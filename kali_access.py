"""One command at a time on the user's Kali VM, over verified SSH."""

import codecs
from contextlib import contextmanager
from datetime import datetime, timezone
import getpass
import hashlib
import json
import os
import re
import shlex
import time
import uuid
from pathlib import Path, PurePosixPath
from execution_mode import unrestricted_execution_enabled


HOST = os.environ.get("DEEP_AGENT_VM_HOST", "192.168.56.103")
PORT = int(os.environ.get("DEEP_AGENT_VM_PORT", "22"))
USER = os.environ.get("DEEP_AGENT_VM_USER", "kali")
KNOWN_HOSTS = Path.home() / ".ssh" / "known_hosts"
COMMAND_TIMEOUT = max(5, min(300, int(os.environ.get("DEEP_AGENT_COMMAND_TIMEOUT", "90"))))
OUTPUT_LIMIT = 120_000
MAX_BACKGROUND_PROCESSES = 32
MAX_BACKGROUND_REGISTRY_ENTRIES = 100
MAX_INTERACTIVE_SESSIONS = 4
MAX_INTERACTIVE_OUTPUT = 12_000
MAX_INTERACTIVE_WAIT_MS = 5_000
_INTERACTIVE_SHELLS = {"sh", "bash", "dash", "zsh", "fish", "csh", "tcsh", "ksh", "ash"}
_INTERACTIVE_SENSITIVE_PROMPT = re.compile(
    r"(?i)\b(?:password|passphrase|passcode|pin|one[- ]time code|otp|verification code|"
    r"access token|token|credential|secret)\b[^\r\n]{0,80}[:?]?\s*$"
)
_ANSI_ESCAPE = re.compile(
    r"\x1b(?:\[[0-?]*[ -/]*[@-~]|\][^\x07]*(?:\x07|\x1b\\)|[@-_])"
)


def _read_password(prompt: str) -> str:
    """Headless-automation fallback for the local password prompts.

    Windows getpass always reads the console, so a piped stdin driver can never
    answer it. An automation driver may instead pre-share the credential in
    process memory via DEEP_AGENT_VM_PASSWORD; it stays out of the model context
    and off disk, like the session sudo cache.
    """
    automated = os.environ.get("DEEP_AGENT_VM_PASSWORD", "")
    if automated:
        return automated
    return getpass.getpass(prompt)


_BACKGROUND_PROCESS_HELPER = r'''
import json, os, signal, subprocess, sys, time

def snapshot(pid):
    try:
        raw = open(f"/proc/{pid}/stat", "r", encoding="ascii").read()
        close = raw.rfind(")")
        fields = raw[close + 2:].split()
        if close < 0 or len(fields) < 20:
            return None
        return {"state": fields[0], "pgrp": int(fields[2]), "start_ticks": fields[19]}
    except (OSError, ValueError, IndexError):
        return None

def emit(value, status=0):
    print(json.dumps(value, separators=(",", ":")))
    raise SystemExit(status)

try:
    operation = sys.argv[1]
    data = json.loads(sys.argv[2])
    pid = int(data["pid"]) if operation != "start" else None
    if operation == "start":
        cwd = data.get("cwd")
        if cwd and not os.path.isdir(cwd):
            emit({"error": "working directory does not exist or is not a directory"}, 2)
        env = os.environ.copy()
        env.update(data.get("env") or {})
        process = subprocess.Popen(
            data["argv"], cwd=cwd or None, env=env,
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL, close_fds=True, start_new_session=True,
        )
        time.sleep(0.15)
        current = snapshot(process.pid)
        if process.poll() is not None or not current or current["state"] in {"Z", "X"}:
            emit({"state": "stopped", "pid": process.pid,
                  "exit_code": process.poll()})
        emit({"state": "running", "pid": process.pid,
              "start_ticks": current["start_ticks"]})

    expected_ticks = str(data["start_ticks"])
    current = snapshot(pid)
    if not current or current["start_ticks"] != expected_ticks:
        emit({"state": "unknown", "pid": pid,
              "error": "process identity no longer matches; no signal was sent"})
    if current["state"] in {"Z", "X"}:
        emit({"state": "stopped", "pid": pid})
    if operation == "check":
        emit({"state": "running", "pid": pid})
    if operation == "stop":
        if current["pgrp"] != pid:
            emit({"state": "unknown", "pid": pid,
                  "error": "process group identity changed; no signal was sent"})
        os.killpg(pid, signal.SIGTERM)
        for _ in range(20):
            time.sleep(0.1)
            current = snapshot(pid)
            if not current or current["start_ticks"] != expected_ticks or current["state"] in {"Z", "X"}:
                emit({"state": "stopped", "pid": pid, "signal": "SIGTERM"})
        emit({"state": "running", "pid": pid,
              "error": "SIGTERM was sent, but the process is still running"})
    emit({"error": "unsupported process operation"}, 2)
except SystemExit:
    raise
except Exception as exc:
    emit({"error": f"{type(exc).__name__}: {exc}"}, 1)
'''

_LISTENING_OUTPUT = re.compile(
    r"(?i)\b(?:serving\s+https?\b|listening\s+(?:on|at)\b|"
    r"(?:web\s+)?server\s+(?:is\s+)?running\b|server\s+started\b|"
    r"ready\s+on\s+https?://|development\s+server\s+(?:is\s+)?running\b|"
    r"(?:local|network):\s*https?://|(?:running|serving|listening)\s+on\s+https?://)"
)


def _failure_type(command: str, stdout: str, stderr: str, exit_code,
                  timed_out: bool) -> str | None:
    """Classify observable execution failures for bounded replanning."""
    output = f"{stderr}\n{stdout}"
    if timed_out and _LISTENING_OUTPUT.search(output):
        return "LONG_RUNNING_PROCESS_USED_AS_ONE_SHOT"
    if re.search(r"(?i)(?:command not found|not found: \S+)", output):
        return "COMMAND_NOT_FOUND"
    if re.search(r"(?i)No such file or directory", output):
        # A running find/cat command can report missing input paths. Shell
        # status 127 instead indicates that an executable could not be run.
        return "COMMAND_NOT_FOUND" if exit_code == 127 else "PATH_NOT_FOUND"
    if re.search(r"(?i)(?:permission denied|operation not permitted)", output):
        return "PERMISSION_DENIED"
    if re.search(r"(?i)(?:address already in use|only one usage of each socket address)", output):
        return "PORT_IN_USE"
    if re.search(r"(?i)(?:network is unreachable|no route to host|host is unreachable)", output):
        return "NETWORK_UNREACHABLE"
    if re.search(r"(?i)(?:invalid argument|invalid option|unrecognized option|usage:)", output):
        return "INVALID_ARGUMENT"
    if timed_out or exit_code in {124, 137}:
        return "TIMEOUT"
    if exit_code not in {None, 0}:
        return "COMMAND_FAILED"
    return None


def _process_strategy_key(command: str) -> tuple[str, ...] | None:
    """Normalize port-valued arguments so port changes do not disguise a retry."""
    try:
        tokens = shlex.split(command)
    except ValueError:
        return None
    if not tokens:
        return None
    normalized = []
    for token in tokens:
        if token.isdigit() and 1 <= int(token) <= 65535:
            normalized.append("<PORT>")
        else:
            def normalize_port(match):
                port = int(match.group("port"))
                return match.group("prefix") + ("<PORT>" if 1 <= port <= 65535 else match.group("port"))

            normalized.append(re.sub(
                r"(?i)(?P<prefix>--?port=|:)(?P<port>\d{1,5})(?=$|[/\]])",
                normalize_port,
                token,
            ))
    return tuple(normalized)


@contextmanager
def _exclusive_file_lock(path: Path):
    """Serialize registry read/merge/writes across agent processes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as lock_file:
        if os.name == "nt":
            import msvcrt

            lock_file.seek(0, os.SEEK_END)
            if lock_file.tell() == 0:
                lock_file.write(b"\0")
                lock_file.flush()
            lock_file.seek(0)
            msvcrt.locking(lock_file.fileno(), msvcrt.LK_LOCK, 1)
            try:
                yield
            finally:
                lock_file.seek(0)
                msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def sudo_command_parts(command: str) -> list[str] | None:
    """Validate sudo input; an empty payload requests a credential check only."""
    if command.strip().lower() == "/sudo":
        raise ValueError("Use `sudo` directly; `/sudo` is a filesystem path, not a sudo command.")
    try:
        tokens = shlex.split(command)
    except ValueError as exc:
        if re.search(r"(?:^|\s)sudo(?:\s|$)", command):
            raise ValueError("The sudo command could not be parsed safely.") from exc
        return None
    if (len(tokens) == 1 and re.fullmatch(r"sudo\s+.+", tokens[0], re.I)):
        raise ValueError(
            "Remove the quotes around sudo and replace the placeholder with a real command, e.g. `sudo id -u`."
        )
    positions = [index for index, token in enumerate(tokens)
                 if os.path.basename(token).lower() == "sudo"]
    if not positions:
        return None
    if positions != [0]:
        raise ValueError("sudo must be the first and only command; shell chaining is not allowed.")
    if "\n" in command or "\r" in command or "$(" in command or "`" in command:
        raise ValueError("sudo accepts one command without multiline input or shell substitution.")

    if len(tokens) == 1:
        return []  # `sudo` alone securely validates credentials without opening a shell.
    if len(tokens) < 2:
        raise ValueError("Add a noninteractive command after sudo.")
    if tokens[1].startswith("-"):
        raise ValueError("Use `sudo COMMAND`; sudo options are managed by the controller.")
    command_parts = tokens[1:]
    if command_parts[0].strip("<>[]{}").lower() in {"command", "cmd", "your_command"}:
        raise ValueError(
            "Replace the sudo placeholder with a real command, e.g. `sudo id -u`."
        )

    lexer = shlex.shlex(command, posix=True, punctuation_chars=";&|<>")
    lexer.whitespace_split = True
    lexer.commenters = ""
    try:
        if any(token and all(character in ";&|<>" for character in token) for token in lexer):
            raise ValueError("sudo accepts one command without shell chaining or redirection.")
    except ValueError as exc:
        if "sudo accepts" in str(exc):
            raise
        raise ValueError("The sudo command could not be parsed safely.") from exc

    executable = os.path.basename(command_parts[0]).lower()
    if executable in {"su", "bash", "sh", "zsh", "fish", "dash", "login"} and not any(
        part in {"-c", "-lc", "--command"} for part in command_parts[1:]
    ):
        raise ValueError("Interactive root shells are not supported; use `sudo COMMAND` for one task.")
    return command_parts


def sudo_authorization_failed(output: str) -> bool:
    return bool(re.search(
        r"(?:sudo:.*(?:password is required|a terminal is required|no password was provided)|sorry,\s*try again|"
        r"incorrect password|not in the sudoers|not allowed to execute)",
        output, re.I,
    ))


def sudo_password_rejected(output: str) -> bool:
    """Detect a bad password, excluding permission and policy failures."""
    return bool(re.search(r"(?:sorry,\s*try again|incorrect password)", output, re.I))


class KaliAccess:
    """Connect lazily and use a fresh, noninteractive SSH channel per command."""

    def __init__(self):
        self.client = None
        self.last_record = None
        self.sudo_mode = False
        self._sudo_password = None
        self._sudo_attempt_number = 1
        self.background_processes = {}
        self.interactive_processes = {}
        self._failed_foreground_process_strategies = set()
        self._process_registry_path = Path(__file__).with_name("runtime") / "background_processes.json"
        self._load_background_processes()

    @staticmethod
    def _process_registry_target():
        return {"host": HOST, "port": PORT, "user": USER}

    def _read_background_processes(self) -> dict:
        try:
            if not self._process_registry_path.is_file() or self._process_registry_path.stat().st_size > 1_000_000:
                return {}
            payload = json.loads(self._process_registry_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            return {}
        if not isinstance(payload, dict) or payload.get("target") != self._process_registry_target():
            return {}
        entries = payload.get("processes")
        if not isinstance(entries, dict):
            return {}
        valid = {}
        for process_id, item in entries.items():
            if not isinstance(item, dict):
                continue
            pid = item.get("pid")
            ticks = item.get("start_ticks")
            state = item.get("state")
            cwd = item.get("cwd")
            program = item.get("program")
            if (not isinstance(process_id, str)
                    or not re.fullmatch(r"proc-[a-f0-9]{32}", process_id)
                    or type(pid) is not int or pid < 1
                    or not isinstance(ticks, str) or not ticks.isdigit()
                    or not isinstance(state, str)
                    or state not in {"running", "stopping", "stopped", "unknown"}
                    or (cwd is not None and (
                        not isinstance(cwd, str) or not PurePosixPath(cwd).is_absolute()
                        or any(ord(char) < 32 for char in cwd)
                    ))
                    or not isinstance(program, str)
                    or len(program) > 256
                    or os.path.basename(program) != program
                    or any(ord(char) < 32 for char in program)):
                continue
            started_at = item.get("started_at", "")
            updated_at = item.get("updated_at", started_at)
            if (not isinstance(started_at, str) or len(started_at) > 80
                    or any(ord(char) < 32 for char in started_at)
                    or not isinstance(updated_at, str) or len(updated_at) > 80
                    or any(ord(char) < 32 for char in updated_at)):
                continue
            valid[process_id] = {
                "process_id": process_id,
                "pid": pid,
                "start_ticks": ticks,
                "program": program,
                "cwd": cwd,
                "state": state,
                "started_at": started_at,
                "updated_at": updated_at,
            }
        return valid

    def _load_background_processes(self):
        self.background_processes.update(self._read_background_processes())

    def _save_background_processes(self):
        self._process_registry_path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = self._process_registry_path.with_name(self._process_registry_path.name + ".lock")
        with _exclusive_file_lock(lock_path):
            merged = self._read_background_processes()
            for process_id, process in self.background_processes.items():
                current = merged.get(process_id)
                process_updated = process.get("updated_at", process.get("started_at", ""))
                current_updated = current.get("updated_at", current.get("started_at", "")) if current else ""
                if current is None or process_updated >= current_updated:
                    merged[process_id] = dict(process)
            active = {
                process_id: process for process_id, process in merged.items()
                if process.get("state") != "stopped"
            }
            stopped = sorted(
                (
                    (process_id, process) for process_id, process in merged.items()
                    if process.get("state") == "stopped"
                ),
                key=lambda entry: entry[1].get("updated_at", entry[1].get("started_at", "")),
                reverse=True,
            )
            retained_stopped = stopped[:max(0, MAX_BACKGROUND_REGISTRY_ENTRIES - len(active))]
            merged = dict(active)
            merged.update(retained_stopped)
            self.background_processes = merged
            payload = {
                "target": self._process_registry_target(),
                "processes": merged,
            }
            temporary = self._process_registry_path.with_name(
                self._process_registry_path.name + "." + uuid.uuid4().hex + ".tmp"
            )
            try:
                temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
                os.replace(temporary, self._process_registry_path)
            finally:
                try:
                    temporary.unlink(missing_ok=True)
                except OSError:
                    pass

    def clear_sudo_mode(self):
        if self._sudo_password is not None:
            self._sudo_password[:] = b"\0" * len(self._sudo_password)
        self._sudo_password = None
        self.sudo_mode = False

    def connect(self):
        if self.client is not None:
            transport = self.client.get_transport()
            if transport is not None and transport.is_active():
                return self.client
            self.close()
        if not KNOWN_HOSTS.is_file():
            raise RuntimeError(
                f"No trusted Kali SSH key found at {KNOWN_HOSTS}. Connect once with "
                f"ssh -p {PORT} {USER}@{HOST} and accept the host key."
            )
        try:
            import paramiko
        except ImportError as exc:
            raise RuntimeError("Paramiko is missing. Install requirements.txt first.") from exc
        client = paramiko.SSHClient()
        client.load_system_host_keys(str(KNOWN_HOSTS))
        client.set_missing_host_key_policy(paramiko.RejectPolicy())
        password = _read_password(f"{USER}@{HOST} Kali password: ")
        try:
            client.connect(
                hostname=HOST, port=PORT, username=USER, password=password,
                allow_agent=False, look_for_keys=False, timeout=5,
                banner_timeout=10, auth_timeout=15,
            )
        except Exception:
            client.close()
            raise
        finally:
            password = None
        self.client = client
        return client

    def run(self, command: str, *, cwd: str | None = None, use_sudo: bool = True,
            display_command: str | None = None, display_output: bool = True) -> str:
        attempt_history = []
        try:
            for attempt in range(1, 4):
                self._sudo_attempt_number = attempt
                output = self._run_once(
                    command, cwd=cwd, use_sudo=use_sudo,
                    display_command=display_command, display_output=display_output,
                )
                record = self.last_record if isinstance(self.last_record, dict) else {}
                is_sudo = record.get("privilege_mode") in {"sudo", "sudo_validation"}
                failed_password = is_sudo and sudo_password_rejected(record.get("stderr", ""))
                if is_sudo:
                    attempt_history.append({
                        "attempt": attempt,
                        "password_rejected": bool(failed_password),
                    })
                    record["sudo_attempt_history"] = list(attempt_history)
                if record and is_sudo:
                    record["sudo_attempt"] = attempt
                if not failed_password:
                    return output
                if attempt < 3:
                    print(
                        f"\n[Kali rejected the sudo password; no privileged command ran. "
                        f"Enter it again (attempt {attempt + 1}/3).]",
                        flush=True,
                    )
                else:
                    print("\n[Kali rejected the sudo password after 3 attempts; no privileged command ran.]", flush=True)
            return output
        finally:
            self._sudo_attempt_number = 1

    def _process_helper(self, operation: str, payload: dict, display_label: str) -> dict:
        """Run a short controller helper; the managed process never owns this SSH channel."""
        encoded = json.dumps(payload, separators=(",", ":"), ensure_ascii=False)
        helper_command = (
            "python3 -c " + shlex.quote(_BACKGROUND_PROCESS_HELPER)
            + " " + shlex.quote(operation) + " " + shlex.quote(encoded)
        )
        if len(helper_command) > 60_000:
            raise ValueError("Background process request is too large for the SSH command channel.")
        self.run(
            helper_command, use_sudo=False, display_command=display_label,
            display_output=False,
        )
        record = self.last_record if isinstance(self.last_record, dict) else {}
        if record.get("execution_state") != "completed" or record.get("exit_code") != 0:
            if record.get("execution_state") != "completed":
                return {"error": record.get("error") or record.get("stderr")
                        or f"controller helper ended in {record.get('execution_state', 'unknown')}"}
        try:
            result = json.loads(str(record.get("stdout", "")).strip())
        except (json.JSONDecodeError, TypeError):
            return {"error": record.get("error") or record.get("stderr")
                    or "controller helper returned invalid process status"}
        return result if isinstance(result, dict) else {"error": "controller helper returned invalid process status"}

    def _record_process_operation(self, operation: str, command: str, result: dict,
                                  *, cwd: str | None = None) -> str:
        record = self.last_record if isinstance(self.last_record, dict) else {}
        state = str(result.get("state", "unknown"))
        error = str(result.get("error", "")) or None
        if operation == "start_background" and state == "stopped" and not error:
            error = "process exited during startup"
        exit_code = 0 if state in {"running", "stopped"} and not error else 1
        lines = [f"operation={operation}"]
        if result.get("process_id"):
            lines.append(f"process_id={result['process_id']}")
        if result.get("pid") is not None:
            lines.append(f"pid={result['pid']}")
        if result.get("program"):
            lines.append(f"program={result['program']}")
        lines.append(f"process_state={state}")
        if cwd:
            lines.append(f"cwd={cwd}")
        if result.get("signal"):
            lines.append(f"signal={result['signal']}")
        if result.get("exit_code") is not None:
            lines.append(f"process_exit_code={result['exit_code']}")
        if error:
            lines.append(f"error={error}")
        output = "\n".join(lines)
        record.update({
            "command": command,
            "operation": operation,
            "cwd": cwd,
            "state": "OBSERVED",
            "execution_state": "completed",
            "exit_code": exit_code,
            "stdout": output,
            "stderr": "",
            "timed_out": False,
            "error": error,
            "failure_type": "PROCESS_START_FAILED" if operation == "start_background" and exit_code else None,
            "process": {
                key: result[key] for key in (
                    "process_id", "pid", "program", "state", "start_ticks", "signal", "exit_code",
                ) if key in result
            },
        })
        return output

    @staticmethod
    def _validate_background_request(command: str, cwd: str | None,
                                     env: dict | None) -> tuple[list[str], dict]:
        unrestricted = unrestricted_execution_enabled()
        if (not isinstance(command, str) or not command.strip() or len(command) > 64_000
                or "\x00" in command or (not unrestricted and any(char in command for char in "\r\n"))):
            raise ValueError("Background command must be one nonempty, single-line command.")
        try:
            argv = ["bash", "-o", "pipefail", "-c", command] if unrestricted else shlex.split(command)
        except ValueError as exc:
            raise ValueError("Background command quoting is invalid.") from exc
        if not argv:
            raise ValueError("Background command must include an executable.")
        if not unrestricted and os.path.basename(argv[0]).lower() == "sudo":
            raise ValueError("Background services run as the authenticated Kali SSH user; sudo is not supported here.")
        if (not unrestricted and os.path.basename(argv[0]).lower() in {"sh", "bash", "dash", "zsh", "fish"}
                and any(arg in {"-c", "-lc", "--command"} for arg in argv[1:])):
            raise ValueError("Background services must use a direct process command, not a nested shell string.")
        if cwd is not None and (
            not isinstance(cwd, str) or not cwd or "\x00" in cwd
            or "\n" in cwd or "\r" in cwd or not PurePosixPath(cwd).is_absolute()
        ):
            raise ValueError("Background working directory must be an absolute POSIX path.")
        if env is None:
            env = {}
        if not isinstance(env, dict) or len(env) > 64:
            raise ValueError("Background environment must be an object with at most 64 entries.")
        clean_env = {}
        env_text_size = 0
        for key, value in env.items():
            if (not isinstance(key, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key)
                    or not isinstance(value, str) or "\x00" in value or len(value) > 8192):
                raise ValueError("Background environment names and values must be valid text.")
            env_text_size += len(key) + len(value)
            if env_text_size > 16_384:
                raise ValueError("Background environment text must not exceed 16 KB.")
            clean_env[key] = value
        return argv, clean_env

    def start_background(self, command: str, *, cwd: str | None = None,
                         env: dict | None = None) -> str:
        """Start a noninteractive detached process under a controller-owned handle."""
        argv, clean_env = self._validate_background_request(command, cwd, env)
        active = sum(item.get("state") in {"running", "stopping"}
                     for item in self.background_processes.values())
        if active >= MAX_BACKGROUND_PROCESSES and not unrestricted_execution_enabled():
            raise RuntimeError(f"At most {MAX_BACKGROUND_PROCESSES} background processes may be active in one session.")
        process_id = "proc-" + uuid.uuid4().hex
        label = f"start background process: {command}" + (f" (cwd {cwd})" if cwd else "")
        result = self._process_helper("start", {"argv": argv, "cwd": cwd, "env": clean_env}, label)
        if result.get("state") == "running" and type(result.get("pid")) is int and result.get("start_ticks"):
            started_at = datetime.now(timezone.utc).isoformat()
            item = {
                "process_id": process_id,
                "pid": result["pid"],
                "start_ticks": str(result["start_ticks"]),
                "program": os.path.basename(argv[0]),
                "cwd": cwd,
                "state": "running",
                "started_at": started_at,
                "updated_at": started_at,
            }
            self.background_processes[process_id] = item
            try:
                self._save_background_processes()
            except OSError as exc:
                self.background_processes.pop(process_id, None)
                stopped = self._process_helper(
                    "stop", {"pid": item["pid"], "start_ticks": item["start_ticks"]},
                    f"roll back unrecordable process {process_id} (SIGTERM)",
                )
                result = {
                    "state": stopped.get("state", "unknown"),
                    "pid": item["pid"],
                    "program": item["program"],
                    "error": f"could not persist process handle; attempted SIGTERM rollback ({type(exc).__name__})",
                }
                output = self._record_process_operation("start_background", command, result, cwd=cwd)
                print(f"\n[Kali process] {output}", flush=True)
                return output
            self._failed_foreground_process_strategies.discard(
                _process_strategy_key(command),
            )
            result.update(item)
        output = self._record_process_operation("start_background", command, result, cwd=cwd)
        print(f"\n[Kali process] {output}", flush=True)
        return output

    def _background_process(self, process_id: str) -> dict:
        if not isinstance(process_id, str) or len(process_id) > 100:
            raise ValueError("Unknown background process handle.")
        self._load_background_processes()
        process = self.background_processes.get(process_id)
        if process is None:
            raise ValueError("Unknown background process handle; only handles recorded for the configured Kali account can be used.")
        return process

    def check_process(self, process_id: str) -> str:
        process = self._background_process(process_id)
        result = self._process_helper(
            "check", {"pid": process["pid"], "start_ticks": process["start_ticks"]},
            f"check background process {process_id}",
        )
        if result.get("state") in {"running", "stopped", "unknown"}:
            process["state"] = result["state"]
            process["updated_at"] = datetime.now(timezone.utc).isoformat()
            self._save_background_processes()
        result["process_id"] = process_id
        result.update({key: process[key] for key in ("pid", "program", "cwd") if key in process})
        output = self._record_process_operation(
            "check_process", f"check_process {process_id}", result, cwd=process.get("cwd"),
        )
        print(f"\n[Kali process] {output}", flush=True)
        return output

    def stop_process(self, process_id: str) -> str:
        process = self._background_process(process_id)
        if process.get("state") == "stopped":
            result = {
                "state": "stopped", "pid": process["pid"],
                "process_id": process_id, "program": process["program"],
                "cwd": process.get("cwd"),
            }
        else:
            process["state"] = "stopping"
            result = self._process_helper(
                "stop", {"pid": process["pid"], "start_ticks": process["start_ticks"]},
                f"stop background process {process_id} (SIGTERM)",
            )
            if result.get("state") in {"running", "stopped", "unknown"}:
                process["state"] = result["state"]
                process["updated_at"] = datetime.now(timezone.utc).isoformat()
                self._save_background_processes()
            result["process_id"] = process_id
            result.update({key: process[key] for key in ("pid", "program", "cwd") if key in process})
        output = self._record_process_operation(
            "stop_process", f"stop_process {process_id}", result, cwd=process.get("cwd"),
        )
        print(f"\n[Kali process] {output}", flush=True)
        return output

    @staticmethod
    def _validate_interactive_request(command: str, cwd: str | None,
                                      env: dict | None) -> tuple[list[str], dict]:
        if unrestricted_execution_enabled():
            return KaliAccess._validate_background_request(command, cwd, env)
        try:
            argv = shlex.split(command)
        except (TypeError, ValueError):
            argv = []
        if argv and os.path.basename(argv[0]).lower() == "env":
            raise ValueError(
                "Interactive mode does not accept env command wrappers; pass environment overrides in the env field."
            )
        first_tokens = argv[:5]
        if any(os.path.basename(token).lower() in _INTERACTIVE_SHELLS | {"sudo"}
               for token in first_tokens):
            raise ValueError(
                "Interactive mode accepts TTY applications, not shells or sudo; use a one-shot Kali command instead."
            )
        return KaliAccess._validate_background_request(command, cwd, env)

    @staticmethod
    def _safe_terminal_text(data: bytes) -> str:
        decoded = data.decode("utf-8", errors="replace")
        decoded = _ANSI_ESCAPE.sub("", decoded)
        return "".join(
            char for char in decoded
            if char in "\t\n\r" or (ord(char) >= 32 and ord(char) != 127)
        )

    def _new_interactive_record(self, operation: str, command: str,
                                *, cwd: str | None = None) -> None:
        self._interactive_record_started = time.monotonic()
        self.last_record = {
            "evidence_id": uuid.uuid4().hex,
            "command": command,
            "operation": operation,
            "cwd": cwd,
            "state": "UNVERIFIED",
            "execution_state": "not_started",
            "exit_code": None,
            "stdout": "",
            "stderr": "",
            "timed_out": False,
            "duration_seconds": 0,
            "side_effect_causality": "unknown",
            "output_truncated": False,
            "error": None,
            "failure_type": None,
            "privilege_mode": "user",
            "started_at": datetime.now(timezone.utc).isoformat(),
        }

    def _record_interactive_operation(self, operation: str, command: str,
                                      session: dict | None, *, output: str = "",
                                      input_length: int | None = None,
                                      error: str | None = None,
                                      execution_state: str = "completed",
                                      cwd: str | None = None,
                                      output_truncated: bool = False) -> str:
        process_id = session.get("process_id") if session else None
        process_state = session.get("state", "unknown") if session else "unknown"
        process = {}
        lines = [f"operation={operation}"]
        if process_id:
            lines.append(f"tty_id={process_id}")
        if session and session.get("program"):
            lines.append(f"program={session['program']}")
        lines.append(f"process_state={process_state}")
        if cwd or (session and session.get("cwd")):
            lines.append(f"cwd={cwd or session.get('cwd')}")
        if input_length is not None:
            lines.append(f"input_characters={input_length}")
        if session and session.get("exit_code") is not None:
            lines.append(f"process_exit_code={session['exit_code']}")
        if output:
            lines.append("terminal_output:\n" + output)
        else:
            lines.append("terminal_output=(no new output)")
        if session:
            process = {
                "process_id": process_id,
                "program": session.get("program"),
                "state": process_state,
                "exit_code": session.get("exit_code"),
            }
        if error:
            lines.append(f"error={error}")
        formatted = "\n".join(lines)
        record = self.last_record if isinstance(self.last_record, dict) else {}
        record.update({
            "command": command,
            "operation": operation,
            "cwd": cwd or (session.get("cwd") if session else None),
            "state": "OBSERVED" if session or output else "UNVERIFIED",
            "execution_state": execution_state,
            "exit_code": 0 if execution_state == "completed" and not error else None,
            "stdout": formatted,
            "stderr": "",
            "timed_out": False,
            "error": error,
            "failure_type": "CONTROLLER_REJECTED" if execution_state == "not_started" and error else None,
            "output_truncated": output_truncated,
            "process": process,
            "duration_seconds": round(
                time.monotonic() - getattr(self, "_interactive_record_started", time.monotonic()), 3,
            ),
            "finished_at": datetime.now(timezone.utc).isoformat(),
        })
        self.last_record = record
        return formatted

    @staticmethod
    def _interactive_process_id(process_id: str) -> bool:
        return isinstance(process_id, str) and bool(re.fullmatch(r"tty-[a-f0-9]{32}", process_id))

    def _interactive_process(self, process_id: str) -> dict:
        if not self._interactive_process_id(process_id):
            raise ValueError("Unknown interactive process handle.")
        session = self.interactive_processes.get(process_id)
        if session is None:
            raise ValueError("Unknown interactive process handle; TTY handles last for this app session only.")
        return session

    def _capture_interactive_output(self, session: dict, wait_ms: int) -> tuple[str, bool]:
        channel = session["channel"]
        deadline = time.monotonic() + wait_ms / 1000
        output_parts = []
        captured = 0
        truncated = False
        raw_tail = session.get("raw_output_tail", b"")
        while True:
            while channel.recv_ready():
                remaining = MAX_INTERACTIVE_OUTPUT - captured
                if remaining <= 0:
                    truncated = True
                    break
                data = channel.recv(min(65536, remaining * 4))
                if not data:
                    break
                raw_tail = (raw_tail + data)[-8192:]
                clean = self._safe_terminal_text(data)
                piece = clean[:remaining]
                output_parts.append(piece)
                captured += len(piece)
                if len(piece) < len(clean):
                    truncated = True
                    break
            if channel.exit_status_ready():
                session["state"] = "stopped"
                session["exit_code"] = channel.recv_exit_status()
                break
            if truncated or captured >= MAX_INTERACTIVE_OUTPUT or time.monotonic() >= deadline:
                break
            if captured and not channel.recv_ready():
                break
            if wait_ms == 0:
                break
            time.sleep(0.025)
        output = "".join(output_parts)
        session["raw_output_tail"] = raw_tail
        if output:
            session["output_tail"] = (session.get("output_tail", "") + output)[-2048:]
        session["output_truncated"] = bool(session.get("output_truncated") or truncated)
        return output, truncated

    def start_interactive(self, command: str, *, cwd: str | None = None,
                          env: dict | None = None) -> str:
        self._new_interactive_record("start_interactive", command, cwd=cwd)
        channel = None
        session = None
        submission_attempted = False
        try:
            argv, clean_env = self._validate_interactive_request(command, cwd, env)
            if self.sudo_mode and not unrestricted_execution_enabled():
                raise ValueError("Interactive TTY sessions do not inherit sudo mode; disable sudo mode first.")
            active = sum(session.get("state") == "running"
                         for session in self.interactive_processes.values())
            if active >= MAX_INTERACTIVE_SESSIONS and not unrestricted_execution_enabled():
                raise RuntimeError(f"At most {MAX_INTERACTIVE_SESSIONS} interactive TTY sessions may run at once.")
            if (len(self.interactive_processes) >= MAX_INTERACTIVE_SESSIONS * 8
                    and not unrestricted_execution_enabled()):
                raise RuntimeError("The interactive TTY session history is full; restart the app to clear it.")
            launch_argv = argv
            if clean_env:
                launch_argv = ["env", *[f"{key}={value}" for key, value in clean_env.items()], *argv]
            remote_command = (f"cd -- {shlex.quote(cwd)} && " if cwd else "") + "exec " + shlex.join(launch_argv)
            if len(remote_command) > 64_000:
                raise ValueError("Interactive process request is too large for the SSH command channel.")
            client = self.connect()
            transport = client.get_transport()
            if transport is None or not transport.is_active():
                raise ConnectionError("Kali SSH connection closed before the TTY session started.")
            channel = transport.open_session(timeout=10)
            channel.settimeout(10)
            channel.get_pty(term="xterm", width=120, height=40)
            process_id = "tty-" + uuid.uuid4().hex
            session = {
                "process_id": process_id,
                "channel": channel,
                "program": os.path.basename(argv[0]),
                "cwd": cwd,
                "state": "unknown",
                "exit_code": None,
                "output_tail": "",
                "raw_output_tail": b"",
                "output_truncated": False,
                "started_at": datetime.now(timezone.utc).isoformat(),
            }
            self.interactive_processes[process_id] = session
            submission_attempted = True
            channel.exec_command(remote_command)
            session["state"] = "running"
            output, truncated = self._capture_interactive_output(session, 250)
            result = self._record_interactive_operation(
                "start_interactive", command, session, output=output, cwd=cwd,
                output_truncated=truncated,
            )
            if truncated:
                result += "\n[terminal output capture reached its per-read limit]"
            print(f"\n[Kali TTY {process_id}]\n{result}", flush=True)
            return result
        except KeyboardInterrupt:
            self._record_interactive_operation(
                "start_interactive", command, session,
                error="TTY startup or output read was interrupted by the user.",
                execution_state="unknown" if submission_attempted else "not_started",
                cwd=cwd,
            )
            raise
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            if session is None and channel is not None:
                try:
                    channel.close()
                except Exception:
                    pass
            self._record_interactive_operation(
                "start_interactive", command, session, error=error,
                execution_state="unknown" if submission_attempted else "not_started",
                cwd=cwd,
            )
            raise

    def read_interactive(self, process_id: str, *, wait_ms: int = 0) -> str:
        command = f"read_interactive {process_id}"
        self._new_interactive_record("read_interactive", command)
        session = None
        try:
            if type(wait_ms) is not int or not 0 <= wait_ms <= MAX_INTERACTIVE_WAIT_MS:
                raise ValueError(f"Interactive read wait must be between 0 and {MAX_INTERACTIVE_WAIT_MS} milliseconds.")
            session = self._interactive_process(process_id)
            output, truncated = self._capture_interactive_output(session, wait_ms)
            result = self._record_interactive_operation(
                "read_interactive", command, session, output=output,
                output_truncated=truncated,
            )
            if truncated:
                result += "\n[terminal output capture reached its per-read limit]"
            print(f"\n[Kali TTY {process_id}]\n{result}", flush=True)
            return result
        except KeyboardInterrupt:
            self._record_interactive_operation(
                "read_interactive", command, session,
                error="TTY output read was interrupted by the user.",
                execution_state="interrupted",
            )
            raise
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            self._record_interactive_operation(
                "read_interactive", command, None, error=error,
                execution_state="not_started",
            )
            raise

    def send_interactive_input(self, process_id: str, input_text: str) -> str:
        input_digest = hashlib.sha256(input_text.encode("utf-8", errors="replace")).hexdigest()[:16] if isinstance(input_text, str) else "invalid"
        command = f"send_interactive_input {process_id} line_sha256={input_digest}"
        self._new_interactive_record("send_interactive_input", command)
        send_attempted = False
        session = None
        try:
            unrestricted = unrestricted_execution_enabled()
            if (not isinstance(input_text, str) or len(input_text) > 4096 or "\x00" in input_text
                    or (not unrestricted and ("\r" in input_text or "\n" in input_text
                                              or any(ord(char) < 32 for char in input_text)))):
                raise ValueError("Interactive input must be one line of at most 4,096 printable characters.")
            session = self._interactive_process(process_id)
            if session.get("state") != "running":
                raise ValueError("Interactive process is not running; no input was sent.")
            pending_output, truncated = self._capture_interactive_output(session, 0)
            recent_terminal_text = self._safe_terminal_text(session.get("raw_output_tail", b""))
            if not unrestricted and _INTERACTIVE_SENSITIVE_PROMPT.search(recent_terminal_text.rstrip()):
                error = "The TTY is asking for a password or secret; the controller will not send model-provided credentials."
                result = self._record_interactive_operation(
                    "send_interactive_input", command, session, output=pending_output,
                    input_length=len(input_text), error=error, execution_state="not_started",
                    output_truncated=truncated,
                )
                print(f"\n[Kali TTY {process_id}]\n{result}", flush=True)
                return result
            if session.get("state") != "running":
                raise ValueError("Interactive process ended before input was sent; no input was sent.")
            send_attempted = True
            session["channel"].sendall((input_text + "\n").encode("utf-8"))
            output, post_send_truncated = self._capture_interactive_output(session, 200)
            result = self._record_interactive_operation(
                "send_interactive_input", command, session,
                output=pending_output + output,
                input_length=len(input_text),
                output_truncated=truncated or post_send_truncated,
            )
            if truncated or post_send_truncated:
                result += "\n[terminal output capture reached its per-read limit]"
            print(f"\n[Kali TTY {process_id}]\n{result}", flush=True)
            return result
        except KeyboardInterrupt:
            self._record_interactive_operation(
                "send_interactive_input", command, session,
                input_length=len(input_text) if isinstance(input_text, str) else None,
                error="TTY input handling was interrupted by the user.",
                execution_state="unknown" if send_attempted else "interrupted",
            )
            raise
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            session = self.interactive_processes.get(process_id) if self._interactive_process_id(process_id) else None
            self._record_interactive_operation(
                "send_interactive_input", command, session, error=error,
                execution_state="unknown" if send_attempted else "not_started",
            )
            raise

    def interrupt_interactive(self, process_id: str) -> str:
        command = f"interrupt_interactive {process_id}"
        self._new_interactive_record("interrupt_interactive", command)
        interrupt_attempted = False
        session = None
        try:
            session = self._interactive_process(process_id)
            if session.get("state") == "running":
                interrupt_attempted = True
                session["channel"].sendall(b"\x03")
            output, truncated = self._capture_interactive_output(session, 1_000)
            result = self._record_interactive_operation(
                "interrupt_interactive", command, session, output=output,
                output_truncated=truncated,
            )
            if truncated:
                result += "\n[terminal output capture reached its per-read limit]"
            print(f"\n[Kali TTY {process_id}]\n{result}", flush=True)
            return result
        except KeyboardInterrupt:
            self._record_interactive_operation(
                "interrupt_interactive", command, session,
                error="TTY interrupt handling was interrupted by the user.",
                execution_state="unknown" if interrupt_attempted else "interrupted",
            )
            raise
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            self._record_interactive_operation(
                "interrupt_interactive", command, None, error=error,
                execution_state="unknown" if interrupt_attempted else "not_started",
            )
            raise

    def background_state_for_prompt(self) -> str:
        lines = []
        for process in self.background_processes.values():
            lines.append(
                f"{process['process_id']} pid={process['pid']} "
                f"program={process['program']} last_observed={process['state']} "
                f"cwd={process.get('cwd') or '(default)'}"
            )
        for session in self.interactive_processes.values():
            lines.append(
                f"{session['process_id']} program={session['program']} "
                f"last_observed={session['state']} cwd={session.get('cwd') or '(default)'}"
            )
        if self._failed_foreground_process_strategies:
            lines.append(
                "A foreground command previously emitted a listening/server-start banner and then timed out. "
                "Do not assume it exited: first check whether the requested endpoint is already available. "
                "If it is absent, use start_background; do not retry foreground startup with changed ports or arguments."
            )
        if not lines:
            return ""
        return (
            "Controller-owned process state (status is the last observation and may be stale):\n"
            + "\n".join(lines)
        )

    def list_background_processes(self) -> str:
        self._load_background_processes()
        if not self.background_processes and not self.interactive_processes:
            return "No controller-managed background processes are recorded for the configured Kali account."
        lines = ["Controller-managed processes (status may be stale):"]
        for process in self.background_processes.values():
            line = (
                f"{process['process_id']} | PID {process['pid']} | "
                f"{process['program']} | last observed {process['state']}"
            )
            if process.get("cwd"):
                line += f" | cwd {process['cwd']}"
            lines.append(line)
        for session in self.interactive_processes.values():
            line = (
                f"{session['process_id']} | TTY | {session['program']} | "
                f"last observed {session['state']}"
            )
            if session.get("cwd"):
                line += f" | cwd {session['cwd']}"
            lines.append(line)
        return "\n".join(lines)

    def failed_foreground_process_issue(self, command: str) -> str | None:
        strategy = _process_strategy_key(command)
        if strategy and strategy in self._failed_foreground_process_strategies:
            return (
                "this foreground process strategy already timed out after reporting that it was listening; "
                "changing a port or another numeric argument does not change the execution mode. "
                "Use start_background with an explicit working directory, then verify the requested endpoint"
            )
        return None

    def _run_once(self, command: str, *, cwd: str | None = None, use_sudo: bool = True,
                  display_command: str | None = None, display_output: bool = True) -> str:
        command = str(command).replace("\r\n", "\n").strip()
        started = time.monotonic()
        self.last_record = {
            "evidence_id": uuid.uuid4().hex,
            "command": command,
            "cwd": cwd,
            "operation": "run_kali_command",
            "state": "UNVERIFIED",
            "execution_state": "not_started",
            "exit_code": None,
            "stdout": "",
            "stderr": "",
            "timed_out": False,
            "duration_seconds": None,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "submitted_at": None,
            "finished_at": None,
            "side_effect_causality": "unknown",
            "output_truncated": False,
            "error": None,
            "privilege_mode": "user",
            "sudo_password_prompted": False,
            "sudo_password_sent": False,
            "sudo_access_validated": False,
            "sudo_attempt": None,
            "sudo_attempt_history": [],
            "failure_type": None,
        }
        if cwd is not None and (
            not isinstance(cwd, str) or "\x00" in cwd or "\n" in cwd or "\r" in cwd
            or not PurePosixPath(cwd).is_absolute()
        ):
            error = "Kali working directory must be an absolute POSIX path without control characters."
            self.last_record["error"] = error
            self._finish_record(started)
            raise ValueError(error)
        if not command or len(command) > 64_000 or "\x00" in command:
            error = "Kali command must be nonempty, contain no NUL, and be at most 64,000 characters."
            self.last_record["error"] = error
            self._finish_record(started)
            raise ValueError(error)
        try:
            sudo_parts = sudo_command_parts(command)
        except ValueError as exc:
            if unrestricted_execution_enabled():
                # Preserve arbitrary shell syntax. A sudo-prefixed script is
                # passed through the existing authenticated sudo transport.
                sudo_parts = (["bash", "-o", "pipefail", "-c", command]
                              if re.match(r"^sudo(?:\s|$)", command) else None)
            else:
                self.last_record["error"] = str(exc)
                self._finish_record(started)
                raise
        session_sudo = use_sudo and self.sudo_mode and sudo_parts is None
        if session_sudo and self._sudo_password is None:
            error = "Sudo session authorization is unavailable; type `sudo` again."
            self.last_record["error"] = error
            self._finish_record(started)
            self.clear_sudo_mode()
            raise RuntimeError(error)
        if sudo_parts is not None:
            self.last_record["privilege_mode"] = "sudo_validation" if not sudo_parts else "sudo"
        elif session_sudo:
            self.last_record["privilege_mode"] = "sudo_session"
            self.last_record["sudo_access_validated"] = True
        if "\n" in command:
            command += "\n"  # Keep a heredoc terminator on its own line.
            self.last_record["command"] = command
        try:
            client = self.connect()
            transport = client.get_transport()
            if transport is None or not transport.is_active():
                raise ConnectionError("Kali SSH connection closed before the command started.")
            channel = transport.open_session(timeout=10)
        except KeyboardInterrupt:
            self.last_record["execution_state"] = "interrupted_before_submission"
            self.last_record["error"] = "User interrupted before the Kali command was submitted."
            self._finish_record(started)
            raise
        except Exception as exc:
            self.last_record["error"] = f"{type(exc).__name__}: {exc}"
            self._finish_record(started)
            raise
        sudo_password = None
        cache_candidate = None
        using_cached_password = False
        if sudo_parts is not None:
            if self.sudo_mode and self._sudo_password is not None:
                sudo_password = bytearray(self._sudo_password)
                using_cached_password = True
            else:
                if sudo_parts == []:
                    self.clear_sudo_mode()
                try:
                    self.last_record["sudo_password_prompted"] = True
                    prompt = f"{USER}@{HOST} Kali sudo password"
                    if self._sudo_attempt_number > 1:
                        prompt += f" (attempt {self._sudo_attempt_number}/3)"
                    entered_password = _read_password(prompt + ": ")
                    sudo_password = bytearray(entered_password.encode("utf-8"))
                    entered_password = None
                    if sudo_parts == []:
                        cache_candidate = bytearray(sudo_password)
                except KeyboardInterrupt:
                    channel.close()
                    self.last_record["execution_state"] = "interrupted_before_submission"
                    self.last_record["error"] = "User interrupted before the sudo command was submitted."
                    self._finish_record(started)
                    raise
                except Exception as exc:
                    channel.close()
                    self.last_record["error"] = f"{type(exc).__name__}: {exc}"
                    self._finish_record(started)
                    raise
        elif session_sudo:
            sudo_password = bytearray(self._sudo_password)
            using_cached_password = True

        if sudo_parts == [] and sudo_password is not None and cache_candidate is None:
            cache_candidate = bytearray(sudo_password)

        cwd_prefix = f"cd -- {shlex.quote(cwd)} && " if cwd else ""
        if sudo_parts is not None or session_sudo:
            sudo_input = (
                "IFS= read -r -s __deep_agent_sudo_password; "
                "set +o pipefail; "
                "printf '%s\\n' \"$__deep_agent_sudo_password\" | "
                "sudo -S -p '' "
            )
            if sudo_parts:
                sudo_payload = cwd_prefix + "exec </dev/null && " + shlex.join(sudo_parts)
                sudo_input += "-- bash -o pipefail -c " + shlex.quote(sudo_payload)
            elif sudo_parts == []:
                sudo_input += "-v"
            else:
                sudo_payload = cwd_prefix + "exec </dev/null && " + command
                sudo_input += "-- bash -o pipefail -c " + shlex.quote(sudo_payload)
            command_to_run = (
                sudo_input + "; __deep_agent_sudo_status=$?; set -o pipefail; "
                "unset __deep_agent_sudo_password; exit $__deep_agent_sudo_status"
            )
        else:
            command_to_run = cwd_prefix + command
        wrapped = (
            f"timeout --signal=TERM --kill-after=3s {COMMAND_TIMEOUT}s "
            f"bash -o pipefail -c {shlex.quote(command_to_run)}"
        )
        command_started = None
        decoders = [codecs.getincrementaldecoder("utf-8")(errors="replace") for _ in range(2)]
        output = []
        streams = [[], []]
        captured = 0
        omitted = False
        exit_code = None
        print(f"\n[Kali] $ {display_command or command}", flush=True)

        def show(index: int, text: str) -> None:
            nonlocal captured, omitted
            if not text:
                return
            buffered_sudo_stderr = (
                index == 1 and self.last_record["privilege_mode"] in {"sudo", "sudo_validation", "sudo_session"}
            )
            if not buffered_sudo_stderr and display_output:
                print(text, end="", flush=True)
            remaining = max(0, OUTPUT_LIMIT - captured)
            shown = text[:remaining]
            if shown:
                output.append(shown)
                streams[index].append(shown)
                captured += len(shown)
            if len(shown) < len(text):
                omitted = True

        try:
            self.last_record["execution_state"] = "submission_unknown"
            channel.exec_command(wrapped)
            self.last_record["execution_state"] = "submitted"
            self.last_record["submitted_at"] = datetime.now(timezone.utc).isoformat()
            command_started = time.monotonic()
            last_progress = command_started
            if sudo_password is not None:
                channel.sendall(sudo_password + b"\n")
                self.last_record["sudo_password_sent"] = True
            channel.shutdown_write()  # No password or interactive prompt can hold stdin open.
            while True:
                for index, (ready, receive) in enumerate((
                    (channel.recv_ready, channel.recv),
                    (channel.recv_stderr_ready, channel.recv_stderr),
                )):
                    for _ in range(16):
                        if not ready():
                            break
                        data = receive(65536)
                        if not data:
                            break
                        show(index, decoders[index].decode(data))
                if channel.exit_status_ready() and not channel.recv_ready() and not channel.recv_stderr_ready():
                    exit_code = channel.recv_exit_status()
                    break
                elapsed = time.monotonic() - command_started
                if elapsed > COMMAND_TIMEOUT + 8:
                    break
                if time.monotonic() - last_progress >= 5:
                    print(f"\n[Still running {int(elapsed)}s; press Ctrl+C to stop.]", flush=True)
                    last_progress = time.monotonic()
                time.sleep(0.05)
        except KeyboardInterrupt:
            print("\n[Command interrupted.]", flush=True)
            self.last_record["execution_state"] = "interrupted"
            self.last_record["state"] = "OBSERVED" if captured else "UNVERIFIED"
            if cache_candidate is not None:
                cache_candidate[:] = b"\0" * len(cache_candidate)
            raise
        except Exception as exc:
            self.last_record["error"] = f"{type(exc).__name__}: {exc}"
            if self.last_record["execution_state"] == "submission_unknown":
                self.last_record["execution_state"] = "unknown"
            self.last_record["state"] = "OBSERVED" if captured else "UNVERIFIED"
            if cache_candidate is not None:
                cache_candidate[:] = b"\0" * len(cache_candidate)
            raise
        finally:
            if sudo_password is not None:
                sudo_password[:] = b"\0" * len(sudo_password)
            channel.close()
            for index, decoder in enumerate(decoders):
                show(index, decoder.decode(b"", final=True))
            if self.last_record["execution_state"] not in {"completed", "timed_out"}:
                self.last_record.update({
                    "state": "OBSERVED" if captured else "UNVERIFIED",
                    "stdout": "".join(streams[0]),
                    "stderr": "".join(streams[1]),
                    "duration_seconds": round(
                        time.monotonic() - (command_started if command_started is not None else started), 3
                    ),
                    "finished_at": datetime.now(timezone.utc).isoformat(),
                    "output_truncated": omitted,
                })

        elapsed = time.monotonic() - (command_started if command_started is not None else started)
        timed_out = exit_code in (124, 137) or exit_code is None
        suffix = "; timeout" if timed_out else ""
        status = f"[exit {exit_code if exit_code is not None else 'unknown'}; {elapsed:.2f}s{suffix}]"
        print(f"\n{status}", flush=True)
        self.last_record.update({
            "state": "OBSERVED" if exit_code is not None or captured else "UNVERIFIED",
            "execution_state": "timed_out" if timed_out else "completed",
            "exit_code": exit_code,
            "stdout": "".join(streams[0]),
            "stderr": "".join(streams[1]),
            "timed_out": timed_out,
            "duration_seconds": round(elapsed, 3),
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "output_truncated": omitted,
        })
        failure_type = _failure_type(
            command, self.last_record["stdout"], self.last_record["stderr"],
            exit_code, timed_out,
        )
        self.last_record["failure_type"] = failure_type
        if failure_type == "LONG_RUNNING_PROCESS_USED_AS_ONE_SHOT":
            strategy = _process_strategy_key(command)
            if strategy:
                self._failed_foreground_process_strategies.add(strategy)
        authorization_denied = sudo_authorization_failed(
            self.last_record["stderr"]
        )
        privilege_mode = self.last_record["privilege_mode"]
        if privilege_mode in {"sudo", "sudo_validation", "sudo_session"}:
            stderr = self.last_record["stderr"]
            if stderr and not sudo_password_rejected(stderr):
                print(stderr, end="", flush=True)
        if privilege_mode == "sudo_validation":
            validated = exit_code == 0 and not timed_out and not authorization_denied
            self.last_record["sudo_access_validated"] = validated
            if validated:
                if self._sudo_password is not None:
                    self.clear_sudo_mode()
                self._sudo_password = cache_candidate
                cache_candidate = None
                self.sudo_mode = True
            else:
                self.clear_sudo_mode()
                self.last_record["error"] = "Sudo credential validation failed; no privileged task ran."
        elif privilege_mode in {"sudo", "sudo_session"}:
            self.last_record["sudo_access_validated"] = not authorization_denied
            if authorization_denied:
                if using_cached_password or privilege_mode == "sudo_session":
                    self.clear_sudo_mode()
                self.last_record.update({
                    "execution_state": "authorization_failed",
                    "error": "Sudo authorization failed; the requested privileged command was not executed.",
                })
        if cache_candidate is not None:
            cache_candidate[:] = b"\0" * len(cache_candidate)
        result = "".join(output).strip()
        if omitted:
            result += "\n[INCOMPLETE: full output was shown live, but exceeded the model capture limit.]"
        return f"{result}\n{status}".strip()

    def _finish_record(self, started: float) -> None:
        self.last_record["duration_seconds"] = round(time.monotonic() - started, 3)
        self.last_record["finished_at"] = datetime.now(timezone.utc).isoformat()

    def close(self):
        self.clear_sudo_mode()
        for session in list(self.interactive_processes.values()):
            channel = session.get("channel")
            try:
                if session.get("state") == "running":
                    channel.sendall(b"\x03")
            except Exception:
                pass
            try:
                channel.close()
            except Exception:
                pass
            if session.get("state") == "running":
                session["state"] = "unknown"
        if self.client is not None:
            self.client.close()
            self.client = None
