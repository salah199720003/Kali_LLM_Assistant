"""One command at a time on the user's Kali VM, over verified SSH."""

import codecs
from datetime import datetime, timezone
import getpass
import os
import re
import shlex
import time
import uuid
from pathlib import Path


HOST = os.environ.get("DEEP_AGENT_VM_HOST", "127.0.0.1")
PORT = int(os.environ.get("DEEP_AGENT_VM_PORT", "2222"))
USER = os.environ.get("DEEP_AGENT_VM_USER", "kali")
KNOWN_HOSTS = Path.home() / ".ssh" / "known_hosts"
COMMAND_TIMEOUT = max(5, min(300, int(os.environ.get("DEEP_AGENT_COMMAND_TIMEOUT", "90"))))
OUTPUT_LIMIT = 120_000


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
        password = getpass.getpass(f"{USER}@{HOST} Kali password: ")
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

    def run(self, command: str) -> str:
        attempt_history = []
        try:
            for attempt in range(1, 4):
                self._sudo_attempt_number = attempt
                output = self._run_once(command)
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

    def _run_once(self, command: str) -> str:
        command = str(command).replace("\r\n", "\n").strip()
        started = time.monotonic()
        self.last_record = {
            "evidence_id": uuid.uuid4().hex,
            "command": command,
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
        }
        if not command or len(command) > 64_000 or "\x00" in command:
            error = "Kali command must be nonempty, contain no NUL, and be at most 64,000 characters."
            self.last_record["error"] = error
            self._finish_record(started)
            raise ValueError(error)
        try:
            sudo_parts = sudo_command_parts(command)
        except ValueError as exc:
            self.last_record["error"] = str(exc)
            self._finish_record(started)
            raise
        session_sudo = self.sudo_mode and sudo_parts is None
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
                    entered_password = getpass.getpass(prompt + ": ")
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

        if sudo_parts is not None or session_sudo:
            sudo_input = (
                "IFS= read -r -s __deep_agent_sudo_password; "
                "set +o pipefail; "
                "printf '%s\\n' \"$__deep_agent_sudo_password\" | "
                "sudo -S -p '' "
            )
            if sudo_parts:
                sudo_payload = "exec </dev/null; " + shlex.join(sudo_parts)
                sudo_input += "-- bash -o pipefail -c " + shlex.quote(sudo_payload)
            elif sudo_parts == []:
                sudo_input += "-v"
            else:
                sudo_payload = "exec </dev/null; " + command
                sudo_input += "-- bash -o pipefail -c " + shlex.quote(sudo_payload)
            command_to_run = (
                sudo_input + "; __deep_agent_sudo_status=$?; set -o pipefail; "
                "unset __deep_agent_sudo_password; exit $__deep_agent_sudo_status"
            )
        else:
            command_to_run = command
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
        print(f"\n[Kali] $ {command}", flush=True)

        def show(index: int, text: str) -> None:
            nonlocal captured, omitted
            if not text:
                return
            buffered_sudo_stderr = (
                index == 1 and self.last_record["privilege_mode"] in {"sudo", "sudo_validation", "sudo_session"}
            )
            if not buffered_sudo_stderr:
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
        if self.client is not None:
            self.client.close()
            self.client = None
