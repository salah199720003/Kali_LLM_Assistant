"""Controller-owned state for one multi-command Kali request."""

from dataclasses import dataclass, field
from enum import Enum
import re
import shlex


class WorkflowState(str, Enum):
    PLANNING = "planning"
    EXECUTING = "executing"
    BLOCKED = "blocked"
    COMPLETE = "complete"


MAX_FAILED_COMMANDS_PER_HYPOTHESIS = 3
MAX_TEST_COMMANDS_PER_HYPOTHESIS = 3
MAX_TEST_COMMANDS_PER_WORKFLOW = 9
MAX_RESEARCH_CALLS_PER_WORKFLOW = 3
MAX_OUTCOME_CHECK_ATTEMPTS_PER_STATE = 3

_SYSTEMD_UNIT_SUFFIX = re.compile(
    r"\.(?:service|socket|target|timer|path|mount|automount|slice|scope|device|swap)$",
    re.I,
)
_SYSTEMD_UNIT_FILE = re.compile(
    r"^[A-Za-z0-9_@.\\:-]+\.(?:service|socket|target|timer|path|mount|automount|slice|scope|device|swap)$",
    re.I,
)
_SYSTEMD_UNIT_ACTIONS = {
    "start", "stop", "restart", "reload", "try-restart", "reload-or-restart",
    "reload-or-try-restart", "force-reload", "enable", "disable", "reenable",
    "mask", "unmask", "preset", "preset-all", "isolate", "reset-failed", "kill",
    "edit", "revert", "set-property",
}
_SERVICE_UNIT_ACTIONS = {
    "start", "stop", "restart", "reload", "force-reload", "try-restart",
}
_SYSTEMD_UNIT_READ_ACTIONS = {
    "is-active", "is-failed", "status", "show", "is-enabled", "list-units",
}
_SERVICE_UNIT_READ_ACTIONS = {"status"}
_SELF_GENERATED_OUTPUT_COMMANDS = {"echo", "false", "printf", "seq", "true", "yes", ":"}
_SYSTEMCTL_VALUE_OPTIONS = {
    "--host", "-H", "--machine", "-M", "--root", "--image", "--type", "-t",
    "--state", "--job-mode", "--preset-mode", "--kill-who", "--signal", "-s",
    "--property", "-p", "--lines", "-n", "--output", "-o", "--timestamp",
}


def _command_tokens(command: str) -> list[str]:
    try:
        tokens = shlex.split(command)
    except ValueError:
        return []
    while tokens:
        wrapper = tokens[0].rsplit("/", 1)[-1].lower()
        index = 1
        if wrapper in {"sudo", "command"}:
            if index < len(tokens) and tokens[index] == "--":
                index += 1
        elif wrapper == "env":
            while index < len(tokens):
                arg = tokens[index]
                if arg == "--":
                    index += 1
                    break
                if arg in {"-i", "--ignore-environment", "-0", "--null"}:
                    index += 1
                elif arg in {"-u", "--unset", "-C", "--chdir"}:
                    if index + 1 >= len(tokens):
                        return []
                    index += 2
                elif (arg.startswith(("--unset=", "--chdir="))
                      or re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*=.*", arg)):
                    index += 1
                elif arg.startswith("-"):
                    return []
                else:
                    break
        elif wrapper == "timeout":
            while index < len(tokens):
                arg = tokens[index]
                if arg == "--":
                    index += 1
                    break
                if arg in {"--foreground", "--preserve-status", "--verbose"}:
                    index += 1
                elif arg in {"-k", "--kill-after", "-s", "--signal"}:
                    if index + 1 >= len(tokens):
                        return []
                    index += 2
                elif (arg.startswith(("--kill-after=", "--signal="))
                      or (arg.startswith(("-k", "-s")) and len(arg) > 2)):
                    index += 1
                elif arg.startswith("-"):
                    return []
                else:
                    break
            if index >= len(tokens):
                return []
            index += 1
        elif wrapper == "stdbuf":
            while index < len(tokens):
                arg = tokens[index]
                if arg == "--":
                    index += 1
                    break
                if arg in {"-i", "-o", "-e"} and index + 1 < len(tokens):
                    index += 2
                elif re.fullmatch(r"-[ioe].+", arg):
                    index += 1
                elif arg.startswith("-"):
                    return []
                else:
                    break
        else:
            break
        tokens = tokens[index:]
    return tokens


def verification_command_issue(command: str) -> str | None:
    """Reject checks that can satisfy their own expected output without reading state."""
    tokens = _command_tokens(command)
    if not tokens:
        return None
    program = tokens[0].rsplit("/", 1)[-1].lower()
    if program in _SELF_GENERATED_OUTPUT_COMMANDS:
        return (
            "this verification command only generates its own output or exit status; "
            "read the requested system state with a relevant check instead"
        )
    return None


def _systemd_inventory_command(command: str) -> tuple[bool, bool]:
    """Return (is inventory query, covers all unit-file types without patterns)."""
    tokens = _command_tokens(command)
    if (len(tokens) < 2 or tokens[0].rsplit("/", 1)[-1].lower() != "systemctl"
            or "list-unit-files" not in tokens[1:]):
        return False, False
    index = tokens.index("list-unit-files", 1) + 1
    complete = True
    while index < len(tokens):
        token = tokens[index]
        option = token.split("=", 1)[0]
        if token in _SYSTEMCTL_VALUE_OPTIONS:
            complete = False
            index += 2
        elif option in _SYSTEMCTL_VALUE_OPTIONS and "=" in token:
            complete = False
            index += 1
        elif token.startswith("-"):
            index += 1
        else:
            # systemctl accepts patterns after list-unit-files; their results
            # prove those names exist, but a missing name is not proof of absence.
            complete = False
            index += 1
    return True, complete


def _systemd_unit_names(output: str) -> set[str]:
    names = set()
    for line in output.splitlines():
        fields = line.split()
        if fields and _SYSTEMD_UNIT_FILE.fullmatch(fields[0]):
            names.add(fields[0])
    return names


def _canonical_systemd_unit(target: str) -> str | None:
    if (not target or any(char in target for char in "/*?[]")
            or target in {".", ".."}):
        return None
    if _SYSTEMD_UNIT_SUFFIX.search(target):
        return target
    if "." in target:
        return target
    return target + ".service"


def _systemctl_unit_targets(args: list[str], actions: set[str]) -> list[str] | None:
    """Parse targets following an accepted systemctl action."""
    index = 0
    while index < len(args):
        token = args[index]
        option = token.split("=", 1)[0]
        if token == "--":
            index += 1
            break
        if token in _SYSTEMCTL_VALUE_OPTIONS:
            index += 2
        elif option in _SYSTEMCTL_VALUE_OPTIONS and "=" in token:
            index += 1
        elif token.startswith("-"):
            index += 1
        else:
            break
    if index >= len(args) or args[index] not in actions:
        return None

    action = args[index]
    if action == "preset-all":
        return []
    raw_targets = []
    index += 1
    while index < len(args):
        token = args[index]
        option = token.split("=", 1)[0]
        if token in _SYSTEMCTL_VALUE_OPTIONS:
            index += 2
        elif option in _SYSTEMCTL_VALUE_OPTIONS and "=" in token:
            index += 1
        elif token.startswith("-"):
            index += 1
        else:
            raw_targets.append(token)
            index += 1
    targets = [_canonical_systemd_unit(value) for value in raw_targets]
    return [value for value in targets if value] if len(targets) == len(raw_targets) else []


def _service_unit_targets(command: str, systemd_actions: set[str],
                          legacy_actions: set[str]) -> list[str] | None:
    """Return targets for a selected systemd/service action, or None if unrelated."""
    tokens = _command_tokens(command)
    if not tokens:
        return None
    program = tokens[0].rsplit("/", 1)[-1].lower()
    args = tokens[1:]
    if program == "service":
        if args and args[0] in legacy_actions:
            return []
        if len(args) >= 2 and args[1] in legacy_actions:
            target = _canonical_systemd_unit(args[0])
            return [target] if target else []
        return None
    if program != "systemctl":
        return None
    return _systemctl_unit_targets(args, systemd_actions)


def _systemd_unit_change_targets(command: str) -> list[str] | None:
    """Return targets for a systemd/service state change, or None if not one."""
    return _service_unit_targets(command, _SYSTEMD_UNIT_ACTIONS, _SERVICE_UNIT_ACTIONS)


def _systemd_unit_inspection_targets(command: str) -> list[str] | None:
    """Return unit targets from a read-only systemd/service status command."""
    return _service_unit_targets(command, _SYSTEMD_UNIT_READ_ACTIONS, _SERVICE_UNIT_READ_ACTIONS)


def _package_manager_changes_units(command: str) -> bool:
    """Invalidate unit evidence after package changes that can add or remove units."""
    tokens = _command_tokens(command)
    if not tokens:
        return False
    program = tokens[0].rsplit("/", 1)[-1].lower()
    args = tokens[1:]
    if program in {"apt", "apt-get", "aptitude"}:
        return any(arg in {
            "install", "reinstall", "upgrade", "dist-upgrade", "full-upgrade",
            "remove", "purge", "autoremove",
        } for arg in args)
    if program == "dpkg":
        return any(arg in {
            "-i", "--install", "--unpack", "--configure", "-r", "--remove",
            "-P", "--purge",
        } for arg in args)
    return False


def expected_result_matches(expected_result: str, record: dict,
                            *, purpose: str | None = None) -> bool | None:
    """Check an output marker from a successful command or an exact exit code."""
    if not isinstance(expected_result, str) or not isinstance(record, dict):
        return None
    if purpose == "verify" and verification_command_issue(str(record.get("command", ""))):
        return None
    if record.get("execution_state") != "completed" or type(record.get("exit_code")) is not int:
        return None
    if record.get("timed_out"):
        return None

    expected_result = expected_result.strip()
    if not expected_result:
        return None
    exit_code = re.fullmatch(r"exit_code=(-?\d+)", expected_result, re.I)
    if exit_code:
        return record["exit_code"] == int(exit_code.group(1))

    # Output from a failed command can contain stale, partial, or diagnostic
    # text that happens to match the requested marker. It is not a successful
    # outcome check; intentional nonzero conditions must use exit_code=N.
    if record["exit_code"] != 0:
        return None

    if not all(isinstance(record.get(key), str) for key in ("stdout", "stderr")):
        return None
    marker = " ".join(expected_result.split()).casefold()
    output = " ".join(f"{record['stdout']}\n{record['stderr']}".split()).casefold()
    # Word boundaries protect markers such as active from matching inactive.
    # A punctuation suffix is already a boundary: HTTP/ must match HTTP/1.1.
    prefix = r"(?<!\w)" if re.match(r"\w", marker[0]) else ""
    suffix = r"(?!\w)" if re.match(r"\w", marker[-1]) else ""
    if re.search(prefix + re.escape(marker) + suffix, output):
        return True
    if record.get("output_truncated"):
        return None
    return False


def _repeat_command_key(command: str) -> str:
    """Collapse insignificant whitespace while preserving shell quoting.

    Keep multiline commands byte-for-byte because newlines and heredoc bodies
    can be meaningful. On a single line, whitespace outside quotes separates
    shell words; whitespace inside quotes is part of an argument and remains
    unchanged.
    """
    if "\n" in command or "\r" in command:
        return command
    normalized = []
    quote = None
    escaped = False
    pending_space = False
    for character in command:
        if quote:
            normalized.append(character)
            if quote == '"':
                if escaped:
                    escaped = False
                elif character == "\\":
                    escaped = True
                elif character == quote:
                    quote = None
            elif character == quote:
                quote = None
            continue
        if escaped:
            normalized.append(character)
            escaped = False
            continue
        if character == "\\":
            if pending_space and normalized:
                normalized.append(" ")
            pending_space = False
            normalized.append(character)
            escaped = True
            continue
        if character in {"'", '"'}:
            if pending_space and normalized:
                normalized.append(" ")
            pending_space = False
            normalized.append(character)
            quote = character
            continue
        if character.isspace():
            pending_space = True
            continue
        if pending_space and normalized:
            normalized.append(" ")
        pending_space = False
        normalized.append(character)
    return "".join(normalized).strip()


@dataclass
class KaliWorkflow:
    """Track command admission and execution truth for a single user request."""

    request: str
    max_commands: int = 20
    scope_target: str | None = None
    requires_goal_check: bool = False
    requires_preflight_check: bool = False
    requires_explicit_change: bool = False
    state: WorkflowState = WorkflowState.PLANNING
    commands_started: int = 0
    test_commands_started: int = 0
    prior_command_count: int = 0
    prior_test_command_count: int = 0
    research_calls_started: int = 0
    state_generation: int = 0
    outcome_check_attempts_in_generation: int = 0
    commands: set[str] = field(default_factory=set)
    rejected_commands: set[tuple[str, ...]] = field(default_factory=set)
    rejected_issues: set[str] = field(default_factory=set)
    records: list[dict] = field(default_factory=list)
    stop_reason: str | None = None
    discovered_tcp_ports: set[int] = field(default_factory=set)
    port_discovery_complete: bool = False
    observed_systemd_units: set[str] = field(default_factory=set)
    systemd_inventory_complete: bool = False
    pending_systemd_units: set[str] = field(default_factory=set)
    hypothesis_failures: dict[str, int] = field(default_factory=dict)
    hypothesis_attempts: dict[str, int] = field(default_factory=dict)
    change_attempted: bool = False
    explicit_change_prompted: bool = False
    preflight_check_run: bool = False
    preflight_command: str = ""
    preflight_expected_result: str = ""
    preflight_condition_met: bool | None = None
    verification_check_run: bool = False
    verification_condition_met: bool | None = None
    verification_expected_result: str = ""
    verification_prompted: bool = False
    current_purpose: str = "inspect"
    current_hypothesis: str = "unspecified"
    current_expected_result: str = ""

    def seed_task_history(self, records: list[dict]) -> None:
        """Carry command deduplication and investigation budgets across confirmations."""
        generation = 0
        outcome_checks = 0
        hypothesis = lambda value: " ".join(str(value or "unspecified").lower().split())
        for item in records or []:
            tool_name = str(item.get("tool_name") or "")
            if tool_name in {"web_search", "searchsploit"}:
                self.research_calls_started += 1
            if tool_name == "web_search":
                continue
            execution = item.get("execution") or {}
            execution_state = execution.get("execution_state", "unknown")
            if execution_state in {
                "not_started", "interrupted_before_submission", "authorization_failed", "skipped",
            }:
                continue
            if (execution_state == "unknown" and execution.get("submitted_at") is None
                    and execution.get("exit_code") is None):
                continue
            command = str(item.get("command") or "")
            purpose = str(item.get("purpose") or "inspect")
            if command:
                self.prior_command_count += 1
                execution_mode = str(item.get("execution_mode") or "one_shot")
                cwd = str(item.get("cwd") or execution.get("cwd") or "")
                generation_key = "change" if purpose == "change" else str(generation)
                self.commands.add(
                    f"{execution_mode}\0{cwd}\0{generation_key}\0{_repeat_command_key(command)}"
                )
            if purpose == "test":
                self.prior_test_command_count += 1
                label = hypothesis(item.get("hypothesis"))
                self.hypothesis_attempts[label] = self.hypothesis_attempts.get(label, 0) + 1
            if (purpose in {"test", "verify"}
                    and str(item.get("expected_result") or "").strip()):
                label = hypothesis(item.get("hypothesis"))
                match = execution.get("expected_result_match")
                if purpose == "verify" and verification_command_issue(command):
                    match = None
                failed = (
                    match is False
                    or (match is None and execution.get("exit_code") not in {0, None})
                )
                if failed:
                    self.hypothesis_failures[label] = self.hypothesis_failures.get(label, 0) + 1
            if purpose == "verify":
                outcome_checks += 1
                expected = str(item.get("expected_result") or "")
                match = execution.get("expected_result_match")
                if execution_state != "completed" or execution.get("exit_code") is None:
                    match = None
                if not self.change_attempted:
                    self.preflight_check_run = True
                    self.preflight_command = command
                    self.preflight_expected_result = expected
                    self.preflight_condition_met = match
                self.verification_check_run = True
                self.verification_expected_result = expected
                self.verification_condition_met = match
            if purpose == "change":
                self.change_attempted = True
                self.verification_check_run = False
                self.verification_condition_met = None
                self.verification_expected_result = ""
                generation += 1
                outcome_checks = 0
        self.state_generation = generation
        self.outcome_check_attempts_in_generation = outcome_checks

    def seed_rejected_attempts(self, attempts: list[dict]) -> None:
        """Carry pre-submission policy rejections across confirmation turns."""
        for item in attempts or []:
            command = item.get("command")
            issue = item.get("issue")
            if isinstance(command, str) and command.strip():
                self.note_rejected_command(command)
            if isinstance(issue, str) and issue.strip():
                self.note_rejected_issue(issue)

    def note_rejected_command(self, command: str) -> bool:
        """Record a controller-rejected command and report whether it repeats one.

        Tokenizing with shell quoting rules makes harmless spacing differences
        equivalent while preserving case-sensitive arguments.
        """
        try:
            lexer = shlex.shlex(command, posix=True, punctuation_chars=";&|")
            lexer.whitespace_split = True
            lexer.commenters = ""
            normalized = tuple(lexer)
        except (TypeError, ValueError):
            normalized = (" ".join(str(command).split()),)
        if not normalized:
            normalized = (" ".join(str(command).split()),)
        repeated = normalized in self.rejected_commands
        self.rejected_commands.add(normalized)
        return repeated

    def note_rejected_issue(self, issue: str) -> bool:
        """Detect different commands repeating the same controller rejection."""
        normalized = " ".join(str(issue or "").casefold().split())
        if not normalized:
            return False
        repeated = normalized in self.rejected_issues
        self.rejected_issues.add(normalized)
        return repeated

    def begin(self, command: str, *, allow_repeat: bool = False,
              execution_mode: str = "one_shot", cwd: str | None = None,
              purpose: str = "inspect", hypothesis: str = "unspecified",
              expected_result: str = "") -> str | None:
        if self.state is not WorkflowState.PLANNING:
            return self.stop_reason or f"workflow is {self.state.value}"
        if not isinstance(purpose, str) or purpose not in {"inspect", "test", "change", "verify"}:
            purpose = "inspect"
        if purpose == "verify" and not (expected_result or "").strip():
            return "a verification command must name its observable expected result"
        if (purpose == "verify" and self.requires_goal_check
                and (self.requires_preflight_check or self.change_attempted)):
            issue = verification_command_issue(command)
            if issue:
                return issue
        if self.goal_condition_satisfied:
            return "the requested outcome check matched after the state change; stop without further commands"
        if (self.requires_preflight_check and not self.preflight_check_run
                and purpose == "change"):
            return (
                "a completed read-only preflight check of the requested outcome must run "
                "before any state-changing command"
            )
        if (self.requires_preflight_check and self.preflight_check_run
                and self.preflight_condition_met is None and purpose != "verify"):
            return (
                "the preflight result was inconclusive; run another bounded read-only "
                "purpose=verify check before changing state or continuing research"
            )
        if self.preflight_satisfied_request:
            return "the preflight check confirmed the requested outcome already exists; stop without further commands"
        if self.verification_required_before_next_action and purpose != "verify":
            return "a read-only verification check must be the next command after the previous state-changing action or completion prompt"
        # A read-only command's previous result may be stale after a change.
        # State-changing commands remain deduplicated across generations to
        # avoid repeating side effects merely because the environment changed.
        generation_key = "change" if purpose == "change" else str(self.state_generation)
        command_key = (
            f"{execution_mode}\0{cwd or ''}\0{generation_key}\0{_repeat_command_key(command)}"
        )
        fresh_verification = purpose == "verify" and (
            self.verification_required_before_next_action
            or (self.verification_check_run and self.verification_condition_met is None)
        )
        if command_key in self.commands and not allow_repeat and not fresh_verification:
            return "the same command was already attempted in this request"
        normalized_hypothesis = " ".join((hypothesis or "unspecified").lower().split())
        if purpose == "test" and self.prior_test_command_count + self.test_commands_started >= MAX_TEST_COMMANDS_PER_WORKFLOW:
            return self.block(
                f"workflow reached its {MAX_TEST_COMMANDS_PER_WORKFLOW}-test investigation limit"
            )
        if purpose == "test" and self.hypothesis_attempts.get(normalized_hypothesis, 0) >= MAX_TEST_COMMANDS_PER_HYPOTHESIS:
            return self.block(
                f"hypothesis {normalized_hypothesis} reached its {MAX_TEST_COMMANDS_PER_HYPOTHESIS}-test limit; revise it or report the blocker"
            )
        if self.prior_command_count + self.commands_started >= self.max_commands:
            self.block(f"workflow reached its {self.max_commands}-command limit")
            return self.stop_reason
        if (purpose == "verify"
                and self.outcome_check_attempts_in_generation >= MAX_OUTCOME_CHECK_ATTEMPTS_PER_STATE):
            return self.block(
                f"workflow reached its {MAX_OUTCOME_CHECK_ATTEMPTS_PER_STATE}-attempt outcome-check limit for this state; report the result as unverified"
            )
        self.commands.add(command_key)
        self.commands_started += 1
        if purpose == "verify":
            self.outcome_check_attempts_in_generation += 1
        if purpose == "test":
            self.test_commands_started += 1
        self.current_purpose = purpose
        self.current_hypothesis = normalized_hypothesis
        self.current_expected_result = expected_result or ""
        self.state = WorkflowState.EXECUTING
        return None

    def begin_research(self) -> str | None:
        """Admit a bounded documentation or exploit-database lookup."""
        if self.state is not WorkflowState.PLANNING:
            return self.stop_reason or f"workflow is {self.state.value}"
        if self.goal_condition_satisfied:
            return "the requested outcome check matched after the state change; stop without further research"
        if self.preflight_satisfied_request:
            return "the preflight check confirmed the requested outcome already exists; stop without further research"
        if self.verification_required_before_next_action:
            return "a read-only outcome check must run before research after the latest state change or completion prompt"
        if self.research_calls_started >= MAX_RESEARCH_CALLS_PER_WORKFLOW:
            return self.block(
                f"workflow reached its {MAX_RESEARCH_CALLS_PER_WORKFLOW}-call research limit"
            )
        self.research_calls_started += 1
        return None

    def finish_command(self, record: dict | None) -> str | None:
        if self.state is not WorkflowState.EXECUTING:
            return self.stop_reason or "no command was in progress"
        if not isinstance(record, dict):
            if self.current_purpose == "change":
                self.change_attempted = True
                self.verification_check_run = False
                self.verification_condition_met = None
                self.state_generation += 1
                self.outcome_check_attempts_in_generation = 0
            return self.block("the controller did not record a command result")
        record = {
            **record,
            "workflow_purpose": self.current_purpose,
            "workflow_hypothesis": self.current_hypothesis,
            "expected_result": self.current_expected_result,
        }
        if (self.current_purpose in {"test", "verify"}
                and self.current_expected_result.strip()):
            record["expected_result_match"] = expected_result_matches(
                self.current_expected_result, record, purpose=self.current_purpose,
            )
        self.records.append(record)
        execution_state = record.get("execution_state")
        process_mode_failure = (
            record.get("failure_type") == "LONG_RUNNING_PROCESS_USED_AS_ONE_SHOT"
        )
        state_change_attempted = (
            self.current_purpose == "change" or process_mode_failure
        )
        if (state_change_attempted
                and execution_state not in {"not_started", "interrupted_before_submission", "authorization_failed", "skipped"}):
            # A foreground server timeout may leave a child process behind,
            # so establish the requested state before attempting another start.
            self.change_attempted = True
            self.verification_check_run = False
            self.verification_condition_met = None
            self.verification_expected_result = ""
            self.verification_prompted = False
            self.state_generation += 1
            self.outcome_check_attempts_in_generation = 0
            unit_targets = _systemd_unit_change_targets(str(record.get("command", "")))
            self.pending_systemd_units = set(unit_targets or [])
            if _package_manager_changes_units(str(record.get("command", ""))):
                self.observed_systemd_units.clear()
                self.systemd_inventory_complete = False
        if execution_state == "timed_out" or record.get("timed_out"):
            if process_mode_failure:
                self.state = WorkflowState.PLANNING
                return None
            return self.block("command timed out")
        if execution_state != "completed":
            return self.block(f"command ended in state {execution_state or 'unknown'}")
        if record.get("exit_code") is None:
            return self.block("command exit status is unknown")

        inventory_command, inventory_is_complete = _systemd_inventory_command(
            str(record.get("command", "")),
        )
        if (inventory_command and record.get("exit_code") == 0
                and not record.get("timed_out") and not record.get("output_truncated")):
            self.observed_systemd_units.update(
                _systemd_unit_names(str(record.get("stdout", ""))),
            )
            if inventory_is_complete:
                self.systemd_inventory_complete = True

        if self.current_purpose == "test":
            hypothesis = self.current_hypothesis or "unspecified"
            self.hypothesis_attempts[hypothesis] = self.hypothesis_attempts.get(hypothesis, 0) + 1

        if self.current_purpose == "verify":
            # A completed check may confirm or contradict the goal. The
            # controller evaluates the stated condition, not the whole goal.
            self.verification_check_run = True
            self.verification_expected_result = self.current_expected_result
            self.verification_condition_met = record.get("expected_result_match")
            if self.requires_preflight_check and not self.preflight_check_run:
                self.preflight_check_run = True
                self.preflight_command = str(record.get("command", ""))
                self.preflight_expected_result = self.current_expected_result
                self.preflight_condition_met = record.get("expected_result_match")
            elif (self.requires_preflight_check and self.preflight_check_run
                  and self.preflight_condition_met is None and not self.change_attempted):
                self.preflight_command = str(record.get("command", ""))
                self.preflight_expected_result = self.current_expected_result
                self.preflight_condition_met = record.get("expected_result_match")

        expected_match = record.get("expected_result_match")
        failed_or_contradicted = (
            self.current_purpose in {"test", "verify"}
            and bool(self.current_expected_result.strip())
            and (
                expected_match is False
                or (expected_match is None and record.get("exit_code") != 0)
            )
        )
        if failed_or_contradicted:
            hypothesis = self.current_hypothesis or "unspecified"
            failures = self.hypothesis_failures.get(hypothesis, 0) + 1
            self.hypothesis_failures[hypothesis] = failures
            if failures >= MAX_FAILED_COMMANDS_PER_HYPOTHESIS:
                return self.block(
                    f"{MAX_FAILED_COMMANDS_PER_HYPOTHESIS} results failed or contradicted the same hypothesis: {hypothesis}"
                )

        if (self.scope_target and record.get("exit_code") == 0
                and not record.get("output_truncated")):
            try:
                command_tokens = shlex.split(str(record.get("command", "")))
            except ValueError:
                command_tokens = []
            if command_tokens and command_tokens[0].rsplit("/", 1)[-1].lower() == "nmap":
                stdout = str(record.get("stdout", ""))
                self.discovered_tcp_ports.update(
                    int(match.group(1))
                    for match in re.finditer(r"(?m)^\s*(\d{1,5})/tcp\s+open(?:\s|$)", stdout)
                    if 1 <= int(match.group(1)) <= 65535
                )
                self.port_discovery_complete = True
        self.state = WorkflowState.PLANNING
        return None

    def prompt_state(self) -> str:
        lines = [
            f"Original user goal (keep fixed): {self.request}",
            f"Budgets this goal: commands {self.prior_command_count + self.commands_started}/{self.max_commands}; "
            f"research {self.research_calls_started}/{MAX_RESEARCH_CALLS_PER_WORKFLOW}; "
            f"investigation tests {self.prior_test_command_count + self.test_commands_started}/"
            f"{MAX_TEST_COMMANDS_PER_WORKFLOW} ({MAX_TEST_COMMANDS_PER_HYPOTHESIS} per hypothesis; "
            f"{MAX_FAILED_COMMANDS_PER_HYPOTHESIS} failed or contradicted results stop one).",
        ]
        if self.state_generation:
            lines.append(
                f"A state-changing action may have occurred (state generation {self.state_generation}). "
                "Read-only results from earlier generations may be stale; repeat a relevant check before relying on them."
            )
        if (self.requires_preflight_check and self.preflight_check_run
                and self.preflight_condition_met is None):
            lines.append(
                "The preflight result is inconclusive. Do not change state, research, or claim the outcome; "
                "run another bounded read-only purpose=verify check with a concrete expected_result. "
                f"Outcome-check attempts used in this state: {self.outcome_check_attempts_in_generation}/{MAX_OUTCOME_CHECK_ATTEMPTS_PER_STATE}."
            )
        if any(record.get("failure_type") == "LONG_RUNNING_PROCESS_USED_AS_ONE_SHOT"
               for record in self.records):
            lines.append(
                "The one-shot executor observed a server/listening startup message and then timed out. "
                "The process may or may not have exited; first run a read-only check of the requested "
                "endpoint or listener. If the goal is still absent, use start_background. Do not retry "
                "foreground startup with changed ports or arguments."
            )
        if self.observed_systemd_units:
            completeness = "complete" if self.systemd_inventory_complete else "partial"
            lines.append(
                f"Controller recorded a {completeness} systemd unit-file inventory; "
                f"{len(self.observed_systemd_units)} exact unit name(s) were observed. "
                "Model-proposed unit changes must use an observed name."
            )
        if self.requires_preflight_check:
            if not self.preflight_check_run:
                lines.append(
                    "Preflight required before any change: run one bounded, read-only purpose=verify check "
                    "of whether the requested outcome already exists, using a concrete expected_result. "
                    "Prefer a direct check of the requested endpoint or service over broad inventories. "
                    "The controller will reject state changes until this check completes. For explicit create/change "
                    "requests, existing resources are context and do not by themselves fulfill the requested change."
                )
            else:
                match_text = (
                    "matched" if self.preflight_condition_met is True else
                    "did not match" if self.preflight_condition_met is False else
                    "could not be evaluated"
                )
                lines.append(
                    f"Preflight completed with {match_text}: command={self.preflight_command!r}, "
                    f"expected={self.preflight_expected_result!r}. Use this result to choose the minimum next step. "
                    "If a state-changing action runs, perform a separate post-change verification before claiming completion."
                )
        if self.explicit_change_pending:
            lines.append(
                "The original request explicitly requires creating, writing, building, updating, or restarting. "
                "A passing preflight only describes pre-existing state; it does not fulfill that requested action. "
                "Perform the minimum relevant state change, then verify its result."
            )
        for hypothesis, failures in sorted(self.hypothesis_failures.items()):
            attempts = self.hypothesis_attempts.get(hypothesis, 0)
            lines.append(f"Hypothesis {hypothesis!r}: {attempts}/{MAX_TEST_COMMANDS_PER_HYPOTHESIS} tests; {failures}/{MAX_FAILED_COMMANDS_PER_HYPOTHESIS} failed or contradicted results.")
            if failures >= 2:
                lines.append("Two results failed or contradicted this hypothesis. Re-evaluate assumptions and choose a materially different test, or report the blocker; do not repeat command variants.")
            elif attempts >= MAX_TEST_COMMANDS_PER_HYPOTHESIS - 1:
                lines.append("This hypothesis has only one test slot left. Use it only for a materially different check, otherwise report the blocker.")
        for hypothesis, attempts in sorted(self.hypothesis_attempts.items()):
            if hypothesis not in self.hypothesis_failures:
                lines.append(f"Hypothesis {hypothesis!r}: {attempts}/{MAX_TEST_COMMANDS_PER_HYPOTHESIS} tests; no failed or contradicted results.")
                if attempts >= MAX_TEST_COMMANDS_PER_HYPOTHESIS - 1:
                    lines.append("This hypothesis has only one test slot left. Use it only for a materially different check, otherwise report the blocker.")
        if self.change_attempted and not self.verification_check_run:
            lines.append("A state-changing action has run since the last verification. The goal remains unverified; next run one minimal read-only check before claiming success.")
        elif self.requires_goal_check and not self.verification_check_run:
            lines.append("This action request needs one read-only outcome check before completion; use purpose=verify with a concrete expected_result.")
        elif self.goal_condition_satisfied:
            lines.append(
                f"The requested outcome condition {self.verification_expected_result!r} matched after the state change. "
                "The controller will reject further actions; report the recorded result and stop."
            )
        elif self.verification_check_run:
            if self.verification_condition_met is True:
                lines.append(f"The stated verification condition matched: {self.verification_expected_result!r}. This checks the output condition only; the user goal remains a model assessment.")
            elif self.verification_condition_met is False:
                lines.append(f"The stated verification condition did not match: {self.verification_expected_result!r}. Do not claim the requested outcome is confirmed; revise the plan or report the blocker.")
            else:
                lines.append(
                    f"The stated verification condition could not be evaluated: {self.verification_expected_result!r}. "
                    "Output or execution evidence is incomplete; do not change state or research. Run another "
                    f"bounded read-only outcome check ({self.outcome_check_attempts_in_generation}/{MAX_OUTCOME_CHECK_ATTEMPTS_PER_STATE} attempts used), "
                    "or report that the goal remains unverified if the evidence stays incomplete."
                )
        return "\n".join(lines)

    @property
    def outcome_check_pending(self) -> bool:
        return (
            (self.change_attempted or self.requires_goal_check)
            and (not self.verification_check_run or self.verification_condition_met is None)
        )

    @property
    def preflight_satisfied_request(self) -> bool:
        return (
            self.requires_goal_check and self.requires_preflight_check
            and self.preflight_check_run and self.preflight_condition_met is True
            and not self.requires_explicit_change and not self.change_attempted
        )

    @property
    def goal_condition_satisfied(self) -> bool:
        """A declared outcome check matched after a required state change."""
        return (
            self.requires_goal_check
            and self.change_attempted
            and self.verification_check_run
            and self.verification_condition_met is True
        )

    @property
    def explicit_change_pending(self) -> bool:
        return self.requires_explicit_change and not self.change_attempted

    @property
    def verification_required_before_next_action(self) -> bool:
        return (
            (self.change_attempted or self.verification_prompted)
            and (not self.verification_check_run or self.verification_condition_met is None)
        ) or (
            self.requires_preflight_check and self.preflight_check_run
            and self.preflight_condition_met is None
        )

    def systemd_change_issue(self, command: str) -> str | None:
        """Require model-proposed unit changes to use names observed from systemd."""
        targets = _systemd_unit_change_targets(command)
        if targets is None:
            return None
        if not targets:
            return "a systemd/service state change must name one or more explicit units; no service action was run"
        if not self.systemd_inventory_complete and not self.observed_systemd_units:
            return (
                "no usable systemd unit names are captured yet; inspect `systemctl list-unit-files` "
                "(or a name-filtered query) and use only an exact unit returned in complete command output"
            )
        unknown = sorted(set(targets) - self.observed_systemd_units)
        if unknown:
            extent = "the complete" if self.systemd_inventory_complete else "the recorded"
            return (
                f"unit(s) {', '.join(unknown)} were not observed in {extent} systemd unit-file inventory; "
                "do not guess a service name. Refresh or narrow the inventory and use an exact observed unit"
            )
        return None

    def systemd_verification_issue(self, command: str, purpose: str) -> str | None:
        """Keep service outcome checks tied to the units changed in this workflow."""
        if purpose != "verify" or not self.pending_systemd_units:
            return None
        checked_units = _systemd_unit_inspection_targets(command)
        if checked_units is None:
            checked = " ".join(sorted(self.pending_systemd_units))
            return (
                f"the next verification must inspect the changed systemd unit(s) {checked}; "
                "use a read-only `systemctl is-active`, `systemctl status`, or `service ... status` check"
            )
        if set(checked_units) != self.pending_systemd_units:
            expected = " ".join(sorted(self.pending_systemd_units))
            return (
                f"this verification checks {', '.join(checked_units) or 'no explicit unit'}, but the latest "
                f"change targeted {expected}; inspect exactly those changed unit(s) before reporting the outcome"
            )
        return None

    def block(self, reason: str) -> str:
        self.state = WorkflowState.BLOCKED
        self.stop_reason = reason
        return reason

    def complete(self) -> None:
        if self.state is WorkflowState.PLANNING:
            self.state = WorkflowState.COMPLETE
