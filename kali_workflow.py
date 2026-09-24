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


@dataclass
class KaliWorkflow:
    """Track command admission and execution truth for a single user request."""

    request: str
    max_commands: int = 20
    scope_target: str | None = None
    state: WorkflowState = WorkflowState.PLANNING
    commands_started: int = 0
    commands: set[str] = field(default_factory=set)
    records: list[dict] = field(default_factory=list)
    stop_reason: str | None = None
    discovered_tcp_ports: set[int] = field(default_factory=set)
    port_discovery_complete: bool = False

    def begin(self, command: str, *, allow_repeat: bool = False) -> str | None:
        if self.state is not WorkflowState.PLANNING:
            return self.stop_reason or f"workflow is {self.state.value}"
        if command in self.commands and not allow_repeat:
            return "the same command was already attempted in this request"
        if self.commands_started >= self.max_commands:
            self.block(f"workflow reached its {self.max_commands}-command limit")
            return self.stop_reason
        self.commands.add(command)
        self.commands_started += 1
        self.state = WorkflowState.EXECUTING
        return None

    def finish_command(self, record: dict | None) -> str | None:
        if self.state is not WorkflowState.EXECUTING:
            return self.stop_reason or "no command was in progress"
        if not isinstance(record, dict):
            return self.block("the controller did not record a command result")
        self.records.append(record)
        execution_state = record.get("execution_state")
        if execution_state == "timed_out" or record.get("timed_out"):
            return self.block("command timed out")
        if execution_state != "completed":
            return self.block(f"command ended in state {execution_state or 'unknown'}")
        if record.get("exit_code") is None:
            return self.block("command exit status is unknown")
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

    def block(self, reason: str) -> str:
        self.state = WorkflowState.BLOCKED
        self.stop_reason = reason
        return reason

    def complete(self) -> None:
        if self.state is WorkflowState.PLANNING:
            self.state = WorkflowState.COMPLETE
