"""Controller-owned evidence for Kali command outcomes and model claims."""

from dataclasses import asdict, dataclass
from enum import Enum


class EvidenceState(str, Enum):
    OBSERVED = "OBSERVED"
    INFERRED = "INFERRED"
    UNVERIFIED = "UNVERIFIED"
    VERIFIED = "VERIFIED"
    CONTRADICTED = "CONTRADICTED"


@dataclass(frozen=True)
class CommandEvidence:
    evidence_id: str
    command: str
    state: EvidenceState
    execution_state: str
    exit_code: int | None
    stdout: str
    stderr: str
    timed_out: bool
    duration_seconds: float | None
    started_at: str | None
    submitted_at: str | None
    finished_at: str | None
    side_effect_causality: str
    output_truncated: bool
    error: str | None = None
    privilege_mode: str = "user"
    sudo_password_prompted: bool = False
    sudo_password_sent: bool = False
    sudo_access_validated: bool = False

    @classmethod
    def from_result(cls, result: dict) -> "CommandEvidence":
        state = result.get("state", EvidenceState.UNVERIFIED.value)
        try:
            state = EvidenceState(state)
        except ValueError:
            state = EvidenceState.UNVERIFIED
        return cls(
            evidence_id=str(result.get("evidence_id", "")),
            command=str(result.get("command", "")),
            state=state,
            execution_state=str(result.get("execution_state", "unknown")),
            exit_code=result.get("exit_code"),
            stdout=str(result.get("stdout", "")),
            stderr=str(result.get("stderr", "")),
            timed_out=bool(result.get("timed_out", False)),
            duration_seconds=result.get("duration_seconds"),
            started_at=result.get("started_at"),
            submitted_at=result.get("submitted_at"),
            finished_at=result.get("finished_at"),
            side_effect_causality=str(result.get("side_effect_causality", "unknown")),
            output_truncated=bool(result.get("output_truncated", False)),
            error=str(result["error"]) if result.get("error") else None,
            privilege_mode=str(result.get("privilege_mode", "user")),
            sudo_password_prompted=bool(result.get("sudo_password_prompted", False)),
            sudo_password_sent=bool(result.get("sudo_password_sent", False)),
            sudo_access_validated=bool(result.get("sudo_access_validated", False)),
        )

    def to_dict(self, *, include_streams: bool = True) -> dict:
        result = asdict(self)
        result["state"] = self.state.value
        if not include_streams:
            result.pop("stdout")
            result.pop("stderr")
        return result


@dataclass(frozen=True)
class ClaimEvidence:
    claim_id: str
    claim: str
    state: EvidenceState
    evidence_ids: tuple[str, ...]
    validator: str | None = None

    def to_dict(self) -> dict:
        return {
            "claim_id": self.claim_id,
            "claim": self.claim,
            "state": self.state.value,
            "evidence_ids": list(self.evidence_ids),
            "validator": self.validator,
        }


class EvidenceLedger:
    """Store raw command evidence separately from model interpretation."""

    def __init__(self):
        self.commands: dict[str, CommandEvidence] = {}
        self.claims: list[ClaimEvidence] = []

    def record_command(self, call_id: str, result: dict) -> CommandEvidence:
        evidence = CommandEvidence.from_result(result)
        self.commands[call_id] = evidence
        return evidence

    def command_for(self, call_id: str | None) -> CommandEvidence | None:
        return self.commands.get(call_id or "")

    def command_by_evidence_id(self, evidence_id: str | None) -> CommandEvidence | None:
        if not evidence_id:
            return None
        return next((item for item in self.commands.values() if item.evidence_id == evidence_id), None)

    def record_claim(
        self,
        claim_id: str,
        claim: str,
        evidence_ids: list[str],
        *,
        state: EvidenceState = EvidenceState.INFERRED,
        validator: str | None = None,
    ) -> ClaimEvidence:
        if state in {EvidenceState.VERIFIED, EvidenceState.CONTRADICTED} and not validator:
            raise ValueError("Verified and contradicted claims require a deterministic validator.")
        claim_evidence = ClaimEvidence(
            claim_id=claim_id,
            claim=claim,
            state=state,
            evidence_ids=tuple(evidence_ids),
            validator=validator,
        )
        self.claims.append(claim_evidence)
        return claim_evidence

    def snapshot(self, *, include_streams: bool = False) -> dict:
        return {
            "commands": [item.to_dict(include_streams=include_streams) for item in self.commands.values()],
            "claims": [item.to_dict() for item in self.claims],
        }

    def clear(self) -> None:
        self.commands.clear()
        self.claims.clear()
