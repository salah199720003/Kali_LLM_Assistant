"""Controller-owned evidence for Kali command outcomes and model claims."""

import re
import shlex
import urllib.parse
import uuid
from dataclasses import asdict, dataclass
from enum import Enum
from html.parser import HTMLParser


class EvidenceState(str, Enum):
    OBSERVED = "OBSERVED"
    INFERRED = "INFERRED"
    UNVERIFIED = "UNVERIFIED"
    VERIFIED = "VERIFIED"
    CONTRADICTED = "CONTRADICTED"


class EvidenceStage(str, Enum):
    HTTP_RESPONSE_OBSERVED = "HTTP_RESPONSE_OBSERVED"
    SERVICE_FOUND = "SERVICE_FOUND"
    WEB_APP_CONFIRMED = "WEB_APP_CONFIRMED"
    INPUT_SURFACE_FOUND = "INPUT_SURFACE_FOUND"
    REFLECTION_FOUND = "REFLECTION_FOUND"
    SQL_ERROR_EXPOSED = "SQL_ERROR_EXPOSED"
    SECURITY_FINDING_VERIFIED = "SECURITY_FINDING_VERIFIED"


_FACT_VALIDATORS = {
    EvidenceStage.HTTP_RESPONSE_OBSERVED: {"curl_response_parser_v1"},
    EvidenceStage.SERVICE_FOUND: {
        "ss_tcp_listener_line_parser_v1",
        "nmap_open_port_line_parser_v1",
    },
    EvidenceStage.WEB_APP_CONFIRMED: {"curl_success_status_parser_v1"},
    EvidenceStage.INPUT_SURFACE_FOUND: {"curl_query_parameter_parser_v1", "curl_html_input_parser_v1"},
    EvidenceStage.REFLECTION_FOUND: {"curl_literal_query_reflection_parser_v1"},
    EvidenceStage.SQL_ERROR_EXPOSED: {"curl_sql_error_parser_v1"},
    EvidenceStage.SECURITY_FINDING_VERIFIED: {"workflow_marker_match_v1"},
}


_SQL_ERROR_SIGNATURES = (
    "you have an error in your sql syntax",
    "warning: mysql",
    "mysqli?",
    "sqlite3\\.(?:query|exec|prepare)",
    "unterminated quoted string",
    "pg_query\\(",
    "postgresql.*ERROR",
    "ORA-\\d{5}",
    "Microsoft OLE DB Provider",
    "ODBC SQL Server Driver",
    "Unclosed quotation mark",
)


def _sql_error_in_body(body: str) -> str | None:
    """Return the first matching DBMS error signature in a response body."""
    for pattern in _SQL_ERROR_SIGNATURES:
        match = re.search(pattern, body, re.I)
        if match:
            return match.group(0).lower()
    return None


@dataclass(frozen=True)
class EvidenceFact:
    fact_id: str
    evidence_id: str
    stage: EvidenceStage
    value: dict
    validator: str

    def to_dict(self) -> dict:
        return {
            "fact_id": self.fact_id,
            "evidence_id": self.evidence_id,
            "stage": self.stage.value,
            "value": dict(self.value),
            "validator": self.validator,
        }


def _command_words(command: str) -> list[str]:
    try:
        words = shlex.split(command)
    except ValueError:
        return []
    while words:
        program = words[0].rsplit("/", 1)[-1]
        if program == "sudo":
            words.pop(0)
            while words and words[0].startswith("-") and words[0] != "--":
                option = words.pop(0)
                if option in {"-u", "--user", "-g", "--group", "-h", "--host", "-p", "--prompt"} and words:
                    words.pop(0)
            if words and words[0] == "--":
                words.pop(0)
        elif program == "timeout":
            words.pop(0)
            while words and words[0].startswith("-"):
                option = words.pop(0)
                if option in {"-k", "--kill-after", "-s", "--signal"} and words:
                    words.pop(0)
            if words and re.fullmatch(r"\d+(?:\.\d+)?[smhd]?", words[0]):
                words.pop(0)
        elif program == "stdbuf":
            words.pop(0)
            while words and words[0].startswith("-"):
                words.pop(0)
        else:
            break
    return words


def command_http_urls(command: str) -> list[urllib.parse.SplitResult]:
    urls = []
    for raw in re.findall(r"(?i)https?://[^\s\"'<>]+", command):
        raw = raw.rstrip(",;)]}")
        try:
            parsed = urllib.parse.urlsplit(raw)
            parsed.port
        except ValueError:
            continue
        if parsed.scheme in {"http", "https"} and parsed.hostname:
            urls.append(parsed)
    return urls


def curl_response_options(command: str) -> set[str]:
    """Recognize response flags without mistaking option values for flags."""
    words = _command_words(command)
    if not words or words[0].rsplit("/", 1)[-1] != "curl":
        return set()
    result = set()
    takes_value = {
        "--url", "--header", "--user", "--proxy", "--output", "--write-out",
        "--request", "--data", "--data-raw", "--data-binary", "--data-urlencode",
        "--max-time", "--connect-timeout", "--cacert", "--capath", "--cert",
        "--key", "--cookie", "--cookie-jar", "--user-agent", "--referer",
        "--resolve", "--connect-to", "--range", "--upload-file", "--dump-header",
    }
    short_flags = {"i": "include", "I": "head", "L": "location", "k": "insecure"}
    skip_value = False
    for word in words[1:]:
        if skip_value:
            skip_value = False
            continue
        if word == "--":
            break
        if word.startswith("--"):
            option = word.split("=", 1)[0]
            if option in {"--include", "--head", "--location", "--insecure"}:
                result.add(option[2:])
            if option in takes_value and "=" not in word:
                skip_value = True
        elif word.startswith("-"):
            for index, flag in enumerate(word[1:], 1):
                if flag in short_flags:
                    result.add(short_flags[flag])
                if flag in "HoXdmuwUbAceDEFTxrzyYK":
                    skip_value = index == len(word) - 1
                    break
    return result


class _HtmlInputs(HTMLParser):
    """Collect advertised GET inputs; page text remains untrusted data."""

    def __init__(self, page_url: str):
        super().__init__(convert_charrefs=True)
        self.page_url = page_url
        self.form = None
        self.inputs = []
        self.has_base = False

    def handle_starttag(self, tag, attributes):
        attrs = dict(attributes)
        if tag == "base":
            # A base element changes relative URL resolution. Leave those pages
            # to the model rather than attach inputs to an invented origin.
            self.has_base = True
        elif tag == "form":
            method = (attrs.get("method") or "get").lower()
            self.form = urllib.parse.urljoin(self.page_url, attrs.get("action") or self.page_url) if method == "get" else None
        elif tag in {"input", "select", "textarea"} and self.form:
            name = attrs.get("name")
            if (name and "disabled" not in attrs
                    and (attrs.get("type") or "text").lower() not in {
                        "password", "hidden", "submit", "button", "reset", "file",
                    }):
                self.inputs.append((self.form, name, "get_form_in_response"))
        elif tag == "a" and attrs.get("href"):
            target = urllib.parse.urljoin(self.page_url, attrs["href"])
            for name in urllib.parse.parse_qs(urllib.parse.urlsplit(target).query, keep_blank_values=True):
                self.inputs.append((target, name, "query_link_in_response"))

    def handle_endtag(self, tag):
        if tag == "form":
            self.form = None


def _derive_facts(evidence: "CommandEvidence") -> list[tuple[EvidenceStage, dict, str]]:
    """Extract narrow, reproducible facts from common Kali command formats."""
    if (evidence.execution_state != "completed" or evidence.exit_code != 0
            or evidence.timed_out or evidence.output_truncated):
        return []
    command = evidence.command
    stdout = evidence.stdout
    words = _command_words(command)
    program = words[0].rsplit("/", 1)[-1].lower() if words else ""
    facts: list[tuple[EvidenceStage, dict, str]] = []

    if program == "ss":
        options = "".join(word.lstrip("-") for word in words[1:] if word.startswith("-"))
        tcp = "t" in options or "--tcp" in words
        listening = "l" in options or "--listening" in words
        if tcp and listening:
            listener_line = re.compile(
                r"(?m)^\s*LISTEN\s+\d+\s+\d+\s+(\S+)\s+\S+(.*)$"
            )
            for match in listener_line.finditer(stdout):
                local_address, suffix = match.groups()
                port_match = re.search(r":(\d+)\*?$", local_address)
                if not port_match:
                    continue
                local_address = local_address.rstrip("*")
                process_match = re.search(r'users:\(\("([^"\s]+)"', suffix)
                facts.append((EvidenceStage.SERVICE_FOUND, {
                    "protocol": "tcp",
                    "local_address": local_address,
                    "port": int(port_match.group(1)),
                    "process": process_match.group(1) if process_match else None,
                }, "ss_tcp_listener_line_parser_v1"))

    elif program == "nmap":
        host_line = re.compile(r"(?m)^Nmap scan report for (.+)$")
        port_line = re.compile(r"(?m)^\s*(\d{1,5})/(tcp|udp)\s+open\s+(\S+)?")
        reports = list(host_line.finditer(stdout))
        for index, report in enumerate(reports):
            host = report.group(1).strip()
            end = reports[index + 1].start() if index + 1 < len(reports) else len(stdout)
            for port in port_line.finditer(stdout, report.end(), end):
                port_number = int(port.group(1))
                if not 1 <= port_number <= 65535:
                    continue
                facts.append((EvidenceStage.SERVICE_FOUND, {
                    "host": host,
                    "protocol": port.group(2),
                    "port": port_number,
                    "service": port.group(3),
                }, "nmap_open_port_line_parser_v1"))

    if program == "curl":
        urls = command_http_urls(command)
        # A compound shell command may contain several curl requests. Without a
        # structured per-request result, do not attach a response to the wrong URL.
        if len(urls) == 1:
            url = urls[0]
            route = url.path or "/"
            response_options = curl_response_options(command)
            header_options = bool(response_options & {"include", "head"})
            status_matches = (
                list(re.finditer(r"(?im)^HTTP/\S+\s+(\d{3})\b", evidence.stdout))
                if header_options else []
            )
            status_match = status_matches[-1] if status_matches else None
            status = int(status_match.group(1)) if status_match else None
            if "location" in response_options and any(
                    300 <= int(match.group(1)) < 400 for match in status_matches):
                # curl -L can finish at a different route or host. Without the
                # effective URL we cannot attribute its final response safely.
                return facts
            response_headers = ""
            if status_match:
                header_end = re.compile(r"\r?\n\r?\n").search(stdout, status_match.end())
                response_headers = stdout[status_match.end():header_end.start() if header_end else len(stdout)]
            content_type_match = re.search(
                r"(?im)^content-type:\s*([^;\r\n]+)", response_headers,
            )
            content_type = content_type_match.group(1).strip().lower() if content_type_match else None
            response_body = evidence.stdout
            if header_options:
                response_body = ""
                if status_match:
                    body_delimiter = re.compile(r"\r?\n\r?\n").search(
                        evidence.stdout, status_match.end(),
                    )
                    if body_delimiter:
                        response_body = evidence.stdout[body_delimiter.end():]

            origin = {"scheme": url.scheme, "host": url.hostname, "port": url.port}
            location_match = re.search(r"(?im)^location:[ \t]*([^\r\n]+)", response_headers)
            redirect_url = (
                urllib.parse.urljoin(url.geturl(), location_match.group(1).strip())
                if location_match and status is not None and 300 <= status < 400 else None
            )

            if status is not None or response_body:
                facts.append((EvidenceStage.HTTP_RESPONSE_OBSERVED, {
                    "scheme": url.scheme,
                    "host": url.hostname,
                    "port": url.port,
                    "route": route,
                    "status": status,
                    "response_body_present": bool(response_body),
                    "content_type": content_type,
                    "redirect_url": redirect_url,
                }, "curl_response_parser_v1"))
            if status is not None and 200 <= status < 400:
                facts.append((EvidenceStage.WEB_APP_CONFIRMED, {
                    "scheme": url.scheme,
                    "host": url.hostname,
                    "port": url.port,
                    "route": route,
                    "status": status,
                    "redirect_url": redirect_url,
                }, "curl_success_status_parser_v1"))

            if (response_body and status is not None and 200 <= status < 300
                    and content_type == "text/html" and "head" not in response_options):
                parser = _HtmlInputs(url.geturl())
                try:
                    parser.feed(response_body)
                except (ValueError, AssertionError):
                    parser.inputs = []
                for target, parameter, source in ([] if parser.has_base else parser.inputs)[:40]:
                    try:
                        parsed = urllib.parse.urlsplit(target)
                        target_origin = (parsed.scheme, parsed.hostname, parsed.port or (443 if parsed.scheme == "https" else 80))
                    except ValueError:
                        continue
                    if target_origin != (url.scheme, url.hostname, url.port or (443 if url.scheme == "https" else 80)):
                        continue
                    facts.append((EvidenceStage.INPUT_SURFACE_FOUND, {
                        **origin, "route": parsed.path or "/", "parameter": parameter,
                        "url": parsed.geturl(), "source": source,
                        "application_handling": "advertised_in_response",
                    }, "curl_html_input_parser_v1"))

            parameters = urllib.parse.parse_qs(url.query, keep_blank_values=True)
            for parameter, values in parameters.items():
                facts.append((EvidenceStage.INPUT_SURFACE_FOUND, {
                    **origin,
                    "route": route,
                    "parameter": parameter,
                    "source": "query_parameter_in_request",
                    "application_handling": "unknown",
                }, "curl_query_parameter_parser_v1"))
                reflected = any(
                    value and len(value) >= 8 and (
                        value in response_body
                        or urllib.parse.quote(value, safe="") in response_body
                    )
                    for value in values
                )
                if reflected:
                    facts.append((EvidenceStage.REFLECTION_FOUND, {
                        **origin,
                        "route": route,
                        "parameter": parameter,
                        "literal_value_in_response_body": True,
                        "content_type": content_type,
                        "browser_execution_tested": False,
                    }, "curl_literal_query_reflection_parser_v1"))
                probe_quoted = any(
                    value and ("'" in value or "%27" in value.lower())
                    for value in values
                )
                if probe_quoted:
                    signature = _sql_error_in_body(response_body)
                    if signature:
                        facts.append((EvidenceStage.SQL_ERROR_EXPOSED, {
                            **origin,
                            "route": route,
                            "parameter": parameter,
                            "signature": signature,
                            "note": ("a database error surfaced in the response to a quoted "
                                     "probe; the workflow verifies exploitation manually"),
                        }, "curl_sql_error_parser_v1"))

    return facts


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
    operation: str | None = None
    cwd: str | None = None
    failure_type: str | None = None
    process: dict | None = None
    workflow_purpose: str | None = None
    workflow_hypothesis: str | None = None
    expected_result: str | None = None

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
            operation=str(result["operation"]) if result.get("operation") else None,
            cwd=str(result["cwd"]) if result.get("cwd") else None,
            failure_type=str(result["failure_type"]) if result.get("failure_type") else None,
            process=dict(result["process"]) if isinstance(result.get("process"), dict) else None,
            workflow_purpose=str(result["workflow_purpose"]) if result.get("workflow_purpose") else None,
            workflow_hypothesis=str(result["workflow_hypothesis"]) if result.get("workflow_hypothesis") else None,
            expected_result=str(result["expected_result"]) if result.get("expected_result") else None,
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
        self.facts: list[EvidenceFact] = []

    def record_command(self, result: dict) -> CommandEvidence:
        record = dict(result)
        evidence_id = record.get("evidence_id")
        if (not isinstance(evidence_id, str) or not evidence_id.strip()
                or evidence_id in self.commands):
            record["evidence_id"] = uuid.uuid4().hex
        evidence = CommandEvidence.from_result(record)
        self.commands[evidence.evidence_id] = evidence
        for stage, value, validator in _derive_facts(evidence):
            self.record_fact(evidence.evidence_id, stage, value, validator=validator)
        return evidence

    def command_by_evidence_id(self, evidence_id: str | None) -> CommandEvidence | None:
        if not evidence_id:
            return None
        return self.commands.get(evidence_id)

    def facts_for_evidence_id(self, evidence_id: str | None) -> list[EvidenceFact]:
        if not evidence_id:
            return []
        return [fact for fact in self.facts if fact.evidence_id == evidence_id]

    def record_fact(
        self,
        evidence_id: str,
        stage: EvidenceStage,
        value: dict,
        *,
        validator: str,
    ) -> EvidenceFact:
        try:
            stage = EvidenceStage(stage)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Unknown evidence fact stage: {stage!r}") from exc
        if stage is EvidenceStage.SECURITY_FINDING_VERIFIED and validator != "workflow_marker_match_v1":
            raise ValueError(
                "SECURITY_FINDING_VERIFIED is recorded only by the workflow marker-matching "
                "validator; reflection or a model conclusion cannot mark a finding verified."
            )
        if evidence_id not in self.commands:
            raise ValueError(f"Fact references unknown command evidence: {evidence_id}")
        if not isinstance(value, dict):
            raise ValueError("Evidence fact values must be dictionaries.")
        evidence = self.commands[evidence_id]
        if (evidence.execution_state != "completed" or evidence.exit_code != 0
                or evidence.timed_out or evidence.output_truncated):
            raise ValueError("Facts require a completed, successful command with complete output.")
        if validator not in _FACT_VALIDATORS.get(stage, set()):
            raise ValueError(f"No controller parser is registered for {stage.value} / {validator!r}.")
        fact = EvidenceFact(
            fact_id=uuid.uuid4().hex,
            evidence_id=evidence_id,
            stage=stage,
            value=dict(value),
            validator=str(validator),
        )
        self.facts.append(fact)
        return fact

    def record_claim(
        self,
        claim_id: str,
        claim: str,
        evidence_ids: list[str],
        *,
        state: EvidenceState = EvidenceState.INFERRED,
        validator: str | None = None,
    ) -> ClaimEvidence:
        try:
            state = EvidenceState(state)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Unknown claim evidence state: {state!r}") from exc
        if state in {EvidenceState.VERIFIED, EvidenceState.CONTRADICTED}:
            raise ValueError(
                "No deterministic claim validator is implemented; verified or contradicted "
                "claims cannot be recorded directly."
            )
        if validator:
            raise ValueError("A validator name is valid only after its controller check has run.")
        if state is not EvidenceState.UNVERIFIED and not evidence_ids:
            raise ValueError("Observed or inferred claims require linked command evidence.")
        known_ids = {item.evidence_id for item in self.commands.values()}
        unknown_ids = sorted(set(evidence_ids) - known_ids)
        if unknown_ids:
            raise ValueError(f"Claim references unknown command evidence: {', '.join(unknown_ids)}")
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
            "facts": [item.to_dict() for item in self.facts],
        }

    def clear(self) -> None:
        self.commands.clear()
        self.claims.clear()
        self.facts.clear()
