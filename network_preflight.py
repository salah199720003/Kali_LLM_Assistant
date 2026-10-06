"""Network facts and callback-plan checks for the unrestricted lab agent."""

from __future__ import annotations

import ipaddress
import json
import re
import shlex


_IPV4 = re.compile(r"(?<![\d.])(?:\d{1,3}\.){3}\d{1,3}(?![\d.])")
_TARGET_SETTING = re.compile(
    r"(?im)(?:^|[;\n])\s*(?:set\s+)?(?:RHOSTS?|TARGET(?:_HOST)?)"
    r"[ \t]*(?:=[ \t]*|[ \t]+)"
    r"['\"]?((?:\d{1,3}\.){3}\d{1,3})\b"
)
_NETWORK_TASK = re.compile(r"\b(?:attack|exploit|pentest|penetration\s+test)\b", re.I)
_ATTACK_TOOL = re.compile(
    r"\b(?:msfvenom|exploit(?:\.py)?|meterpreter|eternalblue|"
    r"ms17_010_(?:psexec|eternalblue)|reverse_tcp|shellcode|SMB_COM_TRANSACTION)\b|"
    r"\buse\s+exploit/|\bmsfconsole\b[^\n]*(?:-r\s+|--resource\s+)",
    re.I,
)
_LHOST_SETTING = re.compile(
    r"(?i)(?:^|[;\n])\s*(?:set(?:g)?\s+)?LHOST[ \t]*(?:=[ \t]*|[ \t]+)"
    r"['\"]?(\d{1,3}(?:\.\d{1,3}){3})\b"
)
_BIND_SETTING = re.compile(
    r"(?i)(?:^|[;\n])\s*(?:set(?:g)?\s+)?ReverseListenerBindAddress"
    r"[ \t]*(?:=[ \t]*|[ \t]+)"
    r"(\d{1,3}(?:\.\d{1,3}){3}|0\.0\.0\.0|::)(?=[;\s]|$)"
)
_OPENED_SESSION = re.compile(
    r"(?i)\b(?:meterpreter|command shell)\s+session\s+\d+\s+opened\s+"
    r"\((?P<local>(?:\d{1,3}\.){3}\d{1,3}):\d+\s+->\s+"
    r"(?P<remote>(?:\d{1,3}\.){3}\d{1,3}):\d+\)"
)


def ipv4_in_text(text: str) -> str | None:
    for match in _IPV4.finditer(str(text)):
        try:
            return str(ipaddress.IPv4Address(match.group()))
        except ipaddress.AddressValueError:
            continue
    return None


def target_for_request(request: str, explicit_target: str | None = None) -> str | None:
    if explicit_target:
        try:
            return str(ipaddress.IPv4Address(explicit_target))
        except ipaddress.AddressValueError:
            return None
    configured_targets = targets_from_command(request)
    if configured_targets:
        return configured_targets[0]
    return ipv4_in_text(request)


def target_from_command(command: str) -> str | None:
    targets = targets_from_command(command)
    if targets:
        return targets[0]
    without_listener_settings = _LHOST_SETTING.sub(" ", command)
    without_listener_settings = _BIND_SETTING.sub(" ", without_listener_settings)
    return ipv4_in_text(without_listener_settings)


def targets_from_command(command: str) -> list[str]:
    """Return literal targets configured with RHOST(S) or TARGET settings."""
    targets = []
    for match in _TARGET_SETTING.finditer(command):
        try:
            target = str(ipaddress.IPv4Address(match.group(1)))
        except ipaddress.AddressValueError:
            continue
        if target not in targets:
            targets.append(target)
    return targets


def should_preflight(request: str, target: str | None) -> bool:
    return bool(target and _NETWORK_TASK.search(request))


def preflight_command(target: str) -> str:
    target = str(ipaddress.IPv4Address(target))
    return f"ip -j -4 address show && ip -j -4 route get {shlex.quote(target)}"


def parse_preflight(record: dict, target: str) -> dict:
    """Parse only JSON emitted by the fixed `ip` command; never infer reachability."""
    result = {"target": target, "observed": False, "addresses": [], "route": None,
              "source_address": None, "error": None}
    if record.get("execution_state") != "completed" or record.get("exit_code") != 0:
        result["error"] = "Kali did not complete the interface and target-route check."
        return result
    lines = [line.strip() for line in str(record.get("stdout", "")).splitlines() if line.strip()]
    try:
        interface_data = json.loads(lines[0])
        route_data = json.loads(lines[1])
        if not isinstance(interface_data, list) or not isinstance(route_data, list) or not route_data:
            raise ValueError("`ip` returned no interface or target-route data")
        route = route_data[0]
        if not isinstance(route, dict):
            raise ValueError("`ip` returned an invalid target-route entry")
        for interface in interface_data:
            if not isinstance(interface, dict):
                continue
            for address in interface.get("addr_info", []):
                local = address.get("local") if isinstance(address, dict) else None
                if not local:
                    continue
                parsed = str(ipaddress.ip_address(local))
                result["addresses"].append({"interface": interface.get("ifname"),
                                             "address": parsed,
                                             "prefixlen": address.get("prefixlen")})
        result["route"] = route
        source = route.get("prefsrc") or route.get("src")
        if source:
            result["source_address"] = str(ipaddress.ip_address(source))
        observed_addresses = {item["address"] for item in result["addresses"]}
        route_target = route.get("dst")
        if route_target and str(ipaddress.ip_address(route_target)) != target:
            raise ValueError("the returned route does not match the requested target")
        result["observed"] = bool(
            result["addresses"] and result["source_address"] in observed_addresses
        )
        if not result["observed"]:
            result["error"] = "The command did not identify both a Kali interface address and a route source."
    except (IndexError, TypeError, ValueError) as exc:
        result["error"] = f"Could not parse the Kali interface and target-route output: {exc}"
    return result


def is_attack_attempt(command: str) -> bool:
    if _ATTACK_TOOL.search(command):
        return True
    lowered = command.lower()
    return ("struct.pack" in lowered and "socket." in lowered
            and any(marker in lowered for marker in ("smb", "shellcode", "payload")))


def callback_settings(command: str) -> list[str]:
    values = []
    for match in _LHOST_SETTING.finditer(command):
        try:
            values.append(str(ipaddress.IPv4Address(match.group(1))))
        except ipaddress.AddressValueError:
            values.append(match.group(1))
    return values


def bind_settings(command: str) -> list[str]:
    return [match.group(1) for match in _BIND_SETTING.finditer(command)]


def validate_plan(plan: object, assessment: dict, command: str,
                  evidence_lookup) -> str | None:
    """Check the model's stated plan against observed Kali addresses and route."""
    if not isinstance(plan, dict):
        return "Before the first exploit command, include a network_plan with the target, Kali source address, listener bind address, callback address, and callback reachability status."
    if not assessment.get("observed"):
        return "The Kali interface and route check is incomplete. Inspect `ip address` and `ip route get TARGET`, then state the observed network position before an exploit attempt."
    try:
        target = str(ipaddress.IPv4Address(plan.get("target", "")))
        source = str(ipaddress.IPv4Address(plan.get("local_address", "")))
    except ipaddress.AddressValueError:
        return "network_plan.target and network_plan.local_address must be IPv4 addresses observed in the preflight."
    if target != assessment["target"]:
        return f"network_plan.target must match the observed target {assessment['target']}."
    command_targets = targets_from_command(command)
    if len(command_targets) > 1:
        return "Use one target IP per exploit command so the route and callback plan match the host being tested."
    if command_targets and command_targets[0] != target:
        return f"This command targets {command_targets[0]}, but its network_plan and route check cover {target}."
    observed_addresses = {item["address"] for item in assessment["addresses"]}
    if source not in observed_addresses or source != assessment["source_address"]:
        return (f"Kali's observed route to {target} uses source {assessment['source_address']}; "
                "network_plan.local_address must use that source address.")
    bind = str(plan.get("listener_bind_address", "")).strip()
    if bind not in observed_addresses | {"0.0.0.0", "::"}:
        return "network_plan.listener_bind_address must be an address on Kali, or the explicit wildcard 0.0.0.0/::."
    explicit_bind = bind_settings(command)
    if explicit_bind and bind not in explicit_bind:
        return "network_plan.listener_bind_address does not match ReverseListenerBindAddress in this command."
    reachability = plan.get("callback_reachability")
    if reachability not in {"verified", "unverified", "not_applicable"}:
        return "State callback_reachability as verified, unverified, or not_applicable; a route alone does not prove a return connection."
    callback = str(plan.get("callback_address", "")).strip()
    if reachability == "not_applicable":
        if callback or callback_settings(command):
            return "State the callback address and its reachability when the command configures LHOST."
    else:
        try:
            callback = str(ipaddress.IPv4Address(callback))
        except ipaddress.AddressValueError:
            return "network_plan.callback_address must be an IPv4 address for a callback attempt."
        settings = callback_settings(command)
        if settings and any(setting != callback for setting in settings):
            return "network_plan.callback_address does not match every LHOST address in this command."
        if not explicit_bind and bind != callback:
            return ("The command's LHOST is also its default listener bind address. "
                    "Use that Kali interface address, or set ReverseListenerBindAddress explicitly.")
    if reachability == "verified":
        proof_id = plan.get("callback_evidence_id")
        evidence = evidence_lookup(proof_id) if isinstance(proof_id, str) else None
        proof = evidence.to_dict(include_streams=True) if evidence else {}
        output = str(proof.get("stdout", "")) + "\n" + str(proof.get("stderr", ""))
        valid_session = False
        for match in _OPENED_SESSION.finditer(output):
            try:
                local = str(ipaddress.IPv4Address(match.group("local")))
                remote = str(ipaddress.IPv4Address(match.group("remote")))
            except ipaddress.AddressValueError:
                continue
            if local == callback and remote == target:
                valid_session = True
                break
        if (not evidence or proof.get("execution_state") != "completed"
                or proof.get("exit_code") != 0 or proof.get("timed_out")
                or proof.get("output_truncated") or not valid_session):
            return ("Mark callback_reachability verified only with a recorded command-evidence ID "
                    "showing a successful, complete session from the target to the stated callback address; "
                    "otherwise state unverified.")
    return None


def is_bind_failure(output: str) -> bool:
    return bool(re.search(
        r"(?i)(?:handler failed to bind|failed to bind|cannot assign requested address|"
        r"can't assign requested address|address not available)", str(output),
    ))
