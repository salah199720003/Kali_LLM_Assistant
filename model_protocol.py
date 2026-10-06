"""Tool-call decoding and backend history adaptation, without execution policy."""

import json
import re
import uuid

def _balanced_json_block(text: str) -> str | None:
    """Return the first complete balanced {...} or [...] block in text, if any."""
    start = None
    opener = closer = ""
    depth = 0
    quote = None
    escaped = False
    for index, character in enumerate(text):
        if quote:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == quote:
                quote = None
            continue
        if character in {"'", '"'}:
            if start is not None:
                quote = character
            continue
        if character in "{[":
            if start is None:
                start = index
                opener, closer = character, ("}" if character == "{" else "]")
            depth += 1
        elif character in "}]":
            if start is not None:
                depth -= 1
                if depth == 0 and character == closer:
                    return text[start:index + 1]
    return None


def _repair_json_text(text: str):
    """Leniently recover a JSON value local models commonly mal-form."""
    for candidate in (text, _balanced_json_block(text or "")):
        if not candidate:
            continue
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass
        repaired = re.sub(r",\s*([}\]])", r"\1", candidate)
        try:
            return json.loads(repaired)
        except json.JSONDecodeError:
            continue
    return None


def _call_from_text(content: str) -> dict | None:
    """Recover a single K2/IFM text-form command request when native calls fail."""
    raw = (content or "").strip()
    if raw.startswith("```"):
        _, separator, raw = raw.partition("\n")
        if not separator:
            return None
        raw = raw.rsplit("```", 1)[0].strip()
    if "<ifm|tool_calls>" in raw:
        match = re.search(r"<ifm\|tool_call>(.*?)</ifm\|tool_call>", raw, re.S)
        if not match:
            return None
        body = match.group(1).strip()
        name_match = re.match(r"([A-Za-z_][A-Za-z0-9_]*)", body)
        if not name_match:
            return None
        args = {}
        for arg in re.finditer(r"<ifm\|arg_key>(.*?)</ifm\|arg_key>\s*<ifm\|arg_value>(.*?)</ifm\|arg_value>", body, re.S):
            args[arg.group(1).strip()] = arg.group(2).strip()
        return _make_tool_call(name_match.group(1), args)
    if raw.startswith("<tool_call>"):
        raw = raw[len("<tool_call>"):].strip()
    data = None
    try:
        data, _ = json.JSONDecoder().raw_decode(raw)
    except json.JSONDecodeError:
        # Quantized local models emit trailing commas, prose around the JSON,
        # or trailing junk; recover before declaring the reply unusable.
        data = _repair_json_text(raw)
    if isinstance(data, list):
        data = data[0] if data else None
    if not isinstance(data, dict):
        return None
    # Some K2 builds emit their command plan as JSON instead of a function call.
    if isinstance(data.get("command"), str) and any(key in data for key in ("analysis", "plan", "output_format")):
        if data["command"].strip().lower() in {"", "none", "null"}:
            return None
        arguments = {"command": data["command"]}
        for key in ("cwd", "purpose", "hypothesis", "expected_result"):
            if key in data:
                arguments[key] = data[key]
        return _make_tool_call("run_kali_command", arguments)
    name = data.get("name")
    arguments = data.get("arguments")
    if name and isinstance(arguments, dict):
        return _make_tool_call(name, arguments)
    return None


def _make_tool_call(name: str, arguments: dict, call_id: str | None = None) -> dict:
    aliases = {"run_vm_command": "run_kali_command", "run_pentest_command": "run_kali_command"}
    return {"id": call_id or uuid.uuid4().hex, "type": "function", "function": {
        "name": aliases.get(name, name), "arguments": json.dumps(arguments, ensure_ascii=False)}}


def _normalize_reply(message: dict, allow_text_tool: bool = True, *, unrestricted=False) -> dict:
    content = message.get("content") or ""
    calls = []
    for call in message.get("tool_calls") or []:
        if not isinstance(call, dict) or not isinstance(call.get("function"), dict):
            raise ValueError("The model returned a malformed tool call; no Kali command ran.")
        fn = call["function"]
        name = fn.get("name")
        args = fn.get("arguments", {})
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError as exc:
                raise ValueError("The model returned malformed tool arguments; no Kali command ran.") from exc
        if not isinstance(name, str) or not isinstance(args, dict):
            raise ValueError("The model returned malformed tool arguments; no Kali command ran.")
        calls.append(_make_tool_call(name, args, call.get("id")))
    if not calls and allow_text_tool:
        textual = _call_from_text(content)
        if textual:
            calls = [textual]
    # K2's template leaks private reasoning into content as
    # "…thoughts…</ifm|think>answer" or a trailing unterminated "<ifm|think>…";
    # keep the answer side so leaks are neither shown nor stored in history.
    content = content or ""
    if "</ifm|think>" in content:
        content = content.rsplit("</ifm|think>", 1)[1]
    content = re.sub(r"(?s)<ifm\|think>.*$", "", content).strip()
    return {"content": "" if calls else content,
            "tool_calls": calls if unrestricted else calls[:1], "usage": message.get("usage")}

def _ollama_messages(messages: list[dict]) -> list[dict]:
    """Adapt stored JSON-string tool arguments to Ollama's object-shaped schema."""
    adapted = []
    for message in messages:
        item = dict(message)
        calls = item.get("tool_calls") or []
        if calls:
            normalized_calls = []
            for call in calls:
                if not isinstance(call, dict) or not isinstance(call.get("function"), dict):
                    raise ValueError("Malformed tool history for Ollama; no Kali command ran.")
                function = dict(call["function"])
                arguments = function.get("arguments", {})
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except json.JSONDecodeError as exc:
                        raise ValueError("Malformed tool arguments in Ollama history; no Kali command ran.") from exc
                if not isinstance(arguments, dict):
                    raise ValueError("Malformed tool arguments in Ollama history; no Kali command ran.")
                function["arguments"] = arguments
                normalized_call = dict(call)
                normalized_call["function"] = function
                normalized_calls.append(normalized_call)
            item["tool_calls"] = normalized_calls
        adapted.append(item)
    return adapted
