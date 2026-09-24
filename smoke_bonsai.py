"""Live chat + native tool round trip; --kali runs only hostname over SSH."""

import argparse
import json
import time
import urllib.request

import deep_agent
from kali_access import KaliAccess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kali", action="store_true", help="Execute the allowlisted hostname call on Kali")
    args = parser.parse_args()
    if deep_agent.MODEL != "bonsai-2-27b":
        raise RuntimeError("This smoke check requires DEEP_AGENT_MODEL=bonsai-2-27b.")
    base = deep_agent.BASE_URL.rstrip("/")
    with urllib.request.urlopen(f"{base}/models", timeout=5) as response:
        models = json.load(response)
    if "bonsai-2-27b" not in [item["id"] for item in models["data"]]:
        raise RuntimeError("The endpoint is not serving Bonsai 2 27B.")

    started = time.monotonic()
    reply = deep_agent._llama_chat([
        {"role": "system", "content": deep_agent.CHAT_SYSTEM_PROMPT},
        {"role": "user", "content": "Reply with exactly BONSAI_READY and nothing else."},
    ], stream_output=False)
    if reply["content"].strip() != "BONSAI_READY" or reply["tool_calls"]:
        raise AssertionError(f"Chat smoke failed: {reply['content']!r}")
    print(f"PASS streamed chat ({time.monotonic() - started:.2f}s)", flush=True)

    messages = [
        {"role": "system", "content": deep_agent.SHELL_SYSTEM_PROMPT},
        {"role": "user", "content": "Use run_kali_command with exactly hostname to check Kali's hostname. After the tool result, reply with only the observed hostname. Do not guess it."},
    ]
    payload = {
        "model": deep_agent.MODEL, "messages": messages,
        "tools": [deep_agent.KALI_TOOL], "parallel_tool_calls": False,
        "temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.05,
        "max_tokens": 8192, "chat_template_kwargs": {"reasoning_effort": "medium"},
    }
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {deep_agent.API_KEY}"}
    request = urllib.request.Request(f"{base}/chat/completions", data=json.dumps(payload).encode(), headers=headers)
    started = time.monotonic()
    print("Checking native tool call...", flush=True)
    with urllib.request.urlopen(request, timeout=180) as response:
        result = json.load(response)
    message = result["choices"][0]["message"]
    calls = message.get("tool_calls") or []
    if len(calls) != 1 or calls[0]["function"]["name"] != "run_kali_command":
        raise AssertionError("Server did not return exactly one native run_kali_command call.")
    arguments = json.loads(calls[0]["function"]["arguments"])
    if arguments != {"command": "hostname"}:
        raise AssertionError(f"Refusing unexpected command: {arguments!r}")
    print(f"PASS native tool call: hostname ({time.monotonic() - started:.2f}s)", flush=True)

    kali = KaliAccess()
    try:
        if args.kali:
            kali.run("hostname")
            record = kali.last_record
            if record["execution_state"] != "completed" or record["exit_code"] != 0:
                raise AssertionError("Live Kali hostname check failed.")
            hostname = record["stdout"].strip()
            print("PASS live Kali execution and exit status", flush=True)
        else:
            hostname = "bonsai-fixture-host"
            record = {"command": "hostname", "execution_state": "completed",
                      "exit_code": 0, "stdout": hostname + "\n", "stderr": ""}
            print("Using a synthetic tool result; no Kali command executed.", flush=True)
        messages.append({"role": "assistant", "content": message.get("content") or "", "tool_calls": calls})
        messages.append({"role": "tool", "tool_call_id": calls[0]["id"], "content": json.dumps(record)})
        started = time.monotonic()
        reply = deep_agent._llama_chat(messages, stream_output=False)
        if reply["tool_calls"] or reply["content"].strip() != hostname:
            raise AssertionError(f"Tool result round trip failed: {reply['content']!r}")
        print(f"PASS streamed tool-result summary: {hostname} ({time.monotonic() - started:.2f}s)", flush=True)
    finally:
        kali.close()


if __name__ == "__main__":
    main()
