"""Local model HTTP transport, configured explicitly by the terminal controller."""

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Callable
from model_protocol import _ollama_messages
from model_streaming import _streamed_ollama_reply

@dataclass
class ClientSettings:
    model: str
    base_url: str
    api_key: str
    normalize_base_url: Callable
    live_generation: Callable
    k2_effort: Callable
    reasoning_override: Callable
    is_result_report: Callable
    action_budget: Callable
    tool_definitions: Callable
    normalize_reply: Callable
    streamed_reply: Callable
    call_from_text: Callable


def llama_chat(settings: ClientSettings, messages: list[dict], tools: bool | list[dict] = False, stream_output: bool = True) -> dict:
    strata_coder = settings.model.lower() == "qwen3.8-flash-next-coder-iq1_m"
    provider = "Strata" if strata_coder else "llama.cpp"
    base = settings.normalize_base_url(settings.base_url)
    parsed = urllib.parse.urlsplit(base)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("DEEP_AGENT_BASE_URL must be a plain URL such as http://127.0.0.1:8080/v1.")
    endpoint = base if base.endswith("/chat/completions") else f"{base}/chat/completions"
    payload = {
        "model": settings.model,
        "messages": messages,
        "stream": True,
        "stream_options": {"include_usage": True},
        # Request-local parsing: expose thoughts separately without changing
        # the shared server or the browser's reasoning/display preferences.
        "reasoning_format": "deepseek",
        "temperature": 0.2 if "deephat" in settings.model.lower() else 1.0,
        "top_p": 0.95,
        "max_tokens": 32768,
    }
    if settings.model.lower().startswith("k2-horizon"):
        payload["chat_template_kwargs"] = {"reasoning_effort": settings.k2_effort()}
    elif settings.model.lower().startswith("bonsai-2-27b"):
        # Model-card thinking profile. Bound generation independently of the
        # server's context window; native tools use --jinja.
        payload.update({"top_k": 20, "min_p": 0.05,
                        "presence_penalty": 0.0, "repeat_penalty": 1.0,
                        "max_tokens": 8192,
                        "chat_template_kwargs": {"reasoning_effort": settings.reasoning_override() or "medium"}})
    else:
        # Generic models (e.g. Qwen3.8 templates): unbounded thinking at slow
        # decode speeds freezes the loop, so DEEP_AGENT_THINKING gates it.
        thinking = os.environ.get("DEEP_AGENT_THINKING", "").strip().lower()
        if thinking in {"off", "false", "disabled", "0", "no"}:
            payload["chat_template_kwargs"] = {"enable_thinking": False}
        elif thinking in {"on", "true", "enabled", "1", "yes"}:
            payload["chat_template_kwargs"] = {"enable_thinking": True}
        override = settings.reasoning_override()
        if override is not None:
            payload["chat_template_kwargs"] = {"enable_thinking": override != "off"}
            if override != "off" and (settings.model.lower() == "agent-27b" or "qwen3.8" in settings.model.lower()):
                # Qwen3.8 defaults to xhigh when effort is omitted. The agent's
                # high command selects that tier; medium and low select their
                # own template instructions instead of falling back to xhigh.
                payload["chat_template_kwargs"]["reasoning_effort"] = (
                    "xhigh" if override == "high" else override
                )
    if settings.reasoning_override() == "off" and settings.model.lower().startswith(("k2-horizon", "bonsai-2-27b")):
        payload["chat_template_kwargs"] = {"enable_thinking": False}
        payload["reasoning_budget"] = 0
    if settings.is_result_report(messages, tools):
        # Reporting already-recorded results needs no extended thinking. Keep
        # the session's chosen reasoning mode for chat and tool selection.
        payload["chat_template_kwargs"] = {"enable_thinking": False}
        payload["reasoning_budget"] = 0
    if settings.model.lower() == "agent-27b" or strata_coder:
        # The Qwen terminals use the thinking-on sampling profile in both
        # modes. Reasoning choices and brief result reports stay independent.
        thinking_on = payload.get("chat_template_kwargs", {}).get("enable_thinking", True)
        # Each request gets a fresh ceiling; the investigation continues across
        # tool rounds. The server ends thinking before generating the tool call.
        # Tool-selection rounds get a tighter ceiling than chat: deliberation
        # beyond it produces spirals, not better commands.
        budget = 8192 if not tools else settings.action_budget()
        payload.pop("reasoning_budget", None)
        payload.update({
            "reasoning_budget_tokens": budget if thinking_on else 0,
            "temperature": 1.0,
            "top_p": 0.95,
            "top_k": 20, "min_p": 0.0,
            "repeat_penalty": 1.0,
            "presence_penalty": 0.0,
            "frequency_penalty": 0.0, "dry_multiplier": 0.0,
            "xtc_probability": 0.0, "mirostat": 0,
        })
        if strata_coder:
            # Strata's engine spells this field differently. Do not inherit
            # penalties from shared browser defaults when the agent asks for none.
            payload["repetition_penalty"] = payload.pop("repeat_penalty")
            for key in ("dry_multiplier", "xtc_probability", "mirostat", "reasoning_format"):
                payload.pop(key, None)
    tool_definitions = settings.tool_definitions(tools)
    if tool_definitions:
        payload["tools"] = tool_definitions
        payload["parallel_tool_calls"] = False
    headers = {"Content-Type": "application/json"}
    if settings.api_key:
        headers["Authorization"] = f"Bearer {settings.api_key}"
    request = urllib.request.Request(endpoint, data=json.dumps(payload).encode("utf-8"), headers=headers)
    try:
        with settings.live_generation(stream_output) as live, urllib.request.urlopen(request, timeout=180) as response:
            content_type = response.headers.get("Content-Type", "") if hasattr(response, "headers") else ""
            if "application/json" in content_type.lower():
                result = json.load(response)
                item = result["choices"][0]["message"]
                if not isinstance(item, dict):
                    raise ValueError("Malformed chat completion: expected a message object.")
                reasoning = item.get("reasoning_content") or item.get("reasoning")
                if isinstance(reasoning, str):
                    live.reasoning(reasoning)
                if isinstance(item.get("content"), str):
                    live.add(item["content"])
                if item.get("tool_calls"):
                    live.tool()
                result = settings.normalize_reply({**item, "usage": result.get("usage")})
            else:
                result = settings.streamed_reply(response, live)
                result = settings.normalize_reply(result)
            live.finish(result)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:500]
        raise ConnectionError(f"{provider} returned HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise ConnectionError(f"Cannot reach {provider} at {endpoint}. Start its model launcher and check DEEP_AGENT_BASE_URL. ({exc})") from exc
    except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Malformed chat completion from {provider}.") from exc
    return result

def ollama_chat(settings: ClientSettings, messages: list[dict], tools: bool | list[dict] = False, stream_output: bool = True) -> dict:
    host = os.environ.get("OLLAMA_HOST", "127.0.0.1:11434").strip()
    if "://" not in host:
        host = "http://" + host
    parsed = urllib.parse.urlsplit(host)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc or parsed.path not in {"", "/"}:
        raise ValueError("OLLAMA_HOST must be a plain host URL such as http://127.0.0.1:11434.")
    endpoint = host.rstrip("/") + "/api/chat"
    payload = {"model": settings.model, "messages": _ollama_messages(messages), "stream": True}
    tool_definitions = settings.tool_definitions(tools)
    if tool_definitions:
        payload["tools"] = tool_definitions
    requested_context = os.environ.get("DEEP_AGENT_OLLAMA_NUM_CTX", "").strip()
    if requested_context:
        try:
            num_ctx = int(requested_context)
        except ValueError as exc:
            raise ValueError("DEEP_AGENT_OLLAMA_NUM_CTX must be a positive integer.") from exc
        if num_ctx < 1:
            raise ValueError("DEEP_AGENT_OLLAMA_NUM_CTX must be a positive integer.")
        payload["options"] = {"num_ctx": num_ctx}
    requested_think = os.environ.get("DEEP_AGENT_OLLAMA_THINK", "").strip()
    if requested_think:
        payload["think"] = (requested_think.lower() == "true" if requested_think.lower() in {"true", "false"}
                             else requested_think)
    override = settings.reasoning_override()
    if override is not None:
        payload["think"] = (override if "gpt-oss" in settings.model.lower() and override != "off"
                            else override != "off")
    if settings.is_result_report(messages, tools):
        payload["think"] = False
    request = urllib.request.Request(
        endpoint, data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    try:
        with settings.live_generation(stream_output) as live, urllib.request.urlopen(request, timeout=600) as response:
            content_type = response.headers.get("Content-Type", "") if hasattr(response, "headers") else ""
            if "ndjson" in content_type.lower():
                reply = _streamed_ollama_reply(response, live)
            else:
                # Retain compatibility with proxies returning one JSON reply.
                reply = json.load(response)
                message = reply.get("message") if isinstance(reply, dict) else None
                if isinstance(message, dict):
                    if isinstance(message.get("thinking"), str):
                        live.reasoning(message["thinking"])
                    if isinstance(message.get("content"), str):
                        live.add(message["content"])
                    if message.get("tool_calls"):
                        live.tool()
            message = reply.get("message") if isinstance(reply, dict) else None
            if not isinstance(message, dict):
                raise ValueError("Malformed chat response from Ollama: expected a message object.")
            content = message.get("content") or ""
            result = settings.normalize_reply({"content": content, "tool_calls": message.get("tool_calls") or [], "usage": {
                "prompt_tokens": reply.get("prompt_eval_count"),
                "completion_tokens": reply.get("eval_count"),
            }})
            live.finish(result)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:500]
        raise ConnectionError(f"Ollama returned HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise ConnectionError(f"Cannot reach Ollama at {endpoint}. Start Ollama and check OLLAMA_HOST. ({exc})") from exc
    except (json.JSONDecodeError, TypeError) as exc:
        raise ValueError("Malformed chat response from Ollama.") from exc
    if not result["tool_calls"] and tools:
        textual = settings.call_from_text(content)
        if textual:
            result = {"content": "", "tool_calls": [textual], "usage": result.get("usage")}
    return result
