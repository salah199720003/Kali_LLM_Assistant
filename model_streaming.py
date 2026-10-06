"""Terminal generation display and SSE/NDJSON assembly."""

import json
import sys
import threading
import time

TOOL_MARKERS = ("<ifm|tool_call", "<tool_call>")
HELD_PREFIXES = ("{", "[", "<ifm|", "<tool_call>", "```")

class LiveAnswer:
    """Stream ordinary text, buffering replies that begin like tool or plan data."""

    def __init__(self, label: str | None = None, *, assistant_name="Agent", answer_filter=lambda text, tool=False: text.strip()):
        self.answer_filter = answer_filter
        self.content = ""
        self.printed = 0
        self.started = False
        self.held = False
        self.label = label or f"{assistant_name}> "

    def add(self, fragment: str) -> None:
        if not fragment:
            return
        self.content += fragment
        if self.held:
            return
        candidate = self.content.lstrip()
        if not self.started:
            if any(candidate.startswith(prefix) for prefix in HELD_PREFIXES):
                self.held = True
                return
            if any(prefix.startswith(candidate) for prefix in HELD_PREFIXES):
                return
            print(f"\n{self.label}", end="", flush=True)
            self.started = True
        marker_positions = [self.content.find(marker, self.printed) for marker in TOOL_MARKERS]
        marker_positions = [position for position in marker_positions if position >= 0]
        if marker_positions:
            self.held = True
            end = min(marker_positions)
        else:
            # Hold only an incomplete tool marker, rather than delaying every
            # answer by twenty characters (especially noticeable on short replies).
            pending = max((length for marker in TOOL_MARKERS
                           for length in range(1, len(marker))
                           if self.content.endswith(marker[:length])), default=0)
            end = max(self.printed, len(self.content) - pending)
        if end > self.printed:
            print(self.content[self.printed:end], end="", flush=True)
            self.printed = end

    def finish(self, tool_requested: bool = False) -> str:
        if tool_requested:
            if self.started:
                print(flush=True)
            return ""
        answer = self.answer_filter(self.content, tool_requested)
        if self.started and answer == self.content.strip():
            print(self.content[self.printed:], flush=True)
        else:
            if self.started:
                print()
            print(f"\n{self.label}{answer}", flush=True)
        return answer


class LiveGeneration:
    """Show backend thinking and generation without treating drafts as evidence."""

    def __init__(self, stream_output: bool, *, assistant_name="Agent", unrestricted=False, answer_filter=lambda text, tool=False: text.strip()):
        self.assistant_name = assistant_name
        self.stream_output = stream_output
        label = (None if stream_output else
                 f"{self.assistant_name} [live]> " if unrestricted else
                 f"{self.assistant_name} [live draft; pending controller checks]> ")
        self.answer = LiveAnswer(label, assistant_name=assistant_name, answer_filter=answer_filter)
        self.waiting = threading.Event()
        self.worker = None
        self.wait_line_width = 0
        self.started_at = time.monotonic()
        self.first_output_at = None
        self.thinking_open = False
        self.tool_announced = False

    def __enter__(self):
        print(f"\n[{self.assistant_name}: waiting for model / processing prompt...]", flush=True)
        if sys.stdout.isatty():
            self.worker = threading.Thread(target=self._wait_status, daemon=True)
            self.worker.start()
        return self

    def _wait_status(self):
        while not self.waiting.wait(1):
            elapsed = time.monotonic() - self.started_at
            line = f"[{self.assistant_name}: waiting for model / processing prompt... {elapsed:.0f}s]"
            self.wait_line_width = max(self.wait_line_width, len(line))
            print("\r" + line.ljust(self.wait_line_width), end="", flush=True)

    def _stop_waiting(self):
        self.waiting.set()
        if self.worker is not None:
            self.worker.join()
            self.worker = None
        if self.wait_line_width:
            print("\r" + " " * self.wait_line_width + "\r", end="", flush=True)
            self.wait_line_width = 0

    def _output_started(self):
        if self.first_output_at is None:
            self.first_output_at = time.monotonic()
            self._stop_waiting()

    def _end_thinking(self):
        if self.thinking_open:
            print(flush=True)
            self.thinking_open = False

    def reasoning(self, fragment: str):
        if not fragment:
            return
        self._output_started()
        if not self.thinking_open:
            print(f"\n{self.assistant_name} [thinking]> ", end="", flush=True)
            self.thinking_open = True
        print(fragment, end="", flush=True)

    def add(self, fragment: str):
        if fragment:
            self._output_started()
            self._end_thinking()
            self.answer.add(fragment)

    def tool(self):
        self._output_started()
        self._end_thinking()
        if not self.tool_announced:
            if self.answer.started:
                print(flush=True)
            print("[Generating tool call; command has not run yet...]", flush=True)
            self.tool_announced = True

    def finish(self, result: dict):
        self._stop_waiting()
        self._end_thinking()
        self.answer.content = result["content"]
        if self.stream_output:
            result["content"] = self.answer.finish(bool(result["tool_calls"]))
        elif self.answer.content or self.answer.started:
            self.answer.finish(bool(result["tool_calls"]))
        usage = result.get("usage") or {}
        tokens = usage.get("completion_tokens")
        if type(tokens) is int and tokens >= 0:
            elapsed = time.monotonic() - self.started_at
            print(f"[Generation complete: {tokens:,} model tokens; {elapsed:.1f}s total.]", flush=True)

    def __exit__(self, exc_type, *_):
        self._stop_waiting()
        self._end_thinking()
        if exc_type is not None and self.first_output_at is not None:
            print("\n[Generation stopped; displayed output is incomplete.]", flush=True)


def _streamed_reply(response, live: LiveAnswer | LiveGeneration | None = None) -> dict:
    """Read llama.cpp SSE content and assemble streamed function calls."""
    data_lines, content, calls = [], "", {}
    saw_choice = completed = done = False
    usage = None

    def accept(data: str) -> None:
        nonlocal content, saw_choice, completed, done, usage
        if data == "[DONE]":
            done = True
            return
        try:
            event = json.loads(data)
        except json.JSONDecodeError as exc:
            raise ValueError("Malformed streaming reply from llama.cpp: invalid JSON.") from exc
        if not isinstance(event, dict):
            raise ValueError("Malformed streaming reply from llama.cpp: expected an object.")
        if event.get("error"):
            raise ConnectionError(f"llama.cpp streaming error: {event['error']}")
        if isinstance(event.get("usage"), dict):
            usage = event["usage"]
        choices = event.get("choices") or []
        if not choices:
            return
        choice = choices[0]
        if not isinstance(choice, dict):
            raise ValueError("Malformed streaming reply from llama.cpp: invalid choice.")
        saw_choice = True
        if choice.get("finish_reason") is not None:
            completed = True
        delta = choice.get("delta") or {}
        if not isinstance(delta, dict):
            raise ValueError("Malformed streaming reply from llama.cpp: invalid delta.")
        reasoning = delta.get("reasoning_content") or delta.get("reasoning")
        if isinstance(reasoning, str) and isinstance(live, LiveGeneration):
            live.reasoning(reasoning)
        fragment = delta.get("content")
        if isinstance(fragment, str):
            content += fragment
            if live is not None:
                live.add(fragment)
        for part in delta.get("tool_calls") or []:
            if isinstance(live, LiveGeneration):
                live.tool()
            if not isinstance(part, dict) or not isinstance(part.get("index"), int):
                raise ValueError("Malformed streaming tool call: missing index.")
            call = calls.setdefault(part["index"], {"id": "", "type": "function", "function": {"name": "", "arguments": ""}})
            if part.get("id"):
                call["id"] += str(part["id"])
            function = part.get("function") or {}
            if not isinstance(function, dict):
                raise ValueError("Malformed streaming tool call: invalid function.")
            for key in ("name", "arguments"):
                value = function.get(key)
                if value is not None:
                    if key == "arguments" and isinstance(value, (dict, list)):
                        value = json.dumps(value)
                    if not isinstance(value, str):
                        raise ValueError(f"Malformed streaming tool call: invalid {key} fragment.")
                    call["function"][key] += value

    for raw_line in response:
        line = raw_line.decode("utf-8", errors="replace") if isinstance(raw_line, bytes) else str(raw_line)
        line = line.rstrip("\r\n")
        if line.startswith("data:"):
            data_lines.append(line[5:].lstrip())
        elif not line and data_lines:
            accept("\n".join(data_lines))
            data_lines.clear()
            if done:
                break
    if data_lines and not done:
        accept("\n".join(data_lines))
    if not saw_choice or not completed:
        raise ValueError("Incomplete streaming reply from llama.cpp.")
    return {"content": content, "tool_calls": [calls[index] for index in sorted(calls)], "usage": usage}

def _streamed_ollama_reply(response, live: LiveGeneration) -> dict:
    """Accumulate Ollama's NDJSON chunks, requiring its terminal done event."""
    content, calls = "", []
    for raw_line in response:
        if not raw_line.strip():
            continue
        try:
            event = json.loads(raw_line)
        except (TypeError, ValueError) as exc:
            raise ValueError("Malformed streaming reply from Ollama: invalid JSON.") from exc
        if not isinstance(event, dict):
            raise ValueError("Malformed streaming reply from Ollama: expected an object.")
        if event.get("error"):
            raise ConnectionError(f"Ollama streaming error: {event['error']}")
        message = event.get("message")
        if not isinstance(message, dict):
            raise ValueError("Malformed streaming reply from Ollama: expected a message object.")
        thinking = message.get("thinking")
        if isinstance(thinking, str):
            live.reasoning(thinking)
        fragment = message.get("content")
        if isinstance(fragment, str):
            content += fragment
            live.add(fragment)
        chunk_calls = message.get("tool_calls") or []
        if not isinstance(chunk_calls, list):
            raise ValueError("Malformed streaming reply from Ollama: invalid tool calls.")
        if chunk_calls:
            live.tool()
            calls.extend(chunk_calls)
        if event.get("done") is True:
            return {**event, "message": {"content": content, "tool_calls": calls}}
    raise ValueError("Incomplete streaming reply from Ollama.")
