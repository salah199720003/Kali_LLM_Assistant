"""Strata agent request controls and native tool transport (no Kali execution)."""
import contextlib
import io
import json
import os
import unittest
from unittest.mock import patch

import deep_agent
from execution_mode import unrestricted_execution_enabled

MODEL = "qwen3.8-flash-next-coder-iq1_m"
BASE = "http://127.0.0.1:8082/v1"


class StrataAgentTests(unittest.TestCase):
    def setUp(self):
        self.context = contextlib.ExitStack()
        self.addCleanup(self.context.close)
        self.context.enter_context(patch.object(deep_agent, "MODEL", MODEL))
        self.context.enter_context(patch.object(deep_agent, "BASE_URL", BASE))
        self.context.enter_context(patch.dict(os.environ, {
            "DEEP_AGENT_THINKING": "on", "DEEP_AGENT_MODEL": MODEL,
            "DEEP_AGENT_EXECUTION_MODE": "unrestricted",
            "DEEP_AGENT_ACTION_THINKING_TOKENS": "2048",
        }))
        self.setting = {"sticky": None, "effort": None}
        self.context.enter_context(patch.object(deep_agent, "REASONING_OVERRIDE", self.setting))
        self.sent = []

        def respond(request, **_):
            self.sent.append((request.full_url, json.loads(request.data)))
            reply = io.BytesIO(json.dumps({"choices": [{"message": {"content": "Ready"}}]}).encode())
            reply.headers = {"Content-Type": "application/json"}
            return reply

        self.context.enter_context(patch.object(deep_agent.urllib.request, "urlopen", side_effect=respond))
        self.context.enter_context(contextlib.redirect_stdout(io.StringIO()))

    def test_session_efforts_and_off_are_explicit_agent_requests(self):
        for effort in ("low", "medium", "high", "off"):
            self.setting["sticky"] = effort
            deep_agent._llama_chat([{"role": "user", "content": "Hello"}], stream_output=False)
        requests = [payload for _, payload in self.sent]
        self.assertEqual([p["chat_template_kwargs"] for p in requests], [
            {"enable_thinking": True, "reasoning_effort": "low"},
            {"enable_thinking": True, "reasoning_effort": "medium"},
            {"enable_thinking": True, "reasoning_effort": "xhigh"},
            {"enable_thinking": False},
        ])
        self.assertEqual([p["reasoning_budget_tokens"] for p in requests], [8192, 8192, 8192, 0])
        self.assertTrue(all(url == BASE + "/chat/completions" for url, _ in self.sent))
        for request in requests:
            self.assertEqual(request["model"], MODEL)
            self.assertEqual((request["temperature"], request["top_p"], request["top_k"]), (1.0, .95, 20))
            self.assertEqual(request["repetition_penalty"], 1.0)
            self.assertEqual(request["presence_penalty"], 0.0)
            self.assertEqual(request["frequency_penalty"], 0.0)

    def test_tool_round_budget_and_report_do_not_change_session_mode(self):
        self.setting["sticky"] = "high"
        deep_agent._llama_chat([{"role": "user", "content": "Check the hostname"}], tools=True, stream_output=False)
        deep_agent._llama_chat([
            {"role": "system", "content": deep_agent.SUMMARY_SYSTEM_PROMPT},
            {"role": "user", "content": "Summarize the recorded result"},
        ], stream_output=False)
        action, report = [p for _, p in self.sent]
        self.assertEqual(action["reasoning_budget_tokens"], 2048)
        self.assertTrue(action["tools"])
        self.assertFalse(action["parallel_tool_calls"])
        self.assertFalse(report["chat_template_kwargs"]["enable_thinking"])
        self.assertEqual(self.setting["sticky"], "high")

    def test_execution_mode_keeps_existing_agent_behavior(self):
        self.assertTrue(unrestricted_execution_enabled())
        self.assertFalse(unrestricted_execution_enabled("unrelated-model"))
        with patch.dict(os.environ, {"DEEP_AGENT_EXECUTION_MODE": "guarded"}):
            self.assertFalse(unrestricted_execution_enabled())


if __name__ == "__main__":
    unittest.main()
