"""Unrestricted execution checks use mocked SSH; no commands run on Kali."""

import contextlib
import io
import json
import os
import unittest
from unittest.mock import Mock, patch

import deep_agent
import kali_access
from execution_mode import unrestricted_execution_enabled
from test_kali_chat import KaliChatTests


class UnrestrictedAgentTests(unittest.TestCase):
    def setUp(self):
        self.context = contextlib.ExitStack()
        self.addCleanup(self.context.close)
        self.context.enter_context(patch.dict(os.environ, {
            "DEEP_AGENT_EXECUTION_MODE": "unrestricted", "DEEP_AGENT_MODEL": "agent-27b",
        }))
        self.context.enter_context(patch.object(deep_agent, "MODEL", "agent-27b"))

    def test_write_redirection_chaining_and_multiline_go_directly_to_kali(self):
        command = "cat > /tmp/example.conf <<'EOF'\nenabled=true\nEOF\ncat /tmp/example.conf"
        proposal = deep_agent._make_tool_call("run_kali_command", {"command": command})
        with (patch.object(deep_agent, "_model_command_issue") as policy,
              patch.object(deep_agent.KaliWorkflow, "begin") as admission):
            kali, calls, output = KaliChatTests()._run_chat(
                ["create the configuration file"],
                [{"content": "", "tool_calls": [proposal]},
                 {"content": "The command completed.", "tool_calls": []}],
            )
        kali.run.assert_called_once_with(command)
        policy.assert_not_called()
        admission.assert_not_called()
        self.assertEqual(calls[0][2][0]["content"], deep_agent.UNRESTRICTED_SYSTEM_PROMPT)
        self.assertNotIn("Controller-owned task state", str(calls))
        self.assertNotIn("rejected", output.lower())

    def test_package_install_uses_model_command_without_install_workflow(self):
        command = "apt-get install -y curl"
        with patch.object(deep_agent, "_run_package_install_workflow") as bounded_install:
            kali, _, _ = KaliChatTests()._run_chat(
                ["install curl"],
                [{"content": "", "tool_calls": [deep_agent._make_tool_call(
                    "run_kali_command", {"command": command})]},
                 {"content": "The package command completed.", "tool_calls": []}],
            )
        kali.run.assert_called_once_with(command)
        bounded_install.assert_not_called()

    def test_repeated_commands_can_exceed_old_task_limit(self):
        proposal = deep_agent._make_tool_call("run_kali_command", {"command": "printf probe"})
        replies = [{"content": "", "tool_calls": [proposal]} for _ in range(22)]
        kali, calls, output = KaliChatTests()._run_chat(
            ["run the requested repeated checks"],
            [*replies, {"content": "The checks are complete.", "tool_calls": []}],
        )
        self.assertEqual(kali.run.call_count, 22)
        self.assertEqual(len(calls), 23)
        self.assertNotIn("skipped", output.lower())

    def test_every_returned_tool_call_executes_in_order(self):
        calls = [deep_agent._make_tool_call("run_kali_command", {"command": command})
                 for command in ("printf first", "printf second")]
        reply = deep_agent._normalize_reply({"content": "", "tool_calls": calls})
        self.assertEqual(len(reply["tool_calls"]), 2)
        kali, _, _ = KaliChatTests()._run_chat(
            ["run both commands"], [reply, {"content": "Both completed.", "tool_calls": []}],
        )
        self.assertEqual([call.args[0] for call in kali.run.call_args_list],
                         ["printf first", "printf second"])

    def test_mode_is_scoped_to_xxs_and_tools_match_its_instructions(self):
        self.assertTrue(unrestricted_execution_enabled())
        tools = deep_agent._tool_definitions(True)
        command_tool = tools[0]["function"]
        self.assertNotIn("purpose", command_tool["parameters"]["properties"])
        self.assertNotIn("preflight", command_tool["description"])
        with (patch.object(deep_agent, "MODEL", "qwen3.8-q4ks-test"),
              patch.dict(os.environ, {"DEEP_AGENT_MODEL": "qwen3.8-q4ks-test"})):
            self.assertFalse(unrestricted_execution_enabled())
            self.assertEqual(deep_agent._tool_definitions(True), deep_agent.SHELL_TOOLS)
            self.assertEqual(deep_agent._shell_system_prompt(), deep_agent.SHELL_SYSTEM_PROMPT)

    def test_background_and_interactive_accept_shell_scripts(self):
        command = "printf first > /tmp/example\nprintf second >> /tmp/example"
        for validator in (kali_access.KaliAccess._validate_background_request,
                          kali_access.KaliAccess._validate_interactive_request):
            argv, env = validator(command, "/tmp", {"MODE": "example"})
            self.assertEqual(argv, ["bash", "-o", "pipefail", "-c", command])
            self.assertEqual(env, {"MODE": "example"})
        argv, _ = kali_access.KaliAccess._validate_interactive_request("bash", None, None)
        self.assertEqual(argv[-1], "bash")

    def test_complex_sudo_command_reaches_ssh_without_password_in_command(self):
        channel = Mock()
        channel.recv_ready.return_value = False
        channel.recv_stderr_ready.return_value = False
        channel.exit_status_ready.return_value = True
        channel.recv_exit_status.return_value = 0
        client = Mock()
        client.get_transport.return_value.is_active.return_value = True
        client.get_transport.return_value.open_session.return_value = channel
        runner = kali_access.KaliAccess()
        command = "sudo printf example > /tmp/example && cat /tmp/example"
        with (patch.object(runner, "connect", return_value=client),
              patch.object(kali_access, "_read_password", return_value="fixture-password"),
              contextlib.redirect_stdout(io.StringIO())):
            runner.run(command)
        remote_command = channel.exec_command.call_args.args[0]
        self.assertIn("sudo -S", remote_command)
        self.assertNotIn("fixture-password", remote_command)
        self.assertNotIn("fixture-password", json.dumps(runner.last_record))
        self.assertEqual(runner.last_record["execution_state"], "completed")

    def test_interactive_multiline_input_is_sent_at_any_prompt(self):
        channel = Mock()
        channel.exit_status_ready.return_value = False
        tty_id = "tty-" + "a" * 32
        runner = kali_access.KaliAccess()
        runner.interactive_processes[tty_id] = {
            "process_id": tty_id, "channel": channel, "program": "bash", "cwd": "/tmp",
            "state": "running", "exit_code": None, "output_tail": "Password:",
            "raw_output_tail": b"Password:", "output_truncated": False,
        }
        text = "printf first\nprintf second"
        with (patch.object(runner, "_capture_interactive_output", return_value=("", False)),
              contextlib.redirect_stdout(io.StringIO())):
            runner.send_interactive_input(tty_id, text)
        channel.sendall.assert_called_once_with((text + "\n").encode())
        self.assertNotEqual(runner.last_record.get("failure_type"), "CONTROLLER_REJECTED")

    def test_user_interrupt_stops_before_another_model_round(self):
        kali = Mock()
        kali.last_record = {"command": "printf example", "execution_state": "interrupted",
                            "exit_code": None, "stdout": "", "stderr": ""}
        kali.run.side_effect = KeyboardInterrupt
        messages = [{"role": "system", "content": deep_agent.UNRESTRICTED_SYSTEM_PROMPT},
                    {"role": "user", "content": "run the command"}]
        proposal = deep_agent._make_tool_call("run_kali_command", {"command": "printf example"})
        with (patch.object(deep_agent, "_model_chat", return_value={
                "content": "", "tool_calls": [proposal]}) as model,
              contextlib.redirect_stdout(io.StringIO()), self.assertRaises(KeyboardInterrupt)):
            deep_agent._run_unrestricted_kali_turn(kali, messages)
        model.assert_called_once()
        kali.run.assert_called_once()


if __name__ == "__main__":
    unittest.main()
