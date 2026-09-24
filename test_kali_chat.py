"""Offline checks for the normal chat prompt's Kali command flow."""

import contextlib
import io
import json
import os
import shlex
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import deep_agent
import kali_access


class KaliChatTests(unittest.TestCase):
    def _run_chat(self, inputs, replies, *, shell=True, last_record=None, sudo_mode=False):
        kali = Mock()
        kali.run.return_value = "192.168.56.101\n[exit 0; 0.05s]"
        kali.sudo_mode = sudo_mode
        kali.clear_sudo_mode.side_effect = lambda: setattr(kali, "sudo_mode", False)
        if last_record is not None:
            kali.last_record = last_record
        else:
            def run(command):
                kali.last_record = {
                    "command": command,
                    "execution_state": "completed",
                    "exit_code": 0,
                    "stdout": "192.168.56.101\n",
                    "stderr": "",
                    "timed_out": False,
                    "duration_seconds": 0.05,
                    "privilege_mode": "user",
                    "sudo_access_validated": False,
                }
                return "192.168.56.101\n[exit 0; 0.05s]"

            kali.run.side_effect = run
        snapshots = []
        replies = iter(replies)

        def model(messages, *, tools, stream_output):
            snapshots.append((tools, stream_output, [dict(item) for item in messages]))
            return next(replies)

        output = io.StringIO()
        prompts = (["/shell"] if shell else []) + list(inputs) + ["exit"]
        with (patch("builtins.input", side_effect=prompts),
              patch("kali_access.KaliAccess", return_value=kali),
              patch.object(deep_agent, "_model_chat", side_effect=model),
              contextlib.redirect_stdout(output)):
            deep_agent.chat_loop()
        return kali, snapshots, output.getvalue()

    def test_direct_command_runs_once_and_returns_output_to_model(self):
        kali, calls, _ = self._run_chat(["ip a"], [{"content": "Kali reported its address.", "tool_calls": []}])
        kali.run.assert_called_once_with("ip a")
        self.assertEqual([item[0] for item in calls], [False])
        evidence = json.loads(calls[0][2][-1]["content"])
        self.assertEqual(evidence["recorded_results"][0]["command"], "ip a")
        self.assertIn("192.168.56.101", evidence["recorded_results"][0]["output"])

    def test_shell_mode_runs_simple_command_without_model_planning(self):
        kali, calls, _ = self._run_chat(["date"], [{"content": "Kali reported the time.", "tool_calls": []}])
        kali.run.assert_called_once_with("date")
        self.assertEqual([item[0] for item in calls], [False])

    def test_plain_ping_is_bounded(self):
        kali, _, _ = self._run_chat(["ping 127.0.0.1"], [{"content": "Ping completed.", "tool_calls": []}])
        kali.run.assert_called_once_with("ping -c 4 127.0.0.1")

    def test_k2_json_plan_executes_then_model_summarizes(self):
        plan = {"content": '{"analysis":"plan","command":"nmap -sT 192.168.56.101"}', "tool_calls": []}
        parsed = deep_agent._normalize_reply(plan)
        kali, calls, output = self._run_chat(
            ["probe services on 192.168.56.101"],
            [parsed, {"content": "Kali returned scan output.", "tool_calls": []}],
        )
        kali.run.assert_called_once_with("nmap -sT 192.168.56.101")
        self.assertEqual([item[0] for item in calls], [True, True])
        self.assertNotIn('"analysis"', output)

    def test_promise_is_retried_before_reporting_no_action(self):
        tool = deep_agent._make_tool_call("run_kali_command", {"command": "ss -tulpn"})
        kali, calls, _ = self._run_chat(
            ["check listening ports"],
            [{"content": "I'll check.", "tool_calls": []},
             {"content": "", "tool_calls": [tool]},
             {"content": "A service is listening.", "tool_calls": []}],
        )
        kali.run.assert_called_once_with("ss -tulpn")
        self.assertEqual([item[0] for item in calls], [True, True, True])

    def test_missing_scan_target_asks_before_any_command(self):
        kali, calls, output = self._run_chat(["scan ports"], [])
        kali.run.assert_not_called()
        self.assertEqual(calls, [])
        self.assertIn("Which host should I scan?", output)

    def test_scan_target_reply_uses_bounded_unprivileged_scan(self):
        kali, calls, _ = self._run_chat(
            ["scan ports", "Kali itself"],
            [{"content": "The quick scan completed.", "tool_calls": []}],
        )
        kali.run.assert_called_once_with("nmap -n -sT --top-ports 100 --host-timeout 45s 127.0.0.1")
        self.assertEqual([item[0] for item in calls], [False])

    def test_numbered_scan_option_runs_the_selected_target(self):
        first = deep_agent._make_tool_call("run_kali_command", {"command": "hostname -I"})
        kali, _, _ = self._run_chat(
            ["check Kali IP", "Do 1"],
            [{"content": "", "tool_calls": [first]},
             {"content": "1. Scan this Kali VM itself: `nmap 10.0.2.15`\n2. Scan another host", "tool_calls": []},
             {"content": "The selected scan completed.", "tool_calls": []}],
        )
        self.assertEqual(kali.run.call_count, 2)
        self.assertEqual(kali.run.call_args.args[0], "nmap -n -sT --top-ports 100 --host-timeout 45s 10.0.2.15")

    def test_model_sudo_scan_is_rejected_and_replanned(self):
        bad = deep_agent._make_tool_call("run_kali_command", {"command": "sudo nmap -p- localhost | tail -20"})
        good = deep_agent._make_tool_call("run_kali_command", {"command": "nmap -n -sT -F --host-timeout 45s 127.0.0.1"})
        kali, calls, _ = self._run_chat(
            ["inspect Kali network services"],
            [{"content": "", "tool_calls": [bad]},
             {"content": "", "tool_calls": [good]},
             {"content": "The scan completed.", "tool_calls": []}],
        )
        kali.run.assert_called_once_with("nmap -n -sT -F --host-timeout 45s 127.0.0.1")
        self.assertEqual([item[0] for item in calls], [True, True, True])

    def test_followup_tool_call_runs_and_is_visible_to_model(self):
        first = deep_agent._make_tool_call("run_kali_command", {"command": "which nmap"})
        second = deep_agent._make_tool_call("run_kali_command", {"command": "which curl"})
        kali, calls, _ = self._run_chat(
            ["check Kali dependencies"],
            [{"content": "", "tool_calls": [first]},
             {"content": "", "tool_calls": [second]},
             {"content": "Both tools are installed.", "tool_calls": []}],
        )
        self.assertEqual([entry.args[0] for entry in kali.run.call_args_list], ["which nmap", "which curl"])
        self.assertEqual([item[0] for item in calls], [True, True, True])
        self.assertEqual(calls[-1][2][-1]["role"], "tool")

    def test_tool_request_after_explicit_command_gets_grounded_fallback(self):
        extra = deep_agent._make_tool_call("run_kali_command", {"command": "which curl"})
        kali, _, output = self._run_chat(["which nmap"], [{"content": "", "tool_calls": [extra]}])
        kali.run.assert_called_once_with("which nmap")
        self.assertIn("no additional command ran.", output)

    def test_greeting_does_not_open_kali(self):
        kali, calls, _ = self._run_chat(["hi"], [{"content": "Hello.", "tool_calls": []}], shell=False)
        kali.run.assert_not_called()
        self.assertEqual([item[0] for item in calls], [False])

    def test_ordinary_chat_request_does_not_offer_kali_tool(self):
        kali, calls, _ = self._run_chat(["show me a recipe"], [{"content": "Here is one.", "tool_calls": []}], shell=False)
        kali.run.assert_not_called()
        self.assertEqual([item[0] for item in calls], [False])

    def test_chat_mode_tool_request_points_to_shell_mode(self):
        tool = deep_agent._make_tool_call("run_kali_command", {"command": "whoami"})
        kali, calls, output = self._run_chat(["kali"], [{"content": "", "tool_calls": [tool]}], shell=False)
        kali.run.assert_not_called()
        self.assertEqual([item[0] for item in calls], [False])
        self.assertIn("Type /shell", output)

    def test_chat_mode_command_is_redirected_without_model_call(self):
        kali, calls, output = self._run_chat(["ip a"], [], shell=False)
        kali.run.assert_not_called()
        self.assertEqual(calls, [])
        self.assertIn("Switch to /shell", output)

    def test_interactive_sudo_shell_is_rejected(self):
        kali, calls, output = self._run_chat(["escalate privilage", "kali", "sudo su"], [])
        kali.run.assert_not_called()
        self.assertEqual(calls, [])
        self.assertIn("Interactive root shells are not supported", output)

    def test_direct_sudo_command_reaches_kali(self):
        kali, calls, output = self._run_chat(["sudo nmap localhost"], [])
        kali.run.assert_called_once_with("sudo nmap localhost")
        self.assertEqual([item[0] for item in calls], [False])
        self.assertIn("Recorded results follow", output)

    def test_kali_prefix_allows_one_sudo_command_but_blocks_chaining(self):
        kali, calls, output = self._run_chat(["/kali sudo id"], [])
        kali.run.assert_called_once_with("sudo id")
        self.assertEqual([item[0] for item in calls], [False])
        self.assertIn("Recorded results follow", output)

        kali, calls, output = self._run_chat(["/kali id; sudo whoami"], [])
        kali.run.assert_not_called()
        self.assertEqual(calls, [])
        self.assertIn("No Kali command ran", output)
        self.assertIn("sudo must be the first and only command", output)

    def test_exploit_scope_rejects_chained_or_additional_hosts(self):
        target = "192.168.56.101"
        bad_commands = (
            f"nmap -n -sT {target} otherhost",
            f"nmap -n -sT {target}; curl https://outside.example",
            "curl http://outside-host/",
        )
        for command in bad_commands:
            with self.subTest(command=command):
                issue = deep_agent._model_command_issue(command, "validate this lab", target)
                self.assertIsNotNone(issue)

    def test_exploit_scope_accepts_a_single_bounded_target(self):
        target = "192.168.56.101"
        command = f"nmap -n -sT -sV --top-ports 100 --host-timeout 45s {target}"
        self.assertIsNone(deep_agent._model_command_issue(command, "validate this lab", target))

    def test_exploit_followup_scan_uses_only_observed_open_ports(self):
        target = "192.168.56.101"
        observed = {21, 22, 80}
        repeated_top_ports = f"nmap -n -sT -sV --top-ports 100 --host-timeout 45s {target}"
        issue = deep_agent._model_command_issue(
            repeated_top_ports, "validate this lab", target, known_tcp_ports=observed,
        )
        self.assertIn("select previously observed open TCP ports", issue)

        scoped = f"nmap -n -sT -sV -p 21,22,80 --host-timeout 45s {target}"
        self.assertIsNone(deep_agent._model_command_issue(
            scoped, "validate this lab", target, known_tcp_ports=observed,
        ))

        unobserved = f"nmap -n -sT -sV -p 21,22,80,445 --host-timeout 45s {target}"
        issue = deep_agent._model_command_issue(
            unobserved, "validate this lab", target, known_tcp_ports=observed,
        )
        self.assertIn("unobserved ports", issue)

    def test_package_install_plans_do_not_hide_failures_or_use_coreutils_install(self):
        chained = "apt-get update 2>&1 | tail -8; echo done; apt-get install -y google-chrome-stable"
        issue = deep_agent._model_command_issue(chained, "install google chrome")
        self.assertIn("one command at a time", issue)
        issue = deep_agent._model_command_issue("sudo install google chrome", "install google chrome")
        self.assertIn("copies files", issue)
        self.assertIsNone(deep_agent._model_command_issue(
            "apt-cache policy google-chrome-stable", "install google chrome",
        ))
        self.assertIn("apt-key is unavailable", deep_agent._model_command_issue(
            "apt-key add /tmp/key.pub", "install google chrome",
        ))
        self.assertIn("Markdown link markup", deep_agent._model_command_issue(
            "wget -q <https://dl.google.com/linux/linux_signing_key.pub> -O /tmp/key_[key.pub](http://key.pub)",
            "install google chrome",
        ))

    def test_workflow_tracks_admission_repetition_and_terminal_failures(self):
        workflow = deep_agent.KaliWorkflow("test", max_commands=2)
        self.assertIsNone(workflow.begin("hostname"))
        self.assertIsNone(workflow.finish_command({
            "execution_state": "completed", "exit_code": 0, "timed_out": False,
        }))
        self.assertIn("already attempted", workflow.begin("hostname"))
        self.assertIsNone(workflow.begin("whoami"))
        self.assertIn("timed out", workflow.finish_command({
            "execution_state": "timed_out", "exit_code": 124, "timed_out": True,
        }))
        self.assertIs(workflow.state, deep_agent.WorkflowState.BLOCKED)
        self.assertIn("timed out", workflow.stop_reason)

    def test_workflow_records_complete_successful_open_tcp_discovery(self):
        target = "192.168.56.101"
        workflow = deep_agent.KaliWorkflow("assessment", scope_target=target)
        command = f"nmap -n -sT -sV --top-ports 100 --host-timeout 45s {target}"
        self.assertIsNone(workflow.begin(command))
        self.assertIsNone(workflow.finish_command({
            "command": command,
            "execution_state": "completed",
            "exit_code": 0,
            "timed_out": False,
            "output_truncated": False,
            "stdout": "PORT     STATE SERVICE\n22/tcp   open  ssh\n80/tcp   open  http\n111/tcp  closed rpcbind\n",
        }))
        self.assertTrue(workflow.port_discovery_complete)
        self.assertEqual(workflow.discovered_tcp_ports, {22, 80})

    def test_workflow_enforces_command_budget_without_running_extra_steps(self):
        workflow = deep_agent.KaliWorkflow("test", max_commands=1)
        self.assertIsNone(workflow.begin("hostname"))
        workflow.finish_command({"execution_state": "completed", "exit_code": 0})
        self.assertIn("1-command limit", workflow.begin("whoami"))
        self.assertIs(workflow.state, deep_agent.WorkflowState.BLOCKED)

    def test_install_request_router_normalizes_packages_and_chrome_aliases(self):
        self.assertEqual(deep_agent._package_install_request("install nmap"), {
            "display_name": "nmap", "package": "nmap", "vendor": None,
        })
        self.assertEqual(deep_agent._package_install_request("install Google Chrome"), {
            "display_name": "Google Chrome", "package": "google-chrome-stable", "vendor": "google-chrome",
        })
        self.assertEqual(deep_agent._package_install_request("install chrome on Kali"), {
            "display_name": "Google Chrome", "package": "google-chrome-stable", "vendor": "google-chrome",
        })
        self.assertIsNone(deep_agent._package_install_request("install /tmp/local.deb"))

    def test_short_kali_connection_phrases_are_deterministic(self):
        for phrase in ("connect kali", "connect to kali", "connect cali", "connect to cali",
                       "connect kalu", "connect to kalu", "connect me to kali vm"):
            with self.subTest(phrase=phrase):
                self.assertTrue(deep_agent._local_kali_connection_request(phrase))

    def test_exploit_workflow_requires_a_bounded_service_scan_first(self):
        target = "192.168.56.101"
        valid = f"nmap -n -sT -sV --top-ports 100 --host-timeout 45s {target}"
        self.assertIsNotNone(deep_agent._model_command_issue(
            f"ping -c 3 -W 1 {target}", "validate this lab", target,
            require_exploit_scan=True,
        ))
        self.assertIsNone(deep_agent._model_command_issue(
            valid, "validate this lab", target, require_exploit_scan=True,
        ))

    def test_exploit_scope_rejects_unbounded_or_aggressive_nmap_scans(self):
        target = "192.168.56.101"
        bad_commands = (
            f"nmap -sT -sV -p 1-10000 --host-timeout 45s {target}",
            f"nmap -sT -sV --top-ports 1001 --host-timeout 45s {target}",
            f"nmap -sT -sV --top-ports 100 {target}",
            f"nmap -sT -sV --host-timeout 45s {target}",
            f"nmap -sT -sV --top-ports 100 --host-timeout 46s {target}",
            f"nmap -sT --top-ports 100 --host-timeout 45s {target}",
            f"nmap -sT -sV -A --top-ports 100 --host-timeout 45s {target}",
        )
        for command in bad_commands:
            with self.subTest(command=command):
                issue = deep_agent._model_command_issue(command, "validate this lab", target)
                self.assertIsNotNone(issue)

    def test_exploit_workflow_replans_an_oversized_scan_before_ssh(self):
        target = "192.168.56.101"
        oversized = deep_agent._make_tool_call(
            "run_kali_command",
            {"command": f"nmap -sT -sV -p 1-10000 --host-timeout 45s {target}"},
        )
        bounded_command = f"nmap -n -sT -sV --top-ports 100 --host-timeout 45s {target}"
        bounded = deep_agent._make_tool_call("run_kali_command", {"command": bounded_command})
        kali, calls, _ = self._run_chat(
            [f"/exploit {target}"],
            [
                {"content": "", "tool_calls": [oversized]},
                {"content": "", "tool_calls": [bounded]},
                {"content": "The bounded scan completed.", "tool_calls": []},
            ],
        )
        kali.run.assert_called_once_with(bounded_command)
        self.assertEqual([item[0] for item in calls], [True, True, True])
        correction = calls[1][2][-1]["content"]
        self.assertIn("may scan at most 1000", correction)

    def test_natural_exploit_request_uses_the_scoped_workflow(self):
        target = "192.168.56.101"
        oversized_command = f"nmap -n -sT -sV -p 1-10000 --host-timeout 45s {target}"
        bounded_command = f"nmap -n -sT -sV --top-ports 100 --host-timeout 45s {target}"
        oversized = deep_agent._make_tool_call("run_kali_command", {"command": oversized_command})
        bounded = deep_agent._make_tool_call("run_kali_command", {"command": bounded_command})
        kali, calls, _ = self._run_chat(
            [f"exploit {target}"],
            [
                {"content": "", "tool_calls": [oversized]},
                {"content": "", "tool_calls": [bounded]},
                {"content": "The bounded scan completed.", "tool_calls": []},
            ],
        )
        kali.run.assert_called_once_with(bounded_command)
        self.assertEqual([item[0] for item in calls], [True, True, True])
        self.assertIn("may scan at most 1000", calls[1][2][-1]["content"])

    def test_exploit_command_rejects_public_target_before_using_kali(self):
        kali, calls, output = self._run_chat(["/exploit 8.8.8.8"], [])
        kali.run.assert_not_called()
        self.assertEqual(calls, [])
        self.assertIn("PRIVATE_IPV4", output)

    def test_exploit_target_accepts_only_private_or_loopback_ipv4(self):
        for target, expected in (
            ("192.168.56.101", "192.168.56.101"),
            ("127.0.0.1", "127.0.0.1"),
            ("8.8.8.8", None),
            ("example.com", None),
            ("2001:db8::1", None),
        ):
            with self.subTest(target=target):
                self.assertEqual(deep_agent._lab_exploit_target(target), expected)

    def test_slash_sudo_is_not_mistaken_for_bare_sudo(self):
        kali, calls, output = self._run_chat(["/sudo"], [])
        kali.run.assert_not_called()
        self.assertEqual(calls, [])
        self.assertIn("Use `sudo` directly", output)
        self.assertNotIn("sudo..", output)

    def test_bare_sudo_validates_credentials_without_running_a_privileged_task(self):
        kali, calls, output = self._run_chat(
            ["sudo"], [], last_record={
                "command": "sudo",
                "privilege_mode": "sudo_validation",
                "sudo_access_validated": True,
                "exit_code": 0,
            },
        )
        kali.run.assert_called_once_with("sudo")
        self.assertEqual(calls, [])
        self.assertIn("Sudo authentication succeeded", output)
        self.assertIn("Sudo mode is enabled", output)

    def test_bare_sudo_reports_failed_validation_without_claiming_access(self):
        _, calls, output = self._run_chat(
            ["sudo"], [], last_record={
                "command": "sudo",
                "privilege_mode": "sudo_validation",
                "sudo_access_validated": False,
                "exit_code": 1,
            },
        )
        self.assertEqual(calls, [])
        self.assertIn("sudo check did not succeed", output)
        self.assertNotIn("Sudo authentication succeeded", output)

    def test_user_command_disables_session_sudo_mode(self):
        kali, calls, output = self._run_chat(["/user"], [], sudo_mode=True)
        kali.run.assert_not_called()
        self.assertEqual(calls, [])
        self.assertFalse(kali.sudo_mode)
        self.assertIn("Sudo mode disabled", output)

    def test_sudo_placeholders_are_rejected_before_reaching_kali(self):
        cases = (
            ("sudo command", "Replace the sudo placeholder"),
            ("'sudo command'", "Remove the quotes around sudo"),
            ("sudo <COMMAND>", "Replace the sudo placeholder"),
        )
        for user_input, hint in cases:
            with self.subTest(user_input=user_input):
                kali, calls, output = self._run_chat([user_input], [])
                kali.run.assert_not_called()
                self.assertEqual(calls, [])
                self.assertIn("No Kali command ran", output)
                self.assertIn(hint, output)

    def test_shell_requests_from_report_offer_tools(self):
        for request in ("make a personal portfolio website on apache", "give the url",
                        "Where's the url", "connect to the web server"):
            with self.subTest(request=request):
                kali, calls, output = self._run_chat(
                    [request], [{"content": "I need one detail first.", "tool_calls": []}])
                self.assertTrue(calls[0][0])
                self.assertFalse(calls[0][1])
                self.assertEqual(output.count("I need one detail first."), 1)
                kali.run.assert_not_called()

    def test_connect_to_kali_runs_a_direct_connection_check(self):
        kali, calls, output = self._run_chat(
            ["connect to kali"], [], last_record={
                "command": "hostname && whoami",
                "execution_state": "completed",
                "exit_code": 0,
                "stdout": "kali\nkali\n",
                "stderr": "",
                "timed_out": False,
            })
        kali.run.assert_called_once_with("hostname && whoami")
        self.assertEqual(calls, [])
        self.assertIn("Connected to Kali as kali on host kali.", output)

    def test_misspelled_local_kali_connect_does_not_consult_k2(self):
        for phrase in ("connect cali", "connect kalu", "connect kali"):
            with self.subTest(phrase=phrase):
                kali, calls, _ = self._run_chat([phrase], [], last_record={
                    "command": "hostname && whoami",
                    "execution_state": "completed",
                    "exit_code": 0,
                    "stdout": "kali\nkali\n",
                    "stderr": "",
                    "timed_out": False,
                })
                kali.run.assert_called_once_with("hostname && whoami")
                self.assertEqual(calls, [])

    def test_no_command_reports_no_action(self):
        kali, _, output = self._run_chat(
            ["check that server"], [{"content": "Which host did you mean?", "tool_calls": []}])
        kali.run.assert_not_called()
        self.assertIn("No Kali command ran for this request.", output)
        self.assertIn("Which host did you mean?", output)

    def test_natural_package_install_uses_controller_workflow_not_k2(self):
        with patch.object(deep_agent, "_run_package_install_workflow") as install:
            kali, calls, _ = self._run_chat(["install google chrome"], [])
        self.assertEqual(calls, [])
        kali.run.assert_not_called()
        install.assert_called_once_with(kali, unittest.mock.ANY, {
            "display_name": "Google Chrome", "package": "google-chrome-stable", "vendor": "google-chrome",
        })

    def test_google_chrome_workflow_uses_verified_signed_repository_and_checks_install(self):
        kali = Mock()
        kali.sudo_mode = False
        kali.clear_sudo_mode.side_effect = lambda: setattr(kali, "sudo_mode", False)
        commands = []
        policy_count = 0

        def execute(_kali, _messages, command, _call_id):
            nonlocal policy_count
            commands.append(command)
            stdout = ""
            record = {
                "command": command,
                "execution_state": "completed",
                "exit_code": 0,
                "stdout": stdout,
                "stderr": "",
                "timed_out": False,
                "privilege_mode": "user",
                "sudo_access_validated": False,
            }
            if command == "dpkg --print-architecture":
                stdout = "amd64\n"
            elif command == "apt-cache policy google-chrome-stable":
                policy_count += 1
                candidate = "(none)" if policy_count == 1 else "144.0.0.0-1"
                stdout = f"google-chrome-stable:\n  Installed: (none)\n  Candidate: {candidate}\n"
            elif command == "mktemp -d /tmp/deep-agent-apt.XXXXXX":
                stdout = "/tmp/deep-agent-apt.ABC123\n"
            elif command.startswith("gpg --batch --show-keys"):
                stdout = f"pub:::::::::D38B4796:\nfpr:::::::::{deep_agent.GOOGLE_LINUX_KEY_FINGERPRINT}:\n"
            elif command == "sudo":
                record.update({"privilege_mode": "sudo_validation", "sudo_access_validated": True})
                kali.sudo_mode = True
            elif command.startswith("dpkg-query -W"):
                stdout = "installed 144.0.0.0-1\n"
            elif command == "google-chrome --version":
                stdout = "Google Chrome 144.0.0.0\n"
            record["stdout"] = stdout
            kali.last_record = record
            return stdout

        messages = [{"role": "system", "content": "test"}, {"role": "user", "content": "install google chrome"}]
        with patch.object(deep_agent, "_execute_kali", side_effect=execute), contextlib.redirect_stdout(io.StringIO()):
            deep_agent._run_package_install_workflow(kali, messages, deep_agent._package_install_request("install google chrome"))

        self.assertTrue(any("https://dl.google.com/linux/linux_signing_key.pub" in command for command in commands))
        self.assertTrue(any("gpg --batch --show-keys --with-colons --fingerprint" in command for command in commands))
        self.assertTrue(any("signed-by=/etc/apt/keyrings/google-chrome.gpg" in command for command in commands))
        self.assertIn("apt-get update", commands)
        self.assertIn("apt-get install -y --no-install-recommends google-chrome-stable", commands)
        self.assertIn("dpkg-query", commands[-4])
        self.assertIn("google-chrome --version", commands)
        self.assertNotIn("apt-key", "\n".join(commands))
        self.assertNotIn("[key.pub]", "\n".join(commands))
        self.assertLess(commands.index("apt-get update"), commands.index("apt-get install -y --no-install-recommends google-chrome-stable"))
        self.assertIn("Installed and verified Google Chrome", messages[-1]["content"])
        kali.clear_sudo_mode.assert_called_once()
        self.assertFalse(kali.sudo_mode)

    def test_google_chrome_workflow_blocks_untrusted_signing_key_before_privilege(self):
        kali = Mock()
        kali.sudo_mode = False
        commands = []

        def execute(_kali, _messages, command, _call_id):
            commands.append(command)
            stdout = ""
            if command == "dpkg --print-architecture":
                stdout = "amd64\n"
            elif command == "apt-cache policy google-chrome-stable":
                stdout = "google-chrome-stable:\n  Installed: (none)\n  Candidate: (none)\n"
            elif command == "mktemp -d /tmp/deep-agent-apt.XXXXXX":
                stdout = "/tmp/deep-agent-apt.DEF456\n"
            elif command.startswith("gpg --batch --show-keys"):
                stdout = "fpr:::::::::0000000000000000000000000000000000000000:\n"
            kali.last_record = {
                "command": command,
                "execution_state": "completed",
                "exit_code": 0,
                "stdout": stdout,
                "stderr": "",
                "timed_out": False,
            }
            return stdout

        messages = [{"role": "system", "content": "test"}]
        with patch.object(deep_agent, "_execute_kali", side_effect=execute), contextlib.redirect_stdout(io.StringIO()):
            deep_agent._run_package_install_workflow(kali, messages, deep_agent._package_install_request("install google chrome"))

        self.assertIn("did not match Google's published Linux key", messages[-1]["content"])
        self.assertFalse(any(command == "sudo" or command.startswith("install -") or "apt-get install" in command for command in commands))
        self.assertTrue(commands[-2].startswith("rm -f -- /tmp/deep-agent-apt.DEF456/"))
        self.assertEqual(commands[-1], "rmdir -- /tmp/deep-agent-apt.DEF456")

    def test_generic_package_workflow_stops_if_candidate_remains_missing(self):
        kali = Mock()
        kali.sudo_mode = False
        kali.clear_sudo_mode.side_effect = lambda: setattr(kali, "sudo_mode", False)
        commands = []
        policy_count = 0

        def execute(_kali, _messages, command, _call_id):
            nonlocal policy_count
            commands.append(command)
            stdout = ""
            record = {
                "command": command,
                "execution_state": "completed",
                "exit_code": 0,
                "stdout": "",
                "stderr": "",
                "timed_out": False,
                "privilege_mode": "user",
                "sudo_access_validated": False,
            }
            if command == "apt-cache policy sample-package":
                policy_count += 1
                stdout = "sample-package:\n  Installed: (none)\n  Candidate: (none)\n"
            elif command == "sudo":
                record.update({"privilege_mode": "sudo_validation", "sudo_access_validated": True})
                kali.sudo_mode = True
            record["stdout"] = stdout
            kali.last_record = record
            return stdout

        messages = [{"role": "system", "content": "test"}]
        with patch.object(deep_agent, "_execute_kali", side_effect=execute), contextlib.redirect_stdout(io.StringIO()):
            deep_agent._run_package_install_workflow(kali, messages, {
                "display_name": "sample package", "package": "sample-package", "vendor": None,
            })
        self.assertEqual(commands, [
            "apt-cache policy sample-package", "sudo", "apt-get update", "apt-cache policy sample-package",
        ])
        self.assertNotIn("apt-get install", commands)
        self.assertIn("no candidate", messages[-1]["content"])
        self.assertFalse(kali.sudo_mode)

    def test_generic_package_workflow_installs_only_after_candidate_check_and_verifies(self):
        kali = Mock()
        kali.sudo_mode = False
        kali.clear_sudo_mode.side_effect = lambda: setattr(kali, "sudo_mode", False)
        commands = []

        def execute(_kali, _messages, command, _call_id):
            commands.append(command)
            stdout = ""
            record = {
                "command": command,
                "execution_state": "completed",
                "exit_code": 0,
                "stdout": "",
                "stderr": "",
                "timed_out": False,
                "privilege_mode": "user",
                "sudo_access_validated": False,
            }
            if command == "apt-cache policy nmap":
                stdout = "nmap:\n  Installed: (none)\n  Candidate: 7.95+dfsg-1\n"
            elif command == "sudo":
                record.update({"privilege_mode": "sudo_validation", "sudo_access_validated": True})
                kali.sudo_mode = True
            elif command.startswith("dpkg-query -W"):
                stdout = "installed 7.95+dfsg-1\n"
            record["stdout"] = stdout
            kali.last_record = record
            return stdout

        messages = [{"role": "system", "content": "test"}]
        with patch.object(deep_agent, "_execute_kali", side_effect=execute), contextlib.redirect_stdout(io.StringIO()):
            deep_agent._run_package_install_workflow(kali, messages, {
                "display_name": "nmap", "package": "nmap", "vendor": None,
            })
        self.assertEqual(commands[:3], [
            "apt-cache policy nmap",
            "sudo",
            "apt-get install -y --no-install-recommends nmap",
        ])
        self.assertEqual(len(commands), 4)
        self.assertTrue(commands[-1].startswith("dpkg-query -W"))
        self.assertNotIn("apt-get update", commands)
        self.assertIn("Installed and verified nmap package version 7.95+dfsg-1", messages[-1]["content"])
        self.assertFalse(kali.sudo_mode)

    def test_database_request_and_go_ahead_keep_tool_access(self):
        tool = deep_agent._make_tool_call("run_kali_command", {"command": "which mariadb"})
        kali, calls, output = self._run_chat(
            ["link this website to a database", "Go ahead"],
            [{"content": "Which database should I check?", "tool_calls": []},
             {"content": "", "tool_calls": [tool]},
             {"content": "The executable lookup completed.", "tool_calls": []}],
        )
        kali.run.assert_called_once_with("which mariadb")
        self.assertTrue(all(call[0] for call in calls))
        self.assertNotIn("tool request in its answer", output)

    def test_unlisted_shell_request_reaches_tools_without_word_allowlist(self):
        tool = deep_agent._make_tool_call("run_kali_command", {"command": "pwd"})
        kali, calls, _ = self._run_chat(
            ["tell me where we are working"],
            [{"content": "", "tool_calls": [tool]},
             {"content": "The working directory is shown above.", "tool_calls": []}],
        )
        kali.run.assert_called_once_with("pwd")
        self.assertTrue(all(call[0] for call in calls))

    def test_shell_greeting_does_not_execute_as_binary(self):
        kali, _, output = self._run_chat(["hi"], [{"content": "Hello.", "tool_calls": []}])
        kali.run.assert_not_called()
        self.assertIn("Hello.", output)

    def test_result_question_uses_records_not_prior_assistant_claims(self):
        kali, calls, output = self._run_chat(
            ["cat /tmp/example.html", "what were the results?"],
            [{"content": "I created imaginary.html.", "tool_calls": []},
             {"content": "Only a read was recorded.", "tool_calls": []}],
        )
        kali.run.assert_called_once_with("cat /tmp/example.html")
        self.assertEqual([call[0] for call in calls], [False, False])
        self.assertNotIn("imaginary.html", str(calls[-1][2]))
        evidence = json.loads(calls[-1][2][-1]["content"])
        self.assertEqual(evidence["recorded_results"][0]["command"], "cat /tmp/example.html")
        self.assertIn("Only a read was recorded.", output)

    def test_rejected_summary_tool_call_displays_all_current_results(self):
        first = deep_agent._make_tool_call("run_kali_command", {"command": "printf first"})
        second = deep_agent._make_tool_call("run_kali_command", {"command": "printf second"})
        extra = deep_agent._make_tool_call("run_kali_command", {"command": "printf must-not-run"})
        kali, calls, output = self._run_chat(
            ["check Kali"],
            [{"content": "", "tool_calls": [first]},
             {"content": "", "tool_calls": [second]},
             {"content": "", "tool_calls": [second]},
             {"content": "", "tool_calls": [extra]}],
        )
        self.assertEqual(kali.run.call_count, 2)
        evidence = json.loads(calls[-1][2][-1]["content"])
        self.assertEqual([r["command"] for r in evidence["recorded_results"]], ["printf first", "printf second", "printf second"])
        self.assertIn("Repeated command skipped", evidence["recorded_results"][-1]["output"])
        self.assertIn("$ printf first", output)
        self.assertIn("$ printf second", output)
        self.assertNotIn("must-not-run", output)

    def test_summary_failure_retains_observed_output(self):
        messages = [{"role": "system", "content": deep_agent.SHELL_SYSTEM_PROMPT}]
        deep_agent._record_tool_result(messages, "cat /tmp/missing", "test", "No such file\n[exit 1]")
        output = io.StringIO()
        with patch.object(deep_agent, "_model_chat", side_effect=ConnectionError("offline")), contextlib.redirect_stdout(output):
            deep_agent._summarize_results(messages, deep_agent._execution_records(messages), "results?")
        self.assertIn("No such file", output.getvalue())
        self.assertIn("[exit 1]", output.getvalue())

    def test_results_with_no_records_cannot_run_a_tool(self):
        tool = deep_agent._make_tool_call("run_kali_command", {"command": "whoami"})
        kali, calls, output = self._run_chat(["results"], [{"content": "", "tool_calls": [tool]}])
        kali.run.assert_not_called()
        self.assertFalse(calls[0][0])
        self.assertIn("No Kali command results", output)

    def test_multistep_task_can_finish_after_four_commands(self):
        commands = [f"printf step{i}" for i in range(5)]
        replies = [{"content": "I'll do the next step.", "tool_calls": [
            deep_agent._make_tool_call("run_kali_command", {"command": command})
        ]} for command in commands]
        replies.append({"content": "All five steps completed.", "tool_calls": []})
        kali, calls, output = self._run_chat(["create a website on Kali"], replies)
        self.assertEqual([call.args[0] for call in kali.run.call_args_list], commands)
        self.assertTrue(all(not call[1] for call in calls))
        self.assertNotIn("I'll do the next step.", output)
        self.assertEqual(output.count("All five steps completed."), 1)

    def test_task_continues_past_old_command_limit(self):
        replies = [{"content": "", "tool_calls": [deep_agent._make_tool_call(
            "run_kali_command", {"command": f"printf step{i}"})]}
            for i in range(15)]
        replies.append({"content": "All steps completed.", "tool_calls": []})
        kali, calls, output = self._run_chat(["create a website on Kali"], replies)
        self.assertEqual(kali.run.call_count, 15)
        self.assertTrue(calls[-1][0])
        self.assertNotIn("reached the command limit", output)
        self.assertIn("All steps completed.", output)

    def test_model_workflow_stops_at_its_command_budget(self):
        replies = [{"content": "", "tool_calls": [deep_agent._make_tool_call(
            "run_kali_command", {"command": f"printf step{i}"})]}
            for i in range(deep_agent.MAX_KALI_WORKFLOW_COMMANDS + 1)]
        kali, calls, output = self._run_chat(["check this lab"], replies)
        self.assertEqual(kali.run.call_count, deep_agent.MAX_KALI_WORKFLOW_COMMANDS)
        self.assertTrue(calls[-1][0])
        self.assertIn("20-command limit", output)

    def test_timed_out_model_command_stops_before_another_model_call(self):
        first = deep_agent._make_tool_call("run_kali_command", {"command": "sleep 200"})
        unused = deep_agent._make_tool_call("run_kali_command", {"command": "whoami"})
        kali, calls, output = self._run_chat(
            ["check Kali"],
            [{"content": "", "tool_calls": [first]}, {"content": "", "tool_calls": [unused]}],
            last_record={
                "command": "sleep 200",
                "execution_state": "timed_out",
                "exit_code": 124,
                "stdout": "",
                "stderr": "",
                "timed_out": True,
            },
        )
        kali.run.assert_called_once_with("sleep 200")
        self.assertEqual(len(calls), 1)
        self.assertIn("command timed out", output)

    def test_large_multiline_heredoc_reaches_ssh_intact(self):
        command = "cat > ~/portfolio/index.html <<'EOF'\n" + "<p>It's a page: $HOME</p>\n" * 150 + "EOF"
        channel = Mock()
        channel.recv_ready.return_value = False
        channel.recv_stderr_ready.return_value = False
        channel.exit_status_ready.return_value = True
        channel.recv_exit_status.return_value = 0
        client = Mock()
        client.get_transport.return_value.open_session.return_value = channel
        runner = kali_access.KaliAccess()
        with patch.object(runner, "connect", return_value=client), contextlib.redirect_stdout(io.StringIO()):
            result = runner.run(command.replace("\n", "\r\n"))
        wrapped = shlex.split(channel.exec_command.call_args.args[0])
        self.assertEqual(wrapped[-1], command + "\n")
        self.assertIn("[exit 0;", result)

    def test_command_duration_excludes_connection_time(self):
        now = [0.0]
        channel = Mock()
        channel.recv_ready.return_value = False
        channel.recv_stderr_ready.return_value = False

        def finish_command():
            now[0] += 0.25
            return True

        channel.exit_status_ready.side_effect = finish_command
        channel.recv_exit_status.return_value = 0
        client = Mock()
        client.get_transport.return_value.open_session.return_value = channel
        runner = kali_access.KaliAccess()

        def connect():
            now[0] += 10  # Model a slow password prompt or SSH connection.
            return client

        output = io.StringIO()
        with (patch.object(runner, "connect", side_effect=connect),
              patch.object(kali_access.time, "monotonic", side_effect=lambda: now[0]),
              contextlib.redirect_stdout(output)):
            result = runner.run("hostname")

        self.assertEqual(runner.last_record["duration_seconds"], 0.25)
        self.assertIn("[exit 0; 0.25s]", result)

    def test_sudo_password_is_prompted_locally_and_sent_over_stdin(self):
        channel = Mock()
        channel.recv_ready.return_value = False
        channel.recv_stderr_ready.return_value = False
        channel.exit_status_ready.return_value = True
        channel.recv_exit_status.return_value = 0
        transport = Mock()
        transport.is_active.return_value = True
        transport.open_session.return_value = channel
        client = Mock()
        client.get_transport.return_value = transport
        runner = kali_access.KaliAccess()
        output = io.StringIO()
        with (patch.object(runner, "connect", return_value=client),
              patch.object(kali_access.getpass, "getpass", return_value="secret-pass") as prompt,
              contextlib.redirect_stdout(output)):
            result = runner.run("sudo id -u")

        prompt.assert_called_once()
        remote_command = channel.exec_command.call_args.args[0]
        remote_script = shlex.split(remote_command)[-1]
        self.assertIn("sudo -S -p '' -- bash", remote_script)
        self.assertIn("exec </dev/null; id -u", remote_script)
        self.assertNotIn("secret-pass", remote_command)
        self.assertEqual(bytes(channel.sendall.call_args.args[0]), b"secret-pass\n")
        self.assertEqual(runner.last_record["command"], "sudo id -u")
        self.assertNotIn("secret-pass", json.dumps(runner.last_record))
        evidence = deep_agent.EvidenceLedger().record_command("sudo-call", runner.last_record)
        recorded = evidence.to_dict(include_streams=False)
        self.assertEqual(recorded["privilege_mode"], "sudo")
        self.assertTrue(recorded["sudo_password_prompted"])
        self.assertTrue(recorded["sudo_password_sent"])
        self.assertNotIn("secret-pass", json.dumps(recorded))
        self.assertIn("[exit 0;", result)

    def test_bare_sudo_prompts_and_runs_only_a_credential_check(self):
        channel = Mock()
        channel.recv_ready.return_value = False
        channel.recv_stderr_ready.return_value = False
        channel.exit_status_ready.return_value = True
        channel.recv_exit_status.return_value = 0
        transport = Mock()
        transport.is_active.return_value = True
        transport.open_session.return_value = channel
        client = Mock()
        client.get_transport.return_value = transport
        runner = kali_access.KaliAccess()
        with (patch.object(runner, "connect", return_value=client),
              patch.object(kali_access.getpass, "getpass", return_value="secret-pass") as prompt,
              contextlib.redirect_stdout(io.StringIO())):
            runner.run("sudo")

        prompt.assert_called_once()
        remote_command = channel.exec_command.call_args.args[0]
        remote_script = shlex.split(remote_command)[-1]
        self.assertIn("sudo -S -p '' -v", remote_script)
        self.assertNotIn("-- bash", remote_script)
        self.assertNotIn("secret-pass", remote_command)
        self.assertEqual(bytes(channel.sendall.call_args.args[0]), b"secret-pass\n")
        self.assertEqual(runner.last_record["privilege_mode"], "sudo_validation")
        self.assertTrue(runner.last_record["sudo_access_validated"])
        self.assertNotIn("secret-pass", json.dumps(runner.last_record))
        self.assertTrue(runner.sudo_mode)
        self.assertEqual(runner._sudo_password, bytearray(b"secret-pass"))

    def test_bare_sudo_then_command_uses_one_local_password_prompt(self):
        channels = []
        for _ in range(2):
            channel = Mock()
            channel.recv_ready.return_value = False
            channel.recv_stderr_ready.return_value = False
            channel.exit_status_ready.return_value = True
            channel.recv_exit_status.return_value = 0
            channels.append(channel)
        transport = Mock()
        transport.is_active.return_value = True
        transport.open_session.side_effect = channels
        client = Mock()
        client.get_transport.return_value = transport
        runner = kali_access.KaliAccess()
        with (patch.object(runner, "connect", return_value=client),
              patch.object(kali_access.getpass, "getpass", return_value="secret-pass") as prompt,
              contextlib.redirect_stdout(io.StringIO())):
            runner.run("sudo")
            runner.run("id -u")

        prompt.assert_called_once()
        self.assertIn("sudo -S -p '' -v", shlex.split(channels[0].exec_command.call_args.args[0])[-1])
        self.assertIn("sudo -S -p '' -- bash", shlex.split(channels[1].exec_command.call_args.args[0])[-1])
        self.assertEqual(bytes(channels[0].sendall.call_args.args[0]), b"secret-pass\n")
        self.assertEqual(bytes(channels[1].sendall.call_args.args[0]), b"secret-pass\n")
        self.assertTrue(runner.sudo_mode)
        self.assertEqual(runner.last_record["privilege_mode"], "sudo_session")

    def test_wrong_sudo_password_reprompts_and_records_only_attempt_metadata(self):
        class Channel:
            def __init__(self, stderr, exit_code):
                self.pending_stderr = [stderr] if stderr else []
                self.exit_code = exit_code
                self.sent = []
                self.command = None

            def exec_command(self, command):
                self.command = command

            def sendall(self, data):
                self.sent.append(bytes(data))

            def shutdown_write(self):
                pass

            def recv_ready(self):
                return False

            def recv(self, _size):
                return b""

            def recv_stderr_ready(self):
                return bool(self.pending_stderr)

            def recv_stderr(self, _size):
                return self.pending_stderr.pop(0)

            def exit_status_ready(self):
                return not self.pending_stderr

            def recv_exit_status(self):
                return self.exit_code

            def close(self):
                pass

        channels = [
            Channel(b"Sorry, try again.\nsudo: no password was provided\n", 1),
            Channel(b"", 0),
        ]
        transport = Mock()
        transport.is_active.return_value = True
        transport.open_session.side_effect = channels
        client = Mock()
        client.get_transport.return_value = transport
        runner = kali_access.KaliAccess()
        output = io.StringIO()
        with (patch.object(runner, "connect", return_value=client),
              patch.object(kali_access.getpass, "getpass", side_effect=["wrong-pass", "right-pass"]) as prompt,
              contextlib.redirect_stdout(output)):
            runner.run("sudo")

        self.assertEqual(prompt.call_count, 2)
        self.assertIn("attempt 2/3", prompt.call_args_list[1].args[0])
        self.assertEqual(channels[0].sent, [b"wrong-pass\n"])
        self.assertEqual(channels[1].sent, [b"right-pass\n"])
        self.assertTrue(runner.sudo_mode)
        self.assertEqual(runner.last_record["sudo_attempt_history"], [
            {"attempt": 1, "password_rejected": True},
            {"attempt": 2, "password_rejected": False},
        ])
        self.assertNotIn("wrong-pass", json.dumps(runner.last_record))
        self.assertNotIn("right-pass", json.dumps(runner.last_record))
        self.assertNotIn("no password was provided", output.getvalue())
        self.assertIn("Kali rejected the sudo password", output.getvalue())

    def test_sudo_password_retries_stop_after_three_attempts(self):
        class Channel:
            def __init__(self):
                self.pending_stderr = [b"Sorry, try again.\nsudo: no password was provided\n"]

            def exec_command(self, _command): pass
            def sendall(self, _data): pass
            def shutdown_write(self): pass
            def recv_ready(self): return False
            def recv(self, _size): return b""
            def recv_stderr_ready(self): return bool(self.pending_stderr)
            def recv_stderr(self, _size): return self.pending_stderr.pop(0)
            def exit_status_ready(self): return not self.pending_stderr
            def recv_exit_status(self): return 1
            def close(self): pass

        channels = [Channel(), Channel(), Channel()]
        transport = Mock()
        transport.is_active.return_value = True
        transport.open_session.side_effect = channels
        client = Mock()
        client.get_transport.return_value = transport
        runner = kali_access.KaliAccess()
        output = io.StringIO()
        with (patch.object(runner, "connect", return_value=client),
              patch.object(kali_access.getpass, "getpass", side_effect=["bad1", "bad2", "bad3"]) as prompt,
              contextlib.redirect_stdout(output)):
            runner.run("sudo")

        self.assertEqual(prompt.call_count, 3)
        self.assertFalse(runner.sudo_mode)
        self.assertEqual(runner.last_record["sudo_attempt"], 3)
        self.assertEqual(len(runner.last_record["sudo_attempt_history"]), 3)
        self.assertIn("after 3 attempts", output.getvalue())

    def test_session_sudo_mode_runs_commands_with_noninteractive_sudo(self):
        channel = Mock()
        channel.recv_ready.return_value = False
        channel.recv_stderr_ready.return_value = False
        channel.exit_status_ready.return_value = True
        channel.recv_exit_status.return_value = 0
        transport = Mock()
        transport.is_active.return_value = True
        transport.open_session.return_value = channel
        client = Mock()
        client.get_transport.return_value = transport
        runner = kali_access.KaliAccess()
        runner.sudo_mode = True
        runner._sudo_password = bytearray(b"session-pass")
        with (patch.object(runner, "connect", return_value=client),
              contextlib.redirect_stdout(io.StringIO())):
            result = runner.run("id -u")

        remote_command = channel.exec_command.call_args.args[0]
        remote_script = shlex.split(remote_command)[-1]
        self.assertIn("sudo -S -p '' -- bash -o pipefail -c", remote_script)
        self.assertIn("exec </dev/null; id -u", remote_script)
        self.assertNotIn("session-pass", remote_command)
        self.assertEqual(bytes(channel.sendall.call_args.args[0]), b"session-pass\n")
        self.assertEqual(runner.last_record["privilege_mode"], "sudo_session")
        self.assertTrue(runner.last_record["sudo_access_validated"])
        self.assertTrue(runner.last_record["sudo_password_sent"])
        self.assertFalse(runner.last_record["sudo_password_prompted"])
        self.assertTrue(runner.sudo_mode)
        self.assertIn("[exit 0;", result)

    def test_explicit_sudo_command_reuses_the_session_password(self):
        channel = Mock()
        channel.recv_ready.return_value = False
        channel.recv_stderr_ready.return_value = False
        channel.exit_status_ready.return_value = True
        channel.recv_exit_status.return_value = 0
        transport = Mock()
        transport.is_active.return_value = True
        transport.open_session.return_value = channel
        client = Mock()
        client.get_transport.return_value = transport
        runner = kali_access.KaliAccess()
        runner.sudo_mode = True
        runner._sudo_password = bytearray(b"session-pass")
        with (patch.object(runner, "connect", return_value=client),
              patch.object(kali_access.getpass, "getpass") as prompt,
              contextlib.redirect_stdout(io.StringIO())):
            runner.run("sudo id -u")

        prompt.assert_not_called()
        self.assertEqual(bytes(channel.sendall.call_args.args[0]), b"session-pass\n")
        self.assertEqual(runner.last_record["privilege_mode"], "sudo")
        self.assertTrue(runner.sudo_mode)

    def test_expired_session_sudo_mode_fails_closed_and_disables_mode(self):
        class Channel:
            def __init__(self):
                self.stderr_pending = [b"sudo: a password is required\n"]

            def exec_command(self, command):
                self.command = command

            def shutdown_write(self):
                pass

            def sendall(self, _data):
                pass

            def recv_ready(self):
                return False

            def recv(self, _size):
                return b""

            def recv_stderr_ready(self):
                return bool(self.stderr_pending)

            def recv_stderr(self, _size):
                return self.stderr_pending.pop(0)

            def exit_status_ready(self):
                return not self.stderr_pending

            def recv_exit_status(self):
                return 1

            def close(self):
                pass

        channel = Channel()
        transport = Mock()
        transport.is_active.return_value = True
        transport.open_session.return_value = channel
        client = Mock()
        client.get_transport.return_value = transport
        runner = kali_access.KaliAccess()
        runner.sudo_mode = True
        runner._sudo_password = bytearray(b"expired-pass")
        with (patch.object(runner, "connect", return_value=client),
              contextlib.redirect_stdout(io.StringIO())):
            result = runner.run("id -u")

        self.assertIn("sudo -S -p '' -- bash", shlex.split(channel.command)[-1])
        self.assertFalse(runner.sudo_mode)
        self.assertIsNone(runner._sudo_password)
        self.assertFalse(runner.last_record["sudo_access_validated"])
        self.assertIn("password is required", result)

    def test_user_exit_wipes_cached_sudo_password(self):
        runner = kali_access.KaliAccess()
        old_password = bytearray(b"secret-pass")
        runner._sudo_password = old_password
        runner.sudo_mode = True
        runner.clear_sudo_mode()

        self.assertFalse(runner.sudo_mode)
        self.assertIsNone(runner._sudo_password)
        self.assertEqual(old_password, bytearray(len(b"secret-pass")))

    def test_sudo_parser_rejects_options_chaining_and_interactive_shells(self):
        self.assertIsNone(kali_access.sudo_command_parts("hostname"))
        self.assertEqual(kali_access.sudo_command_parts("sudo"), [])
        self.assertEqual(kali_access.sudo_command_parts("sudo id -u"), ["id", "-u"])
        with self.assertRaisesRegex(ValueError, "Use `sudo` directly"):
            kali_access.sudo_command_parts("/sudo")
        for command in (
            "id; sudo whoami",
            "sudo id | cat",
            "sudo -n id",
            "sudo su",
            "sudo -i",
            "sudo id > /tmp/result",
        ):
            with self.subTest(command=command), self.assertRaises(ValueError):
                kali_access.sudo_command_parts(command)

    def test_invalid_commands_are_rejected_before_connecting(self):
        runner = kali_access.KaliAccess()
        with patch.object(runner, "connect") as connect:
            for command in ("  ", "x" * 64001, "echo a\x00b"):
                with self.subTest(length=len(command)), self.assertRaises(ValueError):
                    runner.run(command)
            connect.assert_not_called()

    def test_modes_keep_separate_history(self):
        kali, calls, _ = self._run_chat(
            ["hello", "/shell", "ip a", "/chat", "hello again"],
            [{"content": "Hello.", "tool_calls": []},
             {"content": "Kali has an IP.", "tool_calls": []},
             {"content": "Hello again.", "tool_calls": []}],
            shell=False,
        )
        kali.run.assert_called_once_with("ip a")
        self.assertEqual([item[0] for item in calls], [False, False, False])
        self.assertIn("no Kali tools", calls[0][2][0]["content"])
        self.assertIn("no tools in this turn", calls[1][2][0]["content"])
        self.assertEqual([item["role"] for item in calls[2][2]], ["system", "user", "assistant", "user"])

    def test_streaming_native_call_is_assembled(self):
        events = [
            {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "call_1", "function": {"name": "run_kali_command", "arguments": '{"command":"ip'}}]}}]},
            {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": ' a"}'}}]}, "finish_reason": "tool_calls"}]},
        ]
        stream = io.BytesIO(b"".join(b"data: " + json.dumps(event).encode() + b"\n\n" for event in events))
        reply = deep_agent._normalize_reply(deep_agent._streamed_reply(stream))
        self.assertEqual(deep_agent._tool_command(reply)[0], "ip a")

    def test_stream_usage_arrives_after_finish_reason(self):
        events = [
            {"choices": [{"delta": {"content": "Done"}, "finish_reason": None}]},
            {"choices": [{"delta": {}, "finish_reason": "stop"}]},
            {"choices": [], "usage": {"prompt_tokens": 1100, "completion_tokens": 20, "total_tokens": 1120}},
        ]
        stream = io.BytesIO(b"".join(b"data: " + json.dumps(event).encode() + b"\n\n" for event in events)
                            + b"data: [DONE]\n\n")
        result = deep_agent._normalize_reply(deep_agent._streamed_reply(stream))
        self.assertEqual(result["content"], "Done")
        self.assertEqual(result["usage"]["prompt_tokens"], 1100)

    def test_context_indicator_uses_server_limit_and_reports_unknowns(self):
        with patch.object(deep_agent, "CONTEXT_STATUS", {"chat": (1000, 24), "shell": None}):
            with patch.object(deep_agent, "_context_limit", return_value=32768):
                self.assertIn("1,024/32,768 tokens (3.1%)", deep_agent._context_indicator("chat"))
            self.assertIn("usage unavailable", deep_agent._context_indicator("shell"))
            with patch.object(deep_agent, "_context_limit", return_value=None):
                self.assertIn("server limit unavailable", deep_agent._context_indicator("chat"))

    def test_context_limit_reads_llama_props(self):
        response = io.BytesIO(json.dumps({"default_generation_settings": {"n_ctx": 262144}}).encode())
        with (patch.object(deep_agent, "BACKEND", "llama"),
              patch.object(deep_agent, "BASE_URL", "http://127.0.0.1:8080/v1"),
              patch.object(deep_agent, "CONTEXT_LIMIT", None),
              patch.object(deep_agent.urllib.request, "urlopen", return_value=response) as open_url):
            self.assertEqual(deep_agent._context_limit(), 262144)
            self.assertEqual(open_url.call_args.args[0], "http://127.0.0.1:8080/props")

    def test_context_limit_reads_loaded_ollama_model(self):
        response = io.BytesIO(json.dumps({"models": [
            {"name": "other", "context_length": 4096},
            {"name": "qwen3.8:27b", "context_length": 32768},
        ]}).encode())
        with (patch.object(deep_agent, "BACKEND", "ollama"),
              patch.object(deep_agent, "MODEL", "qwen3.8:27b"),
              patch.object(deep_agent, "CONTEXT_LIMIT", None),
              patch.object(deep_agent.urllib.request, "urlopen", return_value=response)):
            self.assertEqual(deep_agent._context_limit(), 32768)

    def test_context_limit_uses_configured_ollama_context_when_unloaded(self):
        response = io.BytesIO(json.dumps({"models": []}).encode())
        with (patch.object(deep_agent, "BACKEND", "ollama"),
              patch.object(deep_agent, "CONTEXT_LIMIT", None),
              patch.dict(os.environ, {"DEEP_AGENT_OLLAMA_NUM_CTX": "32768"}),
              patch.object(deep_agent.urllib.request, "urlopen", return_value=response)):
            self.assertEqual(deep_agent._context_limit(), 32768)

    def test_context_command_shows_indicator_without_model_call(self):
        kali, calls, output = self._run_chat(["/context"], [])
        kali.run.assert_not_called()
        self.assertEqual(calls, [])
        self.assertIn("Context:", output)

    def test_text_tool_array_and_ifm_format_are_accepted(self):
        forms = [
            '<tool_call>[{"name":"run_vm_command","arguments":{"command":"whoami"}}]</tool_call>',
            '<ifm|tool_calls><ifm|tool_call>run_kali_command<ifm|arg_key>command</ifm|arg_key><ifm|arg_value>whoami</ifm|arg_value></ifm|tool_call></ifm|tool_calls>',
        ]
        for value in forms:
            with self.subTest(value=value[:16]):
                reply = deep_agent._normalize_reply({"content": value, "tool_calls": []})
                self.assertEqual(deep_agent._tool_command(reply)[0], "whoami")

    def test_llama_request_offers_tool_and_reads_native_call(self):
        event = {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "call_1", "function": {
            "name": "run_kali_command", "arguments": '{"command":"whoami"}'}}]}, "finish_reason": "tool_calls"}]}

        class Response:
            headers = {"Content-Type": "text/event-stream"}

            def __enter__(self):
                return self

            def __exit__(self, *_):
                return False

            def __iter__(self):
                yield b"data: " + json.dumps(event).encode() + b"\n\n"

        for model in ("bonsai-2-27b", "k2-horizon", "other-local-model"):
            with (self.subTest(model=model), patch.object(deep_agent, "MODEL", model),
                  patch.object(deep_agent.urllib.request, "urlopen", return_value=Response()) as open_url):
                reply = deep_agent._llama_chat([{"role": "user", "content": "whoami"}], tools=True, stream_output=False)
                sent = json.loads(open_url.call_args.args[0].data)
                self.assertEqual(sent["tools"][0]["function"]["name"], "run_kali_command")
                self.assertEqual(sent["stream_options"], {"include_usage": True})
                self.assertFalse(sent["parallel_tool_calls"])
                self.assertEqual(deep_agent._tool_command(reply)[0], "whoami")
                if model == "bonsai-2-27b":
                    self.assertEqual(sent["chat_template_kwargs"], {"reasoning_effort": "medium"})
                    self.assertEqual(sent["max_tokens"], 8192)
                    self.assertEqual(sent["top_k"], 20)
                    self.assertEqual(sent["min_p"], 0.05)
                    self.assertEqual(sent["presence_penalty"], 0.0)
                elif model == "k2-horizon":
                    self.assertEqual(sent["chat_template_kwargs"], {"reasoning_effort": "high"})
                else:
                    self.assertNotIn("chat_template_kwargs", sent)

    def test_ollama_request_offers_tool_and_reads_native_call(self):
        response = io.BytesIO(json.dumps({"message": {"content": "", "tool_calls": [
            {"function": {"name": "run_kali_command", "arguments": {"command": "whoami"}}}]},
            "prompt_eval_count": 12, "eval_count": 5}).encode())
        with (patch.object(deep_agent.urllib.request, "urlopen", return_value=response) as open_url,
              patch.dict(os.environ, {"DEEP_AGENT_OLLAMA_NUM_CTX": "32768", "DEEP_AGENT_OLLAMA_THINK": "xhigh"})):
            reply = deep_agent._ollama_chat([{"role": "user", "content": "whoami"}], tools=True, stream_output=False)
        sent = json.loads(open_url.call_args.args[0].data)
        self.assertEqual(sent["tools"][0]["function"]["name"], "run_kali_command")
        self.assertEqual(sent["options"], {"num_ctx": 32768})
        self.assertEqual(sent["think"], "xhigh")
        self.assertEqual(open_url.call_args.args[0].full_url, "http://127.0.0.1:11434/api/chat")
        self.assertEqual(deep_agent._tool_command(reply)[0], "whoami")

    def test_ollama_request_converts_historical_tool_arguments_to_object(self):
        tool_call = deep_agent._make_tool_call("run_kali_command", {"command": "hostname -I"})
        messages = [
            {"role": "assistant", "content": "", "tool_calls": [tool_call]},
            {"role": "tool", "tool_call_id": tool_call["id"], "tool_name": "run_kali_command", "content": "10.0.2.15"},
        ]
        response = io.BytesIO(json.dumps({"message": {"content": "The Kali IP is 10.0.2.15.", "tool_calls": []},
                                           "prompt_eval_count": 30, "eval_count": 8}).encode())
        with patch.object(deep_agent.urllib.request, "urlopen", return_value=response) as open_url:
            deep_agent._ollama_chat(messages, tools=True, stream_output=False)
        sent = json.loads(open_url.call_args.args[0].data)
        sent_call = sent["messages"][0]["tool_calls"][0]
        self.assertEqual(sent_call["function"]["arguments"], {"command": "hostname -I"})
        self.assertIsInstance(messages[0]["tool_calls"][0]["function"]["arguments"], str)

    def test_kali_output_stays_live_after_model_capture_limit(self):
        class Channel:
            def __init__(self):
                self.pending = [b"helloworld"]

            def exec_command(self, command):
                self.command = command

            def shutdown_write(self):
                pass

            def recv_ready(self):
                return bool(self.pending)

            def recv(self, _size):
                return self.pending.pop(0)

            def recv_stderr_ready(self):
                return False

            def recv_stderr(self, _size):
                return b""

            def exit_status_ready(self):
                return not self.pending

            def recv_exit_status(self):
                return 0

            def close(self):
                pass

        channel = Channel()
        transport = Mock()
        transport.is_active.return_value = True
        transport.open_session.return_value = channel
        client = Mock()
        client.get_transport.return_value = transport
        runner = kali_access.KaliAccess()
        output = io.StringIO()
        with (patch.object(runner, "connect", return_value=client),
              patch.object(kali_access, "OUTPUT_LIMIT", 5),
              contextlib.redirect_stdout(output)):
            result = runner.run("printf helloworld")
        self.assertIn("helloworld", output.getvalue())
        self.assertIn("bash -o pipefail -c", channel.command)
        self.assertIn("hello", result)
        self.assertNotIn("helloworld", result)
        self.assertIn("INCOMPLETE", result)


if __name__ == "__main__":
    unittest.main()
