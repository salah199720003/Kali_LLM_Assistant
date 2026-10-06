"""Offline tests for context retrieval, history bounds and versioned notes."""

import contextlib
import io
import json
import os
import sqlite3
import tempfile
import unittest
from datetime import timedelta
from pathlib import Path
from unittest.mock import Mock, patch

import deep_agent
from agent_tools import LOCAL_TOOL_NAMES, validate_local_arguments
from conversation_context import ResultArchive, bounded_history, compact_tool_results, serialized_size, ContextCapacityError
from evidence import EvidenceLedger
from persistent_memory import MemoryStore, now


class ContextMemoryTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.archive = ResultArchive(self.root / "results.sqlite3")
        self.memory = MemoryStore(self.root / "lab_notes.md")
        self.ledger = EvidenceLedger()
        stack = contextlib.ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(patch.object(deep_agent, "RESULT_ARCHIVE", self.archive))
        stack.enter_context(patch.object(deep_agent, "_LAB_NOTES_PATH", self.memory.legacy_path))
        stack.enter_context(patch.object(deep_agent, "EVIDENCE_LEDGER", self.ledger))

    def test_pages_reconstruct_exact_capture_including_middle_and_tail(self):
        text = "head\n" + "α😀middle\n" * 1300 + "TAIL"
        self.archive.save("result", "web_search", "", text)
        offset, pages = 0, []
        while True:
            page = self.archive.read("result", offset=offset, limit=333)
            pages.append(page["text"])
            if page["next_offset"] is None:
                break
            offset = page["next_offset"]
        self.assertEqual("".join(pages), text)
        self.assertEqual(self.archive.read("result", offset=len(text)+10)["text"], "")

    def test_capture_limit_is_reported_and_streams_are_separate(self):
        content = json.dumps({"command_evidence": {"stdout": "out", "stderr": "err", "output_truncated": True}})
        self.archive.save("result", "run_kali_command", "example", content)
        page = self.archive.read("result", "stderr")
        self.assertEqual(page["text"], "err")
        self.assertTrue(page["capture_truncated"])

    def test_invalid_ids_pages_and_missing_streams_fail_explicitly(self):
        self.archive.save("r", "web_search", "", "plain text")
        for kwargs in ({"result_id": "missing"}, {"result_id": "r", "stream": "stdout"},
                       {"result_id": "r", "offset": -1}, {"result_id": "r", "limit": 4097},
                       {"result_id": "r", "offset": True}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.archive.read(**kwargs)

    def test_archive_is_persistent_paginated_and_search_is_literal(self):
        for n in range(5):
            self.archive.save(str(n), "web_search", "", f"result {n} %literal")
        reopened = ResultArchive(self.archive.path)
        first = reopened.list("%literal", limit=2)
        self.assertEqual(first["total"], 5)
        self.assertEqual(first["next_offset"], 2)
        self.assertEqual(reopened.read(first["results"][0]["result_id"])["text"], "result 4 %literal")

    def test_compaction_exposes_tails_and_middle_remains_retrievable(self):
        stream = "a"*4000 + "IMPORTANT_MIDDLE" + "b"*4000 + "IMPORTANT_TAIL"
        evidence = self.ledger.record_command({"evidence_id": "ev", "command": "inspect",
                                               "stdout": stream, "stderr": "error tail",
                                               "execution_state": "completed", "exit_code": 0})
        messages = []
        deep_agent._record_tool_result(messages, evidence.command, "call", "", evidence)
        compact_tool_results(messages, self.ledger, self.archive, keep=0)
        compact = json.loads(messages[-1]["content"])
        self.assertIn("IMPORTANT_TAIL", compact["stdout_tail"])
        retrieved = self.archive.read(compact["result_id"], "stdout", offset=4000, limit=16)
        self.assertTrue(retrieved["text"].startswith("IMPORTANT_MIDDLE"))
        self.assertEqual(self.ledger.command_by_evidence_id("ev").stdout, stream)

    def test_research_compaction_does_not_require_repeating_a_query(self):
        text = json.dumps({"results": ["head" + "x"*4000 + "middle fact" + "x"*4000 + "tail"]})
        messages = []
        deep_agent._record_tool_result(messages, "", "call", text, tool_name="web_search")
        compact_tool_results(messages, self.ledger, self.archive, keep=0)
        compact = json.loads(messages[-1]["content"])
        self.assertTrue(compact["research_compact"])
        page = self.archive.read(compact["result_id"], offset=text.index("middle fact"), limit=11)
        self.assertEqual(page["text"], "middle fact")

    def test_repeated_model_call_ids_do_not_overwrite_archived_evidence(self):
        messages = []
        for text in ("first result", "second result"):
            deep_agent._record_tool_result(messages, "", "reused", text, tool_name="web_search")
        ids = [m["_result_id"] for m in messages if m["role"] == "tool"]
        self.assertNotEqual(*ids)
        self.assertEqual([self.archive.read(i)["text"] for i in ids], ["first result", "second result"])

    def test_history_bound_keeps_latest_goal_and_whole_tool_pairs(self):
        messages = [{"role": "system", "content": "system"}, {"role": "user", "content": "original goal"}]
        for n in range(12):
            messages.extend([{"role": "assistant", "content": "", "tool_calls": [deep_agent._make_tool_call("read_tool_result", {"result_id": "r"}, str(n))]},
                             {"role": "tool", "tool_call_id": str(n), "content": "x"*600}])
        view = bounded_history(messages, max_bytes=2200, max_messages=8)
        self.assertLessEqual(serialized_size(view), 2200)
        self.assertLessEqual(len(view), 8)
        self.assertIn({"role": "user", "content": "original goal"}, view)
        for i, message in enumerate(view):
            if message["role"] == "tool":
                self.assertEqual(view[i-1]["tool_calls"][0]["id"], message["tool_call_id"])
        self.assertEqual(len(messages), 26)  # Task-local indexes remain stable.

    def test_history_notice_does_not_accumulate_and_latest_turn_can_be_protected(self):
        messages = [{"role": "system", "content": "system"}]
        for n in range(12):
            messages += [{"role": "user", "content": f"question {n}"}, {"role": "assistant", "content": "x"*400}]
        first = bounded_history(messages, 2200, 10, preserve_latest_turn=True)
        first += [{"role": "user", "content": "new goal"}, {"role": "assistant", "content": "x"*1200}]
        second = bounded_history(first, 2200, 10)
        self.assertEqual(sum(m["role"] == "system" for m in second), 2)
        self.assertIn({"role": "user", "content": "new goal"}, second)

    def test_oversized_latest_request_is_not_silently_cut(self):
        with self.assertRaises(ContextCapacityError):
            bounded_history([{"role": "system", "content": "system"}, {"role": "user", "content": "x"*10000}], 2000, 10)

    def test_omitted_user_constraints_remain_retrievable(self):
        messages = [{"role": "system", "content": "system"},
                    {"role": "user", "content": "Earlier constraint: preserve the original configuration"},
                    {"role": "assistant", "content": "x"*3000},
                    {"role": "user", "content": "current goal"}, {"role": "assistant", "content": "latest"}]
        view = bounded_history(messages, 1600, 8, archive=self.archive)
        self.assertIn({"role": "user", "content": "current goal"}, view)
        result = self.archive.list("Earlier constraint")
        self.assertEqual(result["total"], 1)
        recovered = self.archive.read(result["results"][0]["result_id"])["text"]
        self.assertIn("preserve the original configuration", recovered)

    def test_archive_failure_does_not_erase_command_truth_or_compact_unrecoverable_output(self):
        evidence = self.ledger.record_command({"command": "inspect", "stdout": "x"*5000,
                                               "execution_state": "completed", "exit_code": 0})
        messages = []
        with patch.object(self.archive, "save", side_effect=sqlite3.OperationalError("disk failure")), contextlib.redirect_stdout(io.StringIO()):
            deep_agent._record_tool_result(messages, evidence.command, "call", "", evidence)
            before = messages[-1]["content"]
            compact_tool_results(messages, self.ledger, self.archive, keep=0)
        self.assertEqual(messages[-1]["content"], before)
        self.assertEqual(messages[-1]["role"], "tool")
        self.assertEqual(self.ledger.command_by_evidence_id(evidence.evidence_id).stdout, "x"*5000)

    def test_corrupt_memory_does_not_reactivate_legacy_facts(self):
        self.memory.legacy_path.write_text("outdated fact", encoding="utf-8")
        self.memory.path.write_bytes(b"not a sqlite database")
        block = deep_agent._lab_notes_block(self.memory.legacy_path)
        self.assertIn("unavailable", block)
        self.assertNotIn("outdated fact", block)

    def test_request_view_is_bounded_without_mutating_controller_history(self):
        messages = [{"role": "system", "content": "generic"}, {"role": "user", "content": "goal"}]
        messages += [{"role": "assistant", "content": "x"*1000} for _ in range(20)]
        messages[-1]["_private"] = "not a protocol field"
        with (patch.object(deep_agent, "CONTEXT_LIMIT", None),
              patch.dict(os.environ, {"DEEP_AGENT_HISTORY_MAX_BYTES": "4096", "DEEP_AGENT_HISTORY_MAX_MESSAGES": "8"}),
              patch.object(deep_agent, "BACKEND", "llama"),
              patch.object(deep_agent, "_llama_chat", return_value={"content": "ok"}) as model):
            deep_agent._model_chat(messages, tools=False, stream_output=False)
        sent = model.call_args.args[0]
        self.assertLessEqual(serialized_size(sent), 4096)
        self.assertEqual(len(messages), 22)
        self.assertNotIn("_private", sent[-1])

    def test_legacy_memory_migration_is_once_only_and_preserves_original_text(self):
        text = "# existing notes\n- [2026-01-01] first fact\n"
        self.memory.legacy_path.write_text(text, encoding="utf-8")
        self.assertEqual(self.memory.list()["total"], 2)
        self.assertEqual(MemoryStore(self.memory.legacy_path).list()["total"], 2)
        self.assertEqual(self.memory.legacy_path.read_text(encoding="utf-8"), text)

    def test_key_conflicts_are_visible_without_silently_picking_a_value(self):
        old = self.memory.save("service address A", key="Host.Address")
        new = self.memory.save("service address B", key="host.address")
        rows = MemoryStore(self.memory.legacy_path).list()["notes"]
        self.assertEqual({r["status"] for r in rows}, {"disputed"})
        self.assertIn("disputed", self.memory.prompt_block())
        self.assertNotEqual(old["id"], new["id"])

    def test_explicit_replacement_is_versioned_and_stale_notes_are_excluded(self):
        old = self.memory.save("old endpoint", key="host.endpoint")
        new = self.memory.save("new endpoint", replaces=old["id"])
        self.assertEqual(new["fact_key"], "host.endpoint")
        self.assertEqual([r["text"] for r in self.memory.list()["notes"]], ["new endpoint"])
        historical = {r["id"]: r["status"] for r in self.memory.list(include_inactive=True)["notes"]}
        self.assertEqual(historical[old["id"]], "superseded")
        self.memory.update(new["id"], "stale")
        self.assertEqual(self.memory.list()["total"], 0)
        self.assertNotIn("new endpoint", self.memory.prompt_block())

    def test_expiry_and_reactivation_need_current_observation(self):
        note = self.memory.save("temporary endpoint", key="service.endpoint", ttl_days=1)
        with patch("persistent_memory.now", return_value=now()+timedelta(days=2)):
            self.assertEqual(self.memory.list()["total"], 0)
            self.assertEqual(self.memory.list(include_inactive=True)["notes"][0]["status"], "stale")
            with self.assertRaises(ValueError):
                self.memory.update(note["id"], "active")

    def test_conflict_resolution_requires_explicit_history_actions(self):
        first = self.memory.save("value A", key="host.value")
        second = self.memory.save("value B", key="host.value")
        with self.assertRaises(ValueError):
            self.memory.update(second["id"], "active")
        self.memory.update(first["id"], "stale")
        updated = self.memory.update(second["id"], "active")
        self.assertFalse(updated["verified"])
        self.assertEqual(self.memory.list()["notes"][0]["text"], "value B")

    def test_bad_replacement_does_not_modify_prior_note_and_duplicate_is_idempotent(self):
        first = self.memory.save("value A", key="host.value")
        self.assertEqual(self.memory.save("value A", key="host.value")["id"], first["id"])
        for kwargs in ({"replaces": "missing"}, {"replaces": first["id"], "key": "different"}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.memory.save("bad update", **kwargs)
        self.assertEqual(self.memory.list()["notes"][0]["status"], "active")

    def test_memory_snapshot_is_bounded_and_full_record_remains_readable(self):
        for n in range(25):
            self.memory.save(f"fact {n}: " + "x"*400)
        self.assertLess(len(self.memory.prompt_block()), 4500)
        self.assertEqual(self.memory.list()["total"], 25)
        self.assertIsNotNone(self.memory.list()["next_offset"])
        self.assertEqual(len(self.memory.list(offset=20)["notes"]), 5)

    def test_tool_validation_and_registration_for_both_profiles(self):
        for tools in (deep_agent.SHELL_TOOLS, deep_agent._unrestricted_tools()):
            self.assertTrue(LOCAL_TOOL_NAMES <= {t["function"]["name"] for t in tools})
        call = deep_agent._make_tool_call("save_lab_note", {"text": "fact", "key": "host.value", "ttl_days": 1})
        self.assertTrue(deep_agent._tool_call_schema_valid({"tool_calls": [call]}))
        self.assertEqual(deep_agent._tool_request({"tool_calls": [call]})[0], "save_lab_note")
        for args in ({"result_id": "r", "limit": True}, {"result_id": "r", "limit": 5000},
                     {"result_id": "r", "unknown": "x"}):
            with self.subTest(args=args), self.assertRaises(ValueError):
                validate_local_arguments("read_tool_result", args)

    def test_retrieval_tools_run_in_both_engines_without_ssh(self):
        self.archive.save("captured", "web_search", "", "important captured detail")
        for mode in ("guarded", "unrestricted"):
            with self.subTest(mode=mode):
                messages = [{"role": "system", "content": "system"}, {"role": "user", "content": "review older evidence"}]
                kali = Mock()
                snapshots = []
                replies = iter([{"content": "", "tool_calls": [deep_agent._make_tool_call("read_tool_result", {"result_id": "captured"})]},
                                {"content": "Captured detail retrieved.", "tool_calls": []}])
                def model(history, **kwargs):
                    snapshots.append([dict(m) for m in history])
                    return next(replies)
                with (patch.dict(os.environ, {"DEEP_AGENT_EXECUTION_MODE": mode}),
                      patch.object(deep_agent, "MODEL", "agent-27b"),
                      patch.object(deep_agent, "_model_chat", side_effect=model),
                      contextlib.redirect_stdout(io.StringIO())):
                    deep_agent._run_kali_turn(kali, messages, "review older evidence", None)
                kali.run.assert_not_called()
                self.assertTrue(any("important captured detail" in m.get("content", "")
                                    for snapshot in snapshots[1:] for m in snapshot if m["role"] == "tool"))

    def test_memory_update_is_visible_on_next_model_request_without_restart(self):
        note = self.memory.save("outdated endpoint", key="host.endpoint")
        messages = [{"role": "system", "content": deep_agent._shell_system_prompt()}, {"role": "user", "content": "next"}]
        self.memory.save("current endpoint", replaces=note["id"])
        with (patch.object(deep_agent, "ACTIVE_MODE", "shell"), patch.object(deep_agent, "BACKEND", "llama"),
              patch.object(deep_agent, "CONTEXT_LIMIT", None),
              patch.object(deep_agent, "_llama_chat", return_value={"content": "ok"}) as model):
            deep_agent._model_chat(messages, tools=True, stream_output=False)
        system = model.call_args.args[0][0]["content"]
        self.assertIn("current endpoint", system)
        self.assertNotIn("outdated endpoint", system)


if __name__ == "__main__":
    unittest.main()
