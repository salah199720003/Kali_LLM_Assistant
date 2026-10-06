"""Transactional, versioned local notes; a saved statement is not verified truth."""

import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path


def now():
    return datetime.now(timezone.utc)


def validate_note(text, key="", replaces="", ttl_days=None):
    if not isinstance(text, str):
        raise ValueError("text must be a string")
    cleaned = " ".join(text.split())
    if not cleaned or len(text) > 500 or any(ord(c) < 32 for c in text.replace("\n", "").replace("\t", "")):
        raise ValueError("text must be 1-500 printable characters")
    for name, value in (("key", key), ("replaces", replaces)):
        if not isinstance(value, str) or len(value) > 120 or any(ord(c) < 32 for c in value):
            raise ValueError(f"{name} must be printable text up to 120 characters")
    if ttl_days is not None and (type(ttl_days) is not int or not 1 <= ttl_days <= 3650):
        raise ValueError("ttl_days must be 1-3650")
    return cleaned


class MemoryStore:
    def __init__(self, legacy_path: Path):
        self.legacy_path = legacy_path
        self.path = legacy_path.with_suffix(".sqlite3")

    @contextmanager
    def _connect(self):
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        try:
            connection.execute("CREATE TABLE IF NOT EXISTS notes (id TEXT PRIMARY KEY, fact_key TEXT, "
                               "text TEXT, status TEXT, created TEXT, expires TEXT, replaces TEXT, source TEXT)")
            connection.execute("CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT)")
            with connection:
                connection.execute("BEGIN IMMEDIATE")
                if not connection.execute("SELECT 1 FROM metadata WHERE key='legacy_imported'").fetchone():
                    try:
                        legacy = self.legacy_path.read_text(encoding="utf-8")
                    except FileNotFoundError:
                        legacy = ""
                    # Retain all legacy nonempty lines, including unstructured headers.
                    for line in legacy.splitlines():
                        if line.strip():
                            connection.execute("INSERT INTO notes VALUES (?, '', ?, 'active', ?, NULL, '', 'legacy')",
                                               (uuid.uuid4().hex, line.strip(), now().isoformat()))
                    connection.execute("INSERT INTO metadata VALUES ('legacy_imported', '1')")
            with connection:
                yield connection
        finally:
            connection.close()

    def save(self, text, key="", replaces="", ttl_days=None):
        cleaned = validate_note(text, key, replaces, ttl_days)
        key = key.strip().casefold()
        stamp = now()
        expires = (stamp + timedelta(days=ttl_days)).isoformat() if ttl_days is not None else None
        note_id = uuid.uuid4().hex
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            previous = connection.execute("SELECT * FROM notes WHERE id=?", (replaces,)).fetchone() if replaces else None
            if replaces and previous is None:
                raise ValueError("replaces must identify an existing note; use read_lab_notes")
            if previous:
                if key and previous["fact_key"] and key != previous["fact_key"]:
                    raise ValueError("A replacement must use the same key as its previous note")
                key = key or previous["fact_key"]
                connection.execute("UPDATE notes SET status='superseded' WHERE id=?", (replaces,))
            # Exact repeats are idempotent; expired/stale notes require a new record.
            repeat = connection.execute("SELECT * FROM notes WHERE fact_key=? AND text=? AND status='active' "
                                        "AND (expires IS NULL OR expires>?)", (key, cleaned, stamp.isoformat())).fetchone()
            if repeat and not replaces:
                return dict(repeat)
            conflicts = connection.execute("SELECT id FROM notes WHERE fact_key=? AND status IN ('active','disputed') "
                                           "AND (expires IS NULL OR expires>?)", (key, stamp.isoformat())).fetchall() if key else []
            status = "disputed" if conflicts else "active"
            if conflicts:
                connection.executemany("UPDATE notes SET status='disputed' WHERE id=?", [(r["id"],) for r in conflicts])
            connection.execute("INSERT INTO notes VALUES (?, ?, ?, ?, ?, ?, ?, 'agent')",
                               (note_id, key, cleaned, status, stamp.isoformat(), expires, replaces))
            result = dict(connection.execute("SELECT * FROM notes WHERE id=?", (note_id,)).fetchone())
        # Legacy file remains available as an append-only human audit. SQLite owns status.
        try:
            with self.legacy_path.open("a", encoding="utf-8") as handle:
                handle.write(f"- [{stamp.date()}] {cleaned}\n")
        except OSError:
            result["audit_warning"] = "Structured note saved, but the legacy markdown audit could not be appended"
        return result

    def update(self, note_id, status):
        if not isinstance(note_id, str) or status not in {"stale", "active"}:
            raise ValueError("Provide note_id and status=stale|active")
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute("SELECT * FROM notes WHERE id=?", (note_id,)).fetchone()
            if row is None:
                raise ValueError("Unknown note_id")
            if row["status"] == "superseded":
                raise ValueError("Superseded notes remain historical; save a new replacement")
            if status == "active":
                if row["expires"] and row["expires"] <= now().isoformat():
                    raise ValueError("Expired notes need a new observation and replacement")
                others = connection.execute("SELECT id FROM notes WHERE fact_key=? AND id<>? "
                                            "AND status IN ('active','disputed') AND (expires IS NULL OR expires>?)",
                                            (row["fact_key"], note_id, now().isoformat())).fetchall() if row["fact_key"] else []
                if others:
                    raise ValueError("Resolve competing notes with an explicit replacement before activation")
            connection.execute("UPDATE notes SET status=? WHERE id=?", (status, note_id))
        return {"note_id": note_id, "status": status, "verified": False}

    def list(self, query="", offset=0, limit=20, include_inactive=False):
        if not isinstance(query, str) or len(query) > 240 or type(include_inactive) is not bool:
            raise ValueError("query must be short text; include_inactive must be boolean")
        if type(offset) is not int or offset < 0 or type(limit) is not int or not 1 <= limit <= 20:
            raise ValueError("offset must be nonnegative; limit must be 1-20")
        with self._connect() as connection:
            connection.execute("UPDATE notes SET status='stale' WHERE status IN ('active','disputed') "
                               "AND expires IS NOT NULL AND expires<=?", (now().isoformat(),))
            where = "WHERE instr(lower(id || ' ' || fact_key || ' ' || text), lower(?))>0"
            if not include_inactive:
                where += " AND status IN ('active','disputed')"
            total = connection.execute("SELECT count(*) FROM notes " + where, (query,)).fetchone()[0]
            rows = connection.execute("SELECT * FROM notes " + where + " ORDER BY rowid DESC LIMIT ? OFFSET ?",
                                      (query, limit, offset)).fetchall()
        return {"notes": [dict(row) for row in rows], "total": total,
                "next_offset": offset + len(rows) if offset + len(rows) < total else None,
                "note": "Recorded statements, not verified facts. Disputed notes require reconciliation with live evidence."}

    def prompt_block(self, max_characters=4000):
        if not self.path.exists():
            try:
                text = self.legacy_path.read_text(encoding="utf-8").strip()
            except OSError:
                return ""
            if not text:
                return ""
            body = text[:max_characters]
            if len(text) > max_characters:
                body += "\n[More legacy notes available through read_lab_notes.]"
        else:
            snapshot = self.list(limit=20)
            lines = [f"[{r['id']} {r['status']} key={r['fact_key'] or 'unstructured'}] {r['text']}"
                     for r in snapshot["notes"]]
            body = "\n".join(lines)[:max_characters]
            body += "\nUse read_lab_notes for the complete paginated record and inactive history."
        return ("\n\nPersistent lab notes (recorded statements, not verified truth; verify against live evidence). "
                "Disputed notes conflict; stale/superseded notes are excluded from this snapshot. "
                "Legacy text has no automatic contradiction detection:\n" + body)
