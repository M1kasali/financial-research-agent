"""Local SQLite journal. Trusted local state only; no pickle, credentials or headers.

A process-held OS file lock serializes runners for this database. No lease timeout
can accidentally allow a second runner while a slow model request is still active.
"""

import json
import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path


class RecoveryError(RuntimeError):
    pass


class UncertainRequest(RecoveryError):
    pass


def encode(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


class SQLiteTaskStore:
    VERSION = "1"

    def __init__(self, path: Path):
        self.path = Path(path).resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._locked = False

    @contextmanager
    def _connect(self):
        connection = sqlite3.connect(self.path, timeout=5)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA synchronous=FULL")
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def _initialize(self):
        with self._connect() as db:
            tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            if tables:
                if "store_meta" not in tables:
                    raise RecoveryError("Not a financial-agent state database")
                version = db.execute("SELECT value FROM store_meta WHERE key='schema_version'").fetchone()
                if version is None or version[0] != self.VERSION:
                    raise RecoveryError("Unsupported state schema version")
            else:
                db.executescript("""
                    CREATE TABLE store_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
                    INSERT INTO store_meta VALUES ('schema_version', '1');
                    CREATE TABLE tasks (
                        task_id TEXT PRIMARY KEY, identity TEXT NOT NULL,
                        checkpoint TEXT NOT NULL, result TEXT,
                        status TEXT NOT NULL DEFAULT 'running', revision INTEGER NOT NULL DEFAULT 0);
                    CREATE TABLE calls (
                        task_id TEXT NOT NULL, call_id TEXT NOT NULL, request_hash TEXT NOT NULL,
                        status TEXT NOT NULL, response TEXT, error TEXT,
                        PRIMARY KEY (task_id, call_id));
                    CREATE TABLE attempts (
                        task_id TEXT NOT NULL, number INTEGER NOT NULL, call_id TEXT NOT NULL,
                        reserved_total INTEGER NOT NULL, event TEXT NOT NULL,
                        PRIMARY KEY (task_id, number));
                    CREATE TABLE checkpoints (
                        task_id TEXT NOT NULL, revision INTEGER NOT NULL, phase TEXT NOT NULL,
                        step INTEGER NOT NULL, snapshot TEXT NOT NULL,
                        PRIMARY KEY (task_id, revision));
                """)
        if os.name != "nt":
            self.path.chmod(0o600)

    @contextmanager
    def exclusive(self):
        if self._locked:
            raise RecoveryError("State store is already in use")
        lock_path = self.path.with_suffix(self.path.suffix + ".lock")
        handle = lock_path.open("a+b")
        acquired = False
        try:
            if os.name == "nt":
                import msvcrt

                if handle.seek(0, 2) == 0:
                    handle.write(b"0")
                    handle.flush()
                handle.seek(0)
                try:
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                except OSError:
                    raise RecoveryError("Another runner holds this state database") from None
            else:
                import fcntl

                try:
                    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    raise RecoveryError("Another runner holds this state database") from None
            acquired = self._locked = True
            self._initialize()
            yield self
        finally:
            if acquired:
                self._locked = False
                if os.name == "nt":
                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    fcntl.flock(handle, fcntl.LOCK_UN)
            handle.close()

    def _guard(self):
        if not self._locked:
            raise RecoveryError("State access requires exclusive ownership")

    def start(self, task_id: str, identity: dict, initial: dict) -> dict:
        self._guard()
        with self._connect() as db:
            row = db.execute("SELECT * FROM tasks WHERE task_id=?", (task_id,)).fetchone()
            if row is None:
                db.execute(
                    "INSERT INTO tasks(task_id, identity, checkpoint) VALUES (?, ?, ?)",
                    (task_id, encode(identity), encode(initial)),
                )
                row = db.execute("SELECT * FROM tasks WHERE task_id=?", (task_id,)).fetchone()
            elif row["identity"] != encode(identity):
                raise RecoveryError("Task, corpus, code or model configuration changed; use a new task ID")
            return {
                "checkpoint": json.loads(row["checkpoint"]),
                "result": json.loads(row["result"]) if row["result"] else None,
                "status": row["status"],
            }

    def save(self, task_id: str, checkpoint: dict, *, status="running", result=None):
        self._guard()
        snapshot = encode(checkpoint)
        with self._connect() as db:
            revision = db.execute("SELECT revision FROM tasks WHERE task_id=?", (task_id,)).fetchone()[0] + 1
            db.execute(
                "INSERT INTO checkpoints VALUES (?, ?, ?, ?, ?)",
                (task_id, revision, checkpoint["phase"], checkpoint["step"], snapshot),
            )
            db.execute(
                "UPDATE tasks SET checkpoint=?, result=?, status=?, revision=? WHERE task_id=?",
                (snapshot, encode(result) if result is not None else None, status, revision, task_id),
            )

    def call(self, task_id: str, call_id: str, request_hash: str):
        self._guard()
        with self._connect() as db:
            row = db.execute(
                "SELECT * FROM calls WHERE task_id=? AND call_id=?", (task_id, call_id)
            ).fetchone()
            if row is None:
                db.execute(
                    "INSERT INTO calls(task_id,call_id,request_hash,status) VALUES (?,?,?,'pending')",
                    (task_id, call_id, request_hash),
                )
                return None
            if row["request_hash"] != request_hash:
                raise RecoveryError("Cached decision does not match checkpoint context")
            if row["status"] == "pending":
                raise UncertainRequest("Model outcome is uncertain; automatic resend is forbidden")
            return {
                "status": row["status"],
                "response": json.loads(row["response"]) if row["response"] else None,
                "error": json.loads(row["error"]) if row["error"] else None,
            }

    def finish_call(self, task_id: str, call_id: str, *, response=None, error=None):
        self._guard()
        with self._connect() as db:
            db.execute(
                "UPDATE calls SET status=?, response=?, error=? WHERE task_id=? AND call_id=?",
                (
                    "error" if error else "done",
                    encode(response) if response is not None else None,
                    encode(error) if error else None,
                    task_id,
                    call_id,
                ),
            )

    def begin_attempt(self, task_id: str, call_id: str, number: int, reserved_total: int, model: str):
        self._guard()
        event = {
            "attempt": number,
            "model": model,
            "status": "uncertain",
            "usage_status": "unknown",
            "reason": "Reserved before send; response not yet durably recorded",
        }
        with self._connect() as db:
            db.execute(
                "INSERT INTO attempts VALUES (?, ?, ?, ?, ?)",
                (task_id, number, call_id, reserved_total, encode(event)),
            )

    def finish_attempt(self, task_id: str, event: dict):
        self._guard()
        with self._connect() as db:
            db.execute(
                "UPDATE attempts SET event=? WHERE task_id=? AND number=?",
                (encode(event), task_id, event["attempt"]),
            )

    def ledger(self, task_id: str) -> dict:
        self._guard()
        with self._connect() as db:
            rows = db.execute("SELECT * FROM attempts WHERE task_id=? ORDER BY number", (task_id,)).fetchall()
            pending = [
                r[0]
                for r in db.execute(
                    "SELECT call_id FROM calls WHERE task_id=? AND status='pending'", (task_id,)
                )
            ]
        return {
            "events": [json.loads(r["event"]) for r in rows],
            "attempts": len(rows),
            "reserved_total": rows[-1]["reserved_total"] if rows else 0,
            "pending_calls": pending,
        }

    def inspect(self, task_id: str) -> dict:
        """Local metadata view. The database also contains sensitive evidence and outputs."""
        self._guard()
        with self._connect() as db:
            row = db.execute(
                "SELECT status,revision,checkpoint FROM tasks WHERE task_id=?", (task_id,)
            ).fetchone()
            if row is None:
                raise RecoveryError("Unknown task ID")
        checkpoint = json.loads(row["checkpoint"])
        return {
            "task_id": task_id,
            "status": row["status"],
            "revision": row["revision"],
            "phase": checkpoint["phase"],
            "step": checkpoint["step"],
            "ledger": self.ledger(task_id),
        }
