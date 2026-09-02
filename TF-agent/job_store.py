# -*- coding: utf-8 -*-
"""单机后台任务账本与重启 reconcile。

该模块只保存任务元数据，不保存 prompt、密钥、绝对路径或原始输入。SQLite
提供跨 Streamlit rerun/进程的原子状态迁移；执行器仍可继续使用本地线程。
"""
from __future__ import annotations

import json
import os
import re
import shutil
import sqlite3
import threading
import time
import uuid
from contextlib import closing
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Optional

from agent_context_policy import redact_spatial_metadata, sanitize_external_text

STATUSES = (
    "PENDING", "WAITING_CONFIRMATION", "QUEUED", "RUNNING", "SUCCEEDED",
    "FAILED", "BLOCKED", "CANCELLED", "INTERRUPTED", "WARNING",
)
TERMINAL_STATUSES = frozenset({"SUCCEEDED", "FAILED", "BLOCKED", "CANCELLED", "INTERRUPTED"})
ACTIVE_PLAN_STATUSES = frozenset({"PENDING", "WAITING_CONFIRMATION", "QUEUED", "RUNNING"})
_TRANSITIONS = {
    "PENDING": {"WAITING_CONFIRMATION", "QUEUED", "RUNNING", "BLOCKED", "CANCELLED", "FAILED", "INTERRUPTED"},
    "WAITING_CONFIRMATION": {"QUEUED", "CANCELLED", "BLOCKED", "INTERRUPTED"},
    "QUEUED": {"RUNNING", "CANCELLED", "BLOCKED", "FAILED", "INTERRUPTED"},
    "RUNNING": {"SUCCEEDED", "FAILED", "CANCELLED", "WARNING", "INTERRUPTED"},
    "WARNING": {"RUNNING", "SUCCEEDED", "FAILED", "CANCELLED", "INTERRUPTED"},
}

_SAFE_ID = re.compile(r"[^A-Za-z0-9_.:-]+")
_SENSITIVE = ("token", "secret", "password", "api_key", "authorization", "prompt", "path")
_SPATIAL_KEYS = ("bbox", "bounds", "centroid", "geometry", "map_center", "coordinates", "resolution", "transform", "crs")
_SQLITE_HEADER = b"SQLite format 3\x00"


def worker_success_is_committable(success: Any, stop_requested: Any) -> bool:
    """A user stop request always wins the worker's late success signal."""
    return bool(success) and not bool(stop_requested)


def _safe_text(value: Any, *, limit: int = 500) -> str:
    return redact_spatial_metadata(sanitize_external_text(value)).strip()[:limit]


def _safe_id(value: Any, fallback: str) -> str:
    text = _SAFE_ID.sub("_", str(value or "").strip())[:160]
    return text or fallback


def _safe_json(value: Any) -> str:
    if not isinstance(value, dict):
        return "{}"
    clean: Dict[str, Any] = {}
    for key, item in value.items():
        key_text = str(key)
        low = key_text.lower()
        if any(token in low for token in _SENSITIVE):
            continue
        if any(token in low for token in _SPATIAL_KEYS):
            clean[key_text[:80]] = "<spatial-redacted>"
            continue
        if isinstance(item, (str, int, float, bool)) or item is None:
            clean[key_text[:80]] = _safe_text(item, limit=240) if isinstance(item, str) else item
    return json.dumps(clean, ensure_ascii=False, separators=(",", ":"))


@dataclass(frozen=True)
class JobRecord:
    job_id: str
    task: str
    kind: str
    plan_id: Optional[str]
    status: str
    progress: int
    attempt: int
    started_at: Optional[str]
    updated_at: str
    artifact_ids: List[str]
    error_summary: Optional[str]
    metadata: Dict[str, Any]


class JobStore:
    """SQLite job ledger with atomic claim and corruption preservation."""

    def __init__(self, db_path: str, *, now_fn=None) -> None:
        self.db_path = os.path.abspath(os.path.expanduser(db_path))
        os.makedirs(os.path.dirname(self.db_path) or ".", exist_ok=True)
        self._now_fn = now_fn or (lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        self._lock = threading.RLock()
        self._ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=10, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout=10000")
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _ensure_schema(self) -> None:
        # SQLite may keep a ``-wal`` sidecar after a process crash.  If the
        # main file was replaced/truncated meanwhile, opening it first can
        # replay that stale WAL and make a corrupt ledger appear healthy.  A
        # valid SQLite database always carries this header in its main file;
        # quarantine invalid files before any connection can inspect them.
        if os.path.isfile(self.db_path) and not self._has_sqlite_header(self.db_path):
            self._preserve_corrupt_ledger()
        try:
            # sqlite3.Connection's context manager commits/rolls back but does
            # not close the handle.  ``closing`` is required here so the
            # corruption handler can move the database on Windows, where an
            # open handle makes ``shutil.move`` fail with WinError 32.
            with closing(self._connect()) as conn:
                self._create_schema(conn)
                self._validate_rows(conn)
        except (sqlite3.DatabaseError, ValueError, TypeError, json.JSONDecodeError):
            # Never delete a damaged ledger. Preserve it for diagnosis and start clean.
            self._preserve_corrupt_ledger()
            with closing(self._connect()) as conn:
                self._create_schema(conn)

    @staticmethod
    def _create_schema(conn: sqlite3.Connection) -> None:
        conn.execute("""CREATE TABLE IF NOT EXISTS jobs (
            job_id TEXT PRIMARY KEY, task TEXT NOT NULL, kind TEXT NOT NULL,
            plan_id TEXT, status TEXT NOT NULL, progress INTEGER NOT NULL,
            attempt INTEGER NOT NULL, started_at TEXT, updated_at TEXT NOT NULL,
            artifact_ids TEXT NOT NULL, error_summary TEXT, metadata TEXT NOT NULL
        )""")

    @staticmethod
    def _validate_rows(conn: sqlite3.Connection) -> None:
        """Reject a structurally valid SQLite file whose JSON rows are damaged."""
        rows = conn.execute(
            "SELECT job_id,status,progress,attempt,artifact_ids,metadata FROM jobs"
        ).fetchall()
        for row in rows:
            if not row["job_id"] or row["status"] not in STATUSES:
                raise ValueError("job ledger row has invalid identity or status")
            if not (0 <= int(row["progress"]) <= 100) or int(row["attempt"]) < 0:
                raise ValueError("job ledger row has invalid progress or attempt")
            artifacts = json.loads(row["artifact_ids"] or "[]")
            metadata = json.loads(row["metadata"] or "{}")
            if not isinstance(artifacts, list) or not isinstance(metadata, dict):
                raise ValueError("job ledger row JSON has invalid shape")

    @staticmethod
    def _has_sqlite_header(path: str) -> bool:
        try:
            with open(path, "rb") as handle:
                return handle.read(len(_SQLITE_HEADER)) == _SQLITE_HEADER
        except OSError:
            return True

    def _preserve_corrupt_ledger(self) -> Optional[str]:
        """Move a damaged main DB and its WAL sidecars to unique evidence files."""
        if not os.path.isfile(self.db_path):
            return None
        stamp = time.strftime("%Y%m%d%H%M%S", time.gmtime())
        backup = f"{self.db_path}.corrupt-{stamp}"
        suffix = 1
        while os.path.exists(backup):
            backup = f"{self.db_path}.corrupt-{stamp}-{suffix}"
            suffix += 1
        shutil.move(self.db_path, backup)
        for sidecar in ("-wal", "-shm"):
            side_path = f"{self.db_path}{sidecar}"
            if os.path.isfile(side_path):
                try:
                    shutil.move(side_path, f"{backup}{sidecar}")
                except OSError:
                    # The main ledger remains preserved even if a sidecar is
                    # concurrently removed by SQLite during shutdown.
                    pass
        return backup

    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> JobRecord:
        return JobRecord(
            job_id=row["job_id"], task=row["task"], kind=row["kind"], plan_id=row["plan_id"],
            status=row["status"], progress=int(row["progress"]), attempt=int(row["attempt"]),
            started_at=row["started_at"], updated_at=row["updated_at"],
            artifact_ids=list(json.loads(row["artifact_ids"] or "[]")),
            error_summary=row["error_summary"], metadata=dict(json.loads(row["metadata"] or "{}")),
        )

    def create(
        self, *, task: str, kind: str, plan_id: Optional[str] = None,
        job_id: Optional[str] = None, status: str = "QUEUED", metadata: Optional[Dict[str, Any]] = None,
    ) -> JobRecord:
        if status not in STATUSES:
            raise ValueError(f"invalid job status: {status}")
        jid = _safe_id(job_id, uuid.uuid4().hex)
        pid = _safe_id(plan_id, "") or None
        task_text = _safe_text(task, limit=160)
        kind_text = _safe_text(kind, limit=80)
        now = self._now_fn()
        with self._lock, self._connect() as conn:
            # A plan is the idempotency boundary, not the request UUID.  The
            # transaction prevents two Streamlit processes from both creating
            # a fresh job for the same still-active plan.
            conn.execute("BEGIN IMMEDIATE")
            try:
                if pid:
                    existing = conn.execute(
                        "SELECT * FROM jobs WHERE plan_id=? AND status IN (?,?,?,?) "
                        "ORDER BY updated_at DESC, job_id LIMIT 1",
                        (pid, *sorted(ACTIVE_PLAN_STATUSES)),
                    ).fetchone()
                    if existing is not None:
                        if existing["task"] != task_text or existing["kind"] != kind_text:
                            raise ValueError("plan_id 已绑定其他任务或 kind，拒绝复用")
                        conn.execute("COMMIT")
                        return self._row_to_record(existing)
                conn.execute(
                    "INSERT OR IGNORE INTO jobs(job_id,task,kind,plan_id,status,progress,attempt,started_at,updated_at,artifact_ids,error_summary,metadata) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                    (jid, task_text, kind_text, pid,
                     status, 0, 0, None, now, "[]", None, _safe_json(metadata)),
                )
                row = conn.execute("SELECT * FROM jobs WHERE job_id=?", (jid,)).fetchone()
                # ``job_id`` is the execution-request identity.  SQLite's
                # INSERT OR IGNORE keeps retries idempotent, but silently
                # returning an unrelated row would bind a new plan to an old
                # request.  Reject that identity collision while preserving
                # exact same-request retries.
                if row is not None and (
                    row["plan_id"] != pid
                    or row["task"] != task_text
                    or row["kind"] != kind_text
                ):
                    raise ValueError("job_id 已绑定其他任务或 plan_id，拒绝复用")
                conn.execute("COMMIT")
            except BaseException:
                conn.execute("ROLLBACK")
                raise
        return self._row_to_record(row)

    def get(self, job_id: str) -> Optional[JobRecord]:
        with self._lock, self._connect() as conn:
            row = conn.execute("SELECT * FROM jobs WHERE job_id=?", (str(job_id),)).fetchone()
        return self._row_to_record(row) if row else None

    def list(self, *, statuses: Optional[Iterable[str]] = None) -> List[JobRecord]:
        with self._lock, self._connect() as conn:
            rows = conn.execute("SELECT * FROM jobs ORDER BY updated_at, job_id").fetchall()
        records = [self._row_to_record(row) for row in rows]
        if statuses is not None:
            wanted = set(statuses)
            records = [record for record in records if record.status in wanted]
        return records

    def transition(
        self, job_id: str, status: str, *, progress: Optional[int] = None,
        artifacts: Optional[Iterable[str]] = None, error: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None, expected_status: Optional[str] = None,
    ) -> JobRecord:
        if status not in STATUSES:
            raise ValueError(f"invalid job status: {status}")
        progress_value = min(100, max(0, int(progress))) if progress is not None else None
        now = self._now_fn()
        artifacts_json = json.dumps([_safe_id(a, "") for a in (artifacts or []) if _safe_id(a, "")], ensure_ascii=False)
        with self._lock, self._connect() as conn:
            # ``isolation_level=None`` makes each statement autocommit.  A
            # read-then-write claim would therefore allow two processes to
            # observe QUEUED before either UPDATE commits.  Serialize the
            # state transition so expected_status is an actual atomic gate.
            conn.execute("BEGIN IMMEDIATE")
            try:
                row = conn.execute("SELECT * FROM jobs WHERE job_id=?", (str(job_id),)).fetchone()
                if row is None:
                    raise KeyError(job_id)
                if expected_status is not None and row["status"] != expected_status:
                    raise RuntimeError(f"job already claimed or moved: {row['status']}")
                current_status = row["status"]
                if status != current_status and status not in _TRANSITIONS.get(current_status, set()):
                    raise ValueError(f"invalid job transition: {current_status}->{status}")
                started = row["started_at"] or (now if status == "RUNNING" else None)
                conn.execute(
                    "UPDATE jobs SET status=?, progress=COALESCE(?,progress), attempt=attempt + CASE WHEN ?='RUNNING' AND ?='QUEUED' THEN 1 ELSE 0 END, started_at=?, updated_at=?, artifact_ids=CASE WHEN ?='[]' THEN artifact_ids ELSE ? END, error_summary=?, metadata=CASE WHEN ?='{}' THEN metadata ELSE ? END WHERE job_id=?",
                    (status, progress_value, status, expected_status, started, now, artifacts_json, artifacts_json,
                     _safe_text(error) if error else row["error_summary"], _safe_json(metadata), _safe_json(metadata), str(job_id)),
                )
                updated = conn.execute("SELECT * FROM jobs WHERE job_id=?", (str(job_id),)).fetchone()
                conn.execute("COMMIT")
            except BaseException:
                conn.execute("ROLLBACK")
                raise
        return self._row_to_record(updated)

    def claim(self, job_id: str) -> Optional[JobRecord]:
        """Atomically claim a queued job; a second executor gets None."""
        try:
            return self.transition(job_id, "RUNNING", progress=0, expected_status="QUEUED")
        except RuntimeError:
            return None

    def update_progress(
        self, job_id: str, progress: int, *, metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[JobRecord]:
        """持久化运行进度，不改变任务状态或 attempt 计数。"""
        value = min(100, max(0, int(progress)))
        now = self._now_fn()
        with self._lock, self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                row = conn.execute("SELECT * FROM jobs WHERE job_id=?", (str(job_id),)).fetchone()
                if row is None:
                    conn.execute("COMMIT")
                    return None
                if row["status"] not in {"QUEUED", "RUNNING", "WARNING"}:
                    conn.execute("COMMIT")
                    return self._row_to_record(row)
                clean_metadata = _safe_json(metadata)
                # Keep the status predicate in the write itself.  It protects
                # the ledger even if a trigger or a future storage adapter
                # changes the row after the initial read.
                conn.execute(
                    "UPDATE jobs SET progress=?, updated_at=?, metadata=CASE WHEN ?='{}' THEN metadata ELSE ? END WHERE job_id=? AND status IN ('QUEUED','RUNNING','WARNING')",
                    (value, now, clean_metadata, clean_metadata, str(job_id)),
                )
                updated = conn.execute("SELECT * FROM jobs WHERE job_id=?", (str(job_id),)).fetchone()
                conn.execute("COMMIT")
            except BaseException:
                conn.execute("ROLLBACK")
                raise
        return self._row_to_record(updated)

    def reconcile(self) -> List[JobRecord]:
        """Mark non-terminal in-flight jobs interrupted after process restart."""
        records = self.list(statuses=("QUEUED", "RUNNING"))
        out = []
        for record in records:
            out.append(self.transition(record.job_id, "INTERRUPTED", error="进程重启后任务未完成；等待用户确认恢复或重跑。"))
        return out


__all__ = [
    "JobRecord", "JobStore", "STATUSES", "TERMINAL_STATUSES",
    "ACTIVE_PLAN_STATUSES", "worker_success_is_committable",
]
