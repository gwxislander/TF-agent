# -*- coding: utf-8 -*-
"""受控本地会话存储：SQLite、短保留期、无命令重放。"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import sqlite3
import threading
import time
import uuid
from typing import Any, Dict, List, Optional

from agent_context_policy import redact_spatial_metadata, sanitize_external_text


SCHEMA_VERSION = 2
_COMMAND_RE = re.compile(r"\[SYSTEM_COMMAND_JSON\].*?\[/SYSTEM_COMMAND_JSON\]", re.I | re.S)
_SECRET_RE = re.compile(
    r"(?i)(?:api[_-]?key|token|secret|password|authorization)\s*[:=]\s*[^\s,;，；]+"
)
_URL_CREDENTIAL_RE = re.compile(r"(?i)(https?://)([^/@\s]+):([^/@\s]+)@")
_SAFE_COMMAND_ID_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,160}$")
_BARE_PROVIDER_KEY_RE = re.compile(r"(?i)^(?:sk|rk|pk|ghp|github_pat)[_-][A-Za-z0-9_-]{12,}$")


def ensure_thread_id(store: "ConversationStore", thread_id: Optional[str], *, create: bool = True) -> Optional[str]:
    """Return a usable thread id without creating an empty session by default.

    Clearing the final conversation leaves the UI with no current thread.  A
    new thread is created only at the point where the next message is about to
    be persisted, avoiding an empty history row on every rerun.
    """
    current = str(thread_id or "").strip()
    if current:
        return current
    if not create:
        return None
    return store.create_thread() if store is not None else None


def _safe_attachment_ref(value: Any) -> Optional[str]:
    """Persist only a harmless attachment label, never a pasted credential."""
    if value is None:
        return None
    raw = os.path.basename(str(value)).replace("\x00", "").strip()
    if not raw:
        return None
    stem, ext = os.path.splitext(raw)
    ext = re.sub(r"[^A-Za-z0-9.]", "", ext.lower())[:12]
    if _SECRET_RE.search(raw) or _BARE_PROVIDER_KEY_RE.fullmatch(stem):
        return f"attachment{ext or ''}"
    # Keep a display-friendly name while excluding control chars and extreme
    # lengths that could pollute the history projection.
    clean = re.sub(r"[\x00-\x1f\x7f]", "", raw)
    clean = re.sub(r"\s+", " ", clean)[:180].strip()
    return clean or f"attachment{ext or ''}"


def next_thread_id_after_delete(threads: List[Dict[str, Any]], deleted_thread_id: str) -> Optional[str]:
    """Return the next visible history item after deleting the current one.

    The history projection is newest-first.  The row immediately below the
    deleted item is selected first; deleting the final row wraps to the first
    remaining session.  No replacement session is created here.
    """
    deleted_id = str(deleted_thread_id or "")
    ids = []
    for thread in threads:
        if not isinstance(thread, dict):
            continue
        thread_id = str(thread.get("thread_id") or "")
        if thread_id and thread_id not in ids:
            ids.append(thread_id)
    if not ids:
        return None
    try:
        start = ids.index(deleted_id)
    except ValueError:
        return next((thread_id for thread_id in ids if thread_id != deleted_id), None)
    for offset in range(1, len(ids)):
        candidate = ids[(start + offset) % len(ids)]
        if candidate != deleted_id:
            return candidate
    return None


def _safe_content(content: Any) -> str:
    text = str(content or "")
    text = _COMMAND_RE.sub("[系统命令已执行，历史记录不可重放]", text)
    text = _SECRET_RE.sub("<redacted>", text)
    text = _URL_CREDENTIAL_RE.sub(r"\1<redacted>@", text)
    # Also catch provider key prefixes when users paste a bare key without a
    # field name (for example ``sk-...``).
    # Absolute paths are handled by the shared sanitizer, which preserves
    # ordinary http/https/ftp links while redacting local filesystem paths.
    text = sanitize_external_text(text)
    # AOI/map coordinates are not needed for local history display and must not
    # re-enter a later external-model context through persisted messages.
    text = redact_spatial_metadata(text)
    # 不保存 base64/data URL；附件只保留受控引用字段。
    text = re.sub(r"data:[^;\s]+;base64,[A-Za-z0-9+/=]+", "<attachment-redacted>", text)
    return text[:100_000]


def _safe_command_id(value: Any) -> Optional[str]:
    """Persist only an opaque command identifier, never command text."""
    text = str(value or "").strip()
    if not text:
        return None
    if _SAFE_COMMAND_ID_RE.fullmatch(text):
        return text
    digest = hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()[:16]
    return f"cmd_{digest}"


class ConversationStore:
    def __init__(self, db_path: str, *, retention_days: int = 30, max_sessions: int = 100, now_fn=None):
        self.db_path = os.path.abspath(os.path.expanduser(db_path))
        self.retention_days = max(1, int(retention_days))
        self.max_sessions = max(1, int(max_sessions))
        self._now = now_fn or time.time
        self._lock = threading.RLock()
        self.corruption_backup_path: Optional[str] = None
        os.makedirs(os.path.dirname(self.db_path) or ".", exist_ok=True)
        if not os.path.exists(self.db_path):
            open(self.db_path, "a").close()
        try:
            os.chmod(self.db_path, 0o600)
        except OSError:
            pass
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=10.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=10000")
        return conn

    def _init_db(self) -> None:
        try:
            with self._lock, self._connect() as conn:
                self._migrate(conn)
        except sqlite3.DatabaseError:
            # 损坏的 SQLite 不应覆盖原始证据；保留数据库及 WAL/SHM 旁车文件后重建。
            self.corruption_backup_path = self._preserve_corrupt_database()
            with self._lock, self._connect() as conn:
                self._migrate(conn)

    @staticmethod
    def _add_missing_columns(conn: sqlite3.Connection, table: str, columns: Dict[str, str]) -> None:
        existing = {row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
        for name, definition in columns.items():
            if name not in existing:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {definition}")

    def _migrate(self, conn: sqlite3.Connection) -> None:
        """Create or migrate the local schema without dropping user history."""
        conn.execute("CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        conn.execute("""CREATE TABLE IF NOT EXISTS sessions (
            thread_id TEXT PRIMARY KEY,
            created_at REAL NOT NULL,
            last_seen REAL NOT NULL
        )""")
        conn.execute("""CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            thread_id TEXT NOT NULL REFERENCES sessions(thread_id) ON DELETE CASCADE,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at REAL NOT NULL,
            attachment_ref TEXT,
            command_id TEXT
        )""")
        # v1 数据库可能已创建但缺少 v2 引入的可选字段；只补列，不重写历史消息。
        self._add_missing_columns(conn, "sessions", {
            "created_at": "REAL NOT NULL DEFAULT 0",
            "last_seen": "REAL NOT NULL DEFAULT 0",
        })
        self._add_missing_columns(conn, "messages", {
            "attachment_ref": "TEXT",
            "command_id": "TEXT",
        })
        conn.execute("CREATE INDEX IF NOT EXISTS idx_messages_thread ON messages(thread_id, id)")
        row = conn.execute("SELECT value FROM metadata WHERE key='schema_version'").fetchone()
        try:
            previous = int(row[0]) if row else 0
        except (TypeError, ValueError):
            previous = 0
        if previous > SCHEMA_VERSION:
            raise RuntimeError(f"unsupported conversation schema version: {previous}")
        conn.execute("INSERT OR REPLACE INTO metadata(key,value) VALUES('schema_version',?)", (str(SCHEMA_VERSION),))
        conn.execute("INSERT OR REPLACE INTO metadata(key,value) VALUES('schema_migrated_at',?)", (str(int(time.time())),))
        conn.commit()

    def _preserve_corrupt_database(self) -> Optional[str]:
        if not os.path.exists(self.db_path):
            return None
        stamp = time.strftime("%Y%m%d%H%M%S", time.gmtime())
        backup = f"{self.db_path}.corrupt-{stamp}"
        suffix = 1
        while os.path.exists(backup):
            backup = f"{self.db_path}.corrupt-{stamp}-{suffix}"
            suffix += 1
        shutil.move(self.db_path, backup)
        for sidecar in (f"{self.db_path}-wal", f"{self.db_path}-shm"):
            if os.path.exists(sidecar):
                shutil.move(sidecar, f"{backup}{sidecar[len(self.db_path):]}")
        return backup

    def create_thread(self, thread_id: Optional[str] = None) -> str:
        tid = thread_id or f"thread_{uuid.uuid4().hex}"
        now = float(self._now())
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO sessions(thread_id,created_at,last_seen) VALUES(?,?,?)",
                (tid, now, now),
            )
            conn.execute("UPDATE sessions SET last_seen=? WHERE thread_id=?", (now, tid))
            conn.commit()
        return tid

    def append_message(
        self,
        thread_id: str,
        role: str,
        content: Any,
        *,
        attachment_ref: Optional[str] = None,
        command_id: Optional[str] = None,
    ) -> None:
        tid = self.create_thread(thread_id)
        safe_ref = _safe_attachment_ref(attachment_ref)
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT INTO messages(thread_id,role,content,created_at,attachment_ref,command_id) VALUES(?,?,?,?,?,?)",
                (tid, str(role or "assistant")[:32], _safe_content(content), float(self._now()), safe_ref,
                 _safe_command_id(command_id)),
            )
            conn.execute("UPDATE sessions SET last_seen=? WHERE thread_id=?", (float(self._now()), tid))
            conn.commit()

    def replace_messages(self, thread_id: str, messages: List[Dict[str, Any]]) -> None:
        tid = self.create_thread(thread_id)
        with self._lock, self._connect() as conn:
            conn.execute("DELETE FROM messages WHERE thread_id=?", (tid,))
            for message in messages[-200:]:
                if not isinstance(message, dict):
                    continue
                conn.execute(
                    "INSERT INTO messages(thread_id,role,content,created_at,attachment_ref,command_id) VALUES(?,?,?,?,?,?)",
                    (
                        tid,
                        str(message.get("role") or "assistant")[:32],
                        _safe_content(message.get("content")),
                        float(self._now()),
                        _safe_attachment_ref(message.get("image_name")),
                        _safe_command_id(message.get("command_id")),
                    ),
                )
            conn.execute("UPDATE sessions SET last_seen=? WHERE thread_id=?", (float(self._now()), tid))
            conn.commit()

    def load_messages(self, thread_id: str) -> List[Dict[str, Any]]:
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                "SELECT role,content,attachment_ref FROM messages WHERE thread_id=? ORDER BY id",
                (thread_id,),
            ).fetchall()
        out = []
        for row in rows:
            item = {"role": row["role"], "content": row["content"]}
            if row["attachment_ref"]:
                item["image_name"] = row["attachment_ref"]
            out.append(item)
        return out

    def list_threads(self, limit: Optional[int] = 50, *, include_empty: bool = True) -> List[Dict[str, Any]]:
        """Return recent sessions with an already-redacted preview.

        ``limit=None`` intentionally disables the query-level count limit for
        navigation views.  The UI can then keep every visible session while
        constraining only the viewport with an internal scrollbar.

        The session list is a navigation projection only: it exposes no raw
        message payload, attachment bytes, or execution command text.  The
        preview is read from the sanitized ``messages.content`` column and is
        truncated again for the UI boundary.
        """
        if limit is None:
            safe_limit = None
        else:
            try:
                safe_limit = int(limit)
            except (TypeError, ValueError):
                safe_limit = 50
            safe_limit = max(1, min(safe_limit, 100))
        _nonempty_filter = "" if include_empty else (
            "HAVING SUM(CASE WHEN m.role = 'user' THEN 1 ELSE 0 END) > 0"
        )
        _limit_clause = "" if safe_limit is None else "LIMIT ?"
        _query_params = () if safe_limit is None else (safe_limit,)
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT
                    s.thread_id,
                    s.created_at,
                    s.last_seen,
                    COUNT(m.id) AS message_count,
                    COALESCE((
                        SELECT m2.content
                        FROM messages AS m2
                        WHERE m2.thread_id = s.thread_id AND m2.role = 'user'
                        ORDER BY m2.id DESC
                        LIMIT 1
                    ), '') AS preview
                FROM sessions AS s
                LEFT JOIN messages AS m ON m.thread_id = s.thread_id
                GROUP BY s.thread_id, s.created_at, s.last_seen
                {_nonempty_filter}
                ORDER BY s.last_seen DESC, s.thread_id DESC
                {_limit_clause}
                """,
                _query_params,
            ).fetchall()
        out = []
        for row in rows:
            out.append(
                {
                    "thread_id": str(row["thread_id"]),
                    "created_at": float(row["created_at"] or 0),
                    "last_seen": float(row["last_seen"] or 0),
                    "message_count": int(row["message_count"] or 0),
                    # Re-sanitize on read for legacy databases created before
                    # the current persistence boundary was introduced.
                    "preview": _safe_content(row["preview"])[:240],
                }
            )
        return out

    def delete_thread(self, thread_id: str) -> None:
        with self._lock, self._connect() as conn:
            conn.execute("DELETE FROM messages WHERE thread_id=?", (thread_id,))
            conn.execute("DELETE FROM sessions WHERE thread_id=?", (thread_id,))
            conn.commit()

    def cleanup(self) -> Dict[str, int]:
        cutoff = float(self._now()) - self.retention_days * 86400
        with self._lock, self._connect() as conn:
            old = conn.execute("SELECT thread_id FROM sessions WHERE last_seen < ?", (cutoff,)).fetchall()
            for row in old:
                conn.execute("DELETE FROM sessions WHERE thread_id=?", (row["thread_id"],))
            extra = conn.execute(
                "SELECT thread_id FROM sessions ORDER BY last_seen DESC LIMIT -1 OFFSET ?",
                (self.max_sessions,),
            ).fetchall()
            for row in extra:
                conn.execute("DELETE FROM sessions WHERE thread_id=?", (row["thread_id"],))
            conn.commit()
        return {"expired": len(old), "over_limit": len(extra)}


__all__ = ["ConversationStore", "SCHEMA_VERSION", "ensure_thread_id"]
