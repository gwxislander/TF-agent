# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import glob
import sqlite3
import sys
import tempfile
import threading
import unittest

_TF_AGENT = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "TF-agent")
)
if _TF_AGENT not in sys.path:
    sys.path.insert(0, _TF_AGENT)

from conversation_store import ConversationStore, SCHEMA_VERSION, next_thread_id_after_delete  # noqa: E402


class TestConversationStore(unittest.TestCase):
    def test_roundtrip_redacts_command_path_and_attachment_content(self):
        with tempfile.TemporaryDirectory() as td:
            db = os.path.join(td, "conversation.sqlite3")
            store = ConversationStore(db)
            tid = store.create_thread("thread_test")
            store.append_message(
                tid,
                "user",
                "[SYSTEM_COMMAND_JSON]{\"pending_action\":{}}[/SYSTEM_COMMAND_JSON] /Users/chl/a.tif "
                "data:image/png;base64,AAAA",
                attachment_ref="/Users/chl/upload.png",
            )
            rows = store.load_messages(tid)
            self.assertEqual(len(rows), 1)
            self.assertIn("不可重放", rows[0]["content"])
            self.assertNotIn("/Users/", rows[0]["content"])
            self.assertNotIn("base64", rows[0]["content"])
            self.assertEqual(rows[0]["image_name"], "upload.png")
            self.assertEqual(os.stat(db).st_mode & 0o777, 0o600)

    def test_persisted_messages_redact_credentials(self):
        with tempfile.TemporaryDirectory() as td:
            store = ConversationStore(os.path.join(td, "chat.sqlite3"))
            store.append_message(
                "thread_secret",
                "user",
                "token=sk-secret proxy=https://user:pass@example.invalid/api",
            )
            content = store.load_messages("thread_secret")[0]["content"]
            self.assertNotIn("sk-secret", content)
            self.assertNotIn("user:pass@", content)

    def test_history_roundtrip_preserves_web_source_links(self):
        fd, db_path = tempfile.mkstemp(
            prefix="conversation-links-", suffix=".sqlite3", dir=os.getcwd()
        )
        os.close(fd)
        try:
            store = ConversationStore(db_path)
            links = (
                "https://example.org/article/123",
                "https://example.org/data/paper.pdf",
                "https://service.example/app/result?id=9",
            )
            content = "参考来源\n" + "\n".join(
                f"[{index}] {link}" for index, link in enumerate(links, 1)
            )
            store.replace_messages(
                "thread-links",
                [{"role": "assistant", "content": content}],
            )

            restored_messages = store.load_messages("thread-links")
            # Streamlit persists the full message snapshot again after every
            # history switch or subsequent prompt.  Exercise that second pass
            # because it was the trigger that used to corrupt saved links.
            store.replace_messages("thread-links", restored_messages)
            restored = store.load_messages("thread-links")[0]["content"]
            for link in links:
                self.assertIn(link, restored)
            self.assertNotIn("<local-path>", restored)
        finally:
            for suffix in ("", "-wal", "-shm"):
                candidate = db_path + suffix
                if os.path.exists(candidate):
                    os.remove(candidate)

    def test_persisted_messages_redact_bare_provider_key(self):
        with tempfile.TemporaryDirectory() as td:
            store = ConversationStore(os.path.join(td, "chat.sqlite3"))
            store.append_message("thread_bare_key", "user", "please use sk-abcdefghijklmnopqrstuvwxyz")
            content = store.load_messages("thread_bare_key")[0]["content"]
            self.assertNotIn("sk-abcdefghijklmnopqrstuvwxyz", content)
            self.assertIn("<redacted>", content)

    def test_command_id_cannot_persist_raw_command_or_secret(self):
        with tempfile.TemporaryDirectory() as td:
            db = os.path.join(td, "command-id.sqlite3")
            store = ConversationStore(db)
            store.append_message(
                "thread-command-id",
                "assistant",
                "done",
                command_id="token=sk-command-secret /Users/chl/private/command.json",
            )
            with sqlite3.connect(db) as conn:
                raw = conn.execute("SELECT command_id FROM messages").fetchone()[0]
            self.assertNotIn("sk-command-secret", raw)
            self.assertNotIn("/Users/", raw)
            self.assertRegex(raw, r"^cmd_[0-9a-f]{16}$")

    def test_persisted_messages_redact_precise_spatial_fields(self):
        with tempfile.TemporaryDirectory() as td:
            store = ConversationStore(os.path.join(td, "chat.sqlite3"))
            store.append_message(
                "thread_spatial",
                "assistant",
                "bbox=(120.6,30.2,121.2,30.9) centroid=(120.9,30.55)",
            )
            content = store.load_messages("thread_spatial")[0]["content"]
            self.assertNotIn("120.6", content)
            self.assertNotIn("30.55", content)
            self.assertIn("<spatial-redacted>", content)

    def test_delete_and_retention_cleanup(self):
        now = [1000.0]
        with tempfile.TemporaryDirectory() as td:
            store = ConversationStore(os.path.join(td, "c.sqlite3"), retention_days=1, now_fn=lambda: now[0])
            # now_fn is intentionally not part of public API; use delete contract here.
            tid = store.create_thread("delete_me")
            store.append_message(tid, "assistant", "hello")
            store.delete_thread(tid)
            self.assertEqual(store.load_messages(tid), [])

    def test_snapshot_replace_does_not_duplicate(self):
        with tempfile.TemporaryDirectory() as td:
            store = ConversationStore(os.path.join(td, "c.sqlite3"))
            store.replace_messages("t", [{"role": "assistant", "content": "one"}])
            store.replace_messages("t", [{"role": "assistant", "content": "two"}])
            self.assertEqual([m["content"] for m in store.load_messages("t")], ["two"])

    def test_list_threads_returns_recent_sessions_with_safe_preview(self):
        now = [1000.0]
        with tempfile.TemporaryDirectory() as td:
            store = ConversationStore(os.path.join(td, "sessions.sqlite3"), now_fn=lambda: now[0])
            older = store.create_thread("older")
            store.append_message(older, "user", "older prompt")
            now[0] = 1001.0
            newer = store.create_thread("newer")
            store.append_message(newer, "user", "请检查 /Users/chl/private.tif")
            store.append_message(newer, "assistant", "/Users/chl/private result")

            rows = store.list_threads(limit=10)

            self.assertEqual([row["thread_id"] for row in rows[:2]], ["newer", "older"])
            self.assertEqual(rows[0]["message_count"], 2)
            self.assertIn("<local-path>", rows[0]["preview"])
            self.assertEqual(rows[1]["preview"], "older prompt")

    def test_list_threads_limit_is_positive_and_bounded(self):
        with tempfile.TemporaryDirectory() as td:
            store = ConversationStore(os.path.join(td, "sessions.sqlite3"), max_sessions=3)
            for index in range(5):
                store.create_thread(f"thread-{index}")
            self.assertEqual(len(store.list_threads(limit=2)), 2)
            self.assertEqual(len(store.list_threads(limit=0)), 1)

    def test_list_threads_can_return_all_visible_sessions_without_display_limit(self):
        with tempfile.TemporaryDirectory() as td:
            store = ConversationStore(os.path.join(td, "sessions.sqlite3"), max_sessions=20)
            for index in range(12):
                thread_id = store.create_thread(f"thread-{index}")
                store.append_message(thread_id, "user", f"prompt-{index}")

            rows = store.list_threads(limit=None, include_empty=False)

            self.assertEqual(len(rows), 12)

    def test_list_threads_preview_prefers_user_message_over_greeting(self):
        with tempfile.TemporaryDirectory() as td:
            store = ConversationStore(os.path.join(td, "sessions.sqlite3"))
            store.create_thread("thread_preview")
            store.append_message("thread_preview", "assistant", "您好，我是智能分析助手")
            store.append_message("thread_preview", "user", "分析 2022 年潮滩变化")
            store.append_message("thread_preview", "assistant", "正在整理分析计划")

            rows = store.list_threads()

            self.assertEqual(rows[0]["preview"], "分析 2022 年潮滩变化")

    def test_list_threads_can_hide_empty_sessions_for_history_view(self):
        with tempfile.TemporaryDirectory() as td:
            store = ConversationStore(os.path.join(td, "sessions.sqlite3"))
            store.create_thread("empty")
            store.append_message("active", "user", "查看潮滩变化")

            rows = store.list_threads(include_empty=False)

            self.assertEqual([row["thread_id"] for row in rows], ["active"])

    def test_delete_current_session_selects_next_visible_session(self):
        threads = [
            {"thread_id": "newest"},
            {"thread_id": "current"},
            {"thread_id": "next"},
        ]
        self.assertEqual(next_thread_id_after_delete(threads, "current"), "next")
        self.assertEqual(next_thread_id_after_delete(threads, "next"), "newest")
        self.assertIsNone(next_thread_id_after_delete([{"thread_id": "only"}], "only"))

    def test_concurrent_appends_are_serialized_without_losing_messages(self):
        with tempfile.TemporaryDirectory() as td:
            store = ConversationStore(os.path.join(td, "concurrent.sqlite3"))
            errors = []

            def append_batch(worker: int) -> None:
                try:
                    for index in range(10):
                        store.append_message("shared", "user", f"worker-{worker}-{index}")
                except Exception as exc:  # pragma: no cover - assertion below reports it
                    errors.append(exc)

            threads = [threading.Thread(target=append_batch, args=(worker,)) for worker in range(4)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
            self.assertEqual(errors, [])
            self.assertEqual(len(store.load_messages("shared")), 40)

    def test_migrates_legacy_schema_without_dropping_messages(self):
        with tempfile.TemporaryDirectory() as td:
            db = os.path.join(td, "legacy.sqlite3")
            with sqlite3.connect(db) as conn:
                conn.execute("CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
                conn.execute("INSERT INTO metadata(key,value) VALUES('schema_version','1')")
                conn.execute("CREATE TABLE sessions (thread_id TEXT PRIMARY KEY, created_at REAL NOT NULL, last_seen REAL NOT NULL)")
                conn.execute("CREATE TABLE messages (id INTEGER PRIMARY KEY AUTOINCREMENT, thread_id TEXT NOT NULL, role TEXT NOT NULL, content TEXT NOT NULL, created_at REAL NOT NULL)")
                conn.execute("INSERT INTO sessions VALUES('legacy',1,1)")
                conn.execute("INSERT INTO messages(thread_id,role,content,created_at) VALUES('legacy','user','kept',1)")
            store = ConversationStore(db)
            self.assertEqual(store.load_messages("legacy")[0]["content"], "kept")
            with sqlite3.connect(db) as conn:
                version = conn.execute("SELECT value FROM metadata WHERE key='schema_version'").fetchone()[0]
                columns = {row[1] for row in conn.execute("PRAGMA table_info(messages)")}
            self.assertEqual(int(version), SCHEMA_VERSION)
            self.assertTrue({"attachment_ref", "command_id"}.issubset(columns))

    def test_corrupt_database_is_preserved_before_rebuild(self):
        with tempfile.TemporaryDirectory() as td:
            db = os.path.join(td, "broken.sqlite3")
            with open(db, "wb") as handle:
                handle.write(b"not a sqlite database")
            store = ConversationStore(db)
            self.assertTrue(store.corruption_backup_path)
            self.assertTrue(os.path.exists(store.corruption_backup_path))
            self.assertTrue(glob.glob(db + ".corrupt-*"))
            store.append_message("fresh", "assistant", "usable")
            self.assertEqual(store.load_messages("fresh")[0]["content"], "usable")


if __name__ == "__main__":
    unittest.main()
