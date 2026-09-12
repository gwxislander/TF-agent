# -*- coding: utf-8 -*-
"""功能审查缺陷的回归测试。

这些测试只使用临时账本和 override 执行器，不触发 GEE、GPU、影像下载或
真实 PDF 生成。
"""
from __future__ import annotations

import os
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch
from urllib.error import HTTPError

_TF_AGENT = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "TF-agent")
)
if _TF_AGENT not in sys.path:
    sys.path.insert(0, _TF_AGENT)

import workflow_orchestrator as wo  # noqa: E402
from execution_request import attach_execution_request  # noqa: E402
from conversation_store import ConversationStore, ensure_thread_id  # noqa: E402
from context_budget import bound_messages  # noqa: E402
import globe_server  # noqa: E402


def _aoi() -> dict:
    return {"aoi_id": "audit-aoi", "label": "测试湾", "valid": True, "geometry": {}}


def _build(**kwargs):
    values = {
        "aoi": _aoi(),
        "target_year": 2024,
        "baseline_year": None,
        "region": "audit",
    }
    values.update(kwargs)
    return wo.build_analysis_workflow(**values)


def _ok(step, workflow, exec_ctx):
    return {
        "success": True,
        "status": wo.STEP_SUCCEEDED,
        "outputs": {},
        "assets": [{"asset_id": f"asset_{step['step_id']}", "asset_type": "audit"}],
        "metrics": {},
        "warnings": [],
        "error": None,
        "plan_id": f"plan_{step['step_id']}",
    }


def _ctx(calls=None, gee=None):
    def track(step, workflow, exec_ctx):
        if calls is not None:
            calls.append(step["step_id"])
        return _ok(step, workflow, exec_ctx)

    return {
        "gee_executor": gee or track,
        "inference_executor": track,
        "e1_executor": track,
        "m5_executor": track,
        "report_executor": track,
    }


class TestFunctionalAuditFixes(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.old_ledger = wo.WORKFLOW_LEDGER_PATH
        self.old_data_dir = wo._DEFAULT_DATA_DIR
        wo.WORKFLOW_LEDGER_PATH = os.path.join(self.tmp.name, "workflow_ledger.json")
        wo._DEFAULT_DATA_DIR = self.tmp.name

    def tearDown(self):
        wo.WORKFLOW_LEDGER_PATH = self.old_ledger
        wo._DEFAULT_DATA_DIR = self.old_data_dir
        self.tmp.cleanup()

    @staticmethod
    def confirm(wf):
        state = {wo.STATE_WORKFLOW_PENDING_PLAN: wf}
        ok, error = wo.confirm_workflow(state, wf["workflow_id"])
        assert ok, error
        return state

    def test_unconfirmed_workflow_cannot_execute(self):
        calls = []
        result = wo.run_analysis_workflow(
            _build(), exec_ctx=_ctx(calls), push_log=lambda _msg: None
        )
        self.assertEqual(result["status"], wo.WF_PENDING)
        self.assertEqual(calls, [])
        self.assertTrue(any("确认" in error for error in result["errors"]))

    def test_cancelled_workflow_cannot_restart(self):
        calls = []
        wf = _build()
        state = self.confirm(wf)
        ok, error = wo.cancel_workflow(state)
        self.assertTrue(ok, error)
        result = wo.run_analysis_workflow(
            wf, exec_ctx=_ctx(calls), push_log=lambda _msg: None
        )
        self.assertEqual(result["status"], wo.WF_CANCELLED)
        self.assertEqual(calls, [])

    def test_changed_confirmed_parameters_pause_before_first_step(self):
        calls = []
        wf = _build()
        self.confirm(wf)
        wf["context"]["target_year"] = 2025
        result = wo.run_analysis_workflow(
            wf, exec_ctx=_ctx(calls), push_log=lambda _msg: None
        )
        self.assertEqual(result["status"], wo.WF_PAUSED)
        self.assertEqual(calls, [])
        self.assertIsNotNone(wf.get("approved_spec_hash"))

    def test_cancel_after_step_returns_cannot_commit_late_success(self):
        stop_event = threading.Event()
        calls = []

        def gee_then_stop(step, workflow, exec_ctx):
            calls.append(step["step_id"])
            stop_event.set()
            return _ok(step, workflow, exec_ctx)

        wf = _build()
        self.confirm(wf)
        result = wo.run_analysis_workflow(
            wf,
            exec_ctx=_ctx(calls, gee=gee_then_stop),
            push_log=lambda _msg: None,
            stop_event=stop_event,
        )
        first = wf["steps"][0]
        self.assertEqual(result["status"], wo.WF_CANCELLED)
        self.assertEqual(first["status"], wo.STEP_CANCELLED)
        self.assertIsNone(first.get("asset_id"))

    def test_running_heavy_step_blocks_another_heavy_step(self):
        heavy_tool = next(iter(wo.HEAVY_TOOLS))
        workflow = {
            "context": {},
            "steps": [
                {
                    "step_id": "h1",
                    "tool": heavy_tool,
                    "status": wo.STEP_RUNNING,
                    "depends_on": [],
                    "optional_depends_on": [],
                    "required": True,
                    "condition": None,
                },
                {
                    "step_id": "h2",
                    "tool": heavy_tool,
                    "status": wo.STEP_PENDING,
                    "depends_on": [],
                    "optional_depends_on": [],
                    "required": True,
                    "condition": None,
                },
            ],
        }
        self.assertEqual(wo.find_ready_steps(workflow), [])

    def test_need_report_false_skips_report_step(self):
        wf = _build(user_intent={"need_e1": False, "need_m5": False, "need_report": False})
        by_id = {step["step_id"]: step for step in wf["steps"]}
        by_id["gee_download"]["status"] = wo.STEP_SUCCEEDED
        by_id["local_inference"]["status"] = wo.STEP_SUCCEEDED
        wo.find_ready_steps(wf)
        self.assertEqual(by_id["pdf_report"]["status"], wo.STEP_SKIPPED)
        self.assertFalse(by_id["pdf_report"]["required"])

    def test_workflow_execution_request_uses_workflow_id_as_idempotency_key(self):
        pending = {
            "task": "audit",
            "mode": "workflow",
            "workflow_id": "wf_same",
            "workflow_plan": {"workflow_id": "wf_same"},
        }
        request = attach_execution_request(pending, confirmation_source="ui")["execution_request"]
        self.assertEqual(request["plan_id"], "wf_same")

    def test_ledger_roundtrip_preserves_recovery_contract_and_records_lineage(self):
        wf = _build(user_intent={"need_e1": True, "need_m5": False, "need_report": True})
        self.confirm(wf)
        result = wo.run_analysis_workflow(
            wf, exec_ctx=_ctx(), push_log=lambda _msg: None
        )
        self.assertIn(result["status"], {wo.WF_SUCCEEDED, wo.WF_COMPLETED_WITH_WARNINGS})
        restored = wo.load_workflow(wf["workflow_id"])
        self.assertEqual(restored["context"]["target_year"], 2024)
        self.assertEqual(restored["intent"]["need_report"], True)
        self.assertEqual(restored["approved_spec_hash"], wf["approved_spec_hash"])
        restored_by_id = {step["step_id"]: step for step in restored["steps"]}
        self.assertEqual(restored_by_id["local_inference"]["depends_on"], ["gee_download"])
        self.assertTrue(restored_by_id["local_inference"]["required"])
        self.assertTrue(wo.get_asset_lineage("asset_local_inference")["found"])

    def test_deleted_last_session_is_recreated_only_when_next_message_arrives(self):
        store = ConversationStore(os.path.join(self.tmp.name, "conversations.sqlite3"))
        first = store.create_thread()
        store.append_message(first, "user", "旧消息")
        store.delete_thread(first)
        self.assertEqual(store.list_threads(include_empty=False), [])
        self.assertIsNone(ensure_thread_id(store, None, create=False))
        recreated = ensure_thread_id(store, None, create=True)
        self.assertTrue(recreated)
        store.append_message(recreated, "user", "新消息")
        self.assertEqual(store.list_threads(include_empty=False)[0]["thread_id"], recreated)

    def test_attachment_reference_redacts_secret_like_filename(self):
        store = ConversationStore(os.path.join(self.tmp.name, "attachments.sqlite3"))
        tid = store.create_thread()
        store.append_message(tid, "user", "上传", attachment_ref="api_key=secret-value.tif")
        loaded = store.load_messages(tid)
        self.assertEqual(loaded[0]["image_name"], "attachment.tif")

    def test_context_message_budget_includes_system_head(self):
        rows = [{"role": "system", "content": "system"}]
        rows.extend({"role": "user", "content": str(i)} for i in range(5))
        self.assertLessEqual(len(bound_messages(rows, max_messages=3)), 3)

    def test_aoi_queue_isolated_by_map_channel(self):
        globe_server.reset_map_protocol_state()
        globe_server.push_aoi_message({"kind": "selected", "geometry": {"id": "a"}}, channel_id="chan-a")
        self.assertEqual(globe_server.take_aoi_pending(channel_id="chan-b")["messages"], [])
        self.assertEqual(
            globe_server.take_aoi_pending(channel_id="chan-a")["messages"][0]["geometry"]["id"],
            "a",
        )

    def test_overlay_http_error_is_returned_without_urllib_scope_error(self):
        """瓦片上游返回 HTTP 错误时，代理应返回该状态码而不是作用域异常。"""
        token = "a" * 16
        handler = object.__new__(globe_server._GlobeHandler)
        handler.path = f"/overlay/{token}/1/2/3.png"
        sent = []
        handler._send = lambda *args, **kwargs: sent.append((args, kwargs))
        with globe_server._lock:
            globe_server._tile_templates[token] = "https://tiles.invalid/{z}/{x}/{y}.png"
        try:
            error = HTTPError("https://tiles.invalid/1/2/3.png", 503, "upstream unavailable", {}, None)
            with patch.object(globe_server, "_fetch_upstream_tile", side_effect=error):
                globe_server._GlobeHandler.do_GET(handler)
        finally:
            with globe_server._lock:
                globe_server._tile_templates.pop(token, None)

        self.assertEqual(sent[0][0][0], 503)


if __name__ == "__main__":
    unittest.main()
