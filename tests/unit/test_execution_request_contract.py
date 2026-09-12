# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

_TF_AGENT = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "TF-agent")
)
if _TF_AGENT not in sys.path:
    sys.path.insert(0, _TF_AGENT)

from execution_request import (  # noqa: E402
    EXECUTION_ENTRYPOINTS,
    ExecutionRequest,
    attach_execution_request,
    from_pending_task,
)
from agent_command_bridge import build_pending_task, init_ui_session_defaults  # noqa: E402


class TestExecutionRequestContract(unittest.TestCase):
    def test_mode_mapping_has_single_production_entrypoint(self):
        self.assertEqual(EXECUTION_ENTRYPOINTS["dl"], "inference_agent_loop")
        self.assertEqual(EXECUTION_ENTRYPOINTS["index"], "index_agent_loop")
        self.assertEqual(EXECUTION_ENTRYPOINTS["workflow"], "workflow_orchestrator")
        self.assertNotEqual(EXECUTION_ENTRYPOINTS["dl"], "run_pipeline_sync")

    def test_agent_and_ui_snapshots_differ_only_by_confirmation_source(self):
        agent_pending = {
            "task": "p1",
            "mode": "dl",
            "prob": 0.05,
            "cnt": 2,
            "points_shp": None,
            "force_rerun": False,
        }
        ui = attach_execution_request(agent_pending, confirmation_source="ui")
        agent = attach_execution_request(agent_pending, confirmation_source="agent")
        self.assertEqual(ui["execution_request"]["task"], agent["execution_request"]["task"])
        self.assertEqual(ui["execution_request"]["mode"], agent["execution_request"]["mode"])
        self.assertEqual(ui["execution_request"]["entrypoint"], "inference_agent_loop")
        self.assertEqual(ui["execution_request"]["params"], agent["execution_request"]["params"])
        self.assertEqual(ui["execution_request"]["confirmation_source"], "ui")

    def test_attaching_contract_again_preserves_request_identity_for_reruns(self):
        pending = {
            "task": "p1",
            "mode": "dl",
            "plan_id": "plan-rerun-safe",
            "prob": 0.05,
            "cnt": 2,
        }
        first = attach_execution_request(pending, confirmation_source="ui")
        second = attach_execution_request(first, confirmation_source="ui")

        self.assertEqual(
            first["execution_request"]["request_id"],
            second["execution_request"]["request_id"],
        )

    def test_changed_parameters_get_a_new_request_identity(self):
        first = attach_execution_request(
            {"task": "p1", "mode": "dl", "plan_id": "plan-params", "prob": 0.05},
            confirmation_source="ui",
        )
        changed = dict(first)
        changed["prob"] = 0.10
        second = attach_execution_request(changed, confirmation_source="ui")

        self.assertNotEqual(
            first["execution_request"]["request_id"],
            second["execution_request"]["request_id"],
        )

    def test_legacy_sync_mode_is_explicitly_named(self):
        source = (Path(__file__).parents[2] / "TF-agent" / "app.py").read_text(encoding="utf-8")
        self.assertIn('elif mode == "legacy_dl":', source)
        self.assertIn("未启动旧兼容入口", source)

    def test_legacy_m4_worker_uses_shared_gee_adapter(self):
        source = (Path(__file__).parents[2] / "TF-agent" / "app.py").read_text(encoding="utf-8")
        start = source.index("def _pipeline_worker_entry")
        end = source.index("def _workflow_worker_entry", start)
        worker = source[start:end]
        self.assertIn("build_legacy_m4_plan", worker)
        self.assertIn("_gee_worker_entry", worker)
        self.assertNotIn("ok = run_m4_download_sync(ctx, shared, stop_event)", worker)

    def test_independent_postflight_workers_gate_success_on_registration(self):
        source = (Path(__file__).parents[2] / "TF-agent" / "app.py").read_text(encoding="utf-8")
        for worker_name, register_name in (
            ("run_m5_sync", "register_m5_asset"),
            ("run_e1_sync", "register_e1_asset"),
        ):
            start = source.index(f"def {worker_name}")
            end = source.find("\ndef ", start + 5)
            worker = source[start:end if end >= 0 else len(source)]
            self.assertIn(f"asset_id = {register_name}", worker)
            self.assertIn("if not asset_id", worker)

    def test_independent_asset_registerers_reject_empty_reports(self):
        source = (Path(__file__).parents[2] / "TF-agent" / "app.py").read_text(encoding="utf-8")
        for function_name in ("register_m5_asset", "register_e1_asset"):
            start = source.index(f"def {function_name}")
            end = source.find("\ndef ", start + 5)
            function = source[start:end if end >= 0 else len(source)]
            self.assertIn("_nonempty_file", function)

    def test_legacy_optional_postflight_phases_verify_before_success_timeline(self):
        """兼容主流程的可选 M5/E1 也不能把未校验报告写成成功。"""
        source = (Path(__file__).parents[2] / "TF-agent" / "app.py").read_text(encoding="utf-8")
        m5_start = source.index("def _run_m5_phase")
        m5_end = source.index("def run_m5_sync", m5_start)
        m5_phase = source[m5_start:m5_end]
        e1_start = source.index("def _run_e1_phase")
        e1_end = source.index("def run_pipeline_sync", e1_start)
        e1_phase = source[e1_start:e1_end]
        self.assertIn("verify_m5_outputs", m5_phase)
        self.assertIn("register_m5_asset", m5_phase)
        self.assertIn("verify_e1_outputs", e1_phase)
        self.assertIn("register_e1_asset", e1_phase)

        generic_start = source.index("if success and not _inference_handled")
        generic_end = source.index("_job_status =", generic_start)
        generic_finalize = source[generic_start:generic_end]
        self.assertIn("m5_verification", generic_finalize)
        self.assertIn("e1_verification", generic_finalize)
        self.assertIn('"变化分析校验通过" if _m5_ok', generic_finalize)
        self.assertIn('"精度评价校验通过" if _e1_ok', generic_finalize)
        self.assertIn("_optional_postflight_warning", generic_finalize)

    def test_background_workers_honor_stop_before_verify_and_register(self):
        """停止信号在引擎返回后到达时，后台 worker 也不得继续登记成果。"""
        source = (Path(__file__).parents[2] / "TF-agent" / "app.py").read_text(encoding="utf-8")
        for worker_name, verify_marker, register_marker in (
            ("_inference_worker_entry", "verify_inference_outputs", "register_inference_asset"),
            ("_gee_worker_entry", "verify_gee_outputs", "register_gee_dataset_asset"),
            ("run_m5_sync", "verify_m5_outputs", "register_m5_asset"),
            ("run_e1_sync", "verify_e1_outputs", "register_e1_asset"),
        ):
            start = source.index(f"def {worker_name}")
            end = source.find("\ndef ", start + 5)
            block = source[start:end if end >= 0 else len(source)]
            verify_idx = block.index(verify_marker)
            register_idx = block.index(register_marker)
            prefix = block[:verify_idx]
            self.assertIn("stop_event.is_set()", prefix, worker_name)
            self.assertIn("stop_event.is_set()", block[verify_idx:register_idx], worker_name)

    def test_modern_inference_carries_and_runs_optional_e1_m5(self):
        """现代推理入口不能丢弃侧栏的 E1/M5 设置。"""
        source = (Path(__file__).parents[2] / "TF-agent" / "app.py").read_text(encoding="utf-8")
        start = source.index("def _inference_worker_entry")
        end = source.index("\ndef _gee_worker_entry", start)
        worker = source[start:end]
        for marker in ("_run_m5_phase(", "_run_e1_phase(", "postflight_ctx"):
            self.assertIn(marker, worker)

        modern_start = source.index("# 本地潮滩推理可信执行闭环")
        modern_end = source.index("# GEE 影像下载可信执行闭环", modern_start)
        modern_ctx = source[modern_start:modern_end]
        for key in (
            '"task_aoi_shp"', '"m5_enabled"', '"m5_baseline_shp"',
            '"e1_enabled"', '"e1_data_root"', '"e1_reference"',
            '"e1_compare_sources"', '"e1_export_maps"', '"e1_export_heatmap"',
        ):
            self.assertIn(key, modern_ctx)

    def test_stop_button_persists_cancel_request_until_worker_finishes(self):
        """UI 中断必须立即写入账本并显示等待安全退出，而不是只弹一次 toast。"""
        source = (Path(__file__).parents[2] / "TF-agent" / "app.py").read_text(encoding="utf-8")
        start = source.index('stop_btn = st.button(')
        end = source.index('if tune_btn', start)
        stop_block = source[start:end]
        self.assertIn('st.session_state.stop_requested = True', stop_block)
        self.assertIn('_job_transition(', stop_block)
        self.assertIn('"CANCELLED"', stop_block)
        self.assertIn('正在等待当前阶段安全退出', source)

    def test_late_finalizer_does_not_overwrite_reconciled_interrupted_job(self):
        """进程重启后的 INTERRUPTED 账本不能被旧 worker 改写成 FAILED。"""
        source = (Path(__file__).parents[2] / "TF-agent" / "app.py").read_text(encoding="utf-8")
        start = source.index("def _job_transition")
        end = source.index("\ndef _job_progress_update", start)
        transition = source[start:end]
        self.assertIn("TERMINAL_STATUSES", transition)
        self.assertIn("current.status in TERMINAL_STATUSES", transition)

    def test_e1_engine_receives_cooperative_stop_callback(self):
        """E1 分块比较需把停止回调传入引擎，避免只能等整轮评价完成。"""
        source = (Path(__file__).parents[2] / "TF-agent" / "e1_engine.py").read_text(encoding="utf-8")
        call = source[source.index("result = e1.run_pixel_comparison("):]
        self.assertIn("stop_callback=stop_callback", call)

    def test_compatibility_framework_has_no_production_importers(self):
        """历史框架可保留给旧数据读取，但不能成为新的生产入口。"""
        app_dir = Path(__file__).parents[2] / "TF-agent"
        importers = []
        for path in app_dir.glob("*.py"):
            if path.name == "agent_task_framework.py":
                continue
            text = path.read_text(encoding="utf-8")
            if "agent_task_framework" in text:
                importers.append(path.name)
        self.assertEqual(importers, [])

    def test_bridge_pending_task_gets_contract(self):
        state = {}
        init_ui_session_defaults(state)
        state["ui_selected_task"] = "p1"
        state["ui_run_mode"] = "dl"
        pending, autotune, errors = build_pending_task(
            state, {"type": "run_pipeline", "task": "p1", "confirmed": True}
        )
        self.assertIsNone(autotune)
        self.assertFalse(errors)
        # build_pending_task 保持兼容 schema；apply_system_command 负责附加契约。
        self.assertEqual(pending["mode"], "dl")
        request = from_pending_task(pending, confirmation_source="agent")
        self.assertEqual(request["entrypoint"], "inference_agent_loop")

    def test_autotune_pending_task_gets_contract(self):
        state = {}
        init_ui_session_defaults(state)
        state["ui_selected_task"] = "p1"
        pending, autotune, errors = build_pending_task(
            state,
            {
                "type": "run_autotune",
                "task": "p1",
                "confirmed": True,
                "autotune_params": {"reference_id": "ref-2020", "objective": "iou"},
            },
        )
        self.assertIsNone(pending)
        self.assertFalse(errors)
        self.assertEqual(autotune["mode"], "autotune")
        contract = attach_execution_request(autotune, confirmation_source="agent")
        self.assertEqual(contract["execution_request"]["entrypoint"], "autotune")

    def test_invalid_task_or_mode_is_rejected(self):
        with self.assertRaises(ValueError):
            ExecutionRequest(task="", mode="dl")
        with self.assertRaises(ValueError):
            ExecutionRequest(task="p1", mode="unknown")


if __name__ == "__main__":
    unittest.main()
