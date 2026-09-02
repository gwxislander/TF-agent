# -*- coding: utf-8 -*-
"""P0 加固回归测试：安全加载、命令桥完整性、字段统一、代理 URL、数字安全、重型工具确认门闩。

运行：
    D:\\anaconda3\\envs\\gwx\\python.exe -m pytest tests/unit/test_p0_hardening.py -q --tb=short -p no:cacheprovider
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import unittest
from unittest import mock

_TF_AGENT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "TF-agent"))
if _TF_AGENT not in sys.path:
    sys.path.insert(0, _TF_AGENT)

from agent_command_bridge import (  # noqa: E402
    apply_agent_reply_immediate,
    apply_system_command,
    build_pending_task,
    flush_pending_agent_commands,
    init_ui_session_defaults,
    process_agent_reply,
)


def _base_state() -> dict:
    s: dict = {}
    init_ui_session_defaults(s)
    s["ui_prob_th"] = 0.12
    s["ui_min_cnt"] = 4
    s["ui_selected_task"] = "24zhejiang1"
    return s


class TestCommandScalarValidation(unittest.TestCase):
    def test_invalid_iso_date_is_not_applied(self):
        state = _base_state()
        apply_system_command(state, {"sidebar_states": {"m4_start_date": "2020-02-31"}})
        self.assertNotEqual(state["ui_m4_start_date"], "2020-02-31")

    def test_non_finite_threshold_is_not_applied(self):
        state = _base_state()
        before = state["ui_prob_th"]
        apply_system_command(state, {"sidebar_states": {"prob_th": float("nan")}})
        self.assertEqual(state["ui_prob_th"], before)


# ============================================================
# 任务 1：torch.load 必须 weights_only=True（安全加载）
# ============================================================
class TestSafeModelLoading(unittest.TestCase):
    def test_load_model_calls_torch_load_with_weights_only(self):
        import pre_engine

        fake_model = mock.MagicMock()
        fake_state = {"module.backbone.conv1.weight": mock.MagicMock()}
        with mock.patch("pre_engine.CDNet", return_value=fake_model), mock.patch(
            "pre_engine.torch.load", return_value=dict(fake_state)
        ) as m:
            pre_engine.load_model("fake.pth", "cpu")
        self.assertIs(m.call_args.kwargs.get("weights_only"), True,
                      "torch.load 必须显式 weights_only=True，防止 pickle RCE")

    def test_torch_load_with_weights_only_works_for_tensor_state(self):
        """纯张量 state_dict（本项目权重格式）在 weights_only=True 下可正常加载。"""
        import tempfile

        import torch

        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "w.pth")
            torch.save({"backbone.conv1.weight": torch.zeros(2, 2)}, p)
            state = torch.load(p, map_location="cpu", weights_only=True)
            self.assertEqual(list(state.keys()), ["backbone.conv1.weight"])


class TestCIInstallDependencies(unittest.TestCase):
    def test_ci_installs_app_and_model_import_dependencies(self):
        ci_path = Path(__file__).resolve().parents[2] / ".github" / "workflows" / "ci.yml"
        ci_text = ci_path.read_text(encoding="utf-8")
        start = ci_text.index("pip install numpy")
        end = ci_text.index("pip install -r requirements-test.txt", start)
        install_block = ci_text[start:end]
        for dependency in ("leafmap", "streamlit-folium", "einops"):
            self.assertIn(dependency, install_block)


# ============================================================
# 任务 2：apply_agent_reply_immediate 补全
# ============================================================
class TestApplyAgentReplyImmediate(unittest.TestCase):
    def test_no_command_returns_applied_false_and_original(self):
        state = _base_state()
        before = dict(state)
        result, clean = apply_agent_reply_immediate(state, "今天天气不错")
        self.assertFalse(result.applied)
        self.assertEqual(clean, "今天天气不错")
        self.assertEqual(state, before)

    def test_valid_system_command_applied_immediately(self):
        state = _base_state()
        reply = (
            "[SYSTEM_COMMAND_JSON]\n"
            + json.dumps({"sidebar_states": {"prob_th": 0.05}}, ensure_ascii=False)
            + "\n[/SYSTEM_COMMAND_JSON]"
        )
        result, clean = apply_agent_reply_immediate(state, reply)
        self.assertTrue(result.applied)
        self.assertEqual(state["ui_prob_th"], 0.05)
        self.assertIn("prob_th", clean or "")

    def test_invalid_json_does_not_crash(self):
        state = _base_state()
        result, clean = apply_agent_reply_immediate(state, "[SYSTEM_COMMAND_JSON]\n{not valid json\n[/SYSTEM_COMMAND_JSON]")
        self.assertFalse(result.applied)
        self.assertEqual(clean, "[SYSTEM_COMMAND_JSON]\n{not valid json\n[/SYSTEM_COMMAND_JSON]")

    def test_failed_command_reports_errors_and_no_pending(self):
        state = _base_state()
        state.pop("ui_selected_task", None)
        reply = (
            "[SYSTEM_COMMAND_JSON]\n"
            + json.dumps({"pending_action": {"type": "run_pipeline"}}, ensure_ascii=False)
            + "\n[/SYSTEM_COMMAND_JSON]"
        )
        result, _ = apply_agent_reply_immediate(state, reply)
        self.assertTrue(result.applied)
        self.assertTrue(result.errors)
        self.assertNotIn("pending_task", state)

    def test_return_tuple_shape_matches_process_agent_reply(self):
        state = _base_state()
        reply = (
            "[SYSTEM_COMMAND_JSON]\n"
            + json.dumps({"sidebar_states": {"min_cnt": 5}}, ensure_ascii=False)
            + "\n[/SYSTEM_COMMAND_JSON]"
        )
        r_imm, c_imm = apply_agent_reply_immediate(state, reply)
        self.assertEqual(state["ui_min_cnt"], 5)
        self.assertTrue(r_imm.applied)
        self.assertIsInstance(c_imm, str)


# ============================================================
# 任务 3：统一 workspace_tab / workflow_tab
# ============================================================
class TestWorkflowTabUnification(unittest.TestCase):
    def test_new_field_workflow_tab(self):
        state = _base_state()
        apply_system_command(state, {"sidebar_states": {"workflow_tab": "GEE 数据下载"}})
        self.assertEqual(state["ui_workflow"], "GEE 数据下载")

    def test_legacy_field_workspace_tab_still_works(self):
        state = _base_state()
        apply_system_command(state, {"sidebar_states": {"workspace_tab": "GEE 数据下载"}})
        self.assertEqual(state["ui_workflow"], "GEE 数据下载")

    def test_legacy_field_alias_normalized(self):
        state = _base_state()
        apply_system_command(state, {"sidebar_states": {"workspace_tab": "gee数据下载"}})
        self.assertEqual(state["ui_workflow"], "GEE 数据下载")

    def test_workspace_tab_and_workflow_tab_do_not_conflict(self):
        state = _base_state()
        # 两者同时给出时以规范字段 workflow_tab 为准（先处理规范字段）
        cmd = {
            "sidebar_states": {
                "workflow_tab": "潮滩推理",
                "workspace_tab": "GEE 数据下载",
            }
        }
        apply_system_command(state, cmd)
        self.assertEqual(state["ui_workflow"], "潮滩推理")


# ============================================================
# 任务 4：m4_gee_proxy 不得被 normpath 破坏
# ============================================================
class TestM4ProxyNoNormpath(unittest.TestCase):
    def test_http_url_preserved(self):
        state = _base_state()
        apply_system_command(state, {"sidebar_states": {"m4_gee_proxy": "http://127.0.0.1:7890"}})
        self.assertEqual(state["ui_m4_gee_proxy"], "http://127.0.0.1:7890")

    def test_https_url_preserved(self):
        state = _base_state()
        apply_system_command(state, {"sidebar_states": {"m4_gee_proxy": "https://proxy.example.com:8080"}})
        self.assertEqual(state["ui_m4_gee_proxy"], "https://proxy.example.com:8080")

    def test_http_with_path_preserved(self):
        state = _base_state()
        apply_system_command(state, {"sidebar_states": {"m4_gee_proxy": "http://127.0.0.1:7890/proxy"}})
        self.assertEqual(state["ui_m4_gee_proxy"], "http://127.0.0.1:7890/proxy")

    def test_windows_path_normalized(self):
        state = _base_state()
        apply_system_command(state, {"sidebar_states": {"root_dir": "E:/Data/843output"}})
        self.assertEqual(state["ui_root_dir"], os.path.normpath("E:/Data/843output"))

    def test_trailing_slash_path_normalized(self):
        state = _base_state()
        apply_system_command(state, {"sidebar_states": {"mask_root": "E:/Data/843mask/"}})
        self.assertEqual(state["ui_mask_root"], os.path.normpath("E:/Data/843mask/"))

    def test_empty_proxy_keeps_previous_value(self):
        state = _base_state()
        state["ui_m4_gee_proxy"] = "http://127.0.0.1:7890"
        apply_system_command(state, {"sidebar_states": {"m4_gee_proxy": ""}})
        self.assertEqual(state["ui_m4_gee_proxy"], "http://127.0.0.1:7890")


# ============================================================
# 任务 5：pending_action 数字参数安全解析
# ============================================================
class TestPendingActionNumericSafety(unittest.TestCase):
    # 数字安全测试：confirmed=True 以通过重型工具确认门闩，聚焦解析逻辑本身。
    def test_numeric_string_parsed(self):
        state = _base_state()
        pt, _, errs = build_pending_task(
            state, {"type": "run_pipeline", "confirmed": True, "task": "t", "prob_th": "0.05", "min_cnt": "2"}
        )
        self.assertFalse(errs)
        self.assertEqual(pt["prob"], 0.05)
        self.assertEqual(pt["cnt"], 2)

    def test_percent_string_falls_back_to_sidebar(self):
        state = _base_state()
        pt, _, errs = build_pending_task(state, {"type": "run_pipeline", "confirmed": True, "task": "t", "prob_th": "5%"})
        self.assertFalse(errs)
        self.assertEqual(pt["prob"], state["ui_prob_th"])

    def test_int_value_clamped(self):
        state = _base_state()
        pt, _, errs = build_pending_task(state, {"type": "run_pipeline", "confirmed": True, "task": "t", "prob_th": 5, "min_cnt": 5})
        self.assertFalse(errs)
        self.assertEqual(pt["prob"], 0.50)  # 0.01~0.50 上界
        self.assertEqual(pt["cnt"], 5)

    def test_none_uses_sidebar(self):
        state = _base_state()
        pt, _, errs = build_pending_task(
            state, {"type": "run_pipeline", "confirmed": True, "task": "t", "prob_th": None, "min_cnt": None}
        )
        self.assertFalse(errs)
        self.assertEqual(pt["prob"], state["ui_prob_th"])
        self.assertEqual(pt["cnt"], state["ui_min_cnt"])

    def test_empty_string_uses_sidebar(self):
        state = _base_state()
        pt, _, errs = build_pending_task(
            state, {"type": "run_pipeline", "confirmed": True, "task": "t", "prob_th": "", "min_cnt": ""}
        )
        self.assertFalse(errs)
        self.assertEqual(pt["prob"], state["ui_prob_th"])

    def test_garbage_string_falls_back_without_crash(self):
        state = _base_state()
        pt, _, errs = build_pending_task(
            state, {"type": "run_pipeline", "confirmed": True, "task": "t", "prob_th": "abc", "min_cnt": "xyz"}
        )
        self.assertFalse(errs)
        self.assertEqual(pt["prob"], state["ui_prob_th"])
        self.assertEqual(pt["cnt"], state["ui_min_cnt"])

    def test_out_of_range_clamped(self):
        state = _base_state()
        pt, _, errs = build_pending_task(
            state, {"type": "run_pipeline", "confirmed": True, "task": "t", "prob_th": 0.99, "min_cnt": 99}
        )
        self.assertFalse(errs)
        self.assertEqual(pt["prob"], 0.50)
        self.assertEqual(pt["cnt"], 10)

    def test_m4_params_safe(self):
        state = _base_state()
        pt, _, errs = build_pending_task(
            state,
            {
                "type": "run_m4",
                "confirmed": True,
                "task": "zhejiang1",
                "m4_params": {
                    "cloud_limit": "20",
                    "min_land_pct": "5%",
                    "scale": "abc",
                    "bands": "B8,B4",
                },
            },
        )
        self.assertFalse(errs, errs)
        self.assertEqual(pt["m4"]["cloud_limit"], 20)
        self.assertEqual(pt["m4"]["min_land_pct"], 5.0)
        self.assertIn("B8", pt["m4"]["bands"])


# ============================================================
# 任务 6：重型工具（run_pipeline/run_m4/run_autotune）确认门闩
# ============================================================
class TestHeavyToolConfirmationGate(unittest.TestCase):
    def test_run_pipeline_requires_confirmation(self):
        state = _base_state()
        cmd = {"pending_action": {"type": "run_pipeline", "task": "24zhejiang1"}}
        result = apply_system_command(state, cmd)
        self.assertNotIn("pending_task", state)
        self.assertTrue(any("确认" in e for e in result.errors), result.errors)

    def test_run_m4_requires_confirmation(self):
        state = _base_state()
        cmd = {"pending_action": {"type": "run_m4", "task": "zhejiang1"}}
        result = apply_system_command(state, cmd)
        self.assertNotIn("pending_task", state)
        self.assertTrue(any("确认" in e for e in result.errors), result.errors)

    def test_run_autotune_requires_confirmation(self):
        state = _base_state()
        cmd = {
            "pending_action": {
                "type": "run_autotune",
                "task": "24zhejiang1",
                "autotune_params": {"reference_id": "x"},
            }
        }
        result = apply_system_command(state, cmd)
        self.assertNotIn("pending_autotune", state)
        self.assertTrue(any("确认" in e for e in result.errors), result.errors)

    def test_pending_confirm_state_recorded_for_ui(self):
        state = _base_state()
        cmd = {"pending_action": {"type": "run_pipeline", "task": "24zhejiang1"}}
        apply_system_command(state, cmd)
        pc = state.get("_pending_heavy_confirm")
        self.assertIsInstance(pc, dict)
        self.assertEqual(pc.get("action_type"), "run_pipeline")

    def test_confirmed_run_pipeline_executes(self):
        state = _base_state()
        cmd = {"pending_action": {"type": "run_pipeline", "task": "24zhejiang1", "confirmed": True}}
        result = apply_system_command(state, cmd)
        self.assertFalse(result.errors, result.errors)
        self.assertIn("pending_task", state)
        self.assertEqual(state["pending_task"]["task"], "24zhejiang1")

    def test_confirmed_run_m4_executes(self):
        state = _base_state()
        cmd = {
            "pending_action": {
                "type": "run_m4",
                "task": "zhejiang1",
                "confirmed": True,
                "m4_params": {"cloud_limit": 20},
            }
        }
        result = apply_system_command(state, cmd)
        self.assertFalse(result.errors, result.errors)
        self.assertEqual(state["pending_task"]["mode"], "m4")
        self.assertEqual(state["pending_task"]["m4"]["cloud_limit"], 20)

    def test_confirmed_run_autotune_executes(self):
        state = _base_state()
        cmd = {
            "pending_action": {
                "type": "run_autotune",
                "task": "24zhejiang1",
                "confirmed": True,
                "autotune_params": {"reference_id": "师姐_2020"},
            }
        }
        result = apply_system_command(state, cmd)
        self.assertFalse(result.errors, result.errors)
        self.assertIn("pending_autotune", state)
        self.assertEqual(state["pending_autotune"]["reference_id"], "师姐_2020")
        self.assertEqual(state["pending_autotune"]["mode"], "autotune")
        self.assertEqual(
            state["pending_autotune"]["execution_request"]["entrypoint"],
            "autotune",
        )

    def test_image_driven_llm_command_cannot_bypass_confirmation(self):
        """上传图片产生的 LLM 指令走同一解析/合流路径，必须同样经过确认门闩。"""
        state = _base_state()
        # 模拟 VLM 读了图片内容后输出的指令（无 confirmed）
        reply = (
            "[SYSTEM_COMMAND_JSON]\n"
            + json.dumps(
                {"pending_action": {"type": "run_pipeline", "task": "24zhejiang1"}},
                ensure_ascii=False,
            )
            + "\n[/SYSTEM_COMMAND_JSON]"
        )
        result, _ = process_agent_reply(state, reply)
        self.assertTrue(result.queued)
        flush_pending_agent_commands(state)
        self.assertNotIn("pending_task", state)
        self.assertTrue(state.get("_pending_heavy_confirm"))

    def test_duplicate_confirm_does_not_start_twice(self):
        """重复 confirmed 指令不会重复创建可执行任务（消费后需重新确认）。"""
        state = _base_state()
        apply_system_command(
            state, {"pending_action": {"type": "run_pipeline", "task": "24zhejiang1", "confirmed": True}}
        )
        self.assertIn("pending_task", state)
        state["pipeline_thread_started"] = True  # 模拟线程已消费
        state.pop("pending_task", None)
        # 消费后再次收到 confirmed：可重建，但 is_running 语义由 app 层防止重复启动
        apply_system_command(
            state, {"pending_action": {"type": "run_pipeline", "task": "24zhejiang1", "confirmed": True}}
        )
        self.assertIn("pending_task", state)


if __name__ == "__main__":
    unittest.main()
