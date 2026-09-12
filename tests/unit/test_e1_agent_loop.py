# -*- coding: utf-8 -*-
"""E1 Agent 闭环：预检、确认门闩、计划与校验（短测，不跑全量栅格引擎）。"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

_TF_AGENT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "TF-agent"))
if _TF_AGENT not in sys.path:
    sys.path.insert(0, _TF_AGENT)

from agent_command_bridge import (  # noqa: E402
    apply_system_command,
    build_pending_task,
    init_ui_session_defaults,
    propose_e1_plan,
)
import e1_agent_loop  # noqa: E402


def _touch_shp(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")
    path.with_suffix(".shx").write_bytes(b"")
    path.with_suffix(".dbf").write_bytes(b"")
    path.with_suffix(".prj").write_text('GEOGCS["WGS 84"]', encoding="utf-8")


class TestE1Intent(unittest.TestCase):
    def test_intent_and_confirm(self):
        self.assertTrue(e1_agent_loop.is_e1_intent("做一下多源一致性诊断"))
        self.assertTrue(e1_agent_loop.is_e1_intent("跑 E1 对比师姐2020"))
        self.assertFalse(e1_agent_loop.is_e1_intent("跳转到杭州湾"))
        self.assertTrue(e1_agent_loop.is_e1_confirm_utterance("确认"))
        self.assertTrue(e1_agent_loop.is_e1_confirm_utterance("开始执行 E1"))
        self.assertFalse(e1_agent_loop.is_e1_confirm_utterance("随便聊聊天气"))


class TestE1Preflight(unittest.TestCase):
    def test_ready_when_shp_and_data_root(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            final = root / "final"
            data = root / "datasets"
            data.mkdir()
            shp = final / "24zhejiang1" / "24zhejiang1_Final_p0.05_c2.shp"
            _touch_shp(shp)
            plan = e1_agent_loop.build_e1_preflight(
                final_root=str(final),
                current_task="24zhejiang1",
                data_root=str(data),
                reference="师姐_2020",
                prob=0.05,
                cnt=2,
            )
            self.assertTrue(plan["ready"])
            self.assertEqual(plan["current_task"], "24zhejiang1")
            self.assertTrue(plan["current_shp"])
            self.assertEqual(plan["reference"], "师姐_2020")

    def test_blocked_without_current_shp(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            final = root / "final"
            final.mkdir()
            data = root / "datasets"
            data.mkdir()
            plan = e1_agent_loop.build_e1_preflight(
                final_root=str(final),
                current_task="24zhejiang1",
                data_root=str(data),
            )
            self.assertFalse(plan["ready"])
            self.assertTrue(any("SHP" in b or "不存在" in b for b in plan["blockers"]))

    def test_blocked_without_data_root(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            final = root / "final"
            shp = final / "20zhejiang1" / "20zhejiang1_Final_p0.05_c2.shp"
            _touch_shp(shp)
            plan = e1_agent_loop.build_e1_preflight(
                final_root=str(final),
                current_task="20zhejiang1",
                data_root=str(root / "missing_datasets"),
                prob=0.05,
                cnt=2,
            )
            self.assertFalse(plan["ready"])
            self.assertTrue(any("数据集" in b or "data_root" in b.lower() or "目录" in b for b in plan["blockers"]))


class TestE1VerifyAndSummary(unittest.TestCase):
    def test_empty_report_file_fails_verification(self):
        with tempfile.TemporaryDirectory() as td:
            report_path = os.path.join(td, "empty-report.json")
            Path(report_path).touch()
            report = {
                "reference": "师姐_2020",
                "report_path": report_path,
                "comparisons": {"pair": {"jaccard_iou": 0.2}},
            }

            verification = e1_agent_loop.verify_e1_outputs(report)

            self.assertFalse(verification["ok"])
            failed = {c["name"] for c in verification["checks"] if not c["passed"]}
            self.assertIn("report_json_on_disk", failed)

    def test_verify_and_summary(self):
        with tempfile.TemporaryDirectory() as td:
            report_path = os.path.join(td, "E1_PIXEL_REPORT_24zhejiang1.json")
            heat = os.path.join(td, "disagreement_heatmap.tif")
            Path(heat).write_bytes(b"fake")
            report = {
                "roi_name": "24zhejiang1",
                "reference": "师姐_2020",
                "report_path": report_path,
                "comparisons": {
                    "DCTF_2020_vs_师姐_2020": {
                        "jaccard_iou": 0.42,
                        "intersection_km2": 10.0,
                        "union_km2": 24.0,
                        "causal_analysis": {
                            "disagreement_maps": {"heatmap": heat},
                        },
                    }
                },
                "multi_product_heatmap": {"disagreement_pixel_ratio": 0.15},
            }
            Path(report_path).write_text(json.dumps(report), encoding="utf-8")
            ver = e1_agent_loop.verify_e1_outputs(report)
            self.assertTrue(ver["ok"])
            self.assertEqual(ver["map_candidate"], os.path.normpath(heat))
            text = e1_agent_loop.summarize_e1_report_for_chat(report, ver)
            self.assertIn("已验证", text)
            self.assertIn("0.42", text)
            self.assertIn("师姐_2020", text)

    def test_legacy_e1_asset_resolves_map_from_report_path(self):
        """旧资产把报告路径放在 file_path 时，加载仍应找到可绘制图层。"""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            report_path = root / "E1_PIXEL_REPORT_task.json"
            heat = root / "maps" / "disagreement_heatmap.tif"
            heat.parent.mkdir()
            heat.write_bytes(b"fake-raster")
            report_path.write_text(
                json.dumps(
                    {
                        "roi_name": "task",
                        "reference": "师姐_2020",
                        "report_path": report_path.name,
                        "comparisons": {
                            "pair": {
                                "jaccard_iou": 0.5,
                                "causal_analysis": {
                                    "disagreement_maps": {
                                        "heatmap": os.path.join("maps", heat.name)
                                    }
                                },
                            }
                        },
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            asset = {
                "task": "task",
                "method": "e1",
                "file_path": str(report_path),
                "report_path": str(report_path),
            }
            resolved = e1_agent_loop.resolve_e1_map_asset(asset)
            self.assertEqual(resolved, os.path.normpath(str(heat)))

    def test_e1_asset_prefers_explicit_map_path(self):
        with tempfile.TemporaryDirectory() as td:
            heat = Path(td) / "heatmap.tif"
            heat.write_bytes(b"fake-raster")
            report = Path(td) / "report.json"
            report.write_text("{}", encoding="utf-8")
            resolved = e1_agent_loop.resolve_e1_map_asset(
                {
                    "method": "e1",
                    "file_path": str(report),
                    "map_path": str(heat),
                    "report_path": str(report),
                }
            )
            self.assertEqual(resolved, os.path.normpath(str(heat)))


class TestE1BridgeGate(unittest.TestCase):
    def _state(self, final_root: str, data_root: str, task: str = "24zhejiang1") -> dict:
        s: dict = {}
        init_ui_session_defaults(s)
        s["ui_final_root"] = final_root
        s["ui_e1_data_root"] = data_root
        s["ui_selected_task"] = task
        s["ui_e1_reference"] = "师姐_2020"
        s["ui_prob_th"] = 0.05
        s["ui_min_cnt"] = 2
        return s

    def test_propose_then_run_requires_confirm(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            final = root / "final"
            data = root / "datasets"
            data.mkdir()
            _touch_shp(final / "24zhejiang1" / "24zhejiang1_Final_p0.05_c2.shp")
            state = self._state(str(final), str(data))
            r = apply_system_command(
                state, {"pending_action": {"type": "propose_e1", "task": "24zhejiang1"}}
            )
            self.assertEqual(r.action_type, "propose_e1")
            self.assertTrue(state["_e1_pending_plan"]["ready"])
            self.assertFalse(state.get("is_running"))
            self.assertNotIn("pending_task", state)

            pt, _, errs = build_pending_task(
                state, {"type": "run_e1", "confirmed": False}
            )
            self.assertIsNone(pt)
            self.assertTrue(errs)

            pt2, _, errs2 = build_pending_task(
                state, {"type": "run_e1", "confirmed": True}
            )
            self.assertFalse(errs2)
            self.assertEqual(pt2["mode"], "e1")
            self.assertTrue(pt2["e1"]["target_shp"])

    def test_confirm_e1_starts_pending(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            final = root / "final"
            data = root / "datasets"
            data.mkdir()
            _touch_shp(final / "24zhejiang1" / "24zhejiang1_Final_p0.05_c2.shp")
            state = self._state(str(final), str(data))
            propose_e1_plan(state, {"task": "24zhejiang1"})
            r = apply_system_command(
                state, {"pending_action": {"type": "confirm_e1"}}
            )
            self.assertEqual(r.action_type, "run_e1")
            self.assertTrue(state.get("is_running"))
            self.assertEqual(state["pending_task"]["mode"], "e1")


class TestE1AgentTools(unittest.TestCase):
    """Agent 工具应发出 propose_e1 / run_e1（确认后）。"""

    def test_prepare_and_confirm_tools_emit_json(self):
        import agent as cstf_agent

        prep = cstf_agent.prepare_e1_consistency_check.invoke(
            {"task": "24zhejiang1", "reference": "师姐_2020"}
        )
        self.assertIn("[SYSTEM_COMMAND_JSON]", prep)
        self.assertIn("propose_e1", prep)
        self.assertIn("24zhejiang1", prep)

        conf = cstf_agent.confirm_and_run_e1.invoke({"task": "24zhejiang1"})
        self.assertIn("run_e1", conf)
        self.assertIn("confirmed", conf)


if __name__ == "__main__":
    unittest.main()
