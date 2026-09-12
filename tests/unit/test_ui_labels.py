# -*- coding: utf-8 -*-
"""ui_labels 展示层映射回归测试（用户界面中文化与去技术代号改造）。"""
import os
import sys
import unittest

_TF_AGENT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "TF-agent"))
if _TF_AGENT not in sys.path:
    sys.path.insert(0, _TF_AGENT)

import ui_labels as uil  # noqa: E402


class TestToolLabels(unittest.TestCase):
    """核心功能：tool id -> 面向功能的中文名。"""

    def test_gee_download_label(self):
        self.assertEqual(uil.get_tool_label("gee_download"), "获取卫星影像")

    def test_local_inference_label(self):
        self.assertEqual(uil.get_tool_label("local_inference"), "潮滩智能提取")
        self.assertEqual(uil.get_tool_label("local_tidal_flat_inference"), "潮滩智能提取")

    def test_e1_label(self):
        self.assertEqual(uil.get_tool_label("e1_quality"), "潮滩精度评价")
        self.assertEqual(uil.get_tool_label("e1_quality_evaluation"), "潮滩精度评价")
        self.assertEqual(uil.get_tool_label("e1"), "潮滩精度评价")

    def test_m5_label(self):
        self.assertEqual(uil.get_tool_label("m5_change"), "潮滩变化分析")
        self.assertEqual(uil.get_tool_label("m5_change_detection"), "潮滩变化分析")
        self.assertEqual(uil.get_tool_label("m5"), "潮滩变化分析")

    def test_workflow_label(self):
        self.assertEqual(uil.get_tool_label("analysis_workflow"), "一键潮滩分析")
        self.assertEqual(uil.get_tool_label("workflow"), "一键潮滩分析")

    def test_pdf_report_label(self):
        self.assertEqual(uil.get_tool_label("pdf_report"), "成果报告")
        self.assertEqual(uil.get_tool_label("report"), "成果报告")


class TestPhaseStatusAssetLabels(unittest.TestCase):
    """执行阶段 / 状态 / 资产类型 中文转换。"""

    def test_phase_chinese(self):
        self.assertEqual(uil.get_phase_label("PLAN"), "生成计划")
        self.assertEqual(uil.get_phase_label("VALIDATE"), "条件检查")
        self.assertEqual(uil.get_phase_label("CONFIRM"), "等待确认")
        self.assertEqual(uil.get_phase_label("INFERENCE"), "智能提取")
        self.assertEqual(uil.get_phase_label("POST_PROCESS"), "成果生成")
        self.assertEqual(uil.get_phase_label("REPORT"), "生成报告")

    def test_status_chinese(self):
        self.assertEqual(uil.get_status_label("SUCCEEDED"), "已完成")
        self.assertEqual(uil.get_status_label("FAILED"), "失败")
        self.assertEqual(uil.get_status_label("BLOCKED"), "暂不可执行")
        self.assertEqual(uil.get_status_label("CANCELLED"), "已取消")
        self.assertEqual(uil.get_status_label("WAITING_CONFIRMATION"), "等待确认")

    def test_asset_type_chinese(self):
        self.assertEqual(uil.get_asset_label("dataset"), "卫星影像")
        self.assertEqual(uil.get_asset_label("prediction"), "潮滩提取成果")
        self.assertEqual(uil.get_asset_label("e1_evaluation"), "精度评价结果")
        self.assertEqual(uil.get_asset_label("m5_change"), "变化分析结果")
        self.assertEqual(uil.get_asset_label("report"), "成果报告")

    def test_asset_caption_handles_assets_without_inference_parameters(self):
        """E1/M5/报告资产没有概率与次数参数时，成果列表仍应可展示。"""
        caption = uil.format_asset_caption(
            {
                "task": "2022_5b_liaohekou1",
                "method": "e1",
                "created_at": "2026-09-12 10:00:00",
                "file_size_mb": 1.25,
            }
        )
        self.assertEqual(caption, "精度评价 · 2026-09-12 10:00:00 · 1.25MB")

    def test_asset_caption_keeps_inference_parameter_display(self):
        """深度学习成果仍显示概率阈值和最少次数。"""
        caption = uil.format_asset_caption(
            {
                "method": "dl",
                "prob_threshold": 0.15,
                "min_count": 8,
                "created_at": "2026-09-12 10:00:00",
                "file_size_mb": 2,
            }
        )
        self.assertEqual(caption, "P=0.15 C=8 · 2026-09-12 10:00:00 · 2MB")


class TestFallback(unittest.TestCase):
    """未知键不崩溃，回退原值。"""

    def test_unknown_tool_fallback(self):
        self.assertEqual(uil.get_tool_label("no_such_tool_xyz"), "no_such_tool_xyz")

    def test_unknown_phase_fallback(self):
        self.assertEqual(uil.get_phase_label("NO_SUCH_PHASE"), "NO_SUCH_PHASE")

    def test_unknown_status_fallback(self):
        self.assertEqual(uil.get_status_label("NO_SUCH_STATUS"), "NO_SUCH_STATUS")

    def test_unknown_asset_fallback(self):
        self.assertEqual(uil.get_asset_label("no_such_asset"), "no_such_asset")

    def test_none_safe(self):
        self.assertEqual(uil.get_tool_label(None), "")
        self.assertEqual(uil.get_phase_label(None), "")
        self.assertEqual(uil.get_status_label(None), "")


class TestPresentationRegression(unittest.TestCase):
    """展示层回归：映射值（即用户会看到的文字）不得包含技术代号。

    仅扫描"展示值"（label 字典的 value），不扫描内部 key / 注释 / 调试日志。
    """

    FORBIDDEN = (
        "M5 ",
        "E1 ",
        "GEE ",
        "Local Inference",
        "Workflow Plan",
        "Capability Status",
        "Asset Registry",
    )

    def _collect_label_values(self):
        values = []
        for attr in ("TOOL_LABELS", "PHASE_LABELS", "STATUS_LABELS",
                     "ASSET_LABELS", "MAP_LAYER_LABELS", "CAPABILITY_LABELS",
                     "TERM_LABELS"):
            mapping = getattr(uil, attr, None)
            if isinstance(mapping, dict):
                values.extend(str(v) for v in mapping.values())
        return values

    def test_no_technical_codes_in_label_values(self):
        bad = []
        for v in self._collect_label_values():
            for tok in self.FORBIDDEN:
                if tok in v:
                    bad.append(f"{tok!r} 出现在展示值 {v!r}")
        self.assertEqual(bad, [])

    def test_internal_keys_untouched(self):
        """内部标识（key）必须原样保留，不得被翻译。"""
        for tool_id in ("gee_download", "local_inference", "local_tidal_flat_inference",
                        "e1_quality_evaluation", "m5_change_detection", "pdf_report",
                        "analysis_workflow"):
            self.assertIn(tool_id, uil.TOOL_LABELS)
        for phase in ("PLAN", "VALIDATE", "CONFIRM", "QUEUED", "EXECUTE",
                      "INFERENCE", "POST_PROCESS", "VERIFY", "REGISTER", "MAP", "REPORT"):
            self.assertIn(phase, uil.PHASE_LABELS)
        for status in ("PENDING", "WAITING_CONFIRMATION", "QUEUED", "RUNNING",
                       "SUCCEEDED", "FAILED", "BLOCKED", "CANCELLED"):
            self.assertIn(status, uil.STATUS_LABELS)
        for asset_type in ("dataset", "prediction", "e1_evaluation",
                           "m5_change", "report"):
            self.assertIn(asset_type, uil.ASSET_LABELS)


if __name__ == "__main__":
    unittest.main()
