# -*- coding: utf-8 -*-
"""外部 Agent 上下文最小化策略测试。"""
from __future__ import annotations

import os
import sys
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

_TF_AGENT = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "TF-agent")
)
if _TF_AGENT not in sys.path:
    sys.path.insert(0, _TF_AGENT)

import agent_context_policy as policy  # noqa: E402
import agent_command_bridge as bridge  # noqa: E402


class TestAgentContextPolicy(unittest.TestCase):
    def test_sanitize_removes_paths_secrets_and_proxy_credentials(self):
        raw = (
            "path=/Users/chl/Codespace/TF-agent/data/result.tif "
            "win=C:\\Users\\chl\\secret.pth "
            "unc=\\\\server\\share\\a.tif "
            "token=secret123 proxy=http://user:pass@127.0.0.1:7890"
        )
        clean = policy.sanitize_external_text(raw)
        self.assertNotIn("/Users/", clean)
        self.assertNotIn("C:\\", clean)
        self.assertNotIn("\\\\server", clean)
        self.assertNotIn("/Volumes/", policy.sanitize_external_text("mount=/Volumes/External/secret.tif"))
        self.assertNotIn("secret123", clean)
        self.assertNotIn("user:pass", clean)

    def test_sanitize_preserves_web_urls_including_path_like_segments(self):
        urls = (
            "https://example.org/article/123",
            "https://example.org/data/paper.pdf",
            "https://service.example/app/result?id=9",
            "http://gov.example/private/document.html",
            "ftp://archive.example/home/report.zip",
        )
        raw = "参考来源：" + " ".join(urls) + " local=C:\\Users\\chl\\secret.txt"
        clean = policy.sanitize_external_text(raw)
        for url in urls:
            self.assertIn(url, clean)
        self.assertNotIn("C:\\Users\\chl", clean)
        self.assertIn("<local-path>", clean)

    def test_safe_error_summary_is_bounded_and_redacted(self):
        error = RuntimeError(
            "failed /Users/chl/private/model.pth token=sk-secret "
            "https://user:pass@example.invalid/api"
        )
        summary = policy.safe_error_summary(error, limit=80)
        self.assertLessEqual(len(summary), len("RuntimeError: ") + 80)
        self.assertNotIn("/Users/", summary)
        self.assertNotIn("sk-secret", summary)
        self.assertNotIn("user:pass@", summary)

    def test_safe_error_summary_survives_sanitizer_failure(self):
        with patch.object(policy, "sanitize_external_text", side_effect=RuntimeError("sanitizer down")):
            summary = policy.safe_error_summary(ValueError("unavailable"))
        self.assertEqual(summary, "ValueError: ValueError")

    def test_describe_local_path_handles_windows_drive_and_unc_on_posix(self):
        drive = policy.describe_local_path(r"C:\Users\chl\model.pth")
        unc = policy.describe_local_path(r"\\server\share\result.tif")
        self.assertIn("model.pth", drive)
        self.assertNotIn(r"C:\Users\chl", drive)
        self.assertIn("result.tif", unc)
        self.assertNotIn(r"\\server\share", unc)

    def test_sidebar_context_contains_status_not_absolute_paths(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "input.tif"
            p.write_bytes(b"x")
            state = {"ui_root_dir": str(p), "ui_selected_task": "p1"}
            text = bridge.build_agent_sidebar_context(state)
        self.assertIn("input.tif（存在）", text)
        self.assertNotIn(str(p), text)
        self.assertNotIn("/Users/", text)

    def test_sidebar_context_hides_precise_map_center_without_consent(self):
        state = {"map_center": [30.0, 120.0], "map_zoom": 10}
        text = bridge.build_agent_sidebar_context(state)
        self.assertIn("精确坐标未获授权", text)
        self.assertNotIn("30.0", text)
        self.assertNotIn("120.0", text)

        state["agent_spatial_consent"] = True
        text = bridge.build_agent_sidebar_context(state)
        self.assertIn("30.0", text)
        self.assertIn("120.0", text)

    def test_consent_is_session_scoped(self):
        self.assertFalse(policy.media_consent({}))
        self.assertTrue(policy.media_consent({"agent_external_media_consent": True}))
        self.assertFalse(policy.spatial_consent({"agent_external_media_consent": True}))

    def test_redact_spatial_metadata_targets_labelled_fields(self):
        raw = "bbox=(120.6,30.2,121.2,30.9) map_center=[30.55, 120.9]"
        clean = policy.redact_spatial_metadata(raw)
        self.assertNotIn("120.6", clean)
        self.assertNotIn("120.9", clean)
        self.assertIn("<spatial-redacted>", clean)

    def test_redact_geotiff_bounds_crs_and_resolution_without_consent(self):
        raw = (
            "- crs: EPSG:4326\n"
            "- resolution: x=0.00025, y=0.00025\n"
            "- bounds: left=120.600000, bottom=30.200000, right=121.200000, top=30.900000"
        )
        clean = policy.redact_spatial_metadata(raw)
        self.assertNotIn("EPSG:4326", clean)
        self.assertNotIn("120.600000", clean)
        self.assertNotIn("0.00025", clean)
        self.assertGreaterEqual(clean.count("<spatial-redacted>"), 3)

    def test_raw_command_requires_environment_switch_and_session_consent(self):
        old = os.environ.pop("CSTF_ALLOW_RAW_SYSTEM_COMMAND", None)
        try:
            self.assertFalse(policy.raw_system_command_enabled())
            self.assertFalse(policy.raw_system_command_consent({"agent_raw_system_command_consent": True}))
            os.environ["CSTF_ALLOW_RAW_SYSTEM_COMMAND"] = "1"
            self.assertTrue(policy.raw_system_command_enabled())
            self.assertFalse(policy.raw_system_command_consent({}))
            self.assertTrue(policy.raw_system_command_consent({"agent_raw_system_command_consent": True}))
        finally:
            if old is None:
                os.environ.pop("CSTF_ALLOW_RAW_SYSTEM_COMMAND", None)
            else:
                os.environ["CSTF_ALLOW_RAW_SYSTEM_COMMAND"] = old


if __name__ == "__main__":
    unittest.main()
