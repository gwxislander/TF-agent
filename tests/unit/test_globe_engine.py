# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import sys
import unittest
import importlib.util
import tempfile
from pathlib import Path

_TF_AGENT = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "TF-agent")
)
if _TF_AGENT not in sys.path:
    sys.path.insert(0, _TF_AGENT)

import globe_engine  # noqa: E402


class TestGlobeEngineUi(unittest.TestCase):
    def test_navigation_help_is_hidden_by_default(self):
        html = globe_engine.build_cesium_html({})
        self.assertIn("navigationHelpButton: false", html)
        self.assertIn(".cesium-navigation-help-button", html)

    def test_rectangle_aoi_requires_drag_and_has_live_preview(self):
        html = globe_engine.build_cesium_html({})
        self.assertIn("rectStartScreen", html)
        self.assertIn("Math.hypot(dx, dy) < 8", html)
        self.assertIn("Cesium.ScreenSpaceEventType.MOUSE_MOVE", html)
        self.assertIn("矩形模式：请按住鼠标拖拽框选", html)
        self.assertIn("矩形绘制中…松开鼠标完成", html)
        self.assertIn("screenSpaceCameraController.enableInputs = !mode", html)
        self.assertIn("showLocalAoiPreview(geometry)", html)
        self.assertIn('indexOf("aoi:") === 0', html)
        self.assertIn("AOI 已选定，已同步", html)
        self.assertIn("AOI 发送失败，请重试", html)

    def test_unsequenced_fly_is_rejected_after_a_sequenced_navigation(self):
        """A delayed legacy FLY must not cancel the current sequenced flight."""
        html = globe_engine.build_cesium_html({})
        guard = "if (!hasNavigationSeq && _lastFlyNavigationSeq > 0)"
        self.assertIn(guard, html)
        self.assertLess(html.index(guard), html.index("const navigationOptions ="))

    def test_message_listener_validates_protocol_and_channel_before_binding_origin(self):
        html = globe_engine.build_cesium_html({})
        listener = html[html.index('window.addEventListener("message"'):]
        self.assertIn("if (data.version !== 1) return;", listener)
        self.assertIn('if (data.channel_id !== (CFG.channelId || "default")) return;', listener)
        self.assertLess(listener.index("if (data.version !== 1) return;"), listener.index("_parentOrigin = ev.origin"))
        self.assertLess(listener.index('if (data.channel_id !== (CFG.channelId || "default")) return;'), listener.index("_parentOrigin = ev.origin"))
        self.assertIn('if (type !== "CSTF_FLY" && type !== "CSTF_LAYER_ADD" && type !== "CSTF_LAYER_REMOVE") return;', listener)

    @unittest.skipUnless(importlib.util.find_spec("rasterio"), "rasterio required")
    def test_large_raster_gets_cached_map_preview(self):
        """大栅格地图显示使用降采样缓存，不阻塞原始成果读取。"""
        import numpy as np
        import rasterio
        from rasterio.transform import from_origin

        with tempfile.TemporaryDirectory() as td:
            source = Path(td) / "any_disagreement.tif"
            with rasterio.open(
                source,
                "w",
                driver="GTiff",
                width=4096,
                height=2048,
                count=1,
                dtype="uint8",
                crs="EPSG:4326",
                transform=from_origin(120, 42, 0.001, 0.001),
                compress="lzw",
            ) as dst:
                dst.write(np.zeros((1, 2048, 4096), dtype=np.uint8))

            preview = globe_engine.prepare_map_asset(
                str(source), max_pixels=1_000_000, max_dimension=1024
            )
            self.assertTrue(preview)
            self.assertNotEqual(os.path.normpath(str(source)), preview)
            self.assertTrue(os.path.isfile(preview))
            with rasterio.open(preview) as ds:
                self.assertLessEqual(ds.width * ds.height, 1_000_000)
                self.assertLessEqual(max(ds.width, ds.height), 1024)

            # 第二次调用复用同一个缓存，不重复生成文件。
            self.assertEqual(
                preview,
                globe_engine.prepare_map_asset(
                    str(source), max_pixels=1_000_000, max_dimension=1024
                ),
            )


if __name__ == "__main__":
    unittest.main()
