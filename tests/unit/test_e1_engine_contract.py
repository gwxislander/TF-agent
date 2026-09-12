# -*- coding: utf-8 -*-
"""E1 评价范围契约：无显式 ROI 时不得退化为全国超大网格。"""
from __future__ import annotations

import os
import sys
import unittest
from unittest import mock

_TF_AGENT = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "TF-agent")
)
_JB_DIR = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "research", "jb")
)
for _path in (_TF_AGENT, _JB_DIR):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from E1 import E1Cancelled, _stop_requested, _vector_bounds_4326  # noqa: E402
from E1 import E1_DataCleanerAndDiagnostic  # noqa: E402
import e1_engine  # noqa: E402


class _FakeGeoDataFrame:
    crs = "EPSG:32651"
    total_bounds = (380330.0, 4509150.0, 412940.0, 4540850.0)

    def to_crs(self, crs):
        self.crs = crs
        self.total_bounds = (120.10, 40.70, 120.55, 41.02)
        return self


class TestE1EngineContract(unittest.TestCase):
    def test_stop_callback_is_evaluated(self):
        self.assertTrue(_stop_requested(lambda: True))
        self.assertFalse(_stop_requested(lambda: False))

    def test_pixel_comparison_honors_stop_before_starting(self):
        e1 = object.__new__(E1_DataCleanerAndDiagnostic)
        e1.dataset_specs = {"reference": {"kind": "vector", "path": "reference.shp"}}
        with self.assertRaises(E1Cancelled):
            e1.run_pixel_comparison(
                reference="reference",
                compare_sources=[],
                stop_callback=lambda: True,
            )

    def test_target_bounds_are_transformed_to_wgs84(self):
        with mock.patch("E1.gpd.read_file", return_value=_FakeGeoDataFrame()):
            bounds = _vector_bounds_4326("target.shp")
        self.assertEqual(bounds, (120.10, 40.70, 120.55, 41.02))

    def test_invalid_target_bounds_return_none_for_safe_fallback(self):
        class Empty:
            crs = None
            total_bounds = (0.0, 0.0, 0.0, 0.0)

        with mock.patch("E1.gpd.read_file", return_value=Empty()):
            self.assertIsNone(_vector_bounds_4326("target.shp"))

    def test_loaded_legacy_report_gets_disk_paths_for_verification(self):
        with unittest.mock.patch("builtins.open", mock.mock_open(read_data='{"roi_name":"task","comparisons":{}}')):
            with unittest.mock.patch("os.path.isfile", return_value=True):
                report = e1_engine.load_e1_report("/tmp/e1_workspace", "task")
        self.assertEqual(
            report["report_path"],
            "/tmp/e1_workspace/outputs_e1/E1_PIXEL_REPORT_task.json",
        )
        self.assertEqual(
            report["report_md_path"],
            "/tmp/e1_workspace/outputs_e1/E1_PIXEL_REPORT_task.md",
        )

    def test_local_dataset_read_passes_bbox_to_driver(self):
        import geopandas as gpd
        from shapely.geometry import Point

        e1 = object.__new__(E1_DataCleanerAndDiagnostic)
        e1.dataset_specs = {"source": {"kind": "vector", "path": "source.shp"}}
        e1._apply_filter = lambda gdf, filt: gdf
        frame = gpd.GeoDataFrame({"geometry": [Point(121, 41)]}, crs="EPSG:4326")
        with mock.patch("E1.Path.exists", return_value=True):
            with mock.patch("E1.gpd.read_file", return_value=frame) as read_file:
                e1.load_dataset("source", bbox_4326=(120.0, 40.0, 122.0, 42.0))
        self.assertIn("bbox", read_file.call_args.kwargs)

    def test_target_extent_bounds_localize_evaluation_grid_without_roi(self):
        import geopandas as gpd
        from shapely.geometry import box
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            root = os.path.abspath(td)
            target = os.path.join(root, "target.shp")
            reference = os.path.join(root, "reference.shp")
            frame = gpd.GeoDataFrame(
                {"geometry": [box(121.0, 40.0, 121.1, 40.1)]},
                crs="EPSG:4326",
            )
            frame.to_file(target, driver="ESRI Shapefile")
            frame.to_file(reference, driver="ESRI Shapefile")

            e1 = E1_DataCleanerAndDiagnostic(
                workspace_dir=os.path.join(root, "workspace"),
                data_root=root,
                pixel_size_m=10_000,
            )
            e1.dataset_specs = {
                "reference": {"kind": "vector", "path": reference},
            }
            report = e1.run_pixel_comparison(
                reference="reference",
                target_path=target,
                target_name="target",
                compare_sources=[],
                roi_path=None,
                roi_name="local",
                export_rasters=False,
                export_disagreement_maps=False,
                export_multi_product_heatmap=False,
            )

        self.assertTrue(report["target_extent_used"])
        self.assertLess(report["grid_size"]["width"] * report["grid_size"]["height"], 100)
        self.assertGreaterEqual(report["elapsed_seconds"], 0)


if __name__ == "__main__":
    unittest.main()
