# -*- coding: utf-8 -*-
"""Phase E+: 成果报告引擎（集成 E:\\Code\\pdf report_engine.py）单元测试。"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

_TF_AGENT = str(Path(__file__).resolve().parents[2] / "TF-agent")
if _TF_AGENT not in sys.path:
    sys.path.insert(0, _TF_AGENT)

import asset_report_engine as are


def _make_final_tif(path: Path, width: int = 40, height: int = 30) -> Path:
    """生成一个含少量「潮滩像元」的 final_*.tif 供统计/预览测试。"""
    import rasterio
    from rasterio.transform import from_origin

    arr = np.zeros((height, width), dtype="uint8")
    arr[5:15, 8:28] = 1  # 潮滩像元块
    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": 1,
        "dtype": "uint8",
        "crs": "EPSG:4326",  # 地理坐标 → 走 lat-cosine 面积估算分支
        "transform": from_origin(120.0, 31.0, 0.01, 0.01),
        "nodata": 0,
    }
    with rasterio.open(path, "w", **profile) as ds:
        ds.write(arr, 1)
    return path


@pytest.fixture()
def _tmp_registry(tmp_path, monkeypatch):
    """构造临时 assets_registry.json（含 20fujian1 的 Final TIF）。"""
    tif = _make_final_tif(tmp_path / "20fujian1_Final_p0.05_c3.tif")
    reg = {
        "20fujian1_p0.05_c3": {
            "task": "20fujian1",
            "prob_threshold": 0.05,
            "min_count": 3,
            "file_path": str(tif),
            "created_at": "2026-04-16 16:15:20",
            "file_size_mb": 0.01,
        },
        # 其他任务资产，应被过滤
        "20zhejiang1_p0.05_c3": {
            "task": "20zhejiang1",
            "prob_threshold": 0.05,
            "min_count": 3,
            "file_path": str(tif),
            "created_at": "2026-03-09 16:08:03",
            "file_size_mb": 0.01,
        },
    }
    reg_path = tmp_path / "assets_registry.json"
    reg_path.write_text(json.dumps(reg, ensure_ascii=False), encoding="utf-8")
    monkeypatch.setattr(are, "_default_registry_path", lambda: str(reg_path))
    return str(tmp_path)


def test_get_eligible_assets_filters_by_task(_tmp_registry):
    eligible = are.get_eligible_assets("20fujian1")
    assert set(eligible.keys()) == {"20fujian1_p0.05_c3"}


def test_get_eligible_assets_missing_file(tmp_path, monkeypatch):
    reg = tmp_path / "assets_registry.json"
    reg.write_text(
        json.dumps(
            {
                "k1": {
                    "task": "t1",
                    "file_path": str(tmp_path / "nope_final.tif"),
                    "created_at": "2026-01-01 00:00:00",
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(are, "_default_registry_path", lambda: str(reg))
    assert are.get_eligible_assets("t1") == {}


def test_get_eligible_assets_uses_companion_tif_for_registered_shp(tmp_path, monkeypatch):
    """深度学习历史记录登记 SHP 时，应自动使用同 stem 的 Final TIF。"""
    tif = _make_final_tif(tmp_path / "t1_Final_p0.15_c8.tif")
    shp = tmp_path / "t1_Final_p0.15_c8.shp"
    shp.write_bytes(b"placeholder")
    reg = tmp_path / "assets_registry.json"
    reg.write_text(
        json.dumps({
            "t1_p0.15_c8": {
                "task": "t1",
                "prob_threshold": 0.15,
                "min_count": 8,
                "file_path": str(shp),
                "created_at": "2026-01-01 00:00:00",
            }
        }),
        encoding="utf-8",
    )
    monkeypatch.setattr(are, "_default_registry_path", lambda: str(reg))

    eligible = are.get_eligible_assets("t1")
    assert set(eligible) == {"t1_p0.15_c8"}
    assert eligible["t1_p0.15_c8"]["file_path"] == str(tif)


def test_collect_extraction_summary_includes_complete_assets(tmp_path, monkeypatch):
    tif = _make_final_tif(tmp_path / "t1_Final_p0.15_c8.tif")
    shp = tmp_path / "t1_Final_p0.15_c8.shp"
    shp.write_bytes(b"placeholder")
    reg = tmp_path / "assets_registry.json"
    reg.write_text(json.dumps({
        "t1_p0.15_c8": {
            "task": "t1", "method": "deep_learning", "prob_threshold": 0.15,
            "min_count": 8, "file_path": str(shp),
            "created_at": "2026-01-01 00:00:00",
        }
    }), encoding="utf-8")
    monkeypatch.setattr(are, "_default_registry_path", lambda: str(reg))

    summary = are._collect_extraction_summary("t1")
    assert summary["status"] == "complete"
    assert summary["primary_tif"] == str(tif)
    assert summary["primary_shp"] == str(shp)
    assert summary["parameters"]["prob_threshold"] == 0.15
    assert summary["assets"][0]["format"] == "tif"


def test_get_eligible_assets_rejects_empty_final_file(tmp_path, monkeypatch):
    empty = tmp_path / "t1_Final_empty.tif"
    empty.touch()
    reg = tmp_path / "assets_registry.json"
    reg.write_text(
        json.dumps({
            "k1": {
                "task": "t1",
                "file_path": str(empty),
                "created_at": "2026-01-01 00:00:00",
            }
        }),
        encoding="utf-8",
    )
    monkeypatch.setattr(are, "_default_registry_path", lambda: str(reg))

    assert are.get_eligible_assets("t1") == {}


def test_corrupt_asset_registry_is_rejected_and_preserved(tmp_path, monkeypatch):
    reg = tmp_path / "assets_registry.json"
    reg.write_text('{"broken": [', encoding="utf-8")
    monkeypatch.setattr(are, "_default_registry_path", lambda: str(reg))
    result = are.generate_asset_report("t1")
    assert result.success is False
    assert "注册表" in result.error
    assert list(tmp_path.glob("assets_registry.json.corrupt-*"))


def test_generate_asset_report_success(_tmp_registry, tmp_path, monkeypatch):
    """完整链路：统计 + 7 页 PDF 生成成功。"""
    monkeypatch.setattr(are, "_report_dir", lambda: str(tmp_path))
    res = are.generate_asset_report("20fujian1")
    assert res.success is True
    assert res.error == ""
    assert res.report_path and os.path.isfile(res.report_path)
    assert os.path.getsize(res.report_path) > 0
    assert len(res.sections) == 7
    assert "参考真值对比" in res.sections


def test_generate_asset_report_progress_callback(_tmp_registry, tmp_path, monkeypatch):
    """进度回调被调用且覆盖 0..1。"""
    monkeypatch.setattr(are, "_report_dir", lambda: str(tmp_path))
    seen = []

    def cb(pct, msg):
        seen.append((pct, msg))

    res = are.generate_asset_report("20fujian1", progress_callback=cb)
    assert res.success is True
    assert seen and seen[-1][0] == 1.0
    assert any(p > 0.5 for p, _ in seen)


def test_monitor_report_includes_terminal_results(_tmp_registry, tmp_path, monkeypatch):
    """监测报告也应保留终端中的关键运行结果，便于复核提取过程。"""
    monkeypatch.setattr(are, "_report_dir", lambda: str(tmp_path))
    captured = {}
    original_render = are._render_pages

    def capture_render(*args, **kwargs):
        captured["terminal_logs"] = kwargs.get("terminal_logs")
        captured["execution_results"] = kwargs.get("execution_results")
        return original_render(*args, **kwargs)

    monkeypatch.setattr(are, "_render_pages", capture_render)
    res = are.generate_asset_report(
        "20fujian1",
        terminal_logs=[
            "📊 Top-10 参数组合排行榜:",
            "🏆 最优参数: prob=0.15, cnt=8 | IoU=72.97% F1=84.37%",
            "✅ 自适应优化完成！耗时 61.4s",
        ],
    )
    assert res.success is True
    assert captured["terminal_logs"][0].startswith("📊 Top-10")
    assert captured["execution_results"] is None


def test_dedupe_same_asset(_tmp_registry, tmp_path, monkeypatch):
    """同 task+asset+文件 mtime → 二次生成复用已有路径。"""
    monkeypatch.setattr(are, "_report_dir", lambda: str(tmp_path))
    r1 = are.generate_asset_report("20fujian1")
    r2 = are.generate_asset_report("20fujian1")
    assert r1.success and r2.success
    assert r1.report_path == r2.report_path
    assert any("已有" in w for w in r2.warnings)


def test_no_asset_failure(tmp_path, monkeypatch):
    """任务无资产 → 明确失败而非抛异常。"""
    monkeypatch.setattr(are, "_default_registry_path", lambda: str(tmp_path / "empty.json"))
    res = are.generate_asset_report("20nobody")
    assert res.success is False
    assert "无已入库" in (res.error or "")


def test_rasterio_missing(_tmp_registry, monkeypatch):
    """rasterio 缺失 → 明确失败。"""
    monkeypatch.setattr(are, "_HAS_RASTERIO", False)
    res = are.generate_asset_report("20fujian1")
    assert res.success is False
    assert "rasterio" in (res.error or "")


def test_matplotlib_missing(_tmp_registry, monkeypatch):
    monkeypatch.setattr(are, "_HAS_MATPLOTLIB", False)
    res = are.generate_asset_report("20fujian1")
    assert res.success is False
    assert "matplotlib" in (res.error or "")


def test_result_structure(_tmp_registry, tmp_path, monkeypatch):
    monkeypatch.setattr(are, "_report_dir", lambda: str(tmp_path))
    res = are.generate_asset_report("20fujian1")
    for f in ("success", "task_id", "report_path", "sections", "warnings", "error"):
        assert hasattr(res, f), f"缺少字段 {f}"
    assert isinstance(res.warnings, list)
    assert isinstance(res.sections, list)


def test_no_abs_path_leak(_tmp_registry, tmp_path, monkeypatch):
    """报告中不泄露盘符绝对路径（sanitize 生效）。"""
    monkeypatch.setattr(are, "_report_dir", lambda: str(tmp_path))
    assert are._safe_basename("Z:/data/20fujian1_Final.tif") == "20fujian1_Final.tif"
    assert "Z:" not in are._safe_basename("Z:/a/b.tif")
    assert are._sanitize_text("token=sk-secret") == "[已过滤]"
    assert are._sanitize_key("ion_token") is False
    res = are.generate_asset_report("20fujian1")
    assert res.success is True
    assert "Z:" not in (res.error or "")


def test_report_error_sanitizes_posix_path(_tmp_registry, tmp_path, monkeypatch):
    """底层栅格异常进入结果摘要前必须移除路径和凭据。"""
    monkeypatch.setattr(are, "_report_dir", lambda: str(tmp_path))
    monkeypatch.setattr(
        are,
        "compute_raster_stats",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("failed /Users/chl/private/result.tif token=sk-report-secret")
        ),
    )
    res = are.generate_asset_report("20fujian1")
    assert res.success is False
    assert "/Users/" not in res.error
    assert "sk-report-secret" not in res.error


def test_safe_filename_sanitizes():
    assert are._safe_filename("20fujian1/p0.05..c3") == "20fujian1_p0.05..c3"
    # 路径穿越片段：不以 .. 开头、无路径分隔符、不含盘符冒号
    out = are._safe_filename("..\\..\\evil")
    assert not out.startswith("..")
    assert "/" not in out and "\\" not in out
    assert ":" not in are._safe_filename("Z:\\evil")


def test_compute_raster_stats(_tmp_registry):
    tif = str(Path(_tmp_registry) / "20fujian1_Final_p0.05_c3.tif")
    stats = are.compute_raster_stats(tif)
    assert stats["tidal_pixels"] == 10 * 20  # 10 行 x 20 列
    assert stats["area_km2"] > 0
    assert stats["area_estimated"] is True  # EPSG:4326 → 近似估算
    assert 0.0 < stats["coverage_pct"] < 100.0


def test_render_tif_preview(_tmp_registry):
    tif = str(Path(_tmp_registry) / "20fujian1_Final_p0.05_c3.tif")
    img = are.render_tif_preview(tif)
    assert img.shape[2] == 3
    assert img.dtype == np.uint8
    # 背景为暗色 [28,34,42]（RGB 和 104），预测为红色 [220,62,52]
    assert (img.sum(axis=2) == 104).any()
    assert (img[..., 0] == 220).any()
