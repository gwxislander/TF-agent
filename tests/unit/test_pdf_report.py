# -*- coding: utf-8 -*-
"""Phase E: PDF 报告适配器单元测试（10 项）。"""
import os
import sys
from pathlib import Path

import pytest

_TF_AGENT = str(Path(__file__).resolve().parents[2] / "TF-agent")
if _TF_AGENT not in sys.path:
    sys.path.insert(0, _TF_AGENT)

import report_generator as rg


def _task_ctx(task_id="t1", **kw):
    ctx = {
        "task_id": task_id,
        "task": "水华识别",
        "mode": "dl",
        "prob": 0.5,
        "cnt": 3,
        "plan_id": "plan-1",
    }
    ctx.update(kw)
    return ctx


def _timeline():
    from task_timeline import TimelineEvent

    return [
        TimelineEvent(
            task_id="t1", phase="PLAN", message="计划生成", status="SUCCEEDED",
            progress=100, details={}, artifacts=[],
        ),
        TimelineEvent(
            task_id="t1", phase="EXECUTE", message="执行完成", status="SUCCEEDED",
            progress=100, details={}, artifacts=["result.tif"],
        ),
    ]


@pytest.fixture()
def _tmp_report_dir(monkeypatch, tmp_path):
    monkeypatch.setattr(rg, "_report_dir", lambda: str(tmp_path))
    return tmp_path


def test_report_created_nonempty(_tmp_report_dir):
    """成功生成：文件存在且非空，sections 齐全。"""
    res = rg.generate_task_report(
        _task_ctx(),
        capabilities={"deep_learning_inference": {"status": "AVAILABLE", "summary": "可用"}},
        timeline=_timeline(),
        assets=[{"path": "result.tif", "kind": "raster"}],
    )
    assert res.success is True
    assert res.error == ""
    assert res.report_path and os.path.isfile(res.report_path)
    assert os.path.getsize(res.report_path) > 0
    assert set(res.sections) >= {"基本信息", "能力状态", "执行时间线", "资产清单", "地图截图"}


def test_reportlab_missing(_tmp_report_dir, monkeypatch):
    monkeypatch.setattr(rg, "_HAS_REPORTLAB", False)
    res = rg.generate_task_report(_task_ctx())
    assert res.success is False
    assert "reportlab" in (res.error or "")


def test_dedupe_same_task(_tmp_report_dir):
    """同 task_id+配置：二次生成返回已有路径。"""
    r1 = rg.generate_task_report(_task_ctx(), timeline=_timeline())
    r2 = rg.generate_task_report(_task_ctx(), timeline=_timeline())
    assert r1.success and r2.success
    assert r1.report_path == r2.report_path
    assert any("已有" in w for w in r2.warnings)


def test_screenshot_failure_degrades(_tmp_report_dir):
    """截图损坏 → warning，报告仍生成成功。"""
    res = rg.generate_task_report(
        _task_ctx(), timeline=_timeline(), map_snapshot=b"not-an-image-bytes",
    )
    assert res.success is True
    assert any("截图" in w for w in res.warnings)


def test_cjk_font_missing_warning(_tmp_report_dir, monkeypatch):
    monkeypatch.setattr(rg, "_find_cjk_font", lambda: None)
    res = rg.generate_task_report(_task_ctx(), timeline=_timeline())
    assert res.success is True
    assert any("中文字体" in w for w in res.warnings)


def test_no_token_leak(_tmp_report_dir):
    """任务上下文中带密钥 → 不落入报告内容。"""
    ctx = _task_ctx(task_id="t-sec")
    ctx["ion_token"] = "sk-super-secret-abc123"
    assert rg._sanitize_key("ion_token") is False
    assert rg._sanitize_text("token=sk-super-secret-abc123") == "[已过滤]"
    res = rg.generate_task_report(ctx, timeline=_timeline())
    assert res.success is True
    assert "sk-super-secret" not in (res.error or "")


def test_no_abs_path_in_assets(_tmp_report_dir):
    """资产含本地绝对路径 → 转相对/basename，报告不含盘符。"""
    rel = rg._relative_path("Z:/data/result.tif")
    assert "Z:" not in rel
    assert rel == "result.tif"
    res = rg.generate_task_report(
        _task_ctx(), timeline=_timeline(),
        assets=[{"path": "Z:/data/result.tif", "kind": "raster"}],
    )
    assert res.success is True


def test_report_accepts_complete_extraction_asset_metadata(_tmp_report_dir):
    """成果报告应保留 TIF/SHP、参数、状态和大小等提取结果摘要。"""
    res = rg.generate_task_report(
        _task_ctx(),
        timeline=_timeline(),
        assets=[{
            "path": "/tmp/task_Final_p0.15_c8.tif", "kind": "prediction",
            "status": "verified", "file_size_mb": 12.5,
            "prob_threshold": 0.15, "min_count": 8,
            "model_id": "cdnet_resnet50", "weight_id": "best.pth",
        }],
    )
    assert res.success is True


def test_report_includes_terminal_results_and_structured_autotune_metrics(_tmp_report_dir):
    """报告应携带终端关键结果及 AutoTune 的最佳参数/Top-10 指标。"""
    terminal_logs = [
        "📊 Top-10 参数组合排行榜:",
        "  1  0.15  8  72.97  84.37  79.63  89.72 🏆",
        "🏆 最优参数: prob=0.15, cnt=8",
        "   IoU=72.97% F1=84.37% Precision=79.63% Recall=89.72%",
        "✅ 自适应优化完成！耗时 61.4s",
    ]
    execution_results = {
        "autotune": {
            "best_prob": 0.15,
            "best_cnt": 8,
            "best_iou": 0.7297,
            "best_f1": 0.8437,
            "best_precision": 0.7963,
            "best_recall": 0.8972,
            "total_trials": 500,
            "total_time_sec": 61.4,
            "trials": [
                {
                    "prob": 0.15, "cnt": 8, "iou": 0.7297, "f1": 0.8437,
                    "precision": 0.7963, "recall": 0.8972,
                },
            ],
        },
    }
    res = rg.generate_task_report(
        _task_ctx(task_id="terminal-results"),
        timeline=_timeline(),
        terminal_logs=terminal_logs,
        execution_results=execution_results,
    )
    assert res.success is True
    assert "终端运行结果" in res.sections
    result_text = "\n".join(rg._execution_result_lines(execution_results))
    assert "Top-10" in result_text
    assert "0.15" in result_text
    assert "72.97%" in result_text
    assert "84.37%" in result_text
    assert "61.4" in result_text
    normalized_logs = rg._normalize_terminal_logs(terminal_logs)
    assert normalized_logs[0].startswith("Top-10")
    assert normalized_logs[-1].endswith("耗时 61.4s")


def test_execution_result_lines_include_nested_m5_and_e1_metrics():
    """变化分析/精度评价的嵌套指标也应进入报告摘要。"""
    lines = rg._execution_result_lines({
        "m5": {
            "alert_level": "YELLOW",
            "diagnostic_message": "面积变化需复核",
            "quantitative_metrics": {
                "area_evolution": {
                    "baseline_area_km2": 12.3,
                    "current_area_km2": 13.1,
                    "change_rate_percentage": 6.5,
                },
                "centroid_trajectory": {
                    "drift_distance_meters": 84.2,
                    "migration_azimuth_degrees": 31.0,
                },
            },
        },
        "e1": {
            "reference": "师姐_2020",
            "comparisons": {
                "FCS30": {"jaccard_iou": 0.42, "intersection_km2": 2.1},
            },
        },
    })
    result_text = "\n".join(lines)
    assert "基线=12.3 km²" in result_text
    assert "变化率=6.5%" in result_text
    assert "IoU=42.00%" in result_text
    assert "师姐_2020" in result_text


def test_posix_path_and_spatial_metadata_are_redacted():
    """报告文本不得持久化 POSIX 路径或精确空间字段。"""
    text = rg._sanitize_text("failed /Users/chl/private/result.tif bbox=[120,30,120.1,30.1]")
    assert "/Users/" not in text
    assert "result.tif" not in text
    assert "bbox=" not in text


def test_report_text_escapes_markup_before_reportlab_rendering():
    text = rg._sanitize_text("<b>用户输入</b> & 结果")
    assert "<b>" not in text
    assert "&lt;b&gt;用户输入&lt;/b&gt; &amp; 结果" == text


def test_empty_timeline_warning(_tmp_report_dir):
    """时间线为空 → warning，报告仍生成。"""
    res = rg.generate_task_report(_task_ctx(), timeline=None)
    assert res.success is True
    assert any("时间线为空" in w for w in res.warnings)


def test_generation_error_is_sanitized(_tmp_report_dir, monkeypatch):
    def fail_render(*_args, **_kwargs):
        raise RuntimeError("failed /Users/chl/private/report.pdf token=sk-report-secret")

    monkeypatch.setattr(rg, "_render_pdf", fail_render)
    res = rg.generate_task_report(_task_ctx(), timeline=_timeline())
    assert res.success is False
    assert "/Users/" not in res.error
    assert "sk-report-secret" not in res.error


def test_result_structure(_tmp_report_dir):
    res = rg.generate_task_report(_task_ctx(), timeline=_timeline())
    for f in ("success", "task_id", "report_path", "sections", "warnings", "error"):
        assert hasattr(res, f), f"缺少字段 {f}"
    assert isinstance(res.warnings, list)
    assert isinstance(res.sections, list)


def test_config_hash_changes(_tmp_report_dir):
    """不同配置 → 不同报告文件。"""
    r1 = rg.generate_task_report(_task_ctx(task_id="t-a", prob=0.5), timeline=_timeline())
    r2 = rg.generate_task_report(_task_ctx(task_id="t-a", prob=0.8), timeline=_timeline())
    assert r1.success and r2.success
    assert r1.report_path != r2.report_path
