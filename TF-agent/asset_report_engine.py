# -*- coding: utf-8 -*-
"""
Phase E+（集成自同门 E:\\Code\\pdf 的 report_engine.py，经安全与工程化改造）。

- matplotlib PdfPages 渲染 A4 横版多页「成果报告」（标题/概览/分类统计/空间预览/变化分析/参考真值对比/结论）。
- 数据源：TF-agent assets_registry.json 中已入库 Final TIF（栅格统计 + 空间预览 + 参考 SHP 对比 IoU/P/R/F1）。
- 安全：不打印绝对路径（一律 basename）、敏感 key/value 消毒、任务名与资产 key 消毒后入文件名。
- 健壮：依赖缺失（rasterio/matplotlib/geopandas）或参考真值缺失 → 返回 warning 并继续；绝不抛异常。
- 去重：同 task + asset_key + 资产文件变更时间 → 复用已有报告（与任务报告行为一致）。
- 进度回调 progress_callback(pct, message) 可选。
"""
from __future__ import annotations

import datetime as dt
import hashlib
import importlib.util
import json
import math
import os
import re
import tempfile
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from agent_context_policy import redact_spatial_metadata, safe_error_summary, sanitize_external_text

# ---- 可选依赖探测（缺失时报告生成失败并给出明确 warning，而非栈崩溃）----
_HAS_RASTERIO = importlib.util.find_spec("rasterio") is not None
_HAS_MATPLOTLIB = importlib.util.find_spec("matplotlib") is not None
_HAS_GEOPANDAS = importlib.util.find_spec("geopandas") is not None

if _HAS_RASTERIO:
    import rasterio
    from rasterio.enums import Resampling

# ---- 常量 ----
PAGE_SIZE = (11.69, 8.27)  # A4 landscape
INK = "#1f2933"
MUTED = "#667085"
BLUE = "#2563eb"
RED = "#dc3e34"
DARK = "#333c4a"

_REPORT_DIRNAME = "data/reports"
_SENSITIVE_KEY_SUBSTRINGS = ("token", "secret", "password", "api_key", "ion", "key")
_SENSITIVE_VALUE_SUBSTRINGS = ("Z:/", "C:\\", "/home/", "token=", "key=", "sk-")
_SECTIONS = ("标题页", "成果概览", "分类统计", "空间预览", "变化分析", "参考真值对比", "结论摘要")


def _default_registry_path() -> str:
    """默认使用 TF-agent 根目录下的 assets_registry.json（成果资产注册表）。"""
    base = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base, "assets_registry.json")


def _report_dir() -> str:
    base = os.path.dirname(os.path.abspath(__file__))
    d = os.path.join(base, _REPORT_DIRNAME)
    try:
        os.makedirs(d, exist_ok=True)
        return d
    except Exception:
        return tempfile.gettempdir()


@dataclass
class AssetReportResult:
    success: bool
    task_id: str
    report_path: Optional[str] = None
    sections: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    error: str = ""


# ---- 安全消毒（与 report_generator 同策略）----
def _sanitize_text(text: Any) -> str:
    s = str(text)
    low = s.lower()
    if any(sub in low for sub in _SENSITIVE_VALUE_SUBSTRINGS):
        return "[已过滤]"
    s = redact_spatial_metadata(sanitize_external_text(s))
    s = re.sub(r"(?i)\bsk-[A-Za-z0-9_-]+", "<redacted>", s)
    return s[:400]


def _sanitize_key(name: str) -> bool:
    low = (name or "").lower()
    return not any(sub in low for sub in _SENSITIVE_KEY_SUBSTRINGS)


def _safe_basename(path: Any) -> str:
    """仅返回 basename，绝不泄露本地绝对路径。"""
    p = str(path or "")
    return os.path.basename(p.replace("\\", "/")) or "<local-path>"


def _nonempty_file(path: Any) -> bool:
    """Return true only for an existing artifact with non-zero size."""
    try:
        return bool(path) and os.path.isfile(str(path)) and os.path.getsize(str(path)) > 0
    except OSError:
        return False


def _resolve_registered_path(path: Any, registry_path: Optional[str] = None) -> str:
    """Resolve registry paths without depending on the process working directory."""
    raw = str(path or "").strip().strip('"').strip("'")
    if not raw:
        return ""
    if not os.path.isabs(raw):
        registry = registry_path or _default_registry_path()
        raw = os.path.join(os.path.dirname(os.path.abspath(registry)), raw)
    return os.path.normpath(os.path.abspath(raw))


def _registered_final_tif(row: Dict[str, Any], registry_path: Optional[str] = None) -> str:
    """Return a usable Final TIF, including one paired with a registered Final SHP."""
    raw = str(row.get("file_path") or "")
    candidates = []
    declared_tif = row.get("final_tif")
    if declared_tif:
        candidates.append(declared_tif)
    resolved = _resolve_registered_path(raw, registry_path)
    if resolved:
        candidates.append(resolved)
        if resolved.lower().endswith(".shp"):
            candidates.append(os.path.splitext(resolved)[0] + ".tif")
    for candidate in candidates:
        resolved_candidate = _resolve_registered_path(candidate, registry_path)
        if _is_final_tif(resolved_candidate) and _nonempty_file(resolved_candidate):
            return resolved_candidate
    return ""


def _safe_filename(part: str) -> str:
    s = re.sub(r"[^\w\u4e00-\u9fff.\-]", "_", str(part or ""), flags=re.UNICODE)
    s = re.sub(r"_+", "_", s).strip("_")
    # 防路径穿越：删除前导点/下划线片段与盘符冒号
    s = re.sub(r"^[\._]+(?:\.[\._]*)*", "", s)
    s = s.replace(":", "_")
    return (s or "asset")[:60]


# ---- 资产发现 ----
def _load_asset_registry(registry_path: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
    from workflow_orchestrator import load_assets_registry

    return load_assets_registry(registry_path or _default_registry_path())


def _is_final_tif(path: str) -> bool:
    name = os.path.basename(path or "").lower()
    return name.endswith((".tif", ".tiff")) and "final" in name


def get_eligible_assets(
    task: str, registry_path: Optional[str] = None
) -> Dict[str, Dict[str, Any]]:
    """返回某任务下已入库且文件仍存在的 Final TIF（按入库时间倒序）。"""
    out: Dict[str, Dict[str, Any]] = {}
    for key, row in _load_asset_registry(registry_path).items():
        tif_path = _registered_final_tif(row, registry_path)
        if row.get("task") == task and tif_path:
            item = dict(row)
            item["file_path"] = tif_path
            out[key] = item
    return dict(
        sorted(out.items(), key=lambda kv: str(kv[1].get("created_at") or ""), reverse=True)
    )


def _collect_extraction_summary(task: str, registry_path: Optional[str] = None) -> Dict[str, Any]:
    """Collect a complete, report-safe extraction summary from registered artifacts."""
    registry = registry_path or _default_registry_path()
    rows: List[Dict[str, Any]] = []
    primary_tif = ""
    primary_shp = ""
    expects_shp = False
    params: Dict[str, Any] = {}
    warnings: List[str] = []
    try:
        registered = _load_asset_registry(registry)
    except Exception as exc:
        return {"assets": [], "warnings": [f"资产注册表读取失败: {_sanitize_text(exc)}"]}

    for key, row in registered.items():
        if not isinstance(row, dict) or row.get("task") != task:
            continue
        raw_path = str(row.get("file_path") or "")
        resolved_path = _resolve_registered_path(raw_path, registry)
        tif_path = _registered_final_tif(row, registry)
        shp_path = resolved_path if resolved_path.lower().endswith(".shp") and _nonempty_file(resolved_path) else ""
        expects_shp = expects_shp or resolved_path.lower().endswith(".shp") or bool(row.get("final_shp"))
        declared_shp = row.get("final_shp")
        if not shp_path and declared_shp:
            candidate = _resolve_registered_path(declared_shp, registry)
            if candidate.lower().endswith(".shp") and _nonempty_file(candidate):
                shp_path = candidate
        if tif_path and not primary_tif:
            primary_tif = tif_path
        if shp_path and not primary_shp:
            primary_shp = shp_path
        for field in ("prob_threshold", "min_count", "model_id", "weight_id", "device"):
            if row.get(field) is not None and field not in params:
                params[field] = row.get(field)
        rows.append({
            "key": str(key),
            "method": str(row.get("method") or row.get("asset_type") or "unknown"),
            "file": _safe_basename(tif_path or shp_path or resolved_path),
            "format": "tif" if tif_path else ("shp" if shp_path else "unknown"),
            "status": str(row.get("status") or "registered"),
            "created_at": str(row.get("created_at") or ""),
        })

    if not rows:
        warnings.append("该任务在 assets_registry.json 中没有登记资产")
    if not primary_tif:
        warnings.append("缺少可读取的 Final TIF，无法完成栅格统计")
    if expects_shp and not primary_shp:
        warnings.append("缺少可读取的 Final SHP，矢量成果清单不完整")
    return {
        "assets": rows,
        "primary_tif": primary_tif,
        "primary_shp": primary_shp,
        "parameters": params,
        "warnings": warnings,
        "status": "complete" if primary_tif and (primary_shp or not expects_shp) else "partial",
    }


# ---- 栅格统计 ----
def _task_year(task: str) -> Optional[int]:
    m = re.search(r"(20\d{2})", task or "")
    if m:
        return int(m.group(1))
    m = re.match(r"^(\d{2})", task or "")
    if m:
        yy = int(m.group(1))
        if 0 <= yy <= 80:
            return 2000 + yy
    return None


def _pixel_area_m2(ds: Any) -> Tuple[float, bool]:
    tr = ds.transform
    if ds.crs and not ds.crs.is_geographic:
        return abs(float(tr.a) * float(tr.e)), False
    mid_lat = (ds.bounds.top + ds.bounds.bottom) / 2.0
    m_per_deg_lon = 111320.0 * max(math.cos(math.radians(mid_lat)), 0.01)
    m_per_deg_lat = 110574.0
    return abs(float(tr.a) * m_per_deg_lon * float(tr.e) * m_per_deg_lat), True


def _read_prediction_mask(tif_path: str) -> Tuple[np.ndarray, Dict[str, Any]]:
    with rasterio.open(tif_path) as ds:
        arr = ds.read(1, masked=True)
        data = np.ma.filled(arr, 0)
        valid = ~np.ma.getmaskarray(arr)
        if ds.nodata is not None:
            valid &= data != ds.nodata
        pred = valid & np.isfinite(data) & (data > 0)
        pixel_area, estimated = _pixel_area_m2(ds)
        meta = {
            "width": ds.width,
            "height": ds.height,
            "count": ds.count,
            "crs": str(ds.crs) if ds.crs else "Unknown",
            "bounds": ds.bounds,
            "transform": ds.transform,
            "pixel_area_m2": pixel_area,
            "area_estimated": estimated,
            "resolution": (float(abs(ds.transform.a)), float(abs(ds.transform.e))),
            "nodata": ds.nodata,
            "dtype": str(ds.dtypes[0]),
        }
    return pred, meta


def compute_raster_stats(tif_path: str) -> Dict[str, Any]:
    pred, meta = _read_prediction_mask(tif_path)
    total_pixels = int(pred.size)
    tidal_pixels = int(pred.sum())
    area_m2 = tidal_pixels * float(meta["pixel_area_m2"])
    return {
        **meta,
        "total_pixels": total_pixels,
        "valid_pixels": total_pixels,
        "tidal_pixels": tidal_pixels,
        "background_pixels": total_pixels - tidal_pixels,
        "coverage_pct": (tidal_pixels / total_pixels * 100.0) if total_pixels else 0.0,
        "area_m2": area_m2,
        "area_km2": area_m2 / 1_000_000.0,
    }


def render_tif_preview(tif_path: str, max_side: int = 1024) -> np.ndarray:
    with rasterio.open(tif_path) as ds:
        scale = min(1.0, max_side / max(ds.width, ds.height))
        out_h = max(1, int(ds.height * scale))
        out_w = max(1, int(ds.width * scale))
        data = ds.read(1, out_shape=(out_h, out_w), masked=True, resampling=Resampling.nearest)
    arr = np.ma.filled(data, 0)
    pred = np.isfinite(arr) & (arr > 0)
    img = np.zeros((arr.shape[0], arr.shape[1], 3), dtype=np.uint8)
    img[..., :] = np.array([28, 34, 42], dtype=np.uint8)
    img[pred] = np.array([220, 62, 52], dtype=np.uint8)
    return img


# ---- 参考真值对比 ----
def _find_reference_shp(task: str) -> Optional[Dict[str, Any]]:
    try:
        from dataset_assets import list_datasets
    except Exception:
        return None
    refs = list_datasets(role="reference_truth", format="shapefile", require_file_exists=True)
    if not refs:
        return None
    year = _task_year(task)
    if year is None:
        return refs[-1]
    return min(refs, key=lambda r: abs(int(r.get("year") or year) - year))


def _compare_reference(tif_path: str, ref: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not ref:
        return None
    shp_path = ref.get("_resolved_path") or ref.get("primary_path")
    if not shp_path or not os.path.isfile(shp_path):
        return None
    try:
        import geopandas as gpd
        from rasterio.features import rasterize
    except Exception:
        return None

    try:
        pred, _meta = _read_prediction_mask(tif_path)
        with rasterio.open(tif_path) as ds:
            gdf = gpd.read_file(shp_path)
            if gdf.empty:
                return None
            if ds.crs and gdf.crs and str(gdf.crs) != str(ds.crs):
                gdf = gdf.to_crs(ds.crs)
            geoms = [g for g in gdf.geometry if g is not None and not g.is_empty]
            if not geoms:
                return None
            ref_mask = rasterize(
                [(g, 1) for g in geoms],
                out_shape=(ds.height, ds.width),
                transform=ds.transform,
                fill=0,
                dtype="uint8",
            ).astype(bool)
    except Exception as exc:
        return {"error": safe_error_summary(exc), "reference_title": ref.get("title") or ref.get("id")}

    tp = int(np.logical_and(pred, ref_mask).sum())
    fp = int(np.logical_and(pred, ~ref_mask).sum())
    fn = int(np.logical_and(~pred, ref_mask).sum())
    union = tp + fp + fn
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "reference_title": ref.get("title") or ref.get("id"),
        "reference_year": ref.get("year"),
        "intersection_pixels": tp,
        "pred_only_pixels": fp,
        "ref_only_pixels": fn,
        "iou": tp / union if union else 0.0,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


# ---- matplotlib 渲染 ----
def _configure_matplotlib():
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib import font_manager, rcParams

    preferred = [
        "Microsoft YaHei", "SimHei", "Arial Unicode MS",
        "PingFang SC", "Hiragino Sans GB", "STHeiti",
    ]
    installed = {f.name for f in font_manager.fontManager.ttflist}
    for name in preferred:
        if name in installed:
            rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            break
    rcParams["axes.unicode_minus"] = False
    return plt


def _new_page(plt, title: str, page: str):
    fig = plt.figure(figsize=PAGE_SIZE)
    fig.patch.set_facecolor("white")
    fig.text(0.06, 0.94, title, fontsize=18, weight="bold", color=INK, va="top")
    fig.add_artist(plt.Line2D([0.06, 0.94], [0.90, 0.90], color=BLUE, linewidth=1.6))
    fig.text(0.06, 0.035, "CSTF Regional Tidal Flat Monitoring Report", fontsize=8, color=MUTED)
    fig.text(0.94, 0.035, f"Page {page}", ha="right", fontsize=8, color=MUTED)
    return fig


def _style_table(table, header_color: str = "#eef4ff") -> None:
    for (row, _col), cell in table.get_celld().items():
        cell.set_edgecolor("#d0d5dd")
        cell.set_linewidth(0.6)
        if row == 0:
            cell.set_facecolor(header_color)
            cell.set_text_props(weight="bold", color=INK)
        else:
            cell.set_facecolor("#ffffff" if row % 2 else "#f8fafc")
            cell.set_text_props(color=INK)


def _metric(fig, x: float, y: float, label: str, value: str, accent: str = BLUE) -> None:
    import matplotlib.patches as patches

    ax = fig.add_axes([x, y, 0.25, 0.12])
    ax.axis("off")
    ax.add_patch(
        patches.FancyBboxPatch(
            (0, 0),
            1,
            1,
            boxstyle="round,pad=0.015,rounding_size=0.025",
            facecolor="#f8fafc",
            edgecolor="#d0d5dd",
            linewidth=0.8,
        )
    )
    ax.text(0.06, 0.68, label, fontsize=9, color=MUTED, va="center")
    ax.text(0.06, 0.30, value, fontsize=16, color=accent, weight="bold", va="center")


def _fmt_bounds(bounds: Any) -> str:
    return f"{bounds.left:.5f}, {bounds.bottom:.5f}, {bounds.right:.5f}, {bounds.top:.5f}"


def _terminal_report_lines(terminal_logs: Any = None, execution_results: Any = None) -> List[str]:
    """Reuse the task-report sanitizer/formatter for the monitoring PDF."""
    try:
        from report_generator import _execution_result_lines, _normalize_terminal_logs

        lines = list(_execution_result_lines(execution_results))
        lines.extend(_normalize_terminal_logs(terminal_logs))
    except Exception:
        lines = []
        if isinstance(terminal_logs, str):
            lines = terminal_logs.splitlines()
        elif isinstance(terminal_logs, (list, tuple)):
            lines = [str(item or "") for item in terminal_logs]
    return [_sanitize_text(line)[:220] for line in lines if str(line or "").strip()]


def _render_pages(
    pdf,
    plt,
    task: str,
    asset_key: str,
    stats: Dict[str, Any],
    preview: np.ndarray,
    ref_cmp: Optional[Dict[str, Any]],
    extraction_summary: Optional[Dict[str, Any]],
    progress,
    terminal_logs: Any = None,
    execution_results: Any = None,
) -> None:
    """按 7 章节渲染报告页（与同门 report_engine 结构一致，文本全部消毒）。"""
    progress(0.55, "写入标题页")
    fig = _new_page(plt, "区域潮滩监测报告", "1")
    fig.text(0.06, 0.76, _sanitize_text(task), fontsize=30, weight="bold", color=INK)
    fig.text(0.06, 0.68, _safe_basename(asset_key), fontsize=13, color=MUTED)
    fig.text(
        0.06, 0.60,
        f"生成时间  {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        fontsize=11, color=INK,
    )
    fig.text(
        0.06, 0.54,
        "数据来源  assets_registry.json 中已入库 Final TIF；报告生成不触发推理。",
        fontsize=11, color=INK,
    )
    _metric(fig, 0.06, 0.34, "潮滩面积", f"{stats['area_km2']:.4f} km²", RED)
    _metric(fig, 0.36, 0.34, "覆盖比例", f"{stats['coverage_pct']:.2f}%")
    _metric(fig, 0.66, 0.34, "潮滩像元", f"{stats['tidal_pixels']:,}")
    fig.text(
        0.06, 0.18,
        "本报告用于成果归档、快速质检和初步变化分析。正式结论建议结合多期同源数据与人工复核。",
        fontsize=10, color=MUTED,
    )
    pdf.savefig(fig)
    plt.close(fig)

    progress(0.62, "写入成果概览页")
    fig = _new_page(plt, "成果概览与基础统计", "2")
    ax = fig.add_axes([0.06, 0.15, 0.88, 0.68])
    ax.axis("off")
    rows = [
        ["提取状态", _sanitize_text((extraction_summary or {}).get("status") or "unknown")],
        ["文件", _safe_basename(stats.get("_tif_name") or "")],
        ["CRS", _sanitize_text(stats["crs"])],
        ["尺寸", f"{stats['width']} x {stats['height']}"],
        ["分辨率", f"{stats['resolution'][0]:.6g}, {stats['resolution'][1]:.6g}"],
        ["Bounds", _fmt_bounds(stats["bounds"])],
        ["数据类型 / NoData", f"{stats['dtype']} / {stats['nodata']}"],
        ["总像元", f"{stats['total_pixels']:,}"],
        ["潮滩像元", f"{stats['tidal_pixels']:,}"],
        ["潮滩面积", f"{stats['area_km2']:.4f} km²"],
        ["覆盖比例", f"{stats['coverage_pct']:.2f}%"],
    ]
    summary = extraction_summary or {}
    if summary.get("primary_shp"):
        rows.insert(2, ["Final SHP", _safe_basename(summary["primary_shp"])])
    p = summary.get("parameters") or {}
    if p:
        rows.append(["提取参数", _sanitize_text(", ".join(f"{k}={v}" for k, v in p.items()))])
    rows.append(["登记资产数", str(len(summary.get("assets") or []))])
    for warning in (summary.get("warnings") or []):
        rows.append(["完整性提示", _sanitize_text(warning)])
    if stats["area_estimated"]:
        rows.append(["面积备注", "CRS 为经纬度，面积按像元中心纬度近似估算"])
    table = ax.table(
        cellText=rows, colLabels=["项目", "值"], loc="center", cellLoc="left",
        colWidths=[0.24, 0.72],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9.5)
    table.scale(1, 1.42)
    _style_table(table)
    pdf.savefig(fig)
    plt.close(fig)

    progress(0.70, "写入分类统计图表")
    fig = _new_page(plt, "分类统计", "3")
    axes = [fig.add_axes([0.08, 0.18, 0.38, 0.62]), fig.add_axes([0.56, 0.20, 0.34, 0.55])]
    labels = ["潮滩", "背景"]
    vals = [stats["tidal_pixels"], stats["background_pixels"]]
    axes[0].pie(
        vals, labels=labels, autopct="%1.1f%%", colors=[RED, DARK],
        startangle=90, wedgeprops={"linewidth": 1, "edgecolor": "white"},
    )
    axes[0].set_title("像元占比")
    axes[1].bar(labels, vals, color=[RED, DARK], width=0.55)
    axes[1].set_title("像元数量")
    axes[1].set_ylabel("Pixels")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].spines[["top", "right"]].set_visible(False)
    pdf.savefig(fig)
    plt.close(fig)

    progress(0.78, "写入空间结果预览")
    fig = _new_page(plt, "空间结果预览", "4")
    ax = fig.add_axes([0.08, 0.14, 0.84, 0.70])
    ax.imshow(preview)
    ax.axis("off")
    fig.text(
        0.08, 0.10,
        "红色表示预测潮滩区域，深色表示背景或无效区域。预览图已按最长边 1024 像素降采样。",
        fontsize=10, color=MUTED,
    )
    pdf.savefig(fig)
    plt.close(fig)

    progress(0.84, "写入变化分析说明")
    fig = _new_page(plt, "变化分析", "5")
    fig.text(
        0.08, 0.70,
        "当前版本不将同一任务下不同阈值或计数参数的 Final TIF 视为真实时序变化。",
        fontsize=12, color=INK,
    )
    fig.text(
        0.08, 0.63,
        "如需变化分析，请接入同一区域、不同年份且来源一致的历史成果，再计算面积差异和空间转移。",
        fontsize=12, color=INK,
    )
    _metric(fig, 0.08, 0.42, "当前成果潮滩面积", f"{stats['area_km2']:.4f} km²", RED)
    assets = (extraction_summary or {}).get("assets") or []
    if assets:
        names = "；".join(f"{a.get('file')} ({a.get('format')})" for a in assets[:6])
        fig.text(0.08, 0.30, f"本次提取资产：{_sanitize_text(names)}", fontsize=10, color=INK)
    pdf.savefig(fig)
    plt.close(fig)

    progress(0.90, "写入参考真值对比")
    fig = _new_page(plt, "参考真值对比", "6")
    if ref_cmp and "error" not in ref_cmp:
        rows = [
            ["参考数据", _sanitize_text(ref_cmp.get("reference_title") or "")],
            ["年份", str(ref_cmp.get("reference_year") or "未标注")],
            ["IoU", f"{ref_cmp['iou'] * 100:.2f}%"],
            ["Precision", f"{ref_cmp['precision'] * 100:.2f}%"],
            ["Recall", f"{ref_cmp['recall'] * 100:.2f}%"],
            ["F1", f"{ref_cmp['f1'] * 100:.2f}%"],
        ]
        ax = fig.add_axes([0.08, 0.20, 0.84, 0.56])
        ax.axis("off")
        table = ax.table(
            cellText=rows, colLabels=["指标", "值"], loc="center", cellLoc="left",
            colWidths=[0.26, 0.68],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1, 1.8)
        _style_table(table)
    elif ref_cmp and ref_cmp.get("error"):
        fig.text(
            0.08, 0.68,
            f"参考真值读取或栅格化失败，已跳过: {_sanitize_text(ref_cmp['error'])}",
            fontsize=11, color=INK,
        )
    else:
        fig.text(0.08, 0.68, "未配置可用参考真值 SHP，已跳过该增强章节。", fontsize=12, color=INK)
    pdf.savefig(fig)
    plt.close(fig)

    progress(0.96, "写入结论摘要")
    fig = _new_page(plt, "结论摘要", "7")
    fig.text(
        0.08, 0.70,
        f"本次区域潮滩识别面积为 {stats['area_km2']:.4f} km²，覆盖比例 {stats['coverage_pct']:.2f}%。",
        fontsize=12, color=INK,
    )
    summary = extraction_summary or {}
    fig.text(
        0.08, 0.62,
        f"提取闭环状态：{_sanitize_text(summary.get('status') or 'unknown')}；"
        f"登记资产 {len(summary.get('assets') or [])} 项。",
        fontsize=12, color=INK,
    )
    fig.text(0.08, 0.54, "报告基于已入库成果自动生成，可用于成果归档、快速质检和后续人工复核。", fontsize=12, color=INK)
    fig.text(0.08, 0.46, "若需要正式监测结论，建议结合多期同源数据、现场样本和参考真值进行复核。", fontsize=12, color=INK)
    # 把终端窗口中对复核最有价值的数值结果带入监测报告；完整日志仍
    # 保留在任务执行面板，避免 PDF 被逐景进度刷屏。
    terminal_lines = _terminal_report_lines(terminal_logs, execution_results)
    if terminal_lines:
        fig.text(0.08, 0.37, "终端运行结果（关键数据）", fontsize=11, weight="bold", color=INK)
        display_lines = terminal_lines[:14]
        if len(terminal_lines) > len(display_lines):
            display_lines[-1] = f"…（共 {len(terminal_lines)} 行，详见任务终端日志）…"
        ax = fig.add_axes([0.08, 0.07, 0.84, 0.27])
        ax.axis("off")
        ax.text(
            0.0, 1.0, "\n".join(display_lines), va="top", ha="left",
            # 使用已配置的 CJK 无衬线字体，避免 DejaVu Sans Mono 把中文
            # 终端结果渲染成方框；报告可读性优先于等宽对齐。
            fontsize=7.5, color=INK, family="sans-serif",
        )
    pdf.savefig(fig)
    plt.close(fig)


# ---- 公开入口（绝不抛异常）----
def generate_asset_report(
    task: str,
    asset_key: Optional[str] = None,
    output_dir: Optional[str] = None,
    registry_path: Optional[str] = None,
    ref_shp: Optional[str] = None,
    progress_callback: Optional[Callable[[float, str], None]] = None,
    terminal_logs: Any = None,
    execution_results: Any = None,
) -> AssetReportResult:
    """基于已入库 Final TIF 生成成果 PDF 报告。

    参数：
        task: 任务名（如 "20fujian1"）
        asset_key: 指定资产 key；为空时取该任务最新 Final TIF
        output_dir: 输出目录；默认 TF-agent/data/reports
        registry_path: 资产注册表路径；默认 TF-agent/assets_registry.json
        ref_shp: 手动指定参考 SHP 绝对路径（可选）；否则自动匹配参考真值
        progress_callback: 进度回调 (pct, message)
    返回：AssetReportResult（失败时 success=False + error/warnings）
    """
    task_id = str(task or "task_unknown")

    def progress(value: float, message: str) -> None:
        if progress_callback:
            try:
                progress_callback(value, message)
            except Exception:
                pass

    warnings: List[str] = []
    if not _HAS_RASTERIO:
        return AssetReportResult(
            success=False, task_id=task_id, error="rasterio 未安装，无法生成成果报告",
            warnings=["rasterio 未安装"],
        )
    if not _HAS_MATPLOTLIB:
        return AssetReportResult(
            success=False, task_id=task_id, error="matplotlib 未安装，无法生成成果报告",
            warnings=["matplotlib 未安装"],
        )

    try:
        eligible = get_eligible_assets(task, registry_path=registry_path)
        if not eligible:
            return AssetReportResult(
                success=False, task_id=task_id,
                error=f"任务 {task_id} 无已入库 Final TIF 资产（assets_registry.json）",
                warnings=["无可用成果资产"],
            )
        if asset_key and asset_key not in eligible:
            warnings.append(f"指定资产 {asset_key} 不在该任务下，改用最新成果")
            asset_key = None
        if not asset_key:
            asset_key = next(iter(eligible))
        asset = eligible[asset_key]
        tif_path = str(asset.get("file_path") or "")

        # 去重：同 task + asset_key + 资产 mtime 已生成 → 返回已有
        try:
            asset_mtime = int(os.path.getmtime(tif_path))
        except Exception:
            asset_mtime = 0
        try:
            from report_generator import _execution_result_lines, _normalize_terminal_logs

            report_input_digest = hashlib.md5(
                (
                    "\n".join(_normalize_terminal_logs(terminal_logs))
                    + "\n"
                    + "\n".join(_execution_result_lines(execution_results))
                ).encode("utf-8", errors="replace")
            ).hexdigest()[:10]
        except Exception:
            report_input_digest = ""
        dedupe_raw = hashlib.md5(
            f"{task_id}|{asset_key}|{asset_mtime}|{report_input_digest}".encode("utf-8", errors="replace")
        ).hexdigest()[:10]
        out_dir = output_dir or _report_dir()
        try:
            os.makedirs(out_dir, exist_ok=True)
        except Exception:
            out_dir = tempfile.gettempdir()
        report_path = os.path.join(out_dir, f"asset_report_{_safe_filename(task_id)}_{_safe_filename(asset_key)}_{dedupe_raw}.pdf")
        if os.path.isfile(report_path) and os.path.getsize(report_path) > 0:
            return AssetReportResult(
                success=True, task_id=task_id, report_path=report_path,
                sections=list(_SECTIONS), warnings=["已存在同资产报告，返回已有文件"],
            )

        progress(0.05, "读取 Final TIF 并计算基础统计")
        stats = compute_raster_stats(tif_path)
        stats["_tif_name"] = _safe_basename(tif_path)
        progress(0.18, "生成空间预览图")
        preview = render_tif_preview(tif_path)
        progress(0.30, "匹配可选参考真值")
        ref = (
            {"id": "manual_reference", "title": _safe_basename(ref_shp), "_resolved_path": ref_shp}
            if ref_shp
            else _find_reference_shp(task)
        )
        progress(0.38, "计算参考真值对比")
        ref_cmp = _compare_reference(tif_path, ref)
        if ref_cmp and ref_cmp.get("error"):
            warnings.append(f"参考真值对比失败（已跳过该章）：{_sanitize_text(ref_cmp['error'])}")
        elif ref_cmp is None:
            warnings.append("未匹配到可用参考真值 SHP，参考真值对比章已跳过")

        progress(0.48, "初始化 PDF 模板")
        plt = _configure_matplotlib()
        from matplotlib.backends.backend_pdf import PdfPages
        extraction_summary = _collect_extraction_summary(task_id, registry_path=registry_path)

        with PdfPages(report_path) as pdf:
            _render_pages(
                pdf, plt, task_id, asset_key, stats, preview, ref_cmp,
                extraction_summary, progress,
                terminal_logs=terminal_logs,
                execution_results=execution_results,
            )

        if not os.path.isfile(report_path) or os.path.getsize(report_path) <= 0:
            return AssetReportResult(
                success=False, task_id=task_id, error="报告文件生成校验失败（缺失或为空）",
                warnings=warnings,
            )
        progress(1.0, "PDF 报告生成完成")
        return AssetReportResult(
            success=True, task_id=task_id, report_path=report_path,
            sections=list(_SECTIONS), warnings=warnings,
        )
    except Exception as e:
        return AssetReportResult(
            success=False, task_id=task_id, error=safe_error_summary(e), warnings=warnings,
        )
