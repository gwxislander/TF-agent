# -*- coding: utf-8 -*-
"""
Phase E: 任务 PDF 报告最小接入（仅 A–D 全绿后启用）。

- reportlab 纯 Python 渲染；中文字体探测 msyh.ttc → simhei.ttf → 降级英文。
- 仅使用真实数据（timeline / capabilities / assets / 任务结果），禁止编造。
- 无 token、无本地绝对路径（绝对路径一律转相对或 basename）。
- 生成后校验文件存在且非空，否则 FAILED。
- 同 task_id + 配置哈希去重（已存在 → 返回已有路径）。
- 截图失败 → warning + 占位说明，报告仍生成。
"""
from __future__ import annotations

import hashlib
from html import escape as html_escape
import importlib.util
import os
import re
import tempfile
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from agent_context_policy import (
    redact_spatial_metadata,
    safe_error_summary,
    sanitize_external_text,
)

_SENSITIVE_KEY_SUBSTRINGS = ("token", "secret", "password", "api_key", "ion", "key")
_SENSITIVE_VALUE_SUBSTRINGS = ("Z:/", "C:\\", "/home/", "token=", "key=", "sk-")
_REPORT_DIRNAME = "data/reports"
_SECTIONS = ("基本信息", "能力状态", "执行时间线", "终端运行结果", "资产清单", "地图截图")
_TERMINAL_MAX_LINES = 180
_REPORT_DECORATIVE_RE = re.compile(r"[\U0001F300-\U0001FAFF\u2600-\u27BF\uFE0F]")
_TERMINAL_RESULT_RE = re.compile(
    r"(?i)(top[- ]?\d+|排名|最佳|最优|iou|f1|precision|recall|prec|rec|面积|耗时|"
    r"完成|成功|失败|警告|warning|error|step\s*\d|phase\s*\d|参数|场景|像元|"
    r"潮滩|导出|写入|保存|参考真值|网格|任务)"
)

_HAS_REPORTLAB = importlib.util.find_spec("reportlab") is not None


@dataclass
class ReportResult:
    success: bool
    task_id: str
    report_path: Optional[str] = None
    sections: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    error: str = ""


def _sanitize_text(text: Any) -> str:
    """去除敏感值（token/绝对路径），并限制长度。"""
    raw = str(text or "")
    low = raw.lower()
    if any(sub in low for sub in _SENSITIVE_VALUE_SUBSTRINGS):
        return "[已过滤]"
    s = redact_spatial_metadata(sanitize_external_text(raw))
    # ``sk-`` is a provider key prefix even when the surrounding field name is
    # absent; redact it before report text is persisted.
    s = re.sub(r"(?i)\bsk-[A-Za-z0-9_-]+", "<redacted>", s)
    # Matplotlib/ReportLab installations commonly lack color-emoji glyphs.
    # Remove decorative symbols in the persisted report while retaining all
    # textual metrics and values, preventing tofu boxes in PDF output.
    s = _REPORT_DECORATIVE_RE.sub("", s)
    # ReportLab Paragraph treats ``<...>`` and ``&`` as markup.  Escape after
    # all security redaction so task names, errors and timeline messages cannot
    # break PDF generation or inject unintended formatting.
    return html_escape(s, quote=False)[:400]


def _sanitize_key(name: str) -> bool:
    low = name.lower()
    return not any(sub in low for sub in _SENSITIVE_KEY_SUBSTRINGS)


def _relative_path(p: str, base_dir: Optional[str] = None) -> str:
    """绝对路径转相对（相对报告目录或 TF-agent 目录），无路径则原样。"""
    p = str(p or "")
    if not p:
        return ""
    p = p.replace("\\", "/")
    is_windows_abs = bool(re.match(r"^[A-Za-z]:/", p))
    is_unc = p.startswith("//")
    is_posix_abs = p.startswith("/") and not p.startswith("//")
    if (is_windows_abs or is_unc or is_posix_abs) and not p.startswith(("http://", "https://")):
        if base_dir:
            try:
                rel = os.path.relpath(p, base_dir).replace("\\", "/")
                if not rel.startswith(".."):
                    return rel
            except Exception:
                pass
        return os.path.basename(p) or "<local-path>"
    return p


def _find_cjk_font() -> Optional[str]:
    """返回可用的中文字体名（reportlab 注册名），找不到返回 None。"""
    if not _HAS_REPORTLAB:
        return None
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont

    candidates = [
        # macOS 系统字体（Windows 路径在 macOS 上不存在，会导致中文回退到 Helvetica）。
        ("ArialUnicode", "/System/Library/Fonts/Supplemental/Arial Unicode.ttf"),
        ("ArialUnicode", "/Library/Fonts/Arial Unicode.ttf"),
        ("Msyh", "C:/Windows/Fonts/msyh.ttc"),
        ("Msyh", "C:/Windows/Fonts/msyh.ttf"),
        ("SimHei", "C:/Windows/Fonts/simhei.ttf"),
        ("SimSun", "C:/Windows/Fonts/simsun.ttc"),
    ]
    seen = set()
    for name, path in candidates:
        if name in seen or not os.path.exists(path):
            continue
        seen.add(name)
        try:
            pdfmetrics.registerFont(TTFont(name, path))
            return name
        except Exception:
            continue
    return None


def _config_hash(
    task_context: Dict[str, Any],
    terminal_logs: Any = None,
    execution_results: Any = None,
) -> str:
    payload = {
        "task": task_context.get("task"),
        "mode": task_context.get("mode"),
        "prob": task_context.get("prob"),
        "cnt": task_context.get("cnt"),
        "plan_id": task_context.get("plan_id"),
        # A report generated before post-processing completes must not be
        # reused after the terminal has produced the final metrics.
        "terminal_digest": hashlib.md5(
            "\n".join(_normalize_terminal_logs(terminal_logs)).encode("utf-8", errors="replace")
        ).hexdigest()[:12],
        "results_digest": hashlib.md5(
            "\n".join(_execution_result_lines(execution_results)).encode("utf-8", errors="replace")
        ).hexdigest()[:12],
    }
    raw = hashlib.md5(str(sorted(payload.items())).encode("utf-8", errors="replace")).hexdigest()[:12]
    return raw


def _report_dir() -> str:
    base = os.path.dirname(os.path.abspath(__file__))
    d = os.path.join(base, _REPORT_DIRNAME)
    try:
        os.makedirs(d, exist_ok=True)
    except Exception:
        d = tempfile.gettempdir()
    return d


def _normalize_capabilities(capabilities: Any) -> List[str]:
    """归一化能力摘要为文本行（白名单：仅 id + status + summary）。"""
    lines: List[str] = []
    if capabilities is None:
        return lines
    if isinstance(capabilities, dict):
        for cid, info in capabilities.items():
            if isinstance(info, dict):
                status = info.get("status", "")
                summary = info.get("summary", "")
            else:
                status, summary = str(info), ""
            if not _sanitize_key(str(cid)):
                continue
            line = f"- {cid}: {status}"
            if summary:
                line += f" ({_sanitize_text(summary)})"
            lines.append(line)
    elif isinstance(capabilities, (list, tuple)):
        for item in capabilities:
            lines.append(f"- {_sanitize_text(item)}")
    return lines


def _normalize_timeline(timeline: Any) -> List[Dict[str, Any]]:
    """归一化时间线事件列表（转 dict，过滤敏感详情）。"""
    out: List[Dict[str, Any]] = []
    if not timeline:
        return out
    for ev in timeline:
        if hasattr(ev, "to_dict"):
            try:
                d = ev.to_dict()
            except Exception:
                d = {}
        elif isinstance(ev, dict):
            d = dict(ev)
        else:
            continue
        details = d.get("details")
        if isinstance(details, dict):
            d["details"] = {k: _sanitize_text(v) for k, v in details.items()}
        out.append(d)
    return out


def _normalize_terminal_logs(logs: Any, max_lines: int = _TERMINAL_MAX_LINES) -> List[str]:
    """Normalize terminal output for a report while retaining result lines.

    Worker logs can be much longer than the terminal viewport.  Keep a bounded
    excerpt, but always retain lines that carry metrics, phase outcomes or
    warnings so the PDF remains useful for audit and reproducibility.
    """
    if logs is None:
        return []
    if isinstance(logs, str):
        raw_lines = logs.splitlines()
    elif isinstance(logs, (list, tuple)):
        raw_lines = []
        for item in logs:
            if isinstance(item, dict):
                item = item.get("message") or item.get("text") or ""
            raw_lines.extend(str(item or "").splitlines())
    else:
        raw_lines = [str(logs)]

    lines: List[str] = []
    for raw in raw_lines:
        line = _sanitize_text(raw).strip()
        if line:
            lines.append(line)
    if len(lines) <= max_lines:
        return lines

    # Preserve the beginning/end context and result-bearing middle lines.
    head_n = min(18, max_lines // 4)
    tail_n = min(18, max_lines // 4)
    middle_budget = max(0, max_lines - head_n - tail_n - 1)
    middle = [line for line in lines[head_n:-tail_n] if _TERMINAL_RESULT_RE.search(line)]
    selected = lines[:head_n] + middle[:middle_budget]
    selected.append(f"…（终端日志共 {len(lines)} 行，报告已保留关键结果与首尾上下文）…")
    selected.extend(lines[-tail_n:])
    return selected[:max_lines]


def _safe_result_value(value: Any, *, percent: bool = False) -> str:
    """Format a structured result value without exposing local paths."""
    if value is None or value == "":
        return "—"
    if percent:
        try:
            number = float(value)
            if abs(number) <= 1.0:
                number *= 100.0
            return f"{number:.2f}%"
        except (TypeError, ValueError):
            pass
    if isinstance(value, (int, float)):
        return f"{value:g}"
    raw = str(value).replace("\\", "/")
    if raw.startswith(("/", "//")) or re.match(r"^[A-Za-z]:/", raw):
        raw = os.path.basename(raw)
    return _sanitize_text(raw)


def _execution_result_lines(results: Any) -> List[str]:
    """Build human-readable metric lines from real worker result dictionaries."""
    if not isinstance(results, dict):
        return []
    lines: List[str] = []
    autotune = results.get("autotune") or results.get("auto_tune")
    if isinstance(autotune, dict):
        lines.append(
            "参数优化最佳结果："
            f"P={_safe_result_value(autotune.get('best_prob'))}，"
            f"C={_safe_result_value(autotune.get('best_cnt'))}，"
            f"IoU={_safe_result_value(autotune.get('best_iou'), percent=True)}，"
            f"F1={_safe_result_value(autotune.get('best_f1'), percent=True)}，"
            f"Precision={_safe_result_value(autotune.get('best_precision'), percent=True)}，"
            f"Recall={_safe_result_value(autotune.get('best_recall'), percent=True)}，"
            f"试验数={_safe_result_value(autotune.get('total_trials'))}，"
            f"耗时={_safe_result_value(autotune.get('total_time_sec'))} s"
        )
        trials = autotune.get("trials")
        if isinstance(trials, (list, tuple)) and trials:
            lines.append("Top-10 参数组合（按评分降序）：")
            sorted_trials = sorted(
                (trial for trial in trials if isinstance(trial, dict)),
                key=lambda trial: float(trial.get("score") or trial.get("iou") or 0),
                reverse=True,
            )[:10]
            for rank, trial in enumerate(sorted_trials, 1):
                lines.append(
                    f"{rank}. P={_safe_result_value(trial.get('prob'))} "
                    f"C={_safe_result_value(trial.get('cnt'))} "
                    f"IoU={_safe_result_value(trial.get('iou'), percent=True)} "
                    f"F1={_safe_result_value(trial.get('f1'), percent=True)} "
                    f"Precision={_safe_result_value(trial.get('precision'), percent=True)} "
                    f"Recall={_safe_result_value(trial.get('recall'), percent=True)}"
                )

    for name, label in (
        ("inference", "推理结果"),
        ("gee", "影像获取结果"),
        ("m5", "变化分析结果"),
        ("e1", "精度评价结果"),
    ):
        value = results.get(name)
        if not isinstance(value, dict):
            continue
        metrics = value.get("metrics") if isinstance(value.get("metrics"), dict) else value
        if not isinstance(metrics, dict):
            continue
        picked = []
        for key, metric_label, is_percent in (
            ("elapsed_seconds", "耗时(s)", False),
            ("total_time_sec", "耗时(s)", False),
            ("processed_tiles", "处理瓦片", False),
            ("tif_count", "影像数", False),
            ("scene_count", "场景数", False),
            ("area_km2", "潮滩面积(km²)", False),
            ("change_km2", "变化面积(km²)", False),
            ("iou", "IoU", True),
            ("f1", "F1", True),
            ("precision", "Precision", True),
            ("recall", "Recall", True),
        ):
            if key in metrics:
                picked.append(f"{metric_label}={_safe_result_value(metrics.get(key), percent=is_percent)}")
        if picked:
            lines.append(f"{label}：" + "；".join(picked))
        if name == "m5":
            quantitative = value.get("quantitative_metrics")
            if isinstance(quantitative, dict):
                area = quantitative.get("area_evolution") or {}
                if isinstance(area, dict) and any(
                    area.get(k) is not None
                    for k in ("baseline_area_km2", "current_area_km2", "change_rate_percentage")
                ):
                    lines.append(
                        "变化分析面积："
                        f"基线={_safe_result_value(area.get('baseline_area_km2'))} km²；"
                        f"当前={_safe_result_value(area.get('current_area_km2'))} km²；"
                        f"变化率={_safe_result_value(area.get('change_rate_percentage'))}%"
                    )
                centroid = quantitative.get("centroid_trajectory") or {}
                if isinstance(centroid, dict) and centroid.get("drift_distance_meters") is not None:
                    lines.append(
                        "变化分析空间指标："
                        f"重心漂移={_safe_result_value(centroid.get('drift_distance_meters'))} m；"
                        f"方位角={_safe_result_value(centroid.get('migration_azimuth_degrees'))}°"
                    )
            if value.get("alert_level") is not None:
                lines.append(
                    f"变化分析告警：级别={_safe_result_value(value.get('alert_level'))}；"
                    f"诊断={_safe_result_value(value.get('diagnostic_message') or '—')}"
                )
        elif name == "e1":
            if value.get("reference"):
                lines.append(f"精度评价参考产品：{_safe_result_value(value.get('reference'))}")
            comparisons = value.get("comparisons") or {}
            if isinstance(comparisons, dict):
                shown = 0
                for pair, comparison in comparisons.items():
                    if not isinstance(comparison, dict):
                        continue
                    if comparison.get("error"):
                        lines.append(
                            f"精度评价对比 { _safe_result_value(pair) }："
                            f"失败={_safe_result_value(comparison.get('error'))}"
                        )
                    elif comparison.get("jaccard_iou") is not None:
                        lines.append(
                            f"精度评价对比 { _safe_result_value(pair) }："
                            f"IoU={_safe_result_value(comparison.get('jaccard_iou'), percent=True)}；"
                            f"交集={_safe_result_value(comparison.get('intersection_km2'))} km²；"
                            f"像元={_safe_result_value(comparison.get('intersection_pixels'))}"
                        )
                    shown += 1
                    if shown >= 10:
                        break
                if len(comparisons) > shown:
                    lines.append(f"精度评价其余对比组：{len(comparisons) - shown} 组（详见原始结果）")
    # Allow callers to provide a pre-computed, safe textual summary for a new
    # execution module without changing this renderer.
    for key in ("summary", "terminal_summary"):
        if results.get(key):
            lines.append(_safe_result_value(results[key]))
    return lines


def _render_pdf(task_context: Dict[str, Any], cap_lines: List[str],
                events: List[Dict[str, Any]], assets: List[Dict[str, Any]],
                map_snapshot: Optional[bytes], report_path: str,
                warnings: List[str], terminal_logs: Any = None,
                execution_results: Any = None) -> List[str]:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.pdfbase.pdfmetrics import stringWidth
    from reportlab.platypus import (
        Image, Paragraph, Preformatted, SimpleDocTemplate, Spacer, Table, TableStyle,
    )

    font = _find_cjk_font()
    if font is None:
        warnings.append("中文字体缺失，报告降级为英文/数字")
        body_font = "Helvetica"
        bold_font = "Helvetica-Bold"
    else:
        body_font = font
        bold_font = font

    style_h = ParagraphStyle(
        "h", fontName=bold_font, fontSize=15, leading=19, spaceAfter=6,
        textColor=colors.HexColor("#16324f"),
    )
    style_h2 = ParagraphStyle(
        "h2", fontName=bold_font, fontSize=12, leading=15, spaceBefore=10,
        spaceAfter=4, textColor=colors.HexColor("#24476e"),
    )
    style_body = ParagraphStyle(
        "body", fontName=body_font, fontSize=9.5, leading=13,
    )
    style_small = ParagraphStyle(
        "small", fontName=body_font, fontSize=8, leading=11,
        textColor=colors.HexColor("#555555"),
    )

    doc = SimpleDocTemplate(
        report_path, pagesize=A4,
        leftMargin=18 * mm, rightMargin=18 * mm,
        topMargin=16 * mm, bottomMargin=16 * mm,
        title=f"任务报告 {task_context.get('task_id', '')}",
    )
    story: List[Any] = []
    story.append(Paragraph("任务执行报告", style_h))
    story.append(Paragraph(
        f"task_id: {_sanitize_text(task_context.get('task_id', '—'))}"
        f" &nbsp; 生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        style_small,
    ))
    story.append(Spacer(1, 4))

    # 基本信息
    story.append(Paragraph("一、基本信息", style_h2))
    info_rows = [
        ["任务", _sanitize_text(task_context.get("task", "—"))],
        ["模式", _sanitize_text(task_context.get("mode", "—"))],
        ["概率阈值", _sanitize_text(task_context.get("prob", "—"))],
        ["频次阈值", _sanitize_text(task_context.get("cnt", "—"))],
        ["计划 id", _sanitize_text(task_context.get("plan_id", "—"))],
    ]
    t_info = Table(info_rows, colWidths=[28 * mm, 130 * mm])
    t_info.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#eef3fa")),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#c6d3e4")),
        ("FONTNAME", (0, 0), (0, -1), bold_font),
        ("FONTNAME", (1, 0), (1, -1), body_font),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    story.append(t_info)

    # 能力状态
    story.append(Paragraph("二、能力状态", style_h2))
    if cap_lines:
        for line in cap_lines:
            story.append(Paragraph(_sanitize_text(line), style_body))
    else:
        story.append(Paragraph("（无能力摘要数据）", style_small))

    # 时间线
    story.append(Paragraph("三、执行时间线", style_h2))
    if events:
        tl_rows = [["阶段", "状态", "进度", "消息", "工具"]]
        for ev in events:
            tl_rows.append([
                _sanitize_text(ev.get("phase", "—")),
                _sanitize_text(ev.get("status", "—")),
                _sanitize_text(ev.get("progress")),
                _sanitize_text(ev.get("message", "—"))[:80],
                _sanitize_text(ev.get("tool", "—")),
            ])
        t_tl = Table(tl_rows, colWidths=[18 * mm, 24 * mm, 14 * mm, 82 * mm, 20 * mm])
        t_tl.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eef3fa")),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#c6d3e4")),
            ("FONTNAME", (0, 0), (-1, -1), body_font),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 4),
            ("RIGHTPADDING", (0, 0), (-1, -1), 4),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ]))
        story.append(t_tl)
    else:
        story.append(Paragraph("（无时间线事件）", style_small))

    # 终端结果与结构化指标。原始终端窗口中的 AutoTune/后处理数据不再
    # 只停留在 UI：报告同时保存机器可复核的摘要与经过消毒的日志摘录。
    story.append(Paragraph("四、终端运行结果与指标", style_h2))
    result_lines = _execution_result_lines(execution_results)
    terminal_lines = _normalize_terminal_logs(terminal_logs)
    if result_lines:
        for line in result_lines:
            story.append(Paragraph(_sanitize_text(line), style_body))
    if terminal_lines:
        terminal_style = ParagraphStyle(
            "terminal", parent=style_small, fontName=body_font, fontSize=7.2,
            leading=9.2, backColor=colors.HexColor("#f5f7fa"),
            borderColor=colors.HexColor("#d0d5dd"), borderWidth=0.4,
            borderPadding=4, spaceBefore=4, spaceAfter=5,
        )
        story.append(Preformatted("\n".join(terminal_lines), terminal_style))
    if not result_lines and not terminal_lines:
        story.append(Paragraph("（未提供终端运行结果）", style_small))

    # 资产清单
    story.append(Paragraph("五、资产清单与潮滩提取结果", style_h2))
    if assets:
        for i, a in enumerate(assets, 1):
            name = _relative_path(a.get("path") or a.get("name") or "")
            kind = _sanitize_text(a.get("kind") or a.get("type") or "—")
            status = _sanitize_text(a.get("status") or "已登记")
            size = a.get("file_size_mb")
            size_text = f"；大小 {float(size):.2f} MB" if isinstance(size, (int, float)) else ""
            params = []
            for key, label in (("prob_threshold", "概率"), ("min_count", "频次"), ("model_id", "模型"), ("weight_id", "权重")):
                if a.get(key) is not None:
                    params.append(f"{label}={_sanitize_text(a[key])}")
            param_text = f"；{', '.join(params)}" if params else ""
            story.append(Paragraph(
                f"{i}. {_sanitize_text(name)}  [{kind}]；状态 {status}{size_text}{param_text}",
                style_body,
            ))
    else:
        story.append(Paragraph("（无资产登记）", style_small))

    # 地图截图
    story.append(Paragraph("六、地图截图", style_h2))
    if map_snapshot:
        try:
            import io

            img = Image(io.BytesIO(map_snapshot))
            img._width = 90 * mm
            img._height = 90 * mm
            img.hAlign = "LEFT"
            story.append(img)
        except Exception:
            warnings.append("地图截图嵌入失败，已用占位说明")
            story.append(Paragraph("（截图嵌入失败：格式不受支持）", style_small))
    else:
        story.append(Paragraph("（未提供截图）", style_small))

    if warnings:
        story.append(Spacer(1, 8))
        story.append(Paragraph("报告说明", style_h2))
        for w in warnings:
            story.append(Paragraph("· " + _sanitize_text(w), style_small))

    doc.build(story)
    return list(_SECTIONS)


def generate_task_report(
    task_context: Dict[str, Any],
    capabilities: Any = None,
    timeline: Any = None,
    assets: Optional[List[Dict[str, Any]]] = None,
    map_snapshot: Optional[bytes] = None,
    terminal_logs: Any = None,
    execution_results: Any = None,
) -> ReportResult:
    """生成任务 PDF 报告。返回 ReportResult（绝不抛异常，失败记录 error）。"""
    warnings: List[str] = []
    task_id = str(task_context.get("task_id") or "task_unknown")
    if not _HAS_REPORTLAB:
        return ReportResult(
            success=False, task_id=task_id,
            error="reportlab 未安装，无法生成 PDF 报告",
            warnings=["reportlab 未安装"],
        )
    try:
        cfg_hash = _config_hash(task_context, terminal_logs, execution_results)
        report_dir = _report_dir()
        report_path = os.path.join(report_dir, f"report_{task_id}_{cfg_hash}.pdf")

        # 同任务去重：已存在且非空 → 返回已有路径
        if os.path.isfile(report_path) and os.path.getsize(report_path) > 0:
            return ReportResult(
                success=True, task_id=task_id, report_path=report_path,
                sections=list(_SECTIONS), warnings=["已存在同任务报告，返回已有文件"],
            )

        cap_lines = _normalize_capabilities(capabilities)
        events = _normalize_timeline(timeline)
        if not events:
            warnings.append("时间线为空，报告不含执行阶段明细")
        assets = assets or []

        sections = _render_pdf(
            task_context, cap_lines, events, assets, map_snapshot,
            report_path, warnings, terminal_logs, execution_results,
        )

        if not os.path.isfile(report_path) or os.path.getsize(report_path) <= 0:
            return ReportResult(
                success=False, task_id=task_id,
                error="报告文件生成校验失败（缺失或为空）",
                warnings=warnings,
            )
        return ReportResult(
            success=True, task_id=task_id, report_path=report_path,
            sections=sections, warnings=warnings,
        )
    except Exception as e:
        return ReportResult(
            success=False, task_id=task_id, error=safe_error_summary(e), warnings=warnings,
        )
