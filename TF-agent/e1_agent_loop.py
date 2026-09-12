# -*- coding: utf-8 -*-
"""
E1 多源一致性诊断 Agent 执行闭环：预检 → 计划 → 验证 → 摘要。

不依赖 Streamlit；由 agent_command_bridge / app 调用。
引擎入口保持 e1_engine.run_e1_after_synthesis（不改 E1 核心算法）。
"""
from __future__ import annotations

import os
import re
import json
from datetime import datetime
from typing import Any, Dict, List, Optional

import e1_engine
import m5_engine
from agent_context_policy import safe_error_summary

_E1_INTENT_RE = re.compile(
    r"(多源一致|一致性诊断|\be1\b|E1|和师姐比|对比开源|分歧图|多产品热力)",
    re.IGNORECASE,
)


def _nonempty_file(path: object) -> bool:
    """Return true only for a readable path with at least one byte."""
    try:
        return bool(path) and os.path.isfile(str(path)) and os.path.getsize(str(path)) > 0
    except OSError:
        return False


_MAP_ASSET_SUFFIXES = frozenset({".tif", ".tiff", ".shp", ".geojson", ".gpkg"})


def _path_candidates(raw: object, base_dirs: Optional[List[str]] = None):
    """Yield normalized candidates for absolute and legacy relative paths."""
    if raw is None:
        return
    text = str(raw).strip().strip('"').strip("'")
    if not text:
        return
    normalized = os.path.normpath(os.path.expanduser(text))
    yielded = set()

    def _yield(path: str):
        candidate = os.path.normpath(os.path.abspath(path))
        if candidate not in yielded:
            yielded.add(candidate)
            yield candidate

    if os.path.isabs(normalized):
        yield from _yield(normalized)
        return
    for base in base_dirs or []:
        if base:
            yield from _yield(os.path.join(str(base), normalized))
    # Keep the current process directory as a final compatibility fallback.
    yield from _yield(normalized)


def _report_base_dirs(report: Optional[Dict[str, Any]], extra_base: Optional[str] = None) -> List[str]:
    """Build deterministic bases for paths embedded in old E1 JSON reports."""
    bases: List[str] = []
    for raw in (extra_base, (report or {}).get("workspace_dir")):
        if raw:
            bases.append(os.path.normpath(os.path.abspath(os.path.expanduser(str(raw)))))
    report_path = (report or {}).get("report_path")
    if report_path:
        for candidate in _path_candidates(report_path, bases):
            if os.path.isfile(candidate):
                bases.insert(0, os.path.dirname(candidate))
                break
    bases.extend([os.getcwd(), os.path.dirname(os.path.abspath(__file__))])
    # Preserve order while avoiding repeated filesystem probes.
    return list(dict.fromkeys(bases))


def _resolve_existing_path(raw: object, base_dirs: Optional[List[str]] = None) -> Optional[str]:
    for candidate in _path_candidates(raw, base_dirs):
        if _nonempty_file(candidate):
            return candidate
    return None
_CONFIRM_RE = re.compile(
    r"^(确认|同意|好的?|可以|执行|开始|开始执行|确认执行|确认计划|就这样|ok|OK)[\s!！。．.]*$",
    re.IGNORECASE,
)
_CONFIRM_PHRASE_RE = re.compile(
    r"(确认.?(执行|计划|e1|一致性)|开始.?(执行|e1|一致性)|执行.?(计划|e1)|同意.?(执行|计划))",
    re.IGNORECASE,
)


def is_e1_intent(text: str) -> bool:
    return bool(_E1_INTENT_RE.search(text or ""))


def is_e1_confirm_utterance(text: str) -> bool:
    t = (text or "").strip()
    if not t or len(t) > 80:
        return False
    if _CONFIRM_RE.match(t):
        return True
    return bool(_CONFIRM_PHRASE_RE.search(t))


def resolve_target_shp(
    final_root: str,
    task: str,
    prob: Optional[float] = None,
    cnt: Optional[int] = None,
) -> Optional[str]:
    task_dir = os.path.join(final_root, task)
    return m5_engine.find_final_shp_in_task_dir(task_dir, task, prob, cnt)


def build_e1_preflight(
    *,
    final_root: str,
    current_task: str,
    data_root: str,
    reference: str = "师姐_2020",
    compare_sources: Optional[List[str]] = None,
    prob: Optional[float] = None,
    cnt: Optional[int] = None,
    task_aoi_shp: Optional[str] = None,
    export_disagreement_maps: bool = True,
    export_multi_product_heatmap: bool = True,
) -> Dict[str, Any]:
    """生成结构化 E1 执行计划；ready=True 方可确认执行。"""
    final_root = os.path.normpath(str(final_root or "").strip())
    data_root = os.path.normpath(str(data_root or "").strip())
    current_task = (current_task or "").strip()
    reference = (reference or "师姐_2020").strip()
    blockers: List[str] = []
    warnings: List[str] = []

    if not current_task:
        blockers.append("未指定当前任务（selected_task / task）。")
    if not final_root or not os.path.isdir(final_root):
        blockers.append(f"成果根目录不存在: {final_root or '（空）'}")
    if not data_root or not os.path.isdir(data_root):
        blockers.append(f"E1 数据集根目录不存在: {data_root or '（空）'}")

    current_shp = None
    if current_task and final_root and os.path.isdir(final_root):
        current_shp = resolve_target_shp(final_root, current_task, prob, cnt)
        if not current_shp:
            blockers.append(
                f"当期潮滩成果 SHP 不存在：{os.path.join(final_root, current_task)}"
            )

    available_datasets: List[str] = []
    if data_root and os.path.isdir(data_root):
        try:
            available_datasets = list(e1_engine.list_e1_datasets(data_root) or [])
        except Exception as exc:
            warnings.append(f"列举 E1 数据集失败（执行时将再试）：{safe_error_summary(exc)}")

    if available_datasets and reference not in available_datasets:
        warnings.append(
            f"参考产品「{reference}」未出现在数据集列表中，执行时可能失败。"
        )

    resolved_compare = list(compare_sources) if compare_sources else None
    if resolved_compare is not None:
        resolved_compare = [
            n for n in resolved_compare
            if n != reference and n not in e1_engine._SKIP_COMPARE
        ]
        if not resolved_compare:
            blockers.append("指定的对比产品列表为空（或全部被过滤）。")

    workspace = (
        e1_engine.workspace_for_task(final_root, current_task)
        if current_task and final_root
        else None
    )

    ready = len(blockers) == 0 and bool(current_shp) and bool(data_root and os.path.isdir(data_root))

    return {
        "schema": "e1_execution_plan_v1",
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "ready": ready,
        "blockers": blockers,
        "warnings": warnings,
        "current_task": current_task,
        "final_root": final_root,
        "current_shp": os.path.normpath(current_shp) if current_shp else None,
        "data_root": data_root,
        "reference": reference,
        "compare_sources": resolved_compare,
        "available_datasets": available_datasets[:30],
        "task_aoi_shp": (task_aoi_shp or "").strip() or None,
        "workspace_dir": workspace,
        "export_disagreement_maps": bool(export_disagreement_maps),
        "export_multi_product_heatmap": bool(export_multi_product_heatmap),
        "prob": prob,
        "cnt": cnt,
        "steps": [
            "读取当期潮滩成果",
            "解析任务研究区域（可选 AOI）",
            "运行精度评价，逐像元对比多个产品",
            "校验报告 JSON 与可选热力图",
            "保存评价成果并根据真实指标回复",
        ],
    }


def format_e1_plan_for_user(plan: Dict[str, Any]) -> str:
    if not plan:
        return "尚未生成精度评价计划。"
    lines = ["## 潮滩精度评价 · 执行计划", ""]
    if plan.get("ready"):
        lines.append("**状态：可执行**（请回复「确认」或点击确认按钮后开始）")
    else:
        lines.append("**状态：暂不可执行**")
        for b in plan.get("blockers") or []:
            lines.append(f"- 阻塞：{b}")
    lines.append("")
    lines.append(f"- 当前任务：`{plan.get('current_task') or '—'}`")
    lines.append(f"- 目标 SHP：`{os.path.basename(plan.get('current_shp') or '') or '—'}`")
    lines.append(f"- 参考产品：`{plan.get('reference') or '—'}`")
    lines.append(f"- 数据集根：`{plan.get('data_root') or '—'}`")
    ds = plan.get("available_datasets") or []
    if ds:
        lines.append(f"- 已探测数据集（部分）：{', '.join(ds[:8])}")
    for w in plan.get("warnings") or []:
        lines.append(f"- 注意：{w}")
    lines.append("")
    lines.append("确认后将真实运行精度评价，并根据磁盘报告回复交并比等指标。")
    return "\n".join(lines)


def pick_e1_map_path(
    report: Optional[Dict[str, Any]],
    base_dir: Optional[str] = None,
) -> Optional[str]:
    """Return the first existing map layer referenced by an E1 report.

    Reports generated by older versions stored map paths relative to the
    report/workspace directory.  Resolve those paths here so both the live
    result and the historical asset list use the same compatibility logic.
    """
    if not report:
        return None
    bases = _report_base_dirs(report, base_dir)
    mp = report.get("multi_product_heatmap") or {}
    for key in ("any_disagreement_tif", "agreement_count_tif"):
        p = mp.get(key)
        resolved = _resolve_existing_path(p, bases)
        if resolved:
            return resolved
    comps = report.get("comparisons") or {}
    for _pair, metrics in comps.items():
        if not isinstance(metrics, dict) or "error" in metrics:
            continue
        maps = (metrics.get("causal_analysis") or {}).get("disagreement_maps") or {}
        for key in ("heatmap", "class", "consensus"):
            p = maps.get(key)
            resolved = _resolve_existing_path(p, bases)
            if resolved:
                return resolved
    return None


def resolve_e1_map_asset(
    asset: Optional[Dict[str, Any]],
    base_dir: Optional[str] = None,
) -> Optional[str]:
    """Resolve a historical E1 registry row to a map-loadable asset.

    The registry has existed in several shapes: current rows put the map in
    ``file_path``, while older rows used that field for the JSON/Markdown
    report.  Prefer explicit map fields and fall back to reading the report.
    No registry mutation is performed.
    """
    if not isinstance(asset, dict):
        return None
    bases = _report_base_dirs(asset, base_dir)

    for key in ("map_path", "heatmap_path", "file_path"):
        raw = asset.get(key)
        resolved = _resolve_existing_path(raw, bases)
        if resolved and os.path.splitext(resolved)[1].lower() in _MAP_ASSET_SUFFIXES:
            return resolved

    report_raw = asset.get("report_path")
    if not report_raw:
        file_raw = asset.get("file_path")
        if str(file_raw or "").lower().endswith(".json"):
            report_raw = file_raw
    report_path = _resolve_existing_path(report_raw, bases)
    if not report_path or not report_path.lower().endswith(".json"):
        return None
    try:
        with open(report_path, "r", encoding="utf-8") as handle:
            report = json.load(handle)
    except (OSError, ValueError, TypeError):
        return None
    if not isinstance(report, dict):
        return None
    report.setdefault("report_path", report_path)
    return pick_e1_map_path(report, base_dir=os.path.dirname(report_path))


def verify_e1_outputs(report: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    checks: List[Dict[str, Any]] = []
    ok = True
    if not report or not isinstance(report, dict):
        return {
            "ok": False,
            "checks": [{"name": "report_object", "passed": False, "detail": "无报告对象"}],
            "map_candidate": None,
        }

    def _check(name: str, passed: bool, detail: str = "") -> None:
        nonlocal ok
        if not passed:
            ok = False
        checks.append({"name": name, "passed": passed, "detail": detail})

    _check("reference", bool(report.get("reference")), str(report.get("reference") or ""))
    comps = report.get("comparisons") or {}
    _check("comparisons", isinstance(comps, dict) and len(comps) > 0, f"n={len(comps)}")

    report_path = report.get("report_path")
    _check(
        "report_json_on_disk",
        _nonempty_file(report_path),
        str(report_path or ""),
    )

    has_iou = False
    for _k, m in comps.items():
        if isinstance(m, dict) and m.get("jaccard_iou") is not None:
            has_iou = True
            break
    _check("jaccard_iou_present", has_iou, "")

    map_path = pick_e1_map_path(report)
    checks.append(
        {
            "name": "map_layer",
            "passed": True,
            "detail": map_path or "无（可为空）",
        }
    )
    return {"ok": ok, "checks": checks, "map_candidate": map_path, "report_path": report_path}


def summarize_e1_report_for_chat(
    report: Optional[Dict[str, Any]],
    verification: Optional[Dict[str, Any]] = None,
) -> str:
    if not report:
        return (
            "精度评价未生成有效结果。"
            "请检查当期成果、数据集根目录与参考产品后重试。"
        )
    verification_state = (
        "已验证" if verification and verification.get("ok") is True
        else "校验未完全通过" if verification is not None
        else "待校验"
    )
    lines = [
        f"## 潮滩精度评价结果（{verification_state}）",
        "",
        f"- 任务 / ROI：`{report.get('roi_name') or '—'}`",
        f"- 参考产品：`{report.get('reference') or '—'}`",
    ]
    comps = report.get("comparisons") or {}
    shown = 0
    for pair, m in comps.items():
        if not isinstance(m, dict) or "error" in m:
            continue
        iou = m.get("jaccard_iou", "—")
        inter = m.get("intersection_km2", "—")
        lines.append(f"- `{pair}`：交并比={iou} · 交集 {inter} km²")
        shown += 1
        if shown >= 8:
            break
    if shown == 0:
        lines.append("- 对比组：无有效 IoU（可能全部失败）")
    mp = report.get("multi_product_heatmap") or {}
    if mp.get("disagreement_pixel_ratio") is not None:
        lines.append(f"- 多产品分歧像元占比：{mp.get('disagreement_pixel_ratio'):.2%}")
    rp = report.get("report_path")
    if rp:
        lines.append(f"- 报告：`{os.path.basename(str(rp))}`")
    if verification is not None:
        lines.append(
            f"- 输出校验：{'通过' if verification.get('ok') else '未完全通过'}"
        )
    lines.append("")
    lines.append("以上指标均来自本次精度评价的真实输出，而非模型臆测。")
    return "\n".join(lines)


def build_e1_context_for_agent(
    final_root: str,
    current_task: str,
    data_root: str,
    reference: str = "师姐_2020",
    pending_plan: Optional[Dict[str, Any]] = None,
) -> str:
    lines = ["【潮滩精度评价账本】"]
    lines.append(f"- 当前任务：{current_task or '未选'}")
    lines.append(f"- 参考：{reference or '—'}")
    lines.append(f"- 数据集根：{data_root or '—'}")
    shp = resolve_target_shp(final_root, current_task) if final_root and current_task else None
    lines.append(f"- 当期 SHP：{'有 → ' + os.path.basename(shp) if shp else '无'}")
    if pending_plan:
        lines.append(
            f"- 待确认计划：ready={pending_plan.get('ready')} "
            f"reference={pending_plan.get('reference') or '—'}"
        )
        lines.append("- 用户确认后请 dispatch pending_action.type=run_e1 且 confirmed=true")
    else:
        lines.append(
            "- 用户要做多源一致性/E1 时：先 propose_e1，确认后再 run_e1；"
            "不要用 run_pipeline 代替独立 E1。"
        )
    return "\n".join(lines)
