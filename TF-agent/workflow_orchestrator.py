# -*- coding: utf-8 -*-
"""
潮滩分析 Workflow 编排器（Phase D · 面向用户任务的编排层）。

设计目标（用户规格 一~十九）：
- 复用既有可信闭环（GEE / 推理 / E1 / M5 / PDF），不重新实现任何业务引擎；
- LLM 只解析意图（target_year / baseline_year / need_e1 / need_m5 / need_report），
  实际 DAG 由本模块确定性构建（build_analysis_workflow）；
- 单次父级确认：confirm_workflow 之后子步骤以
  confirmation_source="parent_workflow" + allowed_parent_workflow_id 放行，
  子闭环原有确认门闩保留；
- 参数变化 → PAUSED 重新确认（AOI / 年份 / 模型 / 权重 / 参数任一变化）；
- 资产链：GEE dataset_id → inference input_asset_id → prediction →
  E1/M5 → PDF；血缘 workflow_id/derived_from/produced_by 全程记录；
- 部分成功：required 步骤失败 → FAILED + 下游 BLOCKED；
  可选步骤失败/跳过 → COMPLETED_WITH_WARNINGS（永不 FAILED）；
- 账本 workflow_ledger.json 原子写（temp + os.replace），workflow_id 幂等，
  保留最近 N=50 条历史，不保存任何凭证；
- 状态机：PENDING → READY → RUNNING → VERIFYING → SUCCEEDED
           | SKIPPED / FAILED / BLOCKED / CANCELLED / REUSED
  同一时刻只允许一个重型步骤 RUNNING。
"""
from __future__ import annotations

import json
import copy
import hashlib
import os
import re
import shutil
import tempfile
import threading
import uuid
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

from agent_context_policy import describe_local_path, redact_spatial_metadata, safe_error_summary, sanitize_external_text
from asset_registry_schema import ensure_valid_registry, valid_entries

# ---------------------------------------------------------------
#  1. 常量与状态
# ---------------------------------------------------------------
WORKFLOW_SCHEMA = "analysis_workflow_plan_v1"
WORKFLOW_ID_PREFIX = "wf_"

# 步骤状态机
STEP_PENDING = "PENDING"
STEP_READY = "READY"
STEP_RUNNING = "RUNNING"
STEP_VERIFYING = "VERIFYING"
STEP_SUCCEEDED = "SUCCEEDED"
STEP_SKIPPED = "SKIPPED"
STEP_FAILED = "FAILED"
STEP_BLOCKED = "BLOCKED"
STEP_CANCELLED = "CANCELLED"
STEP_REUSED = "REUSED"  # 验收：复用既有资产

STEP_TERMINAL = {STEP_SUCCEEDED, STEP_SKIPPED, STEP_FAILED, STEP_BLOCKED,
                 STEP_CANCELLED, STEP_REUSED}

# Workflow 整体状态
WF_PENDING = "PENDING"
WF_CONFIRMED = "CONFIRMED"
WF_RUNNING = "RUNNING"
WF_SUCCEEDED = "SUCCEEDED"
WF_COMPLETED_WITH_WARNINGS = "COMPLETED_WITH_WARNINGS"
WF_FAILED = "FAILED"
WF_PAUSED = "PAUSED"
WF_CANCELLED = "CANCELLED"

WF_ACTIVE = {WF_PENDING, WF_CONFIRMED, WF_RUNNING, WF_PAUSED}

# 工具名（与既有闭环工具名一致）
TOOL_GEE_DOWNLOAD = "gee_download"
TOOL_LOCAL_INFERENCE = "local_inference"
TOOL_E1_QUALITY = "e1_quality_evaluation"
TOOL_M5_CHANGE = "m5_change_detection"
TOOL_PDF_REPORT = "pdf_report"

TOOL_LABELS = {
    TOOL_GEE_DOWNLOAD: "获取卫星影像",
    TOOL_LOCAL_INFERENCE: "潮滩智能提取",
    TOOL_E1_QUALITY: "潮滩精度评价",
    TOOL_M5_CHANGE: "潮滩变化分析",
    TOOL_PDF_REPORT: "成果报告",
}

# 重型步骤：同一时刻只允许一个 RUNNING
HEAVY_TOOLS = {TOOL_GEE_DOWNLOAD, TOOL_LOCAL_INFERENCE, TOOL_E1_QUALITY,
               TOOL_M5_CHANGE, TOOL_PDF_REPORT}

# 资产类型
ASSET_DATASET = "dataset"
ASSET_PREDICTION = "prediction"
ASSET_E1 = "e1_evaluation"
ASSET_M5 = "m5_change"
ASSET_REPORT = "report"

# 账本路径（原子写）
_DEFAULT_DATA_DIR = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
)
WORKFLOW_LEDGER_PATH = os.path.join(_DEFAULT_DATA_DIR, "workflow_ledger.json")
MAX_LEDGER_HISTORY = 50

_WORKFLOW_LOCK_GUARD = threading.RLock()
_WORKFLOW_LOCKS: Dict[str, threading.RLock] = {}
_HEAVY_SLOT_COND = threading.Condition(threading.RLock())
_ACTIVE_HEAVY_WORKFLOWS: set[str] = set()

# 会话状态键
STATE_WORKFLOW_PENDING_PLAN = "_workflow_pending_plan"
STATE_WORKFLOW_PLAN_CONFIRMED = "_workflow_plan_confirmed"   # set[workflow_id]
STATE_WORKFLOW_NOTICE = "_workflow_notice"

# 敏感键（与 task_timeline / capability_registry 同策略）
_SENSITIVE_KEY_SUBSTRINGS = ("token", "secret", "password", "api_key", "ion",
                             "key", "credential", "refresh_token")


def _now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def new_workflow_id() -> str:
    return WORKFLOW_ID_PREFIX + uuid.uuid4().hex


def _workflow_lock(workflow_id: str) -> threading.RLock:
    """Return the process-local execution lock for one workflow identity."""
    key = str(workflow_id or "") or "__missing_workflow_id__"
    with _WORKFLOW_LOCK_GUARD:
        lock = _WORKFLOW_LOCKS.get(key)
        if lock is None:
            lock = threading.RLock()
            _WORKFLOW_LOCKS[key] = lock
        return lock


def _acquire_heavy_slot(workflow_id: str) -> None:
    """Serialize heavyweight external jobs across workflow identities."""
    key = str(workflow_id or "") or "__missing_workflow_id__"
    with _HEAVY_SLOT_COND:
        while _ACTIVE_HEAVY_WORKFLOWS and key not in _ACTIVE_HEAVY_WORKFLOWS:
            _HEAVY_SLOT_COND.wait(timeout=0.5)
        _ACTIVE_HEAVY_WORKFLOWS.add(key)


def _release_heavy_slot(workflow_id: str) -> None:
    key = str(workflow_id or "") or "__missing_workflow_id__"
    with _HEAVY_SLOT_COND:
        _ACTIVE_HEAVY_WORKFLOWS.discard(key)
        _HEAVY_SLOT_COND.notify_all()


def _sensitive_filtered(text: str) -> str:
    """去掉敏感值（key/token/credentials 值），防泄漏。"""
    if not text:
        return ""
    raw = str(text)
    low = raw.lower()
    for k in _SENSITIVE_KEY_SUBSTRINGS:
        if k in low:
            return "<redacted>"
    return redact_spatial_metadata(sanitize_external_text(raw))[:500]


# ---------------------------------------------------------------
#  2. Workflow 构建（确定性 DAG · LLM 不参与图结构）
# ---------------------------------------------------------------
def build_analysis_workflow(
    *,
    aoi: Dict[str, Any],
    target_year: int,
    baseline_year: Optional[int] = None,
    capabilities: Optional[Dict[str, Any]] = None,
    assets: Optional[Dict[str, Any]] = None,
    user_intent: Optional[Dict[str, Any]] = None,
    goal: str = "",
    task_id: Optional[str] = None,
    region: str = "",
    prob: float = 0.05,
    cnt: int = 2,
    root_dir: str = "",
    final_root: str = "",
    mask_root: str = "",
    model_path: str = "",
    shp_path: str = "",
    e1_data_root: str = "",
    e1_reference: str = "师姐_2020",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    export_to: str = "local",
    gee_proxy_url: str = "",
    gee_project_id: str = "",
    workflow_id: Optional[str] = None,
) -> Dict[str, Any]:
    """确定性构建潮滩分析 Workflow（不预测 scene_count，只做结构+条件判定）。"""
    intent = dict(user_intent or {})
    need_e1 = _intent_bool(intent, "need_e1", default=None)   # None=自动（有条件则做）
    need_m5 = _intent_bool(intent, "need_m5", default=None)
    need_report = _intent_bool(intent, "need_report", default=True)
    aoi_id = str((aoi or {}).get("aoi_id") or "")
    aoi_summary = str((aoi or {}).get("label") or "") or \
        f"[AOI {aoi_id or '—'}]"

    if not task_id:
        task_id = _default_task_id(target_year, region, aoi)

    goal = goal or (
        f"分析 {aoi_summary} 的 {target_year} 年潮滩"
        + (f"，与 {baseline_year} 年成果比较变化" if baseline_year else "")
        + ("，如有多源真值则评价精度" if need_e1 is not False else "")
        + ("，最终生成 PDF 报告" if need_report else "")
    )

    steps: List[Dict[str, Any]] = []

    # 1) GEE 数据准备（必需）
    gee_condition = "gee_available"
    steps.append(_step(
        step_id="gee_download",
        tool=TOOL_GEE_DOWNLOAD,
        depends_on=[],
        required=True,
        condition=gee_condition,
    ))

    # 2) 潮滩推理（必需，依赖 GEE 数据集）
    steps.append(_step(
        step_id="local_inference",
        tool=TOOL_LOCAL_INFERENCE,
        depends_on=["gee_download"],
        required=True,
        condition="dataset_ready",
    ))

    # 3) E1 多源一致性评价（条件执行：用户明确要求则必做，否则有真值才做）
    e1_condition = None
    if need_e1 is False:
        e1_condition = "user_skipped"
    elif need_e1 is True:
        e1_condition = "reference_required"
    else:
        e1_condition = "reference_available"
    steps.append(_step(
        step_id="e1_quality",
        tool=TOOL_E1_QUALITY,
        depends_on=["local_inference"],
        required=(need_e1 is True),
        condition=e1_condition,
    ))

    # 4) M5 时空变化检测（条件执行：用户明确要求则必做，否则有基线才做）
    m5_condition = None
    if need_m5 is False or not baseline_year:
        m5_condition = "user_skipped" if need_m5 is False else "no_baseline_year"
    elif need_m5 is True:
        m5_condition = "baseline_required"
    else:
        m5_condition = "baseline_available"
    steps.append(_step(
        step_id="m5_change",
        tool=TOOL_M5_CHANGE,
        depends_on=["local_inference"],
        required=(need_m5 is True),
        condition=m5_condition,
    ))

    # 5) PDF 报告（必需；用户明确要求的 E1/M5 是硬依赖，自动步骤是软依赖）
    report_deps = ["local_inference"]
    report_optional_deps: List[str] = []
    if e1_condition not in ("user_skipped",):
        (report_deps if need_e1 is True else report_optional_deps).append("e1_quality")
    if m5_condition not in ("user_skipped", "no_baseline_year"):
        (report_deps if need_m5 is True else report_optional_deps).append("m5_change")
    steps.append(_step(
        step_id="pdf_report",
        tool=TOOL_PDF_REPORT,
        depends_on=report_deps,
        optional_depends_on=report_optional_deps,
        required=(need_report is not False),
        condition="report_required",
    ))

    capabilities = capabilities or {}
    warnings: List[str] = []
    blockers: List[str] = []

    # 能力快照（只读现状，不预测）
    _annotate_from_capabilities(steps, capabilities, warnings, blockers)

    wf_id = workflow_id or new_workflow_id()
    workflow: Dict[str, Any] = {
        "schema": WORKFLOW_SCHEMA,
        "workflow_id": wf_id,
        "task_id": task_id,
        "goal": goal,
        "context": {
            "aoi_id": aoi_id,
            "aoi_summary": aoi_summary,
            "target_year": int(target_year),
            "baseline_year": int(baseline_year) if baseline_year else None,
            "region": region,
            "prob": round(float(prob), 2),
            "cnt": int(cnt),
            "root_dir": root_dir,
            "final_root": final_root,
            "mask_root": mask_root,
            "model_path": model_path,
            "shp_path": shp_path or "",
            "e1_data_root": e1_data_root,
            "e1_reference": e1_reference,
            "start_date": start_date or "",
            "end_date": end_date or "",
            "export_to": export_to,
            "gee_proxy_url": gee_proxy_url,
            "gee_project_id": gee_project_id,
        },
        "intent": {
            "need_e1": need_e1,
            "need_m5": need_m5,
            "need_report": need_report,
        },
        "steps": steps,
        "status": WF_PENDING,
        "confirmed": False,
        "approved_params": None,      # 确认时的参数快照
        "approved_spec_hash": None,
        "revision": 1,
        "approved_revision": None,
        "confirmation_source": None,
        "allowed_parent_workflow_id": None,
        "warnings": warnings,
        "blockers": blockers,
        "errors": [],
        "assets": {},                 # step_id → asset info
        "final_result": None,
        "created_at": _now_str(),
        "updated_at": _now_str(),
    }
    workflow["spec_hash"] = _spec_hash(_extract_params_snapshot(workflow))
    if not blockers:
        workflow["status"] = WF_PENDING
    else:
        workflow["status"] = WF_PENDING  # 全局校验时再降级 BLOCKED 展示
    return workflow


def _step(
    *,
    step_id: str,
    tool: str,
    depends_on: List[str],
    optional_depends_on: Optional[List[str]] = None,
    required: bool,
    condition: Optional[str],
) -> Dict[str, Any]:
    return {
        "step_id": step_id,
        "tool": tool,
        "depends_on": list(depends_on),
        "optional_depends_on": list(optional_depends_on or []),
        "required": bool(required),
        "condition": condition,
        "status": STEP_PENDING,
        "plan_id": None,
        "asset_id": None,
        "result": None,
        "started_at": None,
        "finished_at": None,
        "error": None,
    }


def _intent_bool(intent: Dict[str, Any], key: str, default: Optional[bool]):
    v = intent.get(key, default)
    if v is None:
        return default
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in ("1", "true", "yes", "y", "on", "要", "做", "需要", "有"):
        return True
    if s in ("0", "false", "no", "n", "off", "不要", "不做", "跳过", "无"):
        return False
    return default


def _default_task_id(target_year: int, region: str, aoi: Dict[str, Any]) -> str:
    """任务命名约定 {yy}{region}（与 m5_engine.parse_task_identity 兼容）。"""
    yy = int(target_year) % 100
    region = (region or "").strip().lower()
    if not region:
        label = str((aoi or {}).get("label") or "").strip()[:12]
        region = label or "aoi"
    return f"{yy:02d}{region}"


def _annotate_from_capabilities(
    steps: List[Dict[str, Any]],
    capabilities: Dict[str, Any],
    warnings: List[str],
    blockers: List[str],
) -> None:
    """读取能力快照（gee_download / local_inference 等），附加到步骤与全局。"""
    if not isinstance(capabilities, dict):
        return
    gee_state = str(capabilities.get("gee_download") or capabilities.get(
        "GEE 数据下载") or "")
    infer_state = str(capabilities.get("local_inference") or capabilities.get(
        "本地潮滩推理") or "")
    if gee_state and "可用" in gee_state:
        blockers.append("GEE 能力不可用：缺少 Earth Engine 初始化或凭证。")
    elif gee_state and "BLOCKED" in gee_state.upper():
        blockers.append(f"GEE 能力被阻断（{gee_state}）。")
    if infer_state and "CONDITIONAL" in infer_state.upper():
        warnings.append("推理能力为条件可用：需先就绪 GEE 数据集。")
    elif infer_state and ("BLOCKED" in infer_state.upper() or "不可用" in infer_state):
        blockers.append(f"推理能力不可用（{infer_state}）。")


# ---------------------------------------------------------------
#  3. 全局校验 + 面向用户计划文本
# ---------------------------------------------------------------
def validate_analysis_workflow(
    workflow: Dict[str, Any],
    capabilities: Optional[Dict[str, Any]] = None,
    registry: Optional[Dict[str, Any]] = None,
    dataset_registry: Optional[Dict[str, Any]] = None,
) -> Tuple[bool, List[str], List[str]]:
    """全局校验：返回 (ok, blockers, warnings)。

    只读磁盘/账本现状；不预测 scene_count / 精度。
    """
    blockers: List[str] = []
    warnings: List[str] = []

    ctx = workflow.get("context") or {}
    if not ctx.get("aoi_id"):
        blockers.append("缺少有效 AOI（aoi_id 为空）。")
    if not ctx.get("target_year"):
        blockers.append("缺少目标年份（target_year）。")

    root_dir = str(ctx.get("root_dir") or "")
    final_root = str(ctx.get("final_root") or "")
    mask_root = str(ctx.get("mask_root") or "")
    model_path = str(ctx.get("model_path") or "")
    if not root_dir:
        blockers.append("缺少影像根目录。")
    elif not os.path.isdir(root_dir):
        blockers.append(f"影像根目录不存在: {describe_local_path(root_dir)}")
    if not final_root:
        blockers.append("缺少成果根目录。")
    elif not os.path.isdir(final_root):
        blockers.append(f"成果根目录不存在: {describe_local_path(final_root)}")
    if not mask_root:
        blockers.append("缺少掩膜根目录。")
    elif not os.path.isdir(mask_root):
        blockers.append(f"掩膜根目录不存在: {describe_local_path(mask_root)}")
    if not model_path:
        blockers.append("缺少模型权重路径。")
    elif not os.path.isfile(model_path):
        blockers.append(f"模型权重路径不存在: {describe_local_path(model_path)}")

    for step in workflow.get("steps") or []:
        tool = step.get("tool")
        cond = step.get("condition")
        if tool == TOOL_GEE_DOWNLOAD:
            if cond == "gee_available" and capabilities:
                if str(capabilities.get("gee_download") or "").upper() in (
                        "BLOCKED", "UNAVAILABLE"):
                    blockers.append("GEE 下载能力不可用，无法准备数据。")
        elif tool == TOOL_E1_QUALITY:
            if cond == "reference_required":
                refs = _list_e1_references(ctx.get("e1_data_root") or "")
                if not refs:
                    blockers.append(
                        f"用户要求 E1 评价，但参考真值目录无数据集: "
                        f"{ctx.get('e1_data_root') or '（空）'}"
                    )
            elif cond == "reference_available":
                refs = _list_e1_references(ctx.get("e1_data_root") or "")
                if not refs:
                    warnings.append("未探测到 E1 参考真值，E1 步骤将自动跳过。")
        elif tool == TOOL_M5_CHANGE:
            if cond == "baseline_required":
                if not _find_baseline_shp(workflow):
                    blockers.append(
                        f"用户要求 M5 变化检测，但未找到 {ctx.get('baseline_year')} "
                        f"年同区域基线成果。"
                    )
            elif cond == "baseline_available":
                if not _find_baseline_shp(workflow):
                    warnings.append(
                        "未找到基线年份同区域成果，M5 步骤将自动跳过。"
                    )
        elif tool == TOOL_PDF_REPORT:
            # 报告依赖推理产物；推理未就绪时给出提示（不阻断）
            if not _find_prediction_asset(workflow, registry=registry):
                warnings.append("PDF 报告将等待推理成果就绪后生成。")

    return len(blockers) == 0, blockers, warnings


def format_workflow_plan_for_user(workflow: Optional[Dict[str, Any]]) -> str:
    """面向用户的「潮滩分析计划」文本（只含真实信息）。"""
    if not workflow:
        return "尚未生成一键潮滩分析计划。"
    lines = ["## 一键潮滩分析 · 执行计划", ""]
    if workflow.get("status") in (WF_PENDING, WF_CONFIRMED):
        lines.append("**状态：待确认**（请回复「确认」或点击确认按钮后开始）")
    elif workflow.get("status") == WF_RUNNING:
        lines.append("**状态：处理中**")
    elif workflow.get("status") == WF_PAUSED:
        lines.append("**状态：已暂停（参数变化，需重新确认）**")
    else:
        lines.append(f"**状态：{workflow.get('status')}**")
    for b in workflow.get("blockers") or []:
        lines.append(f"- 阻塞：{b}")
    for w in workflow.get("warnings") or []:
        lines.append(f"- 注意：{w}")
    lines.append("")
    lines.append(f"- Workflow ID：`{workflow.get('workflow_id') or '—'}`")
    lines.append(f"- 任务：`{workflow.get('task_id') or '—'}`")
    ctx = workflow.get("context") or {}
    lines.append(f"- 目标年份：`{ctx.get('target_year')}`"
                 + (f" ｜ 基线年份：`{ctx.get('baseline_year')}`"
                    if ctx.get("baseline_year") else "（无基线）"))
    lines.append(f"- AOI：{ctx.get('aoi_summary') or '—'}")
    lines.append(f"- 概率阈值：`{ctx.get('prob')}` ｜ 频次阈值：`{ctx.get('cnt')}`")
    lines.append(f"- 目标：{workflow.get('goal') or '—'}")
    lines.append("")
    lines.append("步骤：")
    for i, s in enumerate(workflow.get("steps") or [], 1):
        tool = s.get("tool") or ""
        cond = s.get("condition") or ""
        cond_txt = {
            "gee_available": "需 GEE 可用",
            "dataset_ready": "需数据集就绪",
            "reference_available": "有真值则执行，否则跳过",
            "reference_required": "用户要求（必须有真值）",
            "baseline_available": "有基线则执行，否则跳过",
            "baseline_required": "用户要求（必须有基线）",
            "user_skipped": "用户明确跳过",
            "no_baseline_year": "未指定基线年份",
            "report_required": "生成 PDF 报告",
        }.get(cond, cond or "—")
        mark = "必" if s.get("required") else "选"
        lines.append(f"{i}. [{'必' if s.get('required') else '选'}] "
                     f"{TOOL_LABELS.get(tool, tool)}（{cond_txt}）")
    lines.append("")
    lines.append("确认后将按依赖顺序真实调用现有闭环（获取影像→提取→评价/变化→报告），"
                 "每一步仅基于真实工具结果登记资产与血缘。")
    return "\n".join(lines)


# ---------------------------------------------------------------
#  4. 单次父级确认门闩
# ---------------------------------------------------------------
def confirm_workflow(
    state: Dict[str, Any],
    workflow_id: str,
    *,
    approved_params: Optional[Dict[str, Any]] = None,
) -> Tuple[bool, Optional[str]]:
    """Workflow 级单次确认（父确认）。

    - 幂等：同一 workflow_id 只确认一次；
    - 确认后步骤获得 confirmation_source="parent_workflow"；
    - approved_params 为确认时的参数快照，用于参数变化检测。
    """
    plan = state.get(STATE_WORKFLOW_PENDING_PLAN)
    if not isinstance(plan, dict):
        return False, "当前没有待确认的潮滩分析 Workflow。"
    if str(plan.get("workflow_id")) != str(workflow_id):
        return False, "Workflow ID 与待确认计划不一致。"
    confirmed = state.get(STATE_WORKFLOW_PLAN_CONFIRMED)
    if not isinstance(confirmed, set):
        confirmed = set()
        state[STATE_WORKFLOW_PLAN_CONFIRMED] = confirmed
    if workflow_id in confirmed:
        return True, None  # 幂等
    confirmed.add(workflow_id)
    plan["confirmed"] = True
    plan["status"] = WF_CONFIRMED
    plan["confirmation_source"] = "parent_workflow"
    plan["approved_params"] = approved_params or _extract_params_snapshot(plan)
    plan["approved_spec_hash"] = _spec_hash(plan["approved_params"])
    plan["approved_revision"] = int(plan.get("revision") or 1)
    plan["spec_hash"] = _spec_hash(_extract_params_snapshot(plan))
    plan["updated_at"] = _now_str()
    state[STATE_WORKFLOW_PENDING_PLAN] = plan
    _ledger_upsert(plan, status=WF_CONFIRMED, note="confirmed")
    return True, None


def is_workflow_confirmed(state: Dict[str, Any], workflow_id: Optional[str]) -> bool:
    if not workflow_id:
        return False
    confirmed = state.get(STATE_WORKFLOW_PLAN_CONFIRMED)
    return isinstance(confirmed, set) and workflow_id in confirmed


def cancel_workflow(state: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    plan = state.get(STATE_WORKFLOW_PENDING_PLAN)
    if not isinstance(plan, dict):
        return False, "当前没有待取消的 Workflow。"
    wf_id = str(plan.get("workflow_id") or "")
    confirmed = state.get(STATE_WORKFLOW_PLAN_CONFIRMED)
    if isinstance(confirmed, set):
        confirmed.discard(wf_id)
        state[STATE_WORKFLOW_PLAN_CONFIRMED] = confirmed
    plan["status"] = WF_CANCELLED
    plan["updated_at"] = _now_str()
    state[STATE_WORKFLOW_PENDING_PLAN] = plan
    _ledger_upsert(plan, status=WF_CANCELLED, note="cancelled")
    return True, None


def _extract_params_snapshot(workflow: Dict[str, Any]) -> Dict[str, Any]:
    """确认时捕获的子计划参数快照（用于参数变化检测）。"""
    ctx = workflow.get("context") or {}
    return {
        "aoi_id": ctx.get("aoi_id"),
        "target_year": ctx.get("target_year"),
        "baseline_year": ctx.get("baseline_year"),
        "prob": ctx.get("prob"),
        "cnt": ctx.get("cnt"),
        "model_path": ctx.get("model_path"),
        "root_dir": ctx.get("root_dir"),
        "final_root": ctx.get("final_root"),
        "mask_root": ctx.get("mask_root"),
        "shp_path": ctx.get("shp_path"),
        "e1_data_root": ctx.get("e1_data_root"),
        "e1_reference": ctx.get("e1_reference"),
        "start_date": ctx.get("start_date"),
        "end_date": ctx.get("end_date"),
        "export_to": ctx.get("export_to"),
        "gee_proxy_url": ctx.get("gee_proxy_url"),
        "gee_project_id": ctx.get("gee_project_id"),
        "steps": [
            {"step_id": s["step_id"], "tool": s["tool"], "required": s["required"],
             "condition": s.get("condition")}
            for s in workflow.get("steps") or []
        ],
    }


def _spec_hash(snapshot: Dict[str, Any]) -> str:
    payload = json.dumps(snapshot or {}, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8", errors="replace")).hexdigest()[:24]


def check_params_changed(workflow: Dict[str, Any]) -> List[str]:
    """确认后的参数变化检测：变化 → workflow 置 PAUSED 并返回变化项。"""
    approved = workflow.get("approved_params")
    if not isinstance(approved, dict):
        return []
    current = _extract_params_snapshot(workflow)
    current_hash = _spec_hash(current)
    workflow["spec_hash"] = current_hash
    changes: List[str] = []
    for key in ("aoi_id", "target_year", "baseline_year", "prob", "cnt",
                "model_path", "root_dir", "final_root", "mask_root",
                "shp_path", "e1_data_root", "e1_reference", "start_date",
                "end_date", "export_to", "gee_proxy_url", "gee_project_id"):
        if approved.get(key) != current.get(key):
            changes.append(
                f"{key}: {approved.get(key)!r} → {current.get(key)!r}"
            )
    # 步骤结构变化（新增/删除重型步骤）
    a_steps = [s["step_id"] for s in approved.get("steps") or []]
    c_steps = [s["step_id"] for s in current.get("steps") or []]
    if a_steps != c_steps:
        changes.append(f"steps: {a_steps} → {c_steps}")
    approved_hash = workflow.get("approved_spec_hash")
    if approved_hash and current_hash != approved_hash and not changes:
        changes.append("approved_spec_hash 已变化，需要重新确认")
    if changes:
        if workflow.get("status") != WF_PAUSED:
            workflow["revision"] = int(workflow.get("revision") or 1) + 1
        workflow["status"] = WF_PAUSED
        workflow["updated_at"] = _now_str()
        _ledger_upsert(workflow, status=WF_PAUSED, note="params_changed")
    return changes


# ---------------------------------------------------------------
#  5. DAG 执行器
# ---------------------------------------------------------------
def find_ready_steps(workflow: Dict[str, Any]) -> List[Dict[str, Any]]:
    """返回当前 READY 的步骤（硬/软依赖全部终态后再 READY）。

    - 硬依赖 FAILED / BLOCKED → 本步 BLOCKED；
    - 软依赖 FAILED / BLOCKED → 等待其终态后继续，不阻断本步；
    - 条件为 user_skipped / no_baseline_year / 无参考 → SKIPPED。
    """
    steps = {s["step_id"]: s for s in workflow.get("steps") or []}
    ready: List[Dict[str, Any]] = []
    running_heavy = any(
        s.get("status") == STEP_RUNNING and s.get("tool") in HEAVY_TOOLS
        for s in steps.values()
    )

    for step in steps.values():
        status = step.get("status")
        if running_heavy and step.get("tool") in HEAVY_TOOLS:
            continue
        if status == STEP_READY:
            # 上轮已置 READY 但未执行（多个 READY 时一次只跑一个）→ 继续排队
            ready.append(step)
            continue
        if status != STEP_PENDING:
            continue

        # 依赖判定（先于条件跳过：依赖失败必须 BLOCKED，而非 SKIPPED）
        hard_deps = [steps.get(d) for d in step.get("depends_on") or []]
        optional_deps = [steps.get(d) for d in step.get("optional_depends_on") or []]
        blocked_by = [
            d["step_id"] for d in hard_deps
            if d and d.get("status") in (STEP_FAILED, STEP_BLOCKED)
        ]
        if blocked_by:
            step["status"] = STEP_BLOCKED
            step["error"] = f"依赖失败: {', '.join(blocked_by)}"
            step["finished_at"] = _now_str()
            continue
        all_deps = [d for d in hard_deps + optional_deps if d]
        deps_terminal = bool(all_deps) and all(
            d.get("status") in (STEP_SUCCEEDED, STEP_REUSED, STEP_SKIPPED)
            or d.get("status") in (STEP_FAILED, STEP_BLOCKED)
            for d in all_deps
        )
        if all_deps and not deps_terminal:
            continue  # 依赖尚未终态

        # 条件判定（自动跳过）——依赖已满足后才评估
        if _should_skip(workflow, step):
            step["status"] = STEP_SKIPPED
            step["finished_at"] = _now_str()
            continue

        step["status"] = STEP_READY
        ready.append(step)
    return ready


def _should_skip(workflow: Dict[str, Any], step: Dict[str, Any]) -> bool:
    cond = step.get("condition")
    ctx = workflow.get("context") or {}
    if cond == "user_skipped":
        return True
    if cond == "no_baseline_year":
        return True
    if cond == "reference_available":
        return not _list_e1_references(ctx.get("e1_data_root") or "")
    if cond == "baseline_available":
        return not _find_baseline_shp(workflow)
    if cond == "report_required":
        return (workflow.get("intent") or {}).get("need_report", True) is False
    return False


def _list_e1_references(data_root: str) -> List[str]:
    if not data_root or not os.path.isdir(data_root):
        return []
    try:
        import e1_engine
        return list(e1_engine.list_e1_datasets(data_root) or [])
    except Exception:  # noqa: BLE001
        return []


def _find_baseline_shp(workflow: Dict[str, Any]) -> Optional[str]:
    """在 final_root 中查找基线年份同区域成果 SHP。"""
    ctx = workflow.get("context") or {}
    final_root = str(ctx.get("final_root") or "")
    baseline_year = ctx.get("baseline_year")
    task_id = str(workflow.get("task_id") or "")
    region = str(ctx.get("region") or "")
    if not final_root or not baseline_year or not task_id:
        return None
    # 用 m5_engine 的既有查找逻辑（parse_task_identity + find_final_shp_in_task_dir）
    try:
        import m5_engine
        yy = int(baseline_year) % 100
        if region:
            base_task = f"{yy:02d}{region}"
            base_dir = os.path.join(final_root, base_task)
            shp = m5_engine.find_final_shp_in_task_dir(base_dir, base_task)
            if shp:
                return shp
        # 兜底：同区域解析
        y, r = m5_engine.parse_task_identity(task_id)
        if r:
            for cand in sorted(os.listdir(final_root)):
                cy, cr = m5_engine.parse_task_identity(cand)
                if cy is not None and cr == r and cy < int(ctx.get("target_year", 0)):
                    cd = os.path.join(final_root, cand)
                    shp = m5_engine.find_final_shp_in_task_dir(cd, cand)
                    if shp:
                        return shp
    except Exception:  # noqa: BLE001
        pass
    return None


def _find_prediction_asset(
    workflow: Dict[str, Any],
    registry: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """在 assets_registry 中查找当前任务预测资产（final_tif/final_shp）。"""
    task_id = str(workflow.get("task_id") or "")
    if not task_id:
        return None
    reg = registry if isinstance(registry, dict) else {}
    best = None
    for key, row in reg.items():
        if not isinstance(row, dict):
            continue
        if row.get("task") != task_id:
            continue
        fp = str(row.get("file_path") or "")
        if not _nonempty_path(fp):
            continue
        if fp.lower().endswith((".tif", ".tiff", ".shp")):
            if best is None or str(row.get("created_at") or "") > str(
                    best.get("created_at") or ""):
                best = dict(row)
                best["_key"] = key
    return best


# ---------------------------------------------------------------
#  6. 步骤执行适配器（转发既有闭环，不重新实现）
# ---------------------------------------------------------------
def run_workflow_step(
    step: Dict[str, Any],
    workflow: Dict[str, Any],
    *,
    exec_ctx: Optional[Dict[str, Any]] = None,
    push_log: Callable[[str], None] = print,
    stop_event: Optional[Any] = None,
) -> Dict[str, Any]:
    """统一适配器：按 tool 转发到既有闭环，返回统一结果。

    统一结果：
      {"success": bool, "status": str, "outputs": dict, "assets": list,
       "metrics": dict, "warnings": list, "error": Optional[str]}
    """
    tool = step.get("tool")
    if step.get("status") == STEP_REUSED:
        return {
            "success": True, "status": STEP_REUSED, "outputs": {},
            "assets": step.get("asset_id") and [{"asset_id": step["asset_id"]}]
            or [], "metrics": {}, "warnings": [], "error": None,
        }
    if tool == TOOL_GEE_DOWNLOAD:
        return _run_gee_step(step, workflow, exec_ctx=exec_ctx,
                             push_log=push_log, stop_event=stop_event)
    if tool == TOOL_LOCAL_INFERENCE:
        return _run_inference_step(step, workflow, exec_ctx=exec_ctx,
                                   push_log=push_log, stop_event=stop_event)
    if tool == TOOL_E1_QUALITY:
        return _run_e1_step(step, workflow, exec_ctx=exec_ctx,
                            push_log=push_log, stop_event=stop_event)
    if tool == TOOL_M5_CHANGE:
        return _run_m5_step(step, workflow, exec_ctx=exec_ctx,
                            push_log=push_log, stop_event=stop_event)
    if tool == TOOL_PDF_REPORT:
        return _run_report_step(step, workflow, exec_ctx=exec_ctx,
                                push_log=push_log, stop_event=stop_event)
    return {
        "success": False, "status": STEP_FAILED, "outputs": {},
        "assets": [], "metrics": {}, "warnings": [],
        "error": f"未知步骤工具: {tool}",
    }


def _run_gee_step(step, workflow, *, exec_ctx, push_log, stop_event) -> Dict[str, Any]:
    ctx = workflow.get("context") or {}
    override = (exec_ctx or {}).get("gee_executor")
    if override is not None:
        return _apply_override(override, step, workflow, exec_ctx)
    import gee_agent_loop as gal

    local_out_dir = os.path.join(
        str(ctx.get("root_dir") or ""), str(workflow.get("task_id") or "aoi")
    )
    # aoi 可能来自 session_state 的 AOIContext 对象（旧路径）或 dict（新路径），
    # 统一归一化为带几何的 dict；无几何时给出明确、可操作的失败原因。
    aoi_dict = (exec_ctx or {}).get("aoi")
    if hasattr(aoi_dict, "to_dict"):
        try:
            aoi_dict = aoi_dict.to_dict()
        except Exception:  # noqa: BLE001
            aoi_dict = None
    if not isinstance(aoi_dict, dict) or not aoi_dict.get("geometry"):
        # 无几何必然被 build_gee_download_plan 阻塞；直接给出可操作提示，避免
        # 晦涩的「AOI 无效（必须是合法 GeoJSON Polygon）」。
        return {
            "success": False, "status": STEP_FAILED, "outputs": {},
            "assets": [], "metrics": {}, "warnings": [],
            "error": "缺少有效 AOI 几何（请先在三维地图绘制/选择研究区域 AOI 后再执行一键潮滩分析）。",
        }
    plan = gal.build_gee_download_plan(
        task_id=str(workflow.get("task_id") or ""),
        aoi=aoi_dict,
        start_date=str(ctx.get("start_date") or f"{ctx.get('target_year')}-01-01"),
        end_date=str(ctx.get("end_date") or f"{ctx.get('target_year')}-12-31"),
        bands=(exec_ctx or {}).get("bands") or gal.DEFAULT_BANDS,
        cloud_limit=int((exec_ctx or {}).get("cloud_limit") or 60),
        export_to=str(ctx.get("export_to") or "local"),
        drive_folder=(exec_ctx or {}).get("drive_folder") or "GEE_Downloads",
        local_out_dir=local_out_dir,
        gee_proxy_url=ctx.get("gee_proxy_url") or "",
        gee_project_id=ctx.get("gee_project_id") or "",
    )
    return _run_with_child_confirmation(
        step, workflow, plan,
        execute_fn=lambda p, lg, se: gal.execute_gee_download(
            p, stop_event=se, push_log=lg,
            push_progress=(exec_ctx or {}).get("push_progress"),
            m4_engine_mod=(exec_ctx or {}).get("m4_engine_mod"),
        ),
        verify_fn=lambda p, r: gal.verify_gee_outputs(p, r),
        register_fn=lambda p, r, v: gal.register_gee_dataset_asset(
            p, r, v, registry_path=(exec_ctx or {}).get("registry_path")),
        asset_type=ASSET_DATASET,
        exec_ctx=exec_ctx,
        push_log=push_log,
        stop_event=stop_event,
    )


def _run_inference_step(step, workflow, *, exec_ctx, push_log, stop_event) -> Dict[str, Any]:
    ctx = workflow.get("context") or {}
    override = (exec_ctx or {}).get("inference_executor")
    if override is not None:
        return _apply_override(override, step, workflow, exec_ctx)
    import inference_agent_loop as ial

    dataset_id = None
    prev = _find_step(workflow, "gee_download")
    if prev and prev.get("asset_id"):
        dataset_id = prev["asset_id"]
    plan = ial.build_inference_plan(
        task_id=str(workflow.get("task_id") or ""),
        root_dir=str(ctx.get("root_dir") or ""),
        final_root=str(ctx.get("final_root") or ""),
        mask_root=str(ctx.get("mask_root") or ""),
        model_path=str(ctx.get("model_path") or ""),
        prob_threshold=float(ctx.get("prob") or 0.05),
        count_threshold=int(ctx.get("cnt") or 2),
        input_asset_id=dataset_id,
        weight_id=(exec_ctx or {}).get("weight_id"),
        device_policy=str((exec_ctx or {}).get("device_policy") or "auto"),
        shp_path=str(ctx.get("shp_path") or "") or None,
    )
    # 复用推理闭环既有执行前验证（输入/A1 单景/权重/设备 → 写入真实 device）。
    # 缺失此步会导致 plan.device 为空、加载模型时 map_location="" 而失败。
    if isinstance(plan, dict) and plan.get("ready"):
        v_ok, v_blockers, v_device = ial.validate_inference_plan(
            plan, check_weight_load=True)
        if not v_ok:
            return {
                "success": False, "status": STEP_FAILED, "outputs": {},
                "assets": [], "metrics": {}, "warnings": plan.get("warnings") or [],
                "error": "；".join(v_blockers or ["推理执行前验证未通过"]),
            }
    return _run_with_child_confirmation(
        step, workflow, plan,
        execute_fn=lambda p, lg, se: ial.execute_local_inference(
            p, stop_event=se, push_log=lg,
            push_progress=(exec_ctx or {}).get("push_progress"),
            pre_engine_mod=(exec_ctx or {}).get("pre_engine_mod"),
            post_engine_mod=(exec_ctx or {}).get("post_engine_mod"),
        ),
        verify_fn=lambda p, r: ial.verify_inference_outputs(p, r),
        register_fn=lambda p, r, v: ial.register_inference_asset(
            p, r, v, registry_path=(exec_ctx or {}).get("registry_path")),
        asset_type=ASSET_PREDICTION,
        exec_ctx=exec_ctx,
        push_log=push_log,
        stop_event=stop_event,
    )


def _run_e1_step(step, workflow, *, exec_ctx, push_log, stop_event) -> Dict[str, Any]:
    ctx = workflow.get("context") or {}
    override = (exec_ctx or {}).get("e1_executor")
    if override is not None:
        return _apply_override(override, step, workflow, exec_ctx)
    import e1_agent_loop as e1al
    import e1_engine

    task_id = str(workflow.get("task_id") or "")
    final_root = str(ctx.get("final_root") or "")
    pred = _find_prediction_asset(workflow, registry=(exec_ctx or {}).get("registry"))
    target_shp = pred.get("final_shp") if pred else None
    if not target_shp or not os.path.isfile(str(target_shp)):
        target_shp = e1al.resolve_target_shp(
            final_root, task_id, ctx.get("prob"), ctx.get("cnt"))
    if not target_shp or not os.path.isfile(str(target_shp)):
        return {
            "success": False, "status": STEP_FAILED, "outputs": {},
            "assets": [], "metrics": {}, "warnings": [],
            "error": f"E1 所需当期潮滩 SHP 不存在: {describe_local_path(target_shp) if target_shp else '（空）'}",
        }
    plan = e1al.build_e1_preflight(
        final_root=final_root,
        current_task=task_id,
        data_root=str(ctx.get("e1_data_root") or ""),
        reference=str(ctx.get("e1_reference") or "师姐_2020"),
        prob=ctx.get("prob"),
        cnt=ctx.get("cnt"),
        task_aoi_shp=str(ctx.get("shp_path") or "") or None,
    )
    if not plan.get("ready"):
        return {
            "success": False, "status": STEP_FAILED, "outputs": {},
            "assets": [], "metrics": {}, "warnings": plan.get("warnings") or [],
            "error": "；".join(plan.get("blockers") or ["E1 计划未就绪"]),
        }

    def _run(p, lg, se):
        workspace = p.get("workspace_dir") or e1_engine.workspace_for_task(
            final_root, task_id)
        report = e1_engine.run_e1_after_synthesis(
            target_shp=p.get("current_shp") or target_shp,
            roi_name=task_id,
            workspace_dir=workspace,
            data_root=p.get("data_root") or "",
            reference=p.get("reference") or "师姐_2020",
            compare_sources=p.get("compare_sources"),
            roi_path=p.get("task_aoi_shp"),
            export_disagreement_maps=p.get("export_disagreement_maps", True),
            export_multi_product_heatmap=p.get(
                "export_multi_product_heatmap", True),
            logger=lg,
            stop_callback=se,
        )
        if not report:
            return {"success": False, "report": None,
                    "error": "E1 引擎未生成报告"}
        return {"success": True, "report": report}

    return _run_engine_adapter(
        step, workflow, plan,
        run_fn=_run,
        verify_fn=lambda r: e1al.verify_e1_outputs(r.get("report")),
        register_fn=lambda p, r, v: _register_e1_workflow_asset(
            task_id, r.get("report"), exec_ctx=exec_ctx),
        asset_type=ASSET_E1,
        exec_ctx=exec_ctx,
        push_log=push_log,
        stop_event=stop_event,
    )


def _run_m5_step(step, workflow, *, exec_ctx, push_log, stop_event) -> Dict[str, Any]:
    ctx = workflow.get("context") or {}
    override = (exec_ctx or {}).get("m5_executor")
    if override is not None:
        return _apply_override(override, step, workflow, exec_ctx)
    import m5_agent_loop as m5al
    import m5_engine

    task_id = str(workflow.get("task_id") or "")
    final_root = str(ctx.get("final_root") or "")
    pred = _find_prediction_asset(workflow, registry=(exec_ctx or {}).get("registry"))
    current_shp = pred.get("final_shp") if pred else None
    if not current_shp or not os.path.isfile(str(current_shp)):
        current_shp = m5al.resolve_current_shp(
            final_root, task_id, ctx.get("prob"), ctx.get("cnt"))
    baseline_shp = _find_baseline_shp(workflow)
    if not current_shp or not os.path.isfile(str(current_shp)):
        return {
            "success": False, "status": STEP_FAILED, "outputs": {},
            "assets": [], "metrics": {}, "warnings": [],
            "error": f"M5 所需当期潮滩 SHP 不存在: {describe_local_path(current_shp) if current_shp else '（空）'}",
        }
    if not baseline_shp:
        return {
            "success": False, "status": STEP_SKIPPED, "outputs": {},
            "assets": [], "metrics": {}, "warnings": [],
            "error": "未找到基线年份同区域成果，M5 跳过。",
        }
    plan = m5al.build_m5_preflight(
        final_root=final_root,
        current_task=task_id,
        task_options=None,
        prob=ctx.get("prob"),
        cnt=ctx.get("cnt"),
        baseline_task=(exec_ctx or {}).get("baseline_task"),
        baseline_shp_override=baseline_shp,
    )

    def _run(p, lg, se):
        report = m5_engine.run_m5_after_synthesis(
            current_shp=p.get("current_shp") or current_shp,
            current_task=task_id,
            final_root=final_root,
            task_options=None,
            prob=p.get("prob"),
            cnt=p.get("cnt"),
            baseline_shp_override=baseline_shp,
            workspace_dir=final_root,
            logger=lg,
        )
        if not report:
            return {"success": False, "report": None,
                    "error": "M5 引擎未生成报告"}
        return {"success": True, "report": report}

    return _run_engine_adapter(
        step, workflow, plan,
        run_fn=_run,
        verify_fn=lambda r: m5al.verify_m5_outputs(
            r.get("report"), workspace_dir=final_root),
        register_fn=lambda p, r, v: _register_m5_workflow_asset(
            task_id, r.get("report"), exec_ctx=exec_ctx),
        asset_type=ASSET_M5,
        exec_ctx=exec_ctx,
        push_log=push_log,
        stop_event=stop_event,
    )


def _run_report_step(step, workflow, *, exec_ctx, push_log, stop_event) -> Dict[str, Any]:
    ctx = workflow.get("context") or {}
    override = (exec_ctx or {}).get("report_executor")
    if override is not None:
        return _apply_override(override, step, workflow, exec_ctx)
    if _stop_requested(stop_event):
        return _cancelled_step_result(step, "报告生成前收到取消请求")
    import asset_report_engine as are

    task_id = str(workflow.get("task_id") or "")
    registry = (exec_ctx or {}).get("registry")
    pred = _find_prediction_asset(workflow, registry=registry)
    asset_key = (pred or {}).get("_key")
    _terminal_logs = (exec_ctx or {}).get("terminal_logs")
    if callable(_terminal_logs):
        try:
            _terminal_logs = _terminal_logs()
        except Exception:
            _terminal_logs = None
    _provided_results = (exec_ctx or {}).get("execution_results")
    _execution_results = dict(_provided_results) if isinstance(_provided_results, dict) else {}
    if not _execution_results:
        _step_result_names = {
            TOOL_LOCAL_INFERENCE: "inference",
            TOOL_GEE_DOWNLOAD: "gee",
            TOOL_M5_CHANGE: "m5",
            TOOL_E1_QUALITY: "e1",
        }
        for _wf_step in workflow.get("steps") or []:
            if not isinstance(_wf_step, dict):
                continue
            _wf_result = _wf_step.get("result")
            _result_name = _step_result_names.get(_wf_step.get("tool"))
            if _result_name and isinstance(_wf_result, dict):
                _execution_results[_result_name] = _wf_result
    try:
        result = are.generate_asset_report(
            task=task_id,
            asset_key=asset_key,
            output_dir=(exec_ctx or {}).get("report_output_dir"),
            registry_path=(exec_ctx or {}).get("registry_path"),
            ref_shp=None,
            progress_callback=(exec_ctx or {}).get("push_progress"),
            terminal_logs=_terminal_logs,
            execution_results=_execution_results,
        )
    except Exception as e:  # noqa: BLE001
        return {
            "success": False, "status": STEP_FAILED, "outputs": {},
            "assets": [], "metrics": {}, "warnings": [],
            "error": f"PDF 报告生成异常: {safe_error_summary(e)}",
        }
    if not result or not getattr(result, "success", False):
        err = getattr(result, "error", "") or "PDF 报告生成失败"
        return {
            "success": False, "status": STEP_FAILED, "outputs": {},
            "assets": [], "metrics": {}, "warnings": list(getattr(result, "warnings", []) or []),
            "error": err,
        }
    if _stop_requested(stop_event):
        return _cancelled_step_result(step, "报告生成后收到取消请求，结果不会登记")
    report_path = getattr(result, "report_path", None)
    asset_id = _register_report_asset(task_id, report_path, exec_ctx=exec_ctx)
    if _stop_requested(stop_event):
        return _cancelled_step_result(step, "报告资产登记后收到取消请求，结果不会提交")
    if not asset_id:
        return {
            "success": False, "status": STEP_FAILED,
            "outputs": {"report_path": report_path},
            "assets": [],
            "metrics": {"sections": len(getattr(result, "sections", []) or [])},
            "warnings": list(getattr(result, "warnings", []) or []),
            "error": "报告文件已生成，但报告资产登记失败。",
        }
    return {
        "success": True, "status": STEP_SUCCEEDED, "outputs": {"report_path": report_path},
        "assets": [{"asset_id": asset_id, "asset_type": ASSET_REPORT,
                    "path": report_path}],
        "metrics": {"sections": len(getattr(result, "sections", []) or [])},
        "warnings": list(getattr(result, "warnings", []) or []),
        "error": None,
    }


def _apply_override(override, step, workflow, exec_ctx) -> Dict[str, Any]:
    """测试用注入执行器：必须返回统一结果 dict。"""
    res = override(step, workflow, exec_ctx or {})
    if not isinstance(res, dict):
        return {
            "success": False, "status": STEP_FAILED, "outputs": {},
            "assets": [], "metrics": {}, "warnings": [],
            "error": "override 执行器返回非 dict",
        }
    return res


def _stop_requested(stop_event: Optional[Any]) -> bool:
    return bool(stop_event is not None and stop_event.is_set())


def _cancelled_step_result(step: Dict[str, Any], reason: str) -> Dict[str, Any]:
    step["status"] = STEP_CANCELLED
    step["asset_id"] = None
    step["error"] = reason
    step["finished_at"] = _now_str()
    return {
        "success": False,
        "status": STEP_CANCELLED,
        "outputs": {},
        "assets": [],
        "metrics": {},
        "warnings": [],
        "error": reason,
    }


def _run_with_child_confirmation(
    step, workflow, plan, *, execute_fn, verify_fn, register_fn,
    asset_type, exec_ctx, push_log, stop_event,
) -> Dict[str, Any]:
    """执行子闭环（保留子确认门闩 + 父级放行）。

    仅用于 GEE / 推理（含 execute/verify/register 三件套的闭环）。
    """
    if not isinstance(plan, dict) or not plan.get("ready"):
        return {
            "success": False, "status": STEP_FAILED, "outputs": {},
            "assets": [], "metrics": {}, "warnings": plan.get("warnings") or [],
            "error": "；".join(plan.get("blockers") or ["子计划未就绪"]),
        }
    plan_id = str(plan.get("plan_id") or "")
    step["plan_id"] = plan_id
    # 父级放行：confirmation_source + allowed_parent_workflow_id
    plan["confirmation_source"] = "parent_workflow"
    plan["allowed_parent_workflow_id"] = workflow.get("workflow_id")

    step["status"] = STEP_RUNNING
    step["started_at"] = _now_str()
    push_log(f"[WF:{workflow.get('workflow_id')[:8]}] "
             f"{TOOL_LABELS.get(step.get('tool'), step.get('tool'))} 启动 "
             f"(plan_id={plan_id})")
    try:
        result = execute_fn(plan, push_log, stop_event)
    except Exception as e:  # noqa: BLE001
        step["status"] = STEP_FAILED
        safe_error = safe_error_summary(e)
        step["error"] = safe_error
        step["finished_at"] = _now_str()
        push_log(f"[WF:{workflow.get('workflow_id')[:8]}] ❌ 执行异常: {safe_error}")
        return {
            "success": False, "status": STEP_FAILED, "outputs": {},
            "assets": [], "metrics": {}, "warnings": [],
            "error": safe_error,
        }
    if _stop_requested(stop_event):
        return _cancelled_step_result(step, "步骤执行后收到取消请求，结果不会登记")
    if not result or result.get("success") is not True:
        err = sanitize_external_text((result or {}).get("error") or "子闭环执行失败")
        step["status"] = STEP_FAILED
        step["error"] = err
        step["finished_at"] = _now_str()
        push_log(f"[WF:{workflow.get('workflow_id')[:8]}] ❌ {err}")
        return {
            "success": False, "status": STEP_FAILED, "outputs": result or {},
            "assets": [], "metrics": (result or {}).get("metrics") or {},
            "warnings": (result or {}).get("warnings") or [],
            "error": err,
        }
    step["status"] = STEP_VERIFYING
    verification = verify_fn(plan, result) or {}
    if not verification.get("ok"):
        failed = [c.get("name") for c in verification.get("checks") or []
                  if not c.get("passed")]
        step["status"] = STEP_FAILED
        step["error"] = f"成果校验未通过: {', '.join(failed) or '未知'}"
        step["finished_at"] = _now_str()
        return {
            "success": False, "status": STEP_FAILED, "outputs": result,
            "assets": [], "metrics": (result or {}).get("metrics") or {},
            "warnings": (result or {}).get("warnings") or [],
            "error": step["error"],
        }
    if _stop_requested(stop_event):
        return _cancelled_step_result(step, "输出校验后收到取消请求，结果不会登记")
    registration_error = ""
    try:
        asset_id = register_fn(plan, result, verification)
    except Exception as e:  # noqa: BLE001
        asset_id = None
        registration_error = f"资产登记异常: {safe_error_summary(e)}"
    if _stop_requested(stop_event):
        return _cancelled_step_result(step, "资产登记后收到取消请求，结果不会提交")
    if not asset_id:
        step["status"] = STEP_FAILED
        step["error"] = registration_error or "校验通过但资产登记失败"
        step["finished_at"] = _now_str()
        return {
            "success": False, "status": STEP_FAILED, "outputs": result,
            "assets": [], "metrics": (result or {}).get("metrics") or {},
            "warnings": (result or {}).get("warnings") or [],
            "error": step["error"],
        }
    step["asset_id"] = asset_id
    step["status"] = STEP_SUCCEEDED
    step["finished_at"] = _now_str()
    push_log(f"[WF:{workflow.get('workflow_id')[:8]}] ✅ "
             f"{TOOL_LABELS.get(step.get('tool'), step.get('tool'))} 完成 "
             f"asset_id={asset_id}")
    return {
        "success": True, "status": STEP_SUCCEEDED, "outputs": result,
        "assets": [{"asset_id": asset_id, "asset_type": asset_type}],
        "metrics": (result or {}).get("metrics") or {},
        "warnings": (result or {}).get("warnings") or [],
        "error": None,
    }


def _run_engine_adapter(
    step, workflow, plan, *, run_fn, verify_fn, register_fn,
    asset_type, exec_ctx, push_log, stop_event,
) -> Dict[str, Any]:
    """执行 E1/M5 这类「引擎直接返回报告」的步骤。"""
    step["plan_id"] = (plan or {}).get("plan_id")
    step["status"] = STEP_RUNNING
    step["started_at"] = _now_str()
    push_log(f"[WF:{workflow.get('workflow_id')[:8]}] "
             f"{TOOL_LABELS.get(step.get('tool'), step.get('tool'))} 启动")
    try:
        raw = run_fn(plan, push_log, stop_event)
    except Exception as e:  # noqa: BLE001
        step["status"] = STEP_FAILED
        safe_error = safe_error_summary(e)
        step["error"] = safe_error
        step["finished_at"] = _now_str()
        return {
            "success": False, "status": STEP_FAILED, "outputs": {},
            "assets": [], "metrics": {}, "warnings": [], "error": safe_error,
        }
    if _stop_requested(stop_event):
        return _cancelled_step_result(step, "步骤执行后收到取消请求，结果不会登记")
    if not raw or raw.get("success") is not True:
        err = sanitize_external_text((raw or {}).get("error") or "引擎未生成结果")
        step["status"] = STEP_FAILED
        step["error"] = err
        step["finished_at"] = _now_str()
        return {
            "success": False, "status": STEP_FAILED, "outputs": {},
            "assets": [], "metrics": {}, "warnings": [], "error": err,
        }
    step["status"] = STEP_VERIFYING
    verification = verify_fn(raw) or {}
    if not verification.get("ok"):
        failed = [c.get("name") for c in verification.get("checks") or []
                  if not c.get("passed")]
        step["status"] = STEP_FAILED
        step["error"] = f"输出校验未通过: {', '.join(failed) or '未知'}"
        step["finished_at"] = _now_str()
        return {
            "success": False, "status": STEP_FAILED, "outputs": raw,
            "assets": [], "metrics": {}, "warnings": [], "error": step["error"],
        }
    if _stop_requested(stop_event):
        return _cancelled_step_result(step, "输出校验后收到取消请求，结果不会登记")
    registration_error = ""
    try:
        asset_id = register_fn(plan, raw, verification)
    except Exception as e:  # noqa: BLE001
        asset_id = None
        registration_error = f"资产登记异常: {safe_error_summary(e)}"
    if _stop_requested(stop_event):
        return _cancelled_step_result(step, "资产登记后收到取消请求，结果不会提交")
    if not asset_id:
        step["status"] = STEP_FAILED
        step["error"] = registration_error or "输出校验通过但资产登记失败"
        step["finished_at"] = _now_str()
        push_log(f"[WF:{workflow.get('workflow_id')[:8]}] ❌ {step['error']}")
        return {
            "success": False, "status": STEP_FAILED, "outputs": raw,
            "assets": [], "metrics": {}, "warnings": [],
            "error": step["error"],
        }
    step["asset_id"] = asset_id
    step["status"] = STEP_SUCCEEDED
    step["finished_at"] = _now_str()
    push_log(f"[WF:{workflow.get('workflow_id')[:8]}] ✅ "
             f"{TOOL_LABELS.get(step.get('tool'), step.get('tool'))} 完成 "
             f"asset_id={asset_id or '—'}")
    return {
        "success": True, "status": STEP_SUCCEEDED, "outputs": raw,
        "assets": [{"asset_id": asset_id, "asset_type": asset_type}]
        if asset_id else [],
        "metrics": {},
        "warnings": [],
        "error": None,
    }


def _find_step(workflow: Dict[str, Any], step_id: str) -> Optional[Dict[str, Any]]:
    for s in workflow.get("steps") or []:
        if s.get("step_id") == step_id:
            return s
    return None


# ---------------------------------------------------------------
#  7. E1 / M5 / PDF 资产登记（workflow 侧；key 约定与 app.py 一致）
# ---------------------------------------------------------------
def _nonempty_path(path: Any) -> bool:
    """Postflight gate: an asset path must exist and contain bytes."""
    try:
        return bool(path) and os.path.isfile(str(path)) and os.path.getsize(str(path)) > 0
    except OSError:
        return False


def _load_assets_registry(registry_path: Optional[str] = None) -> Dict[str, Any]:
    path = registry_path or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "assets_registry.json")
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, UnicodeError) as exc:
        _preserve_corrupt_assets_registry(path)
        raise ValueError("成果资产注册表 JSON 无效；原文件已保留，已停止写入。") from exc
    except OSError:
        raise
    valid, errors = valid_entries(data)
    if errors:
        _preserve_corrupt_assets_registry(path)
        raise ValueError("成果资产注册表记录结构无效；原文件已保留，已停止写入。")
    return valid


def _preserve_corrupt_file(path: str) -> Optional[str]:
    """保留损坏的持久化 JSON，避免后续增量写入覆盖原始证据。"""
    if not os.path.isfile(path):
        return None
    stamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    backup = f"{path}.corrupt-{stamp}"
    suffix = 1
    while os.path.exists(backup):
        backup = f"{path}.corrupt-{stamp}-{suffix}"
        suffix += 1
    try:
        shutil.copy2(path, backup)
    except OSError:
        return None
    return backup


def _preserve_corrupt_assets_registry(path: str) -> Optional[str]:
    """保留损坏资产注册表证据，避免后续增量登记覆盖原文件。"""
    return _preserve_corrupt_file(path)


def load_assets_registry(registry_path: Optional[str] = None) -> Dict[str, Any]:
    """读取成果资产注册表的公共只读入口。

    业务调用方不应自行导入不存在的 ``assets_registry`` 模块，也不应重复
    实现 JSON 读取逻辑；保留 `_load_assets_registry` 供历史内部调用兼容。
    """
    return _load_assets_registry(registry_path)


def save_assets_registry(registry: Dict[str, Any], registry_path: Optional[str] = None) -> None:
    """Validate and atomically persist the shared workflow asset registry."""
    ensure_valid_registry(registry)
    path = registry_path or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "assets_registry.json")
    _atomic_write_json(path, registry)


def _save_assets_registry(reg: Dict[str, Any],
                          registry_path: Optional[str] = None) -> None:
    path = registry_path or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "assets_registry.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ensure_valid_registry(reg)
    _atomic_write_json(path, reg)


def _register_e1_workflow_asset(task_id: str, report: Optional[Dict[str, Any]],
                                *, exec_ctx: Optional[Dict[str, Any]] = None) -> Optional[str]:
    if not report:
        return None
    import e1_agent_loop
    if not _nonempty_path((report or {}).get("report_path")):
        return None
    reg = _load_assets_registry((exec_ctx or {}).get("registry_path"))
    key = f"{task_id}_e1"
    map_path = e1_agent_loop.pick_e1_map_path(report)
    reg[key] = {
        "task": task_id,
        "method": "e1",
        "file_path": os.path.normpath(map_path) if map_path else "",
        "report_path": (report or {}).get("report_path"),
        "report_md_path": (report or {}).get("report_md_path"),
        "reference": (report or {}).get("reference"),
        "workflow_id": (exec_ctx or {}).get("workflow_id"),
        "created_at": _now_str(),
    }
    _save_assets_registry(reg, (exec_ctx or {}).get("registry_path"))
    return key


def _register_m5_workflow_asset(task_id: str, report: Optional[Dict[str, Any]],
                                *, exec_ctx: Optional[Dict[str, Any]] = None) -> Optional[str]:
    if not report:
        return None
    import m5_agent_loop
    if not _nonempty_path((report or {}).get("report_path")):
        return None
    reg = _load_assets_registry((exec_ctx or {}).get("registry_path"))
    key = f"{task_id}_m5"
    map_path = m5_agent_loop.pick_m5_map_path(report)
    spatial = (report or {}).get("spatial_outputs") or {}
    loss = spatial.get("loss_shapefile_path")
    silt = spatial.get("siltation_shapefile_path")
    if loss and str(loss) == "None":
        loss = None
    if silt and str(silt) == "None":
        silt = None
    reg[key] = {
        "task": task_id,
        "method": "m5",
        "file_path": os.path.normpath(map_path) if map_path else "",
        "report_path": (report or {}).get("report_path"),
        "loss_shp": loss if loss and os.path.isfile(str(loss)) else None,
        "siltation_shp": silt if silt and os.path.isfile(str(silt)) else None,
        "baseline_task": (report or {}).get("baseline_task"),
        "alert_level": (report or {}).get("alert_level"),
        "workflow_id": (exec_ctx or {}).get("workflow_id"),
        "created_at": _now_str(),
    }
    _save_assets_registry(reg, (exec_ctx or {}).get("registry_path"))
    return key


def _register_report_asset(task_id: str, report_path: Optional[str],
                           *, exec_ctx: Optional[Dict[str, Any]] = None) -> Optional[str]:
    # Registration is a postflight commit: an existing but empty path is not
    # evidence that report generation completed successfully.
    if (
        not report_path
        or not os.path.isfile(str(report_path))
        or os.path.getsize(str(report_path)) <= 0
    ):
        return None
    reg = _load_assets_registry((exec_ctx or {}).get("registry_path"))
    key = f"{task_id}_report"
    reg[key] = {
        "task": task_id,
        "method": "report",
        "file_path": os.path.normpath(str(report_path)),
        "report_path": os.path.normpath(str(report_path)),
        "workflow_id": (exec_ctx or {}).get("workflow_id"),
        "created_at": _now_str(),
    }
    _save_assets_registry(reg, (exec_ctx or {}).get("registry_path"))
    return key


# ---------------------------------------------------------------
#  8. Workflow 整体执行 + 最终结果 + grounded 总结
# ---------------------------------------------------------------
def run_analysis_workflow(
    workflow: Dict[str, Any],
    *,
    exec_ctx: Optional[Dict[str, Any]] = None,
    push_log: Callable[[str], None] = print,
    stop_event: Optional[Any] = None,
    max_steps: int = 100,
) -> Dict[str, Any]:
    """Execute one workflow under a process-local workflow identity lock."""
    wf_id = str(workflow.get("workflow_id") or "")
    with _workflow_lock(wf_id):
        return _run_analysis_workflow_impl(
            workflow,
            exec_ctx=exec_ctx,
            push_log=push_log,
            stop_event=stop_event,
            max_steps=max_steps,
        )


def _run_analysis_workflow_impl(
    workflow: Dict[str, Any],
    *,
    exec_ctx: Optional[Dict[str, Any]] = None,
    push_log: Callable[[str], None] = print,
    stop_event: Optional[Any] = None,
    max_steps: int = 100,
) -> Dict[str, Any]:
    """完整 DAG 执行：一次只跑一个重型步骤，直到终态。

    返回最终结果（写入 workflow["final_result"]）。
    """
    wf_id = str(workflow.get("workflow_id") or "")
    push_log(f"[WF:{wf_id[:8]}] 潮滩分析 Workflow 开始执行")

    if not workflow.get("confirmed"):
        return _finalize_result(
            workflow,
            status=workflow.get("status") or WF_PENDING,
            errors=["Workflow 尚未确认，拒绝执行任何外部步骤"],
        )
    if workflow.get("status") == WF_CANCELLED:
        return _finalize_result(
            workflow, status=WF_CANCELLED, errors=["Workflow 已取消，拒绝重新启动"]
        )
    if workflow.get("status") in {
        WF_SUCCEEDED, WF_COMPLETED_WITH_WARNINGS, WF_FAILED,
    }:
        return workflow.get("final_result") or _finalize_result(
            workflow, status=workflow["status"]
        )

    changes = check_params_changed(workflow)
    if changes:
        push_log(f"[WF:{wf_id[:8]}] ⏸ 参数变化，需要重新确认: {'; '.join(changes)}")
        return _finalize_result(
            workflow,
            status=WF_PAUSED,
            errors=[f"参数变化需重新确认: {'; '.join(changes)}"],
        )

    if workflow.get("status") == WF_PAUSED:
        return _finalize_result(
            workflow, status=WF_PAUSED, errors=["Workflow 处于暂停状态，需重新确认后执行"]
        )

    workflow["status"] = WF_RUNNING
    workflow["updated_at"] = _now_str()
    _ledger_upsert(workflow, status=WF_RUNNING, note="started")

    for _round in range(max_steps):
        if stop_event and stop_event.is_set():
            workflow["status"] = WF_CANCELLED
            _ledger_upsert(workflow, status=WF_CANCELLED, note="stopped")
            return _finalize_result(workflow, status=WF_CANCELLED,
                                    errors=["执行被用户中断"])

        if workflow.get("status") == WF_PAUSED:
            changes = check_params_changed(workflow)
            if changes:
                push_log(f"[WF:{wf_id[:8]}] ⏸ 参数变化（重新确认）: {'; '.join(changes)}")
                return _finalize_result(
                    workflow, status=WF_PAUSED,
                    errors=[f"参数变化需重新确认: {'; '.join(changes)}"])

        ready = find_ready_steps(workflow)
        pending = [s for s in workflow.get("steps") or []
                   if s.get("status") == STEP_PENDING]
        if not ready and not pending:
            break  # 全部终态
        if not ready:
            # 存在 PENDING 但无 READY：等待确认/阻塞判定
            blocked = [s for s in pending if s.get("status") == STEP_BLOCKED]
            if blocked:
                push_log(f"[WF:{wf_id[:8]}] ⛔ 步骤被阻塞: "
                         f"{', '.join(s['step_id'] for s in blocked)}")
            break
        step = ready[0]
        step["status"] = STEP_RUNNING
        _heavy_slot = step.get("tool") in HEAVY_TOOLS
        if _heavy_slot:
            _acquire_heavy_slot(wf_id)
        try:
            result = run_workflow_step(step, workflow, exec_ctx=exec_ctx,
                                       push_log=push_log, stop_event=stop_event)
        finally:
            if _heavy_slot:
                _release_heavy_slot(wf_id)
        if stop_event and stop_event.is_set():
            step["status"] = STEP_CANCELLED
            step["asset_id"] = None
            step["error"] = "步骤完成后收到取消请求，迟到成功不予提交"
            result = dict(result or {})
            result.update({
                "success": False,
                "status": STEP_CANCELLED,
                "assets": [],
                "error": step["error"],
            })
            step["result"] = result
            step["finished_at"] = _now_str()
            workflow["status"] = WF_CANCELLED
            workflow["updated_at"] = _now_str()
            _ledger_upsert(workflow, status=WF_CANCELLED, note="stopped_after_step")
            return _finalize_result(
                workflow, status=WF_CANCELLED, errors=[step["error"]]
            )
        status = result.get("status") or step.get("status")
        if result.get("success"):
            step["status"] = status if status in STEP_TERMINAL else STEP_SUCCEEDED
        else:
            # 适配器返回 SKIPPED（可选步骤条件不满足）时保持跳过
            step["status"] = STEP_SKIPPED if status == STEP_SKIPPED else STEP_FAILED
            step["error"] = result.get("error")
        # 统一结果同步回步骤（含 override 路径的资产）
        res_assets = result.get("assets") or []
        if res_assets and not step.get("asset_id"):
            step["asset_id"] = res_assets[0].get("asset_id")
        if result.get("plan_id") and not step.get("plan_id"):
            step["plan_id"] = result["plan_id"]
        step["result"] = result
        step["finished_at"] = _now_str()
        wf_status = _evaluate_workflow_status(workflow)
        workflow["status"] = wf_status
        workflow["updated_at"] = _now_str()
        _ledger_upsert(workflow, status=wf_status,
                       note=f"step:{step['step_id']}->{step['status']}")
        if wf_status in (WF_FAILED,):
            # 级联：把依赖失败的下游步骤标记 BLOCKED（不再执行）
            find_ready_steps(workflow)
            wf_status = _evaluate_workflow_status(workflow)
            workflow["status"] = wf_status
            workflow["updated_at"] = _now_str()
            _ledger_upsert(workflow, status=wf_status,
                           note="cascade_blocked")
            break

    final = _evaluate_workflow_status(workflow)
    workflow["status"] = final
    workflow["updated_at"] = _now_str()
    _ledger_upsert(workflow, status=final, note="finished")
    try:
        record_workflow_lineage(workflow)
    except Exception as exc:  # noqa: BLE001
        push_log(f"[WF:{wf_id[:8]}] ⚠️ 血缘记录失败: {safe_error_summary(exc)}")
    return _finalize_result(workflow, status=final)


def _evaluate_workflow_status(workflow: Dict[str, Any]) -> str:
    """部分成功语义：required 失败 → FAILED；可选失败/跳过 → WITH_WARNINGS。"""
    steps = workflow.get("steps") or []
    required_failed = any(
        s.get("required") and s.get("status") in (STEP_FAILED, STEP_BLOCKED)
        for s in steps
    )
    if required_failed:
        return WF_FAILED
    optional_failed = any(
        not s.get("required") and s.get("status") in (STEP_FAILED, STEP_BLOCKED)
        for s in steps
    )
    pending = any(s.get("status") == STEP_PENDING for s in steps)
    if pending:
        return WF_RUNNING
    succeeded = any(s.get("status") in (STEP_SUCCEEDED, STEP_REUSED)
                    for s in steps)
    if not succeeded:
        return WF_FAILED
    if optional_failed or any(s.get("status") == STEP_SKIPPED for s in steps):
        return WF_COMPLETED_WITH_WARNINGS
    return WF_SUCCEEDED


def workflow_result_timeline_status(status: Optional[str]) -> str:
    """Map a workflow result to the timeline status without hiding warnings."""
    if status == WF_SUCCEEDED:
        return "SUCCEEDED"
    if status == WF_COMPLETED_WITH_WARNINGS:
        return "WARNING"
    return "FAILED"


def _finalize_result(
    workflow: Dict[str, Any],
    status: str,
    errors: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """生成最终结果（含资产索引 + grounded 总结）。"""
    assets: Dict[str, Any] = {}
    step_assets = {}
    for s in workflow.get("steps") or []:
        if s.get("asset_id"):
            step_assets[s["step_id"]] = {
                "asset_id": s["asset_id"],
                "tool": s.get("tool"),
                "status": s.get("status"),
            }
            if s.get("tool") == TOOL_GEE_DOWNLOAD:
                assets["dataset"] = s["asset_id"]
            elif s.get("tool") == TOOL_LOCAL_INFERENCE:
                assets["prediction"] = s["asset_id"]
            elif s.get("tool") == TOOL_E1_QUALITY:
                assets["e1"] = s["asset_id"]
            elif s.get("tool") == TOOL_M5_CHANGE:
                assets["m5"] = s["asset_id"]
            elif s.get("tool") == TOOL_PDF_REPORT:
                assets["report"] = s["asset_id"]

    warnings: List[str] = []
    for s in workflow.get("steps") or []:
        if s.get("status") == STEP_SKIPPED:
            warnings.append(
                f"步骤 {TOOL_LABELS.get(s.get('tool'), s.get('tool'))} 已跳过"
                f"（条件：{s.get('condition') or '—'}）")
        elif s.get("status") in (STEP_FAILED, STEP_BLOCKED) and not s.get("required"):
            issue = "被阻塞" if s.get("status") == STEP_BLOCKED else "失败"
            warnings.append(
                f"可选步骤 {TOOL_LABELS.get(s.get('tool'), s.get('tool'))} {issue}"
                f"（不影响主流程）: {s.get('error') or '—'}")

    result: Dict[str, Any] = {
        "workflow_id": workflow.get("workflow_id"),
        "status": status,
        "task_id": workflow.get("task_id"),
        "steps": {
            s["step_id"]: {
                "tool": s.get("tool"),
                "status": s.get("status"),
                "required": s.get("required"),
                "plan_id": s.get("plan_id"),
                "asset_id": s.get("asset_id"),
                "error": s.get("error"),
            }
            for s in workflow.get("steps") or []
        },
        "assets": assets,
        "step_assets": step_assets,
        "warnings": warnings + list(workflow.get("warnings") or []),
        "errors": list(errors or []) + list(workflow.get("errors") or []),
        "summary": summarize_workflow_result_for_chat(workflow, status),
        "finished_at": _now_str(),
    }
    workflow["final_result"] = result
    workflow["status"] = status
    return result


def summarize_workflow_result_for_chat(
    workflow: Dict[str, Any],
    status: Optional[str] = None,
) -> str:
    """Grounded 总结：只使用真实子步骤结果/资产，禁止臆测。"""
    status = status or workflow.get("status") or WF_PENDING
    ctx = workflow.get("context") or {}
    task_id = str(workflow.get("task_id") or "")
    try:
        from ui_labels import get_status_label as _sl

        def _status_label(v: str) -> str:
            return _sl(v)

        status_txt = _status_label(status)
    except Exception:  # noqa: BLE001
        def _status_label(v: str) -> str:  # type: ignore[misc]
            return str(v)

        status_txt = _status_label(status)
    lines = [f"## 一键潮滩分析 · {status_txt}", ""]
    lines.append(f"- Workflow ID：`{workflow.get('workflow_id') or '—'}`")
    lines.append(f"- 任务：`{task_id or '—'}`")
    lines.append(
        f"- 目标年份：`{ctx.get('target_year')}`"
        + (f" ｜ 基线：`{ctx.get('baseline_year')}`" if ctx.get("baseline_year") else "")
    )
    lines.append(f"- 目标：{workflow.get('goal') or '—'}")
    lines.append("")
    lines.append("步骤结果（均来自真实工具输出）：")
    for s in workflow.get("steps") or []:
        tool = TOOL_LABELS.get(s.get("tool"), s.get("tool"))
        st = s.get("status")
        plan_id = s.get("plan_id")
        asset_id = s.get("asset_id")
        extra = ""
        if st == STEP_SUCCEEDED and s.get("result"):
            r = s.get("result") or {}
            metrics = r.get("metrics") or {}
            if s.get("tool") == TOOL_GEE_DOWNLOAD:
                extra = f"（共 {metrics.get('scene_count', '?')} 景）"
            elif s.get("tool") == TOOL_LOCAL_INFERENCE:
                extra = f"（耗时 {metrics.get('elapsed_seconds', '?')}s）"
            elif s.get("tool") == TOOL_PDF_REPORT:
                out = r.get("outputs") or {}
                if out.get("report_path"):
                    extra = f"（报告：{os.path.basename(str(out['report_path']))}）"
        elif st == STEP_SKIPPED:
            extra = "（跳过）"
        elif st == STEP_FAILED:
            extra = f"（失败：{_sensitive_filtered(s.get('error'))}）"
        elif st == STEP_REUSED:
            extra = "（复用既有资产）"
        elif st == STEP_BLOCKED:
            extra = "（被阻塞）"
        line = f"- {tool}：**{_status_label(st)}**{extra}"
        if plan_id:
            line += f" `plan={plan_id[:8]}`"
        if asset_id:
            line += f" `asset={asset_id}`"
        lines.append(line)
    lines.append("")
    if status == WF_SUCCEEDED:
        lines.append("潮滩分析任务已完成，全部必需步骤成功。")
    elif status == WF_COMPLETED_WITH_WARNINGS:
        lines.append("任务完成但含警告（可选步骤跳过/失败，不影响必需成果）。")
    elif status == WF_FAILED:
        lines.append("任务失败：必需步骤未通过，请查看阻塞原因。")
    elif status == WF_PAUSED:
        lines.append("任务暂停：参数已变化，请重新确认后继续。")
    lines.append("")
    lines.append("以上结果均来自本次分析各步骤的真实工具输出，而非模型臆测。")
    return "\n".join(lines)


# ---------------------------------------------------------------
#  9. 血缘（lineage）
# ---------------------------------------------------------------
LINEAGE_KEYS = (
    "workflow_id", "derived_from", "produced_by", "asset_type",
)

def enrich_asset_metadata(
    entry: Dict[str, Any],
    *,
    workflow_id: Optional[str] = None,
    derived_from: Optional[List[str]] = None,
    produced_by: Optional[Dict[str, Any]] = None,
    asset_type: Optional[str] = None,
) -> Dict[str, Any]:
    """扩展资产元数据（workflow_id/derived_from/produced_by/asset_type）。"""
    out = dict(entry or {})
    if workflow_id:
        out["workflow_id"] = workflow_id
    if derived_from is not None:
        out["derived_from"] = [str(x) for x in derived_from]
    if produced_by:
        out["produced_by"] = {
            "tool": produced_by.get("tool"),
            "plan_id": produced_by.get("plan_id"),
            "step_id": produced_by.get("step_id"),
            "code_commit": produced_by.get("code_commit"),
        }
    if asset_type:
        out["asset_type"] = asset_type
    return out


def get_asset_lineage(
    asset_id: str,
    *,
    registry: Optional[Dict[str, Any]] = None,
    dataset_registry: Optional[Dict[str, Any]] = None,
    lineage_index: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """返回资产血缘：{ancestors, children, workflow_id, ...}。

    读取 assets_registry + dataset_assets_registry + lineage 索引
    （血缘索引由 record_asset_lineage 维护，不引入图数据库）。
    """
    asset_id = str(asset_id or "")
    if not asset_id:
        return {"asset_id": asset_id, "ancestors": [], "children": [],
                "workflow_id": None, "found": False}

    index = lineage_index if isinstance(lineage_index, dict) else (
        _load_lineage_index() or {})
    entry = index.get(asset_id) or {}
    workflow_id = entry.get("workflow_id")
    derived_from = list(entry.get("derived_from") or [])

    # 反向索引：children
    children: List[str] = []
    for aid, row in index.items():
        if asset_id in (row.get("derived_from") or []):
            children.append(aid)

    # 祖先递归
    ancestors: List[str] = []
    seen: set = set()

    def _walk(aid: str) -> None:
        if aid in seen:
            return
        seen.add(aid)
        row = index.get(aid) or {}
        for p in row.get("derived_from") or []:
            ancestors.append(p)
            _walk(p)

    _walk(asset_id)
    ancestors = list(dict.fromkeys(ancestors))

    base = {
        "asset_id": asset_id,
        "ancestors": ancestors,
        "children": sorted(children),
        "workflow_id": workflow_id,
        "derived_from": derived_from,
        "produced_by": entry.get("produced_by"),
        "asset_type": entry.get("asset_type"),
        "found": bool(index.get(asset_id)),
    }
    return base


def _load_lineage_index() -> Dict[str, Any]:
    path = os.path.join(_DEFAULT_DATA_DIR, "workflow_lineage.json")
    if os.path.isfile(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, UnicodeError) as exc:
            _preserve_corrupt_file(path)
            raise ValueError("工作流血缘索引 JSON 无效；原文件已保留，已停止写入。") from exc
        if not isinstance(data, dict) or any(not isinstance(v, dict) for v in data.values()):
            _preserve_corrupt_file(path)
            raise ValueError("工作流血缘索引记录结构无效；原文件已保留，已停止写入。")
        return data
    return {}


def _save_lineage_index(index: Dict[str, Any]) -> None:
    path = os.path.join(_DEFAULT_DATA_DIR, "workflow_lineage.json")
    _atomic_write_json(path, index)


def record_asset_lineage(
    asset_id: str,
    *,
    asset_type: str,
    workflow_id: Optional[str] = None,
    derived_from: Optional[List[str]] = None,
    produced_by: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """记录一条血缘（幂等：同 asset_id 覆盖更新）。"""
    index = _load_lineage_index()
    entry = dict(index.get(asset_id) or {})
    entry["asset_id"] = asset_id
    entry["asset_type"] = asset_type or entry.get("asset_type")
    if workflow_id:
        entry["workflow_id"] = workflow_id
    if derived_from is not None:
        entry["derived_from"] = [str(x) for x in derived_from]
    if produced_by:
        entry["produced_by"] = {
            "tool": produced_by.get("tool"),
            "plan_id": produced_by.get("plan_id"),
            "step_id": produced_by.get("step_id"),
            "code_commit": produced_by.get("code_commit"),
        }
    entry["updated_at"] = _now_str()
    index[asset_id] = entry
    _save_lineage_index(index)
    return dict(entry)


def record_workflow_lineage(workflow: Dict[str, Any]) -> None:
    """执行结束后，按步骤资产链记录完整血缘。"""
    wf_id = str(workflow.get("workflow_id") or "")
    steps = workflow.get("steps") or []
    code_commit = _git_head_or_unknown()
    for s in steps:
        asset_id = s.get("asset_id")
        if not asset_id:
            continue
        derived = []
        for d in s.get("depends_on") or []:
            dep = _find_step(workflow, d)
            if dep and dep.get("asset_id"):
                derived.append(dep["asset_id"])
        record_asset_lineage(
            asset_id,
            asset_type=_asset_type_for_tool(s.get("tool")),
            workflow_id=wf_id,
            derived_from=derived,
            produced_by={
                "tool": s.get("tool"),
                "plan_id": s.get("plan_id"),
                "step_id": s.get("step_id"),
                "code_commit": code_commit,
            },
        )


def _asset_type_for_tool(tool: Optional[str]) -> str:
    return {
        TOOL_GEE_DOWNLOAD: ASSET_DATASET,
        TOOL_LOCAL_INFERENCE: ASSET_PREDICTION,
        TOOL_E1_QUALITY: ASSET_E1,
        TOOL_M5_CHANGE: ASSET_M5,
        TOOL_PDF_REPORT: ASSET_REPORT,
    }.get(tool or "", "")


def _git_head_or_unknown() -> str:
    """Read a short commit id without forking from a threaded Streamlit process.

    Calling ``subprocess.run(['git', ...])`` after Streamlit/Gateway starts
    worker threads can trigger macOS fork-with-threads crashes (especially on
    Python 3.11).  The git metadata is plain text, so resolve HEAD and packed
    refs directly and fall back to ``unknown`` when the checkout is unavailable.
    """
    candidate = os.environ.get("CSTF_GIT_COMMIT") or os.environ.get("GIT_COMMIT")
    if not candidate:
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        git_path = os.path.join(repo_root, ".git")
        try:
            if os.path.isfile(git_path):
                with open(git_path, "r", encoding="utf-8") as handle:
                    git_path = handle.read().strip()
                if git_path.startswith("gitdir:"):
                    git_path = git_path.split(":", 1)[1].strip()
                    if not os.path.isabs(git_path):
                        git_path = os.path.normpath(os.path.join(repo_root, git_path))
            head_path = os.path.join(git_path, "HEAD")
            with open(head_path, "r", encoding="utf-8") as handle:
                head = handle.read().strip()
            if head.startswith("ref: "):
                ref = head[5:].strip()
                ref_path = os.path.join(git_path, ref)
                if os.path.isfile(ref_path):
                    with open(ref_path, "r", encoding="utf-8") as handle:
                        candidate = handle.read().strip()
                else:
                    packed = os.path.join(git_path, "packed-refs")
                    if os.path.isfile(packed):
                        with open(packed, "r", encoding="utf-8") as handle:
                            for line in handle:
                                parts = line.strip().split(" ", 1)
                                if len(parts) == 2 and parts[1] == ref:
                                    candidate = parts[0]
                                    break
            else:
                candidate = head
        except OSError:  # noqa: BLE001
            candidate = ""
    match = re.match(r"^[0-9a-fA-F]{7,40}$", str(candidate or "").strip())
    if match:
        return match.group(0)[:7].lower()
    return "unknown"


# ---------------------------------------------------------------
#  10. 账本（原子写 · workflow_id 幂等 · 保留 N=50）
# ---------------------------------------------------------------
def _atomic_write_json(path: str, data: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, tmp = tempfile.mkstemp(
        prefix=".wf_ledger_", suffix=".tmp", dir=os.path.dirname(path))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)
    except Exception:  # noqa: BLE001
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass
        raise


def load_workflow_ledger(ledger_path: Optional[str] = None) -> Dict[str, Any]:
    path = ledger_path or WORKFLOW_LEDGER_PATH
    if os.path.isfile(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, UnicodeError) as exc:
            _preserve_corrupt_file(path)
            raise ValueError("工作流账本 JSON 无效；原文件已保留，已停止写入。") from exc
        if not isinstance(data, dict) or any(not isinstance(v, dict) for v in data.values()):
            _preserve_corrupt_file(path)
            raise ValueError("工作流账本记录结构无效；原文件已保留，已停止写入。")
        return data
    return {}


def save_workflow_ledger(ledger: Dict[str, Any],
                         ledger_path: Optional[str] = None) -> None:
    _atomic_write_json(ledger_path or WORKFLOW_LEDGER_PATH, ledger)


def _ledger_safe_copy(value: Any, key: str = "") -> Any:
    """Copy recovery data while removing credential-like fields."""
    low_key = str(key or "").lower()
    if any(token in low_key for token in ("token", "secret", "password", "api_key", "credential")):
        return "<redacted>"
    if isinstance(value, dict):
        return {str(k): _ledger_safe_copy(v, str(k)) for k, v in value.items()}
    if isinstance(value, list):
        return [_ledger_safe_copy(item, key) for item in value]
    if isinstance(value, tuple):
        return [_ledger_safe_copy(item, key) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        text = str(value) if isinstance(value, str) else value
        if isinstance(text, str):
            text = re.sub(r"(?i)(https?://)([^/@\s]+):([^/@\s]+)@", r"\1<redacted>@", text)
            return text[:20_000]
        return text
    return str(value)[:20_000]


def _ledger_upsert(workflow: Dict[str, Any], *, status: str, note: str = "") -> None:
    """按 workflow_id 幂等更新账本；保留最近 MAX_LEDGER_HISTORY 条。"""
    wf_id = str(workflow.get("workflow_id") or "")
    if not wf_id:
        return
    ledger = load_workflow_ledger()
    row = dict(ledger.get(wf_id) or {})
    row.update({
        "schema": workflow.get("schema") or WORKFLOW_SCHEMA,
        "workflow_id": wf_id,
        "task_id": workflow.get("task_id"),
        "status": status,
        "goal": workflow.get("goal"),
        "context": _ledger_safe_copy(workflow.get("context") or {}),
        "intent": _ledger_safe_copy(workflow.get("intent") or {}),
        "confirmed": bool(workflow.get("confirmed")),
        "confirmation_source": workflow.get("confirmation_source"),
        "approved_params": _ledger_safe_copy(workflow.get("approved_params")),
        "approved_spec_hash": workflow.get("approved_spec_hash"),
        "spec_hash": workflow.get("spec_hash"),
        "revision": int(workflow.get("revision") or 1),
        "approved_revision": workflow.get("approved_revision"),
        "steps": {
            s["step_id"]: {
                "tool": s.get("tool"),
                "status": s.get("status"),
                "plan_id": s.get("plan_id"),
                "asset_id": s.get("asset_id"),
                "depends_on": list(s.get("depends_on") or []),
                "optional_depends_on": list(s.get("optional_depends_on") or []),
                "required": bool(s.get("required")),
                "condition": s.get("condition"),
                "error": _ledger_safe_copy(s.get("error")),
                "started_at": s.get("started_at"),
                "finished_at": s.get("finished_at"),
            }
            for s in workflow.get("steps") or []
        },
        "warnings": _ledger_safe_copy(workflow.get("warnings") or []),
        "blockers": _ledger_safe_copy(workflow.get("blockers") or []),
        "errors": _ledger_safe_copy(workflow.get("errors") or []),
        "note": note,
        "updated_at": _now_str(),
    })
    if "created_at" not in row:
        row["created_at"] = workflow.get("created_at") or _now_str()
    ledger[wf_id] = row
    # 保留最近 N 条
    if len(ledger) > MAX_LEDGER_HISTORY:
        ordered = sorted(
            ledger.items(),
            key=lambda kv: str(kv[1].get("updated_at") or ""),
            reverse=True,
        )[:MAX_LEDGER_HISTORY]
        ledger = dict(ordered)
    save_workflow_ledger(ledger)


def load_workflow(workflow_id: str,
                  ledger_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """从账本恢复 workflow（rerun/refresh 幂等）。"""
    ledger = load_workflow_ledger(ledger_path)
    row = ledger.get(str(workflow_id) or "")
    if not row:
        return None
    wf: Dict[str, Any] = {
        "schema": row.get("schema") or WORKFLOW_SCHEMA,
        "workflow_id": row["workflow_id"],
        "task_id": row.get("task_id"),
        "goal": row.get("goal"),
        "status": row.get("status"),
        "context": copy.deepcopy(row.get("context") or {}),
        "intent": copy.deepcopy(row.get("intent") or {}),
        "steps": [
            {
                "step_id": k, "tool": v.get("tool"),
                "status": v.get("status"),
                "plan_id": v.get("plan_id"),
                "asset_id": v.get("asset_id"),
                "depends_on": list(v.get("depends_on") or []),
                "optional_depends_on": list(v.get("optional_depends_on") or []),
                "required": bool(v.get("required")),
                "condition": v.get("condition"),
                "result": None, "error": v.get("error"),
                "started_at": v.get("started_at"),
                "finished_at": v.get("finished_at"),
            }
            for k, v in (row.get("steps") or {}).items()
        ],
        "confirmed": bool(row.get("confirmed")) or row.get("status") in (WF_CONFIRMED, WF_RUNNING),
        "confirmation_source": row.get("confirmation_source"),
        "approved_params": copy.deepcopy(row.get("approved_params")),
        "approved_spec_hash": row.get("approved_spec_hash"),
        "spec_hash": row.get("spec_hash"),
        "revision": int(row.get("revision") or 1),
        "approved_revision": row.get("approved_revision"),
        "warnings": list(row.get("warnings") or []),
        "blockers": list(row.get("blockers") or []),
        "errors": list(row.get("errors") or []),
        "assets": {}, "final_result": None,
        "created_at": row.get("created_at"),
        "updated_at": row.get("updated_at"),
    }
    return wf
