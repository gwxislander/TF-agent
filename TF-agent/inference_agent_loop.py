# -*- coding: utf-8 -*-
"""
本地潮滩推理 Agent 可信执行闭环：计划 → 校验 → 确认 → 执行 → 验证 → 登记 → 地图 → 时间线 → 真实回复。

范围：只做「本地潮滩推理」的可信执行闭环；复用现有 pre_engine / post_engine，
不重新实现模型构建 / 预处理 / 推理 / 后处理算法。

可信铁律：
- 输入 / 权重路径只来自侧栏合法值或已登记资产，禁止 LLM 拼接任意路径。
- 全部指标（耗时 / 瓦片数 / 设备 / 产物）必须来自本次真实执行，禁止虚构。
- 验证失败不登记、不回复「推理完成」。
- 本模块不依赖 Streamlit；重型依赖（torch / rasterio / geopandas / pre_engine / post_engine）
  在函数内懒加载，保证模块可被任意环境导入并便于单元测试。
"""
from __future__ import annotations

import glob
import json
import ntpath
import os
import re
import shutil
import subprocess
import time
import uuid
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

from agent_context_policy import safe_error_summary, safe_local_path_label, sanitize_external_text

# ---- 时间线阶段映射（与 task_timeline.PHASES 一致） ----
INFERENCE_TIMELINE_PHASES: Tuple[str, ...] = (
    "PLAN",
    "VALIDATE",
    "CONFIRM",
    "QUEUED",
    "INFERENCE",
    "POST_PROCESS",
    "VERIFY",
    "REGISTER",
    "MAP",
    "REPORT",
)

# ---- session_state 键约定（与 M5/E1 隔离） ----
STATE_INFERENCE_PENDING_PLAN = "_inference_pending_plan"
STATE_INFERENCE_PLAN_CONFIRMED = "_inference_plan_confirmed"  # set of plan_id

# 参数合法范围（与侧栏 slider 一致）
PROB_MIN, PROB_MAX = 0.01, 0.50
CNT_MIN, CNT_MAX = 1, 10

MODEL_ID = "cdnet_resnet50"
TOOL_NAME = "local_tidal_flat_inference"

# 可选模型权重注册表（weight_id → 权重路径）。当前无独立注册表文件时保持空，
# 权重路径一律来自侧栏 model_path 直接值（合法配置）。
MODEL_REGISTRY_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "model_weights_registry.json"
)

# 资产账本路径（与 app.py 的 ASSET_REGISTRY_PATH 同路径，便于兼容 find_asset）
ASSET_REGISTRY_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "assets_registry.json"
)


# =======================================================
#  工具函数
# =======================================================
def _now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _nonempty_file(path: object) -> bool:
    """Return true only for a readable artifact with at least one byte."""
    try:
        return bool(path) and os.path.isfile(str(path)) and os.path.getsize(str(path)) > 0
    except OSError:
        return False


def new_plan_id() -> str:
    return uuid.uuid4().hex


def rel_path(p: str) -> str:
    """Return a usable relative artifact path without echoing foreign paths."""
    raw = str(p or "")
    if not raw:
        return ""
    # Windows drive/UNC paths can be passed through a POSIX test host; those
    # are never usable there and must become a basename-only label.
    if ntpath.splitdrive(raw)[0] or raw.startswith(("\\\\", "//")):
        return safe_local_path_label(raw)
    try:
        return os.path.relpath(raw)
    except (OSError, ValueError):
        return safe_local_path_label(raw)


def basename_or_none(p: Optional[str]) -> str:
    return os.path.basename(str(p or "")) or "—"


def git_head() -> Optional[str]:
    """当前代码提交哈希（git rev-parse HEAD）；非 git 环境返回 None。"""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5,
            cwd=os.path.dirname(os.path.abspath(__file__)),
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:  # noqa: BLE001
        pass
    return None


def resolve_weight_path(weight_id: Optional[str], model_path: Optional[str]) -> Tuple[Optional[str], List[str]]:
    """
    解析权重路径（白名单规则）：
    1. weight_id 命中模型权重注册表 → 使用注册路径；
    2. 否则 model_path 为本地现有文件 → 使用（侧栏合法值）；
    3. 其余一律拒绝（含 URL / 任意拼接路径）。
    返回 (weight_path, warnings)。
    """
    warnings: List[str] = []
    weight_path: Optional[str] = None

    if weight_id:
        try:
            if os.path.isfile(MODEL_REGISTRY_PATH):
                with open(MODEL_REGISTRY_PATH, "r", encoding="utf-8") as f:
                    reg = json.load(f)
                entry = reg.get(weight_id)
                if entry:
                    wp = entry.get("weight_path") or entry.get("path")
                    if wp and os.path.isfile(str(wp)):
                        weight_path = os.path.normpath(str(wp))
                    else:
                        warnings.append(f"权重注册表条目 {weight_id} 的路径无效，忽略。")
        except Exception as e:  # noqa: BLE001
            warnings.append(f"权重注册表读取失败（{safe_error_summary(e)}），回退侧栏配置。")

    mp = str(model_path or "").strip().strip('"').strip("'")
    if weight_path is None:
        if not mp:
            return None, warnings
        low = mp.lower()
        if low.startswith(("http://", "https://", "ftp://", "file://")):
            warnings.append("权重路径为网络地址，已拒绝。")
            return None, warnings
        if not os.path.isfile(mp):
            warnings.append(f"权重文件不存在: {rel_path(mp)}")
            return None, warnings
        weight_path = os.path.normpath(mp)

    if weight_path and not os.path.isfile(weight_path):
        warnings.append(f"权重文件不存在: {rel_path(weight_path)}")
        return None, warnings
    return weight_path, warnings


def _read_model_registry() -> Dict[str, Any]:
    try:
        if os.path.isfile(MODEL_REGISTRY_PATH):
            with open(MODEL_REGISTRY_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:  # noqa: BLE001
        pass
    return {}


# =======================================================
#  1. 计划构建（§2.2 / 用户规格 §三）
# =======================================================
def build_inference_plan(
    *,
    task_id: str,
    root_dir: str,
    final_root: str,
    mask_root: str,
    model_path: str,
    prob_threshold: float,
    count_threshold: int,
    input_asset_id: Optional[str] = None,
    weight_id: Optional[str] = None,
    device_policy: str = "auto",
    shp_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    生成本地潮滩推理执行计划。

    - task_id 必须存在于 root_dir 的子目录（否则 blocker）；
    - weight_path 只接受注册表或侧栏合法值（见 resolve_weight_path）；
    - prob / cnt 越界 → blocker。
    """
    root_dir = os.path.normpath(str(root_dir or "").strip())
    final_root = os.path.normpath(str(final_root or "").strip())
    mask_root = os.path.normpath(str(mask_root or "").strip())
    task_id = (str(task_id or "").strip())
    device_policy = str(device_policy or "auto").strip().lower()

    blockers: List[str] = []
    warnings: List[str] = []

    if not task_id:
        blockers.append("未指定目标任务（task_id）。")
    if not root_dir or not os.path.isdir(root_dir):
        blockers.append(f"原始影像根目录不存在: {rel_path(root_dir) or '（空）'}")
    if not final_root or not os.path.isdir(final_root):
        blockers.append(f"成果输出根目录不存在: {rel_path(final_root) or '（空）'}")
    if not mask_root or not os.path.isdir(mask_root):
        blockers.append(f"预测掩膜根目录不存在: {rel_path(mask_root) or '（空）'}")

    try:
        prob = float(prob_threshold)
    except (TypeError, ValueError):
        prob = float("nan")
    try:
        cnt = int(count_threshold)
    except (TypeError, ValueError):
        cnt = -1

    if not (PROB_MIN <= prob <= PROB_MAX):
        blockers.append(f"概率阈值 {prob!r} 超出范围 [{PROB_MIN}, {PROB_MAX}]。")
    if not (CNT_MIN <= cnt <= CNT_MAX):
        blockers.append(f"最少出现次数 {cnt!r} 超出范围 [{CNT_MIN}, {CNT_MAX}]。")

    # B10：input_asset_id 指向已登记 GEE 数据集时，读取 scene_count 用于 A1 阻断
    asset_scene_count: Optional[int] = None
    if input_asset_id and cnt >= 2:
        try:
            import dataset_assets
            entry = dataset_assets.get_dataset(str(input_asset_id))
            if entry and isinstance(entry.get("scene_count"), (int, float)):
                sc = int(entry["scene_count"])
                asset_scene_count = sc
                if sc < cnt:
                    blockers.append(
                        f"输入数据集（{input_asset_id}）仅有 {sc} 景有效影像，"
                        f"但频次阈值为 {cnt}。双约束后处理无法得到有效结果，"
                        f"请增加同一区域影像或降低频次阈值。"
                    )
        except Exception:  # noqa: BLE001
            pass

    if device_policy not in ("auto", "cuda_required"):
        warnings.append(f"未知设备策略 {device_policy!r}，按 auto 处理。")
        device_policy = "auto"

    # 任务目录校验（input_path 仅来自 root_dir/task_id，禁止拼接）
    input_path = ""
    task_options: List[str] = []
    if root_dir and os.path.isdir(root_dir):
        try:
            task_options = sorted(
                d for d in os.listdir(root_dir)
                if os.path.isdir(os.path.join(root_dir, d))
            )
        except OSError:
            task_options = []
    if task_id and root_dir and os.path.isdir(root_dir):
        candidate = os.path.join(root_dir, task_id)
        if os.path.isdir(candidate):
            input_path = os.path.normpath(candidate)
        else:
            # 兼容：任务名在子目录列表内做精确匹配
            exact = next((d for d in task_options if d == task_id), None)
            if exact:
                input_path = os.path.normpath(os.path.join(root_dir, exact))
            else:
                blockers.append(
                    f"目标任务目录不存在于影像根目录: {os.path.join(root_dir, task_id)}"
                )

    weight_path, wp_warn = resolve_weight_path(weight_id, model_path)
    warnings.extend(wp_warn)
    if not weight_path:
        blockers.append(
            f"未找到可用模型权重（weight_id={weight_id or '—'}，model_path={rel_path(model_path) or '—'}）。"
        )

    # 海岸线裁剪矢量（后处理 Step 3 几何掩膜用）；不存在时仅告警，不阻断
    shp_path = str(shp_path or "").strip().strip('"').strip("'") or None
    if shp_path and not os.path.isfile(shp_path):
        warnings.append(f"海岸线裁剪矢量不存在，后处理将跳过几何裁剪: {rel_path(shp_path)}")
        shp_path = None

    output_dir = os.path.join(final_root, task_id) if task_id else ""
    mask_dir = os.path.join(mask_root, task_id) if task_id else ""

    steps: List[str] = [
        f"校验输入目录与影像（{task_id or '—'}，RGB 波段 [1,2,3]）",
        "加载 CDNet 权重并校验结构",
        "逐景执行深度学习推理（1024×1024 切片，512 重叠）",
        "双重约束时空频次后处理（Final TIF + Final SHP）",
        "验证成果文件并登记预测资产",
        "加载成果到地图并刷新动态能力",
    ]

    ready = len(blockers) == 0

    plan: Dict[str, Any] = {
        "schema": "local_tidal_flat_inference_plan_v1",
        "plan_id": new_plan_id(),
        "task_id": task_id,
        "tool": TOOL_NAME,
        "input_asset_id": input_asset_id or "ui_selected",
        "input_asset_scene_count": asset_scene_count,
        "input_path": input_path,
        "input_type": "local_raster",
        "bands": [1, 2, 3],
        "model_id": MODEL_ID,
        "weight_id": weight_id or "ui_selected",
        "weight_path": weight_path,
        "device_policy": device_policy,
        "device": "",  # validate 后写入实际设备
        "prob_threshold": round(prob, 2) if prob == prob else prob_threshold,
        "count_threshold": cnt,
        "postprocess": True,
        "output_dir": output_dir,
        "mask_dir": mask_dir,
        "shp_path": shp_path,
        "expected_outputs": ["prediction_tif", "final_tif", "final_shp"],
        "ready": ready,
        "blockers": blockers,
        "warnings": warnings,
        "steps": steps,
        "status": "waiting_confirmation" if ready else "blocked",
        "created_at": _now_str(),
        "available_tasks": task_options,
    }
    return plan


# =======================================================
#  2. 执行前验证（§2.3 / 用户规格 §四）
# =======================================================
def _list_raw_tifs(input_dir: str) -> List[str]:
    """输入目录下可参与推理的 *.tif（排除 _mask / Final 前缀，去重）。

    Windows 文件系统大小写不敏感：*.tif/*.TIF/*.tiff/*.TIFF 可能匹配同一文件，
    必须按规范化路径去重，否则同一景会被重复计数（影响 A1 单景阻断判定）。
    """
    if not input_dir or not os.path.isdir(input_dir):
        return []
    all_tifs = glob.glob(os.path.join(input_dir, "*.tif")) + \
        glob.glob(os.path.join(input_dir, "*.TIF")) + \
        glob.glob(os.path.join(input_dir, "*.tiff")) + \
        glob.glob(os.path.join(input_dir, "*.TIFF"))
    seen = set()
    out = []
    for f in all_tifs:
        name = os.path.basename(f)
        if "_mask" in name or "Final" in name:
            continue
        key = os.path.normcase(os.path.normpath(f))
        if key in seen:
            continue
        seen.add(key)
        out.append(f)
    return sorted(out)


def validate_inference_plan(
    plan: Dict[str, Any],
    *,
    check_weight_load: bool = True,
) -> Tuple[bool, List[str], str]:
    """
    执行前验证（§四）。返回 (ok, blockers, device)。

    输入检查：路径存在 / 目录非空 / 含受支持 tif / rasterio 可开首个 tif /
              CRS 可读 / 尺寸合法 / 波段数 ≥ 3 / 输出目录可写。
    A1 单景阻断：有效景数 < count_threshold 时提前阻断（双约束后处理无法得到
              有效结果），不启动 GPU / 不加载模型 / 不创建正式输出。
    权重检查：路径存在 / 非 URL / weights_only=True 安全加载 / strict 匹配 CDNet。
    设备检查：torch.cuda.is_available；auto → cuda/cpu 回退并记录真实设备；
              cuda_required 且无 CUDA → blocker；绝不虚报 GPU。
    """
    blockers: List[str] = []
    device = ""

    if not plan:
        return False, ["计划为空。"], device

    # ---- 输入 ----
    input_dir = plan.get("input_path") or ""
    if not input_dir or not os.path.isdir(input_dir):
        blockers.append(f"输入目录不存在: {rel_path(input_dir) or '（空）'}")
    else:
        tifs = _list_raw_tifs(input_dir)
        if not tifs:
            blockers.append(f"输入目录没有可处理的 *.tif 影像: {rel_path(input_dir)}")
        else:
            # A1：单景输入提前阻断（双约束后处理 E>=min_absolute_count 无法满足）
            _cnt = int(plan.get("count_threshold") or 0)
            if _cnt >= 2 and len(tifs) < _cnt:
                blockers.append(
                    f"当前任务只有 {len(tifs)} 景有效影像，但频次阈值为 {_cnt}。"
                    f"双约束后处理无法得到有效结果，请增加同一区域影像或降低频次阈值。"
                )
                # A1 铁律：提前返回，不加载模型、不探测设备、不创建任何输出
                plan["device"] = ""
                return False, blockers, ""
            first = tifs[0]
            try:
                import rasterio
                with rasterio.open(first) as src:
                    if src.crs is None:
                        blockers.append(f"影像缺少 CRS 定义: {os.path.basename(first)}")
                    if src.width <= 0 or src.height <= 0:
                        blockers.append(f"影像尺寸非法: {os.path.basename(first)}")
                    if src.count < 3:
                        blockers.append(
                            f"影像波段不足 3（实际 {src.count}），模型需 RGB 波段 [1,2,3]: "
                            f"{os.path.basename(first)}"
                        )
            except Exception as e:  # noqa: BLE001
                blockers.append(f"影像不可读取: {os.path.basename(first)}（{safe_error_summary(e)}）")

    # ---- 输出目录可写 ----
    for dname, d in (("mask_dir", plan.get("mask_dir")), ("output_dir", plan.get("output_dir"))):
        d = str(d or "")
        if not d:
            blockers.append(f"{dname} 未配置。")
            continue
        try:
            os.makedirs(d, exist_ok=True)
            probe = os.path.join(d, ".cstf_write_probe")
            with open(probe, "w", encoding="utf-8") as f:
                f.write("ok")
            os.remove(probe)
        except OSError as e:
                blockers.append(f"{dname} 不可写: {rel_path(d)}（{safe_error_summary(e)}）")

    # ---- 权重 ----
    weight_path = plan.get("weight_path") or ""
    if not weight_path or not os.path.isfile(str(weight_path)):
        blockers.append(f"权重文件不存在: {rel_path(weight_path)}")
    else:
        low = str(weight_path).lower()
        if low.startswith(("http://", "https://", "ftp://", "file://")):
            blockers.append("权重路径为网络地址，已拒绝。")
        if check_weight_load:
            try:
                import torch
                torch_load = torch.load(str(weight_path), map_location="cpu", weights_only=True)
                if not isinstance(torch_load, dict) or not any(
                    isinstance(v, torch.Tensor) for v in torch_load.values()
                ):
                    blockers.append("权重文件不是合法 state_dict（无张量项）。")
                else:
                    # strict 匹配 CDNet 结构（试配，不驻留）
                    try:
                        import sys
                        import os as _os
                        _dir = _os.path.dirname(_os.path.abspath(__file__))
                        if _dir not in sys.path:
                            sys.path.insert(0, _dir)
                        from YYnet import CDNet
                        model = CDNet(
                            backbone="resnet50", output_stride=16, img_size=1024,
                            n_class=1, img_chan=3, chan_num=64, fuzzy_num=16,
                        )
                        state = {k.replace("module.", ""): v for k, v in torch_load.items()}
                        try:
                            model.load_state_dict(state, strict=True)
                        except Exception as e:  # noqa: BLE001
                            blockers.append(f"权重与 CDNet 结构不匹配: {safe_error_summary(e)}")
                        del model
                    except Exception as e:  # noqa: BLE001
                        blockers.append(f"权重结构校验失败: {safe_error_summary(e)}")
            except Exception as e:  # noqa: BLE001
                blockers.append(f"权重加载失败（weights_only=True）: {safe_error_summary(e)}")

    # ---- 设备 ----
    try:
        import torch
        cuda_ok = bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001
        cuda_ok = False
    policy = plan.get("device_policy") or "auto"
    if policy == "cuda_required" and not cuda_ok:
        blockers.append("设备策略为 cuda_required，但当前环境无可用 CUDA。")
        device = ""
    else:
        device = "cuda" if cuda_ok else "cpu"
        if not cuda_ok:
            pass  # auto 回退 CPU，记录真实设备（不虚报）

    plan["device"] = device
    return (len(blockers) == 0, blockers, device)


# =======================================================
#  3. 确认机制（§2.4 / 用户规格 §五）
# =======================================================
def confirm_inference_plan(state: Dict[str, Any], plan_id: str) -> Tuple[bool, Optional[str]]:
    """
    确认门闩（与 UI 按钮共用同一逻辑）：
    - 计划必须存在于 _inference_pending_plan 且 plan_id 匹配；
    - 同一 plan_id 只确认一次（重复确认 → 错误，不重复执行）；
    - 确认后写入 _inference_plan_confirmed（set）。
    返回 (ok, error)。
    """
    pending = state.get(STATE_INFERENCE_PENDING_PLAN) or {}
    if not pending:
        return False, "没有待确认的推理计划（请先生成计划）。"
    if str(pending.get("plan_id") or "") != str(plan_id):
        return False, f"计划已变化（plan_id 不匹配），请重新生成计划。"
    confirmed = state.get(STATE_INFERENCE_PLAN_CONFIRMED) or set()
    if plan_id in confirmed:
        return False, "该计划已确认，请勿重复确认（不会重复执行）。"
    confirmed.add(plan_id)
    state[STATE_INFERENCE_PLAN_CONFIRMED] = confirmed
    pending["status"] = "confirmed"
    return True, None


def is_plan_confirmed(state: Dict[str, Any], plan_id: Optional[str]) -> bool:
    if not plan_id:
        return False
    confirmed = state.get(STATE_INFERENCE_PLAN_CONFIRMED) or set()
    return plan_id in confirmed


def cancel_inference_plan(state: Dict[str, Any]) -> None:
    """取消：清除待确认计划（不可恢复）。"""
    state.pop(STATE_INFERENCE_PENDING_PLAN, None)


# =======================================================
#  4. 真实执行（§2.5 / §2.6 / 用户规格 §六）
# =======================================================
def _default_pre_engine() -> Any:
    import pre_engine
    return pre_engine


def _default_post_engine() -> Any:
    import post_engine
    return post_engine


def execute_local_inference(
    plan: Dict[str, Any],
    *,
    stop_event: Optional[Any] = None,
    push_log: Callable[[str], None] = print,
    push_progress: Optional[Callable[[int], None]] = None,
    pre_engine_mod: Optional[Any] = None,
    post_engine_mod: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    真实调用现有推理/后处理代码（复刻 run_pipeline_sync 流程，不重写算法）。

    返回 ToolResult（只填真实数据）：
      success / task_id / plan_id / tool / status / inputs / parameters /
      outputs{prediction_tif, final_tif, final_shp} / metrics{elapsed_seconds,
      processed_tiles, tif_count} / warnings / error
    """
    import time

    started = time.time()
    task_id = str(plan.get("task_id") or "")
    plan_id = str(plan.get("plan_id") or "")
    input_dir = str(plan.get("input_path") or "")
    mask_dir = str(plan.get("mask_dir") or "")
    output_dir = str(plan.get("output_dir") or "")
    prob = float(plan.get("prob_threshold") or 0.05)
    cnt = int(plan.get("count_threshold") or 2)
    weight_path = str(plan.get("weight_path") or "")
    device = str(plan.get("device") or "")
    warnings: List[str] = []

    def check_stop() -> bool:
        return bool(stop_event and stop_event.is_set())

    def pct(v: int) -> None:
        if push_progress:
            push_progress(int(min(100, max(0, v))))

    def cancelled_result(processed_tiles: int = 0) -> Dict[str, Any]:
        """Return a stable non-committable result for a user stop request."""
        return {
            "success": False, "task_id": task_id, "plan_id": plan_id,
            "tool": TOOL_NAME, "status": "cancelled",
            "inputs": {"input_path": rel_path(input_dir), "model_id": MODEL_ID,
                       "weight_id": plan.get("weight_id"), "device": device},
            "parameters": {"prob_threshold": prob, "count_threshold": cnt},
            "outputs": {},
            "metrics": {
                "elapsed_seconds": round(time.time() - started, 2),
                "processed_tiles": processed_tiles,
                "tif_count": 0,
            },
            "warnings": warnings,
            "error": "推理被用户中断。",
        }

    pre = pre_engine_mod if pre_engine_mod is not None else _default_pre_engine()
    post = post_engine_mod if post_engine_mod is not None else _default_post_engine()

    try:
        tifs = _list_raw_tifs(input_dir)
        if not tifs:
            return {
                "success": False, "task_id": task_id, "plan_id": plan_id,
                "tool": TOOL_NAME, "status": "failed",
                "inputs": {"input_path": rel_path(input_dir), "model_id": MODEL_ID,
                           "weight_id": plan.get("weight_id"), "device": device},
                "parameters": {"prob_threshold": prob, "count_threshold": cnt},
                "outputs": {}, "metrics": {"elapsed_seconds": 0.0, "processed_tiles": 0,
                                           "tif_count": 0},
                "warnings": [], "error": "输入目录没有可处理的 TIF 影像。",
            }

        total = len(tifs)
        pct(5)
        push_log(f"正在载入深度学习模型: {os.path.basename(weight_path)}")
        try:
            model = pre.load_model(weight_path, device)
        except Exception as e:  # noqa: BLE001
            return {
                "success": False, "task_id": task_id, "plan_id": plan_id,
                "tool": TOOL_NAME, "status": "failed",
                "inputs": {"input_path": rel_path(input_dir), "model_id": MODEL_ID,
                           "weight_id": plan.get("weight_id"), "device": device},
                "parameters": {"prob_threshold": prob, "count_threshold": cnt},
                "outputs": {}, "metrics": {"elapsed_seconds": round(time.time() - started, 2),
                                           "processed_tiles": 0, "tif_count": total},
                "warnings": [], "error": f"模型加载失败: {safe_error_summary(e)}",
            }
        pct(10)

        os.makedirs(mask_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)
        push_log(">>> [Phase 1] 深度学习推理...")
        success_count = 0
        for idx, tif_path in enumerate(tifs):
            if check_stop():
                return {
                    "success": False, "task_id": task_id, "plan_id": plan_id,
                    "tool": TOOL_NAME, "status": "failed",
                    "inputs": {"input_path": rel_path(input_dir), "model_id": MODEL_ID,
                               "weight_id": plan.get("weight_id"), "device": device},
                    "parameters": {"prob_threshold": prob, "count_threshold": cnt},
                    "outputs": {}, "metrics": {"elapsed_seconds": round(time.time() - started, 2),
                                               "processed_tiles": success_count, "tif_count": total},
                    "warnings": warnings, "error": "推理被用户中断。",
                }
            fname = os.path.basename(tif_path)
            save_name = fname.rsplit(".", 1)[0] + "_mask.tif"
            save_path = os.path.join(mask_dir, save_name)
            push_log(f"  推理: {fname} ({idx + 1}/{total})")
            pct(10 + int((idx / total) * 60))
            if os.path.exists(save_path) and os.path.getsize(save_path) > 0:
                success_count += 1
                continue
            try:
                res = pre.process_geotiff(
                    model, tif_path, save_path, device,
                    current_idx=idx + 1, total_batch=total, stop_callback=check_stop,
                )
                if res is False:
                    if check_stop():
                        return {
                            "success": False, "task_id": task_id, "plan_id": plan_id,
                            "tool": TOOL_NAME, "status": "failed",
                            "inputs": {"input_path": rel_path(input_dir), "model_id": MODEL_ID,
                                       "weight_id": plan.get("weight_id"), "device": device},
                            "parameters": {"prob_threshold": prob, "count_threshold": cnt},
                            "outputs": {}, "metrics": {"elapsed_seconds": round(time.time() - started, 2),
                                                       "processed_tiles": success_count, "tif_count": total},
                            "warnings": warnings, "error": "推理被用户中断。",
                        }
                    return {
                        "success": False, "task_id": task_id, "plan_id": plan_id,
                        "tool": TOOL_NAME, "status": "failed",
                        "inputs": {"input_path": rel_path(input_dir), "model_id": MODEL_ID,
                                   "weight_id": plan.get("weight_id"), "device": device},
                        "parameters": {"prob_threshold": prob, "count_threshold": cnt},
                        "outputs": {}, "metrics": {"elapsed_seconds": round(time.time() - started, 2),
                                                   "processed_tiles": success_count, "tif_count": total},
                        "warnings": warnings,
                        "error": f"单景推理失败: {fname}（非中断，见控制台 traceback）",
                    }
                success_count += 1
            except Exception as e:  # noqa: BLE001
                return {
                    "success": False, "task_id": task_id, "plan_id": plan_id,
                    "tool": TOOL_NAME, "status": "failed",
                    "inputs": {"input_path": rel_path(input_dir), "model_id": MODEL_ID,
                               "weight_id": plan.get("weight_id"), "device": device},
                    "parameters": {"prob_threshold": prob, "count_threshold": cnt},
                    "outputs": {}, "metrics": {"elapsed_seconds": round(time.time() - started, 2),
                                               "processed_tiles": success_count, "tif_count": total},
                    "warnings": warnings, "error": f"单景推理异常: {fname}（{safe_error_summary(e)}）",
                }

        if check_stop():
            return {
                "success": False, "task_id": task_id, "plan_id": plan_id,
                "tool": TOOL_NAME, "status": "failed",
                "inputs": {"input_path": rel_path(input_dir), "model_id": MODEL_ID,
                           "weight_id": plan.get("weight_id"), "device": device},
                "parameters": {"prob_threshold": prob, "count_threshold": cnt},
                "outputs": {}, "metrics": {"elapsed_seconds": round(time.time() - started, 2),
                                           "processed_tiles": success_count, "tif_count": total},
                "warnings": warnings, "error": "推理被用户中断。",
            }

        # 后处理：输出 stem 与 run_pipeline_sync 命名一致
        #   final_tif = output_dir/{task}_Final_p{prob:.2f}_c{cnt}.tif
        #   final_shp = output_dir/{task}_Final_p{prob:.2f}_c{cnt}.shp（同 stem）
        stem = f"{task_id}_Final_p{prob:.2f}_c{cnt}"
        final_tif = os.path.join(output_dir, stem + ".tif")
        final_shp = os.path.join(output_dir, stem + ".shp")

        pct(72)
        push_log(">>> [Phase 2] 双重约束时空频次合成...")
        pct(75)

        def bridge_logger(msg: str) -> None:
            push_log(msg)

        try:
            ok = post.generate_double_constraint_complete(
                source_folder=input_dir, mask_folder=mask_dir,
                output_path=final_tif, shp_path=plan.get("shp_path"),
                prob_threshold=prob, min_absolute_count=cnt,
                logger=bridge_logger, stop_callback=check_stop,
                keep_final_tif=True,
            )
        except Exception as e:  # noqa: BLE001
            return {
                "success": False, "task_id": task_id, "plan_id": plan_id,
                "tool": TOOL_NAME, "status": "failed",
                "inputs": {"input_path": rel_path(input_dir), "model_id": MODEL_ID,
                           "weight_id": plan.get("weight_id"), "device": device},
                "parameters": {"prob_threshold": prob, "count_threshold": cnt},
                "outputs": {}, "metrics": {"elapsed_seconds": round(time.time() - started, 2),
                                           "processed_tiles": success_count, "tif_count": total},
            "warnings": warnings, "error": f"后处理异常: {safe_error_summary(e)}",
            }
        if not ok:
            err = "后处理被用户中断。" if check_stop() else "后处理失败（未生成成果）。"
            return {
                "success": False, "task_id": task_id, "plan_id": plan_id,
                "tool": TOOL_NAME, "status": "failed",
                "inputs": {"input_path": rel_path(input_dir), "model_id": MODEL_ID,
                           "weight_id": plan.get("weight_id"), "device": device},
                "parameters": {"prob_threshold": prob, "count_threshold": cnt},
                "outputs": {}, "metrics": {"elapsed_seconds": round(time.time() - started, 2),
                                           "processed_tiles": success_count, "tif_count": total},
                "warnings": warnings, "error": err,
            }

        # 后处理可能在最后一个分块/矢量写出后才观察到停止请求；
        # 在成果校验和资产登记前再检查一次，禁止迟到成功穿透取消门闩。
        if check_stop():
            return cancelled_result(processed_tiles=success_count)

        # 后处理适配器返回 True 只是表示调用路径未抛错，不能替代对真实
        # 磁盘产物的存在性检查。否则适配器“空成功”会被上层当成可登记成果。
        artifact_paths = {
            "Final TIF": final_tif,
            "Final SHP": final_shp,
        }
        missing_artifacts = [
            label for label, path in artifact_paths.items()
            if not path or not os.path.isfile(path) or os.path.getsize(path) <= 0
        ]
        if missing_artifacts:
            return {
                "success": False, "task_id": task_id, "plan_id": plan_id,
                "tool": TOOL_NAME, "status": "failed",
                "inputs": {"input_path": rel_path(input_dir), "model_id": MODEL_ID,
                           "weight_id": plan.get("weight_id"), "device": device},
                "parameters": {"prob_threshold": prob, "count_threshold": cnt},
                "outputs": {},
                "metrics": {"elapsed_seconds": round(time.time() - started, 2),
                            "processed_tiles": success_count, "tif_count": total},
                "warnings": warnings,
                "error": "后处理返回成功但成果文件缺失或为空: " + "、".join(missing_artifacts),
            }

        pct(92)
        # 收集 prediction_tif（首个生成的 mask 文件）
        mask_files = sorted(glob.glob(os.path.join(mask_dir, "*_mask.tif")))
        prediction_tif = mask_files[0] if mask_files else ""
        pct(100)
        elapsed = round(time.time() - started, 2)
        push_log(f"✅ 推理与后处理完成，耗时 {elapsed}s。")

        return {
            "success": True, "task_id": task_id, "plan_id": plan_id,
            "tool": TOOL_NAME, "status": "completed",
            "inputs": {
                "input_asset_id": plan.get("input_asset_id"),
                "input_path": rel_path(input_dir),
                "model_id": MODEL_ID,
                "weight_id": plan.get("weight_id"),
                "device": device,
            },
            "parameters": {"prob_threshold": prob, "count_threshold": cnt},
            "outputs": {
                "prediction_tif": rel_path(prediction_tif) if prediction_tif else "",
                "final_tif": rel_path(final_tif),
                "final_shp": rel_path(final_shp),
            },
            "metrics": {
                "elapsed_seconds": elapsed,
                "processed_tiles": success_count,
                "tif_count": total,
            },
            "warnings": warnings,
            "error": None,
        }
    except Exception as e:  # noqa: BLE001
        return {
            "success": False, "task_id": task_id, "plan_id": plan_id,
            "tool": TOOL_NAME, "status": "failed",
            "inputs": {"input_path": rel_path(input_dir), "model_id": MODEL_ID,
                       "weight_id": plan.get("weight_id"), "device": device},
            "parameters": {"prob_threshold": prob, "count_threshold": cnt},
            "outputs": {}, "metrics": {"elapsed_seconds": round(time.time() - started, 2),
                                       "processed_tiles": 0, "tif_count": 0},
            "warnings": warnings, "error": f"推理执行异常: {safe_error_summary(e)}",
        }


# =======================================================
#  5. 结果验证（§2.7 / 用户规格 §七）
# =======================================================
def _tif_bbox_intersection_ratio(a: Tuple[float, float, float, float],
                                 b: Tuple[float, float, float, float]) -> float:
    """两个 bbox (minx,miny,maxx,maxy) 的交并比（IoU）。"""
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union


def verify_inference_outputs(
    plan: Dict[str, Any],
    result: Dict[str, Any],
    started_at: Optional[float] = None,
) -> Dict[str, Any]:
    """
    验证 Final TIF 与 Final SHP（§七）。返回 {"ok", "checks", "final_tif", "final_shp"}。

    Final TIF：存在 / 非空 / rasterio 可开 / CRS / transform / w,h>0 / 非全 NoData /
               范围与输入合理关系 / mtime ≥ 任务启动时间 / 属于当前 task_id。
    Final SHP：.shp/.shx/.dbf 存在 / geopandas 可读 / CRS / 几何合法 / 非空 /
               bbox 与 Final TIF 基本一致（SHP 为必要输出，缺失 → 验证失败）。
    """
    import time
    checks: List[Dict[str, Any]] = []
    ok = True

    def _check(name: str, passed: bool, detail: str = "") -> None:
        nonlocal ok
        if not passed:
            ok = False
        checks.append({"name": name, "passed": passed, "detail": detail})

    import os as _os
    final_tif = ""
    final_shp = ""
    outputs = result.get("outputs") or {}
    if isinstance(outputs, dict):
        final_tif = _os.path.normpath(str(outputs.get("final_tif") or ""))
        final_shp = _os.path.normpath(str(outputs.get("final_shp") or ""))

    # ---- Final TIF ----
    _check("final_tif_path", bool(final_tif) and _os.path.isfile(final_tif),
           final_tif or "（未指定）")
    if final_tif and _os.path.isfile(final_tif):
        _check("final_tif_nonempty", _os.path.getsize(final_tif) > 0, f"{_os.path.getsize(final_tif)} B")
        try:
            import rasterio
            with rasterio.open(final_tif) as src:
                _check("final_tif_open", True, f"{src.width}×{src.height}")
                _check("final_tif_crs", src.crs is not None, str(src.crs))
                _check("final_tif_transform", bool(src.transform) and src.transform != src.transform.identity(),
                       "valid")
                _check("final_tif_size", src.width > 0 and src.height > 0,
                       f"{src.width}×{src.height}")
                if src.width > 0 and src.height > 0:
                    # 非全 NoData：降采样读一次
                    try:
                        import numpy as np
                        sample = src.read(1, out_shape=(1, max(1, src.height // 16),
                                                        max(1, src.width // 16)))
                        nodata = src.nodata
                        valid = sample[~np.isnan(sample.astype(float))]
                        if nodata is not None:
                            valid = valid[valid != nodata]
                        _check("final_tif_has_data", bool(np.any(valid > 0)), f"valid_pixels={int(np.count_nonzero(valid))}")
                    except Exception as e:  # noqa: BLE001
                        _check("final_tif_has_data", False, f"读取采样失败: {safe_error_summary(e)}")
                tb = src.bounds
                # 与输入范围合理关系：与任一输入 tif 有交集
                input_dir = plan.get("input_path") or ""
                ratio = 0.0
                if input_dir and _os.path.isdir(input_dir):
                    for tif in _list_raw_tifs(input_dir)[:5]:
                        try:
                            with rasterio.open(tif) as isrc:
                                r = _tif_bbox_intersection_ratio(
                                    (tb.left, tb.bottom, tb.right, tb.top),
                                    (isrc.bounds.left, isrc.bounds.bottom,
                                     isrc.bounds.right, isrc.bounds.top),
                                )
                                ratio = max(ratio, r)
                        except Exception:  # noqa: BLE001
                            continue
                _check("final_tif_overlaps_input", ratio > 0.0, f"max_iou={ratio:.3f}")
        except Exception as e:  # noqa: BLE001
            _check("final_tif_open", False, f"打开失败: {safe_error_summary(e)}")

        try:
            mt = _os.path.getmtime(final_tif)
        except OSError:
            mt = 0.0
        started_at = started_at or (time.time() - 3600)
        _check("final_tif_mtime", mt >= started_at,
               f"mtime={mt:.0f} >= start={started_at:.0f}")

    task_id = str(plan.get("task_id") or "")
    _check("final_tif_belongs_to_task",
           (not task_id) or (task_id in _os.path.basename(final_tif)),
           f"task={task_id}")

    # ---- Final SHP ----
    _check("final_shp_path", bool(final_shp) and _os.path.isfile(final_shp),
           final_shp or "（未指定）")
    if final_shp and _os.path.isfile(final_shp):
        stem = _os.path.splitext(final_shp)[0]
        for ext in (".shx", ".dbf"):
            _check(f"final_shp_sidecar{ext}", _os.path.isfile(stem + ext),
                   stem + ext)
        try:
            import geopandas as gpd
            gdf = gpd.read_file(final_shp)
            _check("final_shp_readable", True, f"{len(gdf)} features")
            _check("final_shp_crs", gdf.crs is not None, str(gdf.crs))
            _check("final_shp_nonempty", len(gdf) > 0, f"{len(gdf)} features")
            if len(gdf) > 0:
                try:
                    geom_valid = bool(gdf.geometry.notna().all()) and bool(gdf.geometry.is_valid.all())
                except Exception:  # noqa: BLE001
                    geom_valid = True
                _check("final_shp_geometry_valid", geom_valid, "")
                try:
                    sb = gdf.total_bounds  # [minx, miny, maxx, maxy]
                    tb = None
                    if final_tif and _os.path.isfile(final_tif):
                        import rasterio
                        with rasterio.open(final_tif) as src:
                            tb = (src.bounds.left, src.bounds.bottom,
                                  src.bounds.right, src.bounds.top)
                    if tb is not None:
                        r = _tif_bbox_intersection_ratio(
                            (sb[0], sb[1], sb[2], sb[3]), tb
                        )
                        _check("final_shp_bbox_matches_tif", r > 0.5,
                               f"iou={r:.3f}")
                except Exception:  # noqa: BLE001
                    pass
        except Exception as e:  # noqa: BLE001
                    _check("final_shp_readable", False, f"读取失败: {safe_error_summary(e)}")

    return {
        "ok": ok,
        "checks": checks,
        "final_tif": final_tif if final_tif and _os.path.isfile(final_tif) else None,
        "final_shp": final_shp if final_shp and _os.path.isfile(final_shp) else None,
    }


def classify_inference_recovery(
    plan: Dict[str, Any],
    result: Optional[Dict[str, Any]] = None,
    *,
    verify_fn: Optional[Callable[..., Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """根据已存在检查点决定恢复动作；只分类，不覆盖或删除文件。

    返回 action：``COMPLETE``（完整 Final 已验证）、``RESUME_POST_PROCESS``
    （mask 完整但 Final 缺失）、``ISOLATE_AND_RETRY``（有部分/无效 mask）或
    ``INTERRUPTED_WAIT_CONFIRMATION``（无可信检查点）。
    """
    plan = dict(plan or {})
    result = dict(result or {})
    output_dir = str(plan.get("output_dir") or "")
    mask_dir = str(plan.get("mask_dir") or "")
    task_id = str(plan.get("task_id") or "")
    prob = float(plan.get("prob_threshold") or 0.05)
    cnt = int(plan.get("count_threshold") or 2)
    stem = f"{task_id}_Final_p{prob:.2f}_c{cnt}"
    outputs = result.get("outputs") if isinstance(result.get("outputs"), dict) else {}
    final_tif = str(outputs.get("final_tif") or os.path.join(output_dir, stem + ".tif"))
    final_shp = str(outputs.get("final_shp") or os.path.join(output_dir, stem + ".shp"))
    final_exists = all(os.path.isfile(path) and os.path.getsize(path) > 0 for path in (final_tif, final_shp))
    if final_exists:
        checker = verify_fn or verify_inference_outputs
        checked = checker(
            plan,
            {"outputs": {"final_tif": final_tif, "final_shp": final_shp}},
            started_at=0,
        )
        if checked.get("ok") is True:
            return {"action": "COMPLETE", "reason": "verified_final_exists", "verification": checked}

    masks = []
    if os.path.isdir(mask_dir):
        masks = sorted(
            os.path.join(mask_dir, name)
            for name in os.listdir(mask_dir)
            if name.lower().endswith((".tif", ".tiff")) and "_mask" in name
        )
    valid_masks = [path for path in masks if os.path.getsize(path) > 0]
    expected_inputs = len(_list_raw_tifs(str(plan.get("input_path") or "")))
    if expected_inputs > 0 and len(valid_masks) == expected_inputs:
        return {"action": "RESUME_POST_PROCESS", "reason": "all_masks_present_final_missing", "mask_count": len(valid_masks)}
    if masks:
        return {"action": "ISOLATE_AND_RETRY", "reason": "partial_or_invalid_masks", "mask_count": len(valid_masks), "invalid_count": len(masks) - len(valid_masks)}
    return {"action": "INTERRUPTED_WAIT_CONFIRMATION", "reason": "no_trusted_checkpoint"}


def isolate_inference_checkpoints(
    plan: Dict[str, Any],
    *,
    confirmed: bool = False,
    quarantine_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """将部分/无效 mask 移入隔离目录；未确认时绝不修改文件。"""
    if not confirmed:
        return {"ok": False, "status": "WAITING_CONFIRMATION", "reason": "用户尚未确认隔离并重跑"}
    plan = dict(plan or {})
    mask_dir = os.path.abspath(str(plan.get("mask_dir") or ""))
    if not mask_dir or not os.path.isdir(mask_dir):
        return {"ok": False, "status": "NO_CHECKPOINTS", "reason": "mask 目录不存在"}
    candidates = [
        os.path.join(mask_dir, name)
        for name in os.listdir(mask_dir)
        if name.lower().endswith((".tif", ".tiff")) and "_mask" in name
    ]
    if not candidates:
        return {"ok": False, "status": "NO_CHECKPOINTS", "reason": "没有可隔离的 mask"}
    target_root = os.path.abspath(
        quarantine_dir
        or os.path.join(str(plan.get("output_dir") or mask_dir), ".recovery_quarantine")
    )
    os.makedirs(target_root, exist_ok=True)
    run_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(plan.get("plan_id") or "recovery"))[:80]
    target_dir = os.path.join(target_root, f"{run_id}-{int(time.time())}")
    os.makedirs(target_dir, exist_ok=False)
    moved: List[str] = []
    try:
        for source in candidates:
            destination = os.path.join(target_dir, os.path.basename(source))
            shutil.move(source, destination)
            moved.append(os.path.basename(source))
    except OSError as exc:
        # 发生部分移动时保留已移动文件，返回明确状态；不伪造“全部隔离成功”。
        return {
            "ok": False,
            "status": "PARTIAL_ISOLATION",
            "reason": safe_error_summary(exc),
            "moved": moved,
            "quarantine_dir": target_dir,
        }
    return {
        "ok": True,
        "status": "ISOLATED",
        "moved": moved,
        "quarantine_dir": target_dir,
    }


# =======================================================
#  6. 资产登记（§2.8 / 用户规格 §八）
# =======================================================
def _load_registry(registry_path: Optional[str] = None) -> Dict[str, Any]:
    from workflow_orchestrator import load_assets_registry

    return load_assets_registry(registry_path or ASSET_REGISTRY_PATH)


def _save_registry(registry: Dict[str, Any], registry_path: Optional[str] = None) -> None:
    from workflow_orchestrator import save_assets_registry

    save_assets_registry(registry, registry_path or ASSET_REGISTRY_PATH)


def register_inference_asset(
    plan: Dict[str, Any],
    result: Dict[str, Any],
    verification: Dict[str, Any],
    *,
    registry_path: Optional[str] = None,
) -> Optional[str]:
    """
    验证成功才登记（§八）。返回 asset_id 或 None。

    规则：验证失败不登记；不覆盖已有成果（新键含 plan_id 前 8 位）；同 plan_id
    不重复登记；主键 {task}_p{prob:.2f}_c{cnt} 同时保留以兼容现有 find_asset。
    """
    if not verification or verification.get("ok") is not True:
        return None
    if not result or result.get("success") is not True:
        return None

    # Re-check the commit boundary instead of trusting a caller-provided
    # verification dictionary that may be stale or forged after cleanup.
    if not _nonempty_file(verification.get("final_tif")) or not _nonempty_file(
        verification.get("final_shp")
    ):
        return None

    task_id = str(plan.get("task_id") or "")
    plan_id = str(plan.get("plan_id") or "")
    if not task_id or not plan_id:
        return None

    registry = _load_registry(registry_path)
    # 同 plan_id 不重复登记
    for _k, _v in registry.items():
        if isinstance(_v, dict) and _v.get("plan_id") == plan_id:
            return str(_v.get("asset_id") or _k)

    prob = float(plan.get("prob_threshold") or 0.05)
    cnt = int(plan.get("count_threshold") or 2)

    # 兼容主键（保持 find_asset 可用）+ 唯一扩展键
    compat_key = f"{task_id}_p{prob:.2f}_c{cnt}"
    asset_key = f"{compat_key}__{plan_id[:8]}"

    final_tif = verification.get("final_tif")
    final_shp = verification.get("final_shp")
    size_mb = 0.0
    for p in (final_tif, final_shp):
        if p and os.path.isfile(str(p)):
            try:
                size_mb += os.path.getsize(str(p)) / (1024 ** 2)
            except OSError:
                pass
    if final_shp and os.path.isfile(str(final_shp)):
        stem = os.path.splitext(str(final_shp))[0]
        for ext in (".shx", ".dbf", ".prj", ".cpg"):
            side = stem + ext
            if os.path.isfile(side):
                try:
                    size_mb += os.path.getsize(side) / (1024 ** 2)
                except OSError:
                    pass

    asset_id = str(uuid.uuid4().hex)
    entry: Dict[str, Any] = {
        "task": task_id,
        "prob_threshold": prob,
        "min_count": cnt,
        "file_path": os.path.normpath(str(final_shp)) if final_shp else "",
        "file_size_mb": round(size_mb, 2),
        "method": "dl",
        "asset_id": asset_id,
        "plan_id": plan_id,
        "asset_type": "tidal_flat_prediction",
        "source_asset_id": plan.get("input_asset_id"),
        "input_path": rel_path(str(plan.get("input_path") or "")),
        "model_id": plan.get("model_id") or MODEL_ID,
        "weight_id": plan.get("weight_id"),
        "code_commit": git_head(),
        "device": plan.get("device") or "",
        "parameters": {
            "prob_threshold": prob,
            "count_threshold": cnt,
        },
        "final_tif": rel_path(str(final_tif)) if final_tif else "",
        "final_shp": rel_path(str(final_shp)) if final_shp else "",
        "created_at": _now_str(),
        "elapsed_seconds": (result.get("metrics") or {}).get("elapsed_seconds", 0.0),
        "status": "verified",
    }
    registry[asset_key] = entry
    # 兼容键：若不存在则同步一份（避免覆盖已有成果）
    if compat_key not in registry:
        registry[compat_key] = dict(entry)
    _save_registry(registry, registry_path)
    return asset_id


def find_inference_asset(plan_id: Optional[str], task_id: Optional[str] = None,
                         registry_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
    registry = _load_registry(registry_path)
    if plan_id:
        for _k, _v in registry.items():
            if isinstance(_v, dict) and _v.get("plan_id") == plan_id:
                return _v
    if task_id:
        for _k, _v in registry.items():
            if isinstance(_v, dict) and _v.get("task") == task_id and \
                    _v.get("asset_type") == "tidal_flat_prediction":
                return _v
    return None


# =======================================================
#  7. 面向用户的计划 / 结果 / 上下文（§2.9-2.10）
# =======================================================
def format_inference_plan_for_user(plan: Dict[str, Any]) -> str:
    """确认前展示的执行计划（只含真实信息，路径用 basename/相对名）。"""
    if not plan:
        return "尚未生成提取计划。"
    lines = ["## 潮滩智能提取 · 执行计划", ""]
    if plan.get("ready"):
        lines.append("**状态：可执行**（请回复「确认」或点击确认按钮后开始）")
    else:
        lines.append("**状态：暂不可执行**")
        for b in plan.get("blockers") or []:
            lines.append(f"- 阻塞：{b}")
    for w in plan.get("warnings") or []:
        lines.append(f"- 注意：{w}")
    lines.append("")
    lines.append(f"- 任务：`{plan.get('task_id') or '—'}`")
    lines.append(f"- 输入目录：`{rel_path(str(plan.get('input_path') or '')) or '—'}`")
    lines.append(f"- 波段：RGB {plan.get('bands') or [1, 2, 3]}")
    lines.append(f"- 模型：`{plan.get('model_id') or '—'}` / 权重：`{basename_or_none(plan.get('weight_path'))}`")
    lines.append(f"- 设备策略：`{plan.get('device_policy') or 'auto'}`" +
                 (f"（实际设备：`{plan.get('device')}`）" if plan.get("device") else ""))
    lines.append(f"- 概率阈值：{plan.get('prob_threshold')} ｜ 最少出现次数：{plan.get('count_threshold')}")
    lines.append(f"- 后处理：{'启用' if plan.get('postprocess') else '关闭'}")
    lines.append("")
    lines.append("步骤：")
    for i, s in enumerate(plan.get("steps") or [], 1):
        lines.append(f"{i}. {s}")
    lines.append("")
    lines.append("确认后将真实运行提取与成果生成，并根据磁盘产物验证后回复。")
    return "\n".join(lines)


def summarize_inference_result_for_chat(
    result: Dict[str, Any],
    verification: Optional[Dict[str, Any]] = None,
) -> str:
    """基于真实工具结果生成 Copilot 回复（禁止编造指标）。"""
    if not result or result.get("success") is not True:
        err = sanitize_external_text((result or {}).get("error") or "提取失败")[:240]
        return (
            "## 潮滩智能提取 · 未完成\n\n"
            f"- 任务：`{(result or {}).get('task_id') or '—'}`\n"
            f"- 状态：**失败**\n"
            f"- 原因：{err}\n\n"
            "未登记预测资产，未生成正式成果。请根据上述原因处理后重试。"
        )
    inputs = result.get("inputs") or {}
    metrics = result.get("metrics") or {}
    outputs = result.get("outputs") or {}
    v_ok = bool(verification and verification.get("ok") is True)
    header = "## 潮滩提取已完成（已验证）" if v_ok else "## 潮滩提取已完成（校验未完全通过）"
    lines = [
        header,
        "",
        f"- 任务：`{result.get('task_id') or '—'}`",
        f"- 输入影像：`{basename_or_none(inputs.get('input_path'))}`",
        f"- 模型：`{inputs.get('model_id') or '—'}`",
        f"- 权重：`{basename_or_none(inputs.get('weight_id'))}`",
        f"- 运行设备：`{inputs.get('device') or '—'}`",
        f"- 运行时间：{metrics.get('elapsed_seconds', '—')} 秒"
        f"（处理 {metrics.get('processed_tiles', '—')}/{metrics.get('tif_count', '—')} 景）",
    ]
    if outputs.get("final_tif"):
        lines.append(f"- Final TIF：`{basename_or_none(outputs.get('final_tif'))}`")
    if outputs.get("final_shp"):
        lines.append(f"- Final SHP：`{basename_or_none(outputs.get('final_shp'))}`")
    if v_ok and verification.get("final_shp"):
        lines.append("- 成果已登记并加载至地图。")
    elif v_ok:
        lines.append("- 成果已登记。")
    else:
        lines.append("- 输出校验：**未完全通过**（请检查日志）。")
    lines.append("")
    lines.append("以上均来自本次提取与成果生成的真实输出，而非模型臆测。")
    return "\n".join(lines)


def build_inference_context_for_agent(
    *,
    root_dir: str = "",
    task_options: Optional[List[str]] = None,
    model_path: str = "",
    prob_threshold: Optional[float] = None,
    count_threshold: Optional[int] = None,
    device: str = "",
    pending_plan: Optional[Dict[str, Any]] = None,
) -> str:
    """注入 Copilot 的推理闭环上下文（白名单信息，不含密钥/无关绝对路径）。"""
    lines = ["【潮滩智能提取 · 可信执行闭环】"]
    lines.append(
        "可用工具：local_tidal_flat_inference(task_id, prob_th, cnt, run_now) 与 "
        "confirm_inference(plan_id)。"
    )
    lines.append(
        "规则：路径只允许使用侧栏已配置的合法值；禁止编造权重路径或输入目录；"
        "prob 范围 0.01~0.50，cnt 范围 1~10；执行前必须先生成计划并确认。"
    )
    if task_options:
        lines.append(f"- 可选任务：{', '.join(task_options[:12])}"
                     + (" …" if len(task_options) > 12 else ""))
    if model_path:
        lines.append(f"- 模型权重：`{basename_or_none(model_path)}`（侧栏配置）")
    if prob_threshold is not None:
        lines.append(f"- 当前概率阈值：{prob_threshold} ｜ 最少出现次数：{count_threshold}")
    if device:
        lines.append(f"- 当前设备：{device}")
    if pending_plan:
        pid = pending_plan.get("plan_id") or ""
        lines.append(
            f"- 待确认计划：plan_id={pid[:8]}…，任务={pending_plan.get('task_id')}，"
            f"就绪={'是' if pending_plan.get('ready') else '否'}"
        )
    return "\n".join(lines)


# 兼容旧导入名（若被外部引用）
build_inference_context = build_inference_context_for_agent
