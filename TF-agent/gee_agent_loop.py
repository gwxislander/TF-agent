# -*- coding: utf-8 -*-
"""
GEE 数据下载 Agent 可信执行闭环：计划 → 校验 → 确认 → 执行 → 验证 → 登记 → 汇总。

范围：只做「GEE Sentinel-2 下载」的可信执行闭环；复用现有 m4_engine（真实 GEE
筛选 / 导出 / 本地下载），不重新实现 GEE 算法逻辑。

可信铁律：
- AOI / 日期 / 波段 / 云量等参数全部来自用户输入或已登记数据资产，禁止 LLM 编造；
- 未确认绝不 task.start()；同一 plan_id 只确认一次；多个 pending 计划不串用；
- 全部状态（gee_task_id / 导出状态 / 本地文件 / scene_count / 耗时）必须来自真实
  执行结果，禁止 Fake success；
- 不输出凭证 / 不把巨大 GeoJSON 注入 LLM prompt（AOI 仅以 compact_summary 注入）；
- 下载完成 ≠ 推理开始：只更新能力状态，推理需用户另行确认；
- 本模块不依赖 Streamlit；重型依赖（ee / geemap / m4_engine / aoi_context /
  dataset_assets）在函数内懒加载，便于单元测试。
"""
from __future__ import annotations

import glob
import json
import ntpath
import os
import re
import shutil
import tempfile
import uuid
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

from agent_context_policy import safe_error_summary, safe_local_path_label, sanitize_external_text

# ---- 时间线阶段映射（与 task_timeline.PHASES 一致，B11） ----
GEE_TIMELINE_PHASES: Tuple[str, ...] = (
    "PLAN",
    "VALIDATE",
    "CONFIRM",
    "QUEUED",
    "GEE_EXPORT",
    "WAIT_REMOTE",
    "FETCH_OUTPUT",
    "VERIFY",
    "REGISTER",
    "MAP",
    "REPORT",
)

# ---- session_state 键约定（与推理/M5/E1 隔离） ----
STATE_GEE_PENDING_PLAN = "_gee_pending_plan"
STATE_GEE_PLAN_CONFIRMED = "_gee_plan_confirmed"  # set of plan_id

TOOL_NAME = "gee_download"

# B3 波段铁律：默认 ["B4","B3","B2"] 顺序 RGB 匹配 pre_engine；不默认 RGB 假设，
# 明确区分 inference bands / index bands；不偷偷改数据格式。
DEFAULT_BANDS = ["B4", "B3", "B2"]
DEFAULT_INDEX_BANDS = ["B8", "B11"]  # mNDWI 用 B3/B11，NIR 用 B8（仅辅助统计，不导出）
ALLOWED_BANDS = (
    "B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B11", "B12",
)

# 云量 / 水陆占比 / 像素 / 分辨率 合法范围（与 m4_engine / 侧栏一致）
CLOUD_MIN, CLOUD_MAX = 0, 100
LAND_MIN, LAND_MAX = 0.0, 100.0
PIXEL_MIN, PIXEL_MAX = 100, 500000
SCALE_OPTIONS = (10, 20, 30)
EXPORT_OPTIONS = ("drive", "local")

# 轻量规模预估上限（约 10^8 像素，避免用户误提交超大导出）
_MAX_ESTIMATED_PIXELS = 200_000_000

# 长任务账本（B7）：最小持久化 {task_id, plan_id, gee_task_id, status, created_at, last_checked_at}
GEE_TASK_LEDGER_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "data", "gee_task_ledger.json"
)

# 导出状态机（B6/B8）
GEE_TASK_STATUSES = ("READY", "RUNNING", "COMPLETED", "FAILED", "CANCELLED")
# 本地资产就绪标记：区分 GEE_EXPORT_COMPLETED（云端导出完成）与 LOCAL_ASSET_READY（本地文件就绪）
LOCAL_ASSET_READY = "LOCAL_ASSET_READY"


# =======================================================
#  工具函数
# =======================================================
def _now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def new_plan_id() -> str:
    return uuid.uuid4().hex


def rel_path(p: str) -> str:
    """Return a usable relative artifact path without echoing foreign paths."""
    raw = str(p or "")
    if not raw:
        return ""
    if ntpath.splitdrive(raw)[0] or raw.startswith(("\\\\", "//")):
        return safe_local_path_label(raw)
    try:
        return os.path.relpath(raw)
    except (OSError, ValueError):
        return safe_local_path_label(raw)


def _proxy_format_ok(url: Any) -> Tuple[bool, str]:
    """校验代理格式：空串（直连/VPN）或 http(s)://host:port 合法。"""
    url = str(url or "").strip()
    if not url:
        return True, ""
    if not re.match(r"^https?://", url):
        return False, "代理必须以 http:// 或 https:// 开头"
    m = re.match(r"^https?://([^:/]+)(?::(\d{1,5}))?/?$", url)
    if not m:
        return False, f"代理格式无效: {sanitize_external_text(url)!r}（应为 http://host:port）"
    port = m.group(2)
    if port and not (0 < int(port) < 65536):
        return False, f"代理端口非法: {port}"
    return True, ""


def _resolve_ee_project_any(gee_project_id: Optional[str]) -> Optional[str]:
    """复用 m4_engine._resolve_ee_project 多源解析（override → env → 文件 → credentials）。"""
    try:
        import m4_engine
        return m4_engine._resolve_ee_project(gee_project_id)
    except Exception:  # noqa: BLE001
        return None


def _credentials_file_ok() -> bool:
    p = os.path.join(os.path.expanduser("~"), ".config", "earthengine", "credentials")
    return os.path.isfile(p)


def _estimate_pixels(area_km2: float, scale: int) -> int:
    """粗略像素预估（平方 km → 像素，仅用于规模提示）。"""
    if area_km2 <= 0 or scale <= 0:
        return 0
    return int(area_km2 * 1_000_000 / (scale * scale))


def _list_local_tifs(out_dir: str) -> List[str]:
    if not out_dir or not os.path.isdir(out_dir):
        return []
    out = []
    for pat in ("*.tif", "*.TIF", "*.tiff", "*.TIFF"):
        for f in glob.glob(os.path.join(out_dir, pat)):
            name = os.path.basename(f)
            if "_mask" in name or "Final" in name or "NUMERATOR" in name or "DENOMINATOR" in name:
                continue
            out.append(f)
    # 去重（Windows 大小写不敏感）
    seen, uniq = set(), []
    for f in sorted(out):
        k = os.path.normcase(os.path.normpath(f))
        if k not in seen:
            seen.add(k)
            uniq.append(f)
    return uniq


# =======================================================
#  长任务账本（B7）：最小持久化 + 恢复
# =======================================================
def _load_task_ledger() -> Dict[str, Any]:
    if not os.path.isfile(GEE_TASK_LEDGER_PATH):
        return {}
    try:
        with open(GEE_TASK_LEDGER_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, UnicodeError) as exc:
        _preserve_corrupt_task_ledger()
        raise ValueError("GEE 任务账本 JSON 无效；原文件已保留，已停止写入。") from exc
    except OSError:
        raise
    if not isinstance(data, dict) or any(not isinstance(row, dict) for row in data.values()):
        _preserve_corrupt_task_ledger()
        raise ValueError("GEE 任务账本记录结构无效；原文件已保留，已停止写入。")
    return data


def _preserve_corrupt_task_ledger() -> Optional[str]:
    path = GEE_TASK_LEDGER_PATH
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


def _save_task_ledger(ledger: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(GEE_TASK_LEDGER_PATH) or ".", exist_ok=True)
    fd, tmp = tempfile.mkstemp(
        prefix=".gee_task_ledger_", suffix=".tmp",
        dir=os.path.dirname(os.path.abspath(GEE_TASK_LEDGER_PATH)) or ".",
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(ledger, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, GEE_TASK_LEDGER_PATH)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _ledger_upsert(task_id: str, **fields: Any) -> None:
    ledger = _load_task_ledger()
    row = dict(ledger.get(str(task_id)) or {})
    clean_fields = dict(fields)
    for key in ("error_message", "description"):
        if key in clean_fields and clean_fields[key] is not None:
            clean_fields[key] = sanitize_external_text(clean_fields[key])[:500]
    row.update(clean_fields)
    row["task_id"] = str(task_id)
    row["last_checked_at"] = _now_str()
    ledger[str(task_id)] = row
    _save_task_ledger(ledger)


def _ledger_get(task_id: str) -> Optional[Dict[str, Any]]:
    return _load_task_ledger().get(str(task_id))


def _poll_gee_task_status(gee_task_id: str) -> Dict[str, Any]:
    """
    轮询真实 GEE 任务状态（B6/B7）：ee.data.getTaskStatus([id])。
    返回 {"state", "error_message", "description", "id"}。
    """
    import ee
    try:
        resp = ee.data.getTaskStatus([gee_task_id])
    except Exception as e:  # noqa: BLE001
        return {"state": "UNKNOWN", "error_message": f"查询任务状态失败: {safe_error_summary(e)}"}
    if not resp or not isinstance(resp, list) or not resp:
        return {"state": "UNKNOWN", "error_message": "GEE 未返回任务状态"}
    item = resp[0] or {}
    return {
        "id": str(item.get("id") or gee_task_id),
        "state": str(item.get("state") or "UNKNOWN"),
        "error_message": str(item.get("error_message") or ""),
        "description": str(item.get("description") or ""),
    }


def reconcile_gee_task(
    task_id: str,
    *,
    plan_id: Optional[str] = None,
    poll_fn: Optional[Callable[[str], Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """恢复账本中的 GEE task_id，返回真实状态，不把未知状态当成功。

    该函数只轮询并更新账本；不会自动下载、启动推理或登记资产。测试可注入
    ``poll_fn``，生产调用默认使用 ``ee.data.getTaskStatus``。
    """
    task_key = str(task_id or "").strip()
    if not task_key:
        return {"ok": False, "status": "BLOCKED", "error": "缺少 task_id。", "tasks": []}
    poll = poll_fn or _poll_gee_task_status
    rows = []
    for row in _load_task_ledger().values():
        if str(row.get("task_id") or "") != task_key:
            continue
        if plan_id and str(row.get("plan_id") or "") != str(plan_id):
            continue
        gid = str(row.get("gee_task_id") or "").strip()
        if not gid:
            continue
        state = poll(gid) or {"state": "UNKNOWN", "error_message": "未返回状态"}
        normalized = str(state.get("state") or "UNKNOWN").upper()
        _ledger_upsert(task_key, plan_id=row.get("plan_id"), gee_task_id=gid,
                       status=normalized, error_message=state.get("error_message") or "")
        rows.append({"gee_task_id": gid, "state": normalized,
                     "error_message": str(state.get("error_message") or "")[:300]})
    if not rows:
        return {"ok": False, "status": "INTERRUPTED", "error": "账本中没有可恢复的 GEE task_id。", "tasks": []}
    states = {row["state"] for row in rows}
    if "UNKNOWN" in states:
        overall = "UNKNOWN"
    elif states.issubset({"COMPLETED"}):
        overall = "COMPLETED"
    elif states & {"FAILED", "CANCELLED"}:
        overall = "FAILED"
    else:
        overall = "RUNNING"
    return {"ok": overall == "COMPLETED", "status": overall, "tasks": rows}


# =======================================================
#  1. 计划构建（B3）
# =======================================================
def build_legacy_m4_plan(config: Optional[Dict[str, Any]] = None,
                          *, task_id: str = "") -> Dict[str, Any]:
    """Normalize the historical M4 config into the shared GEE plan schema.

    The old sidebar/Agent payload used ``m4`` fields and a ROI shapefile path.
    This adapter keeps that payload readable while ensuring execution enters
    the same plan → confirm → execute → verify → register path as GEE.
    """
    cfg = dict(config or {})
    task = str(task_id or cfg.get("task_id") or cfg.get("roi_name") or "").strip()
    aoi = cfg.get("aoi")
    if hasattr(aoi, "to_dict"):
        try:
            aoi = aoi.to_dict()
        except Exception:  # noqa: BLE001
            aoi = None
    if not isinstance(aoi, dict) or not aoi.get("geometry"):
        roi_path = str(cfg.get("roi_path") or "").strip()
        if roi_path and os.path.isfile(roi_path):
            try:
                import geopandas as gpd
                from aoi_context import aoi_from_bbox

                gdf = gpd.read_file(roi_path)
                if not gdf.empty:
                    west, south, east, north = [float(v) for v in gdf.total_bounds]
                    aoi = aoi_from_bbox(
                        west, south, east, north,
                        source="legacy_m4_roi_bbox", label=task or None,
                    ).to_dict()
            except Exception as exc:  # noqa: BLE001
                aoi = None
                cfg["_legacy_aoi_warning"] = f"旧 M4 ROI 无法转换为 AOI: {safe_error_summary(exc)}"
    plan = build_gee_download_plan(
        task_id=task,
        aoi=aoi if isinstance(aoi, dict) else {},
        start_date=str(cfg.get("start_date") or ""),
        end_date=str(cfg.get("end_date") or ""),
        collection=str(cfg.get("collection") or "COPERNICUS/S2_SR_HARMONIZED"),
        bands=list(cfg.get("bands") or DEFAULT_BANDS),
        index_bands=list(cfg.get("index_bands") or DEFAULT_INDEX_BANDS),
        cloud_limit=cfg.get("cloud_limit", 60),
        min_land_pct=cfg.get("min_land_pct", 5.0),
        max_land_pct=cfg.get("max_land_pct", 95.0),
        min_pixel_count=cfg.get("min_pixel_count", 1000),
        scale=cfg.get("scale", 10),
        export_to=str(cfg.get("export_to") or "local"),
        drive_folder=str(cfg.get("drive_folder") or "GEE_Downloads"),
        local_out_dir=str(cfg.get("local_out_dir") or ""),
        gee_proxy_url=str(cfg.get("gee_proxy_url") or ""),
        gee_project_id=str(cfg.get("gee_project_id") or ""),
    )
    warning = cfg.get("_legacy_aoi_warning")
    if warning:
        plan.setdefault("warnings", []).append(str(warning))
    return plan


def build_gee_download_plan(
    *,
    task_id: str,
    aoi: Dict[str, Any],
    start_date: str,
    end_date: str,
    collection: str = "COPERNICUS/S2_SR_HARMONIZED",
    bands: Optional[List[str]] = None,
    index_bands: Optional[List[str]] = None,
    cloud_limit: int = 60,
    min_land_pct: float = 5.0,
    max_land_pct: float = 95.0,
    min_pixel_count: int = 1000,
    scale: int = 10,
    export_to: str = "local",
    drive_folder: str = "GEE_Downloads",
    local_out_dir: str = "",
    gee_proxy_url: str = "",
    gee_project_id: str = "",
) -> Dict[str, Any]:
    """
    构建 GEE 下载执行计划（plan schema: gee_download_plan_v1）。

    - aoi: AOIContext.to_dict()（含 geometry，仅内部使用；对外只用 compact_summary）。
    - bands 默认 ["B4","B3","B2"] 顺序 RGB（匹配 pre_engine），显式传入则必须
      是 ALLOWED_BANDS 子集且非空；index_bands 仅辅助统计不导出。
    """
    from aoi_context import compact_summary

    task_id = str(task_id or "").strip()
    start_date = str(start_date or "").strip()
    end_date = str(end_date or "").strip()
    bands = list(bands or DEFAULT_BANDS)
    index_bands = list(index_bands or DEFAULT_INDEX_BANDS)

    blockers: List[str] = []
    warnings: List[str] = []

    if not task_id:
        blockers.append("未指定目标任务（task_id）。")
    if not start_date:
        blockers.append("未指定开始日期。")
    if not end_date:
        blockers.append("未指定结束日期。")

    aoi_ctx = None
    if isinstance(aoi, dict) and aoi.get("aoi_id"):
        try:
            from aoi_context import AOIContext
            aoi_ctx = AOIContext.from_dict(aoi)
        except Exception:  # noqa: BLE001
            warnings.append("AOI 上下文解析失败，按无效 AOI 处理。")
    if aoi_ctx is None or not aoi_ctx.valid or not aoi_ctx.geometry:
        blockers.append("AOI 无效（必须是合法 GeoJSON Polygon）。")
    else:
        warnings.extend(list(aoi_ctx.warnings or []))

    try:
        cloud = int(cloud_limit)
    except (TypeError, ValueError):
        cloud = -1
    if not (CLOUD_MIN <= cloud <= CLOUD_MAX):
        blockers.append(f"云量上限 {cloud_limit!r} 超出范围 [{CLOUD_MIN}, {CLOUD_MAX}]。")

    try:
        mland = float(min_land_pct)
        xland = float(max_land_pct)
    except (TypeError, ValueError):
        mland, xland = -1.0, -1.0
    if not (LAND_MIN <= mland <= LAND_MAX) or not (LAND_MIN <= xland <= LAND_MAX):
        blockers.append(f"水陆占比范围非法: [{min_land_pct}, {max_land_pct}]。")
    elif mland >= xland:
        blockers.append(f"最小陆地占比 ({mland}) 不能 ≥ 最大陆地占比 ({xland})。")

    try:
        mpix = int(min_pixel_count)
    except (TypeError, ValueError):
        mpix = -1
    if not (PIXEL_MIN <= mpix <= PIXEL_MAX):
        blockers.append(f"最小像素数 {min_pixel_count!r} 超出范围 [{PIXEL_MIN}, {PIXEL_MAX}]。")

    try:
        sc = int(scale)
    except (TypeError, ValueError):
        sc = -1
    if sc not in SCALE_OPTIONS:
        blockers.append(f"导出分辨率 {scale!r} 不合法（可选 {list(SCALE_OPTIONS)}）。")

    if str(export_to or "").strip().lower() not in EXPORT_OPTIONS:
        blockers.append(f"导出方式 {export_to!r} 不合法（可选 drive/local）。")

    if not bands:
        blockers.append("波段列表为空。")
    for b in bands:
        if b not in ALLOWED_BANDS:
            blockers.append(f"波段 {b!r} 不在允许集合内（{list(ALLOWED_BANDS)}）。")
    if len(bands) != len(set(bands)):
        blockers.append("波段列表含重复项。")

    # 导出目标（drive 需文件夹名；local 只在执行前验证目录）。
    # 计划构建应保持无副作用，并允许用户先确认一个尚未创建的目标目录。
    if str(export_to or "").strip().lower() == "drive":
        if not str(drive_folder or "").strip():
            blockers.append("Drive 导出需指定文件夹名。")
    else:
        if not str(local_out_dir or "").strip():
            blockers.append("本地导出需指定输出目录。")
        else:
            local_out_dir = str(local_out_dir).strip()

    # 日期合法性
    try:
        d0 = datetime.strptime(start_date, "%Y-%m-%d")
        d1 = datetime.strptime(end_date, "%Y-%m-%d")
        if d1 < d0:
            blockers.append("结束日期不能早于开始日期。")
        else:
            span_days = (d1 - d0).days + 1
            if span_days > 180:
                warnings.append(f"日期跨度达 {span_days} 天，GEE 将按月分批筛选，耗时可能较长。")
    except ValueError:
        blockers.append("日期格式须为 YYYY-MM-DD。")

    # 规模轻量预估（B4：太大 warning）
    est_pixels = 0
    if aoi_ctx and aoi_ctx.valid and sc in SCALE_OPTIONS:
        est_pixels = _estimate_pixels(aoi_ctx.area_km2, sc)
        if est_pixels > _MAX_ESTIMATED_PIXELS:
            warnings.append(
                f"预估像素量约 {est_pixels:,}（AOI {aoi_ctx.area_km2:.0f} km² @ {sc}m），"
                f"导出耗时与费用可能较高，建议缩小 AOI 或调低分辨率。"
            )

    aoi_summary = compact_summary(aoi_ctx) if aoi_ctx else "[当前AOI] 无效"

    steps: List[str] = [
        "加载 AOI 并转换到 GEE 几何（EPSG:4326）",
        f"初始化 Earth Engine（project 多源解析 + 网络代理 {gee_proxy_url or '直连/VPN'}）",
        f"云端筛选 {collection}（QA60 + Cloud Score+ + 水陆占比）",
        f"导出波段 {list(bands)}（顺序 RGB，匹配推理 pre_engine）",
        "提交导出 / 本地下载并跟踪 GEE 任务状态",
        "校验本地 GeoTIFF（存在/非空/CRS/波段/非全 NoData）",
        "登记 GEE 数据集资产（含 scene_count）",
    ]

    ready = len(blockers) == 0

    plan: Dict[str, Any] = {
        "schema": "gee_download_plan_v1",
        "plan_id": new_plan_id(),
        "task_id": task_id,
        "tool": TOOL_NAME,
        "aoi": dict(aoi) if isinstance(aoi, dict) else {},   # 内部使用（含 geometry）
        "aoi_summary": aoi_summary,                          # LLM 注入仅此文本（无完整 GeoJSON）
        "start_date": start_date,
        "end_date": end_date,
        "collection": collection,
        "bands": bands,
        "index_bands": index_bands,
        "cloud_limit": cloud,
        "min_land_pct": mland,
        "max_land_pct": xland,
        "min_pixel_count": mpix,
        "scale": sc,
        "export_to": str(export_to or "").strip().lower(),
        "drive_folder": str(drive_folder or "").strip(),
        "local_out_dir": os.path.normpath(str(local_out_dir or "").strip()),
        "gee_proxy_url": str(gee_proxy_url or "").strip(),
        "gee_project_id": str(gee_project_id or "").strip(),
        "estimated_pixels": est_pixels,
        "ready": ready,
        "blockers": blockers,
        "warnings": warnings,
        "steps": steps,
        "status": "waiting_confirmation" if ready else "blocked",
        "created_at": _now_str(),
    }
    return plan


# =======================================================
#  2. 执行前校验（B4）
# =======================================================
def validate_gee_download_plan(
    plan: Dict[str, Any],
    *,
    check_ee_initialize: bool = False,
) -> Tuple[bool, List[str]]:
    """
    执行前校验（B4）。返回 (ok, blockers)。
    - ee 可导入；credentials 存在；project 多源可解析；proxy 格式合法（不输出凭证）；
    - AOI 有效 / bbox 合法 / 面积>0 / 面积>50万km² → warning；
    - 时间 / 波段 / 输出可写（build 已检，此处复核关键项）；
    - 初始化只发生在 execute，validate 默认不真实连接 GEE（check_ee_initialize=True
      仅供需要提前探活的场景）。
    """
    blockers: List[str] = []
    warnings_extra: List[str] = []

    if not plan:
        return False, ["计划为空。"]

    # ee 可导入
    try:
        import ee  # noqa: F401
    except Exception as e:  # noqa: BLE001
        return False, [f"GEE Python 包不可用（{safe_error_summary(e)}），请安装 ee 包。"]

    # 凭证文件
    if not _credentials_file_ok():
        blockers.append("未找到 GEE 凭证文件，请先执行 earthengine authenticate。")

    # project 多源解析（不输出凭证内容）
    project = _resolve_ee_project_any(plan.get("gee_project_id"))
    if not project:
        blockers.append(
            "未解析到 GEE Cloud Project（override / EE_PROJECT 等环境变量 / "
            "~/.config/earthengine/project / credentials.project 均未配置）。"
        )

    # proxy 格式
    proxy_ok, proxy_err = _proxy_format_ok(plan.get("gee_proxy_url"))
    if not proxy_ok:
        blockers.append(f"GEE 网络代理格式错误：{proxy_err}")

    # AOI
    aoi = plan.get("aoi") or {}
    if not aoi or not aoi.get("valid") or not aoi.get("geometry"):
        blockers.append("AOI 无效（必须为合法 GeoJSON Polygon）。")
    else:
        bbox = aoi.get("bbox") or (0, 0, 0, 0)
        if len(bbox) != 4:
            blockers.append("AOI bbox 非法。")
        else:
            w, s, e_, n = bbox
            if not (-180.0 <= w <= 180.0 and -180.0 <= e_ <= 180.0
                    and -90.0 <= s <= 90.0 and -90.0 <= n <= 90.0):
                blockers.append(f"AOI bbox 越界: {bbox}（lon∈[-180,180], lat∈[-90,90]）。")
            elif w > e_:
                warnings_extra.append("AOI 横跨反经线（west>east），GEE 导出可能需特殊处理。")
        area = float(aoi.get("area_km2") or 0.0)
        if area <= 0:
            blockers.append("AOI 面积 ≤ 0。")
        elif area > 500_000:
            warnings_extra.append(f"AOI 面积 {area:.0f} km² 过大，建议缩小范围。")

    # 时间
    try:
        d0 = datetime.strptime(str(plan.get("start_date") or ""), "%Y-%m-%d")
        d1 = datetime.strptime(str(plan.get("end_date") or ""), "%Y-%m-%d")
        if d1 < d0:
            blockers.append("结束日期不能早于开始日期。")
    except ValueError:
        blockers.append("日期格式须为 YYYY-MM-DD。")

    # 波段
    bands = list(plan.get("bands") or [])
    if not bands:
        blockers.append("波段列表为空。")
    for b in bands:
        if b not in ALLOWED_BANDS:
            blockers.append(f"波段 {b!r} 不在允许集合内。")

    # 输出可写（local）——已由 build 探测，此处复验目录存在
    if str(plan.get("export_to") or "").strip().lower() == "local":
        out_dir = str(plan.get("local_out_dir") or "")
        if not out_dir:
            blockers.append("本地导出需指定输出目录。")
        elif not os.path.isdir(out_dir):
            blockers.append(f"本地输出目录不存在: {rel_path(out_dir)}")

    # 真实 ee.Initialize（B4：初始化检查）
    if check_ee_initialize and not blockers:
        try:
            import m4_engine
            m4_engine.ensure_ee_initialized(
                gee_proxy_url=plan.get("gee_proxy_url"),
                gee_project_id=plan.get("gee_project_id"),
                push_log=lambda m: None,
            )
        except Exception as e:  # noqa: BLE001
            blockers.append(f"GEE 初始化失败：{safe_error_summary(e)}")

    # 合并 warnings（build 的 warnings + 本次新增）
    plan["warnings"] = list(dict.fromkeys(list(plan.get("warnings") or []) + warnings_extra))
    return len(blockers) == 0, blockers


# =======================================================
#  3. 确认门闩（B5）：未确认绝不 task.start()
# =======================================================
def confirm_gee_download_plan(state: Dict[str, Any], plan_id: str) -> Tuple[bool, Optional[str]]:
    """
    确认门闩（与 UI 按钮共用同一逻辑）：
    - 计划必须存在于 _gee_pending_plan 且 plan_id 匹配；
    - 同一 plan_id 只确认一次（重复确认 → 错误，不重复执行）；
    - 多个 pending 计划不串用（确认必须匹配当前 pending 的 plan_id）。
    返回 (ok, error)。
    """
    pending = state.get(STATE_GEE_PENDING_PLAN) or {}
    if not pending:
        return False, "没有待确认的 GEE 下载计划（请先生成计划）。"
    if str(pending.get("plan_id") or "") != str(plan_id):
        return False, "计划已变化（plan_id 不匹配），请重新生成计划。"
    confirmed = state.get(STATE_GEE_PLAN_CONFIRMED) or set()
    if plan_id in confirmed:
        return False, "该计划已确认，请勿重复确认（不会重复执行）。"
    confirmed.add(plan_id)
    state[STATE_GEE_PLAN_CONFIRMED] = confirmed
    pending["status"] = "confirmed"
    return True, None


def is_gee_plan_confirmed(state: Dict[str, Any], plan_id: Optional[str]) -> bool:
    if not plan_id:
        return False
    confirmed = state.get(STATE_GEE_PLAN_CONFIRMED) or set()
    return plan_id in confirmed


def cancel_gee_download_plan(state: Dict[str, Any]) -> None:
    """取消：清除待确认计划（不可恢复）。"""
    state.pop(STATE_GEE_PENDING_PLAN, None)


# =======================================================
#  4. 真实执行（B6/B7）：复用 m4_engine，不 Fake success
# =======================================================
def execute_gee_download(
    plan: Dict[str, Any],
    *,
    stop_event: Optional[Any] = None,
    push_log: Callable[[str], None] = print,
    push_progress: Optional[Callable[[int], None]] = None,
    m4_engine_mod: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    真实调用 m4_engine.run_m4_download 执行下载。

    返回 ToolResult（只填真实数据）：
      success / task_id / plan_id / tool / status / inputs / parameters /
      outputs{export_to, local_out_dir, drive_folder, gee_task_ids, local_tifs} /
      metrics{elapsed_seconds, image_count, scene_count} /
      export_state{READY|RUNNING|COMPLETED|FAILED|CANCELLED} / warnings / error
    """
    import time

    started = time.time()
    task_id = str(plan.get("task_id") or "")
    plan_id = str(plan.get("plan_id") or "")
    export_to = str(plan.get("export_to") or "local").strip().lower()
    local_out_dir = str(plan.get("local_out_dir") or "")
    drive_folder = str(plan.get("drive_folder") or "GEE_Downloads")
    bands = list(plan.get("bands") or DEFAULT_BANDS)
    warnings: List[str] = []

    def check_stop() -> bool:
        return bool(stop_event and stop_event.is_set())

    def pct(v: int) -> None:
        if push_progress:
            push_progress(int(min(100, max(0, v))))

    m4 = m4_engine_mod if m4_engine_mod is not None else _default_m4_engine()

    # 进程重启后先恢复同一 plan 的远端 task_id，禁止再次提交重复导出。
    if export_to == "drive":
        existing = [
            row for row in _load_task_ledger().values()
            if row.get("plan_id") == plan_id and row.get("gee_task_id")
        ]
        if existing:
            states = []
            for row in existing:
                gid = str(row.get("gee_task_id"))
                state = _poll_gee_task_status(gid)
                normalized = str(state.get("state") or "UNKNOWN").upper()
                states.append((gid, normalized, str(state.get("error_message") or "")))
                _ledger_upsert(task_id, plan_id=plan_id, gee_task_id=gid,
                               status=normalized, error_message=state.get("error_message") or "")
            if any(st == "UNKNOWN" for _, st, _ in states):
                export_state = "UNKNOWN"
                success = False
                error = "无法确认已有 GEE 任务状态，未重新提交导出。"
            elif any(st in {"FAILED", "CANCELLED"} for _, st, _ in states):
                export_state = "FAILED"
                success = False
                error = "已有 GEE 任务失败或已取消；请重新生成计划后再确认。"
            elif all(st == "COMPLETED" for _, st, _ in states):
                export_state = "COMPLETED"
                success = True
                error = None
            else:
                export_state = "RUNNING"
                success = True
                error = None
            local_tifs = [rel_path(f) for f in _list_local_tifs(local_out_dir)]
            return {
                "success": success, "task_id": task_id, "plan_id": plan_id,
                "tool": TOOL_NAME, "status": "completed" if success else "failed",
                "inputs": {"aoi_summary": plan.get("aoi_summary") or "", "export_to": export_to},
                "parameters": {"cloud_limit": plan.get("cloud_limit"), "scale": plan.get("scale")},
                "outputs": {"export_to": export_to, "local_out_dir": local_out_dir,
                            "drive_folder": drive_folder, "gee_task_ids": [gid for gid, _, _ in states],
                            "local_tifs": local_tifs},
                "metrics": {"elapsed_seconds": round(time.time() - started, 2),
                            "image_count": len(states), "scene_count": len(states)},
                "export_state": export_state, "warnings": ["恢复已有 GEE task_id，未重复提交。"],
                "error": error,
            }

    # 复用现有 AOI 上下文构建 ROI 矢量（不重新实现 GEE 逻辑）
    roi_path, roi_name = _materialize_aoi_for_m4(plan, push_log)
    if not roi_path:
        return _gee_failure(
            task_id, plan_id, "AOI 无法转换为 ROI 矢量文件，无法执行下载。",
            started=started, warnings=warnings,
        )

    try:
        push_log(f"[GEE] 调用 m4_engine.run_m4_download（export_to={export_to}）…")
        pct(2)

        def on_task_started(task_obj: Any) -> None:
            """drive 模式：捕获每个 ee.batch.Task 的 id/description/开始时间。"""
            try:
                tid = str(task_obj.id)
                desc = str(task_obj.description or "")
                _ledger_upsert(task_id, plan_id=plan_id, gee_task_id=tid,
                               status="READY", description=desc,
                               export_to="drive", created_at=_now_str())
                push_log(f"[GEE] 任务已提交: {desc} (id={tid})")
            except Exception as e:  # noqa: BLE001
                warnings.append(f"记录 GEE 任务失败: {safe_error_summary(e)}")

        m4_result = m4.run_m4_download(
            roi_path=roi_path,
            roi_name=roi_name,
            start_date=str(plan.get("start_date") or ""),
            end_date=str(plan.get("end_date") or ""),
            export_to=export_to,
            local_out_dir=local_out_dir,
            bands=bands,
            cloud_limit=int(plan.get("cloud_limit") or 60),
            min_land_pct=float(plan.get("min_land_pct") or 5.0),
            max_land_pct=float(plan.get("max_land_pct") or 95.0),
            min_pixel_count=int(plan.get("min_pixel_count") or 1000),
            drive_folder=drive_folder,
            scale=int(plan.get("scale") or 10),
            gee_proxy_url=plan.get("gee_proxy_url"),
            gee_project_id=plan.get("gee_project_id"),
            push_log=push_log,
            push_progress=push_progress,
            stop_callback=check_stop,
            on_task_started=on_task_started,
        )
        if m4_result is None:
            return _gee_failure(
                task_id, plan_id, "下载被用户中断。", started=started, warnings=warnings,
            )

        image_count = int(m4_result.get("image_count") or 0)
        id_list = list(m4_result.get("id_list") or [])
        scene_count = len(id_list)

        outputs: Dict[str, Any] = {
            "export_to": export_to,
            "local_out_dir": local_out_dir if export_to == "local" else None,
            "drive_folder": drive_folder if export_to == "drive" else None,
            "gee_task_ids": [],
            "local_tifs": [],
        }

        export_state = "COMPLETED" if export_to == "local" else "READY"

        if export_to == "drive":
            # 从账本收集本 plan 已提交的 GEE 任务
            ledger = _load_task_ledger()
            gee_ids = []
            for row in ledger.values():
                if row.get("plan_id") == plan_id and row.get("gee_task_id"):
                    gee_ids.append(str(row["gee_task_id"]))
            outputs["gee_task_ids"] = sorted(set(gee_ids))
            if outputs["gee_task_ids"]:
                export_state = "RUNNING"
                # 真实轮询一次当前状态
                states = []
                for gid in outputs["gee_task_ids"]:
                    st = _poll_gee_task_status(gid)
                    states.append(st)
                    _ledger_upsert(task_id, gee_task_id=gid, status=st.get("state"),
                                   error_message=st.get("error_message") or "")
                done = [s for s in states if s.get("state") in ("COMPLETED", "FAILED", "CANCELLED")]
                if done and len(done) == len(states):
                    export_state = "COMPLETED" if all(
                        s.get("state") == "COMPLETED" for s in states) else "FAILED"
                # 本地文件（Drive 同步后可能已有）
                outputs["local_tifs"] = [rel_path(f) for f in _list_local_tifs(local_out_dir)]
            else:
                warnings.append("未捕获到 GEE 任务 ID（任务已提交但无法跟踪状态）。")
        else:
            # Prefer the exact scene filenames returned by M4.  Counting every
            # ``*.tif`` in an output directory would let stale files from a
            # previous plan satisfy the current scene count.
            roi_name = str(m4_result.get("roi_name") or "").strip()
            expected_paths = [
                os.path.join(local_out_dir, f"{roi_name}_{scene_id}.tif")
                for scene_id in id_list
            ] if roi_name and id_list else []
            reported_paths = m4_result.get("local_tifs")
            if expected_paths:
                candidate_paths = expected_paths
            elif isinstance(reported_paths, list):
                candidate_paths = [str(path) for path in reported_paths]
            else:
                candidate_paths = []
            outputs["local_tifs"] = [
                rel_path(path) for path in candidate_paths
                if path and os.path.isfile(path) and os.path.getsize(path) > 0
            ]
            scene_count = max(scene_count, len(outputs["local_tifs"]))
            # ``m4_engine`` is expected to return only after every requested
            # local export has produced a non-empty file.  Keep this boundary
            # fail-closed as well: custom/legacy adapters must not turn a
            # successful cloud query with zero local assets into a completed
            # download result.
            if image_count <= 0 or len(outputs["local_tifs"]) < image_count:
                return _gee_failure(
                    task_id,
                    plan_id,
                    f"本地影像下载未完成：期望至少 {image_count} 景，实际发现 {len(outputs['local_tifs'])} 个文件。",
                    started=started,
                    warnings=warnings,
                )

        elapsed = round(time.time() - started, 2)
        pct(100)
        push_log(f"[GEE] 下载完成（{export_state}），scene_count={scene_count}，耗时 {elapsed}s。")

        _ledger_upsert(task_id, plan_id=plan_id, status=export_state, export_to=export_to)

        return {
            "success": True,
            "task_id": task_id,
            "plan_id": plan_id,
            "tool": TOOL_NAME,
            "status": "completed",
            "inputs": {
                "aoi_summary": plan.get("aoi_summary") or "",
                "start_date": plan.get("start_date"),
                "end_date": plan.get("end_date"),
                "collection": plan.get("collection"),
                "bands": bands,
                "scale": plan.get("scale"),
                "export_to": export_to,
                "gee_proxy_url": plan.get("gee_proxy_url") or "",
            },
            "parameters": {
                "cloud_limit": plan.get("cloud_limit"),
                "min_land_pct": plan.get("min_land_pct"),
                "max_land_pct": plan.get("max_land_pct"),
                "min_pixel_count": plan.get("min_pixel_count"),
            },
            "outputs": outputs,
            "metrics": {
                "elapsed_seconds": elapsed,
                "image_count": image_count,
                "scene_count": scene_count,
            },
            "export_state": export_state,
            "warnings": warnings,
            "error": None,
        }
    except Exception as e:  # noqa: BLE001
        return _gee_failure(task_id, plan_id, f"GEE 下载执行异常: {safe_error_summary(e)}",
                            started=started, warnings=warnings)


def _default_m4_engine() -> Any:
    import m4_engine
    return m4_engine


def _materialize_aoi_for_m4(plan: Dict[str, Any], push_log: Callable[[str], None]) -> Tuple[Optional[str], str]:
    """
    将计划内 AOI（GeoJSON Polygon，EPSG:4326）写成临时 .geojson，供 m4_engine
    复用现有 _load_roi_geometry 加载。返回 (geojson_path, roi_name)。
    """
    aoi = plan.get("aoi") or {}
    geometry = aoi.get("geometry")
    task_id = str(plan.get("task_id") or "aoi")
    if not geometry or geometry.get("type") != "Polygon":
        push_log("[GEE] AOI 无有效 Polygon 几何。")
        return None, task_id
    try:
        import tempfile
        tmp_dir = os.path.join(tempfile.gettempdir(), "cstf_gee_aoi")
        os.makedirs(tmp_dir, exist_ok=True)
        safe = re.sub(r"[^\w\u4e00-\u9fff-]+", "_", task_id, flags=re.UNICODE).strip("_") or "aoi"
        path = os.path.join(tmp_dir, f"{safe}_{plan.get('plan_id', '')[:8]}.geojson")
        with open(path, "w", encoding="utf-8") as f:
            json.dump({
                "type": "FeatureCollection",
                "features": [{"type": "Feature", "properties": {"name": task_id},
                              "geometry": geometry}],
            }, f, ensure_ascii=False)
        return path, task_id
    except Exception as e:  # noqa: BLE001
        push_log(f"[GEE] AOI 写入失败: {safe_error_summary(e)}")
        return None, task_id


def _gee_failure(task_id: str, plan_id: str, error: str, *,
                 started: float, warnings: List[str]) -> Dict[str, Any]:
    import time
    return {
        "success": False,
        "task_id": task_id,
        "plan_id": plan_id,
        "tool": TOOL_NAME,
        "status": "failed",
        "inputs": {},
        "parameters": {},
        "outputs": {},
        "metrics": {"elapsed_seconds": round(time.time() - started, 2),
                    "image_count": 0, "scene_count": 0},
        "export_state": "FAILED",
        "warnings": warnings,
        "error": error,
    }


# =======================================================
#  5. 本地输出验证（B8）：GEE_EXPORT_COMPLETED ≠ LOCAL_ASSET_READY
# =======================================================
def verify_gee_outputs(
    plan: Dict[str, Any],
    result: Dict[str, Any],
    *,
    started_at: Optional[float] = None,
) -> Dict[str, Any]:
    """
    校验本地 GeoTIFF（B8）。返回 {"ok", "checks", "local_tifs", "asset_ready"}。
    - 只对本地文件校验；drive 模式若本地无文件 → ok=False, asset_ready=False
      （GEE_EXPORT_COMPLETED 不代表下载闭环完成）。
    """
    import time
    import rasterio

    started_at = started_at or (time.time() - 3600)
    checks: List[Dict[str, Any]] = []
    bands = list(plan.get("bands") or DEFAULT_BANDS)
    export_to = str(plan.get("export_to") or "local").strip().lower()

    def _check(name: str, passed: bool, detail: str = "") -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": str(detail)})

    local_tifs = [str(f) for f in (result.get("outputs") or {}).get("local_tifs") or []]
    local_tifs = [f for f in local_tifs if f and os.path.isfile(f)]

    _check("local_tifs_present", len(local_tifs) > 0,
           f"{len(local_tifs)} 个本地 GeoTIFF" if local_tifs else "本地无 GeoTIFF（drive 模式需先同步）")
    if not local_tifs:
        # drive 模式：区分云端导出状态
        export_state = str(result.get("export_state") or "")
        detail = ("GEE 云端导出已 COMPLETED，但本地文件未就绪，需从 Drive 同步后才能进入推理。"
                  if export_state == "COMPLETED" else f"云端导出状态: {export_state or '未知'}")
        _check("gee_export_completed", export_state == "COMPLETED", detail)
        return {"ok": False, "checks": checks, "local_tifs": [], "asset_ready": False}

    # 每个文件：存在 / 非空 / 可打开 / CRS / 尺寸 / 波段数 / 非全 NoData / mtime
    ok_files = 0
    for tif in local_tifs:
        try:
            fname = os.path.basename(tif)
            _check(f"exists:{fname}", True)
            _check(f"nonempty:{fname}", os.path.getsize(tif) > 0, f"{os.path.getsize(tif)} B")
            with rasterio.open(tif) as src:
                _check(f"open:{fname}", True, f"{src.width}×{src.height}")
                _check(f"crs:{fname}", src.crs is not None, str(src.crs))
                _check(f"size:{fname}", src.width > 0 and src.height > 0,
                       f"{src.width}×{src.height}")
                _check(f"band_count:{fname}", src.count == len(bands),
                       f"{src.count} vs {len(bands)}")
                # 非全 NoData
                try:
                    import numpy as np
                    sample = src.read(1, out_shape=(1, max(1, src.height // 16),
                                                    max(1, src.width // 16)))
                    nodata = src.nodata
                    valid = sample[~np.isnan(sample.astype(float))]
                    if nodata is not None:
                        valid = valid[valid != nodata]
                    _check(f"has_data:{fname}", bool(np.any(valid > 0)),
                           f"valid_pixels={int(np.count_nonzero(valid))}")
                except Exception as e:  # noqa: BLE001
                    _check(f"has_data:{fname}", False, f"读取采样失败: {safe_error_summary(e)}")
            try:
                mt = os.path.getmtime(tif)
                _check(f"mtime:{fname}", mt >= started_at,
                       f"mtime={mt:.0f} >= start={started_at:.0f}")
            except OSError:
                _check(f"mtime:{fname}", False, "mtime 不可读")
            ok_files += 1
        except Exception as e:  # noqa: BLE001
            _check(f"open:{os.path.basename(tif)}", False, f"打开失败: {safe_error_summary(e)}")

    ok = ok_files == len(local_tifs) and all(c["passed"] for c in checks)
    return {"ok": ok, "checks": checks, "local_tifs": local_tifs,
            "asset_ready": ok and len(local_tifs) > 0}


# =======================================================
#  6. 数据集资产登记（B9）：scene_count 必须进 metadata
# =======================================================
def register_gee_dataset_asset(
    plan: Dict[str, Any],
    result: Dict[str, Any],
    verification: Dict[str, Any],
    *,
    registry_path: Optional[str] = None,
) -> Optional[str]:
    """
    验证通过才登记（B9）。返回 dataset_id 或 None。
    - 验证失败不登记；同 plan_id 不重复登记；
    - metadata 含 asset_id/task_id/plan_id/source=GEE/provider=Earth Engine/
      collection/date_range/bands/cloud_limit/scale/AOI summary/gee_task_id/
      local files/CRS/bbox/scene_count/created_at/status。
    """
    import dataset_assets

    if not verification or verification.get("ok") is not True:
        return None
    if not result or result.get("success") is not True:
        return None

    task_id = str(plan.get("task_id") or "")
    plan_id = str(plan.get("plan_id") or "")
    if not task_id or not plan_id:
        return None

    reg = dataset_assets.load_registry()
    for eid, row in reg.items():
        if isinstance(row, dict) and row.get("plan_id") == plan_id:
            return str(eid)

    local_tifs = [f for f in (verification.get("local_tifs") or []) if f and os.path.isfile(f)]
    if not local_tifs:
        return None

    scene_count = int((result.get("metrics") or {}).get("scene_count") or len(local_tifs))

    # 首景文件元数据（CRS / bbox）
    crs = ""
    bbox = None
    try:
        import rasterio
        with rasterio.open(local_tifs[0]) as src:
            crs = str(src.crs or "")
            bbox = list(src.bounds)
    except Exception:  # noqa: BLE001
        pass

    aoi = plan.get("aoi") or {}
    aoi_bbox = aoi.get("bbox") or (0, 0, 0, 0)

    dataset_id = f"gee_{task_id}_{plan_id[:8]}"
    entry: Dict[str, Any] = {
        "id": dataset_id,
        "title": f"卫星影像 {task_id}（{plan.get('start_date')} ~ {plan.get('end_date')}）",
        "description": "由影像获取闭环导出的 Sentinel-2 多波段 GeoTIFF。",
        "source": "open",
        "provider": "Earth Engine",
        "collection": plan.get("collection") or "COPERNICUS/S2_SR_HARMONIZED",
        "date_range": [plan.get("start_date"), plan.get("end_date")],
        "bands": list(plan.get("bands") or DEFAULT_BANDS),
        "index_bands": list(plan.get("index_bands") or DEFAULT_INDEX_BANDS),
        "cloud_limit": plan.get("cloud_limit"),
        "min_land_pct": plan.get("min_land_pct"),
        "max_land_pct": plan.get("max_land_pct"),
        "min_pixel_count": plan.get("min_pixel_count"),
        "scale": plan.get("scale"),
        "format": "geotiff",
        "role": "auxiliary",
        "coverage_scale": "scene",
        "primary_path": os.path.abspath(local_tifs[0]),
        "task_id": task_id,
        "plan_id": plan_id,
        "asset_id": str(uuid.uuid4().hex),
        "source_kind": "GEE",
        "gee_task_id": (result.get("outputs") or {}).get("gee_task_ids") or [],
        "aoi_summary": plan.get("aoi_summary") or "",
        "aoi_bbox": list(aoi_bbox),
        "bbox": bbox,
        "crs": crs,
        "local_files": [os.path.abspath(f) for f in local_tifs],
        "scene_count": scene_count,
        "export_to": plan.get("export_to"),
        "drive_folder": plan.get("drive_folder"),
        "created_at": _now_str(),
        "status": "verified",
        "registered_at": _now_str(),
    }
    try:
        dataset_assets.register_dataset(entry, overwrite=False)
        return dataset_id
    except ValueError as e:
        # 同 id 已存在且为同一 plan → 视为已登记；否则返回 None
        existing = dataset_assets.get_dataset(dataset_id)
        if existing and existing.get("plan_id") == plan_id:
            return dataset_id
        raise ValueError(f"GEE 数据集登记失败: {safe_error_summary(e)}") from e


# =======================================================
#  7. 面向用户的计划 / 结果 / 上下文（B10）
# =======================================================
def format_gee_plan_for_user(plan: Dict[str, Any]) -> str:
    """确认前展示的执行计划（只含真实信息，无凭证 / 无完整 GeoJSON）。"""
    if not plan:
        return "尚未生成影像获取计划。"
    lines = ["## 获取卫星影像 · 执行计划", ""]
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
    lines.append(f"- AOI：{plan.get('aoi_summary') or '—'}")
    lines.append(f"- 日期：`{plan.get('start_date')} ~ {plan.get('end_date')}`")
    lines.append(f"- 集合：`{plan.get('collection')}`")
    lines.append(f"- 波段：{list(plan.get('bands') or [])}（RGB 顺序，匹配推理 pre_engine）")
    lines.append(f"- 云量上限：`{plan.get('cloud_limit')}` ｜ 水陆占比："
                 f"`[{plan.get('min_land_pct')}, {plan.get('max_land_pct')}]`")
    lines.append(f"- 分辨率：`{plan.get('scale')} m` ｜ 导出方式：`{plan.get('export_to')}`")
    lines.append(f"- 代理：`{plan.get('gee_proxy_url') or '直连/VPN'}` ｜ 项目："
                 f"`{'已解析' if _resolve_ee_project_any(plan.get('gee_project_id')) else '未解析'}`")
    lines.append("")
    lines.append("步骤：")
    for i, s in enumerate(plan.get("steps") or [], 1):
        lines.append(f"{i}. {s}")
    lines.append("")
    lines.append("确认后将真实获取影像，并根据云端任务状态与磁盘产物验证后回复。")
    return "\n".join(lines)


def summarize_gee_result_for_chat(
    result: Dict[str, Any],
    verification: Optional[Dict[str, Any]] = None,
) -> str:
    """基于真实工具结果生成 Copilot 回复（禁止编造指标 / 输出凭证）。"""
    if not result or result.get("success") is not True:
        err = sanitize_external_text((result or {}).get("error") or "影像获取失败")[:240]
        return (
            "## 获取卫星影像 · 未完成\n\n"
            f"- 任务：`{(result or {}).get('task_id') or '—'}`\n"
            f"- 状态：**失败**\n"
            f"- 原因：{err}\n\n"
            "未登记数据资产。请根据上述原因处理后重试。"
        )
    inputs = result.get("inputs") or {}
    metrics = result.get("metrics") or {}
    outputs = result.get("outputs") or {}
    export_state = str(result.get("export_state") or "COMPLETED")
    scene_count = int(metrics.get("scene_count") or 0)
    lines = [
        "## 获取卫星影像 · 完成",
        "",
        f"- 任务：`{result.get('task_id') or '—'}`",
        f"- AOI：{inputs.get('aoi_summary') or '—'}",
        f"- 日期：`{inputs.get('start_date')} ~ {inputs.get('end_date')}`",
        f"- 集合：`{inputs.get('collection')}`",
        f"- 波段：{list(inputs.get('bands') or [])}（RGB 顺序）",
        f"- 分辨率：`{inputs.get('scale')} m` ｜ 导出方式：`{inputs.get('export_to')}`",
        f"- 云端导出状态：`{export_state}`",
        f"- **共 {scene_count} 景影像**",
        f"- 耗时：{metrics.get('elapsed_seconds', 0.0):.1f}s",
    ]
    if verification:
        ok = bool(verification.get("ok"))
        lines.append(f"- 本地资产校验：{'✅ 通过（LOCAL_ASSET_READY）' if ok else '⏳ 未就绪'}")
        if not ok:
            failed = [c.get("name") for c in (verification.get("checks") or []) if not c.get("passed")]
            lines.append(f"  - 未通过项：{', '.join(failed) or '—'}")
    gee_ids = outputs.get("gee_task_ids") or []
    if gee_ids:
        lines.append(f"- GEE 任务数：{len(gee_ids)}（首任务 id：`{gee_ids[0]}`）")
    local = outputs.get("local_tifs") or (verification or {}).get("local_tifs") or []
    if local:
        lines.append(f"- 本地文件：{len(local)} 个 GeoTIFF（首个：`{os.path.basename(str(local[0]))}`）")
    lines.append("")
    lines.append("> 数据已就绪。**不会自动启动提取**——如需潮滩提取，"
                 "请回复「对 XX 做潮滩提取」以生成提取计划，确认后再执行。")
    return "\n".join(lines)


def build_gee_context_for_agent() -> Dict[str, Any]:
    """供 Agent 注入的 GEE 上下文（无凭证 / 无完整 GeoJSON）。"""
    project = _resolve_ee_project_any(None)
    return {
        "gee_project_resolved": bool(project),
        "credentials_file": _credentials_file_ok(),
        "default_bands": list(DEFAULT_BANDS),
        "allowed_bands": list(ALLOWED_BANDS),
        "export_options": list(EXPORT_OPTIONS),
        "scale_options": list(SCALE_OPTIONS),
        "note": "下载完成后仅更新能力状态，推理需用户另行确认。",
    }
