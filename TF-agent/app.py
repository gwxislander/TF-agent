import streamlit as st
import leafmap.foliumap as leafmap
import streamlit.components.v1 as components
from streamlit_folium import st_folium
import sidebar_ui as sbui
import ui_labels as uil
import hashlib
import os
import re
import glob
import time
import io
import tempfile

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))
except ImportError:
    pass

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# Clash 默认混合代理端口（可在 .env 用 GEE_PROXY_URL 覆盖）
DEFAULT_CLASH_PROXY = (os.environ.get("GEE_PROXY_URL") or "http://127.0.0.1:7892").strip()

# localtileserver 访问本机瓦片服务时若走系统代理(如 127.0.0.1:7892)会加载失败
_NO_PROXY = "127.0.0.1,localhost,::1"
for _pk in ("NO_PROXY", "no_proxy"):
    _cur = os.environ.get(_pk, "")
    if _cur:
        if "127.0.0.1" not in _cur:
            os.environ[_pk] = f"{_cur},{_NO_PROXY}"
    else:
        os.environ[_pk] = _NO_PROXY

import torch
import datetime
import json
import math
import contextlib
import threading
import traceback
import uuid
import numpy as np
from agent_context_policy import safe_error_summary, sanitize_external_text
from preview_cache import cleanup_preview_cache, preview_cache_dir


@contextlib.contextmanager
def _local_tile_no_proxy():
    """加载地图瓦片时临时禁用 HTTP 代理，避免 localhost 被转到 Clash 端口。"""
    keys = ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "ALL_PROXY", "all_proxy")
    saved = {k: os.environ.pop(k, None) for k in keys}
    try:
        yield
    finally:
        for k, v in saved.items():
            if v is not None:
                os.environ[k] = v

DEBUG_LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "agent_debug.log")


def _append_debug_log(message: str):
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(DEBUG_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(f"[{ts}] {message}\n")


def _format_agent_exception(exc: Exception) -> str:
    # UI/debug output must not echo provider response bodies, local paths or credentials.
    parts = [safe_error_summary(exc)]
    status = getattr(exc, "status_code", None)
    code = getattr(exc, "code", None)
    if status is not None:
        parts.append(f"status_code={status}")
    if code:
        parts.append(f"code={code}")
    return " | ".join(parts)


def _record_worker_exception(shared, label: str, error: BaseException) -> None:
    """Store only a bounded safe summary in UI worker state, never a traceback."""
    safe = safe_error_summary(error)
    with shared["lock"]:
        lines = list(shared.get("log_lines") or [])
        lines.append(f"[CRASH] {label}: {safe}")
        shared["log_lines"] = lines[-30:]
        shared["status"] = ("error", f"{label}：{safe}")


def _chat_preview_uint8(rgb: np.ndarray) -> np.ndarray:
    """Stretch raster preview to uint8 for chat display."""
    out = np.zeros(rgb.shape, dtype=np.uint8)
    valid = np.isfinite(rgb).all(axis=2)
    if not valid.any():
        return out
    for c in range(rgb.shape[2]):
        ch = rgb[..., c].astype(np.float32)
        vals = ch[valid]
        if vals.size == 0:
            continue
        lo = np.percentile(vals, 2)
        hi = np.percentile(vals, 98)
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            lo = np.min(vals)
            hi = np.max(vals)
        if hi <= lo:
            continue
        norm = np.clip((ch - lo) / (hi - lo), 0.0, 1.0)
        norm = np.where(np.isfinite(norm), norm, 0.0)
        out[..., c] = (norm * 255.0).astype(np.uint8)
    return out


def _save_chat_image_preview(uploaded_file):
    """Save a lightweight PNG preview for chat history rendering."""
    if uploaded_file is None:
        return None, None
    name = os.path.basename(getattr(uploaded_file, "name", "") or "upload_image")
    ext = os.path.splitext(name)[1].lower()
    safe = "".join(c for c in name if c.isalnum() or c in "._-") or "upload_image"
    preview_dir = preview_cache_dir(os.path.dirname(os.path.abspath(__file__)))
    os.makedirs(preview_dir, exist_ok=True)
    preview_path = os.path.join(preview_dir, f"preview_{uuid.uuid4().hex}_{safe}.png")
    raw = uploaded_file.getbuffer()

    try:
        from PIL import Image

        if ext in (".tif", ".tiff"):
            import rasterio
            from rasterio.io import MemoryFile

            with MemoryFile(bytes(raw)) as mem:
                with mem.open() as ds:
                    data = ds.read(masked=True)
            if data.shape[0] >= 3:
                rgb = np.moveaxis(data[:3, :, :], 0, -1)
            elif data.shape[0] == 2:
                two = np.moveaxis(data[:2, :, :], 0, -1)
                rgb = np.concatenate([two, two[..., 1:2]], axis=-1)
            else:
                one = data[0]
                rgb = np.repeat(one[:, :, None], 3, axis=2)

            rgb_plain = np.ma.filled(rgb, np.nan) if np.ma.isMaskedArray(rgb) else rgb
            valid = np.isfinite(rgb_plain).all(axis=2)
            if valid.any() and float(valid.mean()) < 0.70:
                ys, xs = np.where(valid)
                y0, y1 = ys.min(), ys.max()
                x0, x1 = xs.min(), xs.max()
                pad = 16
                y0 = max(0, y0 - pad)
                x0 = max(0, x0 - pad)
                y1 = min(rgb_plain.shape[0] - 1, y1 + pad)
                x1 = min(rgb_plain.shape[1] - 1, x1 + pad)
                rgb_plain = rgb_plain[y0 : y1 + 1, x0 : x1 + 1, :]

            preview = Image.fromarray(_chat_preview_uint8(rgb_plain), mode="RGB")
        else:
            preview = Image.open(io.BytesIO(bytes(raw))).convert("RGB")

        preview.thumbnail((1400, 1400), Image.Resampling.BILINEAR)
        preview.save(preview_path, format="PNG")
        return preview_path, name
    except Exception as e:
        _append_debug_log(f"save_chat_preview_failed: {safe_error_summary(e)}; file={name}")
        return None, name


def _dedupe_uploaded_images(uploaded_files):
    """Remove duplicate browser uploads before creating one chat message.

    A synthetic uploader ``change`` event can make Streamlit receive the same
    browser file twice with different server-side ``file_id`` values.  Use the
    id when available, and fall back to a content fingerprint so a repeated
    upload cannot produce duplicate previews or model inputs.
    """
    unique = []
    seen_ids = set()
    seen_fingerprints = set()
    for uploaded_file in list(uploaded_files or []):
        file_id = str(getattr(uploaded_file, "file_id", "") or "")
        if file_id and file_id in seen_ids:
            continue
        name = os.path.basename(str(getattr(uploaded_file, "name", "") or ""))
        file_type = str(getattr(uploaded_file, "type", "") or "")
        try:
            raw = bytes(uploaded_file.getbuffer())
            fingerprint = (
                name,
                file_type,
                len(raw),
                hashlib.sha256(raw).hexdigest(),
            )
        except Exception:
            fingerprint = (
                name,
                file_type,
                int(getattr(uploaded_file, "size", 0) or 0),
            )
        if fingerprint in seen_fingerprints:
            continue
        if file_id:
            seen_ids.add(file_id)
        seen_fingerprints.add(fingerprint)
        unique.append(uploaded_file)
    return unique


def _render_chat_attachment_previews(message):
    """Render every live local preview attached to one chat message."""
    preview_paths = message.get("image_preview_paths") or []
    image_names = message.get("image_names") or []
    if not isinstance(preview_paths, (list, tuple)):
        preview_paths = [preview_paths]
    if not isinstance(image_names, (list, tuple)):
        image_names = [image_names]
    if not preview_paths and message.get("image_preview_path"):
        preview_paths = [message.get("image_preview_path")]
        image_names = [message.get("image_name") or "uploaded image"]

    live_paths = []
    live_captions = []
    for index, preview_path in enumerate(preview_paths):
        if not preview_path or not os.path.exists(str(preview_path)):
            continue
        live_paths.append(str(preview_path))
        if index < len(image_names) and image_names[index]:
            live_captions.append(str(image_names[index]))
        else:
            live_captions.append(f"附件 {index + 1}")
    if live_paths:
        st.image(live_paths, caption=live_captions, width="stretch")


def _nodata_safe_for_tile_api(value):
    """
    GeoTIFF 的 nodata 常为 nan；传给 localtileserver 的 query 会序列化失败或行为异常，导致整段加载报错。
    返回可安全传入 API 的数，或 None 表示不传参（由服务端按文件元数据处理）。
    """
    if value is None:
        return None
    try:
        if isinstance(value, (float, np.floating)) and (np.isnan(value) or np.isinf(value)):
            return None
    except (TypeError, ValueError):
        return None
    return value


# Agent 暗号解析：须兼容 ① 标准竖线 ② 模型常瞎写的括号逗号 ③ 省略 zoom（默认 8）
_RE_CMD_MAP_PIPE = re.compile(
    r"COMMAND_UPDATE_MAP\s*\|\s*([-\d.]+)\s*\|\s*([-\d.]+)\s*\|\s*(\d+)",
    re.IGNORECASE,
)
_RE_CMD_MAP_PAREN = re.compile(
    r"COMMAND_UPDATE_MAP\s*[\(:（]\s*([-\d.]+)\s*[,，]\s*([-\d.]+)(?:\s*[,，]\s*(\d+))?\s*[\)）]?",
    re.IGNORECASE,
)
_RE_CMD_PIPELINE = re.compile(
    r"COMMAND_RUN_PIPELINE\s*\|\s*([^|\n]+?)\s*\|\s*([-\d.]+)\s*\|\s*(\d+)",
    re.IGNORECASE,
)
# 模型常只写自然语言坐标而未附 SYSTEM_COMMAND / COMMAND_UPDATE_MAP
_RE_MAP_COORDS_CHINESE = re.compile(
    r"(?:北纬|纬度)\s*([+-]?\d+(?:\.\d+)?)\s*[°º度]?\s*[,，、;/\s]+\s*"
    r"(?:东经|经度)\s*([+-]?\d+(?:\.\d+)?)\s*[°º度]?",
)
_RE_MAP_COORDS_NSEW = re.compile(
    r"([-\d.]+)\s*[°º]?\s*[Nn北]\s*[,，/]\s*([-\d.]+)\s*[°º]?\s*[Ee东]",
)
_RE_MAP_COORDS_PLAIN = re.compile(
    r"(?:中心点|中心坐标|中心|经纬度|坐标|定位(?:至|到)?|跳转(?:至|到)?|视角)\s*"
    r"(?:约|为)?\s*[：:=]?\s*[（(]?\s*([-\d.]+)\s*[,，]\s*([-\d.]+)\s*[)）]?",
)
_RE_MAP_ZOOM = re.compile(
    r"(?:缩放(?:级别|等级)?|zoom)\s*(?:为|到|=|：|:)?\s*(\d{1,2})",
    re.IGNORECASE,
)
_RE_MAP_INTENT = re.compile(
    r"(已定位|已跳转|已将地图|已为您定位|地图视角|视角已|飞到|定位到|定位至|跳转到|挪到|中心点)",
)
_RE_MAP_LABEL = re.compile(
    r"(?:已将地图(?:视角)?定位到|已成功定位到|已为您定位至|已定位到|已定位|定位到|定位至)\s*"
    r"([^\s（(，,。；;：:\n]{1,30}?)(?:区域)?(?=[（(，,。；;：:\s]|$)"
)


def _parse_agent_map_command(reply: str):
    """解析地图跳转：标准暗号，或模型自然语言中的坐标+缩放。"""
    stripped = re.sub(r"[`\*_]+", " ", reply or "")
    flat = re.sub(r"[\n\r]+", " ", stripped)
    for text in (flat, stripped, reply or ""):
        m = _RE_CMD_MAP_PIPE.search(text)
        if m:
            try:
                return float(m.group(1)), float(m.group(2)), int(m.group(3)), m.group(0)
            except (ValueError, TypeError):
                pass
        m = _RE_CMD_MAP_PAREN.search(text)
        if m:
            try:
                lat = float(m.group(1))
                lon = float(m.group(2))
                zoom = int(m.group(3)) if m.group(3) else 8
                return lat, lon, zoom, m.group(0)
            except (ValueError, TypeError):
                pass

    # 自然语言回退：仅在明确“定位/跳转”语境下提取，避免误伤普通问答
    if not _RE_MAP_INTENT.search(flat):
        return None
    lat = lon = None
    span = ""
    m = _RE_MAP_COORDS_CHINESE.search(flat)
    if m:
        try:
            lat, lon = float(m.group(1)), float(m.group(2))
            span = m.group(0)
        except (ValueError, TypeError):
            lat = lon = None
    if lat is None:
        m = _RE_MAP_COORDS_NSEW.search(flat)
    else:
        m = None
    if m:
        try:
            lat, lon = float(m.group(1)), float(m.group(2))
            span = m.group(0)
        except (ValueError, TypeError):
            lat = lon = None
    if lat is None:
        m = _RE_MAP_COORDS_PLAIN.search(flat)
        if m:
            try:
                lat, lon = float(m.group(1)), float(m.group(2))
                span = m.group(0)
            except (ValueError, TypeError):
                lat = lon = None
    if lat is None or lon is None:
        return None
    if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
        return None
    zoom = 9
    mz = _RE_MAP_ZOOM.search(flat)
    if mz:
        try:
            zoom = max(1, min(18, int(mz.group(1))))
        except (ValueError, TypeError):
            zoom = 9
    return lat, lon, zoom, span or f"{lat},{lon},{zoom}"


def _parse_agent_map_label(reply: str) -> str:
    """Extract the place name from a model's explicit map-location claim."""
    flat = re.sub(r"[`*_\n\r]+", " " , reply or "")
    match = _RE_MAP_LABEL.search(flat)
    return match.group(1).strip() if match else ""


def _strip_map_command_from_reply(reply: str) -> str:
    """从回复中去掉地图暗号片段（含 Markdown 包裹）。"""
    t = reply
    for pat in (_RE_CMD_MAP_PIPE, _RE_CMD_MAP_PAREN):
        t = pat.sub("", t, count=1)
    t = re.sub(r"[`\*_]+", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def _parse_agent_pipeline_command(reply: str):
    flat = re.sub(r"[\n\r]+", " ", reply)
    m = _RE_CMD_PIPELINE.search(flat) or _RE_CMD_PIPELINE.search(reply)
    if not m:
        return None
    try:
        task = m.group(1).strip()
        prob = float(m.group(2))
        cnt = int(m.group(3))
        return task, prob, cnt, m.group(0)
    except (ValueError, TypeError):
        return None

# =======================================================
#  0. 导入后端引擎与智能体大脑
# =======================================================
try:
    import pre_engine
    import post_engine
except ImportError:
    st.error("⚠️ 未找到后端引擎文件 (pre_engine.py, post_engine.py)。")

# =======================================================
#  数据资产管理
# =======================================================
ASSET_REGISTRY_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets_registry.json")


def load_asset_registry():
    # Keep one strict read path for both the UI and Workflow executors.  A
    # corrupt registry is preserved and rejected instead of being treated as
    # an empty registry that a later registration could overwrite.
    from workflow_orchestrator import load_assets_registry

    return load_assets_registry(ASSET_REGISTRY_PATH)


def save_asset_registry(registry):
    from workflow_orchestrator import save_assets_registry

    save_assets_registry(registry, ASSET_REGISTRY_PATH)


def register_asset(task, prob, cnt, file_path):
    registry = load_asset_registry()
    asset_key = f"{task}_p{prob:.2f}_c{cnt}"
    file_path = os.path.normpath(os.path.abspath(str(file_path).strip().strip('"').strip("'")))
    size_mb = 0
    if os.path.exists(file_path):
        if file_path.lower().endswith(".shp"):
            stem = os.path.splitext(file_path)[0]
            for ext in (".shp", ".shx", ".dbf", ".prj", ".cpg"):
                side = stem + ext
                if os.path.isfile(side):
                    size_mb += os.path.getsize(side)
        else:
            size_mb = os.path.getsize(file_path)
    registry[asset_key] = {
        "task": task,
        "prob_threshold": prob,
        "min_count": cnt,
        "file_path": file_path,
        "created_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "file_size_mb": round(size_mb / (1024 ** 2), 2) if size_mb else 0
    }
    save_asset_registry(registry)
    return asset_key


def find_asset(task, prob, cnt):
    registry = load_asset_registry()
    asset_key = f"{task}_p{prob:.2f}_c{cnt}"
    entry = registry.get(asset_key)
    if entry and os.path.exists(entry.get("file_path", "")):
        return entry
    return None


def register_index_asset(task, file_path):
    registry = load_asset_registry()
    asset_key = f"{task}_index"
    file_path = os.path.normpath(os.path.abspath(str(file_path).strip().strip('"').strip("'")))
    registry[asset_key] = {
        "task": task,
        "method": "index",
        "prob_threshold": None,
        "min_count": None,
        "file_path": file_path,
        "created_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "file_size_mb": round(os.path.getsize(file_path) / (1024 ** 2), 2) if os.path.exists(file_path) else 0,
    }
    save_asset_registry(registry)
    return asset_key


def _nonempty_file(path):
    """Postflight asset gate shared by independent report registries."""
    try:
        return bool(path) and os.path.isfile(str(path)) and os.path.getsize(str(path)) > 0
    except OSError:
        return False


def register_m5_asset(task, report: dict):
    """将 M5 报告与差异面登记到资产账本。"""
    import m5_agent_loop

    report_path = (report or {}).get("report_path")
    if not _nonempty_file(report_path):
        return None
    registry = load_asset_registry()
    asset_key = f"{task}_m5"
    map_path = m5_agent_loop.pick_m5_map_path(report)
    spatial = (report or {}).get("spatial_outputs") or {}
    loss = spatial.get("loss_shapefile_path")
    silt = spatial.get("siltation_shapefile_path")
    if loss and str(loss) == "None":
        loss = None
    if silt and str(silt) == "None":
        silt = None
    size_mb = 0.0
    for p in (map_path, loss, silt, report_path):
        if not p or not os.path.isfile(str(p)):
            continue
        try:
            size_mb += os.path.getsize(str(p)) / (1024 ** 2)
        except OSError:
            pass
    registry[asset_key] = {
        "task": task,
        "method": "m5",
        "file_path": os.path.normpath(map_path) if map_path else "",
        "report_path": report_path,
        "loss_shp": loss if loss and os.path.isfile(str(loss)) else None,
        "siltation_shp": silt if silt and os.path.isfile(str(silt)) else None,
        "baseline_task": (report or {}).get("baseline_task"),
        "alert_level": (report or {}).get("alert_level"),
        "created_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "file_size_mb": round(size_mb, 2),
    }
    save_asset_registry(registry)
    return asset_key


def find_m5_asset(task):
    registry = load_asset_registry()
    entry = registry.get(f"{task}_m5")
    if not entry:
        return None
    rp = entry.get("report_path") or entry.get("file_path")
    if rp and os.path.exists(str(rp)):
        return entry
    if entry.get("file_path") and os.path.exists(entry["file_path"]):
        return entry
    return None


def register_e1_asset(task, report: dict):
    """将 E1 报告与可选热力/分歧图登记到资产账本。"""
    import e1_agent_loop

    report_path = (report or {}).get("report_path")
    if not _nonempty_file(report_path):
        return None
    registry = load_asset_registry()
    asset_key = f"{task}_e1"
    map_path = e1_agent_loop.pick_e1_map_path(report)
    size_mb = 0.0
    for p in (map_path, report_path, (report or {}).get("report_md_path")):
        if not p or not os.path.isfile(str(p)):
            continue
        try:
            size_mb += os.path.getsize(str(p)) / (1024 ** 2)
        except OSError:
            pass
    registry[asset_key] = {
        "task": task,
        "method": "e1",
        "file_path": os.path.normpath(map_path) if map_path else "",
        "report_path": report_path,
        "report_md_path": (report or {}).get("report_md_path"),
        "reference": (report or {}).get("reference"),
        "created_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "file_size_mb": round(size_mb, 2),
    }
    save_asset_registry(registry)
    return asset_key


def find_index_asset(task):
    registry = load_asset_registry()
    entry = registry.get(f"{task}_index")
    if entry and os.path.exists(entry.get("file_path", "")):
        return entry
    return None


def get_task_assets(task):
    registry = load_asset_registry()
    return {k: v for k, v in registry.items()
            if v.get("task") == task and os.path.exists(v.get("file_path", ""))}


def scan_and_register_existing(final_root):
    """首次启动时扫描输出目录，将已有的 Final SHP/TIF 自动注册到资产库。"""
    if not os.path.exists(final_root):
        return
    registry = load_asset_registry()
    changed = False
    for task_dir in os.listdir(final_root):
        task_path = os.path.join(final_root, task_dir)
        if not os.path.isdir(task_path):
            continue
        for f in os.listdir(task_path):
            fpath = os.path.join(task_path, f)
            if f.endswith("_Index_Final.tif"):
                task_name = f.replace("_Index_Final.tif", "")
                key = f"{task_name}_index"
                if key not in registry or not os.path.exists(str(registry.get(key, {}).get("file_path") or "")):
                    registry[key] = {
                        "task": task_name,
                        "method": "index",
                        "prob_threshold": None,
                        "min_count": None,
                        "file_path": fpath,
                        "created_at": datetime.datetime.fromtimestamp(
                            os.path.getmtime(fpath)).strftime("%Y-%m-%d %H:%M:%S"),
                        "file_size_mb": round(os.path.getsize(fpath) / (1024 ** 2), 2),
                    }
                    changed = True
                continue
            if not (f.endswith(".tif") and "_Final_" in f):
                if not (f.endswith(".shp") and "_Final_" in f):
                    continue
            if "_NUMERATOR" in f or "_DENOMINATOR" in f or f.endswith("_work.tif"):
                continue
            try:
                base = f.replace(".tif", "").replace(".shp", "")
                parts = base.split("_Final_")
                task_name = parts[0]
                param_str = parts[1]
                prob = float(param_str.split("_c")[0].replace("p", ""))
                cnt = int(param_str.split("_c")[1])
                key = f"{task_name}_p{prob:.2f}_c{cnt}"
                if key not in registry or not os.path.exists(str(registry.get(key, {}).get("file_path") or "")):
                    registry[key] = {
                        "task": task_name,
                        "prob_threshold": prob,
                        "min_count": cnt,
                        "file_path": fpath,
                        "created_at": datetime.datetime.fromtimestamp(
                            os.path.getmtime(fpath)).strftime("%Y-%m-%d %H:%M:%S"),
                        "file_size_mb": round(os.path.getsize(fpath) / (1024 ** 2), 2)
                    }
                    changed = True
            except Exception:
                continue
    if changed:
        save_asset_registry(registry)


def _zoom_fit_lonlat(left, bottom, right, top, viewport_px=960, margin=0.06):
    """
    按视口宽度估算 Leaflet/WebMercator 缩放级，使给定经纬度范围尽量占满地图（略留边距）。
    margin：相对四边各扩展的比例，避免贴边裁切。
    """
    lat_mid = (bottom + top) / 2.0
    lon_mid = (left + right) / 2.0
    cos_lat = max(abs(math.cos(math.radians(lat_mid))), 0.01)
    lon_span = max(abs(right - left), 1e-7)
    lat_span = max(abs(top - bottom), 1e-7)
    lon_span *= 1.0 + 2.0 * margin
    lat_span *= 1.0 + 2.0 * margin
    # 与 256px 瓦片、360° 经度范围对齐的常见拟合式（经向考虑纬度缩短）
    z_lon = math.log2(360.0 * viewport_px / (256.0 * lon_span * cos_lat))
    # 纬度方向 Web Mercator 约 ±85°，有效跨度按 ~170° 量级估算
    z_lat = math.log2(170.0 * viewport_px / (256.0 * lat_span))
    zoom = int(math.floor(min(z_lon, z_lat)))
    return lat_mid, lon_mid, max(5, min(19, zoom))


def _view_from_raster_path(path: str):
    """
    从 GeoTIFF 得到 WGS84 中心与缩放：优先用「有效像元」外接框（潮滩条带不会被整幅研究区拉大），
    否则退回文件 bounds。供 st_folium 与 add_raster 对齐视角。
    """
    try:
        import rasterio
        from rasterio.warp import transform_bounds
        from rasterio.windows import Window
        from rasterio.transform import array_bounds
        from rasterio.enums import Resampling
    except ImportError:
        return None
    if not path or not os.path.exists(path):
        return None
    try:
        with rasterio.open(path) as src:
            H, W = int(src.height), int(src.width)
            left, bottom, right, top = src.bounds.left, src.bounds.bottom, src.bounds.right, src.bounds.top
            crs = src.crs

            # 降采样读一屏，找非背景像元（掩膜/概率图常见：整幅外框很大、有效仅沿岸一线）
            if H < 2 or W < 2:
                arr = src.read(1)
                th, tw = int(arr.shape[0]), int(arr.shape[1])
            else:
                tw = min(512, W)
                th = min(512, H)
                arr = src.read(1, out_shape=(th, tw), resampling=Resampling.nearest)

            nd = src.nodata
            if arr.dtype.kind == "f":
                valid = np.isfinite(arr) & (arr > 1e-6)
            elif nd is not None:
                valid = arr != nd
            else:
                valid = arr != 0

            if np.any(valid):
                ys, xs = np.where(valid)
                r0, r1 = ys.min(), ys.max()
                c0, c1 = xs.min(), xs.max()
                # 缩略图坐标 → 全图像素窗口（略扩 1 格以免裁到边）
                sy = H / float(th)
                sx = W / float(tw)
                col0 = max(0, int(c0 * sx) - 1)
                row0 = max(0, int(r0 * sy) - 1)
                col1 = min(W, int((c1 + 1) * sx) + 1)
                row1 = min(H, int((r1 + 1) * sy) + 1)
                win = Window(col0, row0, col1 - col0, row1 - row0)
                aff = rasterio.windows.transform(win, src.transform)
                left, bottom, right, top = array_bounds(win.height, win.width, aff)

            if crs is not None:
                try:
                    left, bottom, right, top = transform_bounds(
                        crs, "EPSG:4326", left, bottom, right, top
                    )
                except Exception:
                    pass
    except Exception:
        return None

    return _zoom_fit_lonlat(left, bottom, right, top, viewport_px=960, margin=0.05)


def _view_from_vector_path(path: str):
    """从 Shapefile 得到 WGS84 中心与缩放，供 st_folium 对齐视角。"""
    try:
        import geopandas as gpd
    except ImportError:
        return None
    if not path or not os.path.exists(path):
        return None
    try:
        gdf = gpd.read_file(path)
        if gdf.empty or gdf.crs is None:
            return None
        gdf_wgs = gdf.to_crs(4326)
        minx, miny, maxx, maxy = gdf_wgs.total_bounds
        return _zoom_fit_lonlat(minx, miny, maxx, maxy, viewport_px=960, margin=0.05)
    except Exception:
        return None


def _view_from_asset_path(path: str):
    """按扩展名从栅格或矢量成果推断地图视角。"""
    if not path:
        return None
    ext = os.path.splitext(path)[1].lower()
    if ext == ".shp":
        return _view_from_vector_path(path)
    return _view_from_raster_path(path)


def _cached_view_for_asset_path(session_dict, path: str):
    """同一成果在 session 内按 mtime 缓存视角，减少反复打开大文件。"""
    if not path or not os.path.exists(path):
        return None
    abs_p = os.path.normpath(os.path.abspath(path))
    try:
        mt = os.path.getmtime(abs_p)
    except OSError:
        return _view_from_asset_path(abs_p)
    cache = session_dict.setdefault("_asset_view_cache", {})
    prev = cache.get(abs_p)
    if prev and prev[0] == mt:
        return prev[1]
    v = _view_from_asset_path(abs_p)
    if v is not None:
        cache[abs_p] = (mt, v)
        while len(cache) > 24:
            cache.pop(next(iter(cache)))
    return v


def _cached_view_for_raster_path(session_dict, path: str):
    """兼容旧调用：栅格或矢量成果均可。"""
    return _cached_view_for_asset_path(session_dict, path)


def _add_result_raster_to_map(m, path: str, layer_name: str, opacity: float = 0.5):
    """
    使用 leafmap.add_raster（localtileserver + folium）。
    「Reds」色图会把未标记为 nodata 的 0 值渲成白色；单波段成果若文件未写 nodata，默认按 0 作为透明背景。
    opacity：成果层整体透明度（0~1），由侧栏滑块控制。
    """
    norm = os.path.normpath(os.path.abspath(path))
    if not os.path.exists(norm):
        return False, f"文件不存在: {norm}"
    _nd_api = None
    _nb = 1
    try:
        import rasterio

        with rasterio.open(norm) as _ds:
            _nb = int(_ds.count)
            _nd_api = _nodata_safe_for_tile_api(_ds.nodata)
            # 单波段整型掩膜/概率图常见 0=背景；未声明 nodata 时 localtileserver 仍会对 0 上色 → 大块白底
            if _nb == 1 and _nd_api is None:
                dt = str(_ds.dtypes[0])
                if dt.startswith(("uint", "int")) or "float" in dt:
                    _nd_api = 0
    except Exception as e:
        return False, f"无法读取栅格元数据: {safe_error_summary(e)}"

    op = float(max(0.05, min(1.0, opacity)))
    kw = dict(
        layer_name=layer_name,
        colormap="Reds",
        opacity=op,
        client_args={"cors_all": True},
    )
    if _nb == 1:
        kw["indexes"] = 1
    if _nd_api is not None:
        kw["nodata"] = _nd_api
    try:
        with _local_tile_no_proxy():
            m.add_raster(norm, **kw)
        return True, None
    except Exception as e:
        return False, safe_error_summary(e)


def _add_result_vector_to_map(m, path: str, layer_name: str, opacity: float = 0.5):
    """在 Folium 地图上叠加潮滩 Shapefile 成果层。"""
    norm = os.path.normpath(os.path.abspath(path))
    if not os.path.exists(norm):
        return False, f"文件不存在: {norm}"
    try:
        import geopandas as gpd
        import folium
    except ImportError as e:
        return False, f"缺少 geopandas/folium: {safe_error_summary(e)}"

    try:
        gdf = gpd.read_file(norm)
        if gdf.empty:
            return False, "Shapefile 为空"
        if gdf.crs is not None:
            try:
                epsg = gdf.crs.to_epsg()
            except Exception:
                epsg = None
            if epsg != 4326:
                gdf = gdf.to_crs(4326)
        op = float(max(0.05, min(1.0, opacity)))
        folium.GeoJson(
            gdf,
            name=layer_name,
            style_function=lambda _x: {
                "fillColor": "#e41a1c",
                "color": "#b71c1c",
                "weight": 1,
                "fillOpacity": op,
            },
        ).add_to(m)
        return True, None
    except Exception as e:
        return False, safe_error_summary(e)


def _add_result_to_map(m, path: str, layer_name: str, opacity: float = 0.5):
    """按扩展名选择栅格或矢量方式加载潮滩成果。"""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".shp":
        return _add_result_vector_to_map(m, path, layer_name, opacity=opacity)
    return _add_result_raster_to_map(m, path, layer_name, opacity=opacity)


def _run_m5_phase(ctx, shared, current_shp, actual_task, prob, cnt, push_log, check_stop):
    """合成完成后执行时空异常检测（失败不阻断主流程）。"""
    if not ctx.get("m5_enabled", True):
        return None
    if check_stop():
        return None
    push_log(">>> [Phase 3]  潮滩变化分析…")
    try:
        import m5_engine
        import m5_agent_loop

        report = m5_engine.run_m5_after_synthesis(
            current_shp=current_shp,
            current_task=actual_task,
            final_root=ctx["final_root"],
            task_options=ctx.get("task_options"),
            prob=prob,
            cnt=cnt,
            baseline_shp_override=(ctx.get("m5_baseline_shp") or "").strip() or None,
            workspace_dir=ctx["final_root"],
            logger=push_log,
        )
        if report:
            verification = m5_agent_loop.verify_m5_outputs(
                report, workspace_dir=ctx["final_root"]
            )
            asset_id = None
            if verification.get("ok") is True:
                try:
                    asset_id = register_m5_asset(actual_task, report)
                    if not asset_id:
                        raise RuntimeError("资产登记返回空结果")
                    push_log(f"[M5] 已登记资产 {asset_id}")
                except Exception as reg_e:
                    reg_error = safe_error_summary(reg_e)
                    verification = dict(verification)
                    verification["ok"] = False
                    verification["checks"] = list(verification.get("checks") or []) + [{
                        "name": "asset_registration", "passed": False, "detail": reg_error,
                    }]
                    push_log(f"[M5] 后置资产登记失败（不阻断主流程）: {reg_error}")
            with shared["lock"]:
                shared["m5_report"] = report
                shared["m5_verification"] = verification
                shared["m5_asset_id"] = asset_id if verification.get("ok") is True else None
            lvl = report.get("alert_level", "GREEN")
            if verification.get("ok") is True:
                push_log(f"[M5] 变化分析完成，告警级别: {lvl}，输出已校验并登记")
            else:
                push_log(f"[M5] 变化分析完成但输出校验未完全通过，告警级别: {lvl}")
        else:
            push_log("[M5] 未生成变化告警（可能缺少往年同区域基线）。")
        return report
    except Exception as e:
        push_log(f"[M5] 变化分析异常: {safe_error_summary(e)}")
        return None


def run_m5_sync(ctx, shared, stop_event):
    """独立 M5 闭环：仅调用现有 M5 引擎，不跑推理/GEE。"""
    logs_local = []

    def check_stop():
        return stop_event.is_set()

    def push_log(msg):
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        line = f"[{ts}] root@m5: {msg}"
        logs_local.append(line)
        with shared["lock"]:
            shared["log_lines"] = logs_local[-40:]
        print(msg)

    def push_progress(pct):
        with shared["lock"]:
            shared["progress"] = int(min(100, max(0, pct)))

    def push_status(kind, text):
        with shared["lock"]:
            shared["status"] = (kind, text)

    push_progress(5)
    push_status("info", "潮滩变化分析启动…")
    m5_cfg = ctx.get("m5") or {}
    task = ctx.get("task") or m5_cfg.get("plan", {}).get("current_task")
    current_shp = m5_cfg.get("current_shp")
    baseline_shp = m5_cfg.get("baseline_shp")
    push_log(f"TASK: {task}")
    push_log(f"CURRENT: {current_shp}")
    push_log(f"BASELINE: {baseline_shp} ({m5_cfg.get('baseline_task') or '—'})")

    if check_stop():
        push_status("warning", "变化分析已中断")
        return False
    if not current_shp or not os.path.isfile(str(current_shp)):
        push_status("error", "当期潮滩 SHP 不存在")
        push_log(f"[ERROR] 当期 SHP 无效: {current_shp}")
        return False
    if not baseline_shp or not os.path.isfile(str(baseline_shp)):
        push_status("error", "基线潮滩 SHP 不存在")
        push_log(f"[ERROR] 基线 SHP 无效: {baseline_shp}")
        return False

    push_progress(30)
    try:
        import m5_engine
        import m5_agent_loop

        report = m5_engine.run_m5_after_synthesis(
            current_shp=current_shp,
            current_task=task,
            final_root=ctx["final_root"],
            task_options=ctx.get("task_options"),
            prob=ctx.get("prob"),
            cnt=ctx.get("cnt"),
            baseline_shp_override=baseline_shp,
            workspace_dir=ctx["final_root"],
            logger=push_log,
        )
        push_progress(80)
        if not report:
            push_status("warning", "变化分析未生成结果")
            return False
        report["baseline_task"] = report.get("baseline_task") or m5_cfg.get("baseline_task")
        verification = m5_agent_loop.verify_m5_outputs(report, workspace_dir=ctx["final_root"])
        map_path = verification.get("map_candidate") or m5_agent_loop.pick_m5_map_path(report)
        verified = verification.get("ok") is True
        if verified:
            try:
                asset_id = register_m5_asset(task, report)
                if not asset_id:
                    raise RuntimeError("资产登记返回空结果")
                push_log(f"[M5] 已登记资产 {asset_id}")
            except Exception as reg_e:
                reg_error = safe_error_summary(reg_e)
                verified = False
                verification = dict(verification)
                verification["ok"] = False
                verification["checks"] = list(verification.get("checks") or []) + [{
                    "name": "asset_registration", "passed": False, "detail": reg_error,
                }]
                push_log(f"[M5] 资产登记失败，任务不提交成功: {reg_error}")
        else:
            push_log("[M5] 输出校验未通过，未登记或加载未验证成果。")

        with shared["lock"]:
            shared["m5_report"] = report
            shared["m5_verification"] = verification
            shared["asset_path"] = map_path if verified and map_path and os.path.isfile(str(map_path)) else None
            shared["job_kind"] = "m5"

        if verified:
            push_status(
                "success",
                f"变化分析完成 · 告警 {report.get('alert_level', '—')}",
            )
        else:
            push_status("warning", "变化分析已完成但输出校验未完全通过")
        push_progress(100)
        push_log(m5_agent_loop.summarize_m5_report_for_chat(report, verification).replace("\n", " | "))
        return verified
    except Exception as e:
        safe = safe_error_summary(e)
        push_log(f"[ERROR] {safe}")
        push_status("error", f"变化分析异常: {safe}")
        import traceback

        traceback.print_exc()
        return False


def _m5_worker_entry(ctx, shared, stop_event):
    ok = False
    try:
        ok = run_m5_sync(ctx, shared, stop_event)
    except Exception as e:
        _record_worker_exception(shared, "M5 线程异常", e)
    finally:
        with shared["lock"]:
            shared["success"] = ok
            shared["done"] = True


def run_e1_sync(ctx, shared, stop_event):
    """独立 E1 闭环：仅调用 e1_engine，不跑推理/GEE。"""
    logs_local = []

    def check_stop():
        return stop_event.is_set()

    def push_log(msg):
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        line = f"[{ts}] root@e1: {msg}"
        logs_local.append(line)
        with shared["lock"]:
            shared["log_lines"] = logs_local[-40:]
        print(msg)

    def push_progress(pct):
        with shared["lock"]:
            shared["progress"] = int(min(100, max(0, pct)))

    def push_status(kind, text):
        with shared["lock"]:
            shared["status"] = (kind, text)

    push_progress(5)
    push_status("info", "潮滩精度评价启动…")
    e1_cfg = ctx.get("e1") or {}
    task = ctx.get("task") or e1_cfg.get("plan", {}).get("current_task")
    target_shp = e1_cfg.get("target_shp")
    push_log(f"TASK: {task}")
    push_log(f"TARGET: {target_shp}")
    push_log(f"REF: {e1_cfg.get('reference')} | DATA: {e1_cfg.get('data_root')}")

    if check_stop():
        push_status("warning", "精度评价已中断")
        return False
    if not target_shp or not os.path.isfile(str(target_shp)):
        push_status("error", "当期潮滩 SHP 不存在")
        push_log(f"[ERROR] 目标 SHP 无效: {target_shp}")
        return False

    push_progress(25)
    try:
        import e1_engine
        import e1_agent_loop

        roi_path = e1_engine.resolve_task_roi_path(
            e1_cfg.get("task_aoi_shp") or ctx.get("task_aoi_shp"),
            task,
            ctx["final_root"],
            logger=push_log,
        )
        workspace = e1_cfg.get("workspace_dir") or e1_engine.workspace_for_task(
            ctx["final_root"], task
        )
        report = e1_engine.run_e1_after_synthesis(
            target_shp=target_shp,
            roi_name=task,
            workspace_dir=workspace,
            data_root=e1_cfg.get("data_root") or e1_engine.DEFAULT_E1_DATA_ROOT,
            reference=e1_cfg.get("reference") or "师姐_2020",
            compare_sources=e1_cfg.get("compare_sources"),
            roi_path=roi_path,
            export_disagreement_maps=bool(e1_cfg.get("export_disagreement_maps", True)),
            export_multi_product_heatmap=bool(e1_cfg.get("export_multi_product_heatmap", True)),
            logger=push_log,
        )
        push_progress(80)
        if not report:
            push_status("warning", "精度评价未生成结果")
            return False
        verification = e1_agent_loop.verify_e1_outputs(report)
        map_path = verification.get("map_candidate") or e1_agent_loop.pick_e1_map_path(report)
        verified = verification.get("ok") is True
        if verified:
            try:
                asset_id = register_e1_asset(task, report)
                if not asset_id:
                    raise RuntimeError("资产登记返回空结果")
                push_log(f"[E1] 已登记资产 {asset_id}")
            except Exception as reg_e:
                reg_error = safe_error_summary(reg_e)
                verified = False
                verification = dict(verification)
                verification["ok"] = False
                verification["checks"] = list(verification.get("checks") or []) + [{
                    "name": "asset_registration", "passed": False, "detail": reg_error,
                }]
                push_log(f"[E1] 资产登记失败，任务不提交成功: {reg_error}")
        else:
            push_log("[E1] 输出校验未通过，未登记或加载未验证成果。")

        with shared["lock"]:
            shared["e1_report"] = report
            shared["e1_verification"] = verification
            shared["asset_path"] = map_path if verified and map_path and os.path.isfile(str(map_path)) else None
            shared["job_kind"] = "e1"

        n = len(report.get("comparisons") or {})
        if verified:
            push_status("success", f"精度评价完成 · {n} 组对比")
        else:
            push_status("warning", "精度评价已完成但输出校验未完全通过")
        push_progress(100)
        push_log(e1_agent_loop.summarize_e1_report_for_chat(report, verification).replace("\n", " | "))
        return verified
    except Exception as e:
        safe = safe_error_summary(e)
        push_log(f"[ERROR] {safe}")
        push_status("error", f"精度评价异常: {safe}")
        import traceback as _tb

        _tb.print_exc()
        return False


def _e1_worker_entry(ctx, shared, stop_event):
    ok = False
    try:
        ok = run_e1_sync(ctx, shared, stop_event)
    except Exception as e:
        _record_worker_exception(shared, "E1 线程异常", e)
    finally:
        with shared["lock"]:
            shared["success"] = ok
            shared["done"] = True


def _run_e1_phase(ctx, shared, current_shp, actual_task, push_log, check_stop):
    """合成完成后执行多源潮滩一致性诊断（失败不阻断主流程）。"""
    if not ctx.get("e1_enabled", False):
        return None
    if check_stop():
        return None
    push_log(">>> [Phase 4]  潮滩精度评价…")
    try:
        import e1_engine
        import e1_agent_loop

        roi_path = e1_engine.resolve_task_roi_path(
            ctx.get("task_aoi_shp"),
            actual_task,
            ctx["final_root"],
            logger=push_log,
        )
        workspace = e1_engine.workspace_for_task(ctx["final_root"], actual_task)
        compare_sources = ctx.get("e1_compare_sources") or None
        if compare_sources == []:
            compare_sources = None

        report = e1_engine.run_e1_after_synthesis(
            target_shp=current_shp,
            roi_name=actual_task,
            workspace_dir=workspace,
            data_root=ctx.get("e1_data_root") or e1_engine.DEFAULT_E1_DATA_ROOT,
            reference=ctx.get("e1_reference") or "师姐_2020",
            compare_sources=compare_sources,
            roi_path=roi_path,
            export_disagreement_maps=bool(ctx.get("e1_export_maps", True)),
            export_multi_product_heatmap=bool(ctx.get("e1_export_heatmap", True)),
            logger=push_log,
        )
        if report:
            verification = e1_agent_loop.verify_e1_outputs(report)
            asset_id = None
            if verification.get("ok") is True:
                try:
                    asset_id = register_e1_asset(actual_task, report)
                    if not asset_id:
                        raise RuntimeError("资产登记返回空结果")
                    push_log(f"[E1] 已登记资产 {asset_id}")
                except Exception as reg_e:
                    reg_error = safe_error_summary(reg_e)
                    verification = dict(verification)
                    verification["ok"] = False
                    verification["checks"] = list(verification.get("checks") or []) + [{
                        "name": "asset_registration", "passed": False, "detail": reg_error,
                    }]
                    push_log(f"[E1] 后置资产登记失败（不阻断主流程）: {reg_error}")
            with shared["lock"]:
                shared["e1_report"] = report
                shared["e1_verification"] = verification
                shared["e1_asset_id"] = asset_id if verification.get("ok") is True else None
            if verification.get("ok") is True:
                push_log(f"[E1] 精度评价完成，对比 {len(report.get('comparisons') or {})} 组产品，输出已校验并登记。")
            else:
                push_log(f"[E1] 精度评价完成但输出校验未完全通过，对比 {len(report.get('comparisons') or {})} 组产品。")
        else:
            push_log("[E1] 未生成精度评价结果。")
        return report
    except Exception as e:
        push_log(f"[E1] 精度评价异常: {safe_error_summary(e)}")
        return None


def run_pipeline_sync(ctx, shared, stop_event):
    """兼容旧入口的同步执行适配器。

    新的深度学习 Agent 入口应使用 ``inference_agent_loop``；此函数暂时保留以
    读取旧任务资产，不再作为新的 Agent/UI 请求构建器。
    """
    logs_local = []

    def check_stop():
        return stop_event.is_set()

    def push_log(msg):
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        line = f"[{ts}] root@cstf: {msg}"
        logs_local.append(line)
        with shared["lock"]:
            shared["log_lines"] = logs_local[-30:]
        print(msg)

    def push_progress(pct):
        v = int(min(100, max(0, pct)))
        with shared["lock"]:
            shared["progress"] = v

    def push_status(kind, text):
        with shared["lock"]:
            shared["status"] = (kind, text)

    push_progress(0)
    push_status("info", "初始化系统环境...")

    task = ctx["task"]
    prob = ctx["prob"]
    cnt = ctx["cnt"]
    root_dir = ctx["root_dir"]
    mask_root = ctx["mask_root"]
    final_root = ctx["final_root"]
    model_path = ctx["model_path"]
    shp_path = ctx["shp_path"]
    task_options = ctx["task_options"]

    actual_task = task
    for opt in task_options:
        if task in opt:
            actual_task = opt
            break

    if not actual_task or not root_dir:
        push_status("error", "❌ 未选择有效目标任务，或原始影像目录未配置。请在侧栏选择任务后再运行推理。")
        return False

    input_dir = os.path.join(root_dir, actual_task)
    mask_out_dir = os.path.join(mask_root, actual_task)
    final_out_dir = os.path.join(final_root, actual_task)
    current_final_shp = os.path.join(final_out_dir, f"{actual_task}_Final_p{prob:.2f}_c{cnt}.shp")

    cached = find_asset(actual_task, prob, cnt)
    if cached:
        push_log(f"⚡ 缓存命中: {os.path.basename(cached['file_path'])}")
        cached_shp = cached["file_path"]
        if cached_shp.lower().endswith(".tif"):
            _stem = os.path.splitext(cached_shp)[0]
            _alt = _stem + ".shp"
            if os.path.isfile(_alt):
                cached_shp = _alt
        _run_m5_phase(ctx, shared, cached_shp, actual_task, prob, cnt, push_log, check_stop)
        _run_e1_phase(ctx, shared, cached_shp, actual_task, push_log, check_stop)
        push_status("success", "⚡ 发现已有资产！直接加载，无需重新计算")
        push_progress(100)
        with shared["lock"]:
            shared["asset_path"] = cached["file_path"]
        return True

    push_log(f"INIT TASK: {actual_task} | PROB: {prob} | CNT: {cnt}")

    if not os.path.exists(input_dir):
        push_status("error", f"❌ 找不到原始影像输入目录：{input_dir}")
        return False

    os.makedirs(mask_out_dir, exist_ok=True)
    os.makedirs(final_out_dir, exist_ok=True)

    all_tifs = glob.glob(os.path.join(input_dir, "*.tif"))
    raw_tifs = [f for f in all_tifs if "_mask" not in f and "Final" not in f]
    total = len(raw_tifs)

    if total == 0:
        push_status("warning", "没有找到可以处理的 TIF 影像。")
        return False

    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        push_status("info", "正在载入深度学习模型...")
        model = pre_engine.load_model(model_path, device)
        push_progress(10)
    except Exception as e:
        push_status("error", f"模型加载失败: {safe_error_summary(e)}")
        return False

    push_log(">>> [Phase 1] 开始深度学习推理...")
    success_count = 0

    for idx, tif_path in enumerate(raw_tifs):
        if check_stop():
            push_log("[SYSTEM] 🚨 检测到中断信号，安全终止。")
            push_status("warning", "任务已被手动中止。")
            return False

        fname = os.path.basename(tif_path)
        save_name = fname.replace(".tif", "_mask.tif")
        save_path = os.path.join(mask_out_dir, save_name)

        push_status("info", f"[推理阶段] 处理中: {fname} ({idx + 1}/{total})")

        if os.path.exists(save_path) and os.path.getsize(save_path) > 0:
            success_count += 1
        else:
            try:
                res = pre_engine.process_geotiff(
                    model, tif_path, save_path, device,
                    current_idx=idx + 1, total_batch=total, stop_callback=check_stop
                )
                if res is False:
                    if check_stop():
                        push_log(f"  |-- [STOP] {fname}: 用户已请求中断推理。")
                    else:
                        push_log(
                            f"  |-- [FAIL] {fname}: 单景处理失败（非「中断跑图」；"
                            f"可能为 CUDA/显存/读写出错，见控制台 traceback）。"
                        )
                    return False
                success_count += 1
            except Exception as e:
                push_log(f"  |-- [FAIL] {fname}: {safe_error_summary(e)}")

        push_progress(int(10 + (success_count / total) * 70))

    if check_stop():
        return False

    push_log(">>> [Phase 2] 开始时空频次合成...")
    push_progress(80)
    push_status("info", "正在执行合成算法...")

    def bridge_logger(msg):
        push_log(msg)

    try:
        success = post_engine.generate_double_constraint_complete(
            source_folder=input_dir, mask_folder=mask_out_dir, output_path=current_final_shp,
            shp_path=shp_path, prob_threshold=prob, min_absolute_count=cnt,
            logger=bridge_logger, stop_callback=check_stop
        )
        if success and (
            not os.path.isfile(current_final_shp)
            or os.path.getsize(current_final_shp) <= 0
        ):
            push_log("[SYSTEM] 合成引擎返回成功，但最终成果文件缺失或为空；不登记为成功。")
            push_status("error", "合成结果校验失败：成果文件缺失或为空。")
            return False
        if success:
            register_asset(actual_task, prob, cnt, current_final_shp)
            _run_m5_phase(ctx, shared, current_final_shp, actual_task, prob, cnt, push_log, check_stop)
            _run_e1_phase(ctx, shared, current_final_shp, actual_task, push_log, check_stop)
            push_progress(100)
            push_status("success", "🎉 全流程完毕！结果已生成并注册到资产库。")
            with shared["lock"]:
                shared["asset_path"] = current_final_shp
            time.sleep(1.5)
            return True
        push_log("[SYSTEM] 🚨 合成阶段被强行终止。")
        return False
    except Exception as e:
        safe = safe_error_summary(e)
        push_log(f"[SYSTEM] 合成异常: {safe}")
        push_status("error", f"合成算法崩溃: {safe}")
        return False


def run_index_pipeline_sync(ctx, shared, stop_event):
    """指数法潮滩提取（统一委托 index_agent_loop，保留旧 UI 收尾语义）。"""
    import index_agent_loop

    logs_local = []

    def check_stop():
        return stop_event.is_set()

    def push_log(msg):
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        logs_local.append(f"[{ts}] root@cstf: {msg}")
        with shared["lock"]:
            shared["log_lines"] = logs_local[-30:]
        print(msg)

    def push_progress(pct):
        with shared["lock"]:
            shared["progress"] = int(min(100, max(0, pct)))

    def push_status(kind, text):
        with shared["lock"]:
            shared["status"] = (kind, text)

    task = ctx.get("task")
    actual_task = task
    for opt in ctx.get("task_options") or []:
        if task in opt:
            actual_task = opt
            break
    input_dir = os.path.join(ctx.get("root_dir") or "", actual_task or "")
    final_out_dir = os.path.join(ctx.get("final_root") or "", actual_task or "")
    points_shp = ctx.get("points_shp") or ""

    cached = find_index_asset(actual_task) if actual_task else None
    if cached and not ctx.get("force_rerun"):
        push_log(f"⚡ 指数法缓存命中: {os.path.basename(cached['file_path'])}")
        index_shp = os.path.join(final_out_dir, "Final_Intertidal_Flat.shp")
        if os.path.isfile(index_shp):
            _run_m5_phase(ctx, shared, index_shp, actual_task, None, None, push_log, check_stop)
            _run_e1_phase(ctx, shared, index_shp, actual_task, push_log, check_stop)
        push_status("success", "⚡ 发现已有指数法成果，直接加载")
        push_progress(100)
        with shared["lock"]:
            shared["asset_path"] = cached["file_path"]
        return True

    plan = ctx.get("index_plan")
    if not isinstance(plan, dict):
        plan = index_agent_loop.build_index_plan(
            task=actual_task or "", input_dir=input_dir,
            output_dir=final_out_dir, points_shp=points_shp,
            force_rerun=bool(ctx.get("force_rerun")),
        )
    result = index_agent_loop.execute_index_plan(
        plan, push_log=push_log, push_progress=push_progress,
        stop_callback=check_stop,
        register_asset=lambda path: register_index_asset(actual_task, path),
    )
    if result.get("success") is not True:
        push_status("warning" if result.get("status") == "CANCELLED" else "error", index_agent_loop.summarize_index_result(result))
        return False
    index_shp = os.path.join(final_out_dir, "Final_Intertidal_Flat.shp")
    if os.path.isfile(index_shp):
        _run_m5_phase(ctx, shared, index_shp, actual_task, None, None, push_log, check_stop)
        _run_e1_phase(ctx, shared, index_shp, actual_task, push_log, check_stop)
    push_status("success", index_agent_loop.summarize_index_result(result))
    with shared["lock"]:
        shared["asset_path"] = result.get("result_path")
    return True


def run_m4_download_sync(ctx, shared, stop_event):
    """M4 GEE 数据下载（Drive 提交或本地下载）。"""
    import m4_engine

    logs_local = []

    def check_stop():
        return stop_event.is_set()

    def push_log(msg):
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        logs_local.append(f"[{ts}] root@cstf: {msg}")
        with shared["lock"]:
            shared["log_lines"] = logs_local[-30:]
        print(msg)

    def push_progress(pct):
        with shared["lock"]:
            shared["progress"] = int(min(100, max(0, pct)))

    def push_status(kind, text):
        with shared["lock"]:
            shared["status"] = (kind, text)

    cfg = ctx.get("m4") or {}
    push_progress(0)
    push_status("info", "正在连接 Google Earth Engine…")

    try:
        result = m4_engine.run_m4_download(
            roi_path=cfg["roi_path"],
            roi_name=cfg["roi_name"],
            start_date=cfg["start_date"],
            end_date=cfg["end_date"],
            export_to=cfg["export_to"],
            local_out_dir=cfg["local_out_dir"],
            bands=cfg.get("bands"),
            cloud_limit=int(cfg.get("cloud_limit", 60)),
            min_land_pct=float(cfg.get("min_land_pct", 5.0)),
            max_land_pct=float(cfg.get("max_land_pct", 95.0)),
            min_pixel_count=int(cfg.get("min_pixel_count", 1000)),
            drive_folder=cfg.get("drive_folder", "GEE_Downloads"),
            scale=int(cfg.get("scale", 10)),
            gee_proxy_url=(cfg.get("gee_proxy_url") or "").strip() or None,
            gee_project_id=(cfg.get("gee_project_id") or "").strip() or None,
            push_log=push_log,
            push_progress=push_progress,
            stop_callback=check_stop,
        )
        if check_stop():
            push_status("warning", "影像获取已中止。")
            return False
        if not result:
            return False
        with shared["lock"]:
            shared["m4_result"] = result
        n = result["image_count"]
        if result["export_to"] == "drive":
            push_status(
                "success",
                f"已提交 {n} 个 Drive 任务 → 文件夹「{result['drive_folder']}」。请在影像平台任务列表 / 云盘查看。",
            )
        else:
            push_status("success", f"本地下载完成 {n} 景 → {result['local_out_dir']}")
        push_progress(100)
        return True
    except Exception as e:
        safe = safe_error_summary(e)
        push_log(f"[SYSTEM] M4 异常: {safe}")
        push_status("error", f"影像获取失败: {safe}")
        return False


def _pipeline_worker_entry(ctx, shared, stop_event):
    ok = False
    try:
        mode = ctx.get("mode", "dl")
        if mode == "m4":
            # Historical M4 payloads are normalized into the trusted GEE
            # adapter.  Do not send them through the old synchronous path,
            # which had no shared verify/register gate.
            import gee_agent_loop as _gee_loop

            legacy_plan = _gee_loop.build_legacy_m4_plan(
                ctx.get("m4") or {}, task_id=str(ctx.get("task") or "")
            )
            if not legacy_plan.get("ready"):
                reason = "；".join(legacy_plan.get("blockers") or ["旧 M4 参数无法转换为 GEE 计划"])
                with shared["lock"]:
                    shared["status"] = ("error", reason)
                    shared["gee_result"] = {
                        "success": False,
                        "task_id": legacy_plan.get("task_id"),
                        "plan_id": legacy_plan.get("plan_id"),
                        "error": reason,
                    }
                ok = False
            else:
                _gee_worker_entry(
                    {"task": legacy_plan.get("task_id"), "gee_plan": legacy_plan},
                    shared,
                    stop_event,
                )
                with shared["lock"]:
                    ok = bool(shared.get("success"))
        elif mode == "index":
            ok = run_index_pipeline_sync(ctx, shared, stop_event)
        elif mode == "legacy_dl":
            # 仅为明确标记的历史资产提供兼容入口；新请求不得静默回退。
            ok = run_pipeline_sync(ctx, shared, stop_event)
        else:
            with shared["lock"]:
                shared["status"] = ("error", f"未知执行模式：{mode or '空'}，未启动旧兼容入口。")
    except Exception as e:
        _record_worker_exception(shared, "后台任务线程异常", e)
        ok = False
    finally:
        with shared["lock"]:
            shared["success"] = ok
            shared["done"] = True


def _workflow_worker_entry(ctx, shared, stop_event):
    """端到端潮滩分析 Workflow 后台线程：只调用 workflow_orchestrator（复用子闭环）。

    不调用 Streamlit API；只写 shared / 文件。任何一步失败不伪造成功。
    """
    import time as _time

    ok = False
    try:
        import workflow_orchestrator as _wo

        wf = ctx.get("workflow_plan")
        if not isinstance(wf, dict) or not wf.get("workflow_id"):
            with shared["lock"]:
                shared["status"] = ("error", "一键潮滩分析计划未就绪，无法执行。")
            return

        def push_log(msg):
            ts = datetime.datetime.now().strftime("%H:%M:%S")
            with shared["lock"]:
                lines = list(shared.get("log_lines") or [])
                lines.append(f"[{ts}] root@workflow: {msg}")
                shared["log_lines"] = lines[-40:]
            print(msg)

        def push_progress(pct):
            with shared["lock"]:
                shared["progress"] = int(min(100, max(0, pct)))

        def push_status(kind, text):
            with shared["lock"]:
                shared["status"] = (kind, text)

        push_progress(2)
        push_status("info", "潮滩分析 Workflow 启动…")
        push_log(f"WORKFLOW: {wf.get('workflow_id')} | TASK: {wf.get('task_id')}")

        exec_ctx = {
            "workflow_id": wf.get("workflow_id"),
            "aoi": ctx.get("aoi"),
            "root_dir": ctx.get("root_dir"),
            "final_root": ctx.get("final_root"),
            "mask_root": ctx.get("mask_root"),
            "model_path": ctx.get("model_path"),
            "shp_path": ctx.get("shp_path"),
            "e1_data_root": ctx.get("e1_data_root"),
            "e1_reference": ctx.get("e1_reference"),
            "registry": ctx.get("registry"),
            "registry_path": ctx.get("registry_path"),
            "report_output_dir": ctx.get("report_output_dir"),
            "baseline_task": ctx.get("baseline_task"),
            "push_progress": push_progress,
        }
        result = _wo.run_analysis_workflow(
            wf, exec_ctx=exec_ctx, push_log=push_log, stop_event=stop_event,
        )
        with shared["lock"]:
            shared["workflow_result"] = result
        final_status = result.get("status")
        ok = final_status in ("SUCCEEDED", "COMPLETED_WITH_WARNINGS")
        summary = result.get("summary") or ""
        if ok:
            push_status(
                "success" if final_status == "SUCCEEDED" else "warning",
                f"一键潮滩分析完成 · {uil.get_status_label(final_status)}",
            )
        else:
            push_status("error", f"一键潮滩分析未完成 · {uil.get_status_label(final_status)}")
        push_log(summary.replace("\n", " | "))
        push_progress(100)
        return
    except Exception as e:
        _record_worker_exception(shared, "Workflow 线程异常", e)
        ok = False
    finally:
        with shared["lock"]:
            shared["success"] = ok
            shared["done"] = True


def _inference_worker_entry(ctx, shared, stop_event):
    """本地潮滩推理可信执行闭环后台线程（只写 shared / 文件，不调用 Streamlit API）。

    顺序：真实推理 → 真实后处理 → 磁盘校验 → 验证通过才登记资产。
    任何一步失败：不登记、不伪报完成；shared['inference_result'] 保留真实失败信息。
    """
    import time as _time

    ok = False
    try:
        import inference_agent_loop as ial

        plan = ctx.get("inference_plan")
        if not isinstance(plan, dict) or not plan.get("ready"):
            with shared["lock"]:
                shared["status"] = ("error", "推理计划未就绪，无法执行。")
            return

        task_id = plan.get("task_id") or ctx.get("task") or "unknown"

        def check_stop():
            return stop_event.is_set()

        def push_log(msg):
            ts = datetime.datetime.now().strftime("%H:%M:%S")
            with shared["lock"]:
                lines = list(shared.get("log_lines") or [])
                lines.append(f"[{ts}] root@cstf: {msg}")
                shared["log_lines"] = lines[-30:]
            print(msg)

        def push_progress(pct):
            with shared["lock"]:
                shared["progress"] = int(min(100, max(0, pct)))

        def push_status(kind, text):
            with shared["lock"]:
                shared["status"] = (kind, text)

        started = _time.time()
        push_status("info", "正在启动潮滩智能提取…")
        push_log(f"PLAN: {plan.get('plan_id')} | TASK: {task_id} | "
                 f"P={plan.get('prob_threshold')} C={plan.get('count_threshold')} | "
                 f"DEVICE={plan.get('device') or plan.get('device_policy')}")

        recovery = {"action": "INTERRUPTED_WAIT_CONFIRMATION"}
        result = None
        verification = None
        if not plan.get("force_rerun"):
            recovery = ial.classify_inference_recovery(plan)
            if recovery.get("action") == "COMPLETE":
                verification = recovery.get("verification") or {}
                result = {
                    "success": True,
                    "task_id": task_id,
                    "plan_id": plan.get("plan_id"),
                    "status": "reused",
                    "outputs": {
                        "final_tif": verification.get("final_tif"),
                        "final_shp": verification.get("final_shp"),
                    },
                    "metrics": {"reused_checkpoint": True},
                    "warnings": ["复用已通过验证的 Final 成果，未重复执行推理。"],
                    "error": None,
                }
                push_log("♻️ 检测到已验证 Final 成果，跳过重复推理。")
        if result is None:
            result = ial.execute_local_inference(
                plan,
                stop_event=stop_event,
                push_log=push_log,
                push_progress=push_progress,
            )
        if not result or result.get("success") is not True:
            err = sanitize_external_text((result or {}).get("error") or "提取失败")[:240]
            push_status("error", f"❌ {err}")
            with shared["lock"]:
                shared["inference_result"] = result or {}
            return

        push_status("info", "提取完成，正在校验磁盘成果…")
        verification = verification or ial.verify_inference_outputs(plan, result, started_at=started)
        if not verification or verification.get("ok") is not True:
            failed = [c.get("name") for c in (verification or {}).get("checks") or []
                      if not c.get("passed")]
            push_status("error", f"❌ 成果校验未通过: {', '.join(failed) or '未知'}")
            with shared["lock"]:
                shared["inference_result"] = result
                shared["inference_verification"] = verification or {}
            return

        asset_id = ial.register_inference_asset(plan, result, verification)
        if not asset_id:
            push_status("error", "❌ 校验通过但资产登记失败（未登记成果）。")
            with shared["lock"]:
                shared["inference_result"] = result
                shared["inference_verification"] = verification
            return

        final_tif = (result.get("outputs") or {}).get("final_tif") or ""
        final_shp = (result.get("outputs") or {}).get("final_shp") or ""
        push_log(f"✅ 提取闭环完成 | asset_id={asset_id} | Final TIF={os.path.basename(str(final_tif))} | "
                 f"Final SHP={os.path.basename(str(final_shp))}")
        push_status("success", "🎉 潮滩智能提取完成：成果已验证并登记。")
        with shared["lock"]:
            shared["inference_result"] = result
            shared["inference_verification"] = verification
            shared["asset_id"] = asset_id
            # 绝对路径（供地图加载与资产登记使用）
            _abs_map = os.path.abspath(str(final_shp or final_tif or ""))
            shared["asset_path"] = _abs_map if os.path.isfile(_abs_map) else None
            shared["progress"] = 100
        ok = True
    except Exception as e:
        _record_worker_exception(shared, "推理线程异常", e)
        ok = False
    finally:
        with shared["lock"]:
            shared["success"] = ok
            shared["done"] = True


def _gee_worker_entry(ctx, shared, stop_event):
    """GEE 影像下载可信执行闭环后台线程（B 阶段）。

    顺序：真实 m4_engine 下载 → 磁盘/远程校验 → 验证通过才登记 dataset asset。
    任何一步失败：不登记、不伪报完成；shared['gee_result'] 保留真实失败信息。
    """
    import time as _time

    ok = False
    try:
        import gee_agent_loop as gal

        plan = ctx.get("gee_plan")
        if not isinstance(plan, dict) or not plan.get("ready"):
            with shared["lock"]:
                shared["status"] = ("error", "GEE 下载计划未就绪，无法执行。")
            return

        task_id = plan.get("task_id") or ctx.get("task") or "unknown"

        def check_stop():
            return stop_event.is_set()

        def push_log(msg):
            ts = datetime.datetime.now().strftime("%H:%M:%S")
            with shared["lock"]:
                lines = list(shared.get("log_lines") or [])
                lines.append(f"[{ts}] root@cstf: {msg}")
                shared["log_lines"] = lines[-30:]
            print(msg)

        def push_progress(pct):
            with shared["lock"]:
                shared["progress"] = int(min(100, max(0, pct)))

        def push_status(kind, text):
            with shared["lock"]:
                shared["status"] = (kind, text)

        started = _time.time()
        push_status("info", "正在执行影像获取（可信执行闭环）…")
        push_log(f"PLAN: {plan.get('plan_id')} | TASK: {task_id} | "
                 f"BANDS={plan.get('bands')} | EXPORT={plan.get('export_to')} | "
                 f"COLLECTION={plan.get('collection')}")

        result = gal.execute_gee_download(
            plan,
            stop_event=stop_event,
            push_log=push_log,
            push_progress=push_progress,
        )
        if not result or result.get("success") is not True:
            err = sanitize_external_text((result or {}).get("error") or "影像获取失败")[:240]
            push_status("error", f"❌ {err}")
            with shared["lock"]:
                shared["gee_result"] = result or {}
            return

        push_status("info", "下载结束，正在校验成果…")
        verification = gal.verify_gee_outputs(plan, result, started_at=started)
        if not verification or verification.get("ok") is not True:
            failed = [c.get("name") for c in (verification or {}).get("checks") or []
                      if not c.get("passed")]
            push_status("error", f"❌ 成果校验未通过: {', '.join(failed) or '未知'}")
            with shared["lock"]:
                shared["gee_result"] = result
                shared["gee_verification"] = verification or {}
            return

        asset_id = gal.register_gee_dataset_asset(plan, result, verification)
        if not asset_id:
            push_status("error", "❌ 校验通过但资产登记失败（未登记数据集）。")
            with shared["lock"]:
                shared["gee_result"] = result
                shared["gee_verification"] = verification
            return

        n_tifs = len(verification.get("local_tifs") or [])
        push_log(f"✅ 影像获取闭环完成 | dataset_id={asset_id} | "
                 f"scene_count={result.get('metrics', {}).get('scene_count')} | "
                 f"local_tifs={n_tifs}")
        push_status("success", "🎉 影像获取完成：影像数据已验证并登记。提取不会自动启动。")
        with shared["lock"]:
            shared["gee_result"] = result
            shared["gee_verification"] = verification
            shared["dataset_id"] = asset_id
            shared["progress"] = 100
        ok = True
    except Exception as e:
        _record_worker_exception(shared, "GEE 下载线程异常", e)
        ok = False
    finally:
        with shared["lock"]:
            shared["success"] = ok
            shared["done"] = True


# =======================================================
#  1. 页面全局配置与状态机初始化 (Session State)
# =======================================================
st.set_page_config(
    page_title="CSTF-Cloud | 遥感智能监测平台",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 🌟 初始化系统状态：控制运行/中断的红绿灯
if "is_running" not in st.session_state:
    st.session_state.is_running = False
if "stop_requested" not in st.session_state:
    st.session_state.stop_requested = False
if "pending_task" not in st.session_state:
    st.session_state.pending_task = None
if "asset_override" not in st.session_state:
    st.session_state.asset_override = None
if "assets_scanned" not in st.session_state:
    st.session_state.assets_scanned = False
if "_param_key" not in st.session_state:
    st.session_state._param_key = None
if "asset_just_loaded" not in st.session_state:
    st.session_state.asset_just_loaded = False
if "executing_pipeline" not in st.session_state:
    st.session_state.executing_pipeline = False
if "pipeline_log_snapshot" not in st.session_state:
    st.session_state.pipeline_log_snapshot = []
if "pipeline_progress_value" not in st.session_state:
    st.session_state.pipeline_progress_value = 0
if "pipeline_thread_started" not in st.session_state:
    st.session_state.pipeline_thread_started = False
if "agent_chat_width_pct" not in st.session_state:
    st.session_state.agent_chat_width_pct = 34
if "agent_dock_view" not in st.session_state:
    st.session_state.agent_dock_view = "对话"
if "agent_status_panel_height" not in st.session_state:
    st.session_state.agent_status_panel_height = 220
if "agent_status_panel_collapsed" not in st.session_state:
    st.session_state.agent_status_panel_collapsed = False
# 拖拽尺寸通过 URL 参数跨越 Streamlit rerun 保留；参数只包含 UI 尺寸，不含业务数据。
try:
    _query_agent_width = st.query_params.get("cstf_agent_w")
    if _query_agent_width is not None:
        st.session_state.agent_chat_width_pct = int(float(_query_agent_width))
except (TypeError, ValueError, AttributeError):
    pass
try:
    _query_status_height = st.query_params.get("cstf_status_h")
    if _query_status_height is not None:
        st.session_state.agent_status_panel_height = int(float(_query_status_height))
except (TypeError, ValueError, AttributeError):
    pass

# 🌟 初始化地图状态：控制视角飞跃
if "map_center" not in st.session_state:
    st.session_state.map_center = [35.0, 105.0]
if "_map_channel_id" not in st.session_state:
    st.session_state._map_channel_id = f"map_{uuid.uuid4().hex}"
if "map_zoom" not in st.session_state:
    st.session_state.map_zoom = 3
if "_map_view_synced_for" not in st.session_state:
    st.session_state._map_view_synced_for = None
if "result_overlay_opacity_pct" not in st.session_state:
    st.session_state.result_overlay_opacity_pct = 50
if "globe_show_e1" not in st.session_state:
    st.session_state.globe_show_e1 = True
if "use_2d_map_fallback" not in st.session_state:
    st.session_state.use_2d_map_fallback = False
if "_globe_tile_clients" not in st.session_state:
    st.session_state._globe_tile_clients = {}
if "_asset_pinned" not in st.session_state:
    st.session_state._asset_pinned = False
if "_globe_rev" not in st.session_state:
    st.session_state._globe_rev = 0
if "_globe_iframe_cache_sig" not in st.session_state:
    st.session_state._globe_iframe_cache_sig = None
if "_globe_iframe_url" not in st.session_state:
    st.session_state._globe_iframe_url = None
if "_globe_warn_token" not in st.session_state:
    st.session_state._globe_warn_token = None
if "m5_report" not in st.session_state:
    st.session_state.m5_report = None
if "e1_report" not in st.session_state:
    st.session_state.e1_report = None

from agent_command_bridge import (
    init_ui_session_defaults,
    process_agent_reply,
    build_agent_sidebar_context,
    flush_pending_agent_commands,
    queue_agent_command,
    _aoi_state_to_dict,
)

import map_protocol as _map_proto


# ---- Phase D: 地图 AOI 双向交互（Cesium 绘制 → server → 校验回声 → Copilot 上下文）----
def _send_globe_message(payload):
    """向已加载的 Cesium iframe 发送任意 CSTF_MAP_V1 消息（同 FLY 的 targetOrigin 收紧逻辑）。"""
    import json as _json

    _msg_js = _json.dumps(payload, ensure_ascii=False)
    try:
        components.html(
            f"""
<script>
(() => {{
  const win = window.parent || window;
  const doc = win.document;
  const msg = {_msg_js};
  let origin = "*";
  try {{
    const iframes = doc.querySelectorAll("iframe");
    iframes.forEach((ifr) => {{
      const src = ifr.getAttribute("src") || "";
      if (src.indexOf("/globe") >= 0 || src.indexOf(":8765") >= 0) {{
        try {{ origin = new URL(src, win.location.href).origin; }} catch (e) {{}}
      }}
    }});
  }} catch (e) {{}}
  const send = () => {{
    const iframes = doc.querySelectorAll("iframe");
    let sent = false;
    iframes.forEach((ifr) => {{
      const src = ifr.getAttribute("src") || "";
      if (!src) return;
      if (src.indexOf("/globe") >= 0 || src.indexOf(":8765") >= 0) {{
        try {{
          ifr.contentWindow.postMessage(msg, origin);
          sent = true;
        }} catch (e) {{}}
      }}
    }});
    return sent;
  }};
  if (!send()) {{
    let n = 0;
    const t = setInterval(() => {{
      if (send() || ++n > 40) clearInterval(t);
    }}, 120);
  }}
}})();
</script>
            """,
            height=0,
        )
    except Exception:
        pass


def _poll_aoi_messages():
    """消费 Cesium iframe 的 AOI 消息：校验 → 回声图层 → 注入 Copilot 上下文。"""
    try:
        import aoi_context as _aoi_ctx
        import aoi_map_bridge as _aoi_bridge
    except Exception:
        return
    try:
        import globe_server as _gsrv

        _since = int(st.session_state.get("_aoi_poll_seq") or 0)
        _res = _gsrv.take_aoi_pending(
            _since,
            channel_id=st.session_state.get("_map_channel_id"),
        )
    except Exception:
        return
    if _res.get("last_seq") is not None:
        st.session_state["_aoi_poll_seq"] = int(_res.get("last_seq") or _since)
    for _m in _res.get("messages") or []:
        _kind = _m.get("kind") or "selected"
        try:
            if _kind == "cleared":
                _r = _aoi_bridge.process_aoi_cleared(st.session_state)
            else:
                _r = _aoi_bridge.process_aoi_selected(
                    st.session_state,
                    geometry=_m.get("geometry"),
                    source=_m.get("source") or "map_polygon",
                    label=_m.get("label"),
                )
        except Exception as _ae:
            _r = {"ok": False, "errors": [str(_ae)], "echo": None}
        _echo = _r.get("echo")
        if isinstance(_echo, list):
            for _e in _echo:
                if isinstance(_e, dict):
                    _send_globe_message(_e)
        elif isinstance(_echo, dict):
            _send_globe_message(_echo)
        if not _r.get("ok"):
            _aoi_errors = [sanitize_external_text(e)[:240] for e in (_r.get("errors") or [])]
            st.warning("研究区域无效：" + "; ".join(_aoi_errors))


def _aoi_sidebar_context():
    """AOI 摘要（供 Agent System Prompt）：仅包含紧凑摘要 + 推荐，不含 GeoJSON。"""
    try:
        import aoi_context as _aoi_ctx
        import aoi_map_bridge as _aoi_bridge
    except Exception:
        return ""
    _aoi = st.session_state.get("_active_aoi")
    if not _aoi:
        return ""
    try:
        _cap = st.session_state.get("_capability_reg")
        _caps = {}
        if _cap is not None:
            _snap = _cap.snapshot_for_agent()
            _caps = {cid: v.get("status") for cid, v in _snap.items()}
        return _aoi_bridge.aoi_recommendation_text(
            _aoi,
            _caps,
            include_spatial=bool(st.session_state.get("agent_spatial_consent", False)),
        )
    except Exception:
        return ""


# ---- Phase C: 统一任务执行时间线（惰性单例 + 原子账本）----
def _get_task_timeline():
    tl = st.session_state.get("_task_timeline")
    if tl is None:
        import task_timeline as _tt

        _ledger = _tt.timeline_ledger_path()
        tl = _tt.TimelineStore(ledger_path=_ledger)
        try:
            tl.load()
        except Exception:
            pass
        st.session_state._task_timeline = tl
    return tl


def _tl_add(task_id, phase, message, *, status="PENDING", plan_id=None, tool=None,
            progress=None, details=None, artifacts=None, error=None):
    """记录时间线事件并原子落盘（失败静默，不阻塞主流程）。"""
    try:
        tl = _get_task_timeline()
        ev = tl.add(
            task_id, phase, message, status=status, plan_id=plan_id,
            tool=tool, progress=progress, details=details or {},
            artifacts=artifacts or [], error=error,
        )
        try:
            tl.save()
        except Exception:
            pass
        return ev
    except Exception:
        return None


def _tl_update(event_id, *, status=None, progress=None, message=None, error=None):
    try:
        tl = _get_task_timeline()
        _ok, ev = tl.update(
            event_id, status=status, progress=progress, message=message, error=error
        )
        try:
            tl.save()
        except Exception:
            pass
        return ev
    except Exception:
        return None


# AGENT-011：后台线程仍保留，但任务元数据先写入跨 rerun/进程账本。
def _get_job_store():
    store = st.session_state.get("_job_store")
    if store is None:
        from job_store import JobStore

        path = os.environ.get("CSTF_JOB_DB_PATH") or os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "data", "jobs.sqlite3"
        )
        store = JobStore(path)
        if not st.session_state.get("_job_store_reconciled"):
            recovered = store.reconcile()
            if recovered:
                st.session_state["_job_recovery_records"] = recovered
                st.session_state["_job_recovery_notice"] = (
                    f"检测到 {len(recovered)} 个进程中断任务，已标记为 INTERRUPTED；请确认后重新执行。"
                )
            st.session_state["_job_store_reconciled"] = True
        st.session_state._job_store = store
    return store


def _job_create_for_pending(pending, *, status="QUEUED"):
    """为 UI/Agent 共用 execution_request 建立唯一 job_id。"""
    try:
        request = (pending or {}).get("execution_request") or {}
        store = _get_job_store()
        record = store.create(
            job_id=request.get("request_id"),
            task=(pending or {}).get("task") or "unknown",
            kind=request.get("mode") or (pending or {}).get("mode") or "workflow",
            plan_id=request.get("plan_id") or (pending or {}).get("plan_id"),
            status=status,
            metadata={
                "confirmation_source": request.get("confirmation_source"),
                "request_id": request.get("request_id"),
                "request_schema": request.get("schema"),
                "entrypoint": request.get("entrypoint"),
            },
        )
        st.session_state["_active_job_id"] = record.job_id
        return record
    except Exception as exc:
        _append_debug_log(f"job_create_failed: {safe_error_summary(exc)}")
        return None


def _job_transition(status, *, progress=None, artifacts=None, error=None, metadata=None):
    try:
        job_id = st.session_state.get("_active_job_id")
        if not job_id:
            return None
        return _get_job_store().transition(
            job_id, status, progress=progress, artifacts=artifacts,
            error=error, metadata=metadata,
        )
    except Exception as exc:
        _append_debug_log(f"job_transition_failed: {safe_error_summary(exc)}")
        return None


def _job_progress_update(progress, *, metadata=None, job_id=None):
    """把内存 worker 的最新百分比镜像到 JobStore，供 rerun/重启后读取。"""
    try:
        jid = job_id or st.session_state.get("_active_job_id")
        if not jid:
            return None
        return _get_job_store().update_progress(jid, progress, metadata=metadata)
    except Exception as exc:
        _append_debug_log(f"job_progress_update_failed: {safe_error_summary(exc)}")
        return None


init_ui_session_defaults(st.session_state)
try:
    _get_job_store()
    if st.session_state.get("_job_recovery_notice"):
        st.warning(st.session_state.pop("_job_recovery_notice"))
except Exception as _job_init_err:
    _append_debug_log(f"job_store_init_failed: {safe_error_summary(_job_init_err)}")
_agent_flush = flush_pending_agent_commands(st.session_state)
if st.session_state.get("_job_recovery_replan_notice"):
    st.info(st.session_state.pop("_job_recovery_replan_notice"))
if _agent_flush.applied and _agent_flush.errors:
    for _afe in _agent_flush.errors:
        st.warning(_afe)
if _agent_flush.applied and _agent_flush.m5_plan_text:
    st.session_state._m5_plan_notice = _agent_flush.m5_plan_text
    # 将可验证计划写入对话，便于用户确认
    _msgs = list(st.session_state.get("messages") or [])
    _last = (_msgs[-1].get("content") if _msgs else "") or ""
    if "潮滩变化分析 · 执行计划" not in str(_last):
        _msgs.append({"role": "assistant", "content": _agent_flush.m5_plan_text})
        st.session_state.messages = _msgs
if _agent_flush.applied and _agent_flush.action_type == "run_m5":
    try:
        st.toast("潮滩变化分析已确认，正在执行…", icon="🛰️")
    except Exception:
        pass
if _agent_flush.applied and _agent_flush.e1_plan_text:
    st.session_state._e1_plan_notice = _agent_flush.e1_plan_text
    _msgs_e1 = list(st.session_state.get("messages") or [])
    _last_e1 = (_msgs_e1[-1].get("content") if _msgs_e1 else "") or ""
    if "潮滩精度评价 · 执行计划" not in str(_last_e1):
        _msgs_e1.append({"role": "assistant", "content": _agent_flush.e1_plan_text})
        st.session_state.messages = _msgs_e1
if _agent_flush.applied and _agent_flush.action_type == "run_e1":
    try:
        st.toast("潮滩精度评价已确认，正在执行…", icon="📊")
    except Exception:
        pass
if _agent_flush.applied and _agent_flush.inference_plan_text:
    st.session_state._inference_plan_notice = _agent_flush.inference_plan_text
    _msgs_inf = list(st.session_state.get("messages") or [])
    _last_inf = (_msgs_inf[-1].get("content") if _msgs_inf else "") or ""
    if "潮滩智能提取 · 执行计划" not in str(_last_inf):
        _msgs_inf.append({"role": "assistant", "content": _agent_flush.inference_plan_text})
        st.session_state.messages = _msgs_inf
if _agent_flush.applied and _agent_flush.action_type == "run_inference":
    try:
        st.toast("潮滩智能提取已确认，正在执行…", icon="🌊")
    except Exception:
        pass
if _agent_flush.applied and _agent_flush.gee_plan_text:
    st.session_state._gee_plan_notice = _agent_flush.gee_plan_text
    _msgs_gee = list(st.session_state.get("messages") or [])
    _last_gee = (_msgs_gee[-1].get("content") if _msgs_gee else "") or ""
    if "获取卫星影像 · 执行计划" not in str(_last_gee):
        _msgs_gee.append({"role": "assistant", "content": _agent_flush.gee_plan_text})
        st.session_state.messages = _msgs_gee
if _agent_flush.applied and _agent_flush.action_type == "run_gee_download":
    try:
        st.toast("获取卫星影像已确认，正在执行…", icon="🛰️")
    except Exception:
        pass

# =======================================================
#  🌟 AIE 风格 CSS 深度定制
# =======================================================
st.markdown("""
<style>
    /* Keep the browser and Streamlit root surfaces dark during viewport
       reflow.  Without this, rapid window resizing exposes the default white
       body/.stApp background between the fixed workbench layers. */
    html, body, .stApp, [data-testid="stApp"] {
        background-color: #0e0e0e !important;
    }
    [data-testid="stAppViewContainer"] { background-color: #0e0e0e !important; }
    [data-testid="stHeader"] { background-color: rgba(14, 14, 14, 0); } 
    h1, h2, h3, p, span, div { color: #cccccc !important; }
    [data-testid="stSidebar"] { background-color: #1b1b1d !important; border-right: 1px solid #333333; }
    [data-testid="stSidebar"] * { color: #cccccc !important; }
    /* Streamlit 1.62 adds a white background/border to this wrapper around
       every text input.  During a viewport reflow it becomes a visible white
       flash around the dark field, so keep the wrapper itself transparent. */
    [data-testid="stTextInputRootElement"] {
        background-color: transparent !important;
        border: none !important;
    }
    /* Keep the status rail dark across Streamlit rerenders. */
    [data-testid="stProgress"] [role="progressbar"] > div:first-child {
        background-color: #2a2d3b !important;
        border-radius: 999px !important;
    }
    [data-testid="stProgress"] [role="progressbar"] > div:first-child > div {
        background-color: #3a62d7 !important;
        border-radius: 999px !important;
    }
    .react-aria-ComboBox {
        background-color: transparent !important;
        border: none !important;
    }
    .react-aria-ComboBox > [role="group"] {
        background-color: transparent !important;
        border: none !important;
    }
    [data-testid="stChatMessageAvatar"] {
        background-color: #0d131d !important;
        border: 1px solid #2c3649 !important;
    }
    /* Streamlit 1.62 renders the avatar as the first child without a stable
       data-testid; keep that actual container dark as well. */
    [data-testid="stChatMessage"] > div:first-child {
        background-color: #0d131d !important;
        border: 1px solid #2c3649 !important;
    }
    .stTextInput>div>div>input, .stSelectbox>div>div>div { background-color: #252526 !important; color: #eeeeee !important; border: 1px solid #3d3d3d !important; border-radius: 2px !important; }
    .stTextInput>div>div>input:focus, .stSelectbox>div>div>div:focus { border-color: #3A62D7 !important; box-shadow: none !important; }
    div.stButton > button { border-radius: 2px !important; font-weight: 600 !important; letter-spacing: 1px; padding: 0.5rem 1rem !important; }
    [data-testid="stExpander"] { background-color: #1b1b1d !important; border: 1px solid #333333 !important; border-radius: 2px !important; }
    [data-testid="stVerticalBlock"] > div.element-container > div.stMarkdown > div > pre { background-color: #000000 !important; border: 1px solid #333333 !important; color: #00ff00 !important; }
    .main-title { font-size: 1.8rem; font-weight: 600; color: #eeeeee !important; margin-bottom: 0px; border-left: 4px solid #3A62D7; padding-left: 10px;}
    .sub-title { font-size: 0.9rem; color: #888888 !important; margin-bottom: 15px; margin-top: 5px; padding-left: 14px;}
    .stProgress > div > div > div > div { background-color: #3A62D7 !important; }
    .msg-role {
        display: inline-block;
        padding: 0.1rem 0.45rem;
        border-radius: 999px;
        font-size: 0.74rem;
        font-weight: 700;
        margin-bottom: 0.35rem;
        letter-spacing: 0.2px;
    }
    .msg-role-user {
        background: #1f355a;
        color: #cfe1ff !important;
        border: 1px solid #3a62d7;
    }
    .msg-role-assistant {
        background: #23452f;
        color: #cbf1d4 !important;
        border: 1px solid #4ea56a;
    }
    [data-testid="stChatMessage"] {
        display: flex !important;
        flex: 0 0 auto !important;
        width: fit-content !important;
        min-width: 7rem !important;
        max-width: 86% !important;
        background: linear-gradient(180deg, #141a25 0%, #10151f 100%);
        border: 1px solid #2c3649;
        border-radius: 12px;
        padding: 0.45rem 0.65rem;
        margin-bottom: 0.45rem;
        box-sizing: border-box;
    }
    /* 对话采用双侧气泡：助手在左，用户在右。 */
    [data-testid="stChatMessage"]:has(.msg-role-assistant) {
        margin-left: 0 !important;
        margin-right: auto !important;
        border-left: 3px solid #4ea56a;
    }
    [data-testid="stChatMessage"]:has(.msg-role-user) {
        flex-direction: row-reverse !important;
        margin-left: auto !important;
        margin-right: 0 !important;
        background: linear-gradient(180deg, #182b4a 0%, #13223a 100%);
        border-right: 3px solid #5d82e8;
        border-left: 1px solid #2c4678;
    }
    [data-testid="stChatMessage"]:has(.msg-role-user) [data-testid="stChatMessageAvatar"] {
        margin-left: 0.55rem !important;
        margin-right: 0 !important;
    }
    [data-testid="stChatMessage"]:has(.msg-role-assistant) [data-testid="stChatMessageAvatar"] {
        margin-right: 0.55rem !important;
    }
    [data-testid="stChatMessage"] [data-testid="stChatMessageContent"] {
        width: fit-content !important;
        max-width: 100% !important;
        min-width: 0 !important;
        overflow-wrap: anywhere;
    }
    [data-testid="stChatMessage"] pre,
    [data-testid="stChatMessage"] img,
    [data-testid="stChatMessage"] table {
        max-width: 100% !important;
    }
    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] p {
        color: #e6ecf6 !important;
        line-height: 1.5;
    }
    :root {
        --workbench-h: calc(100vh - 3.5rem);
    }
    /* 工作台固定视口：禁止整页滚动（不用 position:fixed，避免上次压缩问题） */
    html, body {
        overflow: hidden !important;
        height: 100% !important;
        overscroll-behavior: none !important;
    }
    .stApp,
    [data-testid="stAppViewContainer"],
    section[data-testid="stMain"],
    div[data-testid="stMainBlockContainer"] {
        overflow: hidden !important;
        max-height: 100vh !important;
        overscroll-behavior: none !important;
    }
    div[data-testid="stMainBlockContainer"] > div[data-testid="stVerticalBlock"]:has([data-testid="stHorizontalBlock"]:has(.cockpit-map-col)) {
        height: var(--workbench-h) !important;
        max-height: var(--workbench-h) !important;
        overflow: hidden !important;
    }
    div[data-testid="stHorizontalBlock"]:has(.cockpit-map-col) {
        height: var(--workbench-h) !important;
        max-height: var(--workbench-h) !important;
        overflow: hidden !important;
        align-items: stretch !important;
        /* Keep map and Agent side-by-side after a drag or viewport resize.
           Streamlit otherwise wraps the second column below the fixed-height
           workbench, making the Agent UI appear to disappear. */
        flex-wrap: nowrap !important;
    }
    div[data-testid="stHorizontalBlock"]:has(.cockpit-map-col) > div[data-testid="stColumn"] {
        height: var(--workbench-h) !important;
        max-height: var(--workbench-h) !important;
        overflow: hidden !important;
        align-self: stretch !important;
        min-width: 0 !important;
    }
    div[data-testid="stColumn"]:has(.cockpit-map-col) > div[data-testid="stVerticalBlock"] {
        height: 100% !important;
        max-height: var(--workbench-h) !important;
        overflow: visible !important;
        /* The map, zero-height bridge, and status drawer share one column.
           Streamlit's default 1rem child gap otherwise creates an empty row
           above the drawer and moves the edge control away from the boundary. */
        gap: 0 !important;
    }
    .cockpit-map-col,
    .cockpit-chat-anchor,
    .cockpit-copilot-zone-start {
        display: none !important;
        height: 0 !important;
        min-height: 0 !important;
        margin: 0 !important;
        padding: 0 !important;
        border: none !important;
        overflow: hidden !important;
    }
    section[data-testid="stMain"] > div,
    section[data-testid="stMain"] .block-container,
    div[data-testid="stMainBlockContainer"] {
        padding-top: 0.25rem;
        padding-bottom: 0.25rem;
        padding-left: 0.5rem;
        padding-right: 0.5rem;
        max-width: 100% !important;
        width: 100% !important;
    }
    /* 三维地球 iframe 撑满主区域（仅地图列） */
    div[data-testid="stColumn"]:has(.cockpit-map-col) [data-testid="stIFrame"],
    div[data-testid="stColumn"]:has(.cockpit-map-col) [data-testid="stHtml"],
    div[data-testid="stColumn"]:has(.cockpit-map-col) div[data-testid="stElementContainer"]:has(iframe.yy-globe-frame) {
        width: 100% !important;
        max-width: 100% !important;
    }
    div[data-testid="stColumn"]:has(.cockpit-map-col) iframe.yy-globe-frame,
    div[data-testid="stColumn"]:has(.cockpit-map-col) iframe[src*="/globe"],
    div[data-testid="stColumn"]:has(.cockpit-map-col) iframe[title*="streamlit_folium"] {
        border: none !important;
        width: 100% !important;
        max-width: 100% !important;
        height: calc(var(--workbench-h) - var(--cstf-status-panel-reserve, 0px)) !important;
        min-height: 280px !important;
        max-height: calc(var(--workbench-h) - var(--cstf-status-panel-reserve, 0px)) !important;
        display: block !important;
        background: #0a1628;
    }
    div[data-testid="stColumn"]:has(.cockpit-map-col) [data-testid="stIFrame"]:has(iframe[src*="/globe"]),
    div[data-testid="stColumn"]:has(.cockpit-map-col) [data-testid="stCustomComponentV1"]:has(iframe[title*="streamlit_folium"]) {
        height: calc(var(--workbench-h) - var(--cstf-status-panel-reserve, 0px)) !important;
        min-height: 280px !important;
        max-height: calc(var(--workbench-h) - var(--cstf-status-panel-reserve, 0px)) !important;
    }
    div[data-testid="stColumn"]:has(.cockpit-map-col) > div[data-testid="stVerticalBlock"] > [data-testid="stElementContainer"]:has(iframe[src*="/globe"]),
    div[data-testid="stColumn"]:has(.cockpit-map-col) > div[data-testid="stVerticalBlock"] > [data-testid="stElementContainer"]:has(iframe[title*="streamlit_folium"]) {
        flex: 0 0 auto !important;
        height: calc(var(--workbench-h) - var(--cstf-status-panel-reserve, 0px)) !important;
        min-height: 280px !important;
        max-height: calc(var(--workbench-h) - var(--cstf-status-panel-reserve, 0px)) !important;
        overflow: hidden !important;
    }
    /* 地图下方状态抽屉：高度由顶部拖拽边缘控制，内容过多时只在抽屉内滚动。 */
    div[data-testid="stColumn"]:has(.cstf-map-status-zone) > div[data-testid="stVerticalBlock"] {
        overflow: visible !important;
    }
    .cstf-map-status-zone {
        height: 0 !important;
        min-height: 0 !important;
        margin: 0 !important;
        padding: 0 !important;
        overflow: hidden !important;
    }
    .cstf-layout-defaults {
        display: none !important;
        width: 0 !important;
        height: 0 !important;
    }
    /* 通知脱离主布局流，避免错误/警告把地图与 Agent 挤出视口；每条通知可独立关闭。 */
    [data-testid="stAlert"].cstf-dismissible-alert {
        position: fixed !important;
        right: 1rem !important;
        top: var(--cstf-alert-top, 1rem) !important;
        z-index: 1400 !important;
        width: min(30rem, calc(100vw - 2rem)) !important;
        max-height: 8rem !important;
        overflow: auto !important;
        margin: 0 !important;
        padding-right: 2.4rem !important;
        border: 1px solid rgba(131, 151, 190, 0.35) !important;
        box-shadow: 0 12px 32px rgba(0, 0, 0, 0.36) !important;
        backdrop-filter: blur(8px);
    }
    [data-testid="stAlert"].cstf-dismissible-alert .cstf-alert-close {
        position: absolute;
        top: 0.45rem;
        right: 0.45rem;
        width: 1.65rem;
        height: 1.65rem;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        border: 1px solid rgba(180, 193, 221, 0.38);
        border-radius: 999px;
        background: rgba(15, 22, 35, 0.68);
        color: #e8edf7;
        cursor: pointer;
        font-size: 1rem;
        line-height: 1;
    }
    [data-testid="stAlert"].cstf-dismissible-alert .cstf-alert-close:hover,
    [data-testid="stAlert"].cstf-dismissible-alert .cstf-alert-close:focus-visible {
        background: #334a82;
        border-color: #6e8ee1;
        outline: none;
    }
    .cstf-status-toolbar-spacer {
        /* Streamlit button bridge is kept in the DOM for state sync, but must
           never reserve a visible row between the map and the drawer. */
        min-height: 0 !important;
        width: 100%;
        height: 0 !important;
        margin: 0 !important;
        padding: 0 !important;
    }
    /* 原生按钮仅作为 Streamlit 状态桥接，实际控制显示在地图底边。 */
    div[data-testid="stHorizontalBlock"]:has(> div[data-testid="stColumn"] > div[data-testid="stVerticalBlock"] > [data-testid="stElementContainer"] .cstf-status-toolbar-spacer) {
        /* Keep the Streamlit widget mounted so a second bridge click still
           reaches its event handler, while removing the row from layout. */
        display: flex !important;
        position: absolute !important;
        left: 0 !important;
        top: 0 !important;
        width: 0 !important;
        height: 0 !important;
        min-height: 0 !important;
        max-height: 0 !important;
        margin: 0 !important;
        padding: 0 !important;
        overflow: visible !important;
        pointer-events: none !important;
    }
    /* The bridge remains a real, event-capable Streamlit button.  Keep it
       visually inert and out of the layout while allowing the edge triangle
       to trigger its click handler after a collapse rerun. */
    div.st-key-agent_status_panel_toggle button {
        position: absolute !important;
        left: 0 !important;
        top: 0 !important;
        width: 1px !important;
        height: 1px !important;
        min-width: 1px !important;
        min-height: 1px !important;
        padding: 0 !important;
        margin: 0 !important;
        opacity: 0 !important;
        pointer-events: auto !important;
    }
    .cstf-status-toggle-state,
    .cstf-status-toggle-host {
        display: none !important;
    }
    .cstf-status-edge-toggle {
        position: fixed;
        z-index: 1400;
        width: 2.25rem;
        height: 1.6rem;
        /* 水平居中；按钮自身的底边由脚本贴到地图/状态区分界线。 */
        transform: translateX(-50%);
        display: inline-flex;
        align-items: center;
        justify-content: center;
        border: 1px solid #536891;
        border-radius: 0.45rem;
        background: #101a2d;
        color: #cddaff;
        cursor: pointer;
        font-size: 1rem;
        line-height: 1;
        opacity: 0.82;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.32);
        transition: opacity 160ms ease, background 160ms ease, border-color 160ms ease;
    }
    .cstf-status-edge-toggle:hover,
    .cstf-status-edge-toggle:focus-visible {
        opacity: 1;
        background: #243967;
        border-color: #89a7f0;
        outline: none;
    }
    /* 状态区顶部边缘：默认只保留命中区域，鼠标靠近时显示分隔提示。 */
    .cstf-status-edge-handle {
        position: fixed;
        z-index: 1350;
        height: 14px;
        transform: translateY(-50%);
        cursor: row-resize;
        touch-action: none;
        user-select: none;
        border-radius: 999px;
        opacity: 0.28;
        transition: opacity 160ms ease;
    }
    .cstf-status-edge-handle,
    .cstf-status-edge-handle:focus-visible,
    .cstf-dock-resize-handle,
    .cstf-dock-resize-handle:focus-visible {
        outline: none !important;
        border: 0 !important;
        box-shadow: none !important;
    }
    .cstf-status-edge-handle::after {
        content: "";
        position: absolute;
        left: 35%;
        top: 5px;
        width: 30%;
        min-width: 96px;
        height: 4px;
        border-radius: 999px;
        background: #5e72a1;
        transition: background 160ms ease, transform 160ms ease;
    }
    .cstf-status-edge-handle::before {
        content: "↕";
        position: absolute;
        left: 50%;
        top: 50%;
        transform: translate(-50%, -50%);
        color: #9bb2ff;
        font-size: 0.72rem;
        line-height: 1;
        opacity: 0;
        transition: opacity 160ms ease;
    }
    .cstf-status-edge-handle:hover,
    .cstf-status-edge-handle:focus-visible,
    body.cstf-resizing-status .cstf-status-edge-handle {
        opacity: 1;
    }
    .cstf-status-edge-handle:hover::after,
    .cstf-status-edge-handle:focus-visible::after,
    body.cstf-resizing-status .cstf-status-edge-handle::after {
        background: #9bb2ff;
        transform: scaleX(1.08);
    }
    .cstf-status-edge-handle:hover::before,
    .cstf-status-edge-handle:focus-visible::before,
    body.cstf-resizing-status .cstf-status-edge-handle::before {
        opacity: 1;
    }
    /* 地图与 Agent 之间的边缘分隔条，命中区比可视线略宽，避免精确点按。 */
    .cstf-dock-resize-handle {
        position: fixed;
        z-index: 1600;
        width: 16px;
        transform: translateX(-50%);
        cursor: col-resize;
        touch-action: none;
        user-select: none;
        border-radius: 999px;
        opacity: 0.34;
        transition: opacity 160ms ease;
    }
    .cstf-dock-resize-handle::after {
        content: "";
        position: absolute;
        left: 5px;
        top: 35%;
        width: 4px;
        height: 30%;
        min-height: 48px;
        border-radius: 999px;
        background: #3d4c67;
        transition: background 160ms ease, transform 160ms ease;
    }
    .cstf-dock-resize-handle::before {
        content: "◀ ▶";
        position: absolute;
        left: 50%;
        top: 50%;
        transform: translate(-50%, -50%);
        color: #9bb2ff;
        font-size: 0.68rem;
        line-height: 1;
        letter-spacing: 0.05rem;
        white-space: nowrap;
        opacity: 0;
        transition: opacity 160ms ease;
    }
    .cstf-dock-resize-handle:hover,
    .cstf-dock-resize-handle:focus-visible,
    body.cstf-resizing-agent .cstf-dock-resize-handle {
        opacity: 1;
    }
    .cstf-dock-resize-handle:hover::after,
    .cstf-dock-resize-handle:focus-visible::after,
    body.cstf-resizing-agent .cstf-dock-resize-handle::after {
        background: #6d8fe8;
        transform: scaleX(1.25);
    }
    .cstf-dock-resize-handle:hover::before,
    .cstf-dock-resize-handle:focus-visible::before,
    body.cstf-resizing-agent .cstf-dock-resize-handle::before {
        opacity: 1;
    }
    div[data-testid="stColumn"]:has(.cstf-map-status-zone) [data-testid="stVerticalBlockBorderWrapper"] {
        border-color: #2a3548 !important;
        background: #0d131d !important;
    }
    .command-deck {
        border-top: 1px solid #333;
        padding-top: 12px;
        margin-top: 4px;
    }
    .command-deck-side {
        border-left: 1px solid #2a3548;
        padding-left: 10px;
        margin-left: 2px;
        height: 100%;
    }
    /* 右侧仅保留 Agent 对话 Dock（Streamlit 1.3x 使用 stLayoutWrapper）。 */
    div[data-testid="stColumn"]:has(.command-deck-side) > div[data-testid="stVerticalBlock"] {
        display: flex !important;
        flex-direction: column !important;
        height: var(--workbench-h) !important;
        max-height: var(--workbench-h) !important;
        overflow: hidden !important;
        gap: 0.3rem !important;
    }
    div[data-testid="stColumn"]:has(.command-deck-side) > div[data-testid="stVerticalBlock"] > [data-testid="stLayoutWrapper"]:has([data-testid="stProgress"]),
    div[data-testid="stColumn"]:has(.command-deck-side) > div[data-testid="stVerticalBlock"] > [data-testid="stElementContainer"]:has(.deck-section-title) {
        flex: 0 0 auto !important;
    }
    div[data-testid="stColumn"]:has(.command-deck-side) > div[data-testid="stVerticalBlock"] > [data-testid="stLayoutWrapper"]:has([data-testid="stChatMessage"]):not(:has([data-testid="stForm"])) {
        flex: 1 1 auto !important;
        min-height: 140px !important;
        overflow-x: hidden !important;
        overflow-y: auto !important;
    }
    div[data-testid="stColumn"]:has(.command-deck-side) > div[data-testid="stVerticalBlock"] > [data-testid="stLayoutWrapper"]:has([data-testid="stForm"] input[aria-label="chat_input"]) {
        flex: 0 0 auto !important;
        margin-top: auto !important;
    }
    div[data-testid="stColumn"]:has(.command-deck-side) > div[data-testid="stVerticalBlock"] > [data-testid="stElementContainer"]:has([data-testid="stVerticalBlockBorderWrapper"] [data-testid="stChatMessage"]) {
        flex: 1 1 auto !important;
        min-height: 140px !important;
        overflow: hidden !important;
        display: flex !important;
        flex-direction: column !important;
    }
    div[data-testid="stColumn"]:has(.command-deck-side) [data-testid="stVerticalBlockBorderWrapper"]:has([data-testid="stChatMessage"]) {
        flex: 1 1 auto !important;
        min-height: 0 !important;
        overflow-y: auto !important;
        overflow-x: hidden !important;
    }
    div[data-testid="stColumn"]:has(.command-deck-side) > div[data-testid="stVerticalBlock"] > [data-testid="stElementContainer"]:has([data-testid="stForm"] input[aria-label="chat_input"]) {
        flex: 0 0 auto !important;
        margin-top: auto !important;
    }
    div[data-testid="stColumn"]:has(.command-deck-side) > div[data-testid="stVerticalBlock"] > [data-testid="stElementContainer"]:has(iframe),
    div[data-testid="stColumn"]:has(.command-deck-side) > div[data-testid="stVerticalBlock"] > [data-testid="stLayoutWrapper"]:has(iframe) {
        flex: 0 0 0 !important;
        height: 0 !important;
        min-height: 0 !important;
        overflow: hidden !important;
        margin: 0 !important;
        padding: 0 !important;
    }
    div[data-testid="stColumn"]:has(.command-deck-side) {
        background: linear-gradient(180deg, #0f141c 0%, #0b1018 100%);
        border-radius: 8px;
        padding: 8px 6px 8px 4px !important;
    }
    /* 收起 Agent Dock 时，右侧只保留展开入口，地图占据主区域。 */
    div[data-testid="stColumn"]:has(.cstf-dock-collapsed-marker) {
        background: #0b1018 !important;
        padding: 8px 2px !important;
    }
    div[data-testid="stColumn"]:has(.cstf-dock-collapsed-marker) > div[data-testid="stVerticalBlock"] > *:not(:has(.cstf-dock-collapsed-marker)) {
        display: none !important;
    }
    div[data-testid="stColumn"]:has(.cstf-dock-collapse-control) button {
        white-space: nowrap !important;
        font-size: 0.72rem !important;
        padding: 0.28rem 0.35rem !important;
    }
    div[data-testid="stColumn"]:has(.cockpit-map-col) {
        padding-right: 4px !important;
    }
    .cstf-log-panel-host-marker,
    div[data-testid="stElementContainer"]:has(> .cstf-copilot-dock:empty),
    div[data-testid="stElementContainer"]:has(> .cstf-chat-compose-host:empty) {
        display: none !important;
        height: 0 !important;
        min-height: 0 !important;
        margin: 0 !important;
        padding: 0 !important;
        overflow: hidden !important;
    }
    /* CSTF-Copilot 输入区：文字在上，下方 + 号附件 */
    div[data-testid="stForm"]:has(input[aria-label="chat_input"]) {
        border: 1px solid #2e384c !important;
        border-radius: 18px !important;
        padding: 8px 10px 6px !important;
        background: linear-gradient(180deg, #151b28 0%, #121720 100%) !important;
        margin-top: 4px !important;
    }
    div[data-testid="stForm"]:has(input[aria-label="chat_input"]) [data-testid="stHorizontalBlock"] {
        align-items: center !important;
        gap: 0.35rem !important;
    }
    div[data-testid="stForm"]:has(input[aria-label="chat_input"]) [data-testid="stHorizontalBlock"]:has(.cstf-attach-bar) {
        display: flex !important;
        flex-direction: row !important;
        flex-wrap: nowrap !important;
        align-items: center !important;
        width: 100% !important;
        min-width: 0 !important;
        max-width: none !important;
        overflow: visible !important;
    }
    div[data-testid="stForm"]:has(input[aria-label="chat_input"]) [data-testid="stHorizontalBlock"]:has(.cstf-attach-bar) > [data-testid="stColumn"] {
        min-width: 0 !important;
        min-height: 0 !important;
        margin: 0 !important;
        padding-top: 0 !important;
        padding-bottom: 0 !important;
    }
    div[data-testid="stForm"]:has(input[aria-label="chat_input"]) [data-testid="stHorizontalBlock"]:has(.cstf-attach-bar) > .cstf-attach-bar {
        flex: 0 0 2.65rem !important;
        width: 2.65rem !important;
        max-width: 2.65rem !important;
        order: 0 !important;
    }
    div[data-testid="stForm"]:has(input[aria-label="chat_input"]) [data-testid="stHorizontalBlock"]:has(.cstf-attach-bar) > [data-testid="stColumn"]:has(input[aria-label="chat_input"]) {
        flex: 1 1 auto !important;
        width: auto !important;
        max-width: none !important;
        order: 1 !important;
    }
    div[data-testid="stForm"]:has(input[aria-label="chat_input"]) [data-testid="stHorizontalBlock"]:has(.cstf-attach-bar) > [data-testid="stColumn"]:has([data-testid="stFormSubmitButton"]) {
        flex: 0 0 2.65rem !important;
        width: 2.65rem !important;
        min-width: 2.65rem !important;
        max-width: 2.65rem !important;
        order: 2 !important;
    }
    div[data-testid="stForm"]:has(input[aria-label="chat_input"]) input[aria-label="chat_input"] {
        border-radius: 14px !important;
        border: 1px solid #2a3548 !important;
        background: #0c1018 !important;
        color: #e8edf7 !important;
        padding: 11px 14px !important;
        font-size: 0.92rem !important;
        min-height: 2.65rem !important;
    }
    div[data-testid="stForm"]:has(input[aria-label="chat_input"]) input[aria-label="chat_input"]::placeholder {
        color: #6b7a94 !important;
    }
    div[data-testid="stForm"]:has(input[aria-label="chat_input"]) input[aria-label="chat_input"]:focus {
        border-color: #4a6cf0 !important;
        box-shadow: 0 0 0 1px rgba(74, 108, 240, 0.35) !important;
    }
    div[data-testid="stForm"]:has(input[aria-label="chat_input"]) [data-testid="stFormSubmitButton"] button {
        border-radius: 50% !important;
        width: 2.65rem !important;
        height: 2.65rem !important;
        min-width: 2.65rem !important;
        padding: 0 !important;
        font-size: 1.05rem !important;
        background: #2a3f7a !important;
        border: 1px solid #3d56a8 !important;
        color: #e8eeff !important;
    }
    div[data-testid="stForm"]:has(input[aria-label="chat_input"]) [data-testid="stFormSubmitButton"] button:hover {
        background: #3552a0 !important;
        border-color: #5a7fd4 !important;
    }
    div[data-testid="stForm"]:has(input[aria-label="chat_input"]) [data-testid="stFileUploader"] {
        position: absolute !important;
        width: 1px !important;
        height: 1px !important;
        margin: -1px !important;
        padding: 0 !important;
        overflow: hidden !important;
        clip: rect(0 0 0 0) !important;
    }
    div[data-testid="stForm"]:has(input[aria-label="chat_input"]) [data-testid="stFileUploaderDropzone"],
    div[data-testid="stForm"]:has(input[aria-label="chat_input"]) [data-testid="stFileUploader"] > label,
    div[data-testid="stForm"]:has(input[aria-label="chat_input"]) [data-testid="stFileUploader"] section,
    div[data-testid="stForm"]:has(input[aria-label="chat_input"]) [data-testid="stFileUploader"] small {
        display: none !important;
    }
    div[data-testid="stForm"]:has(input[aria-label="chat_input"]) [data-testid="stFileUploader"] input[type="file"] {
        position: absolute !important;
        width: 1px !important;
        height: 1px !important;
        opacity: 0 !important;
        overflow: hidden !important;
        clip: rect(0, 0, 0, 0) !important;
    }
    div[data-testid="stForm"]:has(input[aria-label="chat_input"]) .cstf-attach-bar {
        display: flex;
        align-items: center;
        gap: 8px;
        flex: 0 0 2.65rem;
        width: 2.65rem;
        height: 2.65rem;
        margin: 0;
        padding: 0;
        min-height: 0;
        justify-content: center;
        order: -1;
    }
    div[data-testid="stForm"]:has(input[aria-label="chat_input"]) .cstf-plus-btn {
        position: relative;
        width: 32px;
        height: 32px;
        border-radius: 50%;
        border: 1px solid #3d4a63;
        background: transparent;
        color: #d0daf0;
        font-size: 1.4rem;
        font-weight: 300;
        line-height: 1;
        cursor: pointer;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        padding: 0;
        flex-shrink: 0;
        transition: background 0.15s, border-color 0.15s;
    }
    /* 限制说明固定显示在加号上方，不依赖浏览器原生 title 气泡。 */
    div[data-testid="stForm"]:has(input[aria-label="chat_input"]) .cstf-plus-btn::after {
        content: attr(data-tooltip);
        position: absolute;
        /* 加号位于 Agent Dock 左缘，向右展开避免提示被地图/面板边界裁掉。 */
        left: 0;
        bottom: calc(100% + 8px);
        transform: translate(0, 4px);
        width: max-content;
        max-width: min(19rem, calc(100vw - 2rem));
        padding: 6px 9px;
        border: 1px solid #4a5b7b;
        border-radius: 7px;
        background: #111a2a;
        color: #edf2ff;
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.38);
        font-size: 0.72rem;
        font-weight: 400;
        line-height: 1.35;
        text-align: left;
        white-space: normal;
        opacity: 0;
        pointer-events: none;
        z-index: 1500;
        transition: opacity 120ms ease, transform 120ms ease;
    }
    div[data-testid="stForm"]:has(input[aria-label="chat_input"]) .cstf-plus-btn:hover::after,
    div[data-testid="stForm"]:has(input[aria-label="chat_input"]) .cstf-plus-btn:focus-visible::after {
        opacity: 1;
        transform: translate(0, 0);
    }
    div[data-testid="stForm"]:has(input[aria-label="chat_input"]) .cstf-plus-btn:hover {
        background: #1e2838;
        border-color: #5a6d92;
        color: #ffffff;
    }
    div[data-testid="stForm"]:has(input[aria-label="chat_input"]) [data-testid="stCaptionContainer"] {
        margin-top: 2px !important;
        margin-bottom: 0 !important;
        line-height: 1.25 !important;
        font-size: 0.68rem !important;
    }
    /* Streamlit 会按浏览器语言翻译 widget label；以 JS 标记的真实聊天表单
       作为最终布局锚点，避免 “chat_input” 被翻译后恢复成默认上传器布局。 */
    div[data-testid="stForm"].cstf-chat-compose {
        position: relative !important;
        overflow: visible !important;
        border: 1px solid #2e384c !important;
        border-radius: 18px !important;
        padding: 8px 10px 6px !important;
        background: linear-gradient(180deg, #151b28 0%, #121720 100%) !important;
        margin-top: 4px !important;
    }
    /* 预览条是输入框的一部分：显示时为输入行预留缩略图高度，避免
       预览脱离外框漂浮到消息区。 */
    div[data-testid="stForm"].cstf-chat-compose:has(.cstf-attach-preview.is-visible) {
        padding-top: 5.8rem !important;
    }
    /* 预览条需要以聊天表单为定位上下文，而不是以只有加号宽度的入口为定位上下文。 */
    div[data-testid="stForm"].cstf-chat-compose .cstf-attach-bar {
        position: static !important;
    }
    div[data-testid="stForm"].cstf-chat-compose .cstf-chat-input-row {
        display: flex !important;
        flex-direction: row !important;
        flex-wrap: nowrap !important;
        align-items: center !important;
        gap: 0.35rem !important;
        width: 100% !important;
        min-width: 0 !important;
        overflow: visible !important;
    }
    div[data-testid="stForm"].cstf-chat-compose .cstf-chat-input-row > .cstf-attach-bar {
        display: flex !important;
        flex: 0 0 2.65rem !important;
        order: 0 !important;
        width: 2.65rem !important;
        height: 2.65rem !important;
        align-items: center !important;
        justify-content: center !important;
        margin: 0 !important;
        padding: 0 !important;
    }
    div[data-testid="stForm"].cstf-chat-compose .cstf-chat-input-column {
        flex: 1 1 auto !important;
        order: 1 !important;
        width: auto !important;
        min-width: 0 !important;
        max-width: none !important;
    }
    div[data-testid="stForm"].cstf-chat-compose .cstf-chat-send-column {
        flex: 0 0 2.65rem !important;
        order: 2 !important;
        width: 2.65rem !important;
        min-width: 2.65rem !important;
        max-width: 2.65rem !important;
    }
    div[data-testid="stForm"].cstf-chat-compose input[data-testid="stTextInputField"] {
        min-height: 2.65rem !important;
        width: 100% !important;
    }
    div[data-testid="stForm"].cstf-chat-compose [data-testid="stFormSubmitButton"] button {
        width: 2.65rem !important;
        min-width: 2.65rem !important;
        height: 2.65rem !important;
        padding: 0 !important;
        border-radius: 50% !important;
    }
    /* 原生上传器不参与表单流：只保留供 + 按钮触发的隐藏 file input。 */
    div[data-testid="stForm"].cstf-chat-compose [data-testid="stFileUploader"] {
        position: absolute !important;
        width: 1px !important;
        height: 1px !important;
        margin: -1px !important;
        padding: 0 !important;
        overflow: hidden !important;
        clip: rect(0 0 0 0) !important;
    }
    div[data-testid="stForm"].cstf-chat-compose [data-testid="stFileUploader"] > label,
    div[data-testid="stForm"].cstf-chat-compose [data-testid="stFileUploaderDropzone"],
    div[data-testid="stForm"].cstf-chat-compose [data-testid="stFileUploader"] section,
    div[data-testid="stForm"].cstf-chat-compose [data-testid="stFileUploader"] small {
        display: none !important;
    }
    div[data-testid="stForm"].cstf-chat-compose div[data-testid="stElementContainer"]:has(
        .cstf-attachment-epoch-marker
    ) {
        display: none !important;
        height: 0 !important;
        min-height: 0 !important;
        margin: 0 !important;
        padding: 0 !important;
    }
    div[data-testid="stForm"].cstf-chat-compose .cstf-attach-bar {
        position: static !important;
    }
    /* 选择附件后在聊天输入外框内显示缩略图/格式卡片，不改变消息行宽度。 */
    div[data-testid="stForm"].cstf-chat-compose .cstf-attach-preview {
        position: absolute !important;
        top: 8px !important;
        right: 10px !important;
        bottom: auto !important;
        left: 10px !important;
        display: none !important;
        align-items: center !important;
        flex-direction: row !important;
        flex-wrap: nowrap !important;
        gap: 6px !important;
        width: auto !important;
        min-width: 0 !important;
        max-width: none !important;
        max-height: 5.5rem !important;
        padding: 5px !important;
        overflow-x: auto !important;
        overflow-y: hidden !important;
        border: 1px solid #41506c !important;
        border-radius: 10px !important;
        background: #111827 !important;
        box-shadow: 0 10px 24px rgba(0, 0, 0, 0.42) !important;
        z-index: 1550 !important;
    }
    div[data-testid="stForm"].cstf-chat-compose .cstf-attach-preview.is-visible {
        display: flex !important;
    }
    div[data-testid="stForm"].cstf-chat-compose .cstf-attach-preview-card {
        position: relative !important;
        display: inline-flex !important;
        flex: 0 0 4.5rem !important;
        width: 4.5rem !important;
        height: 4.5rem !important;
        align-items: center !important;
        justify-content: center !important;
        overflow: hidden !important;
        border: 1px solid #41506c !important;
        border-radius: 8px !important;
        background: #0d131d !important;
        color: #c9d5ea !important;
        font-size: 0.62rem !important;
        text-align: center !important;
    }
    div[data-testid="stForm"].cstf-chat-compose .cstf-attach-preview-card img {
        width: 100% !important;
        height: 100% !important;
        object-fit: cover !important;
    }
    div[data-testid="stForm"].cstf-chat-compose .cstf-attach-preview-label {
        max-width: 100% !important;
        padding: 4px !important;
        overflow: hidden !important;
        color: #c9d5ea !important;
        font-size: 0.62rem !important;
        line-height: 1.25 !important;
        text-overflow: ellipsis !important;
        white-space: nowrap !important;
    }
    div[data-testid="stForm"].cstf-chat-compose .cstf-attach-preview-clear {
        align-self: flex-start !important;
        flex: 0 0 1.35rem !important;
        width: 1.35rem !important;
        height: 1.35rem !important;
        margin: -2px -2px 0 0 !important;
        padding: 0 !important;
        border: 1px solid #41506c !important;
        border-radius: 50% !important;
        background: #0d131d !important;
        color: #edf2ff !important;
        font-size: 0.9rem !important;
        line-height: 1 !important;
        cursor: pointer !important;
    }
    div[data-testid="stForm"].cstf-chat-compose .cstf-attach-preview-clear:hover,
    div[data-testid="stForm"].cstf-chat-compose .cstf-attach-preview-clear:focus-visible {
        border-color: #6f8fd7 !important;
        background: #1e2838 !important;
        outline: none !important;
    }
    div[data-testid="stForm"].cstf-chat-compose .cstf-plus-btn {
        position: relative !important;
        width: 32px !important;
        height: 32px !important;
        flex: 0 0 32px !important;
    }
    div[data-testid="stForm"].cstf-chat-compose .cstf-plus-btn::after {
        content: attr(data-tooltip);
        position: absolute;
        left: 0;
        bottom: calc(100% + 8px);
        transform: translate(0, 4px);
        width: max-content;
        max-width: min(19rem, calc(100vw - 2rem));
        padding: 6px 9px;
        border: 1px solid #4a5b7b;
        border-radius: 7px;
        background: #111a2a;
        color: #edf2ff;
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.38);
        font-size: 0.72rem;
        font-weight: 400;
        line-height: 1.35;
        white-space: normal;
        opacity: 0;
        pointer-events: none;
        z-index: 1500;
        transition: opacity 120ms ease, transform 120ms ease;
    }
    div[data-testid="stForm"].cstf-chat-compose .cstf-plus-btn:hover::after,
    div[data-testid="stForm"].cstf-chat-compose .cstf-plus-btn:focus-visible::after {
        opacity: 1;
        transform: translate(0, 0);
    }
    .deck-section-title {
        font-size: 0.95rem !important;
        font-weight: 600 !important;
        color: #d0d0d0 !important;
        margin-bottom: 6px !important;
        border-left: 3px solid #3A62D7;
        padding-left: 8px;
    }
    /* 历史页切换为会话导航；选中会话后才进入对话流。 */
    div[data-testid="stColumn"]:has(.cstf-agent-view-history) > div[data-testid="stVerticalBlock"] > [data-testid="stElementContainer"]:has([data-testid="stVerticalBlockBorderWrapper"] [data-testid="stChatMessage"]) {
        flex: 0 1 auto !important;
        min-height: 120px !important;
        max-height: 250px !important;
        overflow: hidden !important;
    }
    div[data-testid="stColumn"]:has(.cstf-agent-view-history) [data-testid="stVerticalBlockBorderWrapper"]:has([data-testid="stChatMessage"]) {
        flex: 0 1 auto !important;
        min-height: 0 !important;
        max-height: 250px !important;
        overflow-y: auto !important;
        overflow-x: hidden !important;
    }
    /* 历史会话数量不截断；只让带标记的历史列表容器在自身范围内滚动。 */
    div[data-testid="stColumn"]:has(.cstf-agent-view-history) > div[data-testid="stVerticalBlock"] > [data-testid="stElementContainer"]:has(.cstf-history-list-marker) {
        flex: 1 1 auto !important;
        min-height: 0 !important;
        overflow: hidden !important;
        display: flex !important;
        flex-direction: column !important;
    }
    div[data-testid="stColumn"]:has(.cstf-agent-view-history) [data-testid="stVerticalBlockBorderWrapper"]:has(.cstf-history-list-marker) {
        flex: 1 1 auto !important;
        min-height: 0 !important;
        height: 100% !important;
        max-height: 100% !important;
        overflow-y: auto !important;
        overflow-x: hidden !important;
    }
    div[data-testid="stColumn"]:has(.cstf-agent-view-history) [data-testid="stLayoutWrapper"]:has(.cstf-history-list-marker) {
        flex: 1 1 0 !important;
        min-height: 0 !important;
        height: 100% !important;
        max-height: 100% !important;
        overflow-y: auto !important;
        overflow-x: hidden !important;
        /* Keep the frame on the fixed scroll viewport; the inner block moves
           with scrollTop and must not carry the visible border. */
        border: 1px solid rgba(250, 250, 250, 0.2) !important;
        border-radius: 8px !important;
        box-sizing: border-box !important;
        background: #0d131d !important;
    }
    div[data-testid="stColumn"]:has(.cstf-agent-view-history) [data-testid="stLayoutWrapper"]:has(.cstf-history-list-marker) > [data-testid="stVerticalBlock"] {
        border: 0 !important;
        border-radius: 0 !important;
        background: transparent !important;
        box-shadow: none !important;
    }
    div[data-testid="stColumn"]:has(.cstf-agent-view-history) [data-testid="stLayoutWrapper"]:has(.cstf-history-list-marker) > [data-testid="stVerticalBlock"] {
        min-height: 0 !important;
    }
    div[data-testid="stColumn"]:has(.cstf-agent-view-history) [data-testid="stForm"] input[aria-label="chat_input"] {
        min-height: 2.4rem !important;
    }
    /* 历史页是纯会话导航：只展示记录，进入具体会话后再显示聊天流。 */
    /* 直接按 stForm 隐藏，避免 aria-label 随浏览器语言被翻译后漏出发送框。 */
    div[data-testid="stColumn"]:has(.cstf-agent-view-history) [data-testid="stForm"],
    div[data-testid="stColumn"]:has(.cstf-agent-view-history) [data-testid="stForm"]:has(input[aria-label="chat_input"]),
    div[data-testid="stColumn"]:has(.cstf-agent-view-history) > div[data-testid="stVerticalBlock"] > [data-testid="stLayoutWrapper"]:has([data-testid="stChatMessage"]),
    /* 空历史时也彻底移除聊天流的外层布局包装，避免留下空白行。 */
    div[data-testid="stColumn"]:has(.cstf-agent-view-history) [data-testid="stLayoutWrapper"]:has(.cstf-chat-stream-marker),
    div[data-testid="stColumn"]:has(.cstf-agent-view-history) [data-testid="stChatMessage"],
    div[data-testid="stColumn"]:has(.cstf-agent-view-history) [data-testid="stElementContainer"]:has([data-testid="stVerticalBlockBorderWrapper"] [data-testid="stChatMessage"]),
    div[data-testid="stColumn"]:has(.cstf-agent-view-history) [data-testid="stVerticalBlockBorderWrapper"]:has([data-testid="stChatMessage"]),
    /* 空会话时，聊天容器没有 stChatMessage；用专属标记仍将其移除。 */
    div[data-testid="stColumn"]:has(.cstf-agent-view-history) [data-testid="stVerticalBlockBorderWrapper"]:has(.cstf-chat-stream-marker) {
        display: none !important;
        height: 0 !important;
        min-height: 0 !important;
        margin: 0 !important;
        padding: 0 !important;
        overflow: hidden !important;
    }
    .header-badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 600;
        border: 1px solid #3d3d3d;
        background: #1b1b1d;
        color: #aaa !important;
    }
    .header-badge-running {
        border-color: #3A62D7;
        color: #8ab4ff !important;
        background: #1a2540;
    }
</style>
""", unsafe_allow_html=True)

# =======================================================
#  2. 侧边栏：任务管理中心
# =======================================================
with st.sidebar:
    sbui.inject_sidebar_css()
    map_display_path = None

    _wf_options = ["潮滩推理", "GEE 数据下载"]
    if st.session_state.ui_workflow not in _wf_options:
        st.session_state.ui_workflow = "潮滩推理"
    workflow = st.radio(
        "工作台",
        _wf_options,
        format_func=lambda x: "潮滩智能提取" if x == "潮滩推理" else "获取卫星影像",
        horizontal=True,
        key="ui_workflow",
        help="潮滩智能提取：用模型或指数法从本地影像提取潮滩；获取卫星影像：从影像平台筛选并导出 Sentinel-2 数据。",
    )
    use_gee_download = st.session_state.ui_workflow == "GEE 数据下载"

    sbui.section("任务与数据")
    with st.container(border=True):
        root_dir = st.text_input("原始影像目录", key="ui_root_dir")

        task_options = []
        if os.path.exists(root_dir):
            sub_dirs = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
            task_options = sorted(sub_dirs)

        if not task_options:
            sbui.hint("未发现可用任务", "warn")
            selected_task = None
        else:
            _sel = st.session_state.get("ui_selected_task")
            _task_opts = list(task_options)
            if _sel and _sel not in _task_opts:
                _task_opts = [_sel] + _task_opts
            if st.session_state.get("ui_selected_task") not in _task_opts:
                st.session_state.ui_selected_task = task_options[0]
            selected_task = st.selectbox("目标任务", options=_task_opts, key="ui_selected_task")
            if st.session_state.get("_last_selected_task") != selected_task:
                st.session_state._asset_pinned = False
                st.session_state.asset_override = None
                st.session_state._map_view_synced_for = None
            st.session_state._last_selected_task = selected_task
            sbui.hint(f"当前任务 · {selected_task}", "ok")

    _default_aoi = r"E:\Data\CHINA_tf_city\china_costal.shp"

    def _ui_to_date(key: str, fallback: datetime.date) -> datetime.date:
        v = st.session_state.get(key, fallback)
        if isinstance(v, datetime.date):
            return v
        try:
            return datetime.datetime.strptime(str(v)[:10], "%Y-%m-%d").date()
        except (ValueError, TypeError):
            return fallback

    st.session_state.ui_m4_start_date = _ui_to_date("ui_m4_start_date", datetime.date(2020, 1, 1))
    st.session_state.ui_m4_end_date = _ui_to_date("ui_m4_end_date", datetime.date(2020, 1, 31))

    m4_roi_path = st.session_state.get("ui_m4_roi_path") or _default_aoi
    m4_roi_name = st.session_state.get("ui_m4_roi_name") or ""
    m4_start_date = st.session_state.ui_m4_start_date
    m4_end_date = st.session_state.ui_m4_end_date
    m4_export_to = st.session_state.get("ui_m4_export_to") or "drive"
    m4_drive_folder = st.session_state.get("ui_m4_drive_folder") or (selected_task or "GEE_Downloads")
    m4_local_dir = st.session_state.get("ui_m4_local_dir") or (
        os.path.join(root_dir, m4_drive_folder) if selected_task else root_dir
    )
    m4_cloud = int(st.session_state.get("ui_m4_cloud_limit") or 60)
    m4_min_land = float(st.session_state.get("ui_m4_min_land") or 5.0)
    m4_max_land = float(st.session_state.get("ui_m4_max_land") or 95.0)
    m4_min_pix = int(st.session_state.get("ui_m4_min_pixel_count") or 1000)
    m4_bands = list(st.session_state.get("ui_m4_bands") or ["B8", "B4", "B3", "B2", "B11"])
    m4_scale = int(st.session_state.get("ui_m4_scale") or 10)
    m4_gee_proxy = st.session_state.get("ui_m4_gee_proxy") or ""
    m4_gee_project = (st.session_state.get("ui_m4_gee_project") or os.environ.get("EE_PROJECT", "")).strip()

    if use_gee_download:
        sbui.section("获取卫星影像")
        with st.expander("影像筛选参数", expanded=True):
            m4_roi_path = st.text_input("研究区域矢量 (.shp)", key="ui_m4_roi_path")
            _roi_names = []
            try:
                import m4_engine as _m4e
                _roi_names = _m4e.list_roi_names(m4_roi_path)
            except Exception:
                pass
            if _roi_names:
                _def_roi = selected_task if selected_task in _roi_names else _roi_names[0]
                m4_roi_name = st.selectbox("研究区域名称 (name 字段)", _roi_names, index=_roi_names.index(_def_roi) if _def_roi in _roi_names else 0)
            else:
                m4_roi_name = st.text_input("研究区域名称 (name 字段)", key="ui_m4_roi_name", placeholder=selected_task or "zhejiang1")
            _c1, _c2 = st.columns(2)
            with _c1:
                m4_start_date = st.date_input("开始日期", key="ui_m4_start_date")
            with _c2:
                m4_end_date = st.date_input("结束日期", key="ui_m4_end_date")
            if m4_end_date < m4_start_date:
                st.error("结束日期不能早于开始日期")
            else:
                _span_days = (m4_end_date - m4_start_date).days + 1
                if _span_days > 31:
                    st.caption(f"已选 {_span_days} 天，将自动按月分批筛选影像，避免单次查询超时。")
            m4_export_to = st.radio(
                "导出方式",
                ["drive", "local"],
                format_func=lambda x: "Google Drive" if x == "drive" else "本机直链",
                horizontal=True,
                key="ui_m4_export_to",
            )
            m4_drive_folder = st.text_input("云端文件夹 / 任务子目录名", key="ui_m4_drive_folder")
            if m4_export_to == "local":
                m4_local_dir = st.text_input("本地下载目录", key="ui_m4_local_dir")
            else:
                if not st.session_state.get("ui_m4_local_dir"):
                    st.session_state.ui_m4_local_dir = os.path.join(root_dir, m4_drive_folder)
                m4_local_dir = st.session_state.ui_m4_local_dir
                st.caption(f"本地提取目录建议：`{os.path.join(root_dir, m4_drive_folder)}`（云端同步后放此处）")
            m4_bands = st.multiselect(
                "导出波段",
                ["B8", "B4", "B3", "B2", "B11", "B8A", "B5", "B6", "B7", "B12"],
                key="ui_m4_bands",
            )
            m4_cloud = st.slider("云量上限 (%)", 0, 100, key="ui_m4_cloud_limit")
            _lc1, _lc2 = st.columns(2)
            with _lc1:
                m4_min_land = st.number_input("最小陆地占比 (%)", 0.0, 100.0, key="ui_m4_min_land", step=0.5)
            with _lc2:
                m4_max_land = st.number_input("最大陆地占比 (%)", 0.0, 100.0, key="ui_m4_max_land", step=0.5)
            m4_min_pix = st.number_input("最小有效像素数", 100, 500000, key="ui_m4_min_pixel_count", step=100)
            _scale_opts = [10, 20, 30]
            if st.session_state.get("ui_m4_scale") not in _scale_opts:
                st.session_state.ui_m4_scale = 10
            m4_scale = st.selectbox("导出分辨率 (m)", _scale_opts, key="ui_m4_scale")
            m4_gee_proxy = st.text_input(
                "影像平台网络代理 (可选)",
                key="ui_m4_gee_proxy",
                placeholder=DEFAULT_CLASH_PROXY,
                help="Clash 混合代理端口（默认 7892），与 Clash 设置保持一致。",
            )
            m4_gee_project = st.text_input(
                "影像平台项目 ID（必填）",
                key="ui_m4_gee_project",
                placeholder="例如 ee-yourname 或 GCP 项目名",
                help="在 https://code.earthengine.google.com 登录后，右上角可见；"
                "或终端执行 earthengine set_project 项目ID 后填同一 ID。",
            )
        _m4_last = st.session_state.get("m4_last_result")
        if _m4_last:
            st.info(
                f"上次：{ _m4_last.get('roi_name')} · {_m4_last.get('image_count')} 景 · "
                f"{ 'Drive/' + str(_m4_last.get('drive_folder')) if _m4_last.get('export_to') == 'drive' else _m4_last.get('local_out_dir') }"
            )

    with st.expander("路径与模型环境", expanded=False):
        mask_root = st.text_input("预测掩膜根目录 (Mask)", key="ui_mask_root")
        final_root = st.text_input("最终合成根目录 (Output)", key="ui_final_root")
        if selected_task:
            task_mask_dir = os.path.join(mask_root, selected_task)
            task_final_dir = os.path.join(final_root, selected_task)
            st.text_input("当前任务 Mask", task_mask_dir, disabled=True)
            st.text_input("当前任务 Final", task_final_dir, disabled=True)
        else:
            task_mask_dir, task_final_dir = "", ""

        model_path = st.text_input("提取模型权重 (.pth)", key="ui_model_path")
        shp_path = st.text_input("岸线约束矢量 (.shp)", key="ui_shp_path")
        points_shp = st.text_input(
            "海洋种子点 (.shp，指数法)",
            key="ui_points_shp",
            help="用于从水体中筛选真实海洋面，需落在海水上的点要素。",
        )
        task_aoi_shp = st.text_input(
            "任务分区研究区域（裁剪参考真值，用于指标）",
            key="ui_task_aoi_shp",
            help="与侧栏「目标任务」同名的要素用于裁剪参考真值，再与预测比交并比/F1；自适应与合成阶段一致。文件不存在则跳过裁剪。",
        )

    if not use_gee_download:
        with st.expander("提取参数", expanded=False):
            _im_opts = ["深度学习", "指数法"]
            if st.session_state.get("ui_inference_mode") not in _im_opts:
                st.session_state.ui_inference_mode = "深度学习"
            inference_mode = st.radio(
                "提取方式",
                _im_opts,
                horizontal=True,
                key="ui_inference_mode",
                help="深度学习：模型逐景掩膜 + 时空合成；指数法：mNDWI 海面 + ACWI 频率 + 空间交集。",
            )
            use_index_mode = st.session_state.ui_inference_mode == "指数法"
            adaptive_mode = st.checkbox(
                "参数自动优化",
                key="ui_adaptive_mode",
                disabled=use_index_mode,
                help="自动搜索最优 (提取概率阈值, 最少有效影像次数)，使合成图与参考真值的交并比 / F1 最优。",
            )
            if use_index_mode:
                adaptive_mode = False
                prob_th, min_cnt = 0.05, 2
                st.caption("指数法输出 `{任务}_Index_Final.tif`；深度学习输出 `{任务}_Final_p*.shp`。")
            elif adaptive_mode:
                prob_th = 0.05
                min_cnt = 2
            else:
                prob_th = st.slider("提取概率阈值", 0.01, 0.50, step=0.01, key="ui_prob_th")
                min_cnt = st.slider("最少有效影像次数", 1, 10, step=1, key="ui_min_cnt")

        with st.expander("成果分析", expanded=False):
            m5_enabled = st.checkbox(
                "潮滩变化分析",
                key="ui_m5_enabled",
                help="合成完成后对比往年同区域潮滩，输出变化告警。",
            )
            m5_baseline_shp = st.text_input(
                "历史对比成果 SHP（可选）",
                key="ui_m5_baseline_shp",
                placeholder="留空自动匹配往年成果",
                disabled=not m5_enabled,
            )
            e1_enabled = st.checkbox(
                "潮滩精度评价",
                key="ui_e1_enabled",
                help="与开源潮滩产品做像元级对比，输出交并比、分歧图与成因分析。",
            )
            e1_data_root = st.text_input(
                "参考数据根目录",
                key="ui_e1_data_root",
                disabled=not e1_enabled,
            )
            _e1_ref_options = ["师姐_2020", "师姐_2022", "师姐_2024", "师姐_2025"]
            if st.session_state.get("ui_e1_reference") not in _e1_ref_options:
                st.session_state.ui_e1_reference = _e1_ref_options[0]
            e1_reference = st.selectbox(
                "参考数据",
                _e1_ref_options,
                key="ui_e1_reference",
                disabled=not e1_enabled,
            )
            _e1_default_compare = [
                "DCTF_2020", "FCS30_2020", "GTF30_2020", "CHN_2024",
                "MTWM_2020", "TFMC_2020", "national_10m_2020",
            ]
            if e1_enabled:
                try:
                    import e1_engine as _e1e

                    _e1_all = _e1e.list_e1_datasets(e1_data_root)
                    _e1_choices = [d for d in _e1_all if d != e1_reference and d not in _e1e._SKIP_COMPARE]
                    e1_compare_sources = st.multiselect(
                        "对比数据",
                        _e1_choices,
                        default=[d for d in _e1_default_compare if d in _e1_choices],
                    )
                except Exception:
                    e1_compare_sources = st.multiselect(
                        "对比数据",
                        _e1_default_compare,
                        default=_e1_default_compare,
                    )
                e1_export_maps = st.checkbox("导出分歧 GeoTIFF", key="ui_e1_export_maps")
                e1_export_heatmap = st.checkbox("导出一致热力图", key="ui_e1_export_heatmap")
            else:
                e1_compare_sources = _e1_default_compare
                e1_export_maps = True
                e1_export_heatmap = True
    else:
        use_index_mode = False
        adaptive_mode = False
        prob_th, min_cnt = 0.05, 2
        m5_enabled = False
        m5_baseline_shp = ""
        e1_enabled = False
        e1_data_root = r"E:\潮滩数据集"
        e1_reference = "师姐_2020"
        e1_compare_sources = []
        e1_export_maps = True
        e1_export_heatmap = True

    cache_hit = None
    force_rerun = bool(st.session_state.get("ui_force_rerun", False))
    tune_btn = False
    run_btn = False
    m4_run_btn = False

    _autotune_ready = False
    _ref_id = None
    _tune_objective = "iou_f1"

    if adaptive_mode and selected_task and not use_gee_download:
        with st.expander("参数自动优化配置", expanded=True):
            try:
                from dataset_assets import list_datasets, get_primary_path as _ds_get_path
                _ref_rows = list_datasets(role="reference_truth")
            except Exception:
                _ref_rows = []
            if not _ref_rows:
                st.warning("参考数据中无真值数据，请先登记真值数据集。")
                adaptive_mode = False
            else:
                _ref_opts = {}
                _default_idx = 0
                _task_year = None
                _ym = re.match(r"(\d{2})", selected_task or "")
                if _ym:
                    _task_year = 2000 + int(_ym.group(1))
                for _i, _d in enumerate(_ref_rows):
                    _label = f"{_d.get('title', _d['id'])} ({_d.get('year', '?')})"
                    _ref_opts[_label] = _d["id"]
                    if _task_year and _d.get("year") == _task_year:
                        _default_idx = _i
                _ref_label = st.selectbox("参考真值数据集", list(_ref_opts.keys()), index=_default_idx)
                _ref_id = _ref_opts[_ref_label]
                _obj_label = st.radio(
                    "优化目标",
                    ["交并比 + F1 (均衡)", "交并比 (优先)", "F1 (精确-召回优先)"],
                    horizontal=True,
                )
                _tune_objective = {"交并比 + F1 (均衡)": "iou_f1", "交并比 (优先)": "iou", "F1 (精确-召回优先)": "f1"}[_obj_label]

                _task_mask_check = os.path.join(mask_root, selected_task) if selected_task else ""
                _mask_count = len(glob.glob(os.path.join(_task_mask_check, "**", "*_mask.tif"), recursive=True)) if os.path.isdir(_task_mask_check) else 0
                if _mask_count == 0:
                    sbui.hint("尚无 Mask，请先运行提取", "warn")
                else:
                    sbui.hint(f"可优化 · {_mask_count} 个 Mask", "ok")
                    _autotune_ready = True

                st.caption("prob ∈ [0.01, 0.50] × cnt ∈ [1, 10] 自动搜索最优组合")

    if selected_task:
        if use_gee_download:
            final_tif_path = ""
        elif use_index_mode:
            final_tif_path = os.path.join(task_final_dir, f"{selected_task}_Index_Final.tif")
        else:
            final_tif_path = os.path.join(task_final_dir, f"{selected_task}_Final_p{prob_th:.2f}_c{min_cnt}.shp")
    else:
        final_tif_path = ""

    # --- 首次启动时扫描已有产出并注册到资产库 ---
    if not st.session_state.assets_scanned:
        scan_and_register_existing(final_root)
        st.session_state.assets_scanned = True

    map_display_path = None
    if not use_gee_download:
        # --- 参数变更时清除手动加载的资产覆盖 ---
        if use_index_mode:
            current_param_key = f"{selected_task}_index"
        else:
            current_param_key = f"{selected_task}_p{prob_th:.2f}_c{min_cnt}"
        if st.session_state._param_key is not None and st.session_state._param_key != current_param_key:
            if not st.session_state.get("_asset_pinned"):
                st.session_state.asset_override = None
        st.session_state._param_key = current_param_key

        # --- 资产缓存检测 ---
        if selected_task:
            cache_hit = find_index_asset(selected_task) if use_index_mode else find_asset(selected_task, prob_th, min_cnt)
        else:
            cache_hit = None

        if st.session_state.asset_override and os.path.exists(st.session_state.asset_override):
            map_display_path = st.session_state.asset_override
        elif cache_hit:
            map_display_path = cache_hit["file_path"]
        elif final_tif_path and os.path.exists(final_tif_path):
            map_display_path = final_tif_path

        sbui.section("成果管理")
        with st.container(border=True):
            if cache_hit:
                sbui.hint(f"缓存命中 · {cache_hit['file_size_mb']} MB · {cache_hit['created_at']}", "ok")
            elif selected_task:
                sbui.hint("暂无缓存，运行提取后生成")

            st.slider(
                "图层透明度",
                min_value=5,
                max_value=100,
                step=5,
                key="result_overlay_opacity_pct",
            )

            if selected_task:
                task_assets = get_task_assets(selected_task)
                if task_assets:
                    with st.expander(f"历史成果 ({len(task_assets)})", expanded=False):
                        for key, asset in task_assets.items():
                            a_cols = st.columns([5, 2])
                            with a_cols[0]:
                                if asset.get("method") == "index":
                                    _lbl = f"指数 · {asset['created_at']} · {asset['file_size_mb']}MB"
                                else:
                                    _lbl = (
                                        f"P={asset['prob_threshold']} C={asset['min_count']} "
                                        f"· {asset['created_at']} · {asset['file_size_mb']}MB"
                                    )
                                st.caption(_lbl)
                            with a_cols[1]:
                                if st.button("加载", key=f"load_{key}", use_container_width=True):
                                    st.session_state.asset_override = asset["file_path"]
                                    st.session_state._asset_pinned = True
                                    st.session_state._map_view_synced_for = None
                                    st.session_state.asset_just_loaded = True
                                    st.session_state._globe_rev = int(st.session_state.get("_globe_rev", 0)) + 1
                                    st.rerun()

            if cache_hit and not adaptive_mode:
                force_rerun = st.checkbox("强制重新生成", key="ui_force_rerun", help="忽略缓存，重新运行提取。")
    elif st.session_state.asset_override and os.path.exists(st.session_state.asset_override):
        map_display_path = st.session_state.asset_override

    # --- 自适应优化历史结果 ---
    _at_res = st.session_state.get("autotune_result")
    if _at_res:
        sbui.section("参数自动优化结果")
        with st.container(border=True):
            st.caption(f"最优概率 P={_at_res['best_prob']:.2f} · 次数 C={_at_res['best_cnt']}")
            _mc1, _mc2 = st.columns(2)
            with _mc1:
                st.metric("交并比 (IoU)", f"{_at_res['best_iou'] * 100:.1f}%")
            with _mc2:
                st.metric("F1 综合评分", f"{_at_res['best_f1'] * 100:.1f}%")
            st.caption(
                f"精确率 {_at_res['best_precision'] * 100:.1f}% · 召回率 {_at_res['best_recall'] * 100:.1f}% · "
                f"{_at_res['total_trials']} 组 · {_at_res['total_time_sec']:.0f}s"
            )
            _at_trials = _at_res.get("trials") or []
            if _at_trials:
                _sorted_t = sorted(_at_trials, key=lambda t: t["score"], reverse=True)
                with st.expander(f"Top-10 ({len(_sorted_t)})", expanded=False):
                    import pandas as pd
                    _top = _sorted_t[:10]
                    st.dataframe(
                        pd.DataFrame({
                            "#": range(1, len(_top) + 1),
                            "概率": [t["prob"] for t in _top],
                            "次数": [t["cnt"] for t in _top],
                            "交并比%": [round(t["iou"] * 100, 2) for t in _top],
                            "F1%": [round(t["f1"] * 100, 2) for t in _top],
                        }),
                        use_container_width=True,
                        hide_index=True,
                    )
            if st.button("清除参数优化结果", key="clear_autotune_result"):
                st.session_state.pop("autotune_result", None)
                st.rerun()

    if selected_task and final_root and not st.session_state.is_running:
        try:
            import m5_engine as _m5e
            _disk_m5 = _m5e.load_m5_report(final_root, selected_task)
            _cur_m5 = st.session_state.get("m5_report")
            if _disk_m5 and (not _cur_m5 or _cur_m5.get("target_roi") != selected_task):
                st.session_state.m5_report = _disk_m5
        except Exception:
            pass

    # 独立 M5 预检入口（不经 LLM，便于验收与无模型时使用）
    if selected_task and final_root and not st.session_state.is_running:
        if st.button("预检并生成变化分析计划", key="propose_m5_manual_btn", use_container_width=True):
            queue_agent_command(
                st.session_state,
                {
                    "sidebar_states": {
                        "m5_enabled": True,
                        "selected_task": selected_task,
                        "final_root": final_root,
                        "root_dir": root_dir,
                    },
                    "pending_action": {"type": "propose_m5", "task": selected_task},
                },
            )
            st.rerun()

    _m5_res = st.session_state.get("m5_report")
    _m5_plan = st.session_state.get("_m5_pending_plan")
    if isinstance(_m5_plan, dict) and not st.session_state.is_running:
        sbui.section("潮滩变化分析计划")
        with st.container(border=True):
            if _m5_plan.get("ready"):
                st.success("条件已满足，确认后将运行变化分析")
            else:
                st.warning("条件未满足，暂不可执行")
                for _b in _m5_plan.get("blockers") or []:
                    st.caption(f"· {_b}")
            st.caption(
                f"当前 `{_m5_plan.get('current_task') or '—'}` · "
                f"历史对比成果 `{_m5_plan.get('baseline_task') or '—'}` · "
                f"可用时期 {len(_m5_plan.get('available_periods') or [])}"
            )
            _pc1, _pc2 = st.columns(2)
            with _pc1:
                if st.button(
                    "确认执行变化分析",
                    key="confirm_m5_plan_btn",
                    type="primary",
                    use_container_width=True,
                    disabled=not bool(_m5_plan.get("ready")),
                ):
                    queue_agent_command(
                        st.session_state,
                        {"pending_action": {"type": "run_m5", "confirmed": True}},
                    )
                    st.rerun()
            with _pc2:
                if st.button("取消计划", key="cancel_m5_plan_btn", use_container_width=True):
                    st.session_state.pop("_m5_pending_plan", None)
                    st.session_state.pop("_m5_plan_confirmed", None)
                    st.session_state.pop("_m5_plan_notice", None)
                    st.rerun()

    _e1_plan = st.session_state.get("_e1_pending_plan")
    if isinstance(_e1_plan, dict) and not st.session_state.is_running:
        sbui.section("潮滩精度评价计划")
        with st.container(border=True):
            if _e1_plan.get("ready"):
                st.success("条件已满足，确认后将运行精度评价")
            else:
                st.warning("条件未满足，暂不可执行")
                for _b in _e1_plan.get("blockers") or []:
                    st.caption(f"· {_b}")
            st.caption(
                f"当前 `{_e1_plan.get('current_task') or '—'}` · "
                f"参考数据 `{_e1_plan.get('reference') or '—'}`"
            )
            _ec1, _ec2 = st.columns(2)
            with _ec1:
                if st.button(
                    "确认执行精度评价",
                    key="confirm_e1_plan_btn",
                    type="primary",
                    use_container_width=True,
                    disabled=not bool(_e1_plan.get("ready")),
                ):
                    queue_agent_command(
                        st.session_state,
                        {"pending_action": {"type": "run_e1", "confirmed": True}},
                    )
                    st.rerun()
            with _ec2:
                if st.button("取消精度评价计划", key="cancel_e1_plan_btn", use_container_width=True):
                    st.session_state.pop("_e1_pending_plan", None)
                    st.session_state.pop("_e1_plan_confirmed", None)
                    st.session_state.pop("_e1_plan_notice", None)
                    st.rerun()

    # 潮滩智能提取执行计划（可信执行闭环：先计划后确认）
    _inf_plan = st.session_state.get("_inference_pending_plan")
    if isinstance(_inf_plan, dict) and not st.session_state.is_running:
        sbui.section("潮滩智能提取计划")
        with st.container(border=True):
            if _inf_plan.get("ready"):
                st.success("条件已满足，确认后将真实调用提取/成果生成代码")
            else:
                st.warning("条件未满足，暂不可执行")
                for _b in _inf_plan.get("blockers") or []:
                    st.caption(f"· {_b}")
            st.caption(
                f"任务 `{_inf_plan.get('task_id') or '—'}` · "
                f"概率 P={_inf_plan.get('prob_threshold')} 次数 C={_inf_plan.get('count_threshold')} · "
                f"设备策略 `{_inf_plan.get('device_policy') or 'auto'}`"
                + (f"（实际 `{_inf_plan.get('device')}`）" if _inf_plan.get("device") else "")
            )
            _infc1, _infc2 = st.columns(2)
            with _infc1:
                if st.button(
                    "确认执行提取",
                    key="confirm_inference_plan_btn",
                    type="primary",
                    use_container_width=True,
                    disabled=not bool(_inf_plan.get("ready")),
                ):
                    from agent_command_bridge import confirm_inference_plan as _bridge_confirm_inf

                    _pid = _inf_plan.get("plan_id")
                    _ok, _cerr = _bridge_confirm_inf(st.session_state, str(_pid))
                    if not _ok:
                        st.warning(_cerr or "确认失败，请重新生成计划。")
                        st.rerun()
                    queue_agent_command(
                        st.session_state,
                        {"pending_action": {"type": "run_inference", "confirmed": True,
                                            "task": _inf_plan.get("task_id"), "plan_id": _pid}},
                    )
                    st.rerun()
            with _infc2:
                if st.button("取消计划", key="cancel_inference_plan_btn", use_container_width=True):
                    st.session_state.pop("_inference_pending_plan", None)
                    st.session_state.pop("_inference_plan_confirmed", None)
                    st.session_state.pop("_inference_plan_notice", None)
                    st.rerun()

    # 指数法执行计划（与深度学习共用先计划后确认的 UI 语义）
    _index_plan = st.session_state.get("_index_pending_plan")
    if isinstance(_index_plan, dict) and not st.session_state.is_running:
        sbui.section("指数法潮滩提取计划")
        with st.container(border=True):
            if _index_plan.get("ready"):
                st.success("条件已满足，确认后将调用指数法执行适配器")
            else:
                st.warning("指数法计划暂不可执行")
                for _b in _index_plan.get("blockers") or []:
                    st.caption(f"· {_b}")
            st.caption(
                f"任务 `{_index_plan.get('task') or '—'}` · "
                f"输入影像目录已校验 · 海洋种子点已校验"
            )
            _ixc1, _ixc2 = st.columns(2)
            with _ixc1:
                if st.button(
                    "确认执行指数法",
                    key="confirm_index_plan_btn",
                    type="primary",
                    use_container_width=True,
                    disabled=not bool(_index_plan.get("ready")),
                ):
                    _tl_add(
                        _index_plan.get("task") or "unknown",
                        "QUEUED",
                        "指数法计划已确认并入队",
                        status="QUEUED",
                        tool="index_agent_loop",
                    )
                    st.session_state.pending_task = {
                        "task": _index_plan.get("task"),
                        "mode": "index",
                        "points_shp": _index_plan.get("points_shp"),
                        "force_rerun": bool(_index_plan.get("force_rerun")),
                        "index_plan": dict(_index_plan),
                    }
                    st.session_state.is_running = True
                    st.session_state.stop_requested = False
                    st.session_state.pop("_index_pending_plan", None)
                    st.rerun()
            with _ixc2:
                if st.button("取消指数法计划", key="cancel_index_plan_btn", use_container_width=True):
                    st.session_state.pop("_index_pending_plan", None)
                    st.rerun()

    # 获取卫星影像执行计划（可信执行闭环：先计划后确认）
    _gee_plan = st.session_state.get("_gee_pending_plan")
    if isinstance(_gee_plan, dict) and not st.session_state.is_running:
        sbui.section("获取卫星影像计划")
        with st.container(border=True):
            if _gee_plan.get("ready"):
                st.success("条件已满足，确认后将真实下载卫星影像（不自动启动提取）")
            else:
                st.warning("条件未满足，暂不可执行")
                for _b in _gee_plan.get("blockers") or []:
                    st.caption(f"· {_b}")
            st.caption(
                f"任务 `{_gee_plan.get('task_id') or '—'}` · "
                f"{_gee_plan.get('start_date') or '—'} → {_gee_plan.get('end_date') or '—'} · "
                f"波段 {','.join((_gee_plan.get('bands') or ['B4','B3','B2']))} · "
                f"导出 `{_gee_plan.get('export_to')}`"
            )
            _gc1, _gc2 = st.columns(2)
            with _gc1:
                if st.button(
                    "确认下载影像",
                    key="confirm_gee_plan_btn",
                    type="primary",
                    use_container_width=True,
                    disabled=not bool(_gee_plan.get("ready")),
                ):
                    from agent_command_bridge import confirm_gee_plan as _bridge_confirm_gee

                    _gpid = _gee_plan.get("plan_id")
                    _gok, _gcerr = _bridge_confirm_gee(st.session_state, str(_gpid))
                    if not _gok:
                        st.warning(_gcerr or "确认失败，请重新生成计划。")
                        st.rerun()
                    queue_agent_command(
                        st.session_state,
                        {"pending_action": {"type": "run_gee_download", "confirmed": True,
                                            "task": _gee_plan.get("task_id"), "plan_id": _gpid}},
                    )
                    st.rerun()
            with _gc2:
                if st.button("取消计划", key="cancel_gee_plan_btn", use_container_width=True):
                    st.session_state.pop("_gee_pending_plan", None)
                    st.session_state.pop("_gee_plan_confirmed", None)
                    st.session_state.pop("_gee_plan_notice", None)
                    st.rerun()

    # 端到端一键潮滩分析：先计划后确认（父级确认门闩）
    _wf_plan = st.session_state.get("_workflow_pending_plan")
    if isinstance(_wf_plan, dict) and not st.session_state.is_running:
        sbui.section("一键潮滩分析")
        with st.container(border=True):
            import workflow_orchestrator as _wo

            _wf_id = str(_wf_plan.get("workflow_id") or "")
            _wf_confirmed = _wo.is_workflow_confirmed(st.session_state, _wf_id)
            _wf_status = str(_wf_plan.get("status") or "PENDING")
            if _wf_status == "PAUSED":
                st.warning("参数已变化，需重新确认后执行")
            elif _wf_confirmed:
                st.success(f"已确认 · `{_wf_id[:12]}…`")
            else:
                st.info(f"待确认 · `{_wf_id[:12]}…`")
            _wf_blockers = _wf_plan.get("blockers") or []
            if _wf_blockers:
                st.warning("全局校验未通过，暂不可执行")
                for _b in _wf_blockers:
                    st.caption(f"· {_b}")
            st.caption(
                f"任务 `{_wf_plan.get('task_id') or '—'}` · "
                f"{(_wf_plan.get('context') or {}).get('target_year')} 年潮滩"
                + (f" · 历史对比 {(_wf_plan.get('context') or {}).get('baseline_year')}" if (_wf_plan.get('context') or {}).get('baseline_year') else "")
            )
            _wf_steps = _wf_plan.get("steps") or []
            st.markdown(
                "\n".join(
                    f"- {'必' if s.get('required') else '选'} · "
                    f"{uil.get_tool_label(s.get('tool'))}"
                    f"（{uil.get_status_label(s.get('status') or 'PENDING')}）"
                    for s in _wf_steps
                )
            )
            _wfc1, _wfc2 = st.columns(2)
            with _wfc1:
                if st.button(
                    "确认执行一键分析",
                    key="confirm_workflow_btn",
                    type="primary",
                    use_container_width=True,
                    disabled=bool(_wf_blockers) or _wf_confirmed,
                ):
                    _ok, _cerr = _wo.confirm_workflow(st.session_state, _wf_id)
                    if not _ok:
                        st.warning(_cerr or "一键分析确认失败。")
                        st.rerun()
                    queue_agent_command(
                        st.session_state,
                        {"pending_action": {"type": "run_workflow", "confirmed": True,
                                            "workflow_id": _wf_id,
                                            "task": _wf_plan.get("task_id")}},
                    )
                    st.rerun()
            with _wfc2:
                if st.button("取消一键分析", key="cancel_workflow_btn", use_container_width=True):
                    st.session_state.pop("_workflow_pending_plan", None)
                    st.session_state.pop("_workflow_plan_confirmed", None)
                    st.session_state.pop("_workflow_notice", None)
                    st.rerun()

    # 侧栏 AutoTune 计划确认：不能因按钮来自本地 UI 就绕过重型操作门闩。
    _autotune_plan = st.session_state.get("_autotune_pending_plan")
    if isinstance(_autotune_plan, dict) and not st.session_state.is_running:
        sbui.section("参数自动优化计划")
        with st.container(border=True):
            _at_plan_valid = (
                bool(adaptive_mode)
                and str(_autotune_plan.get("task") or "") == str(selected_task or "")
                and str(_autotune_plan.get("reference_id") or "") == str(_ref_id or "")
            )
            st.info(
                f"任务 `{_autotune_plan.get('task') or '—'}` · "
                f"参考真值 `{_autotune_plan.get('reference_id') or '—'}` · "
                f"目标 `{_autotune_plan.get('objective') or '—'}`"
            )
            if not _at_plan_valid:
                st.warning("当前侧栏参数已变化，请取消旧计划后重新生成。")
            st.caption("确认后才会搜索参数组合，并可能执行后置 M5/E1 评价。")
            _atc1, _atc2 = st.columns(2)
            with _atc1:
                if st.button(
                    "确认执行参数优化",
                    key="confirm_autotune_plan_btn",
                    type="primary",
                    use_container_width=True,
                    disabled=not _at_plan_valid,
                ):
                    from execution_request import attach_execution_request

                    _at_pending = attach_execution_request(
                        dict(_autotune_plan), confirmation_source="ui"
                    )
                    st.session_state.pending_autotune = _at_pending
                    st.session_state.is_running = True
                    st.session_state.stop_requested = False
                    _tl_add(
                        str(_autotune_plan.get("task") or "unknown"),
                        "CONFIRM",
                        "参数优化计划已确认",
                        status="SUCCEEDED",
                        plan_id=_autotune_plan.get("plan_id"),
                        tool="run_autotune",
                    )
                    _tl_add(
                        str(_autotune_plan.get("task") or "unknown"),
                        "QUEUED",
                        "参数优化任务已入队",
                        status="QUEUED",
                        plan_id=_autotune_plan.get("plan_id"),
                        tool="run_autotune",
                    )
                    st.session_state.pop("_autotune_pending_plan", None)
                    st.rerun()
            with _atc2:
                if st.button("取消参数优化计划", key="cancel_autotune_plan_btn", use_container_width=True):
                    _tl_add(
                        str(_autotune_plan.get("task") or "unknown"),
                        "REPORT",
                        "参数优化计划已取消",
                        status="CANCELLED",
                        plan_id=_autotune_plan.get("plan_id"),
                        tool="run_autotune",
                    )
                    st.session_state.pop("_autotune_pending_plan", None)
                    st.rerun()

    # 重型工具确认门闩：Agent 请求 run_pipeline/run_m4/run_gee_download/run_autotune 未确认时在此待命
    _pending_heavy = st.session_state.get("_pending_heavy_confirm")
    if isinstance(_pending_heavy, dict) and not st.session_state.is_running:
        sbui.section("待确认操作")
        with st.container(border=True):
            _h_label = _pending_heavy.get("label") or _pending_heavy.get("action_type") or "潮滩智能提取"
            _h_task = _pending_heavy.get("task") or "—"
            st.warning(f"Agent 请求执行 **{_h_label}**（任务 `{_h_task}`），需要你确认后才会启动。")
            _hc1, _hc2 = st.columns(2)
            with _hc1:
                if st.button(
                    "确认执行",
                    key="confirm_heavy_btn",
                    type="primary",
                    use_container_width=True,
                ):
                    _orig = dict(_pending_heavy.get("action") or {})
                    _orig["confirmed"] = True
                    queue_agent_command(st.session_state, {"pending_action": _orig})
                    st.session_state.pop("_pending_heavy_confirm", None)
                    st.rerun()
            with _hc2:
                if st.button("取消", key="cancel_heavy_btn", use_container_width=True):
                    st.session_state.pop("_pending_heavy_confirm", None)
                    st.rerun()

    if _m5_res and (not selected_task or _m5_res.get("target_roi") == selected_task):
        sbui.section("潮滩变化分析结果")
        with st.container(border=True):
            _lvl = _m5_res.get("alert_level", "GREEN")
            _msg = _m5_res.get("diagnostic_message", "")
            if _lvl == "RED":
                st.error(_msg)
            elif _lvl == "YELLOW":
                st.warning(_msg)
            else:
                st.success(_msg)
            _qm = _m5_res.get("quantitative_metrics") or {}
            _ae = _qm.get("area_evolution") or {}
            _ct = _qm.get("centroid_trajectory") or {}
            st.caption(
                f"历史对比成果 {_m5_res.get('baseline_task') or '—'} · "
                f"面积 {_ae.get('baseline_area_km2', '?')}→{_ae.get('current_area_km2', '?')} km² "
                f"({_ae.get('change_rate_percentage', '?')}%) · "
                f"漂移 {_ct.get('drift_distance_meters', '?')} m"
            )
            with st.expander("详细指标", expanded=False):
                st.json(_m5_res)
            _mc1, _mc2, _mc3 = st.columns(3)
            _spatial = _m5_res.get("spatial_outputs") or {}
            _loss_p = _spatial.get("loss_shapefile_path")
            _silt_p = _spatial.get("siltation_shapefile_path")
            with _mc1:
                if (
                    _loss_p
                    and str(_loss_p) != "None"
                    and os.path.isfile(str(_loss_p))
                    and st.button("加载变化区域（萎缩）", key="load_m5_loss", use_container_width=True)
                ):
                    st.session_state.asset_override = _loss_p
                    st.session_state._asset_pinned = True
                    st.session_state._map_view_synced_for = None
                    st.session_state._map_prefer_center = False
                    st.session_state.asset_just_loaded = True
                    st.session_state._globe_rev = int(st.session_state.get("_globe_rev", 0)) + 1
                    st.rerun()
            with _mc2:
                if (
                    _silt_p
                    and str(_silt_p) != "None"
                    and os.path.isfile(str(_silt_p))
                    and st.button("加载变化区域（淤积）", key="load_m5_silt", use_container_width=True)
                ):
                    st.session_state.asset_override = _silt_p
                    st.session_state._asset_pinned = True
                    st.session_state._map_view_synced_for = None
                    st.session_state._map_prefer_center = False
                    st.session_state.asset_just_loaded = True
                    st.session_state._globe_rev = int(st.session_state.get("_globe_rev", 0)) + 1
                    st.rerun()
            with _mc3:
                if st.button("清除变化分析结果", key="clear_m5_report", use_container_width=True):
                    st.session_state.pop("m5_report", None)
                    st.rerun()

    if selected_task and final_root and not st.session_state.is_running:
        try:
            import e1_engine as _e1e
            _e1_ws = _e1e.workspace_for_task(final_root, selected_task)
            _disk_e1 = _e1e.load_e1_report(_e1_ws, selected_task)
            _cur_e1 = st.session_state.get("e1_report")
            if _disk_e1 and (not _cur_e1 or _cur_e1.get("roi_name") != selected_task):
                st.session_state.e1_report = _disk_e1
        except Exception:
            pass

    _e1_res = st.session_state.get("e1_report")
    if _e1_res and (not selected_task or _e1_res.get("roi_name") == selected_task):
        sbui.section("潮滩精度评价结果")
        with st.container(border=True):
            _comps = _e1_res.get("comparisons") or {}
            _rows = []
            _heat_path = None
            for _pair, _m in _comps.items():
                if "error" in _m:
                    _rows.append({"对比数据": _pair, "交并比 (IoU)": "ERR", "交集 km²": "-"})
                    continue
                _rows.append({
                    "对比数据": _pair,
                    "交并比 (IoU)": _m.get("jaccard_iou", "-"),
                    "交集 km²": _m.get("intersection_km2", "-"),
                })
                _maps = (_m.get("causal_analysis") or {}).get("disagreement_maps") or {}
                if not _heat_path and _maps.get("heatmap") and os.path.isfile(_maps["heatmap"]):
                    _heat_path = _maps["heatmap"]
            if _rows:
                st.dataframe(_rows, use_container_width=True, hide_index=True)
            _mp = _e1_res.get("multi_product_heatmap") or {}
            if _mp.get("disagreement_pixel_ratio") is not None:
                st.caption(f"分歧像元 {_mp.get('disagreement_pixel_ratio', 0):.2%}")
            _e1c1, _e1c2 = st.columns(2)
            with _e1c1:
                if _heat_path and st.button("加载热力图", key="load_e1_heatmap", use_container_width=True):
                    st.session_state.asset_override = _heat_path
                    st.session_state._asset_pinned = True
                    st.session_state._map_view_synced_for = None
                    st.session_state.asset_just_loaded = True
                    st.session_state._globe_rev = int(st.session_state.get("_globe_rev", 0)) + 1
                    st.rerun()
            with _e1c2:
                if st.button("清除精度评价结果", key="clear_e1_report", use_container_width=True):
                    st.session_state.pop("e1_report", None)
                    st.rerun()
            st.checkbox("球面叠加精度评价图层", key="globe_show_e1")
            with st.expander("详细报告", expanded=False):
                _md_path = _e1_res.get("report_md_path") or ""
                if _md_path and os.path.isfile(_md_path):
                    try:
                        with open(_md_path, "r", encoding="utf-8") as _mf:
                            st.markdown(_mf.read())
                    except Exception:
                        st.json(_e1_res)
                else:
                    st.json(_e1_res)

    st.markdown("---")
    if st.session_state.is_running:
        sbui.hint("任务运行中…", "run")

    if use_gee_download:
        m4_run_btn = st.button(
            "开始获取影像",
            type="primary",
            use_container_width=True,
            disabled=st.session_state.is_running,
        )
    elif adaptive_mode:
        tune_btn = st.button(
            "开始参数优化",
            type="primary",
            use_container_width=True,
            disabled=(
                st.session_state.is_running
                or not _autotune_ready
                or isinstance(st.session_state.get("_autotune_pending_plan"), dict)
            ),
        )
    elif cache_hit and not force_rerun:
        run_btn = st.button(
            "加载已有成果",
            type="primary",
            use_container_width=True,
        )
    else:
        _run_label = "开始指数法提取" if use_index_mode else "开始模型提取"
        run_btn = st.button(
            _run_label,
            type="primary",
            use_container_width=True,
            disabled=st.session_state.is_running,
        )

    stop_btn = st.button(
        "中断任务",
        type="secondary",
        use_container_width=True,
        disabled=not st.session_state.is_running,
    )

    if stop_btn:
        st.session_state.stop_requested = True
        st.session_state.pending_task = None
        st.session_state.pop("pending_autotune", None)
        if st.session_state.get("pipeline_stop_event") is not None:
            st.session_state.pipeline_stop_event.set()
        _tl_add(selected_task or st.session_state.get("_tl_current_task") or "system",
                "EXECUTE", "任务已被用户中断", status="CANCELLED", tool="stop_button")
        st.toast("正在请求安全终止…", icon="🛑")
        st.rerun()

    if tune_btn and _autotune_ready and _ref_id:
        _aoi_path = (task_aoi_shp or "").strip()
        _aoi_use = _aoi_path if _aoi_path and os.path.isfile(_aoi_path) else None
        _at_plan_id = f"autotune_{uuid.uuid4().hex}"
        st.session_state["_autotune_pending_plan"] = {
            "task": selected_task,
            "mode": "autotune",
            "reference_id": _ref_id,
            "objective": _tune_objective,
            "task_aoi_shp": _aoi_use,
            "plan_id": _at_plan_id,
        }
        _tl_add(
            selected_task or "unknown",
            "PLAN",
            "参数优化计划已生成，等待确认",
            status="WAITING_CONFIRMATION",
            plan_id=_at_plan_id,
            tool="run_autotune",
        )
        st.rerun()

    if m4_run_btn:
        if m4_end_date < m4_start_date:
            st.error("结束日期不能早于开始日期。")
        elif not m4_bands:
            st.error("请至少选择一个导出波段。")
        elif not os.path.isfile(m4_roi_path):
            st.error(f"研究区域矢量不存在: {m4_roi_path}")
        else:
            try:
                from agent_command_bridge import propose_gee_plan as _propose_manual_gee

                _manual_gee_plan, _manual_gee_errors = _propose_manual_gee(
                    st.session_state,
                    {
                        "task": selected_task or m4_roi_name,
                        "roi_name": m4_roi_name,
                        "start_date": m4_start_date.isoformat(),
                        "end_date": m4_end_date.isoformat(),
                        "bands": list(m4_bands),
                        "cloud_limit": int(m4_cloud),
                        "min_land_pct": float(m4_min_land),
                        "max_land_pct": float(m4_max_land),
                        "min_pixel_count": int(m4_min_pix),
                        "scale": int(m4_scale),
                        "export_to": m4_export_to,
                        "local_out_dir": os.path.normpath(m4_local_dir.strip()),
                        "drive_folder": m4_drive_folder.strip(),
                        "gee_proxy_url": (m4_gee_proxy or "").strip(),
                        "gee_project_id": (m4_gee_project or "").strip(),
                    },
                )
                if _manual_gee_plan.get("ready"):
                    st.info("已生成影像获取计划，请在下方确认后执行。")
                else:
                    st.warning("影像获取计划暂不可执行，请先修复以下条件：")
                for _plan_error in _manual_gee_errors or _manual_gee_plan.get("blockers") or []:
                    st.caption(f"· {_plan_error}")
            except Exception as _manual_gee_exc:
                st.error(f"影像获取计划生成失败：{type(_manual_gee_exc).__name__}")
            st.rerun()

    if run_btn:
        if cache_hit and not force_rerun:
            st.session_state.asset_override = cache_hit["file_path"]
            st.session_state._asset_pinned = True
            st.session_state.asset_just_loaded = True
            st.session_state._map_view_synced_for = None
            st.session_state._globe_rev = int(st.session_state.get("_globe_rev", 0)) + 1
            _tl_add(selected_task or "unknown", "REGISTER", "加载缓存成果",
                    status="SUCCEEDED", tool="cache_load",
                    artifacts=[os.path.basename(str(cache_hit["file_path"]))])
            st.rerun()
        else:
            if use_index_mode:
                # 指数法也先生成可审阅计划，确认按钮才会入队执行。
                try:
                    import index_agent_loop as _index_loop

                    _index_plan = _index_loop.build_index_plan(
                        task=selected_task or "",
                        input_dir=os.path.join(root_dir or "", selected_task or ""),
                        output_dir=os.path.join(final_root or "", selected_task or ""),
                        points_shp=(points_shp or "").strip(),
                        force_rerun=bool(force_rerun),
                    )
                    _index_ok, _index_blockers = _index_loop.validate_index_plan(_index_plan)
                    _index_plan["ready"] = _index_ok
                    _index_plan["blockers"] = _index_blockers
                    _index_plan["status"] = "waiting_confirmation" if _index_ok else "blocked"
                    st.session_state["_index_pending_plan"] = _index_plan
                except Exception as _index_plan_exc:
                    st.error(f"指数法计划生成失败：{type(_index_plan_exc).__name__}")
                st.rerun()
            # 深度学习手动入口与 Agent 共用同一份“计划 → 确认 → 执行”闭环；
            # 不再让侧栏按钮直接落入旧 run_pipeline_sync 兼容路径。
            if not use_index_mode:
                try:
                    from agent_command_bridge import propose_inference_plan as _propose_manual_inference

                    _manual_plan, _manual_plan_errors = _propose_manual_inference(
                        st.session_state,
                        {
                            "task": selected_task,
                            "prob_th": prob_th,
                            "min_cnt": min_cnt,
                            "force_rerun": bool(force_rerun),
                        },
                    )
                    if _manual_plan.get("ready"):
                        st.info("已生成潮滩智能提取计划，请在下方确认后执行。")
                    else:
                        st.warning("提取计划暂不可执行，请先修复以下条件：")
                    for _plan_error in _manual_plan_errors or _manual_plan.get("blockers") or []:
                        st.caption(f"· {_plan_error}")
                except Exception as _manual_plan_exc:
                    st.error(f"提取计划生成失败：{type(_manual_plan_exc).__name__}")
                st.rerun()
            _tl_add(selected_task or "unknown", "QUEUED", "提取任务已入队",
                    status="QUEUED", tool="run_pipeline")
            st.session_state.pending_task = {
                "task": selected_task,
                "prob": prob_th,
                "cnt": min_cnt,
                "mode": "index" if use_index_mode else "dl",
                "points_shp": (points_shp or "").strip() if use_index_mode else None,
                "force_rerun": bool(force_rerun),
            }
            st.session_state.is_running = True
            st.session_state.stop_requested = False
            st.rerun()

    # ---- 功能状态面板（B 阶段）：折叠、可刷新、不含敏感路径 ----
    with st.expander("功能状态", expanded=False):
        try:
            import capability_registry as _cap
        except Exception:
            _cap = None
        if _cap is not None:
            _app_dir = os.path.dirname(os.path.abspath(__file__))
            _cap_ctx = _cap.build_context(
                app_dir=_app_dir,
                model_path=st.session_state.get("ui_model_path") or "",
                task=selected_task or "",
            )
            _cap_sig = hashlib.md5(
                (f"{_cap_ctx['model_path']}|{_cap_ctx['task']}").encode("utf-8", errors="replace")
            ).hexdigest()[:12]
            _cap_reg = st.session_state.get("_capability_reg")
            if _cap_reg is None or st.session_state.get("_capability_ctx_sig") != _cap_sig:
                _cap_reg = _cap.CapabilityRegistry(context=_cap_ctx)
                st.session_state._capability_reg = _cap_reg
                st.session_state._capability_ctx_sig = _cap_sig
            _cap_c1, _cap_c2 = st.columns([3, 1])
            with _cap_c1:
                st.caption("动态功能状态（不含敏感路径）")
            with _cap_c2:
                if st.button("刷新", key="cap_refresh_btn", use_container_width=True):
                    _cap_reg.bump()
                    st.rerun()
            _status_labels = {
                "AVAILABLE": "🟢 可用",
                "CONDITIONAL": "🟡 条件可用",
                "BLOCKED": "🔴 暂不可用",
                "UNAVAILABLE": "⚪ 不可用",
                "UNKNOWN": "❔ 状态未知",
            }
            for _cid in _cap_reg.ids():
                _cst = _cap_reg.check(_cid)
                st.markdown(
                    f"**{uil.get_capability_label(_cid)}** · {_status_labels.get(_cst.status, _cst.status)}"
                )
                st.caption(_cst.summary)

    st.session_state["map_display_path"] = map_display_path

# ---- 主舱布局：左侧地图 + 地图下方状态/日志，右侧纯 Agent Dock ----
# Cesium 内部画布高度；可视 iframe 高度由 CSS --workbench-h 控制
GLOBE_HEIGHT = 1000
LOG_PANEL_HEIGHT = 88

try:
    _status_panel_height = int(st.session_state.get("agent_status_panel_height", 220))
except (TypeError, ValueError):
    _status_panel_height = 220
_status_panel_height = max(140, min(340, _status_panel_height))
st.session_state.agent_status_panel_height = _status_panel_height
_status_panel_collapsed = bool(st.session_state.get("agent_status_panel_collapsed", False))
# 地图下方预留状态工具栏 + 可调状态区；收起时只保留工具栏。
# 状态区不再单独占用工具栏行；仅保留 8px 边界间距，收起时地图几乎填满工作区。
_status_panel_reserve = 8 if _status_panel_collapsed else _status_panel_height + 8
st.markdown(
    f'<style>:root{{--cstf-status-panel-reserve:{_status_panel_reserve}px !important;}}</style>',
    unsafe_allow_html=True,
)

try:
    _chat_width_pct = int(st.session_state.get("agent_chat_width_pct", 34))
except (TypeError, ValueError):
    _chat_width_pct = 34
_chat_width_pct = max(24, min(48, _chat_width_pct))
st.session_state.agent_chat_width_pct = _chat_width_pct
_map_width_pct = 100 - _chat_width_pct
_side_width_pct = _chat_width_pct
st.markdown(
    f'<div class="cstf-layout-defaults" data-status-reserve="{_status_panel_reserve}" '
    f'data-agent-width="{_chat_width_pct}"></div>',
    unsafe_allow_html=True,
)
col_map, col_side = st.columns([_map_width_pct, _side_width_pct], gap="small")

_log_panel_slot = None
uploaded_images = []
prompt = ""
send_btn = False
raster_load_error = None
map_state = None

with col_map:
    st.markdown('<div class="cockpit-map-col"></div>', unsafe_allow_html=True)

    map_display_path = st.session_state.get("map_display_path")

    try:
        _mc = st.session_state.get("map_center") or [35.0, 105.0]
        st.session_state.map_center = [float(_mc[0]), float(_mc[1])]
    except (TypeError, IndexError, ValueError):
        st.session_state.map_center = [35.0, 105.0]
    try:
        st.session_state.map_zoom = int(st.session_state.get("map_zoom", 3))
    except (TypeError, ValueError):
        st.session_state.map_zoom = 3

    # st_folium 会用 session 的 center/zoom 覆盖 Folium 内部视角；换缓存/换 TIF 时必须先把视角对齐到数据范围
    if map_display_path and os.path.exists(map_display_path):
        if st.session_state.get("_map_view_synced_for") != map_display_path:
            # Copilot 刚跳转时会把 _map_view_synced_for 置空；此时保留跳转中心，勿拉回成果范围
            if (
                st.session_state.get("_map_prefer_center")
                and st.session_state.get("_map_view_synced_for") is None
            ):
                st.session_state._map_view_synced_for = map_display_path
            else:
                _rv = _cached_view_for_asset_path(st.session_state, map_display_path)
                if _rv is not None:
                    _rla, _rlo, _rzm = _rv
                    st.session_state.map_center = [_rla, _rlo]
                    st.session_state.map_zoom = int(_rzm)
                    st.session_state._map_view_synced_for = map_display_path
                    # 新加载成果时恢复「飞到图层范围」，覆盖上一轮 Copilot 跳转优先标志
                    st.session_state._map_prefer_center = False
    else:
        st.session_state._map_view_synced_for = None

    raster_load_error = None
    _use_2d = bool(st.session_state.get("use_2d_map_fallback", False))
    map_state = None
    _globe_open_url = None
    _payload: dict = {}

    if not _use_2d:
        try:
            import globe_engine as _globe
            import globe_server as _globe_srv

            if "_globe_server_port" not in st.session_state:
                st.session_state._globe_server_port = 8765
            _globe_port = _globe_srv.ensure_running(
                preferred_port=int(st.session_state._globe_server_port)
            )
            st.session_state._globe_server_port = _globe_port

            _grev = int(st.session_state.get("_globe_rev", 0))
            _e1_tag = ""
            _e1r = st.session_state.get("e1_report")
            if isinstance(_e1r, dict):
                _e1_tag = str(_e1r.get("roi_name") or _e1r.get("reference") or "")

            _amt = 0.0
            _ap = ""
            if map_display_path and os.path.exists(map_display_path):
                _ap = os.path.normpath(os.path.abspath(map_display_path))
                try:
                    _amt = os.path.getmtime(_ap)
                except OSError:
                    _amt = 0.0

            _prefer_center = bool(st.session_state.get("_map_prefer_center", False))

            # 本机打开页面时强制本地地球，避免 iframe 走失效/过载的 ngrok（ERR_NGROK_3004）
            _page_host = ""
            try:
                _hdrs = getattr(getattr(st, "context", None), "headers", None) or {}
                _page_host = str(_hdrs.get("Host") or _hdrs.get("host") or "")
            except Exception:
                _page_host = ""
            _force_local_globe = _globe_srv.is_local_page_host(_page_host)

            # 缓存签名不含 map_center/zoom：Agent 纯跳转时复用同一 iframe，避免 Viewer 重建
            # 但包含 force_local，避免本机/远程 URL 混用
            _cache_sig = hashlib.md5(
                (
                    f"{_ap}|{_amt:.4f}|{_grev}|"
                    f"{st.session_state.get('result_overlay_opacity_pct', 50)}|"
                    f"{st.session_state.get('globe_show_e1', True)}|{_e1_tag}|"
                    f"local={int(_force_local_globe)}|cam=v3"
                ).encode("utf-8", errors="replace")
            ).hexdigest()

            _globe_warn = _globe_srv.globe_public_url_warning(_globe_port)
            if _globe_warn and not _force_local_globe:
                _warn_tok = hashlib.md5(_globe_warn.encode("utf-8", errors="replace")).hexdigest()[:12]
                if st.session_state.get("_globe_warn_token") != _warn_tok:
                    st.session_state._globe_warn_token = _warn_tok
                    st.warning(_globe_warn)

            _cached_url = st.session_state.get("_globe_iframe_url")
            _globe_cache_hit = (
                st.session_state.get("_globe_iframe_cache_sig") == _cache_sig
                and bool(_cached_url)
                and _globe_srv.same_globe_origin(
                    _cached_url, _globe_port, force_local=_force_local_globe
                )
            )

            if _globe_cache_hit:
                _globe_open_url = _cached_url
                _serve_ok = bool((st.session_state.get("_last_globe_payload") or {}).get("serve_ok", True))
            else:
                _alive_tiles = {}
                for _tk, _tc in list(st.session_state._globe_tile_clients.items()):
                    if _globe._tile_client_alive(_tc):
                        _alive_tiles[_tk] = _tc
                st.session_state._globe_tile_clients = _alive_tiles

                with _local_tile_no_proxy():
                    _payload = _globe.build_globe_payload(
                        center=tuple(st.session_state.map_center),
                        zoom=int(st.session_state.map_zoom),
                        result_path=map_display_path if map_display_path and os.path.exists(map_display_path) else None,
                        opacity_pct=float(st.session_state.get("result_overlay_opacity_pct", 50)),
                        e1_report=st.session_state.get("e1_report"),
                        show_e1_overlay=bool(st.session_state.get("globe_show_e1", True)),
                        tile_clients=st.session_state._globe_tile_clients,
                        ion_token=os.environ.get("CESIUM_ION_TOKEN"),
                        show_borders=False,
                        globe_port=_globe_port,
                        prefer_center=_prefer_center,
                        force_local=_force_local_globe,
                        channel_id=st.session_state.get("_map_channel_id"),
                    )

                if map_display_path and os.path.exists(map_display_path):
                    _has_layer = bool(_payload.get("geojsonLayers") or _payload.get("imageryLayers"))
                    if not _has_layer:
                        raster_load_error = f"无法解析或加载资产: {map_display_path}"

                _html = _globe.build_cesium_html(_payload, height_px=GLOBE_HEIGHT, full_viewport=True)

                _gkey = hashlib.md5(_html.encode("utf-8", errors="replace")).hexdigest()[:10]
                if _ap:
                    _gkey = hashlib.md5(
                        f"{_gkey}|{_ap}|{_amt:.4f}|{_grev}|{st.session_state.get('result_overlay_opacity_pct', 50)}".encode(
                            "utf-8", errors="replace"
                        )
                    ).hexdigest()[:16]
                else:
                    _gkey = f"{_gkey}_{_grev}"

                _globe_srv.publish_html(_html, _gkey)
                _globe_open_url = _globe_srv.globe_url(
                    _globe_port, _gkey, bust=_grev, force_local=_force_local_globe
                )
                _html_disk = _globe_srv.html_dir() / f"{_gkey}.html"

                _serve_ok = False
                _serve_err = ""
                if _html_disk.is_file() and _payload.get("assetName"):
                    try:
                        import urllib.request as _urlreq

                        with _urlreq.urlopen(_globe_open_url, timeout=4) as _resp:
                            _body = _resp.read().decode("utf-8", errors="replace")
                        _serve_ok = _payload["assetName"] in _body
                        if not _serve_ok:
                            _serve_err = "HTTP 服务返回的页面不含资产数据（可能是旧版 globe 进程占用端口）"
                    except Exception as _se:
                        _serve_err = str(_se)
                elif _html_disk.is_file():
                    _serve_ok = True

                st.session_state["_last_globe_payload"] = {
                    "path": map_display_path,
                    "flyRectangle": bool(_payload.get("flyRectangle")),
                    "geojson": len(_payload.get("geojsonLayers") or []),
                    "imagery": len(_payload.get("imageryLayers") or []),
                    "assetName": _payload.get("assetName"),
                    "key": _gkey,
                    "url": _globe_open_url,
                    "port": _globe_port,
                    "serve_ok": _serve_ok,
                }

                if not _serve_ok and _serve_err:
                    raster_load_error = _serve_err or raster_load_error

                st.session_state._globe_iframe_cache_sig = _cache_sig
                st.session_state._globe_iframe_url = _globe_open_url

            if _globe_open_url:
                components.iframe(
                    _globe_open_url,
                    height=GLOBE_HEIGHT,
                    scrolling=False,
                )
                # Phase D: 消费 Cesium AOI 消息（绘制 → 校验 → 回声图层）
                _poll_aoi_messages()
                # Copilot 地图跳转：向已加载的 Cesium iframe 发 CSTF_FLY，避免重建 Viewer
                _fly = st.session_state.pop("_pending_camera_fly", None)
                if isinstance(_fly, dict) and _fly.get("lat") is not None and _fly.get("lon") is not None:
                    try:
                        _fly_lat = float(_fly["lat"])
                        _fly_lon = float(_fly["lon"])
                        _fly_zoom = int(_fly.get("zoom", 9))
                        _fly_height = _fly.get("height")
                        if _fly_height is None:
                            _fly_height = float(_globe.zoom_to_height_m(_fly_zoom, _fly_lat))
                        _fly_label = _fly.get("label") or f"({_fly_lat:.2f}°N, {_fly_lon:.2f}°E)"
                        _fly_payload, _fly_errs = _map_proto.make_fly_message(
                            _fly_lon,
                            _fly_lat,
                            zoom=_fly_zoom,
                            height=_fly_height,
                            pitch=float(_globe.DEFAULT_CAMERA["pitch_deg"]),
                            heading=float(_globe.DEFAULT_CAMERA["heading_deg"]),
                            duration=float(_fly.get("duration", 1.0)),
                            preset=_fly.get("preset"),
                            label=_fly_label,
                            source=str(_fly.get("source") or "agent"),
                        )
                        if _fly_payload is None:
                            st.warning("地图跳转参数无效：" + "; ".join(_fly_errs or []))
                        else:
                            # READY 握手：等 Cesium 就绪后发；等待窗口超 3s 仍未就绪则带警告发送
                            _map_ready_warning = False
                            _map_channel_id = st.session_state.get("_map_channel_id")
                            _mp_state = _globe_srv.map_protocol_state(
                                channel_id=_map_channel_id
                            )
                            _ready_ok = bool(_mp_state.get("ready_ts"))
                            if not _ready_ok:
                                _wait_started = st.session_state.get("_map_ready_wait_started")
                                if _wait_started is None:
                                    st.session_state._map_ready_wait_started = time.time()
                                elif (time.time() - float(_wait_started)) > 3.0:
                                    _map_ready_warning = True
                            import json as _json

                            _fly_js = _json.dumps(_fly_payload, ensure_ascii=False)
                            components.html(
                                f"""
<script>
(() => {{
  const win = window.parent || window;
  const doc = win.document;
  const msg = {_fly_js};
  // targetOrigin 收紧：从 iframe src 提取精确 origin；取不到时回退当前页面 origin
  let origin = "*";
  try {{
    const iframes = doc.querySelectorAll("iframe");
    iframes.forEach((ifr) => {{
      const src = ifr.getAttribute("src") || "";
      if (src.indexOf("/globe") >= 0 || src.indexOf(":8765") >= 0) {{
        try {{ origin = new URL(src, win.location.href).origin; }} catch (e) {{}}
      }}
    }});
  }} catch (e) {{}}
  const send = () => {{
    const iframes = doc.querySelectorAll("iframe");
    let sent = false;
    iframes.forEach((ifr) => {{
      const src = ifr.getAttribute("src") || "";
      if (!src) return;
      if (src.indexOf("/globe") >= 0 || src.indexOf(":8765") >= 0) {{
        try {{
          ifr.contentWindow.postMessage(msg, origin);
          sent = true;
        }} catch (e) {{}}
      }}
    }});
    return sent;
  }};
  // 每次定位都会注入一个很短的隐藏组件。若上一轮定位的延迟重试
  // 仍在 parent window 中排队，它们可能在本轮定位之后再次把相机拉回旧位置。
  // 将定时器挂在 parent 上并在新命令开始时统一取消，保证“最后一次定位”胜出。
  try {{
    const oldTimers = Array.isArray(win.__cstfFlyRetryTimers)
      ? win.__cstfFlyRetryTimers : [];
    oldTimers.forEach((timerId) => win.clearTimeout(timerId));
    win.__cstfFlyRetryTimers = [];
  }} catch (e) {{}}
  // The iframe element can exist before Cesium installs its message listener.
  // Retry at a few increasing delays; avoid restarting the camera flight
  // continuously while the viewer is animating.
  const retryDelays = [150, 400, 900, 1800, 3200, 5000];
  send();
  retryDelays.forEach((delay) => {{
    try {{
      const timerId = win.setTimeout(send, delay);
      win.__cstfFlyRetryTimers.push(timerId);
    }} catch (e) {{
      setTimeout(send, delay);
    }}
  }});
}})();
</script>
                            """,
                                height=0,
                            )
                            # 短等待 FLY_ACK（最多 ~1.2s），成功则 toast；未确认不阻塞
                            _ack = _globe_srv.wait_map_ack(
                                _fly_payload.get("command_id", ""),
                                timeout=1.2,
                                channel_id=_map_channel_id,
                            )
                            if _ack:
                                st.session_state["_map_ready_wait_started"] = None
                                if _ack.get("ok"):
                                    st.toast(f"地图已定位：{_fly_label}", icon="🗺️")
                                else:
                                    st.warning("地图跳转未完成，请检查地球页面状态。")
                            if _ack is None and _map_ready_warning:
                                st.caption("⚠️ 地图尚未确认就绪（可能仍在加载），已尝试跳转。")
                    except (TypeError, ValueError):
                        pass
            else:
                raster_load_error = raster_load_error or "未能生成三维地球 URL"
        except Exception as _globe_err:
            _safe_globe_err = safe_error_summary(_globe_err)
            raster_load_error = _safe_globe_err
            st.error(f"三维地球加载失败：{_safe_globe_err}")
            st.caption(
                "本机请用 http://localhost:8501 打开；远程演示需同时启动 ngrok 并设置 CSTF_GLOBE_PUBLIC_URL。"
            )
            st.toast(f"三维地球加载失败，已切换 2D 地图: {_safe_globe_err}", icon="⚠️")
            _use_2d = True

    if _use_2d:
        m = leafmap.Map(
            center=st.session_state.map_center,
            zoom=st.session_state.map_zoom,
            draw_control=True,
            measure_control=True,
        )
        try:
            m.add_basemap("OpenStreetMap")
        except Exception:
            pass

        if map_display_path and os.path.exists(map_display_path):
            layer_label = os.path.splitext(os.path.basename(map_display_path))[0]
            _rop = st.session_state.get("result_overlay_opacity_pct", 50) / 100.0
            _ok, _rerr = _add_result_to_map(
                m, map_display_path, f"成果: {layer_label}", opacity=_rop
            )
            if not _ok:
                raster_load_error = _rerr

        m.add_layer_control()
        _lat, _lon = float(st.session_state.map_center[0]), float(st.session_state.map_center[1])
        _folium_key = "cstf_main_map"
        if map_display_path and os.path.exists(map_display_path):
            _ap = os.path.normpath(os.path.abspath(map_display_path))
            try:
                _sig = hashlib.md5(
                    f"{_ap}\0{os.path.getmtime(_ap):.6f}".encode("utf-8", errors="replace")
                ).hexdigest()[:12]
                _folium_key = f"cstf_{_sig}"
            except OSError:
                pass
        map_state = st_folium(
            m,
            height=GLOBE_HEIGHT,
            width=None,
            use_container_width=True,
            center=(_lat, _lon),
            zoom=int(st.session_state.map_zoom),
            key=_folium_key,
        )

    if st.session_state.asset_just_loaded:
        st.session_state.asset_just_loaded = False
        _lp = st.session_state.get("_last_globe_payload") or {}
        if raster_load_error:
            st.toast(f"成果图层加载失败: {raster_load_error}", icon="⚠️")
        elif _lp.get("flyRectangle") and _lp.get("assetName"):
            st.toast(
                f"✅ 已加载 {_lp.get('assetName')} · 矢量:{_lp.get('geojson', 0)} 栅格:{_lp.get('imagery', 0)}",
                icon="✅",
            )
        elif map_display_path:
            st.toast("⚡ 成果路径已更新，但未能解析图层范围", icon="⚠️")
        else:
            st.toast("⚡ 已有成果已加载到地图", icon="✅")

    if raster_load_error and not st.session_state.asset_just_loaded:
        st.toast(f"成果图层加载异常: {raster_load_error}", icon="⚠️")
        with st.expander("🛰️ 地图加载诊断", expanded=False):
            _lp = st.session_state.get("_last_globe_payload") or {}
            st.write(f"**错误**：{raster_load_error}")
            if _lp.get("url"):
                st.write(f"**地球 URL**：{_lp.get('url')}")
            st.write(f"**本机地球端口**：{st.session_state.get('_globe_server_port', '—')}")
            st.write(
                "**建议**：① 用 http://localhost:8501 打开（不要用局域网 IP，除非已配 ngrok）；"
                "② 重启 Streamlit；③ 远程演示见 REMOTE_DEMO.md"
            )
            if st.button("切换为 2D 地图并重试", key="btn_force_2d_map"):
                st.session_state.use_2d_map_fallback = True
                st.rerun()

    # 任务状态和终端日志位于地图下方，不再占用右侧 Agent Dock。
    st.markdown('<div class="cstf-map-status-zone"></div>', unsafe_allow_html=True)
    _status_toolbar_c1, _status_toolbar_c2 = st.columns([6, 1], gap="small")
    with _status_toolbar_c1:
        st.markdown('<div class="cstf-status-toolbar-spacer" aria-hidden="true"></div>', unsafe_allow_html=True)
    with _status_toolbar_c2:
        st.markdown(
            f'<div class="cstf-status-toggle-host"><div class="cstf-status-toggle-state" '
            f'data-collapsed="{1 if _status_panel_collapsed else 0}"></div></div>',
            unsafe_allow_html=True,
        )
        if st.button(
            "展开" if _status_panel_collapsed else "收起",
            key="agent_status_panel_toggle",
            use_container_width=True,
        ):
            st.session_state.agent_status_panel_collapsed = not _status_panel_collapsed
            st.rerun()
    if not _status_panel_collapsed:
        _log_panel_slot = st.container(height="stretch", border=True)

with col_side:
    st.markdown('<div class="command-deck-side">', unsafe_allow_html=True)
    st.markdown('<div class="cstf-copilot-dock">', unsafe_allow_html=True)
    st.markdown('<div class="cockpit-copilot-zone-start"></div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="deck-section-title">🤖 智能分析助手</div>',
        unsafe_allow_html=True,
    )

    if "_conversation_store" not in st.session_state:
        # Chat previews are disposable UI artefacts, not conversation data.
        # Bound their lifetime on each new Streamlit session so abandoned
        # uploads cannot accumulate indefinitely in the checkout.
        try:
            cleanup_preview_cache()
        except Exception:
            # Cleanup is best-effort and must never prevent the workbench from
            # starting; the helper itself returns only aggregate counters.
            pass
        try:
            from conversation_store import ConversationStore

            _conversation_db = os.environ.get(
                "CSTF_CONVERSATION_DB_PATH",
                os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "conversations.sqlite3"),
            )
            st.session_state._conversation_store = ConversationStore(_conversation_db)
            # Do not materialize an empty session on page load.  A thread is
            # created lazily when the first message is actually submitted.
            st.session_state._conversation_thread_id = st.session_state.get(
                "_conversation_thread_id"
            ) or None
            st.session_state._conversation_store.cleanup()
        except Exception:
            st.session_state._conversation_store = None
    if "messages" not in st.session_state:
        _restored_messages = []
        if st.session_state.get("_conversation_store") is not None:
            try:
                _current_thread = st.session_state.get("_conversation_thread_id")
                if _current_thread:
                    _restored_messages = st.session_state._conversation_store.load_messages(
                        _current_thread
                    )
            except Exception:
                _restored_messages = []
        st.session_state.messages = _restored_messages or [
            {"role": "assistant", "content": "您好！我是智能分析助手。请告诉我您想分析的区域和年份，或上传截图让我识别。"}
        ]

    _default_chat_message = {
        "role": "assistant",
        "content": "您好！我是智能分析助手。请告诉我您想分析的区域和年份，或上传截图让我识别。",
    }
    st.session_state.setdefault("agent_spatial_consent", False)
    # Session buttons are rendered below the radio widget. Defer the widget
    # value change to the next rerun so Streamlit does not reject a post-
    # instantiation session-state mutation.
    if st.session_state.pop("_conversation_open_dialog", False):
        st.session_state.agent_dock_view = "对话"
    _agent_dock_view = st.radio(
        "Agent 面板",
        ["对话", "历史"],
        horizontal=True,
        key="agent_dock_view",
        label_visibility="collapsed",
    )
    _dock_view_class = "history" if _agent_dock_view == "历史" else "chat"
    st.markdown(
        f'<div class="cstf-agent-view-marker cstf-agent-view-{_dock_view_class}"></div>',
        unsafe_allow_html=True,
    )
    if _agent_dock_view == "历史" and st.session_state.get("_conversation_store") is not None:
        try:
            _conversation_threads = st.session_state._conversation_store.list_threads(
                limit=None,
                include_empty=False,
            )
        except Exception:
            _conversation_threads = []
        st.markdown('<div class="cstf-session-list-heading">会话列表</div>', unsafe_allow_html=True)
        # 历史列表填充 Agent Dock 的剩余高度，内容过多时仅在列表内部滚动。
        with st.container(height="stretch", border=True):
            st.markdown('<div class="cstf-history-list-marker" aria-hidden="true"></div>', unsafe_allow_html=True)
            if not _conversation_threads:
                st.caption("暂无历史会话")
            else:
                for _thread in _conversation_threads:
                    _tid = str(_thread.get("thread_id") or "")
                    if not _tid:
                        continue
                    try:
                        _stamp = datetime.datetime.fromtimestamp(
                            float(_thread.get("last_seen") or 0)
                        ).strftime("%m-%d %H:%M")
                    except (TypeError, ValueError, OSError):
                        _stamp = "—"
                    _preview = re.sub(r"\s+", " ", str(_thread.get("preview") or "")).strip()
                    _preview = _preview[:34] if _preview else "空会话"
                    _is_current = _tid == str(st.session_state.get("_conversation_thread_id") or "")
                    _label = f"{'当前 · ' if _is_current else ''}{_stamp} · {_preview}"
                    if st.button(
                        _label,
                        key=f"conversation_switch_{_tid}",
                        use_container_width=True,
                        disabled=_is_current,
                    ):
                        st.session_state._conversation_thread_id = _tid
                        st.session_state.messages = (
                            st.session_state._conversation_store.load_messages(_tid)
                            or [_default_chat_message.copy()]
                        )
                        st.session_state._conversation_open_dialog = True
                        st.rerun()
        # 历史页底部的两个操作均分可用宽度，避免右侧留下无意义空白。
        _conv_c1, _conv_c2 = st.columns(2)
        with _conv_c1:
            if st.button("新会话", key="conversation_new", use_container_width=True):
                st.session_state._conversation_thread_id = st.session_state._conversation_store.create_thread()
                st.session_state.messages = [_default_chat_message.copy()]
                st.session_state._conversation_open_dialog = True
                st.rerun()
        with _conv_c2:
            if st.button(
                "清空会话",
                key="conversation_clear",
                help="删除当前会话；有下一条则选中，否则创建新会话并进入对话",
                use_container_width=True,
                disabled=not bool(st.session_state.get("_conversation_thread_id")),
            ):
                _deleted_thread_id = str(st.session_state.get("_conversation_thread_id") or "")
                from conversation_store import next_thread_id_after_delete

                _next_thread_id = next_thread_id_after_delete(
                    _conversation_threads,
                    _deleted_thread_id,
                )
                st.session_state._conversation_store.delete_thread(_deleted_thread_id)
                if _next_thread_id:
                    # Keep the history view active: the selection changes,
                    # but the user is not pushed into the chat view.
                    st.session_state._conversation_thread_id = _next_thread_id
                    st.session_state.messages = (
                        st.session_state._conversation_store.load_messages(_next_thread_id)
                        or [_default_chat_message.copy()]
                    )
                else:
                    # Do not leave the dock with an empty chat container after
                    # the final session is removed.  Create a fresh draft and
                    # use the existing deferred flag to switch back to chat
                    # view on the next Streamlit rerun.
                    st.session_state._conversation_thread_id = (
                        st.session_state._conversation_store.create_thread()
                    )
                    st.session_state.messages = [_default_chat_message.copy()]
                    st.session_state._conversation_open_dialog = True
                st.rerun()
    st.markdown('<div class="cockpit-chat-anchor"></div>', unsafe_allow_html=True)
    # 历史页不需要聊天容器边框；避免无会话时在底部留下空白行。
    chat_box = st.container(border=_agent_dock_view == "对话")

    with chat_box:
        # Keep a dedicated hook even when there are no messages.  The history
        # view uses it to remove this otherwise empty bordered chat container.
        st.markdown('<div class="cstf-chat-stream-marker" aria-hidden="true"></div>', unsafe_allow_html=True)
        for msg in st.session_state.messages:
            avatar = "🧑‍💻" if msg["role"] == "user" else "🤖"
            with st.chat_message(msg["role"], avatar=avatar):
                if msg["role"] == "user":
                    st.markdown('<div class="msg-role msg-role-user">用户</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="msg-role msg-role-assistant">智能体</div>', unsafe_allow_html=True)
                st.markdown(msg["content"])
                _render_chat_attachment_previews(msg)

    st.markdown('<div class="cstf-chat-compose-host">', unsafe_allow_html=True)
    with st.form(key="chat_form", clear_on_submit=True):
        _input_cols = st.columns([10, 1])
        with _input_cols[0]:
            prompt = st.text_input(
                "chat_input",
                placeholder="问问智能分析助手…",
                label_visibility="collapsed",
            )
        with _input_cols[1]:
            send_btn = st.form_submit_button("➤", use_container_width=True)

        try:
            _attachment_uploader_epoch = int(
                st.session_state.get("_attachment_uploader_epoch", 0)
            )
        except (TypeError, ValueError):
            _attachment_uploader_epoch = 0
        uploaded_images = st.file_uploader(
            "chat_attach",
            type=["png", "jpg", "jpeg", "webp", "tif", "tiff"],
            accept_multiple_files=True,
            label_visibility="collapsed",
            key=f"chat_attach_uploader_{_attachment_uploader_epoch}",
        )
        uploaded_images = list(uploaded_images or [])
        st.markdown(
            f'<span class="cstf-attachment-epoch-marker" data-epoch="{_attachment_uploader_epoch}" '
            'aria-hidden="true"></span>',
            unsafe_allow_html=True,
        )
        if os.environ.get("CSTF_ALLOW_RAW_SYSTEM_COMMAND", "").strip().lower() in {"1", "true", "yes", "on"}:
            st.checkbox(
                "开发模式：允许本轮直接执行聊天框系统命令",
                value=False,
                key="agent_raw_system_command_consent",
                help="仅用于本地验收；需同时配置 CSTF_ALLOW_RAW_SYSTEM_COMMAND=1，且仍受重型任务确认门闩约束。",
            )

    components.html(
        """
        <script>
        (() => {
          const win = window.parent || window;
          const doc = win.document;
          const bindingToken = `attach-${Date.now()}-${Math.random().toString(36).slice(2)}`;
          // Only the newest Streamlit component execution may install or poll
          // the parent-page attachment bridge. Older iframe timers stop as
          // soon as a newer generation is announced.
          win.__cstfAttachmentInstallerGeneration = bindingToken;
          if (!doc || doc.body?.dataset?.cstfChatComposeBound === "1") {
            /* allow re-bind on streamlit rerun via observer */
          }

          const bindChatCompose = () => {
            if (win.__cstfAttachmentInstallerGeneration !== bindingToken) return false;
            const chatInputSelector =
              'input[aria-label="chat_input"], input[aria-label="聊天输入"], input[placeholder*="智能分析助手"]';
            const forms = doc.querySelectorAll('[data-testid="stForm"]');
            let chatForm = null;
            forms.forEach((f) => {
              if (f.querySelector(chatInputSelector)) chatForm = f;
            });
            if (!chatForm) return false;
            chatForm.classList.add('cstf-chat-compose');
            const fileWrap = chatForm.querySelector('[data-testid="stFileUploader"]');
            const fileInput = chatForm.querySelector('input[type="file"]');
            const epochMarker = chatForm.querySelector('.cstf-attachment-epoch-marker');
            const uploaderEpoch = epochMarker?.dataset?.epoch || 'unknown';
            if (!fileWrap || !fileInput) return false;

            // Streamlit can retain the same native file input node while the
            // server rotates its uploader key. Treat that epoch boundary as a
            // hard compose reset before reading `files`; otherwise a browser
            // can re-advertise the previous round's filename after rerun.
            const existingController = win.__cstfAttachmentController;
            const previousEpoch = existingController?.uploaderEpoch
              ?? win.__cstfAttachmentLastEpoch;
            const attachmentEpochChanged = (
              previousEpoch != null
              && previousEpoch !== 'unknown'
              && previousEpoch !== uploaderEpoch
            );
            if (attachmentEpochChanged) {
              try {
                fileInput.value = '';
              } catch (_) {
                /* The server-side uploader key remains the final fallback. */
              }
            }
            win.__cstfAttachmentLastEpoch = uploaderEpoch;

            const inputRow = [...chatForm.querySelectorAll('[data-testid="stHorizontalBlock"]')]
              .find((row) => row.querySelector(chatInputSelector));
            if (!inputRow) return false;
            inputRow.classList.add('cstf-chat-input-row');
            const chatInput = inputRow.querySelector(chatInputSelector);
            const inputColumn = chatInput?.closest('[data-testid="stColumn"]');
            const sendButton = inputRow.querySelector(
              '[data-testid="stFormSubmitButton"] button'
            );
            const sendColumn = sendButton?.closest('[data-testid="stColumn"]');
            inputColumn?.classList.add('cstf-chat-input-column');
            sendColumn?.classList.add('cstf-chat-send-column');

            const canReuseController = Boolean(
              existingController?.version === '2026-08-23-attachment-v7'
              && existingController?.fileInput === fileInput
              && existingController?.fileInput?.isConnected
              && existingController?.sendButton === sendButton
              && existingController?.chatForm === chatForm
              && existingController?.uploaderEpoch === uploaderEpoch
              && existingController?.bar?.isConnected
              && typeof existingController.sync === 'function'
            );
            if (canReuseController) {
              existingController.sync();
              return true;
            }
            if (existingController && typeof existingController.destroy === 'function') {
              existingController.destroy();
            }
            chatForm.querySelectorAll('.cstf-attach-bar').forEach((node) => node.remove());

            const bar = doc.createElement('div');
            bar.className = 'cstf-attach-bar';
            bar.dataset.bindingToken = bindingToken;
            bar.dataset.uploaderEpoch = uploaderEpoch;
            bar.innerHTML =
                '<button type="button" class="cstf-plus-btn" ' +
                'data-tooltip="每个文件≤200MB · PNG / JPG / WebP / TIFF" ' +
                'aria-label="选择附件（每个文件≤200MB；PNG、JPG、WebP、TIFF）">+</button>' +
                '<div class="cstf-attach-preview" role="list" aria-label="已选择附件预览"></div>';
            // 将附件入口放到消息输入行最左侧；文件选择器本身仍隐藏在表单内。
            if (bar.parentElement !== inputRow) inputRow.insertBefore(bar, inputRow.firstChild);

            const plusBtn = bar.querySelector('.cstf-plus-btn');
            const previewPanel = bar.querySelector('.cstf-attach-preview');

            const defaultTooltip = '每个文件≤200MB · PNG / JPG / WebP / TIFF';
            const previewUrls = new Set();
            // 浏览器每次重新打开文件选择器都会替换原生 FileList；在本轮
            // 会话内显式累积 File 对象，再同步回 input，避免只剩最后一次选择。
            let selectedFiles = Array.from(fileInput.files || []).slice(0, 6);
            let selectedFilesSyncing = false;
            let renderedPreviewSignature = null;
            let active = true;

            const fileIdentity = (file) => [
              String(file?.name || ''),
              String(file?.size || 0),
              // Streamlit/browser bridges can reconstruct File objects with a
              // fresh lastModified timestamp on every reconciliation. Use
              // stable display and MIME fields so one selected attachment
              // remains one preview instead of visibly blinking.
              String(file?.type || ''),
            ].join('::');

            const assignSelectedFiles = (files, notify = false) => {
              try {
                const transfer = new win.DataTransfer();
                files.slice(0, 6).forEach((file) => transfer.items.add(file));
                fileInput.files = transfer.files;
                if (notify) {
                  // Streamlit 的上传器通过 change 事件同步内部 widget 状态；
                  // 赋值 FileList 本身不会触发该事件，因此补发一次聚合后的变更。
                  selectedFilesSyncing = true;
                  fileInput.dispatchEvent(new win.Event('change', { bubbles: true }));
                }
                return true;
              } catch (_) {
                // 现代 Chromium 支持 DataTransfer；若浏览器禁用赋值，
                // 仍保留本地预览，服务端会按原生 FileList 能力处理。
                return false;
              }
            };

            const mergeSelectedFiles = (files) => {
              const priorSelectionCount = selectedFiles.length;
              const merged = [...selectedFiles];
              const seen = new Set(merged.map(fileIdentity));
              files.forEach((file) => {
                const identity = fileIdentity(file);
                if (!seen.has(identity)) {
                  seen.add(identity);
                  merged.push(file);
                }
              });
              selectedFiles = merged.slice(0, 6);
              // The browser's first native change already contains the full
              // selection.  Only notify Streamlit when a later chooser round
              // actually adds a new file; dispatching again on the first
              // selection makes one upload appear twice in the chat message.
              const shouldNotify = priorSelectionCount > 0
                && selectedFiles.length > priorSelectionCount;
              assignSelectedFiles(selectedFiles, shouldNotify);
              return selectedFiles;
            };

            const clearAttachmentPreview = (preserveSignature = false) => {
              previewUrls.forEach((url) => {
                try { win.URL.revokeObjectURL(url); } catch (_) {}
              });
              previewUrls.clear();
              if (!preserveSignature) renderedPreviewSignature = null;
              if (previewPanel) {
                previewPanel.replaceChildren();
                previewPanel.classList.remove('is-visible');
              }
            };

            const fileExtension = (file) => {
              const match = String(file?.name || '').match(/[.]([a-z0-9]+)$/i);
              return match ? match[1].toUpperCase() : '文件';
            };

            const browserPreviewable = (file) => {
              const type = String(file?.type || '').toLowerCase();
              const name = String(file?.name || '').toLowerCase();
              return type === 'image/png'
                || type === 'image/jpeg'
                || type === 'image/webp'
                || /[.](png|jpe?g|webp)$/i.test(name);
            };

            const isGeoTiff = (file) => {
              const type = String(file?.type || '').toLowerCase();
              const name = String(file?.name || '').toLowerCase();
              return type === 'image/tiff' || /[.]tiff?$/i.test(name);
            };

            const renderAttachmentPreview = (files) => {
              if (!previewPanel) return;
              const previewSignature = files
                .slice(0, 6)
                .map(fileIdentity)
                .join('|');
              // The attachment bridge reconciles the Streamlit DOM every
              // 260ms. Recreating Blob URLs for an unchanged selection makes
              // image thumbnails visibly blink, so only redraw a real change.
              if (previewSignature === renderedPreviewSignature) return;
              clearAttachmentPreview(true);
              renderedPreviewSignature = previewSignature;
              if (!files.length) return;
              files.slice(0, 6).forEach((file) => {
                const card = doc.createElement('div');
                card.className = 'cstf-attach-preview-card';
                card.setAttribute('role', 'listitem');
                card.title = file.name;
                if (browserPreviewable(file)) {
                  const url = win.URL.createObjectURL(file);
                  previewUrls.add(url);
                  const image = doc.createElement('img');
                  image.src = url;
                  image.alt = file.name;
                  card.appendChild(image);
                } else {
                  const label = doc.createElement('span');
                  label.className = 'cstf-attach-preview-label';
                  label.textContent = `${isGeoTiff(file) ? 'TIFF' : fileExtension(file)} · ${file.name}`;
                  card.appendChild(label);
                }
                previewPanel.appendChild(card);
              });
              const clearButton = doc.createElement('button');
              clearButton.type = 'button';
              clearButton.className = 'cstf-attach-preview-clear';
              clearButton.setAttribute('aria-label', '清除已选附件');
              clearButton.textContent = '×';
              clearButton.addEventListener('click', (event) => {
                event.preventDefault();
                event.stopPropagation();
                win.__cstfAttachmentClearRequested = true;
                selectedFiles = [];
                try { fileInput.value = ''; } catch (_) {}
                clearAttachmentPreview();
                syncAttach([]);
              });
              previewPanel.appendChild(clearButton);
              previewPanel.classList.add('is-visible');
            };

            if (plusBtn && plusBtn.dataset.bound !== '1') {
              plusBtn.dataset.bound = '1';
              plusBtn.addEventListener('click', (e) => {
                e.preventDefault();
                e.stopPropagation();
                // Match the original main.py flow: + opens the native chooser
                // directly, and the selected files are sent to the active
                // multimodal executor on submit.
                fileInput.click();
              });
            }

            const syncAttach = (filesOverride = undefined) => {
              if (!active) return;
              // A submitted attachment must stay hidden across every
              // Streamlit rerender and uploader-key rotation. This is
              // intentionally not keyed by uploaderEpoch: the key changes
              // after submit while a retained native FileList may still exist.
              const mustRemainCleared = Boolean(
                win.__cstfAttachmentClearRequested
              );
              if (mustRemainCleared) {
                selectedFiles = [];
              }
              const files = filesOverride === undefined
                ? (mustRemainCleared ? [] : (selectedFiles.length ? selectedFiles : Array.from(fileInput.files || [])))
                : Array.from(filesOverride || []);
              selectedFiles = files.slice(0, 6);
              renderAttachmentPreview(selectedFiles);
              plusBtn.removeAttribute('title');
              // Tooltip is deliberately invariant; selected names belong in
              // the preview cards, never in the format/help bubble.
              plusBtn.dataset.tooltip = defaultTooltip;
              plusBtn.setAttribute('aria-label', `选择附件上传方式（${defaultTooltip}）`);
            };

            if (fileInput.dataset.cstfBound !== bindingToken) {
              fileInput.dataset.cstfBound = bindingToken;
              fileInput.addEventListener('change', (event) => {
                if (selectedFilesSyncing) {
                  selectedFilesSyncing = false;
                  syncAttach(selectedFiles);
                  return;
                }
                // A genuine chooser action starts a new compose cycle even if
                // Streamlit has not yet mounted the next server-side epoch.
                // It is the only event that releases the submitted-clear
                // guard; rerenders and stale FileLists never may do so.
                const incomingFiles = Array.from(fileInput.files || []);
                if (incomingFiles.length) {
                  win.__cstfAttachmentClearRequested = false;
                }
                const known = new Set(selectedFiles.map(fileIdentity));
                const addsNewFile = incomingFiles.some(
                  (file) => !known.has(fileIdentity(file))
                );
                // A later chooser round is replaced by one synthetic
                // aggregated event below. Stop the native event first, or
                // React/Streamlit would process the same FileList twice.
                if (selectedFiles.length > 0 && addsNewFile) {
                  event.stopImmediatePropagation();
                }
                syncAttach(mergeSelectedFiles(incomingFiles));
                setTimeout(() => {
                  const chatInput =
                    doc.querySelector(chatInputSelector);
                  if (chatInput) chatInput.focus();
                }, 60);
              });
            }
            syncAttach();

            const resetAttachmentChrome = () => {
              if (!active) return;
              plusBtn?.removeAttribute('title');
              if (plusBtn) {
                plusBtn.dataset.tooltip = defaultTooltip;
                plusBtn.setAttribute('aria-label', `选择附件上传方式（${defaultTooltip}）`);
              }
            };

            const handleSendClick = (event) => {
              const clicked = event.target?.closest?.(
                '[data-testid="stFormSubmitButton"] button'
              );
              if (!clicked || !chatForm.contains(clicked)) return;
              // Listen on the parent document in capture phase: Streamlit may
              // replace the submit button during the same click, so a handler
              // attached only to that ephemeral button is not reliable across
              // consecutive attachment messages.
              // Hide immediately, but leave the native FileList intact until
              // Streamlit has serialized the submitted form and the server
              // rotates the uploader key on its post-submit rerun.
              win.__cstfAttachmentClearRequested = true;
              selectedFiles = [];
              resetAttachmentChrome();
              renderAttachmentPreview([]);
            };
            doc.addEventListener('click', handleSendClick, true);

            if (chatForm.dataset.cstfAttachSubmitToken !== bindingToken) {
              chatForm.dataset.cstfAttachSubmitToken = bindingToken;
              chatForm.addEventListener('submit', () => {
                // Keep the original native FileList through this browser task:
                // Streamlit may serialize the form asynchronously. The
                // server-side uploader epoch is the authoritative clear once
                // the submission is accepted.
                win.__cstfAttachmentClearRequested = true;
                selectedFiles = [];
                resetAttachmentChrome();
                renderAttachmentPreview([]);
              });
            }

            const destroy = () => {
              active = false;
              doc.removeEventListener('click', handleSendClick, true);
              clearAttachmentPreview();
              if (bar.isConnected) bar.remove();
            };
            win.__cstfAttachmentController = {
              version: '2026-08-23-attachment-v7',
              token: bindingToken,
              uploaderEpoch,
              fileInput,
              sendButton,
              chatForm,
              bar,
              sync: syncAttach,
              destroy,
            };
            return true;
          };

          let tries = 0;
          const tick = () => {
            if (bindChatCompose() || tries++ > 40) return;
            win.setTimeout(tick, 120);
          };
          tick();

          const bindChatComposeObserver = () => {
            if (!doc || !doc.documentElement || doc.documentElement.nodeType !== 1) {
              win.setTimeout(bindChatComposeObserver, 120);
              return;
            }
            if (doc.body.dataset.cstfChatComposeObs === '1') return;
            doc.body.dataset.cstfChatComposeObs = '1';
            // Do not observe a parent-document node from the component
            // iframe: Chromium rejects that cross-realm target.  The bounded
            // polling loop is enough because Streamlit reruns replace the
            // compose form and re-execute this bridge.
            let n = 0;
            const poll = () => {
              if (win.__cstfAttachmentInstallerGeneration !== bindingToken) return;
              bindChatCompose();
              if (n++ < 20) win.setTimeout(poll, 180);
            };
            poll();
          };
          bindChatComposeObserver();

          // The component iframe may survive only for the current Streamlit
          // render. Keep one parent-page reconciler alive so a later rerun can
          // replace the native file input without leaving a stale + bar or
          // listeners attached to the disconnected input.
          const reconcileAttachment = () => {
            if (win.__cstfAttachmentInstallerGeneration !== bindingToken) {
              const current = win.__cstfAttachmentReconciler;
              if (current?.token === bindingToken && typeof current.stop === 'function') {
                current.stop();
              }
              return;
            }
            bindChatCompose();
          };
          const previousReconciler = win.__cstfAttachmentReconciler;
          if (previousReconciler && previousReconciler.token !== bindingToken
              && typeof previousReconciler.stop === 'function') {
            previousReconciler.stop();
          }
          const reconcileTimer = win.setInterval(reconcileAttachment, 260);
          win.__cstfAttachmentReconciler = {
            token: bindingToken,
            timer: reconcileTimer,
            stop: () => win.clearInterval(reconcileTimer),
          };
          reconcileAttachment();
        })();
        </script>
        """,
        height=0,
    )
    st.markdown('</div></div>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

_has_text = bool(prompt and prompt.strip())
# Only fingerprint files on an actual submit.  A large local TIFF should not
# be hashed on unrelated Streamlit reruns while the user is still composing.
if send_btn:
    uploaded_images = _dedupe_uploaded_images(uploaded_images)
_has_image = bool(uploaded_images)
_user_submitted = send_btn and (_has_text or _has_image)

if _user_submitted:
    if len(uploaded_images) > 6:
        st.warning("单轮最多支持 6 张图片附件；本轮仅处理前 6 张。")
        uploaded_images = uploaded_images[:6]

    preview_items = []
    preview_failures = []
    for uploaded_image in uploaded_images:
        preview_path, image_name = _save_chat_image_preview(uploaded_image)
        if preview_path:
            preview_items.append((preview_path, image_name))
        else:
            preview_failures.append(image_name or "未知附件")
    if preview_failures:
        st.warning(
            "以下附件无法生成本地预览，已跳过："
            + "、".join(preview_failures[:3])
            + ("…" if len(preview_failures) > 3 else "")
        )

    # Restore the original main.py behavior: every selected attachment is
    # sent to the active chat executor in the same multimodal request.  The
    # local preview remains separate and is cleared by the browser bridge
    # after submission.
    external_preview_paths = [path for path, _name in preview_items]

    used_default_prompt = False
    if _has_text:
        prompt = prompt.strip()
    else:
        prompt = "请结合上传的遥感/地图影像进行专业解译，说明可能的地物、波段组合或异常现象。"
        used_default_prompt = True
        st.toast("未输入文本，已自动使用默认解译指令发送。", icon="ℹ️")

    drawn_context = ""
    if isinstance(map_state, dict) and map_state.get("last_active_drawing"):
        geo_info = map_state["last_active_drawing"]["geometry"]
        drawn_context = "\n\n[系统空间上下文：用户已在地图上选择区域；完整几何仅保留在本地执行层，不发送给外部模型]"

    full_prompt_for_agent = prompt + drawn_context
    display_prompt = prompt
    if used_default_prompt:
        display_prompt = prompt + "\n\n`（未输入文本，系统已自动填充默认解译指令）`"

    user_msg = {"role": "user", "content": display_prompt}
    if preview_items:
        user_msg["image_preview_paths"] = [path for path, _name in preview_items]
        user_msg["image_names"] = [name for _path, name in preview_items]
        # Preserve the existing single-attachment fields for old sessions and
        # the current SQLite projection, which stores only a safe label.
        user_msg["image_preview_path"] = preview_items[0][0]
        user_msg["image_name"] = preview_items[0][1]

    with chat_box:
        with st.chat_message("user", avatar="🧑‍💻"):
            st.markdown('<div class="msg-role msg-role-user">用户</div>', unsafe_allow_html=True)
            st.markdown(display_prompt)
            _render_chat_attachment_previews(user_msg)
    if st.session_state.get("_conversation_store") is not None:
        from conversation_store import ensure_thread_id

        st.session_state._conversation_thread_id = ensure_thread_id(
            st.session_state._conversation_store,
            st.session_state.get("_conversation_thread_id"),
            create=True,
        )
    st.session_state.messages.append(user_msg)
    # File uploaders cannot be cleared by assigning their instantiated widget
    # state. Rotate the key so the post-submit rerun mounts a fresh input.
    st.session_state["_attachment_uploader_epoch"] = _attachment_uploader_epoch + 1

    with chat_box:
        with st.chat_message("assistant", avatar="🤖"):
            st.markdown('<div class="msg-role msg-role-assistant">智能体</div>', unsafe_allow_html=True)
            with st.spinner("🧠 智能体思考中..."):
                try:
                    import agent
                    import m5_agent_loop
                    import e1_agent_loop

                    # 待确认 M5 计划时：用户短句确认 → 直接执行，不绕开条件检查
                    _pending_m5 = st.session_state.get("_m5_pending_plan")
                    if (
                        isinstance(_pending_m5, dict)
                        and m5_agent_loop.is_m5_confirm_utterance(prompt)
                        and not st.session_state.is_running
                    ):
                        if not _pending_m5.get("ready"):
                            _block = "；".join(_pending_m5.get("blockers") or ["条件未满足"])
                            _msg = f"当前潮滩变化分析计划尚不可执行：{_block}"
                            st.warning(_msg)
                            st.session_state.messages.append(
                                {"role": "assistant", "content": _msg}
                            )
                        else:
                            queue_agent_command(
                                st.session_state,
                                {
                                    "pending_action": {
                                        "type": "run_m5",
                                        "confirmed": True,
                                        "task": _pending_m5.get("current_task"),
                                    }
                                },
                            )
                            _msg = (
                                "已确认潮滩变化分析计划，正在调用现有分析引擎。"
                                "完成后将根据真实报告与差异面回复。"
                            )
                            st.success(_msg)
                            st.session_state.messages.append(
                                {"role": "assistant", "content": _msg}
                            )
                        st.rerun()

                    # 待确认 E1 计划时：短句确认 → 直接执行
                    _pending_e1 = st.session_state.get("_e1_pending_plan")
                    if (
                        isinstance(_pending_e1, dict)
                        and e1_agent_loop.is_e1_confirm_utterance(prompt)
                        and not st.session_state.is_running
                    ):
                        # 若同时有 M5 待确认，优先已处理的 M5；此处仅当无 M5 待确认或用户明确提 E1
                        _pending_m5_chk = st.session_state.get("_m5_pending_plan")
                        if not isinstance(_pending_m5_chk, dict) or "e1" in (prompt or "").lower() or "一致" in (prompt or ""):
                            if not _pending_e1.get("ready"):
                                _block = "；".join(_pending_e1.get("blockers") or ["条件未满足"])
                                _msg = f"当前潮滩精度评价计划尚不可执行：{_block}"
                                st.warning(_msg)
                                st.session_state.messages.append(
                                    {"role": "assistant", "content": _msg}
                                )
                            else:
                                queue_agent_command(
                                    st.session_state,
                                    {
                                        "pending_action": {
                                            "type": "run_e1",
                                            "confirmed": True,
                                            "task": _pending_e1.get("current_task"),
                                        }
                                    },
                                )
                                _msg = (
                                    "已确认潮滩精度评价计划，正在调用现有评价引擎。"
                                    "完成后将根据真实报告回复交并比等指标。"
                                )
                                st.success(_msg)
                                st.session_state.messages.append(
                                    {"role": "assistant", "content": _msg}
                                )
                            st.rerun()

                    # 验收/高级：用户消息中直接粘贴 SYSTEM_COMMAND_JSON 时入队（不经 LLM）
                    if "[SYSTEM_COMMAND_JSON]" in (prompt or ""):
                        from agent_context_policy import raw_system_command_consent

                        if not raw_system_command_consent(st.session_state):
                            _msg = (
                                "为避免聊天文本绕过智能体边界，直接系统命令默认关闭。"
                                "本地验收请配置 CSTF_ALLOW_RAW_SYSTEM_COMMAND=1，"
                                "并勾选当前会话的开发授权后重试。"
                            )
                            st.warning(_msg)
                            st.session_state.messages.append({"role": "assistant", "content": _msg})
                            st.rerun()
                        cmd_result, clean_reply = process_agent_reply(st.session_state, prompt)
                        for _ce in cmd_result.errors:
                            st.warning(_ce)
                        _msg = clean_reply or (
                            f"已接收系统指令：{cmd_result.action_type or 'sidebar/map'}"
                        )
                        st.markdown(_msg)
                        st.session_state.messages.append(
                            {"role": "assistant", "content": _msg}
                        )
                        st.rerun()

                    try:
                        from dataset_assets import build_dataset_catalog_for_agent
                        _ds_cat = build_dataset_catalog_for_agent()
                    except Exception:
                        _ds_cat = ""

                    _sidebar_ctx = build_agent_sidebar_context(st.session_state)
                    # Phase D: AOI 空间上下文（紧凑摘要 + 能力推荐；AOI 选定 ≠ 确认）
                    _aoi_ctx_text = _aoi_sidebar_context()
                    if _aoi_ctx_text:
                        _sidebar_ctx = (_sidebar_ctx + "\n\n" + _aoi_ctx_text) if _sidebar_ctx else _aoi_ctx_text
                    # 能力状态快照（白名单，无路径/密钥）：仅首条消息或刷新后注入一次
                    _cap_snap_text = None
                    try:
                        import capability_registry as _cap
                        _cap_reg = st.session_state.get("_capability_reg")
                        if _cap_reg is None:
                            _cap_reg = _cap.CapabilityRegistry(
                                context=_cap.build_context(
                                    app_dir=os.path.dirname(os.path.abspath(__file__)),
                                    model_path=st.session_state.get("ui_model_path") or "",
                                    task=st.session_state.get("ui_selected_task") or "",
                                )
                            )
                            st.session_state._capability_reg = _cap_reg
                        if not st.session_state.get("_cap_snapshot_injected"):
                            _snap = _cap_reg.snapshot_for_agent()
                            _groups = _cap_reg.grouped_summary()
                            _lines = [
                                "可用: " + ",".join(_groups.get("AVAILABLE", [])),
                                "受限: " + ",".join(_groups.get("CONDITIONAL", [])),
                                "阻断: " + ",".join(_groups.get("BLOCKED", [])),
                                "未启用: " + ",".join(_groups.get("UNAVAILABLE", [])),
                                "未知: " + ",".join(_groups.get("UNKNOWN", [])),
                            ]
                            _reasons = {cid: e["summary"] for cid, e in _snap.items()}
                            _cap_snap_text = "\n".join(_lines) + "\n原因: " + str(_reasons)
                            st.session_state._cap_snapshot_injected = True
                    except Exception:
                        _cap_snap_text = None
                    from context_budget import bound_messages

                    reply = agent.chat_with_vlm(
                        full_prompt_for_agent,
                        bound_messages(
                            st.session_state.messages,
                            allow_spatial_metadata=bool(
                                st.session_state.get("agent_spatial_consent", False)
                            ),
                        ),
                        available_tasks=task_options,
                        dataset_catalog_text=_ds_cat or None,
                        sidebar_context=_sidebar_ctx,
                        capability_summary=_cap_snap_text,
                        allow_spatial_metadata=bool(
                            st.session_state.get("agent_spatial_consent", False)
                        ),
                        allow_external_media=True,
                        image_paths=external_preview_paths,
                    )

                    cmd_result, clean_reply = process_agent_reply(st.session_state, reply)
                    if cmd_result.applied:
                        for _ce in cmd_result.errors:
                            st.warning(_ce)
                        if cmd_result.action_type:
                            st.success(f"⚙️ 已接收指令：{cmd_result.action_type}")
                        display = clean_reply or "已更新系统设置。"
                        st.markdown(display)
                        st.session_state.messages.append({"role": "assistant", "content": display})
                        st.rerun()

                    parsed_map = _parse_agent_map_command(reply)
                    if parsed_map is not None:
                        target_lat, target_lon, target_zoom, _cmd_span = parsed_map
                        _map_label = _parse_agent_map_label(reply)
                        _map_command = {
                            "lat": target_lat,
                            "lon": target_lon,
                            "zoom": target_zoom,
                        }
                        if _map_label:
                            _map_command["label"] = _map_label
                        queue_agent_command(
                            st.session_state,
                            {"map": _map_command},
                        )
                        clean_reply = _strip_map_command_from_reply(reply)
                        if not clean_reply:
                            clean_reply = "已为您将地图视角定位至目标区域 🛰️。"
                        st.markdown(clean_reply)
                        st.session_state.messages.append({"role": "assistant", "content": clean_reply})
                        st.rerun()

                    parsed_pipe = _parse_agent_pipeline_command(reply)
                    if parsed_pipe is not None:
                        agent_task, agent_prob, agent_cnt, cmd_span = parsed_pipe
                        st.success(f"⚙️ 准备执行任务: {agent_task}")
                        clean_reply = reply.replace(cmd_span, "").replace("**", "").strip()
                        if not clean_reply:
                            clean_reply = "好的，已收到您的指令，正在后台为您执行调度任务..."
                        st.markdown(clean_reply)
                        st.session_state.messages.append({"role": "assistant", "content": clean_reply})
                        queue_agent_command(
                            st.session_state,
                            {
                                "sidebar_states": {
                                    "selected_task": agent_task,
                                    "prob_th": agent_prob,
                                    "min_cnt": agent_cnt,
                                },
                                "pending_action": {
                                    "type": "run_pipeline",
                                    "task": agent_task,
                                },
                            },
                        )
                        st.rerun()

                    _has_map_kw = re.search(r"COMMAND_UPDATE_MAP", reply, re.I) is not None
                    _has_pipe_kw = re.search(r"COMMAND_RUN_PIPELINE", reply, re.I) is not None
                    if (_has_map_kw or _has_pipe_kw) and parsed_map is None and parsed_pipe is None:
                        st.warning(
                            "模型提到了地图/跑图暗号但无法解析。推荐：让模型调用工具 `change_map_view`；"
                            "或正文含 `COMMAND_UPDATE_MAP|纬度|经度|缩放`（竖线）。"
                            "括号格式 `(lat,lon)` 已做兼容，若仍失败请重试。"
                        )
                        st.markdown(reply)
                        st.session_state.messages.append({"role": "assistant", "content": reply})
                    elif not _has_map_kw and not _has_pipe_kw:
                        st.markdown(reply)
                        st.session_state.messages.append({"role": "assistant", "content": reply})

                except Exception as e:
                    _append_debug_log(f"agent_chat_failed: {_format_agent_exception(e)}")
                    _error_reply = f"连接智能体出错：{_format_agent_exception(e)}"
                    st.error(_error_reply)
                    # Keep the failure in the bounded conversation stream so
                    # a send from History still has an observable assistant
                    # reply after the next rerun.
                    st.session_state.messages.append({"role": "assistant", "content": _error_reply})

    # Normal replies do not otherwise trigger a rerun. Re-render once so the
    # rotated uploader key clears every selected attachment from the composer.
    st.rerun()

# =======================================================
#  4. 后台流水线：启动 + 收尾 + 监控区（须放在 root_dir 等侧栏变量之后）
# =======================================================
def finalize_background_pipeline():
    shared = st.session_state.get("pipeline_shared")
    if not shared or not shared.get("done") or not st.session_state.is_running:
        return False
    _job_was_stopped = bool(st.session_state.get("stop_requested"))
    with shared["lock"]:
        # A stop request can race with a worker's final success signal.  The
        # user intent is authoritative: do not register/load artifacts or emit
        # success timeline events after an explicit interruption.
        from job_store import worker_success_is_committable

        success = worker_success_is_committable(
            shared.get("success", False), _job_was_stopped
        )
        asset_path = shared.get("asset_path")
        lines = list(shared.get("log_lines") or [])
        prog = int(shared.get("progress", 0))
        at_result = shared.get("autotune_result")
        m5_report = shared.get("m5_report")
        m5_verification = shared.get("m5_verification")
        m5_asset_id = shared.get("m5_asset_id")
        e1_report = shared.get("e1_report")
        e1_verification = shared.get("e1_verification")
        e1_asset_id = shared.get("e1_asset_id")
        job_kind = shared.get("job_kind")
        inference_result = shared.get("inference_result")
        inference_verification = shared.get("inference_verification")
        inference_asset_id = shared.get("asset_id")
    _failure_timeline_status = "CANCELLED" if _job_was_stopped else "FAILED"
    _tl_task = st.session_state.get("_tl_current_task") or "unknown"
    st.session_state.pipeline_log_snapshot = lines
    st.session_state.pipeline_progress_value = prog
    st.session_state.is_running = False
    st.session_state.pipeline_thread_started = False
    st.session_state.pipeline_shared = None
    st.session_state.pipeline_stop_event = None
    st.session_state.stop_requested = False
    st.session_state.executing_pipeline = False
    if at_result:
        st.session_state.autotune_result = at_result
    # ---- 本地潮滩推理可信执行闭环收尾 ----
    if inference_result is not None or inference_asset_id:
        _iv_ok = bool(inference_verification and inference_verification.get("ok") is True)
        if success and _iv_ok:
            # 真实结果写回 Copilot（只展示真实数据）
            try:
                import inference_agent_loop as _ial

                summary = _ial.summarize_inference_result_for_chat(
                    inference_result, inference_verification
                )
                st.session_state.messages = list(st.session_state.get("messages") or [])
                st.session_state.messages.append(
                    {"role": "assistant", "content": summary}
                )
                st.session_state._inference_last_summary = summary
            except Exception:
                pass
            _tl_add(_tl_task, "INFERENCE", "智能提取完成",
                    status="SUCCEEDED", tool="run_inference")
            _tl_add(_tl_task, "POST_PROCESS", "成果生成完成（潮滩栅格/矢量成果）",
                    status="SUCCEEDED", tool="post_engine")
            _tl_add(_tl_task, "VERIFY", "成果校验通过（潮滩栅格/矢量成果）",
                    status="SUCCEEDED", tool="verify_inference")
            if inference_asset_id:
                _tl_add(_tl_task, "REGISTER", "提取成果已登记",
                        status="SUCCEEDED", tool="register_inference",
                        artifacts=[str(inference_asset_id)])
            # 地图加载（不重建 iframe / 不重置相机）：成果路径已由校验确认
            _map_path = (inference_verification or {}).get("final_tif") or \
                        (inference_verification or {}).get("final_shp") or \
                        (asset_path or "")
            if _map_path and os.path.isfile(str(_map_path)):
                st.session_state.asset_override = os.path.abspath(str(_map_path))
                st.session_state._asset_pinned = True
                st.session_state.asset_just_loaded = True
                st.session_state._map_view_synced_for = None
                st.session_state._map_prefer_center = False
                st.session_state._globe_rev = int(st.session_state.get("_globe_rev", 0)) + 1
                _tl_add(_tl_task, "MAP", "成果已加载到地图",
                        status="SUCCEEDED", tool="map_load",
                        artifacts=[os.path.basename(str(_map_path))])
            _tl_add(_tl_task, "REPORT", "结果已回复智能助手",
                    status="SUCCEEDED", tool="report")
            # 动态能力状态刷新（含深度学习推理能力）
            try:
                import capability_registry as _cap
                _cap_reg = st.session_state.get("_capability_reg")
                if _cap_reg is not None:
                    _cap_reg.mark_runtime_verified("deep_learning_inference")
                    _cap_reg.bump()
                st.session_state._cap_snapshot_injected = False
            except Exception:
                pass
        else:
            _err = (inference_result or {}).get("error") or "提取失败（详见终端日志）"
            _tl_add(
                _tl_task,
                "INFERENCE",
                "提取已取消" if _job_was_stopped else f"提取未完成：{_err[:60]}",
                status=_failure_timeline_status,
                error=None if _job_was_stopped else _err,
                tool="run_inference",
            )
    # ---- GEE 影像下载可信执行闭环收尾 ----
    gee_result = shared.get("gee_result") if shared else None
    gee_verification = shared.get("gee_verification") if shared else None
    gee_dataset_id = shared.get("dataset_id") if shared else None
    if gee_result is not None or gee_dataset_id:
        _gv_ok = bool(gee_verification and gee_verification.get("ok") is True)
        if success and _gv_ok:
            try:
                import gee_agent_loop as _gal

                summary = _gal.summarize_gee_result_for_chat(gee_result, gee_verification)
                st.session_state.messages = list(st.session_state.get("messages") or [])
                st.session_state.messages.append(
                    {"role": "assistant", "content": summary}
                )
                st.session_state._gee_last_summary = summary
            except Exception:
                pass
            _tl_add(_tl_task, "GEE_EXPORT", "影像获取完成",
                    status="SUCCEEDED", tool="run_gee_download")
            _tl_add(_tl_task, "VERIFY", "影像校验通过",
                    status="SUCCEEDED", tool="verify_gee")
            if gee_dataset_id:
                _tl_add(_tl_task, "REGISTER", "影像数据已登记",
                        status="SUCCEEDED", tool="register_gee",
                        artifacts=[str(gee_dataset_id)])
            _tl_add(_tl_task, "REPORT", "结果已回复智能助手",
                    status="SUCCEEDED", tool="report")
            # 动态能力状态刷新（GEE 能力 / 推理能力 scene_count 感知）
            try:
                import capability_registry as _cap
                _cap_reg = st.session_state.get("_capability_reg")
                if _cap_reg is not None:
                    _cap_reg.mark_runtime_verified("gee_download")
                    _cap_reg.bump()
                st.session_state._cap_snapshot_injected = False
            except Exception:
                pass
        else:
            _err = (gee_result or {}).get("error") or "影像获取失败（详见终端日志）"
            _tl_add(
                _tl_task,
                "GEE_EXPORT",
                "影像获取已取消" if _job_was_stopped else f"获取未完成：{_err[:60]}",
                status=_failure_timeline_status,
                error=None if _job_was_stopped else _err,
                tool="run_gee_download",
            )
    if m5_report:
        st.session_state.m5_report = m5_report
        _lvl = m5_report.get("alert_level", "GREEN")
        _m5_verified = bool(m5_verification and m5_verification.get("ok") is True)
        if _lvl in ("RED", "YELLOW"):
            try:
                st.toast(
                    f"变化分析告警 [{_lvl}]: {m5_report.get('diagnostic_message', '')[:80]}",
                    icon="🚨" if _lvl == "RED" else "⚠️",
                )
            except Exception:
                pass
        # 独立 M5 闭环：把真实结果写回 Copilot，并加载地图
        if not _job_was_stopped and (job_kind == "m5" or (success and m5_verification is not None)):
            try:
                import m5_agent_loop

                summary = m5_agent_loop.summarize_m5_report_for_chat(
                    m5_report, m5_verification
                )
                st.session_state.messages = list(st.session_state.get("messages") or [])
                st.session_state.messages.append(
                    {"role": "assistant", "content": summary}
                )
                st.session_state._m5_last_summary = summary
            except Exception:
                pass
            if m5_verification is not None:
                _tl_add(
                    _tl_task,
                    "VERIFY",
                    "变化分析输出校验通过" if _m5_verified else "变化分析输出校验未通过",
                    status="SUCCEEDED" if _m5_verified else _failure_timeline_status,
                    error=None if _m5_verified else "输出校验未完全通过；未登记成果。",
                    tool="verify_m5",
                )
            map_path = asset_path
            if not map_path:
                try:
                    import m5_agent_loop

                    map_path = m5_agent_loop.pick_m5_map_path(m5_report)
                except Exception:
                    map_path = None
            if _m5_verified and map_path and os.path.isfile(str(map_path)):
                st.session_state.asset_override = map_path
                st.session_state._asset_pinned = True
                st.session_state.asset_just_loaded = True
                st.session_state._map_view_synced_for = None
                st.session_state._map_prefer_center = False
                st.session_state._globe_rev = int(st.session_state.get("_globe_rev", 0)) + 1
    if e1_report:
        st.session_state.e1_report = e1_report
        _e1_verified = bool(e1_verification and e1_verification.get("ok") is True)
        if not _job_was_stopped:
            try:
                st.toast("精度评价已完成", icon="📊")
            except Exception:
                pass
        # 独立 E1 闭环：真实指标写回 Copilot，并优先加载分歧热力图
        if not _job_was_stopped and (job_kind == "e1" or (success and e1_verification is not None)):
            try:
                import e1_agent_loop

                summary = e1_agent_loop.summarize_e1_report_for_chat(
                    e1_report, e1_verification
                )
                st.session_state.messages = list(st.session_state.get("messages") or [])
                st.session_state.messages.append(
                    {"role": "assistant", "content": summary}
                )
                st.session_state._e1_last_summary = summary
            except Exception:
                pass
            if e1_verification is not None:
                _tl_add(
                    _tl_task,
                    "VERIFY",
                    "精度评价输出校验通过" if _e1_verified else "精度评价输出校验未通过",
                    status="SUCCEEDED" if _e1_verified else _failure_timeline_status,
                    error=None if _e1_verified else "输出校验未完全通过；未登记成果。",
                    tool="verify_e1",
                )
            map_path = asset_path
            if not map_path:
                try:
                    import e1_agent_loop

                    map_path = e1_agent_loop.pick_e1_map_path(e1_report)
                except Exception:
                    map_path = None
            if _e1_verified and map_path and os.path.isfile(str(map_path)):
                st.session_state.asset_override = map_path
                st.session_state._asset_pinned = True
                st.session_state.asset_just_loaded = True
                st.session_state._map_view_synced_for = None
                st.session_state._map_prefer_center = False
                st.session_state._globe_rev = int(st.session_state.get("_globe_rev", 0)) + 1
    m4_result = shared.get("m4_result") if shared else None
    if m4_result:
        st.session_state.m4_last_result = m4_result
    # ---- 端到端潮滩分析 Workflow 收尾 ----
    workflow_result = shared.get("workflow_result") if shared else None
    if workflow_result:
        st.session_state.workflow_last_result = workflow_result
        wf_status = workflow_result.get("status")
        try:
            import workflow_orchestrator as _workflow_orchestrator

            _wf_timeline_status = _workflow_orchestrator.workflow_result_timeline_status(wf_status)
        except Exception:
            _wf_timeline_status = (
                "WARNING" if wf_status == "COMPLETED_WITH_WARNINGS"
                else ("SUCCEEDED" if success else "FAILED")
            )
        try:
            summary = workflow_result.get("summary") or ""
            st.session_state.messages = list(st.session_state.get("messages") or [])
            st.session_state.messages.append(
                {"role": "assistant", "content": summary}
            )
            st.session_state._workflow_last_summary = summary
        except Exception:
            pass
        step_line = " | ".join(
            f"{sid}:{s.get('status')}"
            for sid, s in (workflow_result.get("steps") or {}).items()
        )
        if success:
            _tl_add(_tl_task, "WORKFLOW",
                    f"一键潮滩分析完成（{uil.get_status_label(wf_status)}）",
                    status=_wf_timeline_status, progress=100,
                    tool="run_workflow",
                    artifacts=[str(workflow_result.get("workflow_id") or "")])
            _tl_add(_tl_task, "WORKFLOW", f"步骤: {step_line}",
                    status=_wf_timeline_status, tool="run_workflow")
            try:
                st.balloons()
            except Exception:
                pass
        else:
            _tl_add(_tl_task, "WORKFLOW", f"一键潮滩分析未完成（{uil.get_status_label(wf_status)}）",
                    status=_failure_timeline_status,
                    error=None if _job_was_stopped else step_line,
                    tool="run_workflow")
    # 推理闭环已在上面自行登记 EXECUTE/REGISTER/VERIFY/MAP/REPORT，这里避免重复
    _inference_handled = inference_result is not None or inference_asset_id is not None
    _gee_handled = gee_result is not None or gee_dataset_id is not None
    _workflow_handled = workflow_result is not None
    _m5_independent_handled = job_kind == "m5" and m5_report is not None
    _e1_independent_handled = job_kind == "e1" and e1_report is not None
    _optional_postflight_warning = bool(
        success
        and job_kind not in ("m5", "e1")
        and (
            (m5_report is not None and not (m5_verification and m5_verification.get("ok") is True))
            or (e1_report is not None and not (e1_verification and e1_verification.get("ok") is True))
        )
    )
    if success and asset_path and job_kind not in ("m5", "e1") and not _inference_handled and not _gee_handled and not _workflow_handled:
        st.session_state.asset_override = asset_path
    if success and not _inference_handled and not _gee_handled and not _workflow_handled:
        _tl_add(
            _tl_task,
            "EXECUTE",
            (f"任务执行完成（{job_kind or 'pipeline'}，含可选后置警告）"
             if _optional_postflight_warning
             else f"任务执行完成（{job_kind or 'pipeline'}）"),
            status="WARNING" if _optional_postflight_warning else "SUCCEEDED",
            progress=100,
            tool=job_kind or "run_pipeline",
        )
        if asset_path:
            _tl_add(_tl_task, "REGISTER", "成果已登记",
                    status="SUCCEEDED", tool="register_asset",
                    artifacts=[os.path.basename(str(asset_path))])
        if m5_report:
            _m5_ok = bool(m5_verification and m5_verification.get("ok") is True)
            _tl_add(
                _tl_task,
                "VERIFY",
                "变化分析校验通过" if _m5_ok else "变化分析输出校验未完全通过",
                status="SUCCEEDED" if _m5_ok else "WARNING",
                error=None if _m5_ok else "输出校验未完全通过；未登记成果。",
                tool="verify_m5",
            )
            if _m5_ok and m5_asset_id:
                _tl_add(_tl_task, "REGISTER", "变化分析成果已登记",
                        status="SUCCEEDED", tool="register_m5",
                        artifacts=[str(m5_asset_id)])
        if e1_report:
            _e1_ok = bool(e1_verification and e1_verification.get("ok") is True)
            _tl_add(
                _tl_task,
                "VERIFY",
                "精度评价校验通过" if _e1_ok else "精度评价输出校验未完全通过",
                status="SUCCEEDED" if _e1_ok else "WARNING",
                error=None if _e1_ok else "输出校验未完全通过；未登记成果。",
                tool="verify_e1",
            )
            if _e1_ok and e1_asset_id:
                _tl_add(_tl_task, "REGISTER", "精度评价成果已登记",
                        status="SUCCEEDED", tool="register_e1",
                        artifacts=[str(e1_asset_id)])
        if not _optional_postflight_warning:
            try:
                st.balloons()
            except Exception:
                pass
    else:
        if (
            not _inference_handled
            and not _gee_handled
            and not _workflow_handled
            and not _m5_independent_handled
            and not _e1_independent_handled
        ):
            _tl_add(_tl_task, "EXECUTE",
                    "任务已取消" if _job_was_stopped else "任务执行失败",
                    status=_failure_timeline_status,
                    error=None if _job_was_stopped else "任务执行失败，详见终端日志",
                    tool=job_kind or "run_pipeline")
        time.sleep(2)
    _job_status = "CANCELLED" if _job_was_stopped else ("SUCCEEDED" if success else "FAILED")
    if not success and (_m5_independent_handled or _e1_independent_handled):
        _postflight_verification = m5_verification if _m5_independent_handled else e1_verification
        if isinstance(_postflight_verification, dict) and _postflight_verification.get("ok") is False:
            _job_status = "WARNING"
    if _optional_postflight_warning:
        _job_status = "WARNING"
    if workflow_result and workflow_result.get("status") == "COMPLETED_WITH_WARNINGS" and success:
        _job_status = "WARNING"
    _job_error = (
        None
        if success
        else (
            "用户已请求中断；本次成果未登记。"
            if _job_was_stopped
            else "后台任务未完成；请查看时间线并确认后重试。"
        )
    )
    if _job_status == "WARNING" and _optional_postflight_warning:
        _job_error = "主流程已完成，但可选后置输出校验未完全通过；相关成果未登记或加载。"
    elif _job_status == "WARNING" and not success:
        _job_error = "输出校验未完全通过；成果未登记或加载。"
    _job_transition(
        _job_status,
        progress=100 if success else prog,
        artifacts=[x for x in (asset_path, inference_asset_id, gee_dataset_id) if x],
        error=_job_error,
        metadata={"job_kind": job_kind or "pipeline"},
    )
    st.session_state.pop("_active_job_id", None)
    return True


# ---- 自适应调参后台线程 ----

def run_autotune_sync(ctx, shared, stop_event):
    """后台线程：自适应参数搜索（假设 Mask 已存在）。"""
    logs_local = []

    def check_stop():
        return stop_event.is_set()

    def push_log(msg):
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        line = f"[{ts}] root@autotune: {msg}"
        logs_local.append(line)
        with shared["lock"]:
            shared["log_lines"] = logs_local[-40:]
        print(msg)

    def push_progress(pct):
        with shared["lock"]:
            shared["progress"] = int(min(100, max(0, pct)))

    def push_status(kind, text):
        with shared["lock"]:
            shared["status"] = (kind, text)

    push_progress(0)
    push_status("info", "🔬 参数自动优化启动…")

    task = ctx["task"]
    task_options_local = ctx["task_options"]
    actual_task = task
    for opt in task_options_local:
        if task in opt:
            actual_task = opt
            break

    input_dir = os.path.join(ctx["root_dir"], actual_task)
    mask_out_dir = os.path.join(ctx["mask_root"], actual_task)
    final_out_dir = os.path.join(ctx["final_root"], actual_task)
    os.makedirs(final_out_dir, exist_ok=True)

    push_log(f"TASK: {actual_task} | REF: {ctx['reference_id']} | OBJ: {ctx['objective']}")
    push_progress(80)
    push_status("info", "🔬 正在搜索最优参数…")

    try:
        import auto_tune
        from dataset_assets import get_primary_path as ds_get_path

        ref_shp = ds_get_path(ctx["reference_id"])
        if not ref_shp:
            push_status("error", f"❌ 参考真值 {ctx['reference_id']} 文件不存在")
            return False

        result = auto_tune.run_adaptive_tuning(
            source_folder=input_dir,
            mask_folder=mask_out_dir,
            final_out_dir=final_out_dir,
            task_name=actual_task,
            reference_shp_path=ref_shp,
            shp_clip_path=ctx.get("shp_path"),
            task_aoi_shp_path=ctx.get("task_aoi_shp"),
            objective=ctx["objective"],
            logger=push_log,
            progress_callback=push_progress,
            stop_callback=check_stop,
        )

        _best_shp = str((result or {}).get("best_shp_path") or "")
        if result and _best_shp and os.path.isfile(_best_shp) and os.path.getsize(_best_shp) > 0:
            register_asset(actual_task, result["best_prob"], result["best_cnt"], _best_shp)
            _run_m5_phase(
                ctx, shared, _best_shp, actual_task,
                result["best_prob"], result["best_cnt"], push_log, check_stop,
            )
            _run_e1_phase(ctx, shared, result["best_shp_path"], actual_task, push_log, check_stop)
            push_status(
                "success",
                f"🏆 最优参数: P={result['best_prob']:.2f} C={result['best_cnt']} | "
                f"交并比={result['best_iou'] * 100:.1f}% F1={result['best_f1'] * 100:.1f}%",
            )
            with shared["lock"]:
                shared["asset_path"] = _best_shp
                shared["autotune_result"] = result
            return True
        if result and result.get("best_shp_path"):
            push_log("[AutoTune] 引擎返回结果，但最佳成果文件缺失或为空；不登记为成功。")
        push_status("warning", "参数优化未能得出结果（可能被中断或无有效真值像元）。")
        return False
    except Exception as e:
        safe = safe_error_summary(e)
        push_log(f"[ERROR] {safe}")
        push_status("error", f"参数优化异常: {safe}")
        import traceback
        traceback.print_exc()
        return False


def _autotune_worker_entry(ctx, shared, stop_event):
    ok = False
    try:
        ok = run_autotune_sync(ctx, shared, stop_event)
    except Exception as e:
        _record_worker_exception(shared, "参数优化线程异常", e)
    finally:
        with shared["lock"]:
            shared["success"] = ok
            shared["done"] = True


def maybe_start_pipeline_thread():
    # ---- 自适应优化分支 ----
    at_info = st.session_state.get("pending_autotune")
    if at_info and st.session_state.is_running and not st.session_state.pipeline_thread_started:
        st.session_state.pop("pending_autotune", None)
        st.session_state.pipeline_thread_started = True
        st.session_state.pipeline_log_snapshot = []
        st.session_state.pipeline_progress_value = 0
        st.session_state.executing_pipeline = True
        st.session_state._tl_current_task = at_info["task"]
        _at_request = at_info.get("execution_request")
        if not isinstance(_at_request, dict):
            try:
                from execution_request import attach_execution_request

                _at_request = attach_execution_request(
                    at_info,
                    confirmation_source=str(at_info.get("confirmation_source") or "ui"),
                ).get("execution_request")
            except Exception:
                _at_request = None
        _at_pending = {
            "task": at_info.get("task") or "unknown",
            "mode": "autotune",
            "plan_id": at_info.get("plan_id"),
            "execution_request": _at_request,
        }
        _at_job = _job_create_for_pending(_at_pending, status="QUEUED")
        if _at_job is None or _get_job_store().claim(_at_job.job_id) is None:
            st.error("任务账本不可用或任务已被占用，未启动参数优化。")
            st.session_state.is_running = False
            return

        stop_ev = threading.Event()
        st.session_state.pipeline_stop_event = stop_ev
        _tl_add(at_info["task"], "EXECUTE", "参数自动优化已启动",
                status="RUNNING", tool="run_autotune", progress=0)
        shared = {
            "lock": threading.Lock(),
            "job_id": _at_job.job_id,
            "log_lines": [],
            "progress": 0,
            "status": ("info", "🔬 正在启动参数优化线程…"),
            "done": False,
            "success": False,
            "asset_path": None,
            "autotune_result": None,
            "m5_report": None,
            "e1_report": None,
        }
        st.session_state.pipeline_shared = shared
        ctx = {
            "root_dir": root_dir,
            "mask_root": mask_root,
            "final_root": final_root,
            "shp_path": shp_path,
            "task_options": list(task_options),
            "task": at_info["task"],
            "reference_id": at_info["reference_id"],
            "objective": at_info["objective"],
            "task_aoi_shp": at_info.get("task_aoi_shp"),
            "m5_enabled": m5_enabled,
            "m5_baseline_shp": m5_baseline_shp,
            "e1_enabled": e1_enabled,
            "e1_data_root": e1_data_root,
            "e1_reference": e1_reference,
            "e1_compare_sources": list(e1_compare_sources),
            "e1_export_maps": e1_export_maps,
            "e1_export_heatmap": e1_export_heatmap,
        }
        threading.Thread(target=_autotune_worker_entry, args=(ctx, shared, stop_ev), daemon=True).start()
        return

    # ---- 常规推理 / M4 / 独立 M5 分支 ----
    if not (st.session_state.pending_task and st.session_state.is_running):
        return
    if st.session_state.pipeline_thread_started:
        return
    task_info = st.session_state.pending_task
    try:
        from execution_request import attach_execution_request

        _existing_request = task_info.get("execution_request") if isinstance(task_info, dict) else {}
        _confirmation_source = str((_existing_request or {}).get("confirmation_source") or "ui")
        task_info = attach_execution_request(task_info, confirmation_source=_confirmation_source)

        # 兼容旧的 run_pipeline pending schema：在真正启动线程前补齐可信计划，
        # 让 Agent/手动入口都落到同一生产执行适配器，而不是旧同步算法。
        if task_info.get("mode") == "dl" and not task_info.get("inference_plan"):
            from agent_command_bridge import propose_inference_plan as _propose_inference

            _dl_plan, _dl_errors = _propose_inference(st.session_state, task_info)
            if not _dl_plan.get("ready"):
                st.error("推理计划校验未通过：" + "；".join(_dl_errors or _dl_plan.get("blockers") or ["未知条件"]))
                st.session_state.pending_task = None
                st.session_state.is_running = False
                return
            task_info["inference_plan"] = _dl_plan
            task_info["plan_id"] = _dl_plan.get("plan_id")
            task_info = attach_execution_request(task_info, confirmation_source=_confirmation_source)
        elif task_info.get("mode") == "index" and not task_info.get("index_plan"):
            import index_agent_loop as _index_loop

            _idx_plan = _index_loop.build_index_plan(
                task=task_info.get("task") or "",
                input_dir=os.path.join(root_dir or "", task_info.get("task") or ""),
                output_dir=os.path.join(final_root or "", task_info.get("task") or ""),
                points_shp=task_info.get("points_shp") or points_shp or "",
                force_rerun=bool(task_info.get("force_rerun")),
            )
            _idx_ok, _idx_errors = _index_loop.validate_index_plan(_idx_plan)
            if not _idx_ok:
                st.error("指数法计划校验未通过：" + "；".join(_idx_errors))
                st.session_state.pending_task = None
                st.session_state.is_running = False
                return
            _idx_plan["ready"] = True
            _idx_plan["status"] = "confirmed"
            task_info["index_plan"] = _idx_plan
            task_info["plan_id"] = task_info.get("plan_id") or _idx_plan.get("plan_id")
            task_info = attach_execution_request(task_info, confirmation_source=_confirmation_source)
        st.session_state.pending_task = task_info
    except (TypeError, ValueError) as _exec_req_err:
        st.error(f"执行请求校验失败：{_exec_req_err}")
        st.session_state.pending_task = None
        st.session_state.is_running = False
        return
    _job_record = _job_create_for_pending(task_info, status="QUEUED")
    if _job_record is None:
        st.error("任务账本不可用，已停止启动后台任务；请检查 data 目录权限后重试。")
        st.session_state.pending_task = None
        st.session_state.is_running = False
        return
    if _get_job_store().claim(_job_record.job_id) is None:
        st.error("该任务已被另一个执行实例占用，未重复启动。")
        st.session_state.pending_task = None
        st.session_state.is_running = False
        return
    st.session_state.pending_task = None
    st.session_state.pipeline_thread_started = True
    st.session_state.pipeline_log_snapshot = []
    st.session_state.pipeline_progress_value = 0
    st.session_state.executing_pipeline = True
    st.session_state._tl_current_task = task_info.get("task") or "unknown"

    stop_ev = threading.Event()
    st.session_state.pipeline_stop_event = stop_ev
    _mode_txt = {"m4": "获取卫星影像", "index": "指数法提取", "dl": "深度学习提取"}.get(
        task_info.get("mode"), task_info.get("mode") or "提取"
    )
    _tl_add(task_info.get("task") or "unknown", "EXECUTE",
            f"任务已启动（{_mode_txt}）",
            status="RUNNING", tool="run_pipeline", progress=0)
    shared = {
        "lock": threading.Lock(),
        "job_id": _job_record.job_id,
        "log_lines": [],
        "progress": 0,
        "status": ("info", "正在启动后台线程…"),
        "done": False,
        "success": False,
        "asset_path": None,
        "m5_report": None,
        "m5_verification": None,
        "m5_asset_id": None,
        "e1_report": None,
        "e1_verification": None,
        "e1_asset_id": None,
        "job_kind": task_info.get("mode"),
    }
    st.session_state.pipeline_shared = shared

    # 本地潮滩推理可信执行闭环（不进入 run_pipeline_sync 旧路径）
    if task_info.get("inference_plan") or task_info.get("mode") == "dl_inference":
        shared["status"] = ("info", "正在启动潮滩智能提取…")
        _tl_add(task_info.get("task") or "unknown", "INFERENCE",
                "潮滩智能提取已启动", status="RUNNING", tool="run_inference", progress=0)
        ctx = {
            "root_dir": root_dir,
            "final_root": final_root,
            "mask_root": mask_root,
            "model_path": model_path,
            "shp_path": shp_path,
            "task_options": list(task_options),
            "task": task_info.get("task"),
            "inference_plan": task_info.get("inference_plan"),
        }
        threading.Thread(
            target=_inference_worker_entry,
            args=(ctx, shared, stop_ev),
            daemon=True,
        ).start()
        return

    # GEE 影像下载可信执行闭环（不跑推理）
    if task_info.get("mode") == "gee":
        shared["status"] = ("info", "正在启动影像获取…")
        _tl_add(task_info.get("task") or "unknown", "GEE_EXPORT",
                "影像获取已启动", status="RUNNING", tool="run_gee_download", progress=0)
        ctx = {
            "root_dir": root_dir,
            "task": task_info.get("task"),
            "gee_plan": task_info.get("gee_plan"),
        }
        threading.Thread(
            target=_gee_worker_entry,
            args=(ctx, shared, stop_ev),
            daemon=True,
        ).start()
        return

    # 独立 M5 闭环（不跑推理）
    if task_info.get("mode") == "m5":
        shared["status"] = ("info", "正在启动潮滩变化分析…")
        ctx = {
            "root_dir": root_dir,
            "final_root": final_root,
            "task_options": list(task_options),
            "task": task_info.get("task"),
            "prob": task_info.get("prob"),
            "cnt": task_info.get("cnt"),
            "m5": task_info.get("m5") or {},
            "m5_baseline_shp": m5_baseline_shp,
        }
        threading.Thread(
            target=_m5_worker_entry,
            args=(ctx, shared, stop_ev),
            daemon=True,
        ).start()
        return

    # 独立 E1 闭环（不跑推理）
    if task_info.get("mode") == "e1":
        shared["status"] = ("info", "正在启动潮滩精度评价…")
        shared["e1_verification"] = None
        ctx = {
            "root_dir": root_dir,
            "final_root": final_root,
            "task_options": list(task_options),
            "task": task_info.get("task"),
            "prob": task_info.get("prob"),
            "cnt": task_info.get("cnt"),
            "task_aoi_shp": task_aoi_shp,
            "e1": task_info.get("e1") or {},
            "e1_data_root": e1_data_root,
            "e1_reference": e1_reference,
            "e1_compare_sources": list(e1_compare_sources),
            "e1_export_maps": e1_export_maps,
            "e1_export_heatmap": e1_export_heatmap,
        }
        threading.Thread(
            target=_e1_worker_entry,
            args=(ctx, shared, stop_ev),
            daemon=True,
        ).start()
        return

    # 端到端潮滩分析 Workflow（复用子闭环编排）
    if task_info.get("mode") == "workflow":
        shared["status"] = ("info", "正在启动一键潮滩分析（获取影像→提取→评价/变化→报告）…")
        ctx = {
            "root_dir": root_dir,
            "final_root": final_root,
            "mask_root": mask_root,
            "model_path": model_path,
            "shp_path": shp_path,
            "e1_data_root": e1_data_root,
            "e1_reference": e1_reference,
            "task": task_info.get("task"),
            "workflow_plan": task_info.get("workflow_plan"),
            # _active_aoi 可能是 AOIContext 对象，先序列化为 dict 再交给编排器
            "aoi": _aoi_state_to_dict(st.session_state),
            "registry": None,
            "registry_path": os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "assets_registry.json"),
            "report_output_dir": None,
            "baseline_task": None,
        }
        threading.Thread(
            target=_workflow_worker_entry,
            args=(ctx, shared, stop_ev),
            daemon=True,
        ).start()
        return

    ctx = {
        "root_dir": root_dir,
        "mask_root": mask_root,
        "final_root": final_root,
        "model_path": model_path,
        "shp_path": shp_path,
        "task_options": list(task_options),
        "task": task_info.get("task"),
        "prob": task_info.get("prob", 0.05),
        "cnt": task_info.get("cnt", 2),
        "mode": task_info.get("mode", "dl"),
        "index_plan": task_info.get("index_plan"),
        "points_shp": task_info.get("points_shp"),
        "force_rerun": task_info.get("force_rerun", False),
        "m4": task_info.get("m4"),
        "task_aoi_shp": task_aoi_shp,
        "m5_enabled": m5_enabled,
        "m5_baseline_shp": m5_baseline_shp,
        "e1_enabled": e1_enabled,
        "e1_data_root": e1_data_root,
        "e1_reference": e1_reference,
        "e1_compare_sources": list(e1_compare_sources),
        "e1_export_maps": e1_export_maps,
        "e1_export_heatmap": e1_export_heatmap,
    }
    threading.Thread(
        target=_pipeline_worker_entry,
        args=(ctx, shared, stop_ev),
        daemon=True,
    ).start()


def _pipeline_monitor_inner(render: bool = True):
    shared = st.session_state.get("pipeline_shared")
    if shared and shared.get("done") and st.session_state.is_running:
        if finalize_background_pipeline():
            st.rerun()
        return

    if shared:
        with shared["lock"]:
            lines = list(shared.get("log_lines") or [])
            prog = int(shared.get("progress", 0))
            status = shared.get("status", ("info", ""))
            shared_job_id = shared.get("job_id")
    else:
        lines = list(st.session_state.get("pipeline_log_snapshot") or [])
        prog = int(st.session_state.get("pipeline_progress_value", 0))
        status = ("info", "")
        shared_job_id = None

    active_job_id = shared_job_id or st.session_state.get("_active_job_id")
    if active_job_id and st.session_state.get("is_running"):
        _job_progress_update(
            prog,
            job_id=active_job_id,
            metadata={"phase": "RUNNING", "job_kind": (shared or {}).get("job_kind")},
        )

    # Keep the monitor alive while the drawer is collapsed so completion,
    # cleanup and ledger finalization still run, but emit no elements that
    # could reserve a hidden row in the map column.
    if not render:
        return

    st.markdown('<div class="deck-section-title">⏳ 任务执行状态</div>', unsafe_allow_html=True)
    st.progress(min(100, max(0, prog)))
    if isinstance(status, (list, tuple)) and len(status) >= 2:
        kind, text = status[0], status[1]
    else:
        kind, text = "info", ""
    if text:
        if kind == "error":
            st.error(text)
        elif kind == "success":
            st.success(text)
        elif kind == "warning":
            st.warning(text)
        else:
            st.info(text)
    elif st.session_state.is_running:
        st.caption("后台任务运行中，日志与进度将自动刷新…")

    recovered_records = st.session_state.get("_job_recovery_records") or []
    if recovered_records:
        with st.expander(f"🧾 已恢复的任务账本（{len(recovered_records)}）", expanded=False):
            st.caption("这些任务在进程退出前未进入终态，已标记为 INTERRUPTED；不会自动重跑。")
            for record in recovered_records[:20]:
                st.markdown(
                    f"- `{record.job_id}` · {record.task} · {record.kind} · "
                    f"进度 {record.progress}% · `INTERRUPTED`"
                )
                # 仅为普通推理恢复“计划”入口；不复用旧参数直接启动，用户仍需审阅并确认。
                if str(record.kind) == "dl" and st.button(
                    "重新生成推理计划",
                    key=f"replan_interrupted_{record.job_id}",
                    use_container_width=True,
                ):
                    queue_agent_command(
                        st.session_state,
                        {
                            "sidebar_states": {
                                "selected_task": str(record.task),
                                "inference_mode": "深度学习",
                                "force_rerun": True,
                            },
                            "pending_action": {
                                "type": "propose_inference",
                                "task": str(record.task),
                                "force_rerun": True,
                            },
                        },
                    )
                    st.session_state["_job_recovery_replan_notice"] = (
                        f"已为中断任务 `{record.task}` 重新生成推理计划；请检查侧栏参数并确认后执行。"
                    )
                    st.rerun()

    st.markdown('<div class="deck-section-title">🖥️ 系统终端日志</div>', unsafe_allow_html=True)
    with st.container(height=LOG_PANEL_HEIGHT, border=False):
        if lines:
            st.code("\n".join(lines), language="bash")
        elif st.session_state.is_running:
            st.caption("任务启动中…")
        else:
            st.caption("暂无日志。运行提取或影像获取后，终端输出将显示在此处。")

    # ---- Phase C: 任务时间线（倒序事件 + 阶段/状态徽章）----
    try:
        _tl = _get_task_timeline()
        _tl_events = _tl.events(limit=12)
        if _tl_events:
            with st.expander(f"📋 任务进度（{len(_tl_events)}）", expanded=False):
                if _tl.restored_from == "disk":
                    st.caption("历史记录（进程重启后恢复），非实时状态")
                _status_icons = {
                    "PENDING": "⏳", "WAITING_CONFIRMATION": "❓",
                    "QUEUED": "🕓", "RUNNING": "🔵", "SUCCEEDED": "✅",
                    "FAILED": "❌", "BLOCKED": "⛔", "CANCELLED": "🚫",
                    "WARNING": "⚠️",
                }
                for _ev in reversed(_tl_events):
                    _icon = _status_icons.get(_ev.status, "•")
                    _pct = f" {_ev.progress}%" if _ev.progress is not None else ""
                    st.markdown(
                        f"`{_ev.updated_at[11:19]}` {_icon} **{uil.get_phase_label(_ev.phase)}**/"
                        f"{uil.get_status_label(_ev.status)} {_ev.message}{_pct}"
                    )
                # ---- Phase E: PDF 报告入口（任务完成后生成）----
                _tl_col1, _tl_col2 = st.columns(2)
                with _tl_col1:
                    if st.button(
                        "📄 生成成果报告",
                        key="_btn_gen_pdf_report",
                        disabled=isinstance(st.session_state.get("_pending_report_confirm"), dict),
                    ):
                        _queue_report_plan("pdf")
                with _tl_col2:
                    if st.button(
                        "🗺️ 生成监测报告",
                        key="_btn_gen_asset_report",
                        disabled=isinstance(st.session_state.get("_pending_report_confirm"), dict),
                    ):
                        _queue_report_plan("asset")
                    _amsg = st.session_state.get("_asset_report_msg")
                    if _amsg:
                        if _amsg.get("level") == "success":
                            st.markdown("✅ **成果报告已生成**")
                            st.code(_amsg.get("path", ""))
                            try:
                                with open(_amsg["path"], "rb") as _pf:
                                    st.download_button(
                                        "⬇️ 下载成果报告",
                                        _pf.read(),
                                        file_name=os.path.basename(_amsg["path"]),
                                        mime="application/pdf",
                                        key="_btn_dl_asset_report",
                                    )
                            except Exception:
                                pass
                        else:
                            st.markdown(f"⚠️ **{_amsg.get('text', '未知错误')}**")
                        for _w in (_amsg.get("warnings") or []):
                            st.caption(_w)
                _pending_report = st.session_state.get("_pending_report_confirm")
                if isinstance(_pending_report, dict):
                    _kind_label = "成果报告" if _pending_report.get("kind") == "pdf" else "监测报告"
                    _report_plan_valid = (
                        str(_pending_report.get("task_id") or "")
                        == str(_report_task_id() or "")
                        and str(_pending_report.get("task_id") or "") not in {"", "unknown"}
                    )
                    st.info(
                        f"已生成{_kind_label}计划（任务 `{_pending_report.get('task_id') or '—'}`），"
                        "确认后才会读取成果并生成 PDF。"
                    )
                    if not _report_plan_valid:
                        st.warning("当前任务时间线已变化，请取消旧计划后重新生成报告。")
                    _rpc1, _rpc2 = st.columns(2)
                    with _rpc1:
                        if st.button(
                            f"确认生成{_kind_label}",
                            key="confirm_report_plan_btn",
                            type="primary",
                            use_container_width=True,
                            disabled=not _report_plan_valid,
                        ):
                            _kind = str(_pending_report.get("kind") or "pdf")
                            _report_plan_id = _pending_report.get("plan_id")
                            st.session_state.pop("_pending_report_confirm", None)
                            st.session_state["_active_report_plan_id"] = _report_plan_id
                            _tl_add(
                                str(_pending_report.get("task_id") or "unknown"),
                                "CONFIRM",
                                f"{_kind_label}计划已确认",
                                status="SUCCEEDED",
                                plan_id=_report_plan_id,
                                tool="report_generator" if _kind == "pdf" else "asset_report_engine",
                            )
                            if _kind == "asset":
                                _build_asset_report()
                            else:
                                _build_pdf_report()
                    with _rpc2:
                        if st.button("取消报告计划", key="cancel_report_plan_btn", use_container_width=True):
                            _tl_add(
                                str(_pending_report.get("task_id") or "unknown"),
                                "REPORT",
                                f"{_kind_label}计划已取消",
                                status="CANCELLED",
                                plan_id=_pending_report.get("plan_id"),
                                tool="report_generator" if _pending_report.get("kind") == "pdf" else "asset_report_engine",
                            )
                            st.session_state.pop("_pending_report_confirm", None)
                            st.rerun()
    except Exception:
        pass


# ---- Phase E+: 成果报告生成（集成自 E:\\Code\\pdf report_engine.py，栅格统计 + 参考真值对比）----
def _report_task_id() -> str:
    """从时间线解析当前报告任务，避免报告入口自行猜测任务。"""
    try:
        for _e in reversed(_get_task_timeline().events(limit=50)):
            if getattr(_e, "task_id", None):
                return str(_e.task_id)
    except Exception:
        pass
    return str(st.session_state.get("selected_task") or "")


def _queue_report_plan(kind: str) -> None:
    """仅登记报告计划；真正生成必须经过显式确认。"""
    _kind = "asset" if str(kind) == "asset" else "pdf"
    _task_id = _report_task_id() or "unknown"
    _label = "成果报告" if _kind == "pdf" else "监测报告"
    _plan_id = f"report_{uuid.uuid4().hex}"
    st.session_state["_pending_report_confirm"] = {
        "kind": _kind,
        "task_id": _task_id,
        "plan_id": _plan_id,
    }
    _tl_add(
        _task_id,
        "PLAN",
        f"{_label}计划已生成，等待确认",
        status="WAITING_CONFIRMATION",
        plan_id=_plan_id,
        tool="report_generator" if _kind == "pdf" else "asset_report_engine",
    )


def _record_report_outcome(
    task_id: str, tool: str, report_path: str = "", error: str = "", plan_id: str = ""
) -> None:
    """记录报告文件校验与登记，供 UI 和账本共同消费。"""
    _task = str(task_id or "unknown")
    _plan = plan_id or str(st.session_state.get("_active_report_plan_id") or "") or None
    if report_path and os.path.isfile(report_path) and os.path.getsize(report_path) > 0:
        _artifact = os.path.basename(str(report_path))
        _tl_add(_task, "VERIFY", "报告文件存在且非空，校验通过", status="SUCCEEDED", plan_id=_plan, tool=tool)
        _tl_add(_task, "REGISTER", "报告资产已登记", status="SUCCEEDED", plan_id=_plan, tool=tool, artifacts=[_artifact])
        _tl_add(_task, "REPORT", "报告已生成并可下载", status="SUCCEEDED", plan_id=_plan, tool=tool, artifacts=[_artifact])
    else:
        _tl_add(
            _task,
            "VERIFY",
            "报告文件校验失败",
            status="FAILED",
            plan_id=_plan,
            tool=tool,
            error=error or "报告文件缺失或为空",
        )
    st.session_state.pop("_active_report_plan_id", None)


def _build_asset_report():
    try:
        import asset_report_engine as _are

        _tl = _get_task_timeline()
        _events = _tl.events(limit=50)
        _task = ""
        if _events:
            for _e in reversed(_events):
                # 顶层 task_id 为权威字段；details.task 为兼容回退
                if getattr(_e, "task_id", None):
                    _task = str(_e.task_id)
                    break
                if isinstance(_e.details, dict) and _e.details.get("task"):
                    _task = str(_e.details["task"])
                    break
        if not _task:
            _task = str(st.session_state.get("selected_task") or "") or ""
        if not _task:
            st.session_state["_asset_report_msg"] = {
                "level": "warning",
                "text": "未识别到目标任务，请在左侧选择目标任务",
            }
            return
        _res = _are.generate_asset_report(
            _task, progress_callback=lambda p, m: None,
        )
        if _res.success and _res.report_path:
            _record_report_outcome(_task, "asset_report_engine", _res.report_path)
            st.session_state["_asset_report_msg"] = {
                "level": "success",
                "text": "✅ 成果报告已生成",
                "path": _res.report_path,
            }
        else:
            _safe_report_error = safe_error_summary(_res.error or "未知错误")
            _record_report_outcome(_task, "asset_report_engine", error=_safe_report_error)
            _msg = {
                "level": "warning",
                "text": f"成果报告生成失败：{_safe_report_error}",
            }
            _warns = [("· " + w) for w in (_res.warnings or [])]
            if _warns:
                _msg["warnings"] = _warns
            st.session_state["_asset_report_msg"] = _msg
    except Exception as _re:
        _record_report_outcome(_report_task_id(), "asset_report_engine", error=safe_error_summary(_re))
        st.session_state["_asset_report_msg"] = {
            "level": "warning",
            "text": f"成果报告生成异常：{safe_error_summary(_re)}",
        }


# ---- Phase E: PDF 报告生成（真实数据：时间线 + 能力 + 资产）----
def _build_pdf_report():
    try:
        import report_generator as _rg

        _tl = _get_task_timeline()
        _events = _tl.events(limit=50)
        if not _events:
            st.warning("无任务进度记录，无法生成报告")
            return
        _last = _events[-1]
        _task_id = _last.task_id or "task_unknown"
        _task_ctx = {"task_id": _task_id}
        _det = {}
        for _e in reversed(_events):
            if isinstance(_e.details, dict):
                _det.update(_e.details)
        _task_ctx.update(
            {
                "task": _det.get("task") or _task_id,
                "mode": _det.get("mode") or "",
                "prob": _det.get("prob"),
                "cnt": _det.get("cnt"),
                "plan_id": _det.get("plan_id"),
            }
        )
        _caps = {}
        try:
            _creg = st.session_state.get("_capability_reg")
            if _creg is not None:
                _caps = _creg.snapshot_for_agent()
        except Exception:
            pass
        _assets = []
        for _e in _events:
            for _a in (_e.artifacts or []):
                _assets.append({"path": str(_a), "kind": "artifact"})
        _res = _rg.generate_task_report(
            _task_ctx, capabilities=_caps, timeline=_events, assets=_assets,
        )
        if _res.success and _res.report_path:
            _record_report_outcome(_task_id, "report_generator", _res.report_path)
            st.success("✅ 成果报告已生成")
            st.markdown(f"`{_res.report_path}`")
            try:
                with open(_res.report_path, "rb") as _pf:
                    st.download_button(
                        "⬇️ 下载成果报告",
                        _pf.read(),
                        file_name=os.path.basename(_res.report_path),
                        mime="application/pdf",
                        key="_btn_dl_pdf_report",
                    )
            except Exception:
                pass
        else:
            _safe_report_error = safe_error_summary(_res.error or "未知错误")
            _record_report_outcome(_task_id, "report_generator", error=_safe_report_error)
            st.warning(f"成果报告生成失败：{_safe_report_error}")
        for _w in (_res.warnings or []):
            st.caption("· " + _w)
    except Exception as _re:
        _record_report_outcome(_report_task_id(), "report_generator", error=safe_error_summary(_re))
        st.warning(f"成果报告生成异常：{safe_error_summary(_re)}")


_PIPELINE_USE_FRAGMENT = False
try:
    _pipeline_monitor = st.fragment(run_every=2.5)(_pipeline_monitor_inner)
    _PIPELINE_USE_FRAGMENT = True
except (TypeError, AttributeError):
    _pipeline_monitor = _pipeline_monitor_inner


# ---- 地图下方状态抽屉：填充状态 / 日志面板 ----
maybe_start_pipeline_thread()
if _log_panel_slot is not None:
    with _log_panel_slot:
        _pipeline_monitor(render=True)
else:
    _pipeline_monitor(render=False)

components.html(
    """
    <script>
    (() => {
      const win = window.parent || window;
      const doc = win.document;
      const setImp = (el, prop, val) => el?.style?.setProperty(prop, val, "important");
      const safeObserve = (observer, target, options) => {
        // Streamlit component iframes can expose a cross-realm Document
        // proxy that fails the MutationObserver Node brand check.  Initial
        // binding plus timed layout sync is sufficient for reruns; avoid
        // installing a fragile cross-realm observer altogether.
        return false;
      };

      // 只把真正的三维地球/二维地图当作地图画布。定位命令通过
      // components.html 生成的无 src 辅助 iframe 必须保持 0 高度。
      const getPrimaryMapFrame = (mapCol) =>
        mapCol?.querySelector('iframe[src*="/globe"]') ||
        mapCol?.querySelector('iframe[title*="streamlit_folium"]') ||
        null;

      const getPrimaryMapBox = (mapCol) => {
        const frame = getPrimaryMapFrame(mapCol);
        return frame?.closest('[data-testid="stIFrame"]') ||
          frame?.closest('[data-testid="stCustomComponentV1"]') ||
          frame ||
          mapCol;
      };

      const relayoutDismissibleAlerts = () => {
        let index = 0;
        doc.querySelectorAll('[data-testid="stAlert"].cstf-dismissible-alert').forEach((alert) => {
          if (alert.dataset.cstfDismissed === "1") return;
          alert.style.setProperty("--cstf-alert-top", `${16 + index * 92}px`);
          index += 1;
        });
      };

      const bindDismissibleAlerts = () => {
        doc.querySelectorAll('[data-testid="stAlert"]').forEach((alert) => {
          if (alert.dataset.cstfDismissible === "1") return;
          alert.dataset.cstfDismissible = "1";
          const noticeKey = `cstf-dismissed:${encodeURIComponent(
            (alert.textContent || "").replace(/\\s+/g, " ").trim().slice(0, 240)
          )}`;
          try {
            if (win.sessionStorage.getItem(noticeKey) === "1") {
              alert.dataset.cstfDismissed = "1";
              alert.style.display = "none";
              return;
            }
          } catch (_) {}
          alert.classList.add("cstf-dismissible-alert");
          const close = doc.createElement("button");
          close.type = "button";
          close.className = "cstf-alert-close";
          close.setAttribute("aria-label", "关闭通知");
          close.title = "关闭通知";
          close.textContent = "×";
          close.addEventListener("click", () => {
            alert.dataset.cstfDismissed = "1";
            alert.style.display = "none";
            try { win.sessionStorage.setItem(noticeKey, "1"); } catch (_) {}
            relayoutDismissibleAlerts();
          });
          alert.appendChild(close);
        });
        relayoutDismissibleAlerts();
      };

      bindDismissibleAlerts();
      if (!win.__cstfAlertObserver && doc.documentElement && doc.documentElement.nodeType === 1) {
        win.__cstfAlertObserver = new win.MutationObserver(() => {
          win.setTimeout(bindDismissibleAlerts, 20);
        });
        safeObserve(win.__cstfAlertObserver, doc.documentElement, { childList: true, subtree: true });
      }

      const resetLayoutDefaults = () => {
        const defaults = doc.querySelector(".cstf-layout-defaults");
        if (!defaults) return;
        const reserve = defaults.getAttribute("data-status-reserve");
        if (reserve) doc.documentElement.style.setProperty("--cstf-status-panel-reserve", `${reserve}px`, "important");
      };

      // Streamlit 会在按钮交互后替换布局节点，但 iframe 中的脚本不一定重新执行。
      // 监听默认值节点，避免旧拖拽值继续覆盖折叠/展开后的新布局。
      let layoutDefaultsSignature = "";
      const observeLayoutDefaults = () => {
        const sync = () => {
          const defaults = doc.querySelector(".cstf-layout-defaults");
          if (!defaults) return;
          const signature = [
            defaults.getAttribute("data-status-reserve") || "",
            defaults.getAttribute("data-agent-width") || "",
          ].join("|");
          if (signature === layoutDefaultsSignature) return;
          layoutDefaultsSignature = signature;
          resetLayoutDefaults();
          syncWorkbenchHeight();
        };
        sync();
        if (!doc.documentElement || doc.documentElement.nodeType !== 1) return;
        if (win.__cstfLayoutDefaultsObserver) return;
        win.__cstfLayoutDefaultsObserver = new win.MutationObserver(sync);
        safeObserve(win.__cstfLayoutDefaultsObserver, doc.documentElement, {
          subtree: true,
          childList: true,
          attributes: true,
          attributeFilter: ["data-status-reserve", "data-agent-width"],
        });
      };

      const syncStatusHandlePosition = (handle, mapCol) => {
        if (!handle || !mapCol) return;
        const mapRect = getPrimaryMapBox(mapCol)?.getBoundingClientRect() || mapCol.getBoundingClientRect();
        if (!mapRect || mapRect.width < 4 || mapRect.height < 4) return;
        handle.style.left = `${mapRect.left}px`;
        handle.style.top = `${mapRect.bottom}px`;
        handle.style.width = `${mapRect.width}px`;
        const reserve = parseFloat(
          win.getComputedStyle(doc.documentElement).getPropertyValue("--cstf-status-panel-reserve")
        ) || 272;
        handle.setAttribute("aria-valuenow", String(Math.round(reserve)));
      };

      const syncStatusToggle = (toggle, mapCol) => {
        if (!toggle || !mapCol) return;
        const mapRect = getPrimaryMapBox(mapCol)?.getBoundingClientRect() || mapCol.getBoundingClientRect();
        if (!mapRect || mapRect.width < 4 || mapRect.height < 4) return;
        const state = doc.querySelector(".cstf-status-toggle-state");
        const collapsed = state?.getAttribute("data-collapsed") === "1";
        const nextText = collapsed ? "▲" : "▼";
        const nextLabel = collapsed ? "展开任务状态与系统日志" : "收起任务状态与系统日志";
        if (toggle.textContent !== nextText) toggle.textContent = nextText;
        if (toggle.getAttribute("aria-label") !== nextLabel) toggle.setAttribute("aria-label", nextLabel);
        if (toggle.title !== nextLabel) toggle.title = nextLabel;
        // 水平固定在分界线中点：展开时 ▼ 位于状态区内侧；收起时 ▲ 的底边贴合分界线。
        const toggleHeight = toggle.getBoundingClientRect().height;
        toggle.style.left = `${(mapRect.left + mapRect.right) / 2}px`;
        toggle.style.top = `${collapsed ? mapRect.bottom - toggleHeight : mapRect.bottom}px`;
      };

      const bindStatusToggle = () => {
        const mapCol = doc.querySelector('div[data-testid="stColumn"]:has(.cockpit-map-col)');
        if (!mapCol) return;
        let toggle = doc.querySelector(".cstf-status-edge-toggle");
        if (!toggle) {
          toggle = doc.createElement("button");
          toggle.type = "button";
          toggle.className = "cstf-status-edge-toggle";
          doc.body.appendChild(toggle);
        }
        // Streamlit may replace the component iframe or the bridge marker on
        // every rerun.  Rebind the persistent edge button whenever its
        // listener marker is missing, instead of binding only on first create.
        if (toggle.dataset.cstfStatusClickBound !== "1") {
          // Use a parent-document inline handler.  A listener callback
          // created inside this component iframe can be discarded when
          // Streamlit replaces the iframe during a rerun.
          toggle.setAttribute(
            "onclick",
            "const native = document.querySelector('div.st-key-agent_status_panel_toggle button'); if (native) native.click();"
          );
          toggle.__cstfStatusClickBound = true;
          toggle.dataset.cstfStatusClickBound = "1";
        }
        syncStatusToggle(toggle, mapCol);
      };

      // Streamlit replaces the hidden bridge node during a rerun. Observe its
      // data-collapsed state so the edge button immediately flips between
      // ▼ (collapse) and ▲ (expand), instead of retaining the previous label.
      const observeStatusToggleState = () => {
        if (!doc.documentElement || doc.documentElement.nodeType !== 1 || win.__cstfStatusToggleObserver) return;
        const sync = () => {
          syncStatusToggle(
            doc.querySelector(".cstf-status-edge-toggle"),
            doc.querySelector('div[data-testid="stColumn"]:has(.cockpit-map-col)')
          );
        };
        win.__cstfStatusToggleObserver = new win.MutationObserver(sync);
        safeObserve(win.__cstfStatusToggleObserver, doc.documentElement, {
          childList: true,
          subtree: true,
          attributes: true,
          attributeFilter: ["data-collapsed"],
        });
        sync();
      };

      const syncWorkbenchHeight = () => {
        const header = doc.querySelector('[data-testid="stHeader"]');
        const headerH = header ? header.offsetHeight : 56;
        const h = Math.max(480, win.innerHeight - headerH - 6);
        const px = h + "px";
        const reserve = parseFloat(
          win.getComputedStyle(doc.documentElement).getPropertyValue("--cstf-status-panel-reserve")
        ) || 0;
        const mapH = Math.max(280, h - reserve);
        const mapPx = mapH + "px";
        doc.documentElement.style.setProperty("--workbench-h", px);
        const mapCol = doc.querySelector('div[data-testid="stColumn"]:has(.cockpit-map-col)');
        const mapFrame = getPrimaryMapFrame(mapCol);
        const mapHost = getPrimaryMapBox(mapCol);
        const mapContainer = mapFrame?.closest('[data-testid="stElementContainer"]');
        new Set([mapFrame, mapHost, mapContainer]).forEach((el) => {
          if (!el) return;
          setImp(el, "height", mapPx);
          setImp(el, "max-height", mapPx);
        });
        syncStatusHandlePosition(
          doc.querySelector(".cstf-status-edge-handle"),
          mapCol
        );
        syncStatusToggle(
          doc.querySelector(".cstf-status-edge-toggle"),
          mapCol
        );
      };

      const lockPageWheel = () => {
        if (win.__cstfWheelLocked) return;
        win.__cstfWheelLocked = true;
        const canScroll = (el) => {
          if (!el || el === doc.documentElement) return false;
          const oy = win.getComputedStyle(el).overflowY;
          if (!["auto", "scroll", "overlay"].includes(oy)) return false;
          return el.scrollHeight > el.clientHeight + 2;
        };
        win.addEventListener(
          "wheel",
          (e) => {
            let el = e.target;
            while (el && el !== doc.body) {
              if (el.dataset?.testid === "stSidebar") return;
              if (canScroll(el)) {
                const top = el.scrollTop <= 0;
                const bottom = el.scrollTop + el.clientHeight >= el.scrollHeight - 1;
                if ((e.deltaY < 0 && top) || (e.deltaY > 0 && bottom)) e.preventDefault();
                return;
              }
              el = el.parentElement;
            }
            e.preventDefault();
          },
          { passive: false, capture: true }
        );
      };

      const syncDockHandlePosition = (handle, row, mapCol, sideCol) => {
        if (!handle || !row || !mapCol || !sideCol) return;
        const rr = row.getBoundingClientRect();
        const mr = mapCol.getBoundingClientRect();
        const sr = sideCol.getBoundingClientRect();
        handle.style.left = `${(mr.right + sr.left) / 2}px`;
        handle.style.top = `${rr.top}px`;
        handle.style.height = `${rr.height}px`;
        handle.setAttribute("aria-valuenow", String(Math.round(sr.width / Math.max(1, rr.width) * 100)));
      };

      // Inline handlers are compiled by the parent document.  They therefore
      // survive replacement of this temporary Streamlit component iframe and
      // can keep receiving pointer events after the cursor leaves the 14/16px
      // edge hit area.
      const parentResizePointerDown = String.raw`
        const handle = this;
        const win = window;
        const doc = document;
        if (event.button !== 0) return false;
        const kind = handle.dataset.cstfResizeKind;
        if (kind !== "dock" && kind !== "status") return false;
        if (typeof win.__cstfActiveResizeCleanup === "function") {
          win.__cstfActiveResizeCleanup();
        }

        const clamp = (value, min, max) => Math.min(max, Math.max(min, value));
        const setImp = (el, prop, value) => {
          if (el && el.style) el.style.setProperty(prop, value, "important");
        };
        const getPrimaryMapFrame = (mapCol) =>
          (mapCol && mapCol.querySelector('iframe[src*="/globe"]')) ||
          (mapCol && mapCol.querySelector('iframe[title*="streamlit_folium"]')) ||
          null;
        const getPrimaryMapBox = (mapCol) => {
          const frame = getPrimaryMapFrame(mapCol);
          return (frame && frame.closest('[data-testid="stIFrame"]')) ||
            (frame && frame.closest('[data-testid="stCustomComponentV1"]')) ||
            frame || mapCol;
        };
        const getNodes = () => {
          const mapCol = doc.querySelector('div[data-testid="stColumn"]:has(.cockpit-map-col)');
          const sideCol = doc.querySelector('div[data-testid="stColumn"]:has(.command-deck-side)');
          return {
            mapCol,
            sideCol,
            row: mapCol && mapCol.parentElement,
            mapFrame: getPrimaryMapFrame(mapCol),
            mapBox: getPrimaryMapBox(mapCol),
            dockHandle: doc.querySelector(".cstf-dock-resize-handle"),
            statusHandle: doc.querySelector(".cstf-status-edge-handle"),
            statusToggle: doc.querySelector(".cstf-status-edge-toggle"),
          };
        };
        const getReserve = () => parseFloat(
          win.getComputedStyle(doc.documentElement).getPropertyValue("--cstf-status-panel-reserve")
        ) || 272;
        const setLayoutParam = (name, value) => {
          try {
            const url = new URL(win.location.href);
            url.searchParams.set(name, String(Math.round(value)));
            win.history.replaceState({}, "", url.toString());
          } catch (_) {}
        };
        const syncResizeGeometry = () => {
          const nodes = getNodes();
          if (!nodes.mapCol) return;
          const header = doc.querySelector('[data-testid="stHeader"]');
          const headerH = header ? header.offsetHeight : 56;
          const workbenchH = Math.max(480, win.innerHeight - headerH - 6);
          const reserve = getReserve();
          const mapH = Math.max(280, workbenchH - reserve);
          doc.documentElement.style.setProperty("--workbench-h", workbenchH + "px");
          const mapContainer = nodes.mapFrame && nodes.mapFrame.closest('[data-testid="stElementContainer"]');
          [nodes.mapFrame, nodes.mapBox, mapContainer].forEach((el) => {
            setImp(el, "height", mapH + "px");
            setImp(el, "max-height", mapH + "px");
          });

          if (nodes.dockHandle && nodes.row && nodes.sideCol) {
            const rowRect = nodes.row.getBoundingClientRect();
            const mapColRect = nodes.mapCol.getBoundingClientRect();
            const sideRect = nodes.sideCol.getBoundingClientRect();
            nodes.dockHandle.style.left = ((mapColRect.right + sideRect.left) / 2) + "px";
            nodes.dockHandle.style.top = rowRect.top + "px";
            nodes.dockHandle.style.height = rowRect.height + "px";
            nodes.dockHandle.setAttribute(
              "aria-valuenow",
              String(Math.round(sideRect.width / Math.max(1, rowRect.width) * 100))
            );
          }

          const mapRect = nodes.mapBox && nodes.mapBox.getBoundingClientRect();
          if (!mapRect || mapRect.width < 4 || mapRect.height < 4) return;
          if (nodes.statusHandle) {
            const statusHandle = nodes.statusHandle;
            statusHandle.style.left = mapRect.left + "px";
            statusHandle.style.top = mapRect.bottom + "px";
            statusHandle.style.width = mapRect.width + "px";
            statusHandle.setAttribute("aria-valuenow", String(Math.round(reserve)));
          }
          if (nodes.statusToggle) {
            const state = doc.querySelector(".cstf-status-toggle-state");
            const collapsed = state && state.getAttribute("data-collapsed") === "1";
            const label = collapsed ? "展开任务状态与系统日志" : "收起任务状态与系统日志";
            nodes.statusToggle.textContent = collapsed ? "▲" : "▼";
            nodes.statusToggle.setAttribute("aria-label", label);
            nodes.statusToggle.title = label;
            const toggleHeight = nodes.statusToggle.getBoundingClientRect().height;
            nodes.statusToggle.style.left = ((mapRect.left + mapRect.right) / 2) + "px";
            nodes.statusToggle.style.top = (collapsed ? mapRect.bottom - toggleHeight : mapRect.bottom) + "px";
          }
        };
        const applyDockWidth = (sidePct) => {
          const nodes = getNodes();
          if (!nodes.row || !nodes.mapCol || !nodes.sideCol) return sidePct;
          const rowRect = nodes.row.getBoundingClientRect();
          const mapRect = nodes.mapCol.getBoundingClientRect();
          const sideRect = nodes.sideCol.getBoundingClientRect();
          const gap = Math.max(0, sideRect.left - mapRect.right);
          const available = Math.max(1, rowRect.width - gap);
          const pct = clamp(sidePct, 24, 48);
          const gapPx = Math.round(gap);
          const nextSidePct = pct;
          const mapPct = 100 - pct;
          [[nodes.mapCol, mapPct], [nodes.sideCol, nextSidePct]].forEach((pair) => {
            const basis = "calc(" + pair[1] + "% - " + gapPx + "px)";
            setImp(pair[0], "flex", "1 1 " + basis);
            setImp(pair[0], "width", "calc(" + pair[1] + "% - " + gapPx + "px)");
            setImp(pair[0], "max-width", "calc(" + pair[1] + "% - " + gapPx + "px)");
            setImp(pair[0], "min-width", "0");
          });
          syncResizeGeometry();
          return pct;
        };
        const applyStatusReserve = (value) => {
          const next = clamp(value, 192, 392);
          doc.documentElement.style.setProperty(
            "--cstf-status-panel-reserve",
            next + "px",
            "important"
          );
          syncResizeGeometry();
          return next;
        };

        const startNodes = getNodes();
        let drag = null;
        if (kind === "dock") {
          if (!startNodes.row || !startNodes.mapCol || !startNodes.sideCol) return false;
          const rowRect = startNodes.row.getBoundingClientRect();
          const mapRect = startNodes.mapCol.getBoundingClientRect();
          const sideRect = startNodes.sideCol.getBoundingClientRect();
          const available = Math.max(1, rowRect.width - Math.max(0, sideRect.left - mapRect.right));
          drag = {
            kind: "dock",
            startX: event.clientX,
            startSide: sideRect.width,
            startAvailable: available,
            currentPct: sideRect.width / available * 100,
          };
          doc.body.classList.add("cstf-resizing-agent");
        } else {
          const current = getReserve();
          drag = {
            kind: "status",
            startY: event.clientY,
            startReserve: current,
            currentReserve: current,
          };
          doc.body.classList.add("cstf-resizing-status");
        }

        // A full-page transparent capture surface keeps the pointer in the
        // parent document even while it crosses the embedded map iframe.
        const overlay = doc.createElement("div");
        overlay.className = "cstf-resize-capture";
        overlay.setAttribute("aria-hidden", "true");
        overlay.style.position = "fixed";
        overlay.style.inset = "0";
        overlay.style.zIndex = "2147483000";
        overlay.style.background = "transparent";
        overlay.style.cursor = kind === "dock" ? "col-resize" : "row-resize";
        overlay.style.touchAction = "none";
        overlay.style.userSelect = "none";
        doc.body.appendChild(overlay);

        let move = null;
        let stop = null;
        const cleanup = () => {
          if (move) win.removeEventListener("pointermove", move, true);
          if (stop) {
            win.removeEventListener("pointerup", stop, true);
            win.removeEventListener("pointercancel", stop, true);
            win.removeEventListener("blur", stop, true);
          }
          if (move) {
            overlay.removeEventListener("pointermove", move, true);
            overlay.removeEventListener("mousemove", move, true);
          }
          if (stop) {
            overlay.removeEventListener("pointerup", stop, true);
            overlay.removeEventListener("pointercancel", stop, true);
            overlay.removeEventListener("mouseup", stop, true);
          }
          overlay.remove();
          doc.body.classList.remove("cstf-resizing-agent", "cstf-resizing-status");
          if (win.__cstfActiveResizeCleanup === cleanup) {
            win.__cstfActiveResizeCleanup = null;
          }
        };
        move = (moveEvent) => {
          if (!drag) return;
          if (drag.kind === "dock") {
            const pct = clamp(
              (drag.startSide + drag.startX - moveEvent.clientX) /
                Math.max(1, drag.startAvailable) * 100,
              24,
              48
            );
            drag.currentPct = applyDockWidth(pct);
          } else {
            drag.currentReserve = applyStatusReserve(
              drag.startReserve + drag.startY - moveEvent.clientY
            );
          }
          moveEvent.preventDefault();
        };
        stop = () => {
          if (drag) {
            if (drag.kind === "dock") {
              setLayoutParam("cstf_agent_w", drag.currentPct);
            } else {
              setLayoutParam("cstf_status_h", drag.currentReserve - 8);
            }
          }
          drag = null;
          cleanup();
          syncResizeGeometry();
        };
        win.__cstfActiveResizeCleanup = cleanup;
        win.addEventListener("pointermove", move, true);
        win.addEventListener("pointerup", stop, true);
        win.addEventListener("pointercancel", stop, true);
        win.addEventListener("blur", stop, true);
        overlay.addEventListener("pointermove", move, true);
        overlay.addEventListener("mousemove", move, true);
        overlay.addEventListener("pointerup", stop, true);
        overlay.addEventListener("pointercancel", stop, true);
        overlay.addEventListener("mouseup", stop, true);
        try { handle.focus({ preventScroll: true }); } catch (_) {}
        event.preventDefault();
        event.stopPropagation();
        return false;
      `;

      const parentResizeKeyDown = String.raw`
        const kind = this.dataset.cstfResizeKind;
        const key = event.key;
        const dockKey = kind === "dock" && (key === "ArrowLeft" || key === "ArrowRight");
        const statusKey = kind === "status" && (key === "ArrowUp" || key === "ArrowDown");
        if (!dockKey && !statusKey) return;
        const rect = this.getBoundingClientRect();
        const x = (rect.left + rect.right) / 2;
        const y = (rect.top + rect.bottom) / 2;
        const dx = dockKey ? (key === "ArrowLeft" ? -20 : 20) : 0;
        const dy = statusKey ? (key === "ArrowUp" ? -20 : 20) : 0;
        const PointerCtor = window.PointerEvent || window.MouseEvent;
        const init = { bubbles: true, cancelable: true, button: 0, buttons: 1, pointerId: 91 };
        this.dispatchEvent(new PointerCtor("pointerdown", Object.assign({}, init, { clientX: x, clientY: y })));
        window.dispatchEvent(new PointerCtor("pointermove", Object.assign({}, init, { clientX: x + dx, clientY: y + dy })));
        window.dispatchEvent(new PointerCtor("pointerup", Object.assign({}, init, { clientX: x + dx, clientY: y + dy, buttons: 0 })));
        event.preventDefault();
        event.stopPropagation();
        return false;
      `;

      const bindAgentResize = () => {
        const mapCol = doc.querySelector('div[data-testid="stColumn"]:has(.cockpit-map-col)');
        const sideCol = doc.querySelector('div[data-testid="stColumn"]:has(.command-deck-side)');
        const row = mapCol?.parentElement;
        if (!mapCol || !sideCol || !row) return;

        let handle = doc.querySelector(".cstf-dock-resize-handle");
        if (handle && (
          handle.dataset.cstfResizeVersion !== "3" ||
          handle.__cstfMapCol !== mapCol ||
          handle.__cstfSideCol !== sideCol
        )) {
          handle.remove();
          handle = null;
        }
        if (!handle) {
          handle = doc.createElement("div");
          handle.className = "cstf-dock-resize-handle";
          handle.setAttribute("role", "separator");
          handle.setAttribute("aria-orientation", "vertical");
          handle.setAttribute("aria-label", "对话区宽度，可拖拽调整地图与 Agent 宽度");
          handle.setAttribute("aria-valuemin", "24");
          handle.setAttribute("aria-valuemax", "48");
          handle.tabIndex = 0;
          doc.body.appendChild(handle);
        }
        handle.dataset.cstfResizeVersion = "3";
        handle.dataset.cstfResizeKind = "dock";
        handle.setAttribute("onpointerdown", parentResizePointerDown);
        handle.setAttribute("onkeydown", parentResizeKeyDown);
        handle.__cstfMapCol = mapCol;
        handle.__cstfSideCol = sideCol;
        syncDockHandlePosition(handle, row, mapCol, sideCol);
      };

      const bindStatusResize = () => {
        const mapCol = doc.querySelector('div[data-testid="stColumn"]:has(.cockpit-map-col)');
        if (!mapCol) return;
        let handle = doc.querySelector(".cstf-status-edge-handle");
        if (handle && (
          handle.dataset.cstfResizeVersion !== "3" ||
          handle.__cstfMapCol !== mapCol
        )) {
          handle.remove();
          handle = null;
        }
        if (!handle) {
          handle = doc.createElement("div");
          handle.className = "cstf-status-edge-handle";
          handle.setAttribute("role", "separator");
          handle.setAttribute("aria-orientation", "horizontal");
          handle.setAttribute("aria-label", "拖拽调整地图与状态区高度");
          handle.setAttribute("title", "拖拽调整地图与状态区高度");
          handle.setAttribute("aria-valuemin", "192");
          handle.setAttribute("aria-valuemax", "392");
          handle.tabIndex = 0;
          doc.body.appendChild(handle);
        }
        handle.dataset.cstfResizeVersion = "3";
        handle.dataset.cstfResizeKind = "status";
        handle.setAttribute("onpointerdown", parentResizePointerDown);
        handle.setAttribute("onkeydown", parentResizeKeyDown);
        handle.__cstfMapCol = mapCol;
        syncStatusHandlePosition(handle, mapCol);
      };

      // Resize handles live in the parent Streamlit document while this code
      // runs inside a short-lived component iframe.  Install one controller
      // in the parent page so pointer move/up and MutationObserver callbacks
      // remain valid across component reruns.
      const installParentResizeController = () => {
        const controllerVersion = "2026-08-23-edge-resize-v2";
        if (win.__cstfEdgeResizeController?.version === controllerVersion) {
          win.__cstfEdgeResizeController.sync();
          return;
        }
        const controllerSource = String.raw`
          (() => {
            const win = window;
            const doc = win.document;
            const version = "2026-08-23-edge-resize-v2";
            const previous = win.__cstfEdgeResizeController;
            if (previous && typeof previous.destroy === "function") previous.destroy();

            const clamp = (value, min, max) => Math.min(max, Math.max(min, value));
            const setImp = (el, prop, value) => {
              if (el && el.style) el.style.setProperty(prop, value, "important");
            };
            const getPrimaryMapFrame = (mapCol) =>
              (mapCol && mapCol.querySelector('iframe[src*="/globe"]')) ||
              (mapCol && mapCol.querySelector('iframe[title*="streamlit_folium"]')) ||
              null;
            const getPrimaryMapBox = (mapCol) => {
              const frame = getPrimaryMapFrame(mapCol);
              return (frame && frame.closest('[data-testid="stIFrame"]')) ||
                (frame && frame.closest('[data-testid="stCustomComponentV1"]')) ||
                frame || mapCol;
            };
            const getNodes = () => {
              const mapCol = doc.querySelector('div[data-testid="stColumn"]:has(.cockpit-map-col)');
              const sideCol = doc.querySelector('div[data-testid="stColumn"]:has(.command-deck-side)');
              return {
                mapCol,
                sideCol,
                row: mapCol && mapCol.parentElement,
                mapFrame: getPrimaryMapFrame(mapCol),
                mapBox: getPrimaryMapBox(mapCol),
                dockHandle: doc.querySelector(".cstf-dock-resize-handle"),
                statusHandle: doc.querySelector(".cstf-status-edge-handle"),
                statusToggle: doc.querySelector(".cstf-status-edge-toggle"),
              };
            };
            const getReserve = () => parseFloat(
              win.getComputedStyle(doc.documentElement).getPropertyValue("--cstf-status-panel-reserve")
            ) || 272;
            const setLayoutParam = (name, value) => {
              try {
                const url = new URL(win.location.href);
                url.searchParams.set(name, String(Math.round(value)));
                win.history.replaceState({}, "", url.toString());
              } catch (_) {}
            };

            const sync = () => {
              const nodes = getNodes();
              if (!nodes.mapCol) return;
              const header = doc.querySelector('[data-testid="stHeader"]');
              const headerH = header ? header.offsetHeight : 56;
              const workbenchH = Math.max(480, win.innerHeight - headerH - 6);
              const reserve = getReserve();
              const mapH = Math.max(280, workbenchH - reserve);
              doc.documentElement.style.setProperty("--workbench-h", workbenchH + "px");
              const mapContainer = nodes.mapFrame && nodes.mapFrame.closest('[data-testid="stElementContainer"]');
              [nodes.mapFrame, nodes.mapBox, mapContainer].forEach((el) => {
                setImp(el, "height", mapH + "px");
                setImp(el, "max-height", mapH + "px");
              });

              if (nodes.dockHandle && nodes.row && nodes.sideCol) {
                const rowRect = nodes.row.getBoundingClientRect();
                const mapColRect = nodes.mapCol.getBoundingClientRect();
                const sideRect = nodes.sideCol.getBoundingClientRect();
                nodes.dockHandle.style.left = ((mapColRect.right + sideRect.left) / 2) + "px";
                nodes.dockHandle.style.top = rowRect.top + "px";
                nodes.dockHandle.style.height = rowRect.height + "px";
                nodes.dockHandle.setAttribute(
                  "aria-valuenow",
                  String(Math.round(sideRect.width / Math.max(1, rowRect.width) * 100))
                );
              }

              const mapRect = nodes.mapBox && nodes.mapBox.getBoundingClientRect();
              if (!mapRect || mapRect.width < 4 || mapRect.height < 4) return;
              if (nodes.statusHandle) {
                const statusHandle = nodes.statusHandle;
                statusHandle.style.left = mapRect.left + "px";
                statusHandle.style.top = mapRect.bottom + "px";
                statusHandle.style.width = mapRect.width + "px";
                statusHandle.setAttribute("aria-valuenow", String(Math.round(reserve)));
              }
              if (nodes.statusToggle) {
                const state = doc.querySelector(".cstf-status-toggle-state");
                const collapsed = state && state.getAttribute("data-collapsed") === "1";
                const label = collapsed ? "展开任务状态与系统日志" : "收起任务状态与系统日志";
                nodes.statusToggle.textContent = collapsed ? "▲" : "▼";
                nodes.statusToggle.setAttribute("aria-label", label);
                nodes.statusToggle.title = label;
                const toggleHeight = nodes.statusToggle.getBoundingClientRect().height;
                nodes.statusToggle.style.left = ((mapRect.left + mapRect.right) / 2) + "px";
                nodes.statusToggle.style.top = (collapsed ? mapRect.bottom - toggleHeight : mapRect.bottom) + "px";
              }
            };

            const applyDockWidth = (sidePct) => {
              const nodes = getNodes();
              if (!nodes.row || !nodes.mapCol || !nodes.sideCol) return sidePct;
              const rowRect = nodes.row.getBoundingClientRect();
              const mapRect = nodes.mapCol.getBoundingClientRect();
              const sideRect = nodes.sideCol.getBoundingClientRect();
              const gap = Math.max(0, sideRect.left - mapRect.right);
              const available = Math.max(1, rowRect.width - gap);
              const pct = clamp(sidePct, 24, 48);
              const gapPx = Math.round(gap);
              const mapPct = 100 - pct;
              [[nodes.mapCol, mapPct], [nodes.sideCol, pct]].forEach((pair) => {
                const el = pair[0];
                const basis = "calc(" + pair[1] + "% - " + gapPx + "px)";
                setImp(el, "flex", "1 1 " + basis);
                setImp(el, "width", "calc(" + pair[1] + "% - " + gapPx + "px)");
                setImp(el, "max-width", "calc(" + pair[1] + "% - " + gapPx + "px)");
                setImp(el, "min-width", "0");
              });
              sync();
              return pct;
            };
            const applyStatusReserve = (reserve) => {
              const next = clamp(reserve, 192, 392);
              doc.documentElement.style.setProperty(
                "--cstf-status-panel-reserve",
                next + "px",
                "important"
              );
              sync();
              return next;
            };

            let drag = null;
            const pointerDown = (event) => {
              if (event.button !== 0) return;
              const target = event.target && event.target.closest && event.target.closest(
                ".cstf-dock-resize-handle, .cstf-status-edge-handle"
              );
              if (!target) return;
              const nodes = getNodes();
              if (target.classList.contains("cstf-dock-resize-handle")) {
                if (!nodes.row || !nodes.sideCol || !nodes.mapCol) return;
                const rowRect = nodes.row.getBoundingClientRect();
                const mapRect = nodes.mapCol.getBoundingClientRect();
                const sideRect = nodes.sideCol.getBoundingClientRect();
                const gap = Math.max(0, sideRect.left - mapRect.right);
                const available = Math.max(1, rowRect.width - gap);
                drag = {
                  kind: "dock",
                  startX: event.clientX,
                  startSide: sideRect.width,
                  startAvailable: available,
                  currentPct: sideRect.width / available * 100,
                };
                doc.body.classList.add("cstf-resizing-agent");
              } else {
                const current = getReserve();
                drag = {
                  kind: "status",
                  startY: event.clientY,
                  startReserve: current,
                  currentReserve: current,
                };
                doc.body.classList.add("cstf-resizing-status");
              }
              try { target.focus({ preventScroll: true }); } catch (_) {}
              event.preventDefault();
              event.stopPropagation();
            };
            const move = (event) => {
              if (!drag) return;
              if (drag.kind === "dock") {
                const delta = drag.startX - event.clientX;
                const pct = clamp(
                  (drag.startSide + delta) / Math.max(1, drag.startAvailable) * 100,
                  24,
                  48
                );
                drag.currentPct = applyDockWidth(pct);
              } else {
                const next = drag.startReserve + drag.startY - event.clientY;
                drag.currentReserve = applyStatusReserve(next);
              }
              event.preventDefault();
            };
            const stop = () => {
              if (!drag) return;
              if (drag.kind === "dock") {
                setLayoutParam("cstf_agent_w", drag.currentPct);
              } else {
                setLayoutParam("cstf_status_h", drag.currentReserve - 8);
              }
              drag = null;
              doc.body.classList.remove("cstf-resizing-agent", "cstf-resizing-status");
              sync();
            };
            const keyDown = (event) => {
              const target = event.target && event.target.closest && event.target.closest(
                ".cstf-dock-resize-handle, .cstf-status-edge-handle"
              );
              if (!target) return;
              if (target.classList.contains("cstf-dock-resize-handle")) {
                if (event.key !== "ArrowLeft" && event.key !== "ArrowRight") return;
                const nodes = getNodes();
                if (!nodes.row || !nodes.sideCol || !nodes.mapCol) return;
                const rowRect = nodes.row.getBoundingClientRect();
                const mapRect = nodes.mapCol.getBoundingClientRect();
                const sideRect = nodes.sideCol.getBoundingClientRect();
                const available = Math.max(1, rowRect.width - Math.max(0, sideRect.left - mapRect.right));
                const current = sideRect.width / available * 100;
                const pct = applyDockWidth(current + (event.key === "ArrowLeft" ? 2 : -2));
                setLayoutParam("cstf_agent_w", pct);
              } else {
                if (event.key !== "ArrowUp" && event.key !== "ArrowDown") return;
                const next = applyStatusReserve(getReserve() + (event.key === "ArrowUp" ? 20 : -20));
                setLayoutParam("cstf_status_h", next - 8);
              }
              event.preventDefault();
            };

            let syncQueued = false;
            const queueSync = () => {
              if (syncQueued) return;
              syncQueued = true;
              win.requestAnimationFrame(() => {
                syncQueued = false;
                sync();
              });
            };
            doc.addEventListener("pointerdown", pointerDown, true);
            win.addEventListener("pointermove", move, true);
            win.addEventListener("pointerup", stop, true);
            win.addEventListener("pointercancel", stop, true);
            doc.addEventListener("keydown", keyDown, true);
            win.addEventListener("resize", queueSync);
            const observer = new win.MutationObserver(queueSync);
            if (doc.documentElement) observer.observe(doc.documentElement, { childList: true, subtree: true });

            const destroy = () => {
              observer.disconnect();
              doc.removeEventListener("pointerdown", pointerDown, true);
              win.removeEventListener("pointermove", move, true);
              win.removeEventListener("pointerup", stop, true);
              win.removeEventListener("pointercancel", stop, true);
              doc.removeEventListener("keydown", keyDown, true);
              win.removeEventListener("resize", queueSync);
              doc.body.classList.remove("cstf-resizing-agent", "cstf-resizing-status");
              drag = null;
            };
            win.__cstfEdgeResizeController = { version, sync, destroy };
            sync();
          })();
        `;
        try {
          const controllerScript = doc.createElement("script");
          controllerScript.type = "text/javascript";
          controllerScript.textContent = controllerSource;
          (doc.head || doc.documentElement).appendChild(controllerScript);
          controllerScript.remove();
          if (win.__cstfEdgeResizeController?.version !== controllerVersion) {
            throw new Error("parent controller did not initialize");
          }
        } catch (error) {
          win.console?.error?.("CSTF resize controller installation failed", error);
        }
      };

      resetLayoutDefaults();
      syncWorkbenchHeight();
      observeLayoutDefaults();
      lockPageWheel();
      bindAgentResize();
      bindStatusResize();
      bindStatusToggle();
      observeStatusToggleState();
      bindDismissibleAlerts();
      const syncAllResizeGeometry = () => {
        syncWorkbenchHeight();
        bindAgentResize();
        bindStatusResize();
        bindStatusToggle();
      };
      if (win.__cstfResizeWindowHandler) {
        win.removeEventListener("resize", win.__cstfResizeWindowHandler);
      }
      win.__cstfResizeWindowHandler = syncAllResizeGeometry;
      win.addEventListener("resize", win.__cstfResizeWindowHandler);
      [100, 400, 900].forEach((ms) => win.setTimeout(syncAllResizeGeometry, ms));
    })();
    </script>
    """,
    height=0,
)

# 无 st.fragment 时，只能靠整页定时刷新看到后台日志（会略卡顿）
if (
    not _PIPELINE_USE_FRAGMENT
    and st.session_state.is_running
    and st.session_state.get("pipeline_shared")
):
    _ps = st.session_state.pipeline_shared
    if not _ps.get("done"):
        time.sleep(3.0)
        st.rerun()

# 每轮 rerun 以快照方式持久化受控消息；命令原文和敏感路径由 ConversationStore 清理，
# 历史消息只用于展示/上下文，不会被重新入队执行。
if (
    st.session_state.get("_conversation_store") is not None
    and st.session_state.get("_conversation_thread_id")
    and st.session_state.get("messages") is not None
):
    try:
        st.session_state._conversation_store.replace_messages(
            st.session_state._conversation_thread_id,
            st.session_state.messages,
        )
    except Exception:
        pass
