import os
import sys
import glob
import json
import hashlib
import rasterio
from rasterio.features import shapes
from rasterio.windows import Window, from_bounds, transform as window_transform
from rasterio.warp import reproject, Resampling, transform_bounds
import numpy as np
from tqdm import tqdm
import geopandas as gpd
from shapely.geometry import shape
from agent_context_policy import safe_error_summary

# Windows 控制台默认 GBK 编码，无法编码 emoji（✅/❌/⚠️/🗺️ 等），
# 直接 logger 打印会抛 UnicodeEncodeError。强制 UTF-8 输出避免崩掉。
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# 累加缓存（_NUMERATOR/_DENOMINATOR）复用安全门闩：
# 仅当 manifest 存在且 fingerprint 完全一致才允许复用；否则视为不可信旧缓存，
# 删除同 stem 的 NUMERATOR/DENOMINATOR 并重新计算（不删除其他任务缓存）。
CACHE_MANIFEST_SCHEMA_VERSION = 1
CACHE_MANIFEST_FILENAME = "_cache_manifest.json"


# =======================================================
#  辅助函数
# =======================================================
def cv2_add_safe(a, b):
    # 转为 uint16 防止溢出
    return a.astype(np.uint16) + b.astype(np.uint16)


def output_stem(output_path: str) -> str:
    """成果路径 stem（不含扩展名），供 NUMERATOR/DENOMINATOR 等中间 TIF 命名。"""
    root, ext = os.path.splitext(output_path)
    if ext.lower() in (".tif", ".tiff", ".shp"):
        return root
    return output_path


def _cache_manifest_path(stem: str) -> str:
    """缓存 manifest 路径（与 NUMERATOR/DENOMINATOR 同目录，按 stem 隔离）。"""
    return f"{stem}{CACHE_MANIFEST_FILENAME}"


def _resolve_mask_source_pairs(mask_folder: str, source_folder: str):
    """解析 mask→源影像 配对（与阶段 1 一致：缺失源则跳过）。"""
    mask_files = glob.glob(os.path.join(mask_folder, "**", "*_mask.tif"), recursive=True)
    pairs = []
    for tif in mask_files:
        try:
            filename = os.path.basename(tif).replace("_mask.tif", ".tif")
            src_path = os.path.join(source_folder, filename)
            if not os.path.exists(src_path):
                found = glob.glob(os.path.join(source_folder, "**", filename), recursive=True)
                if found:
                    src_path = found[0]
                else:
                    continue
            pairs.append((tif, src_path))
        except Exception:
            continue
    return pairs


def _compute_cache_fingerprint(source_folder, mask_folder, prob_threshold,
                               min_absolute_count, output_path,
                               identity=None) -> str:
    """
    缓存 fingerprint：对实际参与累加的输入（mask + 源影像 path/size/mtime）与
    概率阈值 / 次数阈值 / 输出 stem / schema_version / 任务身份 求 sha256。
    输入或参数任何变化 → fingerprint 变化 → 旧缓存不可复用。
    """
    pairs = _resolve_mask_source_pairs(mask_folder, source_folder)
    inputs = []
    for mask_f, src_f in pairs:
        for p in (mask_f, src_f):
            try:
                st = os.stat(p)
                inputs.append([os.path.normpath(p), st.st_size, int(st.st_mtime)])
            except OSError:
                inputs.append([os.path.normpath(p), 0, 0])
    payload = {
        "schema_version": CACHE_MANIFEST_SCHEMA_VERSION,
        "inputs": sorted(inputs),
        "prob_threshold": float(prob_threshold),
        "count_threshold": int(min_absolute_count),
        "stem": output_stem(output_path),
        "identity": dict(identity or {}),
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _cache_fingerprint_matches(manifest_path, source_folder, mask_folder,
                               prob_threshold, min_absolute_count, output_path,
                               identity=None) -> bool:
    """manifest 存在 且 fingerprint 完全一致 → 可复用。"""
    if not os.path.isfile(manifest_path):
        return False
    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
    except (OSError, json.JSONDecodeError):
        return False
    if manifest.get("schema_version") != CACHE_MANIFEST_SCHEMA_VERSION:
        return False
    cur = _compute_cache_fingerprint(
        source_folder, mask_folder, prob_threshold, min_absolute_count,
        output_path, identity=identity,
    )
    return str(manifest.get("fingerprint") or "") == cur


def _write_cache_manifest(stem, source_folder, mask_folder, prob_threshold,
                          min_absolute_count, output_path, identity=None) -> None:
    """累加完成后写入 manifest（原子写）。"""
    manifest = {
        "schema_version": CACHE_MANIFEST_SCHEMA_VERSION,
        "fingerprint": _compute_cache_fingerprint(
            source_folder, mask_folder, prob_threshold, min_absolute_count,
            output_path, identity=identity,
        ),
        "inputs": _manifest_inputs(mask_folder, source_folder),
        "model_id": (identity or {}).get("model_id"),
        "weight_id": (identity or {}).get("weight_id"),
        "prob_threshold": float(prob_threshold),
        "count_threshold": int(min_absolute_count),
        "task_id": (identity or {}).get("task_id"),
        "plan_id": (identity or {}).get("plan_id"),
        "created_at": _now_str_post(),
    }
    manifest_path = _cache_manifest_path(stem)
    tmp = f"{manifest_path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    os.replace(tmp, manifest_path)


def _manifest_inputs(mask_folder, source_folder):
    out = []
    for mask_f, src_f in _resolve_mask_source_pairs(mask_folder, source_folder):
        for p in (mask_f, src_f):
            try:
                st = os.stat(p)
                out.append({"path": os.path.normpath(p), "size": st.st_size,
                            "mtime": int(st.st_mtime)})
            except OSError:
                out.append({"path": os.path.normpath(p), "size": 0, "mtime": 0})
    return out


def _now_str_post() -> str:
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _drop_stale_cache(stem, numerator_path, denominator_path, logger):
    """manifest 缺失或 fingerprint 不一致 → 删除同 stem 旧缓存（不动其他任务）。"""
    for _p in (numerator_path, denominator_path, _cache_manifest_path(stem)):
        if os.path.isfile(_p):
            try:
                os.remove(_p)
                logger(f"   🗑️ 已清理不可信旧缓存: {os.path.basename(_p)}")
            except OSError:
                pass


def raster_tidal_flat_to_shp(
    tif_path: str,
    shp_path: str,
    tidal_value: int = 255,
    logger=print,
) -> bool:
    """将二值/分类潮滩栅格转为 ESRI Shapefile（dissolve 后便于与参考 SHP 对比）。"""
    if not os.path.isfile(tif_path):
        logger(f"❌ 栅格转矢量失败，找不到: {tif_path}")
        return False

    logger(f"🗺️ 正在将合成栅格转为 Shapefile: {os.path.basename(shp_path)}")
    with rasterio.open(tif_path) as src:
        image = src.read(1)
        transform = src.transform
        crs = src.crs
        nodata = src.nodata

    mask = image > 128 if tidal_value > 1 else (image == tidal_value)
    if nodata is not None:
        mask = mask & (image != nodata)

    records = []
    for geom, value in shapes(image, mask=mask, transform=transform):
        records.append({"geometry": shape(geom), "class_val": int(value)})

    if not records:
        logger("⚠️ 合成结果中未检测到潮滩像元，跳过矢量化。")
        return False

    gdf = gpd.GeoDataFrame(records, crs=crs)
    gdf = gdf.dissolve(by="class_val").reset_index()

    out_dir = os.path.dirname(shp_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    gdf.to_file(shp_path, encoding="utf-8")
    logger(f"✅ 潮滩矢量成果已保存: {shp_path}")
    return True


# =======================================================
#  核心逻辑：生成 + 双重筛选 (适配 Streamlit)
# =======================================================
def generate_double_constraint_complete(source_folder, mask_folder, output_path, shp_path,
                                        prob_threshold=0.05, min_absolute_count=2, logger=print,
                                        stop_callback=None, keep_final_tif=False,
                                        cache_identity=None):
    """
    logger: 传入 st.write 或自定义函数，用于在网页上显示日志

    keep_final_tif: True 时把合成的 work 栅格保留为 Final TIF（output_path 若为 .tif），
                    默认 False 保持历史行为（仅生成 Final SHP，删除中间 work 栅格）。

    cache_identity: 可选 dict（如 {"task_id","plan_id","model_id","weight_id"}），
                    参与缓存 fingerprint 与 manifest 记录；任务身份变化 → 旧缓存不可复用。
    """
    logger(f"\n📊 [Post-Process] 启动双重约束合成")
    logger(f"   🎯 策略: 概率 > {prob_threshold:.1%}  且  绝对次数 >= {min_absolute_count}")

    stem = output_stem(output_path)
    if output_path.lower().endswith(".shp"):
        final_shp_path = output_path
    else:
        final_shp_path = f"{stem}.shp"
    work_tif_path = f"{stem}_work.tif"
    numerator_path = f"{stem}_NUMERATOR.tif"
    denominator_path = f"{stem}_DENOMINATOR.tif"
    manifest_path = _cache_manifest_path(stem)

    # --- 阶段 1: 扫描文件 ---
    mask_files = glob.glob(os.path.join(mask_folder, "**", "*_mask.tif"), recursive=True)
    if not mask_files:
        logger("❌ 未找到预测结果文件 (_mask.tif)")
        return False

    logger(f"📦 找到 {len(mask_files)} 个 Mask 文件，准备计算...")

    # --- 阶段 2: 生成累加缓存（manifest 门控复用，防跨任务/跨输入错误复用） ---
    cache_files_ok = os.path.exists(numerator_path) and os.path.exists(denominator_path)
    manifest_ok = _cache_fingerprint_matches(
        manifest_path, source_folder, mask_folder,
        prob_threshold, min_absolute_count, output_path, identity=cache_identity,
    )
    if cache_files_ok and manifest_ok:
        logger("⚡ 缓存 fingerprint 一致，复用 _NUMERATOR/_DENOMINATOR，跳过累加...")
    else:
        if cache_files_ok:
            logger("⚠️ 缓存存在但 manifest 缺失或 fingerprint 不一致，视为不可信旧缓存，重新计算...")
            _drop_stale_cache(stem, numerator_path, denominator_path, logger)
        else:
            logger("🐢 缓存未找到，正在从头生成累加数据 (这需要几分钟)...")

        # 2.1 计算范围（统一到首张 CRS）
        min_x, min_y, max_x, max_y = float('inf'), float('inf'), float('-inf'), float('-inf')
        ref_meta = None
        ref_crs = None
        valid_pairs = []

        for tif in tqdm(mask_files, desc="Step 1: 扫描范围"):
            try:
                filename = os.path.basename(tif).replace("_mask.tif", ".tif")
                src_path = os.path.join(source_folder, filename)
                if not os.path.exists(src_path):
                    found = glob.glob(os.path.join(source_folder, "**", filename), recursive=True)
                    if found:
                        src_path = found[0]
                    else:
                        continue

                with rasterio.open(tif) as src:
                    left, bottom, right, top = src.bounds
                    if ref_meta is None:
                        ref_meta = src.profile.copy()
                        ref_crs = src.crs
                    elif ref_crs and src.crs and src.crs != ref_crs:
                        left, bottom, right, top = transform_bounds(
                            src.crs, ref_crs, left, bottom, right, top, densify_pts=21
                        )
                    min_x, min_y = min(min_x, left), min(min_y, bottom)
                    max_x, max_y = max(max_x, right), max(max_y, top)

                valid_pairs.append((tif, src_path))
            except Exception:
                continue

        if not valid_pairs:
            logger("❌ 没有找到有效的成对影像！")
            return False

        # 2.2 创建大图
        res_x, res_y = ref_meta['transform'][0], -ref_meta['transform'][4]
        width, height = int((max_x - min_x) / res_x), int((max_y - min_y) / res_y)
        out_trans = rasterio.transform.from_origin(min_x, max_y, res_x, res_y)
        out_crs = ref_meta.get("crs")

        ref_meta.update({"height": height, "width": width, "transform": out_trans,
                         "count": 1, "dtype": "uint16", "compress": "lzw", "bigtiff": "YES"})

        # 2.3 执行累加（reproject 对齐到统一网格，避免 rowcol 错位）
        _interrupted = False
        try:
            with rasterio.open(numerator_path, "w+", **ref_meta) as dst_num, \
                    rasterio.open(denominator_path, "w+", **ref_meta) as dst_den:

                for mask_f, source_f in tqdm(valid_pairs, desc="Step 2: 累加计算"):
                    if stop_callback and stop_callback():
                        logger("🚨 合成阶段收到中断信号，终止累加")
                        _interrupted = True
                        break
                    try:
                        with rasterio.open(mask_f) as src_m:
                            mask_data = src_m.read(1)
                            tidal_count = (mask_data > 128).astype(np.uint16)
                            h, w = tidal_count.shape

                            if src_m.crs and out_crs and src_m.crs != out_crs:
                                bl, bb, br, bt = transform_bounds(
                                    src_m.crs, out_crs, *src_m.bounds, densify_pts=21)
                            else:
                                bl, bb, br, bt = src_m.bounds

                            try:
                                win = from_bounds(bl, bb, br, bt, out_trans)
                                win = win.round_offsets()
                                if hasattr(win, "round_lengths"):
                                    win = win.round_lengths()
                                win = win.intersection(Window(0, 0, width, height))
                                if win.width < 1 or win.height < 1:
                                    continue
                            except Exception:
                                continue

                            dst_h, dst_w = int(win.height), int(win.width)
                            dst_affine = window_transform(win, out_trans)
                            dst_crs_use = out_crs if out_crs else src_m.crs

                            dst_tidal = np.zeros((dst_h, dst_w), dtype=np.uint16)
                            reproject(
                                source=tidal_count,
                                destination=dst_tidal,
                                src_transform=src_m.transform,
                                src_crs=src_m.crs,
                                dst_transform=dst_affine,
                                dst_crs=dst_crs_use,
                                resampling=Resampling.nearest,
                            )

                            with rasterio.open(source_f) as src_s:
                                source_data = src_s.read(1, window=Window(0, 0, w, h))
                                valid_src = (source_data > 0).astype(np.uint16)
                            dst_valid = np.zeros((dst_h, dst_w), dtype=np.uint16)
                            reproject(
                                source=valid_src,
                                destination=dst_valid,
                                src_transform=src_m.transform,
                                src_crs=src_m.crs,
                                dst_transform=dst_affine,
                                dst_crs=dst_crs_use,
                                resampling=Resampling.nearest,
                            )

                            curr_num = dst_num.read(1, window=win)
                            dst_num.write(cv2_add_safe(curr_num, dst_tidal), 1, window=win)
                            curr_den = dst_den.read(1, window=win)
                            dst_den.write(cv2_add_safe(curr_den, dst_valid), 1, window=win)
                    except Exception:
                        continue
        except Exception as e:
            logger(f"❌ 累加文件创建失败: {safe_error_summary(e)}")
            _interrupted = True

        if _interrupted:
            _drop_stale_cache(stem, numerator_path, denominator_path, logger)
            logger("❌ 累加未完成，已清理缓存，任务终止。")
            return False

        # 累加成功 → 写缓存 manifest（记录本次输入的 fingerprint，供下次复用门闩）
        try:
            _write_cache_manifest(stem, source_folder, mask_folder,
                                  prob_threshold, min_absolute_count,
                                  output_path, identity=cache_identity)
        except OSError as e:
            logger(f"   ⚠️ 缓存 manifest 写入失败（不影响本次成果）: {safe_error_summary(e)}")

    # --- 阶段 3: 双重筛选与保存（分块处理，避免超大栅格 OOM） ---
    if stop_callback and stop_callback():
        return False
    logger(f"🚀 Step 3: 应用双重约束 (Prob>{prob_threshold} & Count>={min_absolute_count})...")

    TILE = 4096

    try:
        with rasterio.open(numerator_path) as src_num:
            out_meta = src_num.meta.copy()
            out_transform = src_num.transform
            out_crs = src_num.crs
            full_h, full_w = src_num.height, src_num.width

        # 预加载岸线裁剪掩膜（geometry_mask 必须一次性生成）
        clip_mask = None
        if shp_path and os.path.exists(shp_path) and os.path.normpath(shp_path) != os.path.normpath(final_shp_path):
            logger("   ✂️ 加载岸线裁剪掩膜...")
            try:
                gdf = gpd.read_file(shp_path)
                if out_crs != gdf.crs:
                    gdf = gdf.to_crs(out_crs)
                from rasterio.features import geometry_mask
                clip_mask = geometry_mask(
                    gdf.geometry,
                    out_shape=(full_h, full_w),
                    transform=out_transform,
                )
            except Exception as e:
                logger(f"   ⚠️ 岸线裁剪矢量读取失败，跳过裁剪: {safe_error_summary(e)}")
                clip_mask = None

        out_meta.update({"dtype": "uint8", "compress": "lzw"})
        n_tiles_y = (full_h + TILE - 1) // TILE
        n_tiles_x = (full_w + TILE - 1) // TILE
        logger(f"   📦 栅格 {full_h}×{full_w}，分 {n_tiles_y}×{n_tiles_x} 块处理...")

        with rasterio.open(numerator_path) as src_num, \
             rasterio.open(denominator_path) as src_den, \
             rasterio.open(work_tif_path, "w", **out_meta) as dst:

            for ty in range(n_tiles_y):
                if stop_callback and stop_callback():
                    logger("🚨 筛选阶段收到中断信号")
                    return False
                for tx in range(n_tiles_x):
                    row_off = ty * TILE
                    col_off = tx * TILE
                    th = min(TILE, full_h - row_off)
                    tw = min(TILE, full_w - col_off)
                    win = Window(col_off, row_off, tw, th)

                    E = src_num.read(1, window=win).astype(np.float32)
                    O = src_den.read(1, window=win).astype(np.float32)

                    prob = np.divide(E, O, out=np.zeros_like(E), where=O != 0)
                    result = ((prob >= prob_threshold) & (E >= min_absolute_count)).astype(np.uint8) * 255

                    if clip_mask is not None:
                        result[clip_mask[row_off:row_off + th, col_off:col_off + tw]] = 0

                    dst.write(result, 1, window=win)

        if not raster_tidal_flat_to_shp(work_tif_path, final_shp_path, tidal_value=255, logger=logger):
            return False

        # ✅ 保留 Final TIF：work 栅格内容即最终双重约束合成结果，
        #    把它原子重命名为 output_path（要求 output_path 以 .tif 结尾）。
        if keep_final_tif:
            _final_tif_candidates = [output_path]
            if output_path.lower().endswith(".tif") or output_path.lower().endswith(".tiff"):
                _final_tif = output_path
            else:
                _final_tif = output_stem(output_path) + ".tif"
            try:
                os.replace(work_tif_path, _final_tif)
                logger(f"✅ Final TIF 已保留: {_final_tif}")
            except OSError:
                logger(f"   ⚠️ 未能保留 Final TIF: {_final_tif}")
                try:
                    os.remove(work_tif_path)
                except OSError:
                    pass
        else:
            try:
                os.remove(work_tif_path)
            except OSError:
                logger(f"   ⚠️ 未能删除临时栅格: {work_tif_path}")

        logger(f"🎉 成功！双重约束潮滩矢量已生成: {final_shp_path}")
        return True

    except Exception as e:
        logger(f"❌ 出错: {safe_error_summary(e)}")
        import traceback
        traceback.print_exc()
        for _p in (work_tif_path,):
            if os.path.isfile(_p):
                try:
                    os.remove(_p)
                except OSError:
                    pass
        return False
