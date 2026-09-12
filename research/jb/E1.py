# -*- coding: utf-8 -*-
"""
E1: 多源潮滩数据统一与像元级一致性诊断

- 矢量统一 CRS: EPSG:4326（与师姐 china_tidal_flat_projected_*.shp 一致）
- 像元级对比网格: EPSG:4544（CGCS2000 Albers）30 m 真实面积
- 支持自定义 ROI（不必按省界）
"""
from __future__ import annotations

import glob
import json
import os
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize, shapes
from rasterio.merge import merge
from rasterio.transform import from_origin
from rasterio.transform import array_bounds
from rasterio.warp import Resampling, reproject
from rasterio.windows import Window, transform as window_transform
from shapely.geometry import box, shape

warnings.filterwarnings("ignore")

_EGEE_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
if _EGEE_ROOT not in sys.path:
    sys.path.insert(0, _EGEE_ROOT)
import cstf_ux as ux

# 与师姐 shp 一致
TARGET_VECTOR_CRS = "EPSG:4326"
# 像元级面积计算用等面积投影（30 m 真实边长）
RASTER_CRS = "EPSG:4544"
PIXEL_SIZE_M = 30
PIXEL_AREA_M2 = PIXEL_SIZE_M * PIXEL_SIZE_M
# 全国 30 m 网格约 60k x 90k 像元，全图载入会 OOM；超过则分块统计
MAX_FULL_RASTER_PIXELS = 20_000_000
DEFAULT_TILE_SIZE = 4096

# 师姐数据覆盖的中国海岸范围 (WGS84)
CHINA_BOUNDS_4326 = (107.996, 18.159, 124.229, 41.019)

DEFAULT_DATA_ROOT = r"E:\潮滩数据集"


class E1Cancelled(RuntimeError):
    """Raised when a cooperative stop is requested during a tiled E1 run."""


def _stop_requested(stop_callback: Optional[Callable[[], bool]]) -> bool:
    if not stop_callback:
        return False
    try:
        return bool(stop_callback())
    except Exception:
        # A diagnostic callback must never make E1 fail spuriously.  The
        # worker still has its own event and will perform a final gate.
        return False


def _vector_bounds_4326(path: Union[str, Path]) -> Optional[Tuple[float, float, float, float]]:
    """Read a vector extent and normalize it to WGS84.

    E1 historically used ``CHINA_BOUNDS_4326`` whenever no explicit ROI was
    supplied.  That turns a local prediction into a nationwide 30 m grid and
    can make an otherwise valid evaluation appear hung or fail from resource
    pressure.  The target result is a safe fallback extent because it bounds
    the comparison without treating the prediction geometry as the ROI mask.
    """
    try:
        gdf = gpd.read_file(path)
        crs = getattr(gdf, "crs", None)
        if crs is not None:
            crs_text = str(crs).upper()
            if "4326" not in crs_text and "CRS84" not in crs_text:
                gdf = gdf.to_crs(TARGET_VECTOR_CRS)
        bounds = getattr(gdf, "total_bounds", None)
        if bounds is None or len(bounds) != 4:
            return None
        values = tuple(float(v) for v in bounds)
        west, south, east, north = values
        if not all(np.isfinite(values)) or east <= west or north <= south:
            return None
        if west < -180 or east > 180 or south < -90 or north > 90:
            return None
        return values
    except Exception:
        return None


def _find_child(root: Path, *keywords: str) -> Optional[Path]:
    if not root.is_dir():
        return None
    for name in os.listdir(root):
        if all(k in name for k in keywords):
            return root / name
    return None


def _builtin_dataset_specs(data_root: Path) -> Dict[str, Dict[str, Any]]:
    """本地潮滩数据集路径注册（基于 E:\\潮滩数据集 实际目录）。"""
    dctf_dir = _find_child(data_root, "18", "N") or _find_child(data_root, "DCTF")
    gtf_dir = _find_child(data_root, "GTF30")
    fcs_dir = _find_child(data_root, "FCS30") or _find_child(data_root, "GWL")
    mtwm_dir = _find_child(data_root, "MTWM") or _find_child(data_root, "TidalWetland")
    tfmc_dir = data_root / "TFMC"
    sis_dir = data_root / "师姐数据集"
    nat10_dir = _find_child(data_root, "10m") or _find_child(data_root, "national")
    murray_dir = _find_child(data_root, "2014")

    specs: Dict[str, Dict[str, Any]] = {}

    for year in (2020, 2022, 2024, 2025):
        p = sis_dir / f"china_tidal_flat_projected_{year}.shp"
        if p.exists():
            specs[f"师姐_{year}"] = {"kind": "vector", "path": p}

    chn_dir = data_root / "CHN"
    if chn_dir.is_dir():
        for key, sub in [("CHN_2016", "CHN2016TidalFlats"), ("CHN_2024", "CHN2024TidalFlats")]:
            p = chn_dir / sub / f"{sub}.shp"
            if p.exists():
                specs[key] = {
                    "kind": "vector",
                    "path": p,
                    "filter": {"column": "class", "values": ["TF", "TidalFlats", "flat"]},
                }

    if dctf_dir:
        p = dctf_dir / "DCTF_China_2020.shp"
        if p.exists():
            specs["DCTF_2020"] = {
                "kind": "vector",
                "path": p,
                "filter": {"column": "class", "values": ["TidalFlats", "tidalflats", "TidalFlat"]},
            }

    if fcs_dir:
        p = fcs_dir / "FCS30_china_2020_flat_clip.shp"
        if p.exists():
            specs["FCS30_2020"] = {
                "kind": "vector",
                "path": p,
                "filter": {"column": "gridcode", "values": [187]},
            }

    gtf_china_shp = (gtf_dir / "china" / "GTF30_china.shp") if gtf_dir else None
    gtf_coast_shp = (gtf_dir / "china" / "海岸线.shp") if gtf_dir else None
    gtf_tif_dir = (gtf_dir / "GTF30_2020maps_E95_E120") if gtf_dir else None
    result_shp = data_root / "result" / "海岸线.shp"
    if gtf_coast_shp and gtf_coast_shp.exists():
        specs["GTF30_2020"] = {"kind": "vector", "path": gtf_coast_shp}
    elif result_shp.exists():
        specs["GTF30_2020"] = {"kind": "vector", "path": result_shp}
    elif gtf_tif_dir and gtf_tif_dir.is_dir():
        specs["GTF30_2020"] = {
            "kind": "tif_folder",
            "path": gtf_tif_dir,
            "flat_values": [1],
        }
    elif gtf_china_shp and gtf_china_shp.exists():
        specs["GTF30_2020"] = {
            "kind": "vector",
            "path": gtf_china_shp,
            "filter": {"column": "gridcode", "values": [1]},
            "note": "省级汇总版，像元精度有限",
        }

    if mtwm_dir:
        p = mtwm_dir / "MTWM_flat.shp"
        if p.exists():
            specs["MTWM_2020"] = {
                "kind": "vector",
                "path": p,
                "filter": {"column": "GRIDCODE", "values": [2]},
            }

    if tfmc_dir.is_dir():
        gpkg = tfmc_dir / "TFMC_china.gpkg"
        if gpkg.exists():
            specs["TFMC_2020"] = {"kind": "vector", "path": gpkg}
        else:
            specs["TFMC_2020"] = {"kind": "tfmc_merge", "path": tfmc_dir}

    if nat10_dir:
        for sub in nat10_dir.rglob("national tidal.shp"):
            specs["national_10m_2020"] = {"kind": "vector", "path": sub}
            break

    if murray_dir and murray_dir.is_dir():
        specs["Murray_2014_2016"] = {
            "kind": "tif_folder",
            "path": murray_dir,
            "flat_values": [1],
        }

    return specs


class E1_DataCleanerAndDiagnostic:
    def __init__(
        self,
        workspace_dir: str,
        data_root: str = DEFAULT_DATA_ROOT,
        pixel_size_m: int = PIXEL_SIZE_M,
    ):
        self.workspace = Path(ux.ensure_dir(workspace_dir))
        self.data_root = Path(ux.normalize_path(data_root) or data_root)
        self.pixel_size_m = pixel_size_m
        self.clean_dir = Path(ux.ensure_dir(self.workspace / "E1_cleaned_data"))
        self.unified_dir = Path(ux.ensure_dir(self.workspace / "E1_unified"))
        self.raster_dir = Path(ux.ensure_dir(self.workspace / "E1_rasters"))
        self.output_dir = Path(ux.ensure_dir(self.workspace / "outputs_e1"))
        # Projected geometries are reused by every tile/pair in one run.  The
        # previous implementation transformed and clipped the same national
        # vectors once per tile, which dominated E1 runtime.
        self._raster_projected_cache: Dict[str, gpd.GeoDataFrame] = {}
        self._raster_roi_cache: Dict[int, Optional[gpd.GeoDataFrame]] = {}

        self.dataset_specs = _builtin_dataset_specs(self.data_root)
        ux.banner("E1 多源一致性诊断", print)
        print(f"  工作区: {self.workspace}")
        print(f"  数据根目录: {self.data_root}")
        print(f"  已注册: {', '.join(sorted(self.dataset_specs)) or '(无)'}")

    # ------------------------------------------------------------------
    # 数据加载与矢量统一
    # ------------------------------------------------------------------
    def list_datasets(self) -> List[str]:
        return sorted(self.dataset_specs.keys())

    def _apply_filter(self, gdf: gpd.GeoDataFrame, filt: Optional[dict]) -> gpd.GeoDataFrame:
        if not filt or gdf.empty:
            return gdf
        col = filt.get("column")
        values = filt.get("values", [])
        if col not in gdf.columns:
            return gdf
        series = gdf[col]
        if pd.api.types.is_numeric_dtype(series):
            mask = series.isin(values)
        else:
            mask = series.astype(str).str.lower().isin({str(v).lower() for v in values})
        return gdf.loc[mask].copy()

    def _load_tfmc_merge(
        self,
        folder: Path,
        bbox_4326: Optional[Tuple[float, float, float, float]] = None,
    ) -> gpd.GeoDataFrame:
        parts = []
        bbox = None
        if bbox_4326 is not None:
            bbox = gpd.GeoSeries([box(*bbox_4326)], crs=TARGET_VECTOR_CRS)
        for fp in sorted(folder.glob("*.geojson")):
            try:
                parts.append(gpd.read_file(fp, bbox=bbox) if bbox is not None else gpd.read_file(fp))
            except Exception:
                # Older GDAL/pyogrio combinations may not accept a GeoSeries
                # bbox for GeoJSON (or may reject a CRS conversion); fall back
                # to the normal read path so correctness is preserved.
                parts.append(gpd.read_file(fp))
        if not parts:
            raise FileNotFoundError(f"TFMC 目录下无 geojson: {folder}")
        gdf = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), crs=parts[0].crs)
        return gdf

    def load_dataset(
        self,
        name: str,
        spec: Optional[dict] = None,
        bbox_4326: Optional[Tuple[float, float, float, float]] = None,
    ) -> gpd.GeoDataFrame:
        spec = spec or self.dataset_specs.get(name)
        if not spec:
            raise KeyError(f"未知数据集 [{name}]，可用: {self.list_datasets()}")

        kind = spec["kind"]
        path = Path(spec["path"])

        if kind == "vector":
            if not path.exists():
                raise FileNotFoundError(path)
            if bbox_4326 is not None:
                bbox = gpd.GeoSeries([box(*bbox_4326)], crs=TARGET_VECTOR_CRS)
                try:
                    gdf = gpd.read_file(path, bbox=bbox)
                except Exception:
                    # Keep compatibility with drivers that do not support a
                    # GeoSeries bbox; the later spatial index still clips the
                    # loaded result to the local comparison domain.
                    gdf = gpd.read_file(path)
            else:
                gdf = gpd.read_file(path)
            gdf = self._apply_filter(gdf, spec.get("filter"))
        elif kind == "tfmc_merge":
            gdf = self._load_tfmc_merge(path, bbox_4326=bbox_4326)
        elif kind == "tif_folder":
            gdf = self._tif_folder_to_vectors(path, spec.get("flat_values", [1]))
        else:
            raise ValueError(f"不支持的数据类型: {kind}")

        gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()
        if gdf.crs is None:
            gdf = gdf.set_crs(TARGET_VECTOR_CRS)
        gdf = gdf.to_crs(TARGET_VECTOR_CRS)
        gdf = self._clip_gdf_to_bounds(gdf, bbox_4326)
        gdf = ux.repair_geometries(gdf, logger=print)
        gdf["source"] = name
        return gdf

    def _tif_folder_to_vectors(
        self, folder: Path, flat_values: List[int]
    ) -> gpd.GeoDataFrame:
        tifs = sorted(glob.glob(str(folder / "*.tif")))
        if not tifs:
            raise FileNotFoundError(f"目录下无 tif: {folder}")

        srcs = [rasterio.open(p) for p in tifs]
        try:
            mosaic, transform = merge(srcs)
            crs = srcs[0].crs
        finally:
            for s in srcs:
                s.close()

        band = mosaic[0]
        mask = np.isin(band, flat_values)
        if not mask.any():
            raise ValueError(f"{folder} 中未找到 flat_values={flat_values} 的像元")

        records = []
        for geom, val in shapes(band.astype(np.int16), mask=mask, transform=transform):
            if int(val) in flat_values:
                records.append({"geometry": shape(geom)})
        return gpd.GeoDataFrame(records, crs=crs)

    def normalize_vector(
        self,
        input_path: Union[str, Path],
        asset_name: str,
        filter_spec: Optional[dict] = None,
        save: bool = True,
        bbox_4326: Optional[Tuple[float, float, float, float]] = None,
    ) -> gpd.GeoDataFrame:
        """读取任意 shp/geojson/gpkg，统一为 EPSG:4326 标准字段。"""
        path = Path(ux.normalize_path(input_path, must_exist=True) or input_path)
        if not path.exists():
            raise FileNotFoundError(path)

        if bbox_4326 is not None:
            bbox = gpd.GeoSeries([box(*bbox_4326)], crs=TARGET_VECTOR_CRS)
            try:
                gdf = gpd.read_file(path, bbox=bbox)
            except Exception:
                gdf = gpd.read_file(path)
        else:
            gdf = gpd.read_file(path)
        gdf = self._apply_filter(gdf, filter_spec)
        gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()
        if gdf.crs is None:
            gdf = gdf.set_crs(TARGET_VECTOR_CRS)
        gdf = gdf.to_crs(TARGET_VECTOR_CRS)
        gdf = self._clip_gdf_to_bounds(gdf, bbox_4326)
        gdf = ux.repair_geometries(gdf, logger=print)

        out = gpd.GeoDataFrame(
            {
                "source": asset_name,
                "geometry": gdf.geometry,
            },
            crs=TARGET_VECTOR_CRS,
        )

        if save:
            out_path = self.unified_dir / f"{asset_name}.gpkg"
            out.to_file(out_path, driver="GPKG", layer=asset_name)
            print(f"  已保存统一矢量: {out_path} ({len(out):,} 要素)")

        return out

    def normalize_all_registered(self) -> Dict[str, Path]:
        """批量导出已注册开源数据集的标准 gpkg。"""
        saved = {}
        for name in self.list_datasets():
            print(f"统一 [{name}] ...")
            gdf = self.load_dataset(name)
            out_path = self.unified_dir / f"{name}.gpkg"
            slim = gpd.GeoDataFrame({"source": name, "geometry": gdf.geometry}, crs=TARGET_VECTOR_CRS)
            slim.to_file(out_path, driver="GPKG", layer=name)
            saved[name] = out_path
            print(f"  -> {out_path} ({len(slim):,} 要素)")
        return saved

    # ------------------------------------------------------------------
    # 像元级栅格化与对比
    # ------------------------------------------------------------------
    def _load_roi(
        self, roi_path: Optional[str], bounds_4326: Tuple[float, float, float, float]
    ) -> Tuple[Optional[gpd.GeoDataFrame], Tuple[float, float, float, float]]:
        if not roi_path:
            return None, bounds_4326

        roi = gpd.read_file(roi_path)
        if roi.crs is None:
            roi = roi.set_crs(TARGET_VECTOR_CRS)
        roi = roi.to_crs(TARGET_VECTOR_CRS)
        roi = roi[roi.geometry.notna() & ~roi.geometry.is_empty]
        if roi.empty:
            raise ValueError(f"ROI 为空: {roi_path}")

        minx, miny, maxx, maxy = roi.total_bounds
        clipped = (
            max(bounds_4326[0], minx),
            max(bounds_4326[1], miny),
            min(bounds_4326[2], maxx),
            min(bounds_4326[3], maxy),
        )
        return roi, clipped

    @staticmethod
    def _clip_gdf_to_bounds(
        gdf: gpd.GeoDataFrame,
        bounds_4326: Optional[Tuple[float, float, float, float]],
    ) -> gpd.GeoDataFrame:
        """Clip geometry envelopes before topology repair.

        Coastal products often contain one invalid polygon spanning a large
        part of China.  Repairing that full geometry is much more expensive
        than repairing the small piece that can actually enter this E1 grid.
        ``shapely.clip_by_rect`` is deliberately used as a fast preclip; the
        normal validity repair still runs immediately afterwards.
        """
        if not bounds_4326 or gdf is None or gdf.empty:
            return gdf
        try:
            from shapely import clip_by_rect

            clipped = gdf.copy()
            xmin, ymin, xmax, ymax = (float(v) for v in bounds_4326)
            clipped["geometry"] = clip_by_rect(
                clipped.geometry.array, xmin, ymin, xmax, ymax
            )
            return clipped[clipped.geometry.notna() & ~clipped.geometry.is_empty].copy()
        except Exception:
            # A conservative envelope filter is still useful with older
            # Shapely/GDAL combinations that do not expose clip_by_rect.
            try:
                frame = gdf[gdf.geometry.intersects(box(*bounds_4326))].copy()
                return frame
            except Exception:
                return gdf

    def build_reference_grid(
        self,
        roi_path: Optional[str] = None,
        bounds_4326: Tuple[float, float, float, float] = CHINA_BOUNDS_4326,
    ) -> Tuple[Any, int, int, str]:
        """在 EPSG:4544 下建立 30 m 参考网格。"""
        _, bounds = self._load_roi(roi_path, bounds_4326)

        bbox_gdf = gpd.GeoDataFrame(geometry=[box(*bounds)], crs=TARGET_VECTOR_CRS)
        bbox_p = bbox_gdf.to_crs(RASTER_CRS)
        minx, miny, maxx, maxy = bbox_p.total_bounds

        width = max(1, int(np.ceil((maxx - minx) / self.pixel_size_m)))
        height = max(1, int(np.ceil((maxy - miny) / self.pixel_size_m)))
        transform = from_origin(minx, maxy, self.pixel_size_m, self.pixel_size_m)
        return transform, width, height, RASTER_CRS

    def _projected_roi(self, roi_gdf: Optional[gpd.GeoDataFrame]) -> Optional[gpd.GeoDataFrame]:
        if roi_gdf is None:
            return None
        key = id(roi_gdf)
        cache = getattr(self, "_raster_roi_cache", None)
        if cache is None:
            cache = {}
            self._raster_roi_cache = cache
        if key not in cache:
            cache[key] = roi_gdf.to_crs(RASTER_CRS)
        return cache[key]

    def _projected_gdf(
        self,
        cache_key: str,
        gdf: gpd.GeoDataFrame,
        roi_gdf: Optional[gpd.GeoDataFrame] = None,
    ) -> Tuple[gpd.GeoDataFrame, Optional[gpd.GeoDataFrame]]:
        """Prepare one vector once in the equal-area raster CRS."""
        cache = getattr(self, "_raster_projected_cache", None)
        if cache is None:
            cache = {}
            self._raster_projected_cache = cache
        if cache_key not in cache:
            gdf_p = gdf if str(getattr(gdf, "crs", "")) == RASTER_CRS else gdf.to_crs(RASTER_CRS)
            roi_p = self._projected_roi(roi_gdf)
            if roi_p is not None and not gdf_p.empty:
                try:
                    # Clip once per dataset, never once per tile.
                    gdf_p = gpd.clip(gdf_p, roi_p)
                except Exception:
                    # The tile-level spatial index and ROI mask below remain
                    # a correct fallback when a GDAL spatial index is absent.
                    pass
            cache[cache_key] = gdf_p
        return cache[cache_key], self._projected_roi(roi_gdf)

    def _projected_dataset(
        self,
        name: str,
        gdf_cache: Dict[str, gpd.GeoDataFrame],
        roi_gdf: Optional[gpd.GeoDataFrame],
    ) -> Tuple[gpd.GeoDataFrame, Optional[gpd.GeoDataFrame]]:
        gdf = gdf_cache.get(name)
        if gdf is None:
            # An explicit AOI is just as useful as the target extent for
            # driver-level filtering.  Most vector drivers can reject
            # features outside this bbox before GeoPandas constructs the
            # full national frame; the projected clip below still applies
            # the exact ROI geometry.
            bbox_4326 = None
            if roi_gdf is not None and not roi_gdf.empty:
                try:
                    roi_wgs84 = (
                        roi_gdf
                        if str(getattr(roi_gdf, "crs", "")) == TARGET_VECTOR_CRS
                        else roi_gdf.to_crs(TARGET_VECTOR_CRS)
                    )
                    bounds = tuple(float(v) for v in roi_wgs84.total_bounds)
                    if len(bounds) == 4 and bounds[2] > bounds[0] and bounds[3] > bounds[1]:
                        bbox_4326 = bounds
                except Exception:
                    bbox_4326 = None
            gdf = self.load_dataset(name, bbox_4326=bbox_4326)
            gdf_cache[name] = gdf
        return self._projected_gdf(name, gdf, roi_gdf)

    def _rasterize_projected(
        self,
        gdf_p: gpd.GeoDataFrame,
        transform,
        out_shape: Tuple[int, int],
        roi_p: Optional[gpd.GeoDataFrame] = None,
        tile_bounds: Optional[Tuple[float, float, float, float]] = None,
    ) -> np.ndarray:
        """Rasterize already projected geometries with tile-local filtering."""
        tile_gdf = gdf_p
        if tile_bounds is not None and not tile_gdf.empty:
            minx, miny, maxx, maxy = tile_bounds
            try:
                idx = list(tile_gdf.sindex.intersection(tile_bounds))
                tile_gdf = tile_gdf.iloc[idx]
            except Exception:
                tile_box = box(minx, miny, maxx, maxy)
                tile_gdf = tile_gdf[tile_gdf.intersects(tile_box)]

        geoms = [
            (geom, 1)
            for geom in tile_gdf.geometry
            if geom is not None and not geom.is_empty
        ]
        arr = rasterize(
            geoms,
            out_shape=out_shape,
            transform=transform,
            fill=0,
            dtype=np.uint8,
            all_touched=False,
        ) if geoms else np.zeros(out_shape, dtype=np.uint8)

        if roi_p is not None and not roi_p.empty:
            roi_tile = roi_p
            if tile_bounds is not None:
                minx, miny, maxx, maxy = tile_bounds
                try:
                    idx = list(roi_p.sindex.intersection(tile_bounds))
                    roi_tile = roi_p.iloc[idx]
                except Exception:
                    tile_box = box(minx, miny, maxx, maxy)
                    roi_tile = roi_p[roi_p.intersects(tile_box)]
            roi_geoms = [
                (geom, 1)
                for geom in roi_tile.geometry
                if geom is not None and not geom.is_empty
            ]
            if roi_geoms:
                roi_mask = rasterize(
                    roi_geoms,
                    out_shape=out_shape,
                    transform=transform,
                    fill=0,
                    dtype=np.uint8,
                    all_touched=True,
                )
                arr = np.where(roi_mask == 1, arr, 0).astype(np.uint8)
            else:
                arr.fill(0)
        return arr

    def vector_to_raster(
        self,
        gdf: gpd.GeoDataFrame,
        transform,
        out_shape: Tuple[int, int],
        roi_gdf: Optional[gpd.GeoDataFrame] = None,
        tile_bounds: Optional[Tuple[float, float, float, float]] = None,
    ) -> np.ndarray:
        """矢量 -> 30 m 二值栅格 (1=潮滩, 0=非潮滩)。"""
        gdf_p, roi_p = self._projected_gdf(
            f"__vector_to_raster__{id(gdf)}", gdf, roi_gdf
        )
        return self._rasterize_projected(
            gdf_p,
            transform,
            out_shape,
            roi_p=roi_p,
            tile_bounds=tile_bounds,
        )

    def tif_to_raster(
        self,
        tif_path: Union[str, Path],
        transform,
        out_shape: Tuple[int, int],
        flat_values: Optional[List[int]] = None,
    ) -> np.ndarray:
        flat_values = flat_values or [1]
        dst = np.zeros(out_shape, dtype=np.float32)
        with rasterio.open(tif_path) as src:
            reproject(
                source=rasterio.band(src, 1),
                destination=dst,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=transform,
                dst_crs=RASTER_CRS,
                resampling=Resampling.nearest,
                src_nodata=src.nodata,
                dst_nodata=0,
            )
        return np.isin(dst.astype(np.int32), flat_values).astype(np.uint8)

    def _dataset_to_raster(
        self,
        name: str,
        transform,
        out_shape: Tuple[int, int],
        roi_gdf: Optional[gpd.GeoDataFrame],
        gdf_cache: Dict[str, gpd.GeoDataFrame],
    ) -> np.ndarray:
        spec = self.dataset_specs[name]
        kind = spec["kind"]

        if kind == "tif_folder":
            tifs = sorted(glob.glob(str(Path(spec["path"]) / "*.tif")))
            acc = np.zeros(out_shape, dtype=np.uint8)
            flat_values = spec.get("flat_values", [1])
            for tif in tifs:
                layer = self.tif_to_raster(tif, transform, out_shape, flat_values)
                acc = np.where(layer == 1, 1, acc).astype(np.uint8)
            if roi_gdf is not None:
                roi_p = roi_gdf.to_crs(RASTER_CRS)
                roi_mask = rasterize(
                    [(g, 1) for g in roi_p.geometry if g is not None and not g.is_empty],
                    out_shape=out_shape,
                    transform=transform,
                    fill=0,
                    dtype=np.uint8,
                    all_touched=True,
                )
                acc = np.where(roi_mask == 1, acc, 0).astype(np.uint8)
            return acc

        gdf_p, roi_p = self._projected_dataset(name, gdf_cache, roi_gdf)
        tile_bounds = array_bounds(out_shape[0], out_shape[1], transform)
        return self._rasterize_projected(
            gdf_p,
            transform,
            out_shape,
            roi_p=roi_p,
            tile_bounds=tile_bounds,
        )

    @staticmethod
    def _stats_to_metrics(stats: Dict[str, int], pair_name: str = "") -> Dict[str, float]:
        inter = stats["inter"]
        union = stats["union"]
        iou = ux.safe_div(inter, union, default=0.0)
        if inter == 0 and union == 0:
            ux.warn(ux.zero_overlap_message_e1(pair_name), print)
        to_km2 = lambda n: n * PIXEL_AREA_M2 / 1e6
        return {
            "jaccard_iou": round(iou, 4),
            "intersection_km2": round(to_km2(inter), 3),
            "union_km2": round(to_km2(union), 3),
            "only_a_km2": round(to_km2(stats["only_a"]), 3),
            "only_b_km2": round(to_km2(stats["only_b"]), 3),
            "area_a_km2": round(to_km2(stats["cnt_a"]), 3),
            "area_b_km2": round(to_km2(stats["cnt_b"]), 3),
            "intersection_pixels": inter,
            "union_pixels": union,
            "zero_pixel_overlap": inter == 0,
        }

    @staticmethod
    def _gdf_spatial_overlap(gdf_a: gpd.GeoDataFrame, gdf_b: gpd.GeoDataFrame) -> bool:
        """Cheap overlap gate using bounds + spatial index.

        Dissolving both sources into ``unary_union`` before every E1 run can
        take several seconds for coastal multipart polygons.  The raster
        comparison itself is authoritative, so the gate only needs to reject
        clearly disjoint inputs; an envelope/index false positive simply
        yields IoU=0 in the normal comparison path.
        """
        if gdf_a.empty or gdf_b.empty:
            return False
        try:
            b = gdf_b.to_crs(gdf_a.crs) if gdf_b.crs != gdf_a.crs else gdf_b
            a_minx, a_miny, a_maxx, a_maxy = gdf_a.total_bounds
            b_minx, b_miny, b_maxx, b_maxy = b.total_bounds
            if (
                a_maxx < b_minx
                or b_maxx < a_minx
                or a_maxy < b_miny
                or b_maxy < a_miny
            ):
                return False
            try:
                return bool(len(gdf_a.sindex.query(b.geometry, predicate="intersects")))
            except Exception:
                return True
        except Exception:
            # Do not block a valid evaluation on an optional spatial-index
            # optimization; the pixel pass will determine the real overlap.
            return True

    @staticmethod
    def _accumulate_tile_stats(
        stats: Dict[str, int], raster_a: np.ndarray, raster_b: np.ndarray
    ) -> None:
        a = raster_a == 1
        b = raster_b == 1
        stats["inter"] += int(np.count_nonzero(a & b))
        stats["only_a"] += int(np.count_nonzero(a & ~b))
        stats["only_b"] += int(np.count_nonzero(~a & b))
        stats["union"] += int(np.count_nonzero(a | b))
        stats["cnt_a"] += int(np.count_nonzero(a))
        stats["cnt_b"] += int(np.count_nonzero(b))

    def compare_rasters(
        self, raster_a: np.ndarray, raster_b: np.ndarray
    ) -> Dict[str, float]:
        stats = {"inter": 0, "only_a": 0, "only_b": 0, "union": 0, "cnt_a": 0, "cnt_b": 0}
        self._accumulate_tile_stats(stats, raster_a, raster_b)
        return self._stats_to_metrics(stats)

    def _iter_tiles(self, width: int, height: int, tile_size: int):
        n_cols = int(np.ceil(width / tile_size))
        n_rows = int(np.ceil(height / tile_size))
        total = n_cols * n_rows
        done = 0
        for row in range(0, height, tile_size):
            for col in range(0, width, tile_size):
                h = min(tile_size, height - row)
                w = min(tile_size, width - col)
                done += 1
                yield row, col, h, w, done, total

    def _compare_pair_tiled(
        self,
        name_a: str,
        name_b: str,
        transform,
        width: int,
        height: int,
        roi_gdf: Optional[gpd.GeoDataFrame],
        gdf_cache: Dict[str, gpd.GeoDataFrame],
        tile_size: int = DEFAULT_TILE_SIZE,
        writers: Optional[Dict[str, Any]] = None,
        stop_callback: Optional[Callable[[], bool]] = None,
    ) -> Dict[str, float]:
        stats = {"inter": 0, "only_a": 0, "only_b": 0, "union": 0, "cnt_a": 0, "cnt_b": 0}
        total = 0
        for row, col, h, w, done, total in self._iter_tiles(width, height, tile_size):
            if _stop_requested(stop_callback):
                raise E1Cancelled("E1 精度评价被用户中断")
            win = Window(col, row, w, h)
            sub_transform = window_transform(win, transform)
            shape = (h, w)
            ra = self._dataset_to_raster(name_a, sub_transform, shape, roi_gdf, gdf_cache)
            rb = self._dataset_to_raster(name_b, sub_transform, shape, roi_gdf, gdf_cache)
            self._accumulate_tile_stats(stats, ra, rb)
            if writers:
                cons, oa, ob, cls, heat = self._pair_disagreement_layers(ra, rb)
                self._write_tile(writers["consensus"], cons, col, row)
                self._write_tile(writers["only_a"], oa, col, row)
                self._write_tile(writers["only_b"], ob, col, row)
                self._write_tile(writers["heatmap"], heat, col, row)
                self._write_tile(writers["class"], cls, col, row)
            if done == 1 or done == total or done % max(1, total // 10) == 0:
                print(f"    分块进度 {done}/{total}", flush=True)
        if _stop_requested(stop_callback):
            raise E1Cancelled("E1 精度评价被用户中断")
        metrics = self._stats_to_metrics(stats, pair_name=f"{name_a}_vs_{name_b}")
        metrics["tiles_processed"] = total
        return metrics

    def _compare_gdf_to_dataset_tiled(
        self,
        gdf_a: gpd.GeoDataFrame,
        name_b: str,
        transform,
        width: int,
        height: int,
        roi_gdf: Optional[gpd.GeoDataFrame],
        gdf_cache: Dict[str, gpd.GeoDataFrame],
        tile_size: int = DEFAULT_TILE_SIZE,
        writers: Optional[Dict[str, Any]] = None,
        stop_callback: Optional[Callable[[], bool]] = None,
    ) -> Dict[str, float]:
        stats = {"inter": 0, "only_a": 0, "only_b": 0, "union": 0, "cnt_a": 0, "cnt_b": 0}
        target_p, roi_p = self._projected_gdf(
            f"__target__{id(gdf_a)}", gdf_a, roi_gdf
        )
        total = 0
        for row, col, h, w, done, total in self._iter_tiles(width, height, tile_size):
            if _stop_requested(stop_callback):
                raise E1Cancelled("E1 精度评价被用户中断")
            win = Window(col, row, w, h)
            sub_transform = window_transform(win, transform)
            shape = (h, w)
            tile_bounds = array_bounds(h, w, sub_transform)
            ra = self._rasterize_projected(
                target_p,
                sub_transform,
                shape,
                roi_p=roi_p,
                tile_bounds=tile_bounds,
            )
            rb = self._dataset_to_raster(name_b, sub_transform, shape, roi_gdf, gdf_cache)
            self._accumulate_tile_stats(stats, ra, rb)
            if writers:
                cons, oa, ob, cls, heat = self._pair_disagreement_layers(ra, rb)
                self._write_tile(writers["consensus"], cons, col, row)
                self._write_tile(writers["only_a"], oa, col, row)
                self._write_tile(writers["only_b"], ob, col, row)
                self._write_tile(writers["heatmap"], heat, col, row)
                self._write_tile(writers["class"], cls, col, row)
            if done == 1 or done == total or done % max(1, total // 10) == 0:
                print(f"    分块进度 {done}/{total}", flush=True)
        if _stop_requested(stop_callback):
            raise E1Cancelled("E1 精度评价被用户中断")
        return self._stats_to_metrics(stats, pair_name=f"product_vs_{name_b}")

    def _save_geotiff(
        self,
        arr: np.ndarray,
        transform,
        crs: str,
        path: Path,
        dtype=rasterio.uint8,
        nodata: Optional[int] = None,
        cog: bool = False,
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        profile = {
            "driver": "GTiff",
            "height": arr.shape[0],
            "width": arr.shape[1],
            "count": 1,
            "dtype": dtype,
            "crs": crs,
            "transform": transform,
            "compress": "lzw",
        }
        if cog:
            profile.update(
                tiled=True,
                blockxsize=512,
                blockysize=512,
                interleave="band",
            )
        if nodata is not None:
            profile["nodata"] = nodata
        with rasterio.open(path, "w", **profile) as dst:
            dst.write(arr, 1)

    def _open_raster_writer(
        self,
        path: Path,
        transform,
        crs: str,
        width: int,
        height: int,
        dtype=rasterio.uint8,
        nodata: Optional[int] = None,
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        profile = {
            "driver": "GTiff",
            "height": height,
            "width": width,
            "count": 1,
            "dtype": dtype,
            "crs": crs,
            "transform": transform,
            "compress": "lzw",
            "tiled": True,
            "blockxsize": 512,
            "blockysize": 512,
            "interleave": "band",
        }
        if nodata is not None:
            profile["nodata"] = nodata
        return rasterio.open(path, "w", **profile)

    @staticmethod
    def _write_tile(dst, arr: np.ndarray, col: int, row: int):
        h, w = arr.shape
        dst.write(arr, 1, window=Window(col, row, w, h))

    @staticmethod
    def _pair_disagreement_layers(
        raster_a: np.ndarray, raster_b: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        a = raster_a == 1
        b = raster_b == 1
        consensus = np.where(a & b, 1, 0).astype(np.uint8)
        only_a = np.where(a & ~b, 1, 0).astype(np.uint8)
        only_b = np.where(~a & b, 1, 0).astype(np.uint8)
        # 0=背景 1=一致 2=仅A 3=仅B
        cls = np.zeros_like(consensus, dtype=np.uint8)
        cls[a & b] = 1
        cls[a & ~b] = 2
        cls[~a & b] = 3
        # 0=一致非潮滩 1=一致潮滩 2=分歧
        heat = np.zeros_like(consensus, dtype=np.uint8)
        heat[a & b] = 1
        heat[a ^ b] = 2
        return consensus, only_a, only_b, cls, heat

    def _open_pair_disagreement_writers(
        self,
        out_dir: Path,
        transform,
        crs: str,
        width: int,
        height: int,
    ) -> Dict[str, Any]:
        out_dir.mkdir(parents=True, exist_ok=True)
        return {
            "consensus": self._open_raster_writer(
                out_dir / "consensus.tif", transform, crs, width, height
            ),
            "only_a": self._open_raster_writer(
                out_dir / "only_layer_a.tif", transform, crs, width, height
            ),
            "only_b": self._open_raster_writer(
                out_dir / "only_layer_b.tif", transform, crs, width, height
            ),
            "class": self._open_raster_writer(
                out_dir / "disagreement_class.tif", transform, crs, width, height
            ),
            "heatmap": self._open_raster_writer(
                out_dir / "disagreement_heatmap.tif", transform, crs, width, height
            ),
            "paths": {
                "consensus": out_dir / "consensus.tif",
                "only_a": out_dir / "only_layer_a.tif",
                "only_b": out_dir / "only_layer_b.tif",
                "class": out_dir / "disagreement_class.tif",
                "heatmap": out_dir / "disagreement_heatmap.tif",
            },
        }

    @staticmethod
    def _close_writers(writers: Dict[str, Any]):
        for key in ("consensus", "only_a", "only_b", "class", "heatmap"):
            if key in writers and writers[key] is not None:
                writers[key].close()

    @staticmethod
    def _infer_dataset_traits(name: str, spec: Optional[dict]) -> Dict[str, Any]:
        traits: Dict[str, Any] = {"name": name}
        lower = name.lower()
        if "10m" in lower or "national" in lower or "tfmc" in lower:
            traits["resolution_m"] = 10
        elif "murray" in lower:
            traits["resolution_m"] = 30
        else:
            traits["resolution_m"] = 30

        year = None
        for token in name.replace("-", "_").split("_"):
            if token.isdigit() and len(token) == 4:
                year = int(token)
                break
        traits["year"] = year
        traits["kind"] = (spec or {}).get("kind", "vector")
        traits["note"] = (spec or {}).get("note", "")
        if "师姐" in name:
            traits["category"] = "reference_manual"
        elif "chn" in lower:
            traits["category"] = "national_30m"
        elif "dctf" in lower:
            traits["category"] = "long_series_30m"
        elif "gtf30" in lower or "fcs30" in lower:
            traits["category"] = "global_derived_30m"
        elif "mtwm" in lower:
            traits["category"] = "wetland_multiclass"
        elif "tfmc" in lower:
            traits["category"] = "high_res_10m"
        elif "national" in lower:
            traits["category"] = "survey_10m"
        elif "murray" in lower:
            traits["category"] = "global_raster"
        else:
            traits["category"] = "other"
        return traits

    def _build_causal_analysis(
        self,
        pair_name: str,
        name_a: str,
        name_b: str,
        metrics: Dict[str, float],
        spec_a: Optional[dict] = None,
        spec_b: Optional[dict] = None,
    ) -> Dict[str, Any]:
        ta = self._infer_dataset_traits(name_a, spec_a)
        tb = self._infer_dataset_traits(name_b, spec_b)
        iou = metrics.get("jaccard_iou", 0.0)
        only_a = metrics.get("only_a_km2", 0.0)
        only_b = metrics.get("only_b_km2", 0.0)
        area_a = metrics.get("area_a_km2", 0.0)
        area_b = metrics.get("area_b_km2", 0.0)

        factors: List[Dict[str, str]] = []
        summary_parts: List[str] = []

        if ta.get("year") and tb.get("year") and ta["year"] != tb["year"]:
            factors.append(
                {
                    "factor": "temporal_mismatch",
                    "detail": f"{name_a}({ta['year']}) vs {name_b}({tb['year']}) 年份不一致，潮滩边界随海平面/岸线变化可能偏移",
                    "severity": "high" if abs(ta["year"] - tb["year"]) >= 4 else "medium",
                }
            )
            summary_parts.append("年份差异")

        ra = ta.get("resolution_m", 30)
        rb = tb.get("resolution_m", 30)
        if ra != rb:
            factors.append(
                {
                    "factor": "resolution_mismatch",
                    "detail": f"空间分辨率不同 ({ra}m vs {rb}m)，细边界与像元聚合方式差异会导致 only_a/only_b 条带",
                    "severity": "high",
                }
            )
            summary_parts.append("分辨率差异")

        if area_a > 0 and only_a / max(area_a, 1e-6) > 0.35:
            factors.append(
                {
                    "factor": "over_detection_a",
                    "detail": f"{name_a} 独有面积约 {only_a:.1f} km²（占自身 {only_a/area_a*100:.1f}%），可能过度扩张潮滩或含盐沼/浅水混淆",
                    "severity": "medium",
                }
            )
        if area_b > 0 and only_b / max(area_b, 1e-6) > 0.35:
            factors.append(
                {
                    "factor": "over_detection_b",
                    "detail": f"{name_b} 独有面积约 {only_b:.1f} km²（占自身 {only_b/area_b*100:.1f}%），参考层边界可能偏保守或分类口径不同",
                    "severity": "medium",
                }
            )

        if ta.get("category") == "wetland_multiclass" or tb.get("category") == "wetland_multiclass":
            factors.append(
                {
                    "factor": "class_definition",
                    "detail": "湿地多分类产品（如 MTWM）需子类筛选，潮滩与盐沼/滩涂边界定义可能与 reference 不一致",
                    "severity": "medium",
                }
            )
            summary_parts.append("分类口径")

        if ta.get("kind") == "tif_folder" or tb.get("kind") == "tif_folder":
            factors.append(
                {
                    "factor": "raster_vectorization",
                    "detail": "栅格源经矢量化/重采样至 30m 网格，边界锯齿与重采样邻域会造成系统性分歧",
                    "severity": "low",
                }
            )

        if iou < 0.35:
            factors.append(
                {
                    "factor": "low_agreement",
                    "detail": f"IoU={iou:.3f} 极低，两产品空间重合度差，需优先核查 ROI、年份与潮滩定义",
                    "severity": "high",
                }
            )
        elif iou < 0.55:
            factors.append(
                {
                    "factor": "moderate_agreement",
                    "detail": f"IoU={iou:.3f} 中等，分歧集中在边界带与河口浅滩，属多源潮滩产品常见现象",
                    "severity": "medium",
                }
            )

        if ta.get("note"):
            factors.append(
                {"factor": "dataset_note", "detail": ta["note"], "severity": "info"}
            )
        if tb.get("note"):
            factors.append(
                {"factor": "dataset_note", "detail": tb["note"], "severity": "info"}
            )

        if not summary_parts:
            if iou >= 0.55:
                summary = "两产品总体一致，残余分歧主要来自 30m 边界像元与矢量碎斑"
            else:
                summary = "分歧可能由数据源算法、时相与潮滩定义差异共同造成"
        else:
            summary = "主要成因: " + "、".join(summary_parts)

        return {
            "pair": pair_name,
            "layer_a": name_a,
            "layer_b": name_b,
            "summary": summary,
            "factors": factors,
            "metrics_snapshot": {
                "jaccard_iou": iou,
                "only_a_km2": only_a,
                "only_b_km2": only_b,
            },
        }

    def _export_multi_product_heatmap_tiled(
        self,
        product_names: List[str],
        reference: str,
        transform,
        width: int,
        height: int,
        roi_gdf: Optional[gpd.GeoDataFrame],
        gdf_cache: Dict[str, gpd.GeoDataFrame],
        roi_name: str,
        tile_size: int = DEFAULT_TILE_SIZE,
        stop_callback: Optional[Callable[[], bool]] = None,
    ) -> Dict[str, Any]:
        """多产品一致计数热力图：像元值 = 判定为潮滩的产品数量 (0..N)。"""
        all_names = [reference] + [n for n in product_names if n != reference]
        n_products = len(all_names)
        count_path = self.output_dir / roi_name / "multi_product" / "agreement_count.tif"
        disagree_path = self.output_dir / roi_name / "multi_product" / "any_disagreement.tif"
        count_dst = self._open_raster_writer(
            count_path, transform, RASTER_CRS, width, height, dtype=rasterio.uint8
        )
        disagree_dst = self._open_raster_writer(
            disagree_path, transform, RASTER_CRS, width, height, dtype=rasterio.uint8
        )

        hist = {i: 0 for i in range(n_products + 1)}
        disagree_pixels = 0
        total_valid = 0

        for row, col, h, w, done, total in self._iter_tiles(width, height, tile_size):
            if _stop_requested(stop_callback):
                count_dst.close()
                disagree_dst.close()
                raise E1Cancelled("E1 精度评价被用户中断")
            win = Window(col, row, w, h)
            sub_transform = window_transform(win, transform)
            shape = (h, w)
            stack = []
            for name in all_names:
                layer = self._dataset_to_raster(
                    name, sub_transform, shape, roi_gdf, gdf_cache
                )
                stack.append(layer == 1)
            arr = np.stack(stack, axis=0)
            count = arr.sum(axis=0).astype(np.uint8)
            # 有产品认为是潮滩但不全一致
            any_flat = count > 0
            all_same = (count == 0) | (count == n_products)
            disagree = np.where(any_flat & ~all_same, 1, 0).astype(np.uint8)

            self._write_tile(count_dst, count, col, row)
            self._write_tile(disagree_dst, disagree, col, row)

            for v in range(n_products + 1):
                hist[v] += int(np.count_nonzero(count == v))
            disagree_pixels += int(np.count_nonzero(disagree))
            total_valid += h * w

            if done == 1 or done == total or done % max(1, total // 5) == 0:
                print(f"  多产品热力图 {done}/{total}", flush=True)

        if _stop_requested(stop_callback):
            count_dst.close()
            disagree_dst.close()
            raise E1Cancelled("E1 精度评价被用户中断")

        count_dst.close()
        disagree_dst.close()

        return {
            "products": all_names,
            "n_products": n_products,
            "agreement_count_tif": str(count_path),
            "any_disagreement_tif": str(disagree_path),
            "histogram": hist,
            "disagreement_pixel_ratio": round(
                disagree_pixels / total_valid if total_valid else 0.0, 4
            ),
        }

    def _write_markdown_report(self, results: Dict[str, Any], md_path: Path):
        lines = [
            f"# E1 像元级潮滩对比报告",
            "",
            f"- 时间: {results.get('timestamp')}",
            f"- ROI: {results.get('roi_name')}",
            f"- 参考层: {results.get('reference')}",
            f"- 网格: {results.get('raster_crs')} @ {results.get('pixel_size_m')}m",
            f"- 分块模式: {results.get('tiled_mode')}",
            f"- 评价耗时: {results.get('elapsed_seconds', '-')} 秒",
            "",
            "## 两两对比",
            "",
            "| 对比组 | IoU | 交集(km²) | A面积 | B面积 | 摘要 |",
            "|--------|-----|-----------|-------|-------|------|",
        ]
        for pair, data in results.get("comparisons", {}).items():
            if "error" in data:
                lines.append(f"| {pair} | ERROR | - | - | - | {data['error']} |")
                continue
            causal = data.get("causal_analysis", {})
            summary = causal.get("summary", "-")
            lines.append(
                f"| {pair} | {data.get('jaccard_iou', '-')} | "
                f"{data.get('intersection_km2', '-')} | "
                f"{data.get('area_a_km2', '-')} | "
                f"{data.get('area_b_km2', '-')} | {summary} |"
            )

        lines.extend(["", "## 成因分析详情", ""])
        for pair, data in results.get("comparisons", {}).items():
            causal = data.get("causal_analysis")
            if not causal:
                continue
            lines.append(f"### {pair}")
            lines.append(f"- {causal.get('summary', '')}")
            for f in causal.get("factors", []):
                lines.append(f"  - [{f.get('severity', '?')}] {f.get('detail', '')}")
            lines.append("")

        mp = results.get("multi_product_heatmap")
        if mp:
            lines.extend(
                [
                    "## 多产品一致热力图",
                    "",
                    f"- 参与产品 ({mp.get('n_products')}): {', '.join(mp.get('products', []))}",
                    f"- 一致计数栅格: `{mp.get('agreement_count_tif')}`",
                    f"- 分歧掩膜: `{mp.get('any_disagreement_tif')}`",
                    f"- 任两产品不一致像元占比: {mp.get('disagreement_pixel_ratio', 0):.2%}",
                    "",
                    "### 一致计数直方图 (像元数)",
                ]
            )
            for k, v in sorted((mp.get("histogram") or {}).items(), key=lambda x: int(x[0])):
                lines.append(f"- {k}/{mp.get('n_products')} 产品一致: {v:,}")

        md_path.parent.mkdir(parents=True, exist_ok=True)
        md_path.write_text("\n".join(lines), encoding="utf-8")

    def run_pixel_comparison(
        self,
        reference: str = "师姐_2020",
        target_path: Optional[str] = None,
        target_name: str = "My_Product",
        compare_sources: Optional[List[str]] = None,
        roi_path: Optional[str] = None,
        roi_name: str = "study_area",
        export_rasters: Optional[bool] = None,
        export_disagreement_maps: Optional[bool] = None,
        export_multi_product_heatmap: bool = True,
        tile_size: int = DEFAULT_TILE_SIZE,
        stop_callback: Optional[Callable[[], bool]] = None,
    ) -> Dict[str, Any]:
        """
        像元级多源对比（默认以师姐产品为 reference）。

        :param target_path: 你的潮滩 shp/geojson/gpkg；可为 None（仅对比开源产品 vs 师姐）
        :param compare_sources: 要对比的数据集名；None = 全部已注册数据集（除 reference）
        :param roi_path: 自定义研究区 shp（不必按省界）；None = 中国海岸默认范围
        :param export_rasters: 是否导出各产品二值栅格；全国大范围时自动 False
        :param export_disagreement_maps: 是否导出分歧图（分块 COG 写入，全国可用）
        :param export_multi_product_heatmap: 是否导出多产品一致计数热力图
        """
        if _stop_requested(stop_callback):
            raise E1Cancelled("E1 精度评价被用户中断")
        started_at = time.perf_counter()
        if reference not in self.dataset_specs:
            raise KeyError(f"reference [{reference}] 不在已注册列表: {self.list_datasets()}")

        if compare_sources is None:
            compare_sources = [n for n in self.list_datasets() if n != reference]

        # A missing explicit ROI used to silently expand every local target to
        # the China-wide 30 m grid.  Keep an explicit ROI authoritative; when
        # it is absent, bound the grid by the target vector extent instead.
        grid_bounds = CHINA_BOUNDS_4326
        target_extent_used = False
        if not roi_path and target_path:
            target_bounds = _vector_bounds_4326(target_path)
            if target_bounds is not None:
                grid_bounds = target_bounds
                target_extent_used = True
                print(
                    "  未指定 ROI，已按目标成果范围建立评价网格: "
                    f"({grid_bounds[0]:.6f}, {grid_bounds[1]:.6f}, "
                    f"{grid_bounds[2]:.6f}, {grid_bounds[3]:.6f})"
                )

        roi_gdf, resolved_bounds = self._load_roi(roi_path, grid_bounds)
        if target_extent_used and roi_gdf is None:
            # Use the target *extent* as a comparison mask.  It is deliberately
            # a bbox rather than the prediction geometry, so false positives
            # and false negatives remain measurable inside the local domain.
            roi_gdf = gpd.GeoDataFrame(
                {"geometry": [box(*grid_bounds)]}, crs=TARGET_VECTOR_CRS
            )
        transform, width, height, crs = self.build_reference_grid(
            roi_path, bounds_4326=grid_bounds
        )
        total_pixels = width * height
        use_tiled = total_pixels > MAX_FULL_RASTER_PIXELS
        if export_rasters is None:
            export_rasters = not use_tiled
        if export_disagreement_maps is None:
            export_disagreement_maps = True

        print(f"\n--- 像元级对比 | ROI: {roi_name} | 网格: {RASTER_CRS} @ {self.pixel_size_m}m ---")
        print(f"  网格大小: {width} x {height} = {total_pixels:,} 像元")
        if use_tiled:
            print(f"  模式: 分块统计 (tile={tile_size})，避免全国网格内存溢出")
        if not export_rasters:
            print("  不导出完整产品 GeoTIFF（范围过大或 export_rasters=False）")
        if export_disagreement_maps:
            print("  分歧图: 分块 COG 导出 (consensus / only_a / only_b / heatmap)")
        print(f"  参考层: {reference}")
        print(f"  对比层: {', '.join(compare_sources)}")
        if target_path:
            print(f"  你的产品: {target_path}")

        gdf_cache: Dict[str, gpd.GeoDataFrame] = {}
        # A diagnostic object can be reused for another ROI in tests or an
        # interactive session; never carry projected geometries across runs.
        self._raster_projected_cache = {}
        self._raster_roi_cache = {}
        local_scope = bool(target_extent_used or roi_path)
        if local_scope:
            # Warm the vector cache with spatially filtered data.  Without
            # this, every local E1 comparison would first load all national
            # products and only clip them after the expensive read.  An
            # explicit ROI follows the same fast path as an inferred target
            # extent; the exact polygon remains the raster mask.
            for _name in dict.fromkeys([reference, *compare_sources]):
                _spec = self.dataset_specs.get(_name) or {}
                if _spec.get("kind") not in {"vector", "tfmc_merge"}:
                    continue
                try:
                    gdf_cache[_name] = self.load_dataset(
                        _name, bbox_4326=resolved_bounds
                    )
                except Exception as _cache_exc:
                    print(f"  {_name} 局部预读取失败，将在对比时重试: {_cache_exc}")
        results: Dict[str, Any] = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "roi_name": roi_name,
            "reference": reference,
            "vector_crs": TARGET_VECTOR_CRS,
            "raster_crs": RASTER_CRS,
            "pixel_size_m": self.pixel_size_m,
            "grid_size": {"width": width, "height": height},
            "bounds_4326": [float(v) for v in resolved_bounds],
            "target_extent_used": target_extent_used,
            "tiled_mode": use_tiled,
            "export_rasters": export_rasters,
            "export_disagreement_maps": export_disagreement_maps,
            "comparisons": {},
        }

        if export_rasters and not use_tiled:
            if _stop_requested(stop_callback):
                raise E1Cancelled("E1 精度评价被用户中断")
            ref_raster = self._dataset_to_raster(
                reference, transform, (height, width), roi_gdf, gdf_cache
            )
            ref_tif = self.raster_dir / f"{roi_name}_{reference}_30m.tif"
            self._save_geotiff(ref_raster, transform, crs, ref_tif, cog=True)
            print(f"  参考栅格: {ref_tif}")

        def _attach_causal(pair_name: str, name_a: str, name_b: str, metrics: dict):
            metrics["causal_analysis"] = self._build_causal_analysis(
                pair_name,
                name_a,
                name_b,
                metrics,
                self.dataset_specs.get(name_a),
                self.dataset_specs.get(name_b),
            )
            if export_disagreement_maps and "disagreement_maps" in metrics:
                metrics["causal_analysis"]["disagreement_maps"] = metrics.pop(
                    "disagreement_maps"
                )

        if target_path:
            if _stop_requested(stop_callback):
                raise E1Cancelled("E1 精度评价被用户中断")
            target_path = ux.normalize_path(target_path, must_exist=True)
            target_gdf = self.normalize_vector(
                target_path,
                target_name,
                save=True,
                bbox_4326=resolved_bounds,
            )
            pair_name = f"{target_name}_vs_{reference}"
            ref_gdf = gdf_cache.get(reference)
            if ref_gdf is None:
                ref_gdf = self.load_dataset(
                    reference,
                    bbox_4326=resolved_bounds if local_scope else None,
                )
                gdf_cache[reference] = ref_gdf
            if not self._gdf_spatial_overlap(target_gdf, ref_gdf):
                ux.warn(ux.zero_overlap_message_e1(pair_name), print)
                metrics = {
                    "jaccard_iou": 0.0,
                    "intersection_km2": 0.0,
                    "union_km2": 0.0,
                    "only_a_km2": 0.0,
                    "only_b_km2": 0.0,
                    "area_a_km2": 0.0,
                    "area_b_km2": 0.0,
                    "intersection_pixels": 0,
                    "union_pixels": 0,
                    "zero_pixel_overlap": True,
                    "skipped_raster_compare": True,
                }
                _attach_causal(pair_name, target_name, reference, metrics)
                results["comparisons"][pair_name] = metrics
                print(f"  {pair_name} IoU = 0.0000 (无空间重叠，已安全跳过)")
            else:
                out_dir = self.output_dir / roi_name / pair_name
                writers = None
                metrics: Dict[str, Any] = {}
                if use_tiled and export_disagreement_maps:
                    writers = self._open_pair_disagreement_writers(
                        out_dir, transform, crs, width, height
                    )
                try:
                    if use_tiled:
                        metrics = self._compare_gdf_to_dataset_tiled(
                            target_gdf,
                            reference,
                            transform,
                            width,
                            height,
                            roi_gdf,
                            gdf_cache,
                            tile_size,
                            writers=writers,
                            stop_callback=stop_callback,
                        )
                    else:
                        if _stop_requested(stop_callback):
                            raise E1Cancelled("E1 精度评价被用户中断")
                        target_raster = self.vector_to_raster(
                            target_gdf, transform, (height, width), roi_gdf
                        )
                        ref_raster = self._dataset_to_raster(
                            reference, transform, (height, width), roi_gdf, gdf_cache
                        )
                        if _stop_requested(stop_callback):
                            raise E1Cancelled("E1 精度评价被用户中断")
                        if export_rasters:
                            self._save_geotiff(
                                target_raster,
                                transform,
                                crs,
                                self.raster_dir / f"{roi_name}_{target_name}_30m.tif",
                                cog=True,
                            )
                        metrics = self.compare_rasters(target_raster, ref_raster)
                        if export_disagreement_maps:
                            self._export_disagreement_rasters(
                                target_raster, ref_raster, transform, crs, roi_name, pair_name
                            )
                            metrics["disagreement_maps"] = self._disagreement_map_paths(out_dir)
                finally:
                    if writers:
                        paths = writers.pop("paths")
                        self._close_writers(writers)
                        metrics["disagreement_maps"] = {k: str(v) for k, v in paths.items()}

                _attach_causal(pair_name, target_name, reference, metrics)
                results["comparisons"][pair_name] = metrics
                print(f"  {pair_name} IoU = {metrics['jaccard_iou']:.4f}")

        successful_compare = []
        for src in compare_sources:
            if _stop_requested(stop_callback):
                raise E1Cancelled("E1 精度评价被用户中断")
            if src == reference:
                continue
            if src not in self.dataset_specs:
                print(f"  跳过未知数据集: {src}")
                continue

            pair_name = f"{src}_vs_{reference}"
            print(f"  对比 [{pair_name}] ...")
            out_dir = self.output_dir / roi_name / pair_name
            writers = None
            metrics: Dict[str, Any] = {}
            if use_tiled and export_disagreement_maps:
                writers = self._open_pair_disagreement_writers(
                    out_dir, transform, crs, width, height
                )
            try:
                if use_tiled:
                    metrics = self._compare_pair_tiled(
                        src,
                        reference,
                        transform,
                        width,
                        height,
                        roi_gdf,
                        gdf_cache,
                        tile_size,
                        writers=writers,
                        stop_callback=stop_callback,
                    )
                else:
                    if _stop_requested(stop_callback):
                        raise E1Cancelled("E1 精度评价被用户中断")
                    src_raster = self._dataset_to_raster(
                        src, transform, (height, width), roi_gdf, gdf_cache
                    )
                    ref_raster = self._dataset_to_raster(
                        reference, transform, (height, width), roi_gdf, gdf_cache
                    )
                    if _stop_requested(stop_callback):
                        raise E1Cancelled("E1 精度评价被用户中断")
                    if export_rasters:
                        self._save_geotiff(
                            src_raster,
                            transform,
                            crs,
                            self.raster_dir / f"{roi_name}_{src}_30m.tif",
                            cog=True,
                        )
                    metrics = self.compare_rasters(src_raster, ref_raster)
                    if export_disagreement_maps:
                        self._export_disagreement_rasters(
                            src_raster, ref_raster, transform, crs, roi_name, pair_name
                        )
                        metrics["disagreement_maps"] = self._disagreement_map_paths(out_dir)
            except E1Cancelled:
                if writers:
                    self._close_writers(writers)
                raise
            except Exception as exc:
                print(f"    失败: {exc}")
                results["comparisons"][pair_name] = {"error": str(exc)}
                if writers:
                    self._close_writers(writers)
                continue
            finally:
                if writers:
                    paths = writers.pop("paths")
                    self._close_writers(writers)
                    metrics["disagreement_maps"] = {k: str(v) for k, v in paths.items()}

            _attach_causal(pair_name, src, reference, metrics)
            results["comparisons"][pair_name] = metrics
            successful_compare.append(src)
            print(
                f"    IoU={metrics['jaccard_iou']:.4f} | "
                f"交集={metrics['intersection_km2']} km2 | "
                f"{src}={metrics['area_a_km2']} km2 | "
                f"{reference}={metrics['area_b_km2']} km2"
            )

        if export_multi_product_heatmap and len(successful_compare) >= 2:
            if _stop_requested(stop_callback):
                raise E1Cancelled("E1 精度评价被用户中断")
            print("\n  导出多产品一致热力图 ...")
            try:
                results["multi_product_heatmap"] = self._export_multi_product_heatmap_tiled(
                    successful_compare,
                    reference,
                    transform,
                    width,
                    height,
                    roi_gdf,
                    gdf_cache,
                    roi_name,
                    tile_size,
                    stop_callback=stop_callback,
                )
            except E1Cancelled:
                raise
            except Exception as exc:
                print(f"  多产品热力图失败: {exc}")
                results["multi_product_heatmap"] = {"error": str(exc)}

        if _stop_requested(stop_callback):
            raise E1Cancelled("E1 精度评价被用户中断")

        results["elapsed_seconds"] = round(time.perf_counter() - started_at, 3)
        report_path = self.output_dir / f"E1_PIXEL_REPORT_{roi_name}.json"
        # Keep deterministic provenance in the JSON itself.  This also makes
        # reports generated by the low-level engine verifiable after reload,
        # without relying on the wrapper having been the original caller.
        results["report_path"] = str(report_path)
        results["report_md_path"] = str(self.output_dir / f"E1_PIXEL_REPORT_{roi_name}.md")
        results["workspace_dir"] = str(self.workspace)
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        md_path = Path(results["report_md_path"])
        self._write_markdown_report(results, md_path)

        print(f"\n报告已保存: {report_path}")
        print(f"Markdown: {md_path}")
        return results

    def _disagreement_map_paths(self, out_dir: Path) -> Dict[str, str]:
        return {
            "consensus": str(out_dir / "consensus.tif"),
            "only_a": str(out_dir / "only_layer_a.tif"),
            "only_b": str(out_dir / "only_layer_b.tif"),
            "class": str(out_dir / "disagreement_class.tif"),
            "heatmap": str(out_dir / "disagreement_heatmap.tif"),
        }

    def _export_disagreement_rasters(
        self,
        raster_a: np.ndarray,
        raster_b: np.ndarray,
        transform,
        crs: str,
        roi_name: str,
        pair_name: str,
    ):
        cons, only_a, only_b, cls, heat = self._pair_disagreement_layers(raster_a, raster_b)
        base = self.output_dir / roi_name / pair_name
        self._save_geotiff(cons, transform, crs, base / "consensus.tif", cog=True)
        self._save_geotiff(only_a, transform, crs, base / "only_layer_a.tif", cog=True)
        self._save_geotiff(only_b, transform, crs, base / "only_layer_b.tif", cog=True)
        self._save_geotiff(cls, transform, crs, base / "disagreement_class.tif", cog=True)
        self._save_geotiff(heat, transform, crs, base / "disagreement_heatmap.tif", cog=True)

    # ------------------------------------------------------------------
    # 兼容旧接口
    # ------------------------------------------------------------------
    def clean_external_asset(
        self,
        input_path: str,
        asset_name: str,
        target_crs: str = TARGET_VECTOR_CRS,
        raster_flat_val: int = 1,
        filter_spec: Optional[dict] = None,
    ) -> str:
        """统一矢量至 gpkg（不再 dissolve 成单面）。"""
        path = Path(input_path)
        if path.suffix.lower() in {".tif", ".tiff"}:
            spec = {"kind": "vector", "path": path}
            gdf = self._tif_folder_to_vectors(path.parent, [raster_flat_val])
            gdf["source"] = asset_name
            gdf = gdf.to_crs(target_crs)
            out = self.unified_dir / f"{asset_name}.gpkg"
            gpd.GeoDataFrame({"source": asset_name, "geometry": gdf.geometry}, crs=target_crs).to_file(
                out, driver="GPKG"
            )
            return str(out)

        gdf = self.normalize_vector(path, asset_name, filter_spec=filter_spec, save=True)
        return str(self.unified_dir / f"{asset_name}.gpkg")

    def diagnose_consistency(
        self,
        my_product_shp: str,
        external_shp_dict: Optional[dict] = None,
        roi_name: str = "roi",
        reference: str = "师姐_2020",
        roi_path: Optional[str] = None,
    ) -> dict:
        """
        兼容旧调用方式：以师姐为 reference，你的产品 + 外部产品做像元级 IoU。
        external_shp_dict 若提供，会先把路径注册为临时数据集。
        """
        if external_shp_dict:
            for name, path in external_shp_dict.items():
                self.dataset_specs[name] = {"kind": "vector", "path": Path(path)}

        compare = [n for n in self.list_datasets() if n != reference]
        if external_shp_dict:
            compare = list(external_shp_dict.keys()) + [
                n for n in compare if n not in external_shp_dict
            ]

        return self.run_pixel_comparison(
            reference=reference,
            target_path=my_product_shp,
            target_name="My_Product",
            compare_sources=compare,
            roi_path=roi_path,
            roi_name=roi_name,
        )


if __name__ == "__main__":
    # 建议使用已安装 geopandas 的环境，例如: D:\anaconda3\envs\gwx\python.exe E1.py
    WORKSPACE = r"E:\Code\GEE\jb\e1_workspace"
    e1 = E1_DataCleanerAndDiagnostic(
        workspace_dir=WORKSPACE,
        data_root=DEFAULT_DATA_ROOT,
    )

    print("可用数据集:", e1.list_datasets())

    # 1) 可选：批量统一 -> e1_workspace/E1_unified/*.gpkg（数据量大时较慢）
    # e1.normalize_all_registered()

    # 2) 像元级对比：以师姐 2020 为 reference
    #    roi_path: 你的自定义研究区 shp（不必按省界），None = 中国海岸默认范围
    e1.run_pixel_comparison(
        reference="师姐_2020",
        target_path=None,  # 你的成果 shp 路径（有则填入）
        compare_sources=[
            "DCTF_2020",
            "FCS30_2020",
            "GTF30_2020",
            "CHN_2024",
            "MTWM_2020",
            "TFMC_2020",
            "national_10m_2020",
            # "CHN_2016",
            # "Murray_2014_2016",  # 全球 tif，极慢，需要时再开
        ],
        roi_path=None,
        roi_name="china_coast",
    )

    # 3) 示例：你的产品 vs 师姐 + 自定义 ROI
    # MY = r"E:\Code\GEE\YYnet\DATA\final_result\outputs\Final_Intertidal_Flat.shp"
    # e1.run_pixel_comparison(
    #     reference="师姐_2020",
    #     target_path=MY,
    #     compare_sources=["DCTF_2020", "FCS30_2020"],
    #     roi_path=r"E:\path\to\your_roi.shp",
    #     roi_name="my_study_area",
    # )
