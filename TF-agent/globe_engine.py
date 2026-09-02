"""
Cesium 3D 地球引擎 — 用于 YYnet 中央地理显示区。

- Cesium Ion 高清卫星底图（.env 中 CESIUM_ION_TOKEN）
- 潮滩 SHP → 球面 GeoJSON 叠加
- 潮滩 TIF → localtileserver 瓦片贴图到球面
"""

from __future__ import annotations

import json
import math
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

_CESIUM_VER = "1.128"
_CESIUM_JS = f"https://cesium.com/downloads/cesiumjs/releases/{_CESIUM_VER}/Build/Cesium/Cesium.js"
_CESIUM_CSS = f"https://cesium.com/downloads/cesiumjs/releases/{_CESIUM_VER}/Build/Cesium/Widgets/widgets.css"
_CESIUM_NEII_URL = (
    f"https://cesium.com/downloads/cesiumjs/releases/{_CESIUM_VER}/Build/Cesium/Assets/Textures/NaturalEarthII"
)
_BORDERS_GEOJSON_URL = (
    "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_110m_admin_0_countries.geojson"
)

_BASE_IMAGERY_CANDIDATES = [
    {
        "url": "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        "maxLevel": 19,
        "credit": "Esri World Imagery",
    },
    {
        "url": "https://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        "maxLevel": 19,
        "credit": "Esri",
    },
    {
        "url": "https://tile.openstreetmap.org/{z}/{x}/{y}.png",
        "maxLevel": 19,
        "credit": "OpenStreetMap",
    },
    {
        "url": "https://basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}.png",
        "maxLevel": 19,
        "credit": "CARTO",
    },
]

# 默认框选中国大陆（略收紧边界，使视口尽可能铺满中国）
_CHINA_VIEW_RECT = {"west": 73.0, "south": 17.0, "east": 136.0, "north": 54.0}

# 统一相机视觉风格：初始中国总览与 Agent 跳转共用 heading/pitch/roll
# pitch 需足够俯视（约 -55°），否则 fromDegrees 高度点 + 浅俯角会把地球甩到画面底部/地平线
DEFAULT_CAMERA = {
    "heading_deg": 0.0,
    "pitch_deg": -55.0,
    "roll_deg": 0.0,
    "duration": 1.2,
    # lookAt 距离（米）：对准中国中心，框入完整中国陆域
    "china_range_m": 4_800_000.0,
    "china_center": {"lat": 36.0, "lon": 104.0},
    # 海湾级 / 点位 lookAt 距离
    "region_range_m": 280_000.0,
    "point_range_m": 90_000.0,
}


def zoom_to_height_m(zoom: int, lat: float = 30.0) -> float:
    """将 Web 地图缩放级别映射为 Cesium lookAt 距离（米）。"""
    z = max(1, min(18, int(zoom)))
    if z <= 4:
        return float(DEFAULT_CAMERA["china_range_m"])
    if z <= 7:
        return 1_000_000.0
    if z <= 10:
        return float(DEFAULT_CAMERA["region_range_m"])
    if z <= 12:
        return float(DEFAULT_CAMERA["point_range_m"])
    return 35_000.0


def height_from_rectangle_span(west: float, south: float, east: float, north: float) -> float:
    """由矩形跨度估算观察高度，保持与 DEFAULT_CAMERA 相同的斜视风格。"""
    span = max(abs(east - west), abs(north - south), 0.02)
    h = span * 111_000.0 * 2.4
    return float(max(8_000.0, min(h, 8_000_000.0)))


def view_from_vector_path(path: str) -> Optional[Tuple[float, float, int]]:
    try:
        import geopandas as gpd

        gdf = gpd.read_file(path)
        if gdf.empty:
            return None
        if gdf.crs is not None:
            gdf = gdf.to_crs(4326)
        minx, miny, maxx, maxy = gdf.total_bounds
        return _view_from_bounds(minx, miny, maxx, maxy)
    except Exception:
        return None


def view_from_raster_path(path: str) -> Optional[Tuple[float, float, int]]:
    try:
        import rasterio
        from rasterio.warp import transform_bounds

        with rasterio.open(path) as ds:
            b = ds.bounds
            crs = ds.crs
        if crs is not None:
            west, south, east, north = transform_bounds(crs, "EPSG:4326", b.left, b.bottom, b.right, b.top)
        else:
            west, south, east, north = b.left, b.bottom, b.right, b.top
        return _view_from_bounds(west, south, east, north)
    except Exception:
        return None


def view_from_asset_path(path: str) -> Optional[Tuple[float, float, int]]:
    if not path or not os.path.isfile(path):
        return None
    ext = os.path.splitext(path)[1].lower()
    if ext == ".shp":
        return view_from_vector_path(path)
    if ext in {".tif", ".tiff"}:
        return view_from_raster_path(path)
    return None


def _view_from_bounds(west: float, south: float, east: float, north: float) -> Tuple[float, float, int]:
    lat = (south + north) / 2.0
    lon = (west + east) / 2.0
    span = max(east - west, north - south)
    if span > 8:
        zoom = 5
    elif span > 3:
        zoom = 7
    elif span > 1:
        zoom = 9
    elif span > 0.3:
        zoom = 11
    elif span > 0.08:
        zoom = 13
    else:
        zoom = 15
    return lat, lon, zoom


def bounds_from_asset_path(path: str) -> Optional[Tuple[float, float, float, float]]:
    try:
        import geopandas as gpd
        import rasterio
        from rasterio.warp import transform_bounds
    except ImportError:
        return None
    if not path or not os.path.isfile(path):
        return None
    ext = os.path.splitext(path)[1].lower()
    try:
        if ext == ".shp":
            gdf = gpd.read_file(path)
            if gdf.empty:
                return None
            gdf = gdf.to_crs(4326)
            minx, miny, maxx, maxy = gdf.total_bounds
            return float(minx), float(miny), float(maxx), float(maxy)
        if ext in {".tif", ".tiff"}:
            with rasterio.open(path) as ds:
                b = ds.bounds
                crs = ds.crs
            if crs is not None:
                w, s, e, n = transform_bounds(crs, "EPSG:4326", b.left, b.bottom, b.right, b.top)
            else:
                w, s, e, n = b.left, b.bottom, b.right, b.top
            return float(w), float(s), float(e), float(n)
    except Exception:
        return None
    return None


def _simplify_geojson(geojson: dict, max_features: int = 8000) -> dict:
    feats = geojson.get("features") or []
    if len(feats) <= max_features:
        return geojson
    step = max(1, len(feats) // max_features)
    geojson = dict(geojson)
    geojson["features"] = feats[::step][:max_features]
    return geojson


def load_shp_geojson(path: str, simplify_tolerance: Optional[float] = None) -> Optional[dict]:
    try:
        import geopandas as gpd

        gdf = gpd.read_file(path)
        if gdf.empty:
            return None
        gdf = gdf.to_crs(4326)
        minx, miny, maxx, maxy = gdf.total_bounds
        span = max(maxx - minx, maxy - miny)
        tol = simplify_tolerance
        if tol is None:
            if span > 2:
                tol = 0.01
            elif span > 0.5:
                tol = 0.002
            elif span > 0.1:
                tol = 0.0005
            else:
                tol = 0.0001
        if tol and tol > 0:
            gdf = gdf.copy()
            gdf["geometry"] = gdf.geometry.simplify(tol, preserve_topology=True)
        geojson = json.loads(gdf.to_json())
        return _simplify_geojson(geojson)
    except Exception:
        return None


def _nodata_safe(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        if isinstance(value, (float, np.floating)) and (np.isnan(value) or np.isinf(value)):
            return None
    except (TypeError, ValueError):
        return None
    return float(value)


def infer_raster_tile_params(path: str) -> dict:
    """与 2D 地图一致：单波段成果用 Reds 色图，nodata=0 透明背景。"""
    out: dict[str, Any] = {"indexes": None, "colormap": None, "nodata": None}
    try:
        import rasterio

        with rasterio.open(path) as ds:
            nb = int(ds.count)
            nd = _nodata_safe(ds.nodata)
            if nb == 1:
                out["indexes"] = 1
                out["colormap"] = "reds"
                if nd is None:
                    dt = str(ds.dtypes[0])
                    if dt.startswith(("uint", "int")) or "float" in dt:
                        nd = 0.0
                out["nodata"] = nd
    except Exception:
        pass
    return out


def _tile_client_alive(client: Any) -> bool:
    try:
        import urllib.request

        raw = client.get_tile_url().replace("localhost", "127.0.0.1")
        test = raw.replace("{z}", "0").replace("{x}", "0").replace("{y}", "0")
        with urllib.request.urlopen(test, timeout=4) as resp:
            return int(resp.status) == 200
    except Exception:
        return False


def get_raster_tile_overlay(
    path: str,
    tile_clients: Dict[str, Any],
    globe_port: Optional[int] = None,
    *,
    force_local: bool = False,
) -> Optional[dict]:
    try:
        from localtileserver import TileClient
    except ImportError:
        return None
    if not path or not os.path.isfile(path):
        return None
    try:
        mt = os.path.getmtime(path)
    except OSError:
        mt = 0
    key = f"{os.path.normpath(os.path.abspath(path))}|{mt:.4f}"
    client = tile_clients.get(key)
    if client is None:
        try:
            client = TileClient(path, host="127.0.0.1")
            tile_clients[key] = client
        except Exception:
            return None

    params = infer_raster_tile_params(path)
    url_kwargs: dict[str, Any] = {}
    if params.get("indexes") is not None:
        url_kwargs["indexes"] = params["indexes"]
    if params.get("colormap"):
        url_kwargs["colormap"] = params["colormap"]
    if params.get("nodata") is not None:
        url_kwargs["nodata"] = params["nodata"]

    bounds = client.bounds()
    south, north, west, east = bounds
    raw_url = client.get_tile_url(**url_kwargs).replace("localhost", "127.0.0.1")
    if globe_port:
        import globe_server as gs

        token = gs.register_tile_template(key, raw_url)
        url = gs.overlay_tile_url(globe_port, token, force_local=force_local)
    else:
        url = raw_url
    return {
        "url": url,
        "west": float(west),
        "south": float(south),
        "east": float(east),
        "north": float(north),
        "min_zoom": int(getattr(client, "min_zoom", 0) or 0),
        "max_zoom": int(getattr(client, "max_zoom", 18) or 18),
    }


def find_e1_overlay_path(e1_report: Optional[dict], prefer: str = "heatmap") -> Optional[str]:
    if not e1_report:
        return None
    for _pair, metrics in (e1_report.get("comparisons") or {}).items():
        causal = metrics.get("causal_analysis") or {}
        maps = causal.get("disagreement_maps") or {}
        for key in (prefer, "heatmap", "consensus", "class"):
            p = maps.get(key)
            if p and os.path.isfile(p):
                return p
    mp = e1_report.get("multi_product_heatmap") or {}
    for key in ("any_disagreement_tif", "agreement_count_tif"):
        p = mp.get(key)
        if p and os.path.isfile(p):
            return p
    return None


def build_globe_payload(
    center: Tuple[float, float],
    zoom: int,
    result_path: Optional[str] = None,
    opacity_pct: float = 50.0,
    pitch_deg: float = -55.0,
    e1_report: Optional[dict] = None,
    show_e1_overlay: bool = False,
    tile_clients: Optional[Dict[str, Any]] = None,
    ion_token: Optional[str] = None,
    show_borders: bool = True,
    globe_port: Optional[int] = None,
    prefer_center: bool = False,
    force_local: bool = False,
    channel_id: Optional[str] = None,
) -> dict:
    lat, lon = float(center[0]), float(center[1])
    tile_clients = tile_clients if tile_clients is not None else {}

    payload: Dict[str, Any] = {
        "center": {"lat": lat, "lon": lon},
        "height": zoom_to_height_m(zoom, lat),
        "pitch": float(pitch_deg if pitch_deg is not None else DEFAULT_CAMERA["pitch_deg"]),
        "heading": float(DEFAULT_CAMERA["heading_deg"]),
        "roll": float(DEFAULT_CAMERA["roll_deg"]),
        "duration": float(DEFAULT_CAMERA["duration"]),
        "flyRectangle": None,
        "preferCenter": bool(prefer_center),
        "geojsonLayers": [],
        "imageryLayers": [],
        "opacity": max(0.05, min(1.0, opacity_pct / 100.0)),
        "ionToken": (ion_token or "").strip() or None,
        "showBorders": bool(show_borders),
        "bordersUrl": _BORDERS_GEOJSON_URL,
        "naturalEarthUrl": _CESIUM_NEII_URL,
        "assetName": None,
        "chinaView": dict(_CHINA_VIEW_RECT),
        "defaultCamera": {
            "headingDeg": float(DEFAULT_CAMERA["heading_deg"]),
            "pitchDeg": float(DEFAULT_CAMERA["pitch_deg"]),
            "rollDeg": float(DEFAULT_CAMERA["roll_deg"]),
            "duration": float(DEFAULT_CAMERA["duration"]),
            "chinaRange": float(DEFAULT_CAMERA["china_range_m"]),
            "regionRange": float(DEFAULT_CAMERA["region_range_m"]),
            "pointRange": float(DEFAULT_CAMERA["point_range_m"]),
            "chinaCenter": dict(DEFAULT_CAMERA["china_center"]),
        },
        "debugCamera": bool(os.environ.get("CSTF_GLOBE_DEBUG", "").strip() in {"1", "true", "yes"}),
        # Per-Streamlit-session channel prevents AOI messages from one browser
        # session being consumed by another session sharing this local server.
        "channelId": str(channel_id or "default"),
    }

    rects: List[Tuple[float, float, float, float]] = []

    if result_path and os.path.isfile(result_path):
        ext = os.path.splitext(result_path)[1].lower()
        name = os.path.splitext(os.path.basename(result_path))[0]
        payload["assetName"] = name
        if ext == ".shp":
            gj = load_shp_geojson(result_path)
            if gj:
                payload["geojsonLayers"].append(
                    {"name": name, "data": gj, "color": "#e41a1c", "alpha": payload["opacity"]}
                )
            b = bounds_from_asset_path(result_path)
            if b:
                rects.append(b)
        elif ext in {".tif", ".tiff"}:
            tile = get_raster_tile_overlay(
                result_path, tile_clients, globe_port=globe_port, force_local=force_local
            )
            if tile:
                payload["imageryLayers"].append(
                    {
                        "name": name,
                        "url": tile["url"],
                        "west": tile["west"],
                        "south": tile["south"],
                        "east": tile["east"],
                        "north": tile["north"],
                        "minZoom": tile["min_zoom"],
                        "maxZoom": tile["max_zoom"],
                        "alpha": payload["opacity"],
                    }
                )
                rects.append((tile["west"], tile["south"], tile["east"], tile["north"]))

    if show_e1_overlay and e1_report:
        e1_path = find_e1_overlay_path(e1_report)
        if e1_path and os.path.isfile(e1_path):
            ext = os.path.splitext(e1_path)[1].lower()
            if ext == ".shp":
                gj = load_shp_geojson(e1_path)
                if gj:
                    payload["geojsonLayers"].append(
                        {"name": "精度评价结果", "data": gj, "color": "#ff6b35", "alpha": 0.65}
                    )
            elif ext in {".tif", ".tiff"}:
                tile = get_raster_tile_overlay(
                    e1_path, tile_clients, globe_port=globe_port, force_local=force_local
                )
                if tile:
                    payload["imageryLayers"].append(
                        {
                            "name": "精度评价结果",
                            "url": tile["url"],
                            "west": tile["west"],
                            "south": tile["south"],
                            "east": tile["east"],
                            "north": tile["north"],
                            "minZoom": tile["min_zoom"],
                            "maxZoom": tile["max_zoom"],
                            "alpha": 0.7,
                        }
                    )
                    rects.append((tile["west"], tile["south"], tile["east"], tile["north"]))

    if rects and not prefer_center:
        # 有成果图层时默认飞到图层范围；智能体显式跳转时 prefer_center=True，改用 center/height
        west = min(r[0] for r in rects)
        south = min(r[1] for r in rects)
        east = max(r[2] for r in rects)
        north = max(r[3] for r in rects)
        pad_lon = max(0.02, (east - west) * 0.08)
        pad_lat = max(0.02, (north - south) * 0.08)
        payload["flyRectangle"] = {
            "west": west - pad_lon,
            "south": south - pad_lat,
            "east": east + pad_lon,
            "north": north + pad_lat,
        }
    elif result_path and not prefer_center:
        auto = view_from_asset_path(result_path)
        if auto:
            payload["center"] = {"lat": auto[0], "lon": auto[1]}
            payload["height"] = zoom_to_height_m(auto[2], auto[0])

    return payload


def _json_for_script(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False).replace("</", "<\\/")


def build_cesium_html(payload: dict, height_px: int = 700, full_viewport: bool = True) -> str:
    cfg = _json_for_script(payload)
    imagery_candidates = _json_for_script(_BASE_IMAGERY_CANDIDATES)
    if full_viewport:
        size_rule = "width: 100%; height: 100%;"
        container_rule = "width: 100%; height: 100%;"
    else:
        h = int(max(480, height_px))
        size_rule = f"width: 100%; height: {h}px;"
        container_rule = size_rule
    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <link rel="stylesheet" href="{_CESIUM_CSS}"/>
  <script src="{_CESIUM_JS}"></script>
  <style>
    html, body {{
      {size_rule} margin: 0; padding: 0; overflow: hidden;
      background: #02040a;
    }}
    #cesiumContainer {{
      {container_rule} margin: 0; padding: 0; overflow: hidden;
    }}
    .cesium-viewer,
    .cesium-viewer-cesiumWidgetContainer,
    .cesium-widget,
    .cesium-widget canvas {{
      width: 100% !important;
      height: 100% !important;
      display: block;
    }}
    #cesiumError {{
      display: none; position: absolute; top: 12px; left: 12px; right: 12px;
      padding: 10px 14px; background: rgba(120, 20, 20, 0.92); color: #fff;
      font: 13px/1.4 sans-serif; border-radius: 4px; z-index: 9999;
    }}
    #cesiumStatus {{
      position: fixed; bottom: 10px; left: 10px; z-index: 99999;
      padding: 5px 10px; background: rgba(8, 14, 28, 0.82); color: #b8c8e8;
      font: 12px/1.35 sans-serif; border-radius: 6px; border: 1px solid #2a3a55;
      pointer-events: none;
    }}
    #aoiToolbar {{
      position: fixed; top: 10px; right: 10px; z-index: 99998;
      display: flex; gap: 6px; padding: 6px 8px;
      background: rgba(8, 14, 28, 0.85); border: 1px solid #2a3a55;
      border-radius: 8px; font: 12px/1.3 sans-serif;
    }}
    #aoiToolbar button {{
      border: 1px solid #3a4a6a; background: #111c30; color: #cfe0ff;
      border-radius: 5px; padding: 4px 9px; cursor: pointer; white-space: nowrap;
    }}
    #aoiToolbar button:hover {{ background: #1c2c48; }}
    #aoiToolbar button.active {{
      background: #2e6ac0; border-color: #7fb2ff; color: #fff;
    }}
    .cesium-viewer-bottom {{ display: none !important; }}
    .cesium-navigation-help-button,
    .cesium-navigation-help-wrapper {{ display: none !important; }}
    .cesium-viewer .cesium-widget-credits {{ font-size: 10px; opacity: 0.55; }}
  </style>
</head>
<body>
<div id="cesiumError"></div>
<div id="cesiumStatus">地球初始化中…</div>
<div id="aoiToolbar">
  <button id="aoiBtnClick" title="点选（以点击点为中心的小方框）">点选</button>
  <button id="aoiBtnRect" title="拖拽绘制矩形">矩形</button>
  <button id="aoiBtnPoly" title="左键加点，右键闭合（至少 3 点）">多边形</button>
  <button id="aoiBtnView" title="使用当前视野范围">当前视图</button>
  <button id="aoiBtnClear" title="清除当前 AOI（不影响业务图层）">清除</button>
</div>
<div id="cesiumContainer"></div>
<script>
(async function() {{
  const CFG = {cfg};
  const IMAGERY_CANDIDATES = {imagery_candidates};

  function setStatus(msg) {{
    const el = document.getElementById("cesiumStatus");
    if (el) el.textContent = msg;
  }}

  function showError(msg) {{
    const el = document.getElementById("cesiumError");
    if (el) {{ el.style.display = "block"; el.textContent = msg; }}
    setStatus("底图异常");
    console.error(msg);
  }}

  function disableDynamicLighting(viewer) {{
    viewer.scene.globe.enableLighting = false;
    viewer.scene.globe.showGroundAtmosphere = false;
    viewer.scene.highDynamicRange = false;
    if (viewer.scene.fog) viewer.scene.fog.enabled = false;
    if (viewer.scene.skyAtmosphere) viewer.scene.skyAtmosphere.show = false;
    if (viewer.scene.sun) viewer.scene.sun.show = false;
    if (viewer.scene.moon) viewer.scene.moon.show = false;
    if (viewer.scene.skyBox) viewer.scene.skyBox.show = false;
    if (viewer.scene.atmosphere && Cesium.DynamicAtmosphereLightingType) {{
      viewer.scene.atmosphere.dynamicLighting = Cesium.DynamicAtmosphereLightingType.NONE;
    }}
    if (viewer.scene.globe.dynamicAtmosphereLighting !== undefined) {{
      viewer.scene.globe.dynamicAtmosphereLighting = false;
    }}
  }}

  async function tryAddImagery(viewer, provider, label) {{
    try {{
      viewer.imageryLayers.addImageryProvider(provider);
      setStatus("底图 · " + label);
      return true;
    }} catch (e) {{
      console.warn("imagery add failed:", label, e);
      return false;
    }}
  }}

  async function setupBaseImagery(viewer) {{
    viewer.imageryLayers.removeAll();
    if (CFG.ionToken) {{
      Cesium.Ion.defaultAccessToken = CFG.ionToken;
      try {{
        const ionProvider = await Cesium.IonImageryProvider.fromAssetId(2);
        if (await tryAddImagery(viewer, ionProvider, "Cesium Ion 卫星")) return;
      }} catch (e) {{ console.warn("Ion asset 2 failed", e); }}
      try {{
        const worldProvider = await Cesium.createWorldImageryAsync({{
          style: Cesium.IonWorldImageryStyle.AERIAL,
        }});
        if (await tryAddImagery(viewer, worldProvider, "Cesium Ion World")) return;
      }} catch (e) {{ console.warn("createWorldImageryAsync failed", e); }}
    }}
    for (let i = 0; i < IMAGERY_CANDIDATES.length; i++) {{
      const item = IMAGERY_CANDIDATES[i];
      try {{
        const p = new Cesium.UrlTemplateImageryProvider({{
          url: item.url,
          maximumLevel: item.maxLevel || 18,
          credit: item.credit || "",
        }});
        if (await tryAddImagery(viewer, p, item.credit || ("fallback-" + i))) return;
      }} catch (e) {{ console.warn("url imagery failed", i, e); }}
    }}
    try {{
      const neii = await Cesium.TileMapServiceImageryProvider.fromUrl(
        Cesium.buildModuleUrl("Assets/Textures/NaturalEarthII")
      );
      if (await tryAddImagery(viewer, neii, "Natural Earth II")) return;
    }} catch (e) {{ console.warn("NEII failed", e); }}
    showError("底图加载失败，请检查 CESIUM_ION_TOKEN 或网络。");
  }}

  // Viewer 原则上每个 iframe 生命周期只创建一次；Agent 跳转应走 postMessage，勿重建页面
  window.__cstfViewerInitCount = (window.__cstfViewerInitCount || 0) + 1;
  if (CFG.debugCamera) {{
    console.debug("[CesiumViewer] initialized", window.__cstfViewerInitCount);
  }}

  let viewer;
  try {{
    viewer = new Cesium.Viewer("cesiumContainer", {{
      animation: false,
      timeline: false,
      geocoder: !!CFG.ionToken,
      homeButton: true,
      sceneModePicker: true,
      baseLayerPicker: false,
      navigationHelpButton: false,
      fullscreenButton: true,
      infoBox: false,
      selectionIndicator: false,
      terrainProvider: new Cesium.EllipsoidTerrainProvider(),
      shouldAnimate: false,
      skyBox: false,
      skyAtmosphere: false,
      // 静止时少占 GPU；相机/图层变更后主动 requestRender
      requestRenderMode: true,
      maximumRenderTimeChange: Infinity,
      baseLayer: false,
    }});
  }} catch (err) {{
    showError("地球初始化失败: " + err);
    return;
  }}

  disableDynamicLighting(viewer);
  viewer.scene.mode = Cesium.SceneMode.SCENE3D;
  viewer.scene.backgroundColor = Cesium.Color.fromBytes(2, 4, 12, 255);
  viewer.scene.globe.show = true;
  viewer.scene.globe.depthTestAgainstTerrain = false;
  viewer.scene.globe.maximumScreenSpaceError = 2;
  viewer.scene.fxaa = true;
  if (viewer.cesiumWidget && viewer.cesiumWidget.container) {{
    viewer.cesiumWidget.container.style.width = "100%";
    viewer.cesiumWidget.container.style.height = "100%";
  }}
  // 禁止 Entity 跟踪改相机；Home 按钮走统一导航
  viewer.trackedEntity = undefined;
  try {{
    if (viewer.homeButton && viewer.homeButton.viewModel) {{
      viewer.homeButton.viewModel.command.beforeExecute.addEventListener(function(e) {{
        e.cancel = true;
        navigateToChinaOverview({{ duration: 1.2, source: "homeButton" }});
      }});
    }}
  }} catch (e) {{}}

  await setupBaseImagery(viewer);
  viewer.scene.requestRender();

  function hexToCesiumColor(hex, alpha) {{
    const h = hex.replace("#", "");
    return Cesium.Color.fromBytes(
      parseInt(h.substring(0, 2), 16),
      parseInt(h.substring(2, 4), 16),
      parseInt(h.substring(4, 6), 16),
      Math.round((alpha || 0.5) * 255)
    );
  }}

  (CFG.geojsonLayers || []).forEach(function(layer) {{
    Cesium.GeoJsonDataSource.load(layer.data, {{
      clampToGround: true,
      stroke: hexToCesiumColor(layer.color || "#e41a1c", 0.95),
      fill: hexToCesiumColor(layer.color || "#e41a1c", layer.alpha || 0.5),
      strokeWidth: 2,
    }}).then(function(ds) {{
      ds.name = layer.name || "layer";
      viewer.dataSources.add(ds);
    }}).catch(function(err) {{ console.warn("GeoJSON load failed", err); }});
  }});

  (CFG.imageryLayers || []).forEach(function(layer) {{
    try {{
      const rect = Cesium.Rectangle.fromDegrees(layer.west, layer.south, layer.east, layer.north);
      const provider = new Cesium.UrlTemplateImageryProvider({{
        url: layer.url,
        rectangle: rect,
        minimumLevel: layer.minZoom || 0,
        maximumLevel: layer.maxZoom || 18,
      }});
      const imgLayer = viewer.imageryLayers.addImageryProvider(provider);
      imgLayer.alpha = layer.alpha != null ? layer.alpha : 0.75;
    }} catch (err) {{ console.warn("overlay imagery failed", err); }}
  }});

  function rectFromCfg(box) {{
    return Cesium.Rectangle.fromDegrees(box.west, box.south, box.east, box.north);
  }}

  const CAM = CFG.defaultCamera || {{}};
  const DEG2RAD = Cesium.Math.toRadians;
  let _navSeq = 0;
  let _lastNavKey = "";

  function cameraSnapshot() {{
    const c = viewer.camera.positionCartographic;
    return {{
      lon: Cesium.Math.toDegrees(c.longitude),
      lat: Cesium.Math.toDegrees(c.latitude),
      height: c.height,
      heading: Cesium.Math.toDegrees(viewer.camera.heading),
      pitch: Cesium.Math.toDegrees(viewer.camera.pitch),
      roll: Cesium.Math.toDegrees(viewer.camera.roll),
    }};
  }}

  function debugCamera(tag, extra) {{
    if (!CFG.debugCamera) return;
    console.debug("[MapCamera]", tag, extra || "", cameraSnapshot());
  }}

  function defaultPitchDeg() {{
    const p = (CFG.pitch != null ? CFG.pitch : (CAM.pitchDeg != null ? CAM.pitchDeg : -55));
    return Number(p);
  }}

  function defaultHeadingDeg() {{
    const h = (CFG.heading != null ? CFG.heading : (CAM.headingDeg != null ? CAM.headingDeg : 0));
    return Number(h);
  }}

  function heightForRectangle(box) {{
    const west = Number(box.west), south = Number(box.south);
    const east = Number(box.east), north = Number(box.north);
    const span = Math.max(Math.abs(east - west), Math.abs(north - south), 0.02);
    const h = span * 111000 * 2.8;
    return Math.max(60000, Math.min(h, 6000000));
  }}

  // 用 lookAt 对准目标点：避免「相机站在目标正上方 + 浅俯角」导致地球掉到底部/只见地平线
  function applyLookAtView(lon, lat, range, pitchDeg, headingDeg) {{
    const target = Cesium.Cartesian3.fromDegrees(lon, lat, 0);
    const offset = new Cesium.HeadingPitchRange(
      DEG2RAD(headingDeg),
      DEG2RAD(pitchDeg),
      Math.max(1000, range)
    );
    viewer.camera.lookAt(target, offset);
    // 解锁相机，否则用户无法拖拽，后续 flyTo 也会异常
    viewer.camera.lookAtTransform(Cesium.Matrix4.IDENTITY);
  }}

  function navigateToLocation(opts) {{
    if (!viewer || !opts) return false;
    const lon = Number(opts.longitude ?? opts.lon);
    const lat = Number(opts.latitude ?? opts.lat);
    let range = Number(opts.height ?? opts.heightM ?? opts.range);
    if (!isFinite(lon) || !isFinite(lat) || lon < -180 || lon > 180 || lat < -90 || lat > 90) {{
      console.warn("[MapCamera] invalid lon/lat", opts);
      return false;
    }}
    if (!isFinite(range) || range <= 0) {{
      range = Number(CFG.height) || Number(CAM.regionRange) || 280000;
    }}
    const duration = (opts.duration != null && isFinite(Number(opts.duration)))
      ? Number(opts.duration)
      : Number(CFG.duration || CAM.duration || 1.2);
    const pitchDeg = (opts.pitch != null) ? Number(opts.pitch) : defaultPitchDeg();
    const headingDeg = (opts.heading != null) ? Number(opts.heading) : defaultHeadingDeg();
    const key = [lat.toFixed(4), lon.toFixed(4), range.toFixed(0), pitchDeg.toFixed(1)].join("|");
    if (_lastNavKey === key && !opts.force) {{
      debugCamera("skip duplicate", {{ source: opts.source || "unknown" }});
      return true;
    }}
    _lastNavKey = key;
    const seq = ++_navSeq;
    debugCamera("before navigation", {{
      target: {{ lat, lon, range, pitchDeg, headingDeg }},
      source: opts.source || "unknown",
    }});

    try {{ viewer.trackedEntity = undefined; }} catch (e) {{}}
    try {{
      if (viewer.camera && viewer.camera.cancelFlight) viewer.camera.cancelFlight();
    }} catch (e) {{}}

    const target = Cesium.Cartesian3.fromDegrees(lon, lat, 0);
    const sphere = new Cesium.BoundingSphere(target, Math.max(range * 0.08, 2000));
    const offset = new Cesium.HeadingPitchRange(
      DEG2RAD(headingDeg),
      DEG2RAD(pitchDeg),
      Math.max(1000, range)
    );
    const complete = function() {{
      if (seq !== _navSeq) return;
      try {{ viewer.camera.lookAtTransform(Cesium.Matrix4.IDENTITY); }} catch (e) {{}}
      debugCamera("after navigation", {{ source: opts.source || "unknown" }});
      viewer.scene.requestRender();
    }};

    try {{
      if (duration <= 0.05) {{
        applyLookAtView(lon, lat, range, pitchDeg, headingDeg);
        complete();
      }} else {{
        viewer.camera.flyToBoundingSphere(sphere, {{
          duration: duration,
          offset: offset,
          complete: complete,
        }});
      }}
    }} catch (err) {{
      console.warn("[MapCamera] flyToBoundingSphere failed, fallback lookAt", err);
      try {{
        applyLookAtView(lon, lat, range, pitchDeg, headingDeg);
        complete();
      }} catch (e2) {{
        return false;
      }}
    }}
    viewer.scene.requestRender();
    return true;
  }}

  function navigateToRectangle(box, opts) {{
    if (!box) return false;
    const rect = rectFromCfg(box);
    const center = Cesium.Rectangle.center(rect);
    const lon = Cesium.Math.toDegrees(center.longitude);
    const lat = Cesium.Math.toDegrees(center.latitude);
    const range = heightForRectangle(box);
    return navigateToLocation({{
      longitude: lon,
      latitude: lat,
      height: range,
      duration: (opts && opts.duration != null) ? opts.duration : (CFG.duration || 1.0),
      source: (opts && opts.source) || "rectangle",
      force: !!(opts && opts.force),
    }});
  }}

  function navigateToChinaOverview(opts) {{
    const cv = CFG.chinaView || {{ west: 78, south: 21, east: 128, north: 50 }};
    const rect = rectFromCfg(cv);
    const cc = CAM.chinaCenter || {{ lat: 36, lon: 104 }};
    const range = Number(CAM.chinaRange || 4800000);
    const src = (opts && opts.source) || "chinaOverview";
    const duration = (opts && opts.duration != null) ? opts.duration : 0.8;
    const pitchDeg = defaultPitchDeg();
    const headingDeg = defaultHeadingDeg();

    try {{ viewer.trackedEntity = undefined; }} catch (e) {{}}
    try {{
      if (viewer.camera && viewer.camera.cancelFlight) viewer.camera.cancelFlight();
    }} catch (e) {{}}

    let ok = false;
    try {{
      if (duration <= 0.05) {{
        // 首次加载：矩形铺满视口 = 完整中国地图（最稳妥）
        viewer.camera.setView({{ destination: rect }});
        ok = true;
      }} else {{
        // 带动画：包围球 + 俯视角，既框入中国又保持三维倾斜
        const sphere = Cesium.BoundingSphere.fromRectangle3D(rect);
        const offset = new Cesium.HeadingPitchRange(
          DEG2RAD(headingDeg),
          DEG2RAD(pitchDeg),
          0
        );
        viewer.camera.flyToBoundingSphere(sphere, {{
          duration: duration,
          offset: offset,
          complete: function() {{
            try {{ viewer.camera.lookAtTransform(Cesium.Matrix4.IDENTITY); }} catch (e) {{}}
            viewer.scene.requestRender();
          }},
        }});
        ok = true;
      }}
    }} catch (err) {{
      console.warn("[MapCamera] china overview failed, fallback lookAt", err);
      ok = navigateToLocation({{
        longitude: cc.lon,
        latitude: cc.lat,
        height: range,
        pitch: pitchDeg,
        heading: headingDeg,
        duration: 0,
        source: src,
        force: true,
      }});
    }}

    try {{
      console.info("[Cesium Init Camera]", {{
        longitude: cc.lon,
        latitude: cc.lat,
        range: range,
        heading: headingDeg,
        pitch: pitchDeg,
        roll: 0,
        source: src,
        mode: duration <= 0.05 ? "setView(rect)" : "flyToBoundingSphere",
        ok: !!ok,
      }});
    }} catch (e) {{}}
    viewer.scene.requestRender();
    return ok;
  }}

  function isDefaultChinaOverview(c) {{
    if (!c) return true;
    if (CFG.preferCenter) return false;
    const cc = CAM.chinaCenter || {{ lat: 36, lon: 104 }};
    return Math.abs(Number(c.lat) - Number(cc.lat)) < 2.5 &&
      Math.abs(Number(c.lon) - Number(cc.lon)) < 6.0;
  }}

  function applyCameraView(source) {{
    viewer.resize();
    if (viewer.cesiumWidget && viewer.cesiumWidget.container) {{
      viewer.cesiumWidget.container.style.width = "100%";
      viewer.cesiumWidget.container.style.height = "100%";
    }}
    const cv = CFG.chinaView || {{ west: 78, south: 21, east: 128, north: 50 }};
    Cesium.Camera.DEFAULT_VIEW_RECTANGLE = rectFromCfg(cv);
    const base = document.getElementById("cesiumStatus")?.textContent || "底图就绪";

    // 1) 成果图层：用矩形中心+固定高度+固定姿态（不用无 orientation 的 flyToRect）
    if (CFG.flyRectangle && !CFG.preferCenter) {{
      navigateToRectangle(CFG.flyRectangle, {{ duration: 0.9, source: source || "assetRect" }});
      if (CFG.assetName) setStatus(base + " · 已加载 " + CFG.assetName);
      return;
    }}

    // 2) Copilot / 会话 center+height
    if (CFG.center && !isDefaultChinaOverview(CFG.center)) {{
      if (navigateToLocation({{
        longitude: CFG.center.lon,
        latitude: CFG.center.lat,
        height: CFG.height,
        duration: (source === "init") ? 0 : (CFG.duration || 1.0),
        source: source || "initialCenter",
        force: source === "init",
      }})) {{
        const lat = Number(CFG.center.lat).toFixed(2);
        const lon = Number(CFG.center.lon).toFixed(2);
        setStatus(base + " · 已定位 (" + lat + "°N, " + lon + "°E)");
        return;
      }}
    }}

    // 3) 默认中国视角：首次用 setView（duration=0），避免飞行动画未完成时容器尺寸不对导致地球掉到屏幕下方
    navigateToChinaOverview({{
      duration: (source === "init") ? 0 : 0.6,
      source: source || "chinaDefault",
      force: source === "init",
    }});
    setStatus(base + " · 中国视角");
  }}

  // ---- CSTF_MAP_V1 协议：READY / FLY_ACK / LAYER_*（targetOrigin 收紧到父窗口 origin） ----
  let _parentOrigin = "";
  const _cstfLayers = {{}};
  let _aoiPreviewEntity = null;

  function clearLocalAoiPreview() {{
    if (!_aoiPreviewEntity) return;
    try {{ viewer.entities.remove(_aoiPreviewEntity); }} catch (e) {{}}
    _aoiPreviewEntity = null;
    viewer.scene.requestRender();
  }}

  function showLocalAoiPreview(geometry) {{
    clearLocalAoiPreview();
    const ring = geometry && geometry.type === "Polygon" && geometry.coordinates
      ? geometry.coordinates[0]
      : null;
    if (!Array.isArray(ring) || ring.length < 4) return;
    const flat = [];
    ring.forEach(function(coord) {{
      if (Array.isArray(coord) && coord.length >= 2) {{
        flat.push(Number(coord[0]), Number(coord[1]));
      }}
    }});
    if (flat.length < 8) return;
    _aoiPreviewEntity = viewer.entities.add({{
      polygon: {{
        hierarchy: Cesium.Cartesian3.fromDegreesArray(flat),
        material: Cesium.Color.fromBytes(46, 106, 192, 70),
        height: 0,
        outline: true,
        outlineColor: Cesium.Color.fromBytes(127, 178, 255, 245),
        perPositionHeight: false,
      }},
    }});
    viewer.scene.requestRender();
  }}

  function postToParent(msg) {{
    if (!window.parent) return;
    const origin = _parentOrigin || "*";
    try {{
      window.parent.postMessage(msg, origin);
    }} catch (e) {{
      console.warn("[MapProtocol] postMessage failed", e);
    }}
  }}

  function notifyReadyToServer() {{
    try {{
      fetch("./api/map/ready?channel_id=" + encodeURIComponent(CFG.channelId || "default"), {{ method: "GET", cache: "no-store" }}).catch(function() {{}});
    }} catch (e) {{}}
  }}

  function notifyAckToServer(commandId, ok) {{
    try {{
      fetch(
        "./api/map/ack?channel_id=" + encodeURIComponent(CFG.channelId || "default") + "&command_id=" + encodeURIComponent(commandId || "") + "&ok=" + (ok ? "1" : "0"),
        {{ method: "GET", cache: "no-store" }}
      ).catch(function() {{}});
    }} catch (e) {{}}
  }}

  function sendReady() {{
    postToParent({{
      type: "CSTF_MAP_READY",
      version: 1,
      command_id: "ready-" + Date.now(),
      status: "ready",
      viewer_init_count: window.__cstfViewerInitCount || 1,
      ts: Date.now(),
    }});
    notifyReadyToServer();
  }}

  function sendFlyAck(commandId, ok, extra) {{
    const msg = {{
      type: "CSTF_FLY_ACK",
      version: 1,
      command_id: commandId || "",
      ok: !!ok,
      ts: Date.now(),
    }};
    if (extra) Object.assign(msg, extra);
    postToParent(msg);
    notifyAckToServer(commandId, ok);
  }}

  function sendLayerAck(commandId, layerId, ok, error) {{
    postToParent({{
      type: "CSTF_LAYER_ACK",
      version: 1,
      command_id: commandId || "",
      layer_id: layerId || "",
      ok: !!ok,
      error: error || "",
      ts: Date.now(),
    }});
  }}

  function addCstfLayer(payload) {{
    const layerId = payload.layer_id || ("layer-" + Date.now());
    if (!payload.data) {{
      sendLayerAck(payload.command_id, layerId, false, "缺少 GeoJSON data");
      return;
    }}
    return Cesium.GeoJsonDataSource.load(payload.data, {{
      clampToGround: true,
      stroke: hexToCesiumColor(payload.color || "#e41a1c", 0.95),
      fill: hexToCesiumColor(payload.color || "#e41a1c", payload.alpha != null ? payload.alpha : 0.5),
      strokeWidth: 2,
    }}).then(function(ds) {{
      if (String(layerId).indexOf("aoi:") === 0) clearLocalAoiPreview();
      ds.name = payload.name || layerId;
      if (_cstfLayers[layerId]) {{
        viewer.dataSources.remove(_cstfLayers[layerId]);
      }}
      viewer.dataSources.add(ds);
      _cstfLayers[layerId] = ds;
      if (String(layerId).indexOf("aoi:") === 0) setStatus("AOI 已回显");
      sendLayerAck(payload.command_id, layerId, true);
      viewer.scene.requestRender();
    }}).catch(function(err) {{
      console.warn("[MapProtocol] layer add failed", err);
      sendLayerAck(payload.command_id, layerId, false, String((err && err.message) || err));
    }});
  }}

  function removeCstfLayer(payload) {{
    const layerId = payload.layer_id || "";
    if (String(layerId).indexOf("aoi:") === 0) clearLocalAoiPreview();
    const ds = _cstfLayers[layerId];
    if (ds) {{
      viewer.dataSources.remove(ds);
      delete _cstfLayers[layerId];
      sendLayerAck(payload.command_id, layerId, true);
      viewer.scene.requestRender();
      return;
    }}
    sendLayerAck(payload.command_id, layerId, false, "图层不存在: " + layerId);
  }}

  // Streamlit 侧仅改 center/zoom 时通过 postMessage 飞行，避免 iframe 重建
  window.addEventListener("message", function(ev) {{
    if (!ev.origin) return;
    _parentOrigin = ev.origin;  // 记录父窗口精确 origin，回发收紧 targetOrigin
    const data = ev.data;
    if (!data || typeof data !== "object") return;
    const type = data.type;

    if (type === "CSTF_FLY") {{
      const ok = navigateToLocation({{
        longitude: data.lon,
        latitude: data.lat,
        height: data.height,
        pitch: data.pitch != null ? data.pitch : defaultPitchDeg(),
        heading: data.heading != null ? data.heading : defaultHeadingDeg(),
        duration: data.duration != null ? data.duration : 1.0,
        source: data.source || "postMessage",
        force: true,
      }});
      const label = data.label || (
        Number(data.lat).toFixed(2) + "°N, " + Number(data.lon).toFixed(2) + "°E"
      );
      if (ok) setStatus("底图就绪 · 已定位 " + label);
      sendFlyAck(data.command_id, ok, {{ label: label, source: data.source || "postMessage" }});
      return;
    }}

    if (type === "CSTF_LAYER_ADD") {{
      addCstfLayer(data);
      return;
    }}

    if (type === "CSTF_LAYER_REMOVE") {{
      removeCstfLayer(data);
      return;
    }}
  }});

  // 等容器尺寸稳定后再设相机，避免首次渲染高度为 0 导致地球掉到下方
  function scheduleInitialCamera() {{
    let tries = 0;
    const tick = function() {{
      tries += 1;
      viewer.resize();
      const el = document.getElementById("cesiumContainer");
      const h = el ? el.clientHeight : 0;
      const w = el ? el.clientWidth : 0;
      if ((w > 40 && h > 40) || tries > 20) {{
        applyCameraView("init");
        viewer.scene.requestRender();
        sendReady();
        return;
      }}
      requestAnimationFrame(tick);
    }};
    requestAnimationFrame(tick);
    setTimeout(function() {{
      viewer.resize();
      viewer.scene.requestRender();
    }}, 250);
  }}
  scheduleInitialCamera();

  // ---- AOI 绘制工具：点选 / 矩形 / 多边形 / 当前视图 / 清除 ----
  // 绘制结果 → server（POST /api/map/aoi，Streamlit 轮询消费） + postToParent(CSTF_AOI_*)
  const aoi = {{
    mode: "",          // "" | "click" | "rect" | "poly"
    handler: null,
    rectStart: null,
    rectStartScreen: null,
    rectEntity: null,
    polyPts: [],
    polyEntity: null,
  }};

  function aoiSetMode(mode) {{
    aoi.mode = mode;
    document.querySelectorAll("#aoiToolbar button").forEach(function(b) {{
      b.classList.remove("active");
    }});
    const btn = document.getElementById("aoiBtn" + (mode ? mode[0].toUpperCase() + mode.slice(1) : "X"));
    if (btn) btn.classList.add("active");
    if (aoi.handler) {{
      aoi.handler.destroy();
      aoi.handler = null;
    }}
    if (aoi.polyEntity) {{
      viewer.entities.remove(aoi.polyEntity);
      aoi.polyEntity = null;
    }}
    if (aoi.rectEntity) {{
      viewer.entities.remove(aoi.rectEntity);
      aoi.rectEntity = null;
    }}
    aoi.polyPts = [];
    aoi.rectStart = null;
    aoi.rectStartScreen = null;
    try {{
      viewer.scene.screenSpaceCameraController.enableInputs = !mode;
      viewer.scene.canvas.style.cursor = mode ? "crosshair" : "default";
    }} catch (e) {{}}
    if (!mode) return;
    aoi.handler = new Cesium.ScreenSpaceEventHandler(viewer.scene.canvas);
    const pickPos = function(position) {{
      const cart = viewer.camera.pickEllipsoid(position, viewer.scene.globe.ellipsoid);
      if (!cart) return null;
      const c = Cesium.Cartographic.fromCartesian(cart);
      return {{
        lon: Cesium.Math.toDegrees(c.longitude),
        lat: Cesium.Math.toDegrees(c.latitude),
      }};
    }};
    setStatus(
      mode === "click" ? "点选模式：点击地图选择一点" :
      mode === "rect" ? "矩形模式：按住鼠标拖拽框选" :
      "多边形模式：左键加点，右键完成"
    );
    const toRing = function(pts) {{
      const ring = pts.map(function(p) {{ return [p.lon, p.lat]; }});
      if (ring.length > 0) {{
        const first = ring[0];
        const last = ring[ring.length - 1];
        if (first[0] !== last[0] || first[1] !== last[1]) ring.push([first[0], first[1]]);
      }}
      return ring;
    }};
    if (mode === "click") {{
      aoi.handler.setInputAction(function(movement) {{
        const p = pickPos(movement.position);
        if (!p) return;
        aoiSetMode("");
        sendAoi("selected", {{
          type: "Polygon",
          coordinates: [[
            [p.lon - 0.002, p.lat - 0.002],
            [p.lon + 0.002, p.lat - 0.002],
            [p.lon + 0.002, p.lat + 0.002],
            [p.lon - 0.002, p.lat + 0.002],
            [p.lon - 0.002, p.lat - 0.002],
          ]],
        }}, "map_click");
      }}, Cesium.ScreenSpaceEventType.LEFT_CLICK);
    }}
    if (mode === "rect") {{
      aoi.handler.setInputAction(function(movement) {{
        aoi.rectStart = pickPos(movement.position);
        aoi.rectStartScreen = {{
          x: Number(movement.position.x),
          y: Number(movement.position.y),
        }};
      }}, Cesium.ScreenSpaceEventType.LEFT_DOWN);
      aoi.handler.setInputAction(function(movement) {{
        if (!aoi.rectStart) return;
        const p = pickPos(movement.endPosition);
        if (!p) return;
        const west = Math.min(aoi.rectStart.lon, p.lon);
        const east = Math.max(aoi.rectStart.lon, p.lon);
        const south = Math.min(aoi.rectStart.lat, p.lat);
        const north = Math.max(aoi.rectStart.lat, p.lat);
        if (aoi.rectEntity) viewer.entities.remove(aoi.rectEntity);
        aoi.rectEntity = viewer.entities.add({{
          rectangle: {{
            coordinates: Cesium.Rectangle.fromDegrees(west, south, east, north),
            material: Cesium.Color.fromBytes(46, 106, 192, 55),
            height: 0,
            outline: true,
            outlineColor: Cesium.Color.fromBytes(127, 178, 255, 220),
            outlineWidth: 2,
          }},
        }});
        setStatus("矩形绘制中…松开鼠标完成");
        viewer.scene.requestRender();
      }}, Cesium.ScreenSpaceEventType.MOUSE_MOVE);
      aoi.handler.setInputAction(function(movement) {{
        if (!aoi.rectStart) return;
        const p = pickPos(movement.position);
        if (!p) return;
        const startScreen = aoi.rectStartScreen || {{ x: movement.position.x, y: movement.position.y }};
        const dx = Number(movement.position.x) - Number(startScreen.x);
        const dy = Number(movement.position.y) - Number(startScreen.y);
        if (Math.hypot(dx, dy) < 8) {{
          aoi.rectStart = null;
          aoi.rectStartScreen = null;
          if (aoi.rectEntity) {{
            viewer.entities.remove(aoi.rectEntity);
            aoi.rectEntity = null;
          }}
          setStatus("矩形模式：请按住鼠标拖拽框选");
          return;
        }}
        const w = aoi.rectStart.lon, s = aoi.rectStart.lat;
        aoiSetMode("");
        sendAoi("selected", {{
          type: "Polygon",
          coordinates: [[
            [w, s], [p.lon, s], [p.lon, p.lat], [w, p.lat], [w, s],
          ]],
        }}, "map_rectangle");
      }}, Cesium.ScreenSpaceEventType.LEFT_UP);
    }}
    if (mode === "poly") {{
      aoi.handler.setInputAction(function(movement) {{
        const p = pickPos(movement.position);
        if (!p) return;
        aoi.polyPts.push(p);
        if (aoi.polyEntity) viewer.entities.remove(aoi.polyEntity);
        aoi.polyEntity = viewer.entities.add({{
          polyline: {{
            positions: aoi.polyPts.map(function(q) {{
              return Cesium.Cartesian3.fromDegrees(q.lon, q.lat);
            }}),
            width: 2,
            material: Cesium.Color.YELLOW.withAlpha(0.9),
            clampToGround: true,
          }},
        }});
        setStatus("多边形绘制中（" + aoi.polyPts.length + " 点，右键闭合）");
      }}, Cesium.ScreenSpaceEventType.LEFT_CLICK);
      aoi.handler.setInputAction(function() {{
        if (aoi.polyPts.length < 3) {{
          setStatus("多边形至少需要 3 个点");
          aoiSetMode("");
          return;
        }}
        const ring = toRing(aoi.polyPts);
        aoiSetMode("");
        sendAoi("selected", {{ type: "Polygon", coordinates: [ring] }}, "map_polygon");
      }}, Cesium.ScreenSpaceEventType.RIGHT_CLICK);
    }}
  }}

  function sendAoi(kind, geometry, source, label) {{
    const msg = {{ kind: kind, geometry: geometry, source: source || "", label: label || null }};
    if (kind === "cleared") clearLocalAoiPreview();
    else showLocalAoiPreview(geometry);
    setStatus(kind === "cleared" ? "AOI 正在清除…" : "AOI 正在同步…");
    try {{
      const channelQuery = "?channel_id=" + encodeURIComponent(CFG.channelId || "default");
      fetch("./api/map/aoi" + channelQuery, {{
        method: "POST",
        headers: {{ "Content-Type": "application/json" }},
        body: JSON.stringify(msg),
        cache: "no-store",
      }}).then(function(resp) {{
        if (!resp || !resp.ok) throw new Error("HTTP " + (resp && resp.status));
        setStatus(kind === "cleared" ? "AOI 已清除" : "AOI 已选定，已同步");
        return resp;
      }}).catch(function(e) {{
        console.warn("[AOI] server push failed", e);
        setStatus(kind === "cleared" ? "AOI 清除失败" : "AOI 发送失败，请重试");
      }});
    }} catch (e) {{
      console.warn("[AOI] server push failed", e);
      setStatus(kind === "cleared" ? "AOI 清除失败" : "AOI 发送失败，请重试");
    }}
    postToParent({{
      type: kind === "cleared" ? "CSTF_AOI_CLEARED" : "CSTF_AOI_SELECTED",
      version: 1,
      geometry: geometry,
      source: source || "",
      label: label || null,
      ts: Date.now(),
    }});
  }}

  function aoiCurrentView() {{
    try {{
      const rect = viewer.camera.computeViewRectangle();
      if (!rect) {{ setStatus("当前视野不可用（2D/无矩形）"); return; }}
      sendAoi("selected", {{
        type: "Polygon",
        coordinates: [[
          [Cesium.Math.toDegrees(rect.west), Cesium.Math.toDegrees(rect.south)],
          [Cesium.Math.toDegrees(rect.east), Cesium.Math.toDegrees(rect.south)],
          [Cesium.Math.toDegrees(rect.east), Cesium.Math.toDegrees(rect.north)],
          [Cesium.Math.toDegrees(rect.west), Cesium.Math.toDegrees(rect.north)],
          [Cesium.Math.toDegrees(rect.west), Cesium.Math.toDegrees(rect.south)],
        ]],
      }}, "current_view");
    }} catch (e) {{
      console.warn("[AOI] current view failed", e);
      setStatus("当前视野不可用");
    }}
  }}

  document.getElementById("aoiBtnClick").addEventListener("click", function() {{
    aoiSetMode(aoi.mode === "click" ? "" : "click");
  }});
  document.getElementById("aoiBtnRect").addEventListener("click", function() {{
    aoiSetMode(aoi.mode === "rect" ? "" : "rect");
  }});
  document.getElementById("aoiBtnPoly").addEventListener("click", function() {{
    aoiSetMode(aoi.mode === "poly" ? "" : "poly");
  }});
  document.getElementById("aoiBtnView").addEventListener("click", function() {{
    aoiSetMode("");
    aoiCurrentView();
  }});
  document.getElementById("aoiBtnClear").addEventListener("click", function() {{
    aoiSetMode("");
    sendAoi("cleared", null, "ui_clear");
  }});
}})();
</script>
</body>
</html>"""
