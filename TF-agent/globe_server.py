"""本地 HTTP 服务：Cesium 页面与瓦片代理。"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Optional

_SERVER_VERSION = 11

_lock = threading.Lock()
_html_by_key: dict[str, str] = {}
_tile_templates: dict[str, str] = {}
_server: Optional[ThreadingHTTPServer] = None
_server_port: Optional[int] = None
_server_version_started: int = 0
_probe_cache: dict[str, tuple[float, bool]] = {}
_PROBE_TTL_SEC = 30.0

# CSTF_MAP_V1 协议状态：iframe 内 JS 通过同源 fetch 上报，供 Streamlit 侧读取
_MAP_STATE: dict = {
    "ready_ts": None,
    "ready_count": 0,
    "navigation_seq": 0,
    "pending_fly": None,
    "ack": None,
    "ack_ts": None,
    "aoi_seq": 0,          # 已处理的 AOI 消息最大序号
    "aoi_pending": [],     # 待消费 AOI 消息 [{seq, kind, geometry, source, label, ts}]
}
_MAP_STATES: dict[str, dict] = {"default": _MAP_STATE}
_CHANNEL_RE = re.compile(r"^[A-Za-z0-9._:-]{1,96}$")


def _channel_key(channel_id: Optional[str]) -> str:
    value = str(channel_id or "default").strip()
    return value if _CHANNEL_RE.fullmatch(value) else "default"


def _new_map_state() -> dict:
    return {
        "ready_ts": None,
        "ready_count": 0,
        "navigation_seq": 0,
        "pending_fly": None,
        "ack": None,
        "ack_ts": None,
        "aoi_seq": 0,
        "aoi_pending": [],
    }


def _map_state(channel_id: Optional[str] = None) -> dict:
    key = _channel_key(channel_id)
    state = _MAP_STATES.get(key)
    if state is None:
        state = _new_map_state()
        _MAP_STATES[key] = state
    return state


def map_protocol_state(channel_id: Optional[str] = None) -> dict:
    """读取地图协议状态（READY 时间 / 最近一次 FLY_ACK / AOI 序号）。"""
    with _lock:
        return dict(_map_state(channel_id))


def reset_map_protocol_state() -> None:
    with _lock:
        _MAP_STATES.clear()
        _MAP_STATES["default"] = _MAP_STATE
        _MAP_STATE.clear()
        _MAP_STATE.update(_new_map_state())


def queue_map_fly(payload: dict, *, channel_id: Optional[str] = None) -> dict:
    """Queue the sole deliverable FLY for one map channel.

    A Streamlit rerun can emit a new navigation while the previous iframe
    retry is still alive.  The per-channel sequence makes that newer command
    authoritative and leaves no second pending camera target to replay.
    """
    if not isinstance(payload, dict) or not payload.get("command_id"):
        raise ValueError("FLY payload requires command_id")
    with _lock:
        state = _map_state(channel_id)
        seq = int(state.get("navigation_seq") or 0) + 1
        queued = dict(payload, navigation_seq=seq)
        state["navigation_seq"] = seq
        state["pending_fly"] = queued
        return dict(queued)


def wait_map_ack(
    command_id: str,
    timeout: float = 1.5,
    *,
    channel_id: Optional[str] = None,
    navigation_seq: Optional[int] = None,
) -> Optional[dict]:
    """Wait for the FLY_ACK for this exact command and navigation sequence."""
    deadline = time.time() + float(timeout)
    while time.time() < deadline:
        with _lock:
            ack = _map_state(channel_id).get("ack")
            if ack and ack.get("command_id") == command_id and (
                navigation_seq is None
                or int(ack.get("navigation_seq") or -1) == int(navigation_seq)
            ):
                return dict(ack)
        time.sleep(0.1)
    return None


def push_aoi_message(msg: dict, *, channel_id: Optional[str] = None) -> int:
    """Cesium iframe → Python：追加 AOI 消息，返回新序号。"""
    with _lock:
        state = _map_state(channel_id)
        state["aoi_seq"] = int(state.get("aoi_seq") or 0) + 1
        seq = state["aoi_seq"]
        pending = list(state.get("aoi_pending") or [])
        pending.append(dict(msg, seq=seq, ts=time.time()))
        state["aoi_pending"] = pending[-50:]  # 上限 50 条
        return seq


def take_aoi_pending(since_seq: int = 0, *, channel_id: Optional[str] = None) -> dict:
    """消费序号 > since_seq 的 AOI 消息。返回 {messages, last_seq}。"""
    with _lock:
        pending = list(_map_state(channel_id).get("aoi_pending") or [])
    fresh = [m for m in pending if int(m.get("seq") or 0) > int(since_seq)]
    last_seq = max((int(m.get("seq") or 0) for m in pending), default=int(since_seq))
    return {"messages": fresh, "last_seq": last_seq}


def html_dir() -> Path:
    d = Path(tempfile.gettempdir()) / "yyglobe_html"
    d.mkdir(parents=True, exist_ok=True)
    return d


def register_tile_template(asset_key: str, upstream_template: str) -> str:
    token = hashlib.md5(asset_key.encode("utf-8", errors="replace")).hexdigest()[:16]
    with _lock:
        _tile_templates[token] = upstream_template
    return token


def public_globe_base_url() -> Optional[str]:
    """
    远程试用时设置环境变量 CSTF_GLOBE_PUBLIC_URL，例如 ngrok 转发的地球服务地址：
      https://xxxx.ngrok-free.app
    未设置则使用本机 127.0.0.1（仅本机浏览器可加载三维地球 iframe）。
    """
    raw = (os.environ.get("CSTF_GLOBE_PUBLIC_URL") or "").strip().rstrip("/")
    return raw or None


def _probe_url_ok(url: str, timeout: float = 2.5) -> bool:
    key = url.rstrip("/")
    now = time.time()
    cached = _probe_cache.get(key)
    if cached and (now - cached[0]) < _PROBE_TTL_SEC:
        return cached[1]
    ok = False
    try:
        with urllib.request.urlopen(f"{key}/health", timeout=timeout) as resp:
            ok = resp.status == 200
    except Exception:
        ok = False
    _probe_cache[key] = (now, ok)
    return ok


def globe_service_base(
    port: int,
    *,
    validate_public: bool = True,
    force_local: bool = False,
) -> str:
    """浏览器加载 iframe/瓦片时使用的地球服务根 URL。"""
    if force_local:
        return _local_globe_base(port)
    pub = public_globe_base_url()
    if pub and validate_public:
        if _probe_url_ok(pub):
            return pub
        # 公网地址失效（ngrok 未启动等）时回退本机，避免 iframe 指向死链
        return _local_globe_base(port)
    if pub:
        return pub
    return _local_globe_base(port)


def _local_globe_base(port: int) -> str:
    return f"http://127.0.0.1:{port}"


def is_local_page_host(host: Optional[str]) -> bool:
    """浏览器 Host 为本机时，iframe 必须用 127.0.0.1，禁止走 ngrok。"""
    if not host:
        return False
    h = str(host).split(",")[0].strip().split(":")[0].lower()
    return h in {"localhost", "127.0.0.1", "::1", "[::1]"}


def globe_public_url_warning(port: int) -> Optional[str]:
    """若配置了不可达的 CSTF_GLOBE_PUBLIC_URL，返回提示文案。"""
    pub = public_globe_base_url()
    if not pub:
        return None
    if _probe_url_ok(pub):
        return None
    return (
        f"环境变量 CSTF_GLOBE_PUBLIC_URL={pub} 当前不可达（ngrok/网关未启动？），"
        f"已自动改用本机 http://127.0.0.1:{port}。"
        "若从其他设备访问 Streamlit，三维地球会空白——请启动 ngrok 并重新设置该变量后重启 Streamlit。"
    )


def overlay_tile_url(port: int, token: str, *, force_local: bool = False) -> str:
    base = globe_service_base(port, force_local=force_local)
    return f"{base}/overlay/{token}/{{z}}/{{x}}/{{y}}.png"


def _refresh_tile_upstream(token: str, template: str) -> None:
    with _lock:
        _tile_templates[token] = template


def _build_test_html() -> str:
    try:
        import os

        try:
            from dotenv import load_dotenv

            load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))
        except ImportError:
            pass

        import globe_engine as ge

        payload = ge.build_globe_payload(
            center=(35.0, 105.0),
            zoom=3,
            ion_token=os.environ.get("CESIUM_ION_TOKEN"),
            show_borders=False,
        )
        return ge.build_cesium_html(payload, full_viewport=True)
    except Exception:
        return "<html><body>globe test failed to build</body></html>"


def _build_minimal_html() -> str:
    import os

    try:
        from dotenv import load_dotenv

        load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))
    except ImportError:
        pass

    ver = "1.128"
    ion = (os.environ.get("CESIUM_ION_TOKEN") or "").strip()
    ion_line = f"Cesium.Ion.defaultAccessToken = {json.dumps(ion)};" if ion else ""
    return f"""<!DOCTYPE html>
<html><head>
<meta charset="utf-8"/>
<link rel="stylesheet" href="https://cesium.com/downloads/cesiumjs/releases/{ver}/Build/Cesium/Widgets/widgets.css"/>
<script src="https://cesium.com/downloads/cesiumjs/releases/{ver}/Build/Cesium/Cesium.js"></script>
<style>html,body,#cesiumContainer{{width:100%;height:100%;margin:0;padding:0;overflow:hidden;}}</style>
</head><body>
<div id="cesiumContainer"></div>
<script>
{ion_line}
const viewer = new Cesium.Viewer("cesiumContainer", {{
  baseLayer: Cesium.ImageryLayer.fromProviderAsync(
    Cesium.TileMapServiceImageryProvider.fromUrl(
      Cesium.buildModuleUrl("Assets/Textures/NaturalEarthII")
    )
  ),
  baseLayerPicker: false,
  animation: false, timeline: false, geocoder: false,
  skyAtmosphere: false,
}});
viewer.scene.globe.enableLighting = false;
viewer.camera.flyTo({{
  destination: Cesium.Rectangle.fromDegrees(73, 18, 135, 54),
  duration: 0,
}});
</script></body></html>"""


def _fetch_upstream_tile(template: str, z: str, x: str, y: str) -> tuple[int, bytes, str]:
    url = (
        template.replace("{z}", z)
        .replace("{x}", x)
        .replace("{y}", y)
        .replace("%7Bz%7D", z)
        .replace("%7Bx%7D", x)
        .replace("%7By%7D", y)
    )
    req = urllib.request.Request(url, headers={"User-Agent": "YYnetGlobe/1.0"})
    with urllib.request.urlopen(req, timeout=15) as resp:
        body = resp.read()
        ctype = resp.headers.get("Content-Type", "image/png")
        return int(resp.status), body, ctype


class _GlobeHandler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args) -> None:
        return

    def _send(
        self,
        code: int,
        body: bytes,
        content_type: str = "text/html; charset=utf-8",
        extra_headers: Optional[dict[str, str]] = None,
    ) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
        self.send_header("Access-Control-Allow-Origin", "*")
        if extra_headers:
            for k, v in extra_headers.items():
                self.send_header(k, v)
        self.end_headers()
        try:
            self.wfile.write(body)
        except (ConnectionResetError, BrokenPipeError, OSError):
            pass

    def do_GET(self) -> None:
        raw = self.path or "/"
        path = raw.split("?", 1)[0]
        query = raw.split("?", 1)[1] if "?" in raw else ""
        from urllib.parse import parse_qs

        query_params = parse_qs(query, keep_blank_values=True)
        channel_id = _channel_key((query_params.get("channel_id") or ["default"])[0])
        cache_key = ""
        if query:
            for part in query.split("&"):
                if part.startswith("v="):
                    cache_key = part[2:]
                    break

        overlay_m = re.match(r"^/overlay/([a-f0-9]{16})/(\d+)/(\d+)/(\d+)\.png$", path)
        if overlay_m:
            token, z, x, y = overlay_m.groups()
            with _lock:
                template = _tile_templates.get(token)
            if not template:
                self._send(404, b"tile template not found", "text/plain; charset=utf-8")
                return
            try:
                status, body, ctype = _fetch_upstream_tile(template, z, x, y)
                if status != 200:
                    self._send(status, body, ctype)
                    return
                self._send(200, body, ctype)
            except urllib.error.HTTPError as e:
                self._send(int(e.code), e.read() if e.fp else b"", "text/plain; charset=utf-8")
            except Exception:
                self._send(502, b"tile fetch failed", "text/plain; charset=utf-8")
            return

        if path in ("/", "/globe"):
            html = ""
            if cache_key:
                disk = html_dir() / f"{cache_key}.html"
                if disk.is_file():
                    html = disk.read_text(encoding="utf-8")
                else:
                    with _lock:
                        html = _html_by_key.get(cache_key, "")
            if not html:
                with _lock:
                    if _html_by_key:
                        html = next(reversed(_html_by_key.values()))
            if not html:
                self._send(
                    200,
                    b"<html><body style='background:#111;color:#ccc;font-family:sans-serif;"
                    b"padding:2rem'>Globe page not ready. Reload Streamlit app.</body></html>",
                )
                return
            self._send(200, html.encode("utf-8"))
            return
        if path == "/test":
            self._send(200, _build_test_html().encode("utf-8"))
            return
        if path == "/minimal":
            self._send(200, _build_minimal_html().encode("utf-8"))
            return
        if path == "/health":
            self._send(200, b"ok", "text/plain; charset=utf-8")
            return
        if path == "/api/map/ready":
            with _lock:
                state = _map_state(channel_id)
                state["ready_ts"] = time.time()
                state["ready_count"] = int(state.get("ready_count") or 0) + 1
            self._send(200, b"ok", "text/plain; charset=utf-8")
            return
        if path == "/api/map/ack":
            q = {}
            for part in query.split("&"):
                if "=" in part:
                    k, v = part.split("=", 1)
                    q[k] = urllib.parse.unquote(v)
            try:
                navigation_seq = int(q.get("navigation_seq") or 0)
            except (TypeError, ValueError):
                navigation_seq = 0
            with _lock:
                state = _map_state(channel_id)
                pending = state.get("pending_fly")
                if (
                    isinstance(pending, dict)
                    and pending.get("command_id") == q.get("command_id", "")
                    and int(pending.get("navigation_seq") or -1) == navigation_seq
                ):
                    now = time.time()
                    state["ack"] = {
                        "command_id": q.get("command_id", ""),
                        "navigation_seq": navigation_seq,
                        "ok": q.get("ok") == "1",
                        "ts": now,
                    }
                    state["ack_ts"] = now
                    state["pending_fly"] = None
            self._send(200, b"ok", "text/plain; charset=utf-8")
            return
        self._send(404, b"not found", "text/plain; charset=utf-8")

    def do_POST(self) -> None:
        raw_path = self.path or "/"
        path = raw_path.split("?", 1)[0]
        from urllib.parse import parse_qs

        query_params = parse_qs(raw_path.split("?", 1)[1], keep_blank_values=True) if "?" in raw_path else {}
        channel_id = _channel_key((query_params.get("channel_id") or ["default"])[0])
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except (TypeError, ValueError):
            length = 0
        raw_body = self.rfile.read(length) if length > 0 else b""
        if path == "/api/map/aoi":
            import json as _json

            try:
                msg = _json.loads(raw_body.decode("utf-8") or "{}")
            except Exception:
                self._send(400, b"invalid json", "text/plain; charset=utf-8")
                return
            if not isinstance(msg, dict):
                self._send(400, b"invalid message", "text/plain; charset=utf-8")
                return
            kind = str(msg.get("kind") or "selected")
            geometry = msg.get("geometry")
            source = str(msg.get("source") or "map_polygon")
            label = msg.get("label")
            push_aoi_message(
                {
                    "kind": kind,
                    "geometry": geometry,
                    "source": source,
                    "label": label if isinstance(label, str) else None,
                },
                channel_id=channel_id,
            )
            self._send(200, b'{"ok":true}', "application/json; charset=utf-8")
            return
        self._send(404, b"not found", "text/plain; charset=utf-8")


def _server_healthy(port: int) -> bool:
    import socket

    try:
        with socket.create_connection(("127.0.0.1", port), timeout=1.5) as sock:
            sock.sendall(b"GET /health HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n")
            data = sock.recv(160)
            return b"200 OK" in data
    except Exception:
        return False


def ensure_running(preferred_port: int = 8765) -> int:
    global _server, _server_port, _server_version_started

    if (
        _server is not None
        and _server_port is not None
        and _server_version_started == _SERVER_VERSION
        and _server_healthy(_server_port)
    ):
        return _server_port

    if _server is not None:
        try:
            _server.shutdown()
        except Exception:
            pass
        _server = None
        _server_port = None

    bind_host = (os.environ.get("CSTF_GLOBE_BIND") or "127.0.0.1").strip() or "127.0.0.1"
    ThreadingHTTPServer.allow_reuse_address = True
    httpd: Optional[ThreadingHTTPServer] = None
    port = preferred_port
    for candidate in range(preferred_port, preferred_port + 20):
        try:
            httpd = ThreadingHTTPServer((bind_host, candidate), _GlobeHandler)
            port = candidate
            break
        except OSError:
            httpd = None
    if httpd is None:
        httpd = ThreadingHTTPServer((bind_host, 0), _GlobeHandler)
        port = int(httpd.server_address[1])

    httpd.daemon_threads = True
    thread = threading.Thread(target=httpd.serve_forever, name="yyglobe-http", daemon=True)
    thread.start()
    _server = httpd
    _server_port = port
    _server_version_started = _SERVER_VERSION
    for _ in range(40):
        if _server_healthy(port):
            return port
        time.sleep(0.05)
    return port


def publish_html(html: str, key: str) -> None:
    with _lock:
        _html_by_key[key] = html
    path = html_dir() / f"{key}.html"
    path.write_text(html, encoding="utf-8")


def globe_url(
    port: int,
    cache_key: str = "",
    bust: Optional[int] = None,
    *,
    force_local: bool = False,
) -> str:
    suffix = f"?v={cache_key}" if cache_key else ""
    if bust is not None and cache_key:
        suffix += f"&b={int(bust)}"
    base = globe_service_base(port, force_local=force_local)
    return f"{base}/globe{suffix}"


def same_globe_origin(url: str, port: int, *, force_local: bool = False) -> bool:
    """判断缓存的 iframe URL 是否仍指向当前可用的地球服务。"""
    if not url:
        return False
    base = globe_service_base(port, force_local=force_local).rstrip("/")
    return str(url).startswith(base + "/") or str(url).rstrip("/") == base


def test_url(port: int) -> str:
    return f"{globe_service_base(port)}/test"


def remote_mode_active() -> bool:
    return public_globe_base_url() is not None
