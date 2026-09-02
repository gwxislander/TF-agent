# -*- coding: utf-8 -*-
"""外部 Agent 上下文最小化策略（默认 minimal）。"""
from __future__ import annotations

import os
import ntpath
import re
from typing import Any

_ABS_PATH_RE = re.compile(
    r"(?:[A-Za-z]:[\\/][^\s,，;；)）]+|\\\\[^\s,，;；)）]+|/(?:Users|home|private|tmp|var|opt|Volumes|mnt|srv|workspace|app|data)/[^\s,，;；)）]+)"
)
_WEB_URL_RE = re.compile(r"(?i)\b(?:https?|ftp)://[^\s<>\"']+")
_SECRET_RE = re.compile(
    r"(?i)(?:api[_-]?key|token|secret|password|authorization)\s*[:=]\s*[^\s,;，；]+"
)
_BARE_PROVIDER_KEY_RE = re.compile(r"(?i)\bsk-[A-Za-z0-9][A-Za-z0-9_-]{8,}")
_URL_CREDENTIAL_RE = re.compile(r"(?i)(https?://)([^/@\s]+):([^/@\s]+)@")
_SPATIAL_FIELD_RE = re.compile(
    r"(?i)\b(?:aoi[_ ]?bbox|bbox|centroid)\s*[:=]\s*(?:\([^\)\n]*\)|\[[^\]\n]*\])"
)
_MAP_CENTER_RE = re.compile(
    r"(?i)(?:地图中心|map_center)\s*[:=]\s*\[[^\]\n]+\](?:\s+zoom\s*=\s*[^\s]+)?"
)
_SPATIAL_LINE_RE = re.compile(
    r"(?im)^(?P<prefix>\s*(?:[-*]\s*)?)(?P<label>bounds|crs|resolution|pixel_size|transform)\s*:\s*[^\n]*$"
)


def describe_local_path(path: Any) -> str:
    """只返回 basename + 存在性，不把绝对路径交给外部模型。"""
    raw = str(path or "").strip()
    if not raw:
        return "未配置"
    try:
        # ntpath handles drive-letter and UNC strings even when tests run on
        # POSIX; os.path.basename alone would echo the whole Windows path.
        # Always ask ntpath for the final component first: it understands both
        # slash styles regardless of the host platform.
        normalized = os.path.normpath(raw)
        basename = ntpath.basename(raw.rstrip("\\/")) or os.path.basename(normalized)
        basename = basename or "未命名"
        basename = sanitize_external_text(basename)[:160]
        return f"{basename}（{'存在' if os.path.exists(raw) else '不存在'}）"
    except (OSError, TypeError, ValueError):
        return "已配置（状态未知）"


def safe_local_path_label(path: Any) -> str:
    """Return a bounded basename/existence label suitable for user/model text."""
    return describe_local_path(path)


def sanitize_external_text(text: Any) -> str:
    """移除绝对路径、密钥/代理凭据和完整本地 geometry 文本。"""
    value = str(text or "")
    value = _SECRET_RE.sub("<redacted>", value)
    value = _BARE_PROVIDER_KEY_RE.sub("<redacted>", value)
    value = _URL_CREDENTIAL_RE.sub(r"\1<redacted>@", value)
    # Web URLs must remain intact. Running the absolute-path expression over
    # the whole string makes the tail of ``https:/`` look like a Windows drive
    # path (``s:/``), and URL paths such as ``/data/...`` look local as well.
    chunks = []
    cursor = 0
    for match in _WEB_URL_RE.finditer(value):
        chunks.append(_ABS_PATH_RE.sub("<local-path>", value[cursor:match.start()]))
        chunks.append(match.group(0))
        cursor = match.end()
    chunks.append(_ABS_PATH_RE.sub("<local-path>", value[cursor:]))
    return "".join(chunks)


def redact_spatial_metadata(text: Any) -> str:
    """Remove known precise AOI/map fields from model-facing or persisted text.

    This deliberately targets the application's labelled representations rather
    than arbitrary decimal pairs, avoiding accidental corruption of ordinary
    prose while covering generated AOI summaries and map context.
    """
    value = str(text or "")
    value = _SPATIAL_FIELD_RE.sub("<spatial-redacted>", value)
    value = _MAP_CENTER_RE.sub("<spatial-redacted>", value)
    # GeoTIFF metadata is emitted as labelled lines (for example
    # ``- bounds: left=..., ...``), so field-value regexes above would leave
    # exact CRS, resolution, and coordinate extents in model context.
    return _SPATIAL_LINE_RE.sub(lambda m: f"{m.group('prefix')}{m.group('label')}: <spatial-redacted>", value)


def safe_error_summary(error: BaseException, *, limit: int = 240) -> str:
    """Return a bounded error string safe for UI/Agent-facing summaries."""
    try:
        value = sanitize_external_text(error)
    except Exception:
        # Error reporting must not create a second failure (for example while
        # an import hook is deliberately simulating a missing optional module).
        # In that extreme case the exception class is still a useful, safe
        # diagnostic and contains no provider/path payload.
        value = type(error).__name__
    value = value.strip()[:limit]
    return f"{type(error).__name__}: {value}" if value else type(error).__name__


def spatial_consent(state: dict) -> bool:
    return bool(state.get("agent_spatial_consent"))


def media_consent(state: dict) -> bool:
    return bool(state.get("agent_external_media_consent"))


def raw_system_command_enabled() -> bool:
    """仅在本地显式开启开发开关时允许聊天框注入系统命令。"""
    return os.environ.get("CSTF_ALLOW_RAW_SYSTEM_COMMAND", "").strip().lower() in {
        "1", "true", "yes", "on"
    }


def raw_system_command_consent(state: dict) -> bool:
    """开发开关之外还要求当前会话勾选一次性授权。"""
    return raw_system_command_enabled() and bool(state.get("agent_raw_system_command_consent"))


__all__ = [
    "describe_local_path", "safe_local_path_label", "media_consent", "raw_system_command_consent",
    "raw_system_command_enabled", "safe_error_summary", "sanitize_external_text",
    "spatial_consent", "redact_spatial_metadata"
]
