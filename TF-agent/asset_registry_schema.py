"""结构化校验成果资产注册表。

`assets_registry.json` 已经存在多代历史记录，校验只约束跨版本都稳定的
字段类型，不要求历史文件路径在当前机器存在，也不拒绝未知扩展字段。读取
时可过滤坏记录；写入时则拒绝把坏记录继续落盘。
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Tuple

_OPTIONAL_TEXT_FIELDS = frozenset(
    {
        "task", "method", "file_path", "map_path", "report_path", "report_md_path",
        "loss_shp", "siltation_shp", "baseline_task", "alert_level",
        "reference", "workflow_id", "asset_id", "plan_id", "asset_type",
        "created_at", "status", "code_commit", "device", "model_id",
        "weight_id", "source_asset_id", "input_path", "final_tif", "final_shp",
    }
)
_OPTIONAL_NUMBER_FIELDS = frozenset(
    {"prob_threshold", "min_count", "file_size_mb", "elapsed_seconds"}
)


def validate_entry(key: Any, entry: Any) -> List[str]:
    errors: List[str] = []
    if not isinstance(key, str) or not key.strip():
        errors.append("资产键必须是非空字符串")
    if not isinstance(entry, dict):
        return errors + ["资产记录必须是 JSON object"]
    for field in _OPTIONAL_TEXT_FIELDS:
        if field in entry and entry[field] is not None and not isinstance(entry[field], str):
            errors.append(f"{field} 必须是字符串或 null")
    for field in _OPTIONAL_NUMBER_FIELDS:
        value = entry.get(field)
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            errors.append(f"{field} 必须是数字或 null")
        elif not math.isfinite(float(value)):
            errors.append(f"{field} 必须是有限数字")
    if "parameters" in entry and entry["parameters"] is not None and not isinstance(entry["parameters"], dict):
        errors.append("parameters 必须是 object 或 null")
    return errors


def validate_registry(registry: Any) -> List[str]:
    """返回所有结构错误；未知字段允许向前兼容。"""
    if not isinstance(registry, dict):
        return ["注册表顶层必须是 JSON object"]
    errors: List[str] = []
    for key, entry in registry.items():
        errors.extend(f"{key}: {error}" for error in validate_entry(key, entry))
    return errors


def valid_entries(registry: Any) -> Tuple[Dict[str, Dict[str, Any]], List[str]]:
    """过滤坏记录并返回 `(valid, errors)`，不改变输入对象。"""
    if not isinstance(registry, dict):
        return {}, validate_registry(registry)
    valid: Dict[str, Dict[str, Any]] = {}
    errors: List[str] = []
    for key, entry in registry.items():
        row_errors = validate_entry(key, entry)
        if row_errors:
            errors.extend(f"{key}: {error}" for error in row_errors)
        else:
            valid[key] = dict(entry)
    return valid, errors


def ensure_valid_registry(registry: Any) -> Dict[str, Dict[str, Any]]:
    errors = validate_registry(registry)
    if errors:
        preview = "; ".join(errors[:5])
        suffix = " …" if len(errors) > 5 else ""
        raise ValueError(f"资产注册表结构无效: {preview}{suffix}")
    return registry


__all__ = ["ensure_valid_registry", "valid_entries", "validate_entry", "validate_registry"]
