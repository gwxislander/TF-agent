"""
E1 多源潮滩像元级一致性诊断（封装 jb/E1.py）。

在 YYnet 潮滩合成完成后，将当期 SHP 与开源潮滩产品做像元级 IoU 对比，
输出分歧图、多产品热力图与成因分析（JSON + Markdown）。
"""

from __future__ import annotations

import json
import os
import sys
from typing import Callable, Dict, List, Optional

import geopandas as gpd
from agent_context_policy import safe_error_summary

_JB_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "research", "jb"))
if _JB_DIR not in sys.path:
    sys.path.insert(0, _JB_DIR)

DEFAULT_E1_DATA_ROOT = r"E:\潮滩数据集"
DEFAULT_COMPARE_SOURCES = [
    "DCTF_2020",
    "FCS30_2020",
    "GTF30_2020",
    "CHN_2024",
    "MTWM_2020",
    "TFMC_2020",
    "national_10m_2020",
]
_SKIP_COMPARE = frozenset({"Murray_2014_2016"})


def workspace_for_task(final_root: str, task_name: str) -> str:
    return os.path.join(final_root, task_name, "e1_workspace")


def e1_report_json(workspace_dir: str, roi_name: str) -> str:
    return os.path.join(workspace_dir, "outputs_e1", f"E1_PIXEL_REPORT_{roi_name}.json")


def e1_report_md(workspace_dir: str, roi_name: str) -> str:
    return os.path.join(workspace_dir, "outputs_e1", f"E1_PIXEL_REPORT_{roi_name}.md")


def load_e1_report(workspace_dir: str, roi_name: str) -> Optional[Dict]:
    path = e1_report_json(workspace_dir, roi_name)
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            report = json.load(f)
        if not isinstance(report, dict):
            return None
        # Reports produced before the wrapper started adding provenance do not
        # carry their own disk paths.  Reattach the deterministic paths on
        # load so verification and the UI can recognize these valid reports.
        report.setdefault("report_path", path)
        report.setdefault("report_md_path", e1_report_md(workspace_dir, roi_name))
        report.setdefault("workspace_dir", workspace_dir)
        return report
    except Exception:
        return None


def list_e1_datasets(data_root: str) -> List[str]:
    from E1 import E1_DataCleanerAndDiagnostic

    e1 = E1_DataCleanerAndDiagnostic(workspace_dir=os.path.join(data_root, "_e1_probe"), data_root=data_root)
    return e1.list_datasets()


def resolve_task_roi_path(
    task_aoi_shp: Optional[str],
    task_name: str,
    final_root: str,
    logger: Callable = print,
) -> Optional[str]:
    """从任务分区 AOI 中裁剪当前任务要素，写出临时 ROI shp 供 E1 使用。"""
    if not task_aoi_shp or not str(task_aoi_shp).strip():
        return None
    path = os.path.normpath(os.path.expanduser(str(task_aoi_shp).strip()))
    if not os.path.isfile(path):
        logger(f"[E1] 任务 AOI 不存在，使用全国默认范围: {path}")
        return None

    from evaluation_geo import filter_aoi_for_task

    try:
        aoi = gpd.read_file(path)
        sub = filter_aoi_for_task(aoi, task_name)
        if sub.empty:
            logger(f"[E1] AOI 中未匹配任务 {task_name}，使用全国默认范围。")
            return None
        out_dir = os.path.join(final_root, task_name, "_e1_roi")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{task_name}_aoi.shp")
        sub.to_file(out_path, encoding="utf-8")
        logger(f"[E1] 任务 ROI: {out_path}")
        return out_path
    except Exception as exc:
        logger(f"[E1] 解析任务 ROI 失败，使用全国默认范围: {safe_error_summary(exc)}")
        return None


def run_e1_after_synthesis(
    target_shp: str,
    roi_name: str,
    workspace_dir: str,
    data_root: str = DEFAULT_E1_DATA_ROOT,
    reference: str = "师姐_2020",
    compare_sources: Optional[List[str]] = None,
    roi_path: Optional[str] = None,
    export_disagreement_maps: bool = True,
    export_multi_product_heatmap: bool = True,
    logger: Callable = print,
    stop_callback: Optional[Callable[[], bool]] = None,
) -> Optional[Dict]:
    """
    合成完成后运行 E1 多源对比。失败返回 None，不阻断主流程。
    """
    if not target_shp or not os.path.isfile(target_shp):
        logger("[E1] 当期潮滩 SHP 不存在，跳过多源一致性诊断。")
        return None

    from E1 import E1_DataCleanerAndDiagnostic

    logger(f"\n[E1] 多源一致性诊断 | ROI={roi_name} | reference={reference}")
    os.makedirs(workspace_dir, exist_ok=True)
    e1 = E1_DataCleanerAndDiagnostic(workspace_dir=workspace_dir, data_root=data_root)

    if compare_sources is None:
        compare_sources = [
            n for n in e1.list_datasets()
            if n != reference and n not in _SKIP_COMPARE
        ]
    else:
        compare_sources = [n for n in compare_sources if n != reference and n not in _SKIP_COMPARE]

    if not compare_sources:
        logger("[E1] 无有效对比产品，跳过。")
        return None

    try:
        if stop_callback and stop_callback():
            logger("[E1] 已收到中断请求，未启动精度评价。")
            return None
        result = e1.run_pixel_comparison(
            reference=reference,
            target_path=target_shp,
            target_name="YYnet_Product",
            compare_sources=compare_sources,
            roi_path=roi_path,
            roi_name=roi_name,
            export_rasters=False,
            export_disagreement_maps=export_disagreement_maps,
            export_multi_product_heatmap=export_multi_product_heatmap and len(compare_sources) >= 2,
            stop_callback=stop_callback,
        )
        result["report_path"] = e1_report_json(workspace_dir, roi_name)
        result["report_md_path"] = e1_report_md(workspace_dir, roi_name)
        result["workspace_dir"] = workspace_dir
        logger(f"[E1] 报告已保存: {result['report_path']}")
        return result
    except Exception as exc:
        if stop_callback and stop_callback():
            logger("[E1] 精度评价已中断，未写入成功结果。")
            return None
        logger(f"[E1] 多源一致性诊断失败: {safe_error_summary(exc)}")
        return None
