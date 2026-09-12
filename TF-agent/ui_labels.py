# -*- coding: utf-8 -*-
"""
用户界面展示层统一中文名称映射（纯展示层，禁止用于任何内部标识/协议）。

本模块只做一件事：把用户直接看到的界面文本，从带技术代号的名称
（如 M5、E1、GEE、Workflow、AutoTune、Asset Registry…）统一成
简单、明确、面向实际功能的中文名称。

内部技术标识（tool/plan/task/workflow/asset/step/command id、JSON schema、
Agent Tool 名等）一律不变；本模块仅提供"展示名"。
未知的键一律回退为原值，绝不抛异常。
"""

# ---------------------------------------------------------------
# 一、核心功能（tool id -> 面向功能的中文名）
# ---------------------------------------------------------------
TOOL_LABELS = {
    # 数据准备
    "gee_download": "获取卫星影像",
    "m4": "获取卫星影像",
    "m4_download": "获取卫星影像",
    "run_m4": "获取卫星影像",
    "run_gee_download": "获取卫星影像",
    "gee_export": "获取卫星影像",
    # 潮滩提取
    "local_inference": "潮滩智能提取",
    "inference": "潮滩智能提取",
    "local_tidal_flat_inference": "潮滩智能提取",
    "run_inference": "潮滩智能提取",
    "run_pipeline": "潮滩智能提取",
    "index_inference": "潮滩智能提取（指数法）",
    # 成果生成
    "post_engine": "潮滩成果生成",
    "post_process": "潮滩成果生成",
    "combine": "潮滩成果生成",
    # 成果分析
    "e1": "潮滩精度评价",
    "e1_quality": "潮滩精度评价",
    "e1_quality_evaluation": "潮滩精度评价",
    "run_e1": "潮滩精度评价",
    "e1_consistency_check": "潮滩精度评价",
    "m5": "潮滩变化分析",
    "m5_change": "潮滩变化分析",
    "m5_change_detection": "潮滩变化分析",
    "run_m5": "潮滩变化分析",
    # 成果输出
    "pdf_report": "成果报告",
    "report": "成果报告",
    "report_engine": "成果报告",
    "asset_report": "成果报告",
    # 综合分析
    "analysis_workflow": "一键潮滩分析",
    "workflow": "一键潮滩分析",
    "run_workflow": "一键潮滩分析",
    "workflow_orchestrator": "一键潮滩分析",
    # 参数优化
    "autotune": "参数自动优化",
    "run_autotune": "参数自动优化",
    "auto_tune": "参数自动优化",
    # 其他界面元素
    "cache_load": "加载已有成果",
    "map_load": "加载到地图",
    "stop_button": "中断任务",
    "verify_inference": "提取结果检查",
    "verify_gee": "影像检查",
    "verify_m5": "变化分析校验",
    "verify_e1": "精度评价校验",
    "register_inference": "保存提取成果",
    "register_gee": "保存影像",
    "register_asset": "保存成果",
}

# ---------------------------------------------------------------
# 二、执行阶段（phase -> 中文名，任务时间线使用）
# ---------------------------------------------------------------
PHASE_LABELS = {
    "PLAN": "生成计划",
    "VALIDATE": "条件检查",
    "CONFIRM": "等待确认",
    "QUEUED": "等待处理",
    "EXECUTE": "正在处理",
    "INFERENCE": "智能提取",
    "POST_PROCESS": "成果生成",
    "GEE_EXPORT": "获取影像",
    "WAIT_REMOTE": "等待影像处理",
    "FETCH_OUTPUT": "获取影像文件",
    "VERIFY": "结果检查",
    "REGISTER": "保存成果",
    "MAP": "加载地图",
    "REPORT": "生成报告",
    "WORKFLOW": "一键分析",
    "AUTOTUNE": "参数优化",
}

# ---------------------------------------------------------------
# 三、任务状态（status -> 中文名，任务时间线使用）
# ---------------------------------------------------------------
STATUS_LABELS = {
    "PENDING": "待处理",
    "WAITING_CONFIRMATION": "等待确认",
    "CONFIRMED": "已确认",
    "READY": "已就绪",
    "QUEUED": "等待处理",
    "RUNNING": "处理中",
    "SUCCEEDED": "已完成",
    "COMPLETED": "已完成",
    "FAILED": "失败",
    "BLOCKED": "暂不可执行",
    "CANCELLED": "已取消",
    "SKIPPED": "已跳过",
    "REUSED": "使用已有成果",
    "WARNING": "有注意事项",
    "COMPLETED_WITH_WARNINGS": "已完成（有注意事项）",
    "PAUSED": "已暂停",
    "VERIFYING": "校验中",
}

# ---------------------------------------------------------------
# 四、资产类型（asset_type -> 中文名）
# ---------------------------------------------------------------
ASSET_LABELS = {
    "dataset": "卫星影像",
    "prediction": "潮滩提取成果",
    "tidal_flat_prediction": "潮滩提取成果",
    "e1": "精度评价结果",
    "e1_evaluation": "精度评价结果",
    "m5": "变化分析结果",
    "m5_change": "变化分析结果",
    "report": "成果报告",
    "pdf_report": "成果报告",
    "artifact": "成果文件",
    "reference_truth": "参考数据",
    "index": "指数法提取成果",
}

# ---------------------------------------------------------------
# 五、地图图层（layer -> 中文名）
# ---------------------------------------------------------------
MAP_LAYER_LABELS = {
    "e1": "精度评价结果",
    "e1_result": "精度评价结果",
    "e1_heatmap": "精度评价分歧热力图",
    "m5_difference": "潮滩变化区域",
    "m5_loss": "萎缩区域",
    "m5_silt": "淤积区域",
    "prediction": "潮滩提取成果",
    "gee_dataset": "卫星影像",
    "reference": "参考数据",
    "baseline": "历史对比成果",
    "current_prediction": "当前提取成果",
    "final_tif": "潮滩栅格成果",
    "final_shp": "潮滩矢量成果",
}

# ---------------------------------------------------------------
# 六、能力状态（capability_id -> 中文名，功能状态面板使用）
# ---------------------------------------------------------------
CAPABILITY_LABELS = {
    "map_navigation": "地图定位",
    "map_layer_display": "地图图层",
    "deep_learning_inference": "潮滩智能提取",
    "gee_download": "获取卫星影像",
    "e1_quality_evaluation": "潮滩精度评价",
    "m5_change_detection": "潮滩变化分析",
    "autotune": "参数自动优化",
    "pdf_report": "成果报告",
    "knowledge_search": "知识库检索",
}

# ---------------------------------------------------------------
# 七、通用术语（用户可见界面用，纯展示）
# ---------------------------------------------------------------
TERM_LABELS = {
    "AOI": "研究区域",
    "ROI": "研究区域",
    "GEE": "卫星影像",
    "M4": "获取卫星影像",
    "M5": "潮滩变化分析",
    "E1": "潮滩精度评价",
    "AutoTune": "参数自动优化",
    "Workflow": "一键潮滩分析",
    "Capability Status": "功能状态",
    "Asset Registry": "成果管理",
    "Dataset Asset": "卫星影像",
    "Prediction Asset": "潮滩提取成果",
    "Task Timeline": "任务进度",
    "Workflow Timeline": "分析进度",
    "Map Navigation": "地图定位",
    "Map Layer": "地图图层",
    "Current View": "当前地图范围",
    "Polygon AOI": "多边形选区",
    "Rectangle AOI": "矩形选区",
    "Reference": "参考数据",
    "Ground Truth": "参考数据",
    "Baseline": "历史对比成果",
    "Final TIF": "潮滩栅格成果",
    "Final SHP": "潮滩矢量成果",
    "Blocker": "无法执行的原因",
    "Warning": "注意事项",
    "Result": "处理结果",
    "Plan": "执行计划",
    "Task": "任务",
    "Local Inference": "潮滩智能提取",
    "Post Processing": "潮滩成果生成",
    "PDF Report": "成果报告",
    "Inference": "潮滩智能提取",
}

# ---------------------------------------------------------------
# 安全取用函数：未知键回退原值，绝不崩溃
# ---------------------------------------------------------------
def get_tool_label(tool_id: str) -> str:
    """工具/步骤展示名；未知值原样返回。"""
    if tool_id is None:
        return ""
    return TOOL_LABELS.get(str(tool_id), str(tool_id))


def get_phase_label(phase: str) -> str:
    """执行阶段中文名；未知值原样返回。"""
    if phase is None:
        return ""
    return PHASE_LABELS.get(str(phase), str(phase))


def get_status_label(status: str) -> str:
    """任务状态中文名；未知值原样返回。"""
    if status is None:
        return ""
    return STATUS_LABELS.get(str(status), str(status))


def get_asset_label(asset_type: str) -> str:
    """资产类型中文名；未知值原样返回。"""
    if asset_type is None:
        return ""
    return ASSET_LABELS.get(str(asset_type), str(asset_type))


def format_asset_caption(asset: dict) -> str:
    """格式化成果列表标题，兼容不同资产类型的可选字段。

    深度学习成果包含概率/次数参数，而 E1、M5、报告和指数成果通常不包含
    这些字段。展示层不应假设所有资产都共享同一套元数据。
    """
    if not isinstance(asset, dict):
        return "成果"

    kind = str(asset.get("method") or asset.get("asset_type") or "").strip().lower()
    kind_labels = {
        "index": "指数",
        "e1": "精度评价",
        "e1_evaluation": "精度评价",
        "m5": "变化分析",
        "m5_change": "变化分析",
        "report": "成果报告",
        "pdf_report": "成果报告",
        "dataset": "卫星影像",
    }
    prefix = kind_labels.get(kind)
    if prefix is None:
        parameters = asset.get("parameters")
        if not isinstance(parameters, dict):
            parameters = {}
        prob = asset.get("prob_threshold", parameters.get("prob_threshold"))
        count = asset.get("min_count", parameters.get("count_threshold"))
        if prob is not None or count is not None:
            prefix = f"P={prob if prob is not None else '—'} C={count if count is not None else '—'}"
        else:
            prefix = "成果"

    created_at = asset.get("created_at") or "未知时间"
    file_size = asset.get("file_size_mb")
    size_text = f"{file_size}MB" if file_size is not None else "大小未知"
    return f"{prefix} · {created_at} · {size_text}"


def get_map_layer_label(layer_id: str) -> str:
    """地图图层展示名；未知值原样返回。"""
    if layer_id is None:
        return ""
    return MAP_LAYER_LABELS.get(str(layer_id), str(layer_id))


def get_capability_label(cap_id: str) -> str:
    """能力展示名；未知值原样返回。"""
    if cap_id is None:
        return ""
    return CAPABILITY_LABELS.get(str(cap_id), str(cap_id))


def get_term_label(term: str) -> str:
    """通用术语展示名；未知值原样返回。"""
    if term is None:
        return ""
    return TERM_LABELS.get(str(term), str(term))
