import os
import re
import base64
import io
import threading
import uuid
from contextvars import ContextVar
from datetime import date
from typing import Optional, Sequence

# Workaround for Windows OpenMP runtime duplication from mixed scientific deps.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import httpx
from dotenv import load_dotenv
import chromadb
from chromadb.utils import embedding_functions
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage
from langgraph.prebuilt import ToolRuntime, create_react_agent
from llm_backend import BackendUnavailable, LLMBackendConfig, build_chat_model
from agent_context_policy import redact_spatial_metadata, sanitize_external_text

# Respect explicitly injected environment values (CI, remote gateway and
# isolated tests). The ignored local .env remains the fallback for developers.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(_THIS_DIR, ".env"), override=False)
load_dotenv(override=False)


# ==========================================
# 1. 定义 Agent 的工具箱 (Tools)
# ==========================================

# 文献库懒加载：避免「每次打开 Copilot / 任意首轮对话」就拉 Chroma + 下载/加载 BGE（与问题内容无关）
_kb_collection = None
_kb_lock = threading.Lock()

# 联网检索结果必须按当前调用上下文隔离。模块级 list 会让上一轮、其他会话
# 甚至其他用户的来源串入本轮回答。ContextVar 同时兼容同步调用与异步上下文。
_search_trace_var: ContextVar[dict] = ContextVar(
    "cstf_search_trace",
    default={
        "trace_id": "",
        "used": False,
        "profile": "general",
        "urls": (),
        "results": (),
        "failure": "",
    },
)
_search_trace_registry: dict[str, dict] = {}
_search_trace_registry_lock = threading.Lock()
_SEARCH_TRACE_CONFIG_KEY = "cstf_search_trace_id"


def _new_search_trace_id() -> str:
    return uuid.uuid4().hex


def _reset_search_trace() -> str:
    trace_id = _new_search_trace_id()
    _search_trace_var.set(
        {
            "trace_id": trace_id,
            "used": False,
            "profile": "general",
            "urls": (),
            "results": (),
            "failure": "",
        }
    )
    return trace_id


def _set_search_trace(
    *,
    profile: str,
    results: Optional[list] = None,
    failure: str = "",
    trace_id: Optional[str] = None,
) -> None:
    clean_results = tuple(
        {
            "title": str(item.get("title") or "").strip(),
            "url": str(item.get("url") or "").strip(),
        }
        for item in (results or [])
        if str(item.get("url") or "").strip()
    )
    trace = {
        "trace_id": str(trace_id or _search_trace_var.get().get("trace_id") or ""),
        "used": True,
        "profile": profile,
        "urls": tuple(item["url"] for item in clean_results),
        "results": clean_results,
        "failure": str(failure or "").strip(),
    }
    _search_trace_var.set(trace)
    if trace["trace_id"]:
        with _search_trace_registry_lock:
            _search_trace_registry[trace["trace_id"]] = trace


def _current_search_trace() -> dict:
    return dict(_search_trace_var.get())


def _sync_search_trace(trace_id: str) -> None:
    """Import a tool-worker's search trace into the caller's context.

    LangGraph's synchronous ToolNode executes tool calls in a worker pool, so
    ContextVar values set by ``web_search`` are not visible in ``chat_with_vlm``.
    The per-request registry bridges that boundary without reintroducing a
    module-level "last result" shared by all sessions.
    """
    if not trace_id:
        return
    with _search_trace_registry_lock:
        trace = _search_trace_registry.get(trace_id)
    if trace is not None:
        _search_trace_var.set(dict(trace))


def _discard_search_trace(trace_id: str) -> None:
    if not trace_id:
        return
    with _search_trace_registry_lock:
        _search_trace_registry.pop(trace_id, None)


def _search_invoke_config(trace_id: str) -> dict:
    return {"configurable": {_SEARCH_TRACE_CONFIG_KEY: trace_id}}


def _invoke_agent(executor, payload: dict, trace_id: str):
    """Invoke an agent with trace metadata, preserving lightweight test doubles."""
    try:
        return executor.invoke(payload, config=_search_invoke_config(trace_id))
    except TypeError as exc:
        # A few callers provide minimal executor fakes that only accept the
        # payload.  Real LangGraph executors accept ``config``; only fall back
        # for this narrow signature mismatch, never for model/tool errors.
        if "unexpected keyword argument 'config'" not in str(exc):
            raise
        return executor.invoke(payload)


def _get_knowledge_collection():
    """仅在首次调用 search_knowledge_base 时初始化本地向量库与嵌入模型。"""
    global _kb_collection
    with _kb_lock:
        if _kb_collection is not None:
            return _kb_collection
        print("[CSTF-Agent] 首次触发文献检索，正在连接本地 Chroma 并加载 BGE 嵌入模型…")
        from knowledge_store import knowledge_db_path, knowledge_embedding_model

        db_path = knowledge_db_path()
        os.makedirs(db_path, exist_ok=True)
        chroma_client = chromadb.PersistentClient(path=db_path)
        bge_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=knowledge_embedding_model()
        )
        _kb_collection = chroma_client.get_or_create_collection(
            name="remote_sensing_papers",
            embedding_function=bge_ef,
        )
        return _kb_collection


def _normalize_url(url: str) -> str:
    """去掉 URL 末尾常见标点，便于精确比对。"""
    return (url or "").rstrip(".,;:!?)>]}")


def _domain_of(url: str) -> str:
    """提取 URL 的域名（去 www.、小写），用于同域名判定。"""
    m = re.match(r"https?://([^/]+)", url or "")
    if not m:
        return ""
    d = m.group(1).lower()
    return d[4:] if d.startswith("www.") else d


_URL_RE = re.compile(r"https?://[^\s\"'<>)\]\[]+")


# 非论文/非权威来源域名黑名单（检索学术文献时应排除）
_NON_PAPER_DOMAINS = {
    "scholar.google.com", "github.com", "zhihu.com", "csdn.net", "blog.csdn.net",
    "cnblogs.com", "jianshu.com", "baidu.com", "baike.baidu.com", "wikipedia.org",
    "facebook.com", "twitter.com", "x.com", "youtube.com", "linkedin.com",
    "researchgate.net", "semanticscholar.org", "arxiv.org", "researcher.life",
    "ijcesen.com", "ijcaonline.org", "frontiersin.org", "sciopen.com",
    "preprints.org", "biorxiv.org", "medrxiv.org", "ssrn.com", "openreview.net",
    "wepub.org", "u.osu.edu", "scirp.org", "hindawi.com", "omicsonline.org",
    "huggingface.co", "patents.google.com",
}
_NON_PAPER_PATH_MARKERS = (
    "/commit/", "/blob/", "/tree/", "/supplement/", ".diff",
)
_GOVERNMENT_QUERY_RE = re.compile(
    r"(政策|法规|条例|规划|通知|办法|意见|政府|自然资源厅|生态环境厅|"
    r"发展改革委|人民政府|gov\.cn|policy|regulation|official|government|"
    r"department|ministry)",
    re.IGNORECASE,
)
_ACADEMIC_QUERY_RE = re.compile(
    r"(论文|文献|研究|期刊|会议|综述|语义分割|深度学习|遥感|"
    r"paper|literature|journal|conference|review|semantic\s+segmentation|remote\s+sensing)",
    re.IGNORECASE,
)


def _looks_like_paper(url: str, title: str = "") -> bool:
    """判断一条检索结果是否像真实论文/期刊文章（而非学者主页、代码库、博客等）。"""
    dom = _domain_of(url)
    # 后缀匹配：zhuanlan.zhihu.com 这类子域名也能命中 zhihu.com
    for bad in _NON_PAPER_DOMAINS:
        if dom == bad or dom.endswith("." + bad):
            return False
    if "citations?user=" in (url or ""):
        return False
    # 个人主页常见路径特征
    for marker in ("/publications", "/citations", "/profile", "/author", "personal"):
        if marker in (url or ""):
            return False
    lowered_url = (url or "").lower()
    if any(marker in lowered_url for marker in _NON_PAPER_PATH_MARKERS):
        return False
    lowered_title = (title or "").strip().lower()
    if not lowered_title or lowered_title.startswith("http://") or lowered_title.startswith("https://"):
        return False
    return True


def _search_profile(query: str) -> str:
    """Classify a search so policy queries are not forced through paper filters."""
    text = str(query or "")
    if _GOVERNMENT_QUERY_RE.search(text):
        return "government"
    if _ACADEMIC_QUERY_RE.search(text):
        return "academic"
    return "general"


def _government_domains_for_query(query: str) -> list[str]:
    text = str(query or "")
    if "浙江" in text:
        return ["zj.gov.cn"]
    return ["gov.cn"]


def _is_government_source(url: str, query: str) -> bool:
    domain = _domain_of(url)
    allowed = _government_domains_for_query(query)
    return any(domain == item or domain.endswith("." + item) for item in allowed)


def _dedupe_search_results(results: list) -> list:
    deduped = []
    seen = set()
    for item in results or []:
        if not isinstance(item, dict):
            continue
        url = _normalize_url(str(item.get("url") or "").strip())
        content = str(item.get("content") or "").strip()
        if not url or not content or url in seen:
            continue
        seen.add(url)
        copied = dict(item)
        copied["url"] = url
        deduped.append(copied)
    return deduped


def _flag_unverified_urls(reply: str, verified_urls: list) -> str:
    """后处理校验：把回答中不在真实检索结果里的 URL 标记为「待核实」。

    - 精确匹配真实 URL → 保留不动
    - 域名命中真实 URL 但链接细节不同 → 标「链接待核实」
    - 域名都不命中 → 标「来源待核实」
    - 含 [SYSTEM_COMMAND_JSON] 命令块的回复不做校验（那是系统指令，不是文献引用）
    """
    if not reply or "[SYSTEM_COMMAND_JSON]" in reply:
        return reply
    verified = {_normalize_url(u) for u in verified_urls if u}
    verified_domains = {_domain_of(u) for u in verified_urls if u}

    def _cb(m):
        raw = m.group(0)
        if _normalize_url(raw) in verified:
            return raw
        dom = _domain_of(raw)
        if dom and dom in verified_domains:
            return raw + " ⚠️[链接待核实]"
        return raw + " ⚠️[来源待核实]"

    return _URL_RE.sub(_cb, reply)


_REF_SECTION_HEADING_RE = re.compile(
    r"(?im)^\s*(?:#{1,6}\s*)?(?:\*{1,2})?"
    r"(?:参考来源|参考文献|references)"
    r"(?:\*{1,2})?\s*(?:[：:].*)?$"
)


def _build_verified_reference_list(results: list) -> str:
    """根据真实检索结果，程序化生成参考来源清单（标题 + URL 一一对应，绝不编造）。"""
    if not results:
        return ""
    entries = []
    for i, item in enumerate(results, 1):
        title = (item.get("title") or "").strip()
        url = (item.get("url") or "").strip()
        entries.append(f"[{i}] {title}" + (f" — {url}" if url else ""))
    header = "---\n\n**参考来源**（由系统据本次检索真实结果生成，编号与正文 [n] 对应）：\n"
    return header + "\n\n".join(entries)


def _strip_llm_reference_section(reply: str) -> str:
    """Strip only a trailing LLM-generated reference section.

    A user may legitimately ask the answer to begin with the literal text
    ``参考文献``.  The old substring search cut the entire answer at index 0.
    We now require a line-level heading after substantive body text, with a
    numbered item or URL following it.
    """
    if not reply:
        return reply
    matches = list(_REF_SECTION_HEADING_RE.finditer(reply))
    for match in reversed(matches):
        before = reply[: match.start()].strip()
        after = reply[match.end() :]
        if len(before) < 24:
            continue
        if not (re.search(r"(?m)^\s*\[?\d+\]?", after) or _URL_RE.search(after)):
            continue
        return before
    return reply


def _finalize_search_reply(reply: str) -> str:
    """Apply citation post-processing only to a search performed this turn."""
    trace = _current_search_trace()
    text = str(reply or "").strip()
    if not trace.get("used"):
        return text
    results = list(trace.get("results") or [])
    if not results:
        return str(trace.get("failure") or text).strip()
    text = _strip_llm_reference_section(text)
    text = _flag_unverified_urls(text, list(trace.get("urls") or []))
    references = _build_verified_reference_list(results)
    return text + references if references else text


def _web_search_tavily(
    query: str,
    max_results: int = 12,
    search_depth: str = "advanced",
    *,
    trace_id: Optional[str] = None,
) -> str:
    """调用 Tavily Search API 进行联网检索，返回带来源的上下文字符串。

    - search_depth: basic（快）/ advanced（更全更相关，默认）
    - 无 TAVILY_API_KEY 时优雅降级：返回明确提示，不抛异常、不阻断对话。
    """
    profile = _search_profile(query)
    api_key = os.environ.get("TAVILY_API_KEY", "").strip()
    if not api_key:
        message = (
            "【联网搜索未启用】当前未配置 TAVILY_API_KEY 环境变量，无法进行联网检索。"
            "请在 .env 中配置 TAVILY_API_KEY 后重试，或改用本地知识库。"
        )
        _set_search_trace(profile=profile, failure=message, trace_id=trace_id)
        return message

    try:
        payload = {
            "api_key": api_key,
            "query": query,
            "max_results": max_results,
            "search_depth": search_depth,
            "include_raw_content": False,
            "topic": "general",
        }
        if profile == "academic":
            payload["exclude_domains"] = sorted(_NON_PAPER_DOMAINS)
        elif profile == "government":
            payload["include_domains"] = _government_domains_for_query(query)
        resp = httpx.post(
            "https://api.tavily.com/search",
            json=payload,
            timeout=30.0,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        message = f"【联网搜索失败】调用 Tavily 出错：{type(exc).__name__}。请稍后重试，或改用本地知识库。"
        _set_search_trace(profile=profile, failure=message, trace_id=trace_id)
        return message

    results = _dedupe_search_results(data.get("results") or [])
    if profile == "academic":
        results = [
            item
            for item in results
            if _looks_like_paper(item.get("url") or "", item.get("title") or "")
        ]
    elif profile == "government":
        results = [
            item
            for item in results
            if _is_government_source(item.get("url") or "", query)
        ]
    if not results:
        if profile == "government":
            message = (
                "联网搜索未返回符合政府官网约束的可核实结果。"
                "不能据此断言相关政策不存在、尚未发布或仍然有效；请调整关键词后重试。"
            )
        elif profile == "academic":
            message = (
                "联网搜索未返回可用的论文结果（已过滤代码库、模型仓库、专利、博客和补充材料等非论文来源），"
                "请调整关键词后重试。"
            )
        else:
            message = "联网搜索未返回可核实结果，请调整关键词后重试。"
        _set_search_trace(profile=profile, failure=message, trace_id=trace_id)
        return message

    _set_search_trace(profile=profile, results=results, trace_id=trace_id)

    if profile == "government":
        parts = ["【系统从政府官网检索到的可核实资料如下】："]
    elif profile == "academic":
        parts = ["【系统从互联网检索到的学术资料如下，请仅依据这些真实结果作答】："]
    else:
        parts = ["【系统从互联网检索到的可核实资料如下，请注意甄别】："]
    for i, item in enumerate(results, 1):
        title = (item.get("title") or "").strip()
        url = (item.get("url") or "").strip()
        content = (item.get("content") or "").strip()
        score = item.get("score")
        if not content:
            continue
        parts.append(f"结果 {i}：{title}" + (f"（来源: {url}）" if url else ""))
        if score is not None:
            parts.append(f"  相关度: {score:.2f}")
        parts.append(f"  摘要: {content}")
    if profile == "academic":
        parts.append(
            "请基于以上真实论文结果作答，并在正文中用 [n] 标注引用。"
            "优先归纳方法、年份、来源和适用性；不得补写检索结果中没有的作者、期刊、方法或数值。"
            "不要自行生成参考来源清单，系统会按本轮真实结果追加。"
        )
    elif profile == "government":
        parts.append(
            "请只依据以上政府官网结果回答，并在正文中用 [n] 标注引用。"
            "不得把高校环评、商业转载、外省文件或第三方解读当作浙江省政府政策。"
            "如果结果不足，必须明确说无法核实；不得据此断言政策不存在、尚未发布或仍然有效。"
            "不要自行生成参考来源清单，系统会按本轮真实结果追加。"
        )
    else:
        parts.append(
            "请只依据以上真实结果回答，并在正文中用 [n] 标注引用。"
            "信息不足时明确说明无法核实，不得编造最新事实。"
            "不要自行生成参考来源清单，系统会按本轮真实结果追加。"
        )
    return "\n".join(parts)


@tool
def web_search(query: str, runtime: ToolRuntime) -> str:
    """
    【联网搜索工具】
    当用户询问的是「通用知识、时事动态、概念解释」等本地知识库覆盖不到的内容时调用；
    也用于本地文献库未命中时的兜底。参数 `query` 为一句自然语言检索问句（而非零散关键词）。
    检索学术/文献/论文类问题时，优先用英文关键词（如 remote sensing semantic segmentation），
    以获取更多英文期刊与会议论文。
    """
    print(f"\n[Agent 后台动作] 🌐 正在联网搜索：{query}")
    config = getattr(runtime, "config", {}) or {}
    configurable = config.get("configurable", {}) if isinstance(config, dict) else {}
    trace_id = configurable.get(_SEARCH_TRACE_CONFIG_KEY)
    return _web_search_tavily(query, trace_id=trace_id)


@tool
def search_knowledge_base(keywords: str) -> str:
    """
    【学术与政策检索工具】
    当用户询问关于遥感文献（如潮滩分割、注意力机制、反演公式）或政策法规（如《国家湿地保护法》、管控红线）时，必须调用此工具！
    参数 `keywords` 必须是你提炼出的核心检索词，多个词之间用空格隔开。
    例如："多尺度注意力机制 潮滩边缘" 或 "国家湿地保护法 红树林 管控"
    """
    print(f"\n[Agent 后台动作] 🚀 正在调用 ChromaDB 检索文献，关键词：{keywords}")

    try:
        collection = _get_knowledge_collection()
        results = collection.query(query_texts=[keywords], n_results=2)
    except Exception:
        return "本地知识库当前不可用（可能为空、损坏或 embedding 模型未准备好）；未获得可引用资料。"

    docs_batch = results.get("documents") or []
    if not docs_batch or not docs_batch[0]:
        return "本地知识库中未检索到相关文献或法规，请告知用户该领域暂无数据支撑。"

    metas_batch = results.get("metadatas") or [[]]
    retrieved_context = "【系统从本地数据库中检索到的权威资料如下】：\n"
    docs = docs_batch[0]
    metas = metas_batch[0] if metas_batch else []
    for i, doc in enumerate(docs):
        meta = metas[i] if i < len(metas) and isinstance(metas[i], dict) else {}
        source = redact_spatial_metadata(sanitize_external_text(meta.get("source", "未知来源")))[:240]
        content = redact_spatial_metadata(sanitize_external_text(doc))[:1200]
        candidate = f"文献 {i+1} (来源: {source}): {content}\n"
        # Keep retrieval context bounded even when a backend ignores the
        # requested result size or returns unusually long documents.
        if len(retrieved_context) + len(candidate) > 4200:
            break
        retrieved_context += candidate
        
    retrieved_context += "\n请基于以上检索到的真实资料回答用户的问题，必须在回答中引用文献来源，严禁自行编造公式或法规内容！"
    
    return retrieved_context

@tool
def dispatch_system_command(command_json: str) -> str:
    """
    【系统控制主工具 · 凡改侧栏/跑流程/跳地图必用】
    参数 command_json 为合法 JSON 字符串（不要 markdown 代码块）。

    结构：
    {
      "map": {"lat": 30.2, "lon": 121.5, "zoom": 10},          // 可选，仅跳地图
      "sidebar_states": { ... },                                  // 可选，差量更新侧栏
      "pending_action": { "type": "run_pipeline"|"run_m4"|"run_autotune"|"propose_m5"|"run_m5"|"confirm_m5"|"propose_e1"|"run_e1"|"confirm_e1", ... }  // 可选，启动后台
    }

    sidebar_states 全部可用键（未提及则省略，禁止脑补）：
    workflow_tab(潮滩推理|GEE 数据下载), selected_task, run_mode(dl|index), inference_mode(深度学习|指数法),
    prob_th(0.01~0.50), min_cnt(1~10), adaptive_mode, force_rerun,
    root_dir, mask_root, final_root, model_path, shp_path, points_shp, task_aoi_shp,
    m5_enabled, m5_baseline_shp, e1_enabled, e1_data_root, e1_reference, e1_compare_sources[],
    e1_export_maps, e1_export_heatmap,
    m4_roi_path, m4_roi_name, m4_start_date, m4_end_date, m4_export_to(drive|local),
    m4_drive_folder, m4_local_dir, m4_cloud, m4_min_land, m4_max_land, m4_min_pix,
    m4_bands[], m4_scale, m4_gee_proxy, m4_gee_project

    pending_action：
    - run_pipeline: 需要 task（可来自 selected_task 或快照）；prob/cnt 可省略则用侧栏
    - run_m4: m4_params 可含 roi_path/roi_name/start_date/end_date/cloud_limit 等
    - run_autotune: autotune_params.reference_id + objective(iou|f1|iou_f1)；需 adaptive_mode=true
    - propose_m5: 仅生成 M5 变化检测计划（读账本/时期），不启动线程；推荐用 prepare_m5_change_detection
    - run_m5 / confirm_m5: 用户确认后执行独立 M5（confirmed=true）；推荐用 confirm_and_run_m5
    - propose_e1: 仅生成 E1 多源一致性计划；推荐用 prepare_e1_consistency_check
    - run_e1 / confirm_e1: 用户确认后执行独立 E1；推荐用 confirm_and_run_e1

    重要（重型工具确认门闩）：run_pipeline / run_m4 / run_autotune 属于重型操作，
    必须先以 confirmed=true 显式确认（一般只在用户明确说「开始/执行/启动/下载」时给出）；
    未确认时系统不会启动任务，仅提示用户确认。不可绕过。

    口语速查：
    「跑/推理/合成/开始」→ pending_action.run_pipeline（confirmed=true）
    「本地影像推理/潮滩推理/模型跑图」→ local_tidal_flat_inference，确认后 confirm_inference
    「下载/GEE/下影像」→ gee_download_plan（按地图 AOI 下载，计划展示后确认 confirm_gee_download；
       下载完成后不会自动启动推理，如需推理请再发起推理任务）
    「调参/搜最优/AutoTune」→ adaptive_mode=true + run_autotune（confirmed=true）
    「5%/百分之五」→ prob_th=0.05；「频次2/两次」→ min_cnt=2
    「关M5/不要E1」→ m5_enabled/e1_enabled=false
    「变化检测/M5/两期对比/萎缩淤积」→ 先 prepare_m5_change_detection，确认后再 confirm_and_run_m5
    「多源一致性/E1/和师姐比/分歧图」→ 先 prepare_e1_consistency_check，确认后再 confirm_and_run_e1
    """
    cmd = command_json.strip()
    if cmd.startswith("```"):
        cmd = cmd.strip("`").strip()
        if cmd.lower().startswith("json"):
            cmd = cmd[4:].strip()
    return f"[SYSTEM_COMMAND_JSON]\n{cmd}\n[/SYSTEM_COMMAND_JSON]"


@tool
def prepare_m5_change_detection(
    task: Optional[str] = None,
    baseline_task: Optional[str] = None,
) -> str:
    """
    【M5 时空变化检测 · 预检与计划】
    用户要对「已有潮滩成果」做变化检测 / M5 / 两期对比 / 萎缩淤积告警时，必须先调用本工具。
    会生成可验证执行计划，等待用户确认；不会立刻跑推理流水线。
    task 可省略（用侧栏当前任务）；baseline_task 可省略（自动选最近更早同区域时期）。
    """
    import json as _json

    action: dict = {"type": "propose_m5"}
    if task and str(task).strip():
        action["task"] = str(task).strip()
    if baseline_task and str(baseline_task).strip():
        action["baseline_task"] = str(baseline_task).strip()
    payload = {"pending_action": action, "sidebar_states": {"m5_enabled": True}}
    if task and str(task).strip():
        payload["sidebar_states"]["selected_task"] = str(task).strip()
    return (
        "[SYSTEM_COMMAND_JSON]\n"
        + _json.dumps(payload, ensure_ascii=False)
        + "\n[/SYSTEM_COMMAND_JSON]"
    )


@tool
def confirm_and_run_m5(task: Optional[str] = None) -> str:
    """
    【M5 确认执行】
    仅在用户已明确确认执行计划后调用（如「确认」「开始执行」）。
    将真实调用现有 M5 引擎；禁止在未确认时调用。
    """
    import json as _json

    action: dict = {"type": "run_m5", "confirmed": True}
    if task and str(task).strip():
        action["task"] = str(task).strip()
    payload = {"pending_action": action}
    return (
        "[SYSTEM_COMMAND_JSON]\n"
        + _json.dumps(payload, ensure_ascii=False)
        + "\n[/SYSTEM_COMMAND_JSON]"
    )


@tool
def prepare_e1_consistency_check(
    task: Optional[str] = None,
    reference: Optional[str] = None,
) -> str:
    """
    【E1 多源一致性诊断 · 预检与计划】
    用户要对「已有潮滩成果」做多源一致性 / E1 / 和师姐对比 / 分歧图时，必须先调用本工具。
    生成可验证计划并等待确认；不跑推理、不下载 GEE。
    task / reference 可省略（用侧栏当前任务与 ui_e1_reference）。
    """
    import json as _json

    action: dict = {"type": "propose_e1"}
    if task and str(task).strip():
        action["task"] = str(task).strip()
    sb: dict = {"e1_enabled": True}
    if task and str(task).strip():
        sb["selected_task"] = str(task).strip()
    if reference and str(reference).strip():
        sb["e1_reference"] = str(reference).strip()
        action["reference"] = str(reference).strip()
    payload = {"pending_action": action, "sidebar_states": sb}
    return (
        "[SYSTEM_COMMAND_JSON]\n"
        + _json.dumps(payload, ensure_ascii=False)
        + "\n[/SYSTEM_COMMAND_JSON]"
    )


@tool
def confirm_and_run_e1(task: Optional[str] = None) -> str:
    """
    【E1 确认执行】
    仅在用户已明确确认 E1 计划后调用。真实调用 e1_engine；禁止未确认执行。
    """
    import json as _json

    action: dict = {"type": "run_e1", "confirmed": True}
    if task and str(task).strip():
        action["task"] = str(task).strip()
    payload = {"pending_action": action}
    return (
        "[SYSTEM_COMMAND_JSON]\n"
        + _json.dumps(payload, ensure_ascii=False)
        + "\n[/SYSTEM_COMMAND_JSON]"
    )


@tool
def analyze_workflow(
    target_year: Optional[int] = None,
    baseline_year: Optional[int] = None,
    need_e1: Optional[bool] = None,
    need_m5: Optional[bool] = None,
    skip_e1: Optional[bool] = None,
    skip_m5: Optional[bool] = None,
    region: Optional[str] = None,
    task: Optional[str] = None,
    goal: Optional[str] = None,
) -> str:
    """
    【端到端潮滩分析 Workflow · 生成执行计划（必须先调用！）】
    当用户要求「分析当前 AOI 的 XXXX 年潮滩 / 和 XXXX 年比较变化 / 评价精度 / 生成报告」时，
    必须先调用本工具生成确定性执行计划（GEE→本地推理→E1 精度评价→M5 变化检测→PDF 报告），
    展示给用户确认。未确认前绝不执行任何下载/推理。

    - target_year: 分析年份（如 2024）。省略则用侧栏默认（ui_workflow_target_year）。
    - baseline_year: 对比基线年份（如 2022）。省略则用侧栏默认；用户说「和XX年比较」时必填。
    - need_e1: 用户明确要求「评价精度/和真值对比/师姐」→ True（必做，缺真值则阻塞）。
      用户说「不要精度评价/跳过E1」→ False。省略 → 有真值才做，否则自动跳过。
    - need_m5: 用户明确要求「变化检测/M5/萎缩淤积」→ True（必做，缺基线则阻塞）。
      用户说「不要变化检测/跳过M5」→ False。省略 → 有基线才做，否则自动跳过。
    - skip_e1 / skip_m5: 等价于 need_e1/need_m5=False。
    - region: 区域标识（如 quanzhou）；省略则从当前 AOI 推导。
    - task: 可省略（用侧栏当前任务 / AOI 自动命名）。
    - goal: 可省略（由系统按年份/基线/意图自动生成）。
    """
    import json as _json

    action: dict = {"type": "propose_workflow"}
    if target_year is not None:
        action["target_year"] = int(target_year)
    if baseline_year is not None:
        action["baseline_year"] = int(baseline_year)
    if need_e1 is not None:
        action["need_e1"] = bool(need_e1)
    if need_m5 is not None:
        action["need_m5"] = bool(need_m5)
    if skip_e1 is True:
        action["skip_e1"] = True
    if skip_m5 is True:
        action["skip_m5"] = True
    if region and str(region).strip():
        action["region"] = str(region).strip()
    if task and str(task).strip():
        action["task"] = str(task).strip()
    if goal and str(goal).strip():
        action["goal"] = str(goal).strip()
    payload = {"pending_action": action}
    return (
        "[SYSTEM_COMMAND_JSON]\n"
        + _json.dumps(payload, ensure_ascii=False)
        + "\n[/SYSTEM_COMMAND_JSON]"
    )


@tool
def confirm_workflow(workflow_id: Optional[str] = None) -> str:
    """
    【端到端潮滩分析 Workflow · 确认执行】
    仅在用户已明确确认 Workflow 计划后调用（用户说「确认/开始/执行/就这么办」）。
    真实按依赖顺序调用既有闭环（GEE→推理→E1/M5→PDF）；禁止未确认执行。
    workflow_id 可省略（用当前待确认计划）。
    """
    import json as _json

    action: dict = {"type": "confirm_workflow"}
    if workflow_id and str(workflow_id).strip():
        action["workflow_id"] = str(workflow_id).strip()
    payload = {"pending_action": action}
    return (
        "[SYSTEM_COMMAND_JSON]\n"
        + _json.dumps(payload, ensure_ascii=False)
        + "\n[/SYSTEM_COMMAND_JSON]"
    )


@tool
def trigger_spatial_analysis(
    task_node: str,
    prob_th: float,
    min_cnt: int,
    run_mode: str = "dl",
    m5_enabled: Optional[bool] = None,
    e1_enabled: Optional[bool] = None,
) -> str:
    """
    【兼容工具 · 跑潮滩推理】用户明确要求跑模型/推理时调用。
    run_mode: dl=深度学习, index=指数法。prob/min_cnt 必须来自用户原话，禁止编造。
    推荐改用 dispatch_system_command 以同时调整 M5/E1。
    """
    import json as _json

    payload = {
        "sidebar_states": {
            "selected_task": task_node,
            "prob_th": prob_th,
            "min_cnt": int(min_cnt),
            "run_mode": run_mode,
        },
        "pending_action": {"type": "run_pipeline", "task": task_node},
    }
    if m5_enabled is not None:
        payload["sidebar_states"]["m5_enabled"] = m5_enabled
    if e1_enabled is not None:
        payload["sidebar_states"]["e1_enabled"] = e1_enabled
    return f"[SYSTEM_COMMAND_JSON]\n{_json.dumps(payload, ensure_ascii=False)}\n[/SYSTEM_COMMAND_JSON]"


@tool
def change_map_view(
    location_name: str,
    lat: float,
    lon: float,
    zoom: int,
    preset: Optional[str] = None,
    label: Optional[str] = None,
) -> str:
    """
    地图视角跳转。用户表达查看/定位/跳到某地时必须立即调用。
    可与 dispatch_system_command 合并；单独跳转时用本工具。
    preset: 可选，地名预设（如 杭州湾/乐清湾/中国），用于高度档位与展示。
    label: 可选，状态栏展示名（默认取 location_name）。
    """
    import json as _json

    payload = {"map": {"lat": lat, "lon": lon, "zoom": int(zoom)}}
    if preset:
        payload["map"]["preset"] = str(preset)
    if label:
        payload["map"]["label"] = str(label)
    elif location_name:
        payload["map"]["label"] = str(location_name)
    return f"[SYSTEM_COMMAND_JSON]\n{_json.dumps(payload, ensure_ascii=False)}\n[/SYSTEM_COMMAND_JSON]"


@tool
def assist_gee_download(
    region_name: str,
    year: int,
    cloud_limit: Optional[int] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    run_now: bool = False,
) -> str:
    """
    GEE Sentinel-2 下载（M4）。用户要下载遥感数据时调用。
    若用户说「启动下载/开始下」，设 run_now=true 并在 pending_action 中 type=run_m4。
    """
    import json as _json

    sd = start_date or f"{int(year)}-01-01"
    ed = end_date or f"{int(year)}-01-31"
    payload: dict = {
        "sidebar_states": {
            "workflow_tab": "GEE 数据下载",
            "m4_roi_name": region_name,
            "m4_start_date": sd,
            "m4_end_date": ed,
        },
    }
    if cloud_limit is not None:
        payload["sidebar_states"]["m4_cloud"] = int(cloud_limit)
    if run_now:
        payload["pending_action"] = {
            "type": "run_m4",
            "confirmed": True,  # run_now=true 即用户明确要求启动，满足重型工具确认门闩
            "task": region_name,
            "m4_params": {
                "roi_name": region_name,
                "start_date": sd,
                "end_date": ed,
                "cloud_limit": cloud_limit,
            },
        }
    return f"[SYSTEM_COMMAND_JSON]\n{_json.dumps(payload, ensure_ascii=False)}\n[/SYSTEM_COMMAND_JSON]"


@tool
def local_tidal_flat_inference(
    task_id: Optional[str] = None,
    prob_th: Optional[float] = None,
    cnt: Optional[int] = None,
    run_now: bool = False,
) -> str:
    """
    【本地潮滩推理 · 生成执行计划】
    用户要对「本地遥感影像」做潮滩推理（深度学习/CDNet/模型跑图/推理）时调用。
    只接收 task_id / prob_th / cnt / run_now，**不接收任何路径参数**（路径一律使用
    侧栏已配置的合法值或已登记资产，禁止编造路径）。
    - 先调用本工具生成计划（propose）；plan 展示后必须等用户确认，再调用 confirm_inference。
    - run_now=true 表示用户已明确要求启动（等价「开始/执行/跑」），此时会直接进入
      计划→校验→确认（自动确认）→执行闭环；否则只生成计划等待确认。
    prob_th 范围 0.01~0.50；cnt 范围 1~10；越界将由系统校验拒绝。
    """
    import json as _json

    action: dict = {"type": "propose_inference"}
    if task_id and str(task_id).strip():
        action["task"] = str(task_id).strip()
    if prob_th is not None:
        action["prob_th"] = float(prob_th)
    if cnt is not None:
        action["cnt"] = int(cnt)
    if run_now:
        action["run_now"] = True
    payload = {"pending_action": action}
    if task_id and str(task_id).strip():
        payload["sidebar_states"] = {"selected_task": str(task_id).strip()}
    return (
        "[SYSTEM_COMMAND_JSON]\n"
        + _json.dumps(payload, ensure_ascii=False)
        + "\n[/SYSTEM_COMMAND_JSON]"
    )


@tool
def confirm_inference(plan_id: Optional[str] = None) -> str:
    """
    【本地潮滩推理 · 确认执行】
    仅在用户已明确确认推理计划后调用（如「确认」「开始执行」）。
    同一 plan_id 只确认一次；将真实调用现有 pre_engine / post_engine。
    禁止在未确认时调用；禁止编造 plan_id（从计划中获取）。
    """
    import json as _json

    action: dict = {"type": "confirm_inference", "confirmed": True}
    if plan_id and str(plan_id).strip():
        action["plan_id"] = str(plan_id).strip()
    payload = {"pending_action": action}
    return (
        "[SYSTEM_COMMAND_JSON]\n"
        + _json.dumps(payload, ensure_ascii=False)
        + "\n[/SYSTEM_COMMAND_JSON]"
    )


@tool
def gee_download_plan(
    task_id: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    bands: Optional[str] = None,
    cloud_limit: Optional[int] = None,
    export_to: Optional[str] = None,
    run_now: bool = False,
) -> str:
    """
    【GEE 遥感影像下载 · 生成执行计划】
    用户要「下载/获取 GEE/哨兵/COPERNICUS 影像」「根据地图 AOI 取影像数据」时调用。
    使用当前地图绘制的 AOI（无需传 geometry）；也可传 task_id / 起止日期 / 波段等参数。
    - 波段 bands：逗号分隔字符串，如 "B4,B3,B2"（RGB 顺序，默认）；可含 index_bands 由系统追加。
    - 先调用本工具生成计划（propose）；plan 展示后必须等用户确认，再调用 confirm_gee_download。
    - run_now=true 表示用户已明确要求启动，直接进入计划→校验→确认→执行闭环；
      否则只生成计划等待确认（下载完成后**不会自动启动推理**）。
    """
    import json as _json

    action: dict = {"type": "propose_gee"}
    if task_id and str(task_id).strip():
        action["task"] = str(task_id).strip()
    if start_date and str(start_date).strip():
        action["start_date"] = str(start_date).strip()
    if end_date and str(end_date).strip():
        action["end_date"] = str(end_date).strip()
    if bands and str(bands).strip():
        action["bands"] = [b.strip() for b in str(bands).split(",") if b.strip()]
    if cloud_limit is not None:
        action["cloud_limit"] = int(cloud_limit)
    if export_to and str(export_to).strip():
        action["export_to"] = str(export_to).strip()
    if run_now:
        action["run_now"] = True
    payload = {"pending_action": action}
    if task_id and str(task_id).strip():
        payload["sidebar_states"] = {"m4_roi_name": str(task_id).strip()}
    return (
        "[SYSTEM_COMMAND_JSON]\n"
        + _json.dumps(payload, ensure_ascii=False)
        + "\n[/SYSTEM_COMMAND_JSON]"
    )


@tool
def confirm_gee_download(plan_id: Optional[str] = None) -> str:
    """
    【GEE 遥感影像下载 · 确认执行】
    仅在用户已明确确认 GEE 下载计划后调用（如「确认」「开始下载」）。
    同一 plan_id 只确认一次；将真实调用现有 m4_engine 下载（Drive 提交或本地下载）。
    下载完成后**不会自动启动推理**（如需推理请另行发起推理任务）。
    禁止在未确认时调用；禁止编造 plan_id（从计划中获取）。
    """
    import json as _json

    action: dict = {"type": "confirm_gee", "confirmed": True}
    if plan_id and str(plan_id).strip():
        action["plan_id"] = str(plan_id).strip()
    payload = {"pending_action": action}
    return (
        "[SYSTEM_COMMAND_JSON]\n"
        + _json.dumps(payload, ensure_ascii=False)
        + "\n[/SYSTEM_COMMAND_JSON]"
    )



# ==========================================
# 2. 通义千问 API（阿里云百炼 DashScope · OpenAI 兼容模式）
# ==========================================
# 在系统环境变量或 .env 中设置（勿把 Key 写进代码仓库）：
#   DASHSCOPE_API_KEY=sk-你的Key
# 可选：
#   QWEN_OPENAI_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1  （默认即此）
#   QWEN_CHAT_MODEL=qwen-plus          # 纯文本+工具推荐：qwen-plus / qwen-max / qwen-turbo
#   QWEN_CHAT_MODEL=qwen-vl-plus       # 需要上传图片解译时改用 VL 系列（qwen-vl-plus / qwen-vl-max）
#
# 控制台与计费：https://bailian.console.aliyun.com/  → API-KEY
_backend_config = LLMBackendConfig.from_env()
_dash_key = _backend_config.api_key
_qwen_base = _backend_config.base_url
_qwen_model = _backend_config.model
_tiff_mode = os.environ.get("YYNET_TIFF_MODE", "auto").strip().lower()  # legacy local preview mode; external TIFF always PNG
_attach_geo_meta = os.environ.get("YYNET_ATTACH_GEO_META", "0").strip().lower() not in {"0", "false", "no"}
_tiff_auto_png_mb = float(os.environ.get("YYNET_TIFF_AUTO_PNG_MB", "12"))
_vlm_max_side = int(os.environ.get("YYNET_VLM_MAX_SIDE", "2048"))

try:
    # 只构造轻量客户端，不在导入阶段发起网络请求；无 Key 时保持手动工作台可用。
    llm = build_chat_model(_backend_config, require_tools=True)
except BackendUnavailable:
    llm = None

tools = [
    dispatch_system_command,
    local_tidal_flat_inference,
    confirm_inference,
    gee_download_plan,
    confirm_gee_download,
    prepare_m5_change_detection,
    confirm_and_run_m5,
    prepare_e1_consistency_check,
    confirm_and_run_e1,
    analyze_workflow,
    confirm_workflow,
    trigger_spatial_analysis,
    change_map_view,
    assist_gee_download,
    web_search,
]

# ==========================================
# 3. 组装新一代 LangGraph Agent
# ==========================================
system_prompt_base = """你是 CSTF-Copilot，遥感潮滩分析平台的对话控制中枢。
用户不会按固定句式提问；你必须从碎片化、口语化、多意图混杂的句子中还原真实企图，并调用正确工具。

【回复风格 · 必守】
- 禁止在回复正文中原样罗列内部路径（如 <本地路径>）、工具代码（gee_download /
  local_inference / e1_quality / m5_change / pdf_report / run_m4 等）、侧栏快照原文或
  模板占位符；计划/状态展示由系统按真实数据生成，你只需用自然语言概括要点。
- 涉及数字（scene_count / 精度 / 面积）必须来自工具真实返回，禁止编造或复述猜测值。

═══════════════════════════════════════
第一步 · 意图分类（每轮必做，可多标签）
═══════════════════════════════════════
A. 纯问答 / 文献 / 图片解译 → 只回答或 web_search，**禁止** dispatch_system_command
   一律用 web_search 联网检索（文献/法规/公式/通用知识/时事/概念都走它）
B. 只改侧栏、不立即运行 → dispatch_system_command，**不要** pending_action
   例：「概率改成 8%」「把 E1 打开」「切到下载页」「云量设 20」
C. 改侧栏并立即运行 → dispatch_system_command，sidebar_states + pending_action 同轮给出
   例：「用 5% 跑一下浙江」「下载杭州湾 2020 年 1 月影像并开始」
D. 只跳地图 → map 字段（可合并进 dispatch_system_command）
   例：「看看杭州湾」「定位到南流江口」「地图挪到舟山」
E. 承前省略 / 指代 → 结合【侧栏快照】与对话上文补全 task/参数
   例：「那就跑吧」「同样参数再跑一遍」「云量改成 20 再下」
H. **本地潮滩推理可信执行闭环**
   - 「对本地影像跑推理 / 潮滩推理 / 模型跑图」→ **local_tidal_flat_inference**（生成计划）
   - 展示计划后必须等用户确认；用户说「确认/开始执行」→ **confirm_inference(plan_id)**
   - **禁止**编造权重路径/输入目录；路径一律用侧栏合法值；参数越界由系统校验
   - run_now=true 仅当用户已明确说「开始/执行/跑」；否则只生成计划
   - 同一 plan_id 只确认一次；完成后 Copilot 只回复工具真实输出
F. **M5 时空变化检测闭环**
   - 「做变化检测 / 跑 M5 / 两期对比 / 看萎缩淤积」→ **prepare_m5_change_detection**（propose_m5）
   - 展示计划后等用户确认；用户说「确认/开始执行」→ **confirm_and_run_m5**
   - **禁止**用 run_pipeline 冒充独立 M5；**禁止**未确认就 run_m5
   - 必须结合【M5 变化检测账本】判断当期 SHP / 可用基线时期；条件不足时说明 blockers，不要假装已跑完
G. **E1 多源一致性闭环**
   - 「多源一致性 / 跑 E1 / 和师姐比 / 分歧图」→ **prepare_e1_consistency_check**（propose_e1）
   - 用户确认后 → **confirm_and_run_e1**
   - **禁止**用 run_pipeline 冒充独立 E1；**禁止**未确认就 run_e1
   - 结合【E1 账本】检查当期 SHP / data_root / reference；条件不足说明 blockers
I. **端到端潮滩分析 Workflow（GEE→推理→E1/M5→PDF）**
   - 「分析当前 AOI 的 2024 年潮滩」「和 2022 年比较变化」「评价精度/有真值就评」「生成报告」
     这类**多阶段综合请求** → **analyze_workflow**（生成确定性执行计划，展示后等确认）
   - 用户确认后 → **confirm_workflow**
   - 参数映射：分析年份→target_year；比较年份→baseline_year；
     「评价精度/和真值比」→need_e1=true；「不要E1/跳过精度」→skip_e1=true；
     「变化检测/M5/萎缩淤积」→need_m5=true；「不要M5」→skip_m5=true；
     区域（泉州湾→quanzhou）→region
   - **禁止**：把端到端请求拆成零散 run_pipeline/run_m4/run_e1 逐个手动拼装；
     未确认前绝不执行任何下载/推理；不得编造 scene_count/精度数值

═══════════════════════════════════════
第二步 · 工具选择
═══════════════════════════════════════
- **首选** dispatch_system_command：可同时改多个侧栏项 + 跳地图 + 启动流程
- local_tidal_flat_inference / confirm_inference：本地潮滩推理可信执行闭环（先计划后确认）
- prepare_m5_change_detection / confirm_and_run_m5：独立 M5 变化检测闭环
- prepare_e1_consistency_check / confirm_and_run_e1：独立 E1 多源一致性闭环
- analyze_workflow / confirm_workflow：**端到端潮滩分析**（GEE→推理→E1/M5→PDF，先计划后确认）
- change_map_view：仅当地图跳转且无任何侧栏/运行需求时用
- assist_gee_download：用户明确要 GEE 下载时可快捷调用（等价于 dispatch + run_m4）
- trigger_spatial_analysis：仅简单跑推理且无 M5/E1/Tab 变更时用
- web_search：文献/法规/公式/通用知识/时事/概念一律走联网实时检索（带来源 URL）

调用后必须在回复**末尾原样附上**工具返回的 [SYSTEM_COMMAND_JSON]...[/SYSTEM_COMMAND_JSON] 块。

═══════════════════════════════════════
第三步 · 口语 → JSON 映射规范
═══════════════════════════════════════

【任务名】
- 必须与「可用任务目录」列表模糊匹配：24浙江/浙江2024 → 24zhejiang1
- 不在列表中 → 明确告知无法运行，**禁止**编造任务名或强行 pending_action

【推理方式】
- 深度学习/CDNet/模型/神经网络 → run_mode=dl 或 inference_mode=深度学习
- 指数法/mNDWI/ACWI/不用模型 → run_mode=index 或 inference_mode=指数法

【阈值】
- 5%/百分之五/概率0.05 → prob_th=0.05（注意：5% 不是 5.0）
- 频次2/两次/最少2次/cnt=2 → min_cnt=2
- 「参数默认/按侧栏/当前设置」→ **省略** prob_th/min_cnt（前端保留快照值）

【M5 / E1】
- 开/启用/加上/要做 变化检测（作为推理后置）→ m5_enabled=true；关/不要/跳过 → false
- **独立 M5 闭环**（已有成果、只要变化检测）：prepare_m5_change_detection → 等确认 → confirm_and_run_m5
- **独立 E1 闭环**（多源一致性/分歧图）：prepare_e1_consistency_check → 等确认 → confirm_and_run_e1
- 仅改侧栏开 E1（不立刻跑）→ e1_enabled=true
- 师姐2020/参考2020 → e1_reference=师姐_2020（2022/2024/2025 同理）

【GEE 下载 M4】
- 下载/下数据/GEE/哨兵/Sentinel → workflow_tab=GEE数据下载
- 云量20/云小于30 → m4_cloud=20 或 30
- 2020年1月/2020-01 → m4_start_date/m4_end_date
- 「开始下载/启动M4/现在就下」→ pending_action.type=run_m4（confirmed=true）

【AutoTune】
- 自动调参/搜最优阈值/自适应 → adaptive_mode=true
- 「跑 AutoTune/开始调参」→ 另加 pending_action.type=run_autotune（confirmed=true）
- reference_id 从【数据集资产目录】选取，缺则追问；objective: iou | f1 | iou_f1

【路径】
- 用户给出盘符路径 → 写入对应 root_dir/mask_root/final_root/model_path 等键

【是否立即运行 · 关键判别】
含以下动词 → 通常要 pending_action：跑、执行、开始、启动、下载、推理、合成、调参、来一轮
仅含以下 → 通常**不要** pending_action：改成、设为、打开、关闭、切换、调到、看看（仅地图）

【端到端潮滩分析 Workflow】
- 「分析当前 AOI 的 2024 年潮滩，和 2022 年比较变化，有真值就评价精度，生成报告」
  → analyze_workflow(target_year=2024, baseline_year=2022)（need_e1/need_m5 省略=有条件才做）
- 用户明确要「评价精度」→ need_e1=true；「不要精度评价」→ skip_e1=true
- 用户明确要「变化检测/M5」→ need_m5=true；「不要M5」→ skip_m5=true
- 确认后 → confirm_workflow（一次确认，绝不逐个手动拼装）

═══════════════════════════════════════
第四步 · 差量更新铁律
═══════════════════════════════════════
- JSON 中**只写用户本轮明确提到或可从指代推断的字段**
- 未提及的参数：**省略键**或 null，严禁擅自填默认数字
- 缺关键信息无法安全执行时：**先追问**（task、prob、reference_id、ROI 日期等）

═══════════════════════════════════════
第五步 · 典型多意图句式（必须一次工具搞定）
═══════════════════════════════════════
① 「深度学习跑24zhejiang，5%两次，开M5关E1，开始」
   → selected_task, run_mode=dl, prob_th=0.05, min_cnt=2, m5_enabled=true, e1_enabled=false, run_pipeline(confirmed=true)
② 「指数法跑一下，别的按侧栏」→ run_mode=index, run_pipeline(confirmed=true)（prob/cnt 省略）
③ 「切下载，云量15，2020年6月，启动」→ workflow_tab, m4_cloud=15, 日期, run_m4(confirmed=true)
④ 「看看钱塘江然后跑当前任务」→ map + run_pipeline(confirmed=true)（task 取自快照）
⑤ 「E1打开参考2022，先别跑」→ e1_enabled, e1_reference，**无** pending_action

═══════════════════════════════════════
禁止事项
═══════════════════════════════════════
- 只口头说「已定位/已开始」却不调用工具
- 向用户解释 JSON/暗号/协议细节
- 纯地理知识问答时误触发跑图
- 任务不在硬盘列表时假装能跑"""


_agent_executor_lock = threading.Lock()
agent_executor = None
_text_model_lock = threading.Lock()
_text_model = None


def _get_agent_executor():
    """首次真正聊天时构造工具型 Agent；未配置后端时返回 None。"""
    global agent_executor, llm
    if agent_executor is not None:
        return agent_executor
    with _agent_executor_lock:
        if agent_executor is not None:
            return agent_executor
        if llm is None:
            try:
                llm = build_chat_model(_backend_config, require_tools=True)
            except BackendUnavailable:
                return None
        agent_executor = create_react_agent(llm, tools)
        return agent_executor


def _get_text_model():
    """构造不带工具的纯问答后端，允许本地 text-only 模型参与聊天。"""
    global _text_model
    if _text_model is not None:
        return _text_model
    with _text_model_lock:
        if _text_model is None:
            _text_model = build_chat_model(_backend_config, require_tools=False)
    return _text_model


def _percentile_stretch_to_uint8(arr: np.ndarray, valid_mask: np.ndarray = None) -> np.ndarray:
    """Convert an arbitrary numeric array to uint8 using robust percentile stretch."""
    out = np.zeros(arr.shape, dtype=np.uint8)
    for c in range(arr.shape[-1]):
        ch = arr[..., c].astype(np.float32)
        finite = np.isfinite(ch)
        if valid_mask is not None:
            finite = finite & valid_mask
        if not finite.any():
            continue
        lo = np.percentile(ch[finite], 2)
        hi = np.percentile(ch[finite], 98)
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            lo = np.min(ch[finite])
            hi = np.max(ch[finite])
        if hi <= lo:
            out[..., c] = 0
            continue
        norm = (ch - lo) / (hi - lo)
        norm = np.clip(norm, 0.0, 1.0)
        norm = np.where(np.isfinite(norm), norm, 0.0)
        out[..., c] = (norm * 255.0).astype(np.uint8)
    return out


def _is_tiff_path(image_path: str) -> bool:
    return os.path.splitext(image_path)[1].lower() in (".tif", ".tiff")


def _needs_png_in_auto(image_path: str) -> bool:
    if not _is_tiff_path(image_path):
        return False
    try:
        size_mb = os.path.getsize(image_path) / (1024 * 1024)
    except OSError:
        size_mb = 0.0
    return size_mb >= _tiff_auto_png_mb


def _extract_geotiff_meta_text(image_path: str) -> str:
    """Return compact geospatial metadata text for model context."""
    if not _is_tiff_path(image_path):
        return ""
    try:
        import rasterio

        with rasterio.open(image_path) as ds:
            bounds = ds.bounds
            compress = ds.profile.get("compress", "none")
            tiled = bool(ds.profile.get("tiled", False))
            blockx = ds.profile.get("blockxsize")
            blocky = ds.profile.get("blockysize")
            xres, yres = ds.res if ds.res else (None, None)
            finite_ratio = None
            try:
                sample = ds.read(list(range(1, min(ds.count, 3) + 1)))
                finite_ratio = float(np.isfinite(sample).all(axis=0).mean())
            except Exception:
                finite_ratio = None
            return (
                "[GeoTIFF metadata]\n"
                f"- bands: {ds.count}\n"
                f"- size: {ds.width}x{ds.height}\n"
                f"- dtype: {ds.dtypes[0] if ds.dtypes else 'unknown'}\n"
                f"- crs: {ds.crs}\n"
                f"- resolution: x={xres}, y={yres}\n"
                f"- nodata: {ds.nodata}\n"
                f"- bounds: left={bounds.left:.6f}, bottom={bounds.bottom:.6f}, "
                f"right={bounds.right:.6f}, top={bounds.top:.6f}\n"
                f"- compression: {compress}\n"
                f"- tiled: {tiled}\n"
                + (f"- block_size: {blockx}x{blocky}\n" if blockx and blocky else "")
                + (f"- finite_pixel_ratio: {finite_ratio:.4f}\n" if finite_ratio is not None else "")
            )
    except Exception as exc:
        return f"[GeoTIFF metadata unavailable: {sanitize_external_text(exc)[:240]}]"


def _estimate_finite_pixel_ratio(image_path: str, sample_max_side: int = 1024) -> float:
    """Estimate finite pixel ratio (across first up to 3 bands) on a sampled grid."""
    if not _is_tiff_path(image_path):
        return 1.0
    try:
        import rasterio
        from rasterio.enums import Resampling

        with rasterio.open(image_path) as ds:
            bands = list(range(1, min(ds.count, 3) + 1))
            out_h = min(ds.height, sample_max_side)
            out_w = min(ds.width, sample_max_side)
            sample = ds.read(
                bands,
                out_shape=(len(bands), out_h, out_w),
                resampling=Resampling.nearest,
                masked=False,
            )
        return float(np.isfinite(sample).all(axis=0).mean())
    except Exception:
        return 1.0


def _build_image_data_url(image_path: str, force_png_for_tiff: bool = False) -> str:
    """Build a data URL for VLM; TIFF can be sent raw or converted to PNG preview."""
    ext = os.path.splitext(image_path)[1].lower()

    if ext in (".tif", ".tiff") and not force_png_for_tiff:
        with open(image_path, "rb") as img_file:
            img_b64 = base64.b64encode(img_file.read()).decode("utf-8")
        return f"data:image/tiff;base64,{img_b64}"

    if ext in (".tif", ".tiff") and force_png_for_tiff:
        try:
            import rasterio
            from PIL import Image

            with rasterio.open(image_path) as ds:
                data = ds.read(masked=True)
            if data.size == 0:
                raise ValueError("empty raster data")

            if data.shape[0] >= 3:
                rgb = np.moveaxis(data[:3, :, :], 0, -1)
            elif data.shape[0] == 2:
                two = np.moveaxis(data[:2, :, :], 0, -1)
                rgb = np.concatenate([two, two[..., 1:2]], axis=-1)
            else:
                one = data[0]
                rgb = np.repeat(one[:, :, None], 3, axis=2)

            if np.ma.isMaskedArray(rgb):
                rgb_plain = np.ma.filled(rgb, np.nan)
                valid_mask = (~np.ma.getmaskarray(rgb).any(axis=2)) & np.isfinite(rgb_plain).all(axis=2)
            else:
                valid_mask = np.isfinite(rgb).all(axis=2)
                rgb_plain = rgb

            if valid_mask is not None and valid_mask.any():
                valid_ratio = float(valid_mask.mean())
                if valid_ratio < 0.70:
                    ys, xs = np.where(valid_mask)
                    y0, y1 = ys.min(), ys.max()
                    x0, x1 = xs.min(), xs.max()
                    pad = 16
                    y0 = max(0, y0 - pad)
                    x0 = max(0, x0 - pad)
                    y1 = min(rgb_plain.shape[0] - 1, y1 + pad)
                    x1 = min(rgb_plain.shape[1] - 1, x1 + pad)
                    rgb_plain = rgb_plain[y0 : y1 + 1, x0 : x1 + 1, :]
                    valid_mask = valid_mask[y0 : y1 + 1, x0 : x1 + 1]

            rgb_u8 = _percentile_stretch_to_uint8(rgb_plain, valid_mask=valid_mask)
            img = Image.fromarray(rgb_u8, mode="RGB")
            if _vlm_max_side > 0:
                img.thumbnail((_vlm_max_side, _vlm_max_side), Image.Resampling.BILINEAR)
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode('utf-8')}"
        except Exception as conv_err:
            raise RuntimeError(
                f"TIFF conversion failed: {sanitize_external_text(conv_err)[:240]}"
            ) from conv_err

    mime = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
    }.get(ext, "image/jpeg")
    with open(image_path, "rb") as img_file:
        img_b64 = base64.b64encode(img_file.read()).decode("utf-8")
    return f"data:{mime};base64,{img_b64}"


def chat_with_vlm(
    user_input: str,
    chat_history: list,
    image_path: str = None,
    available_tasks: list = None,
    dataset_catalog_text: str = None,
    sidebar_context: str = None,
    capability_summary: str = None,
    allow_spatial_metadata: bool = False,
    # Keep the original main.py contract: image-bearing callers that do not
    # pass a consent flag still send the selected image to the active model.
    # Callers that need a local-only preview can explicitly pass False.
    allow_external_media: bool = True,
    image_paths: Optional[Sequence[str]] = None,
) -> str:
    """处理对话，完美支持多模态视觉能力与物理感知"""
    # Each user turn owns its search/citation trace.  This prevents references
    # from a previous question or another concurrent session leaking here.
    search_trace_id = _reset_search_trace()

    def _model_text(value: object, *, limit: int = 4000) -> str:
        """Normalize every caller-provided text before it enters model context."""
        text = sanitize_external_text(value)
        if not allow_spatial_metadata:
            text = redact_spatial_metadata(text)
        return text[:limit]

    requested_image_paths = []
    if image_path:
        requested_image_paths.append(os.fspath(image_path))
    if isinstance(image_paths, (str, os.PathLike)):
        image_paths = [os.fspath(image_paths)]
    for candidate in image_paths or []:
        if not candidate:
            continue
        normalized = os.fspath(candidate)
        if normalized not in requested_image_paths:
            requested_image_paths.append(normalized)
    if len(requested_image_paths) > 6:
        return "单轮最多支持 6 张图片附件，请减少选择数量后重试。"

    # Explicit False remains available for local-only previews and for callers
    # that intentionally enforce their own media-consent boundary.  The
    # default stays permissive for compatibility with the original main.py.
    if requested_image_paths and not allow_external_media:
        return "当前会话未授权向外部模型发送影像内容；请先在会话设置中勾选影像发送授权。"

    existing_image_paths = [path for path in requested_image_paths if os.path.exists(path)]

    executor = _get_agent_executor()
    if executor is None:
        # 无 tools 的本地后端仍可提供纯问答；只有控制侧栏/跑任务时才要求工具能力。
        if existing_image_paths and "vision" not in _backend_config.capabilities:
            return "当前本地模型只支持文本问答，不支持图片输入；地图和手动任务仍可继续使用。"
        try:
            text_model = _get_text_model()
            text_messages = [SystemMessage(content="你是遥感潮滩分析助手，只回答用户问题，不执行系统命令。")]
            for msg in chat_history[:-1]:
                text_messages.append({
                    "role": msg["role"],
                    "content": _model_text(msg.get("content", "")),
                })
            if existing_image_paths:
                multimodal_content = [{"type": "text", "text": _model_text(user_input)}]
                for current_path in existing_image_paths:
                    image_url = _build_image_data_url(
                        current_path,
                        force_png_for_tiff=_is_tiff_path(current_path),
                    )
                    multimodal_content.append(
                        {"type": "image_url", "image_url": {"url": image_url}}
                    )
                text_messages.append({"role": "user", "content": multimodal_content})
            else:
                text_messages.append({"role": "user", "content": _model_text(user_input)})
            response = text_model.invoke(text_messages)
            return str(getattr(response, "content", response) or "").strip()
        except BackendUnavailable:
            return (
                "聊天后端尚未配置或当前后端不支持工具调用。请在被忽略的 .env 中配置 "
                "DASHSCOPE_API_KEY，或选择已配置的本地后端；本地地图和手动任务仍可继续使用。"
            )
        except Exception as exc:
            return f"纯问答后端调用失败：{type(exc).__name__}。请检查本地模型服务状态。"

    task_list_str = ", ".join(_model_text(task, limit=240) for task in (available_tasks or [])) or "目前硬盘中没有任何数据"
    _d_today = date.today()
    dynamic_prompt = (
        system_prompt_base
        + f"\n\n【系统当前日期】{_d_today.strftime('%Y-%m-%d')}（当前年份 {_d_today.year} 年。"
        "计算“近两年/近三年/近五年”等时间范围、以及构造检索关键词时，必须以此日期为基准，不要使用你记忆中的年份）"
        + f"\n\n【🚨 硬盘可用任务目录（唯一合法 task 名来源）】\n{task_list_str}"
    )
    if sidebar_context and sidebar_context.strip():
        dynamic_prompt += "\n\n" + _model_text(sidebar_context)
    if dataset_catalog_text and dataset_catalog_text.strip():
        dynamic_prompt += "\n\n【数据集资产目录 · AutoTune reference_id 从此选取】\n" + _model_text(dataset_catalog_text)
    if capability_summary and capability_summary.strip():
        dynamic_prompt += (
            "\n\n【能力状态（只读参考，每轮会话快照一次）】\n"
            + _model_text(capability_summary)
            + "\n铁律：已 BLOCKED/UNKNOWN 的能力不得声称执行成功；CONDITIONAL 的能力需先确认前置条件。"
        )

    messages = [SystemMessage(content=dynamic_prompt)]

    for msg in chat_history[:-1]:
        messages.append({
            "role": msg["role"],
            "content": _model_text(msg.get("content", "")),
        })

    attach_geo_meta = False
    if existing_image_paths:
        for attachment_index, current_path in enumerate(existing_image_paths, start=1):
            if not _is_tiff_path(current_path):
                continue
            finite_ratio_est = _estimate_finite_pixel_ratio(current_path)
            if finite_ratio_est <= 0.0:
                return (
                    f"第 {attachment_index} 个 GeoTIFF 在采样检测中未发现任何有效像素（均为 NaN/Inf），"
                    "因此无法进行地物解译。请检查数据源、导出流程或尝试提供未损坏的影像。"
                )

        image_text = _model_text(user_input)
        attach_geo_meta = bool(_attach_geo_meta and allow_spatial_metadata)
        if attach_geo_meta:
            metadata_chunks = []
            for attachment_index, current_path in enumerate(existing_image_paths, start=1):
                meta_text = _extract_geotiff_meta_text(current_path)
                if meta_text:
                    metadata_chunks.append(f"[附件 {attachment_index}]\n{meta_text}")
            if metadata_chunks:
                image_text = f"{_model_text(user_input)}\n\n" + "\n\n".join(metadata_chunks)

        # Never send a raw GeoTIFF to an external model.  TIFF tags can carry
        # CRS/bounds even when textual spatial consent is disabled.  Convert
        # every TIFF to a metadata-free, bounded PNG first; the legacy
        # YYNET_TIFF_MODE/native setting is intentionally ignored at this
        # external-data boundary.
        multimodal_content = [{"type": "text", "text": image_text}]
        for current_path in existing_image_paths:
            image_data_url = _build_image_data_url(
                current_path,
                force_png_for_tiff=_is_tiff_path(current_path),
            )
            multimodal_content.append(
                {"type": "image_url", "image_url": {"url": image_data_url}}
            )
        messages.append({"role": "user", "content": multimodal_content})
    else:
        messages.append({"role": "user", "content": _model_text(user_input)})

    try:
        response = _invoke_agent(executor, {"messages": messages}, search_trace_id)
    except Exception as e:
        if (
            existing_image_paths
            and any(_is_tiff_path(path) for path in existing_image_paths)
            and _tiff_mode == "auto"
            and "image format is illegal" in str(e).lower()
        ):
            image_text = _model_text(user_input)
            if attach_geo_meta:
                metadata_chunks = []
                for attachment_index, current_path in enumerate(existing_image_paths, start=1):
                    meta_text = _extract_geotiff_meta_text(current_path)
                    if meta_text:
                        metadata_chunks.append(f"[附件 {attachment_index}]\n{meta_text}")
                if metadata_chunks:
                    image_text = f"{_model_text(user_input)}\n\n" + "\n\n".join(metadata_chunks)
            retry_content = [{"type": "text", "text": image_text}]
            for current_path in existing_image_paths:
                retry_url = _build_image_data_url(
                    current_path,
                    force_png_for_tiff=_is_tiff_path(current_path),
                )
                retry_content.append(
                    {"type": "image_url", "image_url": {"url": retry_url}}
                )
            retry_messages = messages[:-1] + [
                {
                    "role": "user",
                    "content": retry_content,
                }
            ]
            response = _invoke_agent(
                executor,
                {"messages": retry_messages},
                search_trace_id,
            )
        else:
            raise
    output_messages = response["messages"]
    _sync_search_trace(search_trace_id)
    final_reply = str(output_messages[-1].content or "")

    for msg in output_messages:
        if msg.type == "tool" and (
            "[SYSTEM_COMMAND_JSON]" in str(msg.content)
            or "COMMAND_RUN_PIPELINE" in str(msg.content)
            or "COMMAND_UPDATE_MAP" in str(msg.content)
        ):
            result = _finalize_search_reply(final_reply) + "\n" + str(msg.content)
            _discard_search_trace(search_trace_id)
            return result

    if "COMMAND_SEARCH_KNOWLEDGE_BASE" in final_reply:
        print("\n🚨 [后台监控] 截获到大模型的检索暗号！改为联网搜索...")

        match = re.search(r"COMMAND_SEARCH_KNOWLEDGE_BASE[\|:：\s]*(.*)", final_reply)
        keywords = match.group(1).strip() if match and match.group(1).strip() else user_input
        keywords = keywords.strip("。.,!！")

        retrieved_info = _web_search_tavily(keywords)
        print(f"✅ [后台监控] 联网搜索完成，正在强迫大模型结合检索结果重新作答...\n")

        messages.append({"role": "assistant", "content": final_reply})
        messages.append(
            {
                "role": "user",
                "content": f"系统已自动联网检索，结果如下：\n{retrieved_info}\n请仔细阅读上述资料，直接回答我最初的问题。回答必须专业严谨，且在文末标注来源 URL。严禁在本次回答中再输出 COMMAND 暗号。",
            }
        )

        response_phase2 = _invoke_agent(executor, {"messages": messages}, search_trace_id)
        _sync_search_trace(search_trace_id)
        result = _finalize_search_reply(response_phase2["messages"][-1].content)
        _discard_search_trace(search_trace_id)
        return result

    try:
        return _finalize_search_reply(final_reply)
    finally:
        _discard_search_trace(search_trace_id)
