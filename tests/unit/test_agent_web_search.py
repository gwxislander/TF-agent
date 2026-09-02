"""联网搜索与引用后处理回归测试。"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[2]
TF_AGENT = ROOT / "TF-agent"
if str(TF_AGENT) not in sys.path:
    sys.path.insert(0, str(TF_AGENT))

import agent  # noqa: E402


def test_reference_heading_at_answer_start_is_not_truncated():
    reply = "参考文献\n参考文献用于说明信息来源。它也方便读者复核论据。"
    assert agent._strip_llm_reference_section(reply) == reply


def test_only_trailing_generated_reference_section_is_removed():
    body = "这是正文内容，已经足够长，用于说明潮滩语义分割研究的发展趋势。"
    reply = body + "\n\n参考来源\n[1] 测试论文 — https://example.org/paper"
    assert agent._strip_llm_reference_section(reply) == body


def test_chat_turn_resets_previous_search_references(monkeypatch):
    agent._set_search_trace(
        profile="academic",
        results=[{"title": "上一轮论文", "url": "https://example.org/old"}],
    )

    class _Executor:
        def invoke(self, _payload):
            return {"messages": [SimpleNamespace(type="ai", content="概率阈值已更新为 8%。")]}

    monkeypatch.setattr(agent, "_get_agent_executor", lambda **_kwargs: _Executor())
    reply = agent.chat_with_vlm(
        "把概率阈值改成8%，不要联网，不要提供参考文献。",
        [],
    )

    assert reply == "概率阈值已更新为 8%。"
    assert "上一轮论文" not in reply
    assert "参考来源" not in reply


class _FakeResponse:
    def __init__(self, results):
        self._results = results

    def raise_for_status(self):
        return None

    def json(self):
        return {"results": self._results}


def test_government_search_keeps_only_zhejiang_government_sources(monkeypatch):
    captured = {}
    results = [
        {
            "title": "浙江省自然资源厅通知",
            "url": "https://zrzyt.zj.gov.cn/art/2026/test.html",
            "content": "浙江省海岸带管理相关通知。",
            "score": 0.9,
        },
        {
            "title": "山东政策",
            "url": "http://gb.shandong.gov.cn/policy/test",
            "content": "山东省政策。",
            "score": 0.8,
        },
        {
            "title": "高校环评报告",
            "url": "https://example.edu.cn/report.pdf",
            "content": "项目环评。",
            "score": 0.7,
        },
    ]

    def _post(_url, *, json, timeout):
        captured.update(json)
        assert timeout == 30.0
        return _FakeResponse(results)

    monkeypatch.setenv("TAVILY_API_KEY", "test-key")
    monkeypatch.setattr(agent.httpx, "post", _post)
    agent._reset_search_trace()
    agent._web_search_tavily("2026年浙江省海岸带管理相关政策")
    trace = agent._current_search_trace()

    assert captured["include_domains"] == ["zj.gov.cn"]
    assert captured.get("exclude_domains") is None
    assert trace["profile"] == "government"
    assert trace["urls"] == ("https://zrzyt.zj.gov.cn/art/2026/test.html",)


def test_academic_search_removes_model_repositories_and_patents(monkeypatch):
    results = [
        {
            "title": "Tidal-flat semantic segmentation with deep learning",
            "url": "https://journal.example.org/article/123",
            "content": "A peer-reviewed semantic segmentation study.",
        },
        {
            "title": "Speech model repository",
            "url": "https://huggingface.co/example/speech-model",
            "content": "A speech synthesis model.",
        },
        {
            "title": "一种潮滩变化监测方法及系统",
            "url": "https://patents.google.com/patent/CN123/zh",
            "content": "专利内容。",
        },
    ]

    monkeypatch.setenv("TAVILY_API_KEY", "test-key")
    monkeypatch.setattr(
        agent.httpx,
        "post",
        lambda *_args, **_kwargs: _FakeResponse(results),
    )
    agent._reset_search_trace()
    agent._web_search_tavily("潮滩语义分割深度学习论文")
    trace = agent._current_search_trace()

    assert trace["profile"] == "academic"
    assert trace["urls"] == ("https://journal.example.org/article/123",)
