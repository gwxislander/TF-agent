"""AGENT-005: 媒体发送授权必须在 Agent 核心函数再次 enforced。"""
from __future__ import annotations

import sys
from types import SimpleNamespace
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
TF_AGENT = ROOT / "TF-agent"
if str(TF_AGENT) not in sys.path:
    sys.path.insert(0, str(TF_AGENT))

import agent  # noqa: E402


def test_chat_with_vlm_rejects_image_without_explicit_media_consent(monkeypatch):
    called = False

    def _unexpected_executor(**_kwargs):
        nonlocal called
        called = True
        raise AssertionError("backend must not be reached without media consent")

    monkeypatch.setattr(agent, "_get_agent_executor", _unexpected_executor)
    reply = agent.chat_with_vlm(
        "请解译影像",
        [],
        image_path="/private/tmp/unauthorized.tif",
        allow_external_media=False,
    )
    assert "未授权" in reply
    assert called is False


def test_geotiff_metadata_error_is_sanitized(monkeypatch):
    class _BrokenRasterio:
        def open(self, _path):
            raise RuntimeError("failed /Users/chl/private/secret.tif token=sk-secret")

    monkeypatch.setitem(sys.modules, "rasterio", _BrokenRasterio())
    text = agent._extract_geotiff_meta_text("/Users/chl/private/secret.tif")
    assert "/Users/" not in text
    assert "sk-secret" not in text
    assert "<redacted>" in text


def test_knowledge_tool_output_redacts_local_paths_and_bounds_context(monkeypatch):
    class _Collection:
        def query(self, **_kwargs):
            return {
                "documents": [[
                    "资料 /Users/chl/private/source.pdf bbox=[120,30,121,31] token=sk-knowledge-secret "
                    + ("x" * 5000),
                ]],
                "metadatas": [[{"source": "/Users/chl/private/source.pdf"}]],
            }

    monkeypatch.setattr(agent, "_get_knowledge_collection", lambda: _Collection())
    text = agent.search_knowledge_base.invoke({"keywords": "潮滩"})
    assert "/Users/" not in text
    assert "source.pdf" not in text
    assert "sk-knowledge-secret" not in text
    assert len(text) <= 4300


def test_chat_model_context_redacts_direct_caller_text(monkeypatch):
    captured = {}

    class _Executor:
        def invoke(self, payload):
            captured.update(payload)
            return {"messages": [SimpleNamespace(type="ai", content="收到")]}

    monkeypatch.setattr(agent, "_get_agent_executor", lambda **_kwargs: _Executor())
    reply = agent.chat_with_vlm(
        "请分析 /Users/chl/private/result.tif bbox=[120,30,121,31]",
        [{"role": "user", "content": "历史 /Users/chl/private/old.tif"}],
        available_tasks=["/Users/chl/private/task"],
        dataset_catalog_text="source=/Users/chl/private/catalog.json",
        sidebar_context="map_center=[30,120] root=/Users/chl/private",
    )
    assert reply == "收到"
    blob = str(captured["messages"])
    assert "/Users/" not in blob
    assert "result.tif" not in blob
    assert "bbox=" not in blob


def test_authorized_image_context_also_redacts_user_text(monkeypatch, tmp_path):
    captured = {}

    class _Executor:
        def invoke(self, payload):
            captured.update(payload)
            return {"messages": [SimpleNamespace(type="ai", content="收到")]}

    image = tmp_path / "preview.png"
    image.write_bytes(b"placeholder")
    monkeypatch.setattr(agent, "_get_agent_executor", lambda **_kwargs: _Executor())
    monkeypatch.setattr(agent, "_build_image_data_url", lambda *_args, **_kwargs: "data:image/png;base64,AA==")
    reply = agent.chat_with_vlm(
        "请分析 /Users/chl/private/result.tif token=sk-image-secret",
        [],
        image_path=str(image),
        allow_external_media=True,
    )
    assert reply == "收到"
    blob = str(captured["messages"])
    assert "/Users/" not in blob
    assert "sk-image-secret" not in blob


def test_legacy_default_image_context_is_sent_without_consent_argument(monkeypatch, tmp_path):
    """兼容原 main：旧调用方未传授权参数时仍把图片交给当前 executor。"""
    captured = {}

    class _Executor:
        def invoke(self, payload):
            captured.update(payload)
            return {"messages": [SimpleNamespace(type="ai", content="看到了")]}

    image = tmp_path / "legacy.png"
    image.write_bytes(b"placeholder")
    monkeypatch.setattr(agent, "_get_agent_executor", lambda **_kwargs: _Executor())
    monkeypatch.setattr(agent, "_build_image_data_url", lambda *_args, **_kwargs: "data:image/png;base64,AA==")

    reply = agent.chat_with_vlm("请识别截图", [], image_path=str(image))

    assert reply == "看到了"
    content = captured["messages"][-1]["content"]
    assert [item["type"] for item in content] == ["text", "image_url"]


def test_authorized_multiple_images_share_one_multimodal_user_message(monkeypatch, tmp_path):
    """多附件必须按选择顺序进入同一轮用户消息，而不是只发送第一张。"""
    captured = {}

    class _Executor:
        def invoke(self, payload):
            captured.update(payload)
            return {"messages": [SimpleNamespace(type="ai", content="收到两张图")]}

    first = tmp_path / "first.png"
    second = tmp_path / "second.webp"
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    monkeypatch.setattr(agent, "_get_agent_executor", lambda **_kwargs: _Executor())
    monkeypatch.setattr(
        agent,
        "_build_image_data_url",
        lambda path, **_kwargs: f"data:image/mock;base64,{Path(path).stem}",
    )

    try:
        reply = agent.chat_with_vlm(
            "比较附件",
            [],
            image_paths=[str(first), str(second)],
            allow_external_media=True,
        )
    except TypeError as exc:
        pytest.fail(f"chat_with_vlm 尚未实现多附件契约：{exc}")

    assert reply == "收到两张图"
    content = captured["messages"][-1]["content"]
    assert [item["type"] for item in content] == ["text", "image_url", "image_url"]
    assert content[1]["image_url"]["url"].endswith("first")
    assert content[2]["image_url"]["url"].endswith("second")


def test_image_metadata_bounds_crs_and_resolution_require_spatial_consent(monkeypatch, tmp_path):
    captured = []

    class _Executor:
        def invoke(self, payload):
            captured.append(payload)
            return {"messages": [SimpleNamespace(type="ai", content="收到")]}

    image = tmp_path / "preview.tif"
    image.write_bytes(b"placeholder")
    monkeypatch.setattr(agent, "_get_agent_executor", lambda **_kwargs: _Executor())
    monkeypatch.setattr(agent, "_build_image_data_url", lambda *_args, **_kwargs: "data:image/tiff;base64,AA==")
    monkeypatch.setattr(
        agent,
        "_extract_geotiff_meta_text",
        lambda _path: (
            "[GeoTIFF metadata]\n"
            "- crs: EPSG:4326\n"
            "- resolution: x=0.00025, y=0.00025\n"
            "- bounds: left=120.600000, bottom=30.200000, right=121.200000, top=30.900000\n"
        ),
    )
    monkeypatch.setattr(agent, "_attach_geo_meta", True)

    reply = agent.chat_with_vlm(
        "请分析影像",
        [],
        image_path=str(image),
        sidebar_context=(
            "- crs: EPSG:4326\n"
            "- resolution: x=0.00025, y=0.00025\n"
            "- bounds: left=120.600000, bottom=30.200000, right=121.200000, top=30.900000"
        ),
        allow_external_media=True,
        allow_spatial_metadata=False,
    )
    assert reply == "收到"
    blob = str(captured[-1]["messages"])
    assert "EPSG:4326" not in blob
    assert "120.600000" not in blob
    assert "0.00025" not in blob
    assert "<spatial-redacted>" in blob

    captured.clear()
    authorized_reply = agent.chat_with_vlm(
        "请分析影像",
        [],
        image_path=str(image),
        sidebar_context=(
            "- crs: EPSG:4326\n"
            "- resolution: x=0.00025, y=0.00025\n"
            "- bounds: left=120.600000, bottom=30.200000, right=121.200000, top=30.900000"
        ),
        allow_external_media=True,
        allow_spatial_metadata=True,
    )
    assert authorized_reply == "收到"
    authorized_blob = str(captured[-1]["messages"])
    assert "EPSG:4326" in authorized_blob
    assert "120.600000" in authorized_blob


def test_system_prompt_has_no_absolute_path_example():
    assert "I:\\GEE_data" not in agent.system_prompt_base
    assert "/Users/" not in agent.system_prompt_base


def test_external_geotiff_is_always_converted_to_metadata_free_png(monkeypatch, tmp_path):
    captured = []

    class _Executor:
        def invoke(self, payload):
            captured.append(payload)
            return {"messages": [SimpleNamespace(type="ai", content="收到")]}

    image = tmp_path / "spatial.tif"
    image.write_bytes(b"placeholder")
    monkeypatch.setattr(agent, "_get_agent_executor", lambda **_kwargs: _Executor())
    calls = []

    def _data_url(path, *, force_png_for_tiff=False):
        calls.append((path, force_png_for_tiff))
        return "data:image/png;base64,AA=="

    monkeypatch.setattr(agent, "_build_image_data_url", _data_url)
    reply = agent.chat_with_vlm(
        "请分析影像", [], image_path=str(image), allow_external_media=True,
    )
    assert reply == "收到"
    assert calls and calls[-1][1] is True
    assert "image/tiff" not in str(captured[-1]["messages"])
