# -*- coding: utf-8 -*-
"""可配置 LLM 后端与能力声明。

配置读取不触发网络请求或模型加载；调用方可在真正发起聊天时再构造客户端。
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import FrozenSet, Optional

from langchain_openai import ChatOpenAI


class BackendUnavailable(RuntimeError):
    pass


# DashScope 的视觉模型命名并不统一：部分模型没有 ``vl`` 后缀。
# 这些前缀覆盖当前项目模型选择器中已知的多模态模型；未知模型仍可
# 通过 CSTF_LLM_SUPPORTS_VISION 显式覆盖，避免依赖名称猜测能力。
_KNOWN_VISION_MODEL_PREFIXES = (
    "qwen3.8",
    "qwen3.7-flash",
    "qwen3.7-plus",
    "qwen3.7-max-2026-06-08",
    "qwen3.5-ocr",
    "qwen3.5-flash",
    "qwen3.5-plus",
    "qwen3-vl",
    "qwen2.5-vl",
    "qwen-vl",
    "kimi-k3",
    "kimi-k2.7-code",
)


def model_supports_vision(model: str) -> bool:
    """Return the conservative built-in visual-capability classification.

    A model identifier containing ``vl`` remains compatible with the original
    behavior.  Known DashScope multimodal families are also recognized even
    when their public identifier omits that suffix.
    """
    normalized = str(model or "").strip().lower()
    return "vl" in normalized or any(
        normalized.startswith(prefix) for prefix in _KNOWN_VISION_MODEL_PREFIXES
    )


def _vision_override_from_env() -> Optional[bool]:
    """Read an explicit remote vision override, if configured."""
    for name in ("CSTF_LLM_SUPPORTS_VISION", "CSTF_DASHSCOPE_SUPPORTS_VISION"):
        if name not in os.environ:
            continue
        value = os.environ.get(name, "").strip().lower()
        if value in {"1", "true", "yes", "on"}:
            return True
        if value in {"0", "false", "no", "off"}:
            return False
    return None


@dataclass(frozen=True)
class LLMBackendConfig:
    provider: str
    model: str
    base_url: str
    api_key: Optional[str]
    capabilities: FrozenSet[str]
    temperature: float = 0.1

    @classmethod
    def from_env(cls) -> "LLMBackendConfig":
        provider = (os.environ.get("CSTF_LLM_BACKEND") or "dashscope").strip().lower()
        model = (
            os.environ.get("CSTF_LLM_MODEL")
            or os.environ.get("QWEN_CHAT_MODEL")
            or "qwen-plus"
        ).strip()
        base_url = (
            os.environ.get("CSTF_LLM_BASE_URL")
            or os.environ.get("QWEN_OPENAI_BASE_URL")
            or "https://dashscope.aliyuncs.com/compatible-mode/v1"
        ).strip()
        api_key = (
            os.environ.get("CSTF_LLM_API_KEY")
            or os.environ.get("DASHSCOPE_API_KEY")
            or os.environ.get("QWEN_API_KEY")
        )
        if provider in {"local", "openai-compatible", "openai_compatible"}:
            provider = "local"
            capabilities = {"text"}
            if os.environ.get("CSTF_LOCAL_SUPPORTS_TOOLS", "").lower() in {"1", "true", "yes"}:
                capabilities.add("tools")
            if os.environ.get("CSTF_LOCAL_SUPPORTS_VISION", "").lower() in {"1", "true", "yes"}:
                capabilities.add("vision")
            api_key = api_key or "local"
        else:
            provider = "dashscope"
            capabilities = {"text", "tools"}
            vision_override = _vision_override_from_env()
            supports_vision = (
                vision_override
                if vision_override is not None
                else model_supports_vision(model)
            )
            if supports_vision:
                capabilities.add("vision")
        return cls(
            provider=provider,
            model=model,
            base_url=base_url,
            api_key=api_key,
            capabilities=frozenset(capabilities),
        )

    @property
    def configured(self) -> bool:
        return bool(self.api_key)


def backend_status(config: Optional[LLMBackendConfig] = None) -> dict:
    cfg = config or LLMBackendConfig.from_env()
    return {
        "provider": cfg.provider,
        "model": cfg.model,
        "configured": cfg.configured,
        "capabilities": sorted(cfg.capabilities),
    }


def build_chat_model(
    config: Optional[LLMBackendConfig] = None,
    *,
    require_tools: bool = False,
    require_vision: bool = False,
) -> ChatOpenAI:
    cfg = config or LLMBackendConfig.from_env()
    if not cfg.configured:
        raise BackendUnavailable(
            "聊天后端未配置：请在被忽略的 .env 中设置 DASHSCOPE_API_KEY，或选择已配置的本地后端。"
        )
    required = set()
    if require_tools:
        required.add("tools")
    if require_vision:
        required.add("vision")
    missing = required - set(cfg.capabilities)
    if missing:
        raise BackendUnavailable(
            f"当前 LLM 后端不具备所需能力：{', '.join(sorted(missing))}。"
        )
    return ChatOpenAI(
        model=cfg.model,
        api_key=cfg.api_key,
        base_url=cfg.base_url,
        temperature=cfg.temperature,
    )


__all__ = [
    "BackendUnavailable",
    "LLMBackendConfig",
    "backend_status",
    "build_chat_model",
    "model_supports_vision",
]
