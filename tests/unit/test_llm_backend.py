# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import sys
import unittest
from unittest import mock

_TF_AGENT = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "TF-agent")
)
if _TF_AGENT not in sys.path:
    sys.path.insert(0, _TF_AGENT)

from llm_backend import (  # noqa: E402
    BackendUnavailable,
    LLMBackendConfig,
    backend_status,
    build_chat_model,
)


class TestLLMBackend(unittest.TestCase):
    def test_missing_remote_key_is_import_safe_and_actionable(self):
        with mock.patch.dict(os.environ, {"CSTF_LLM_BACKEND": "dashscope"}, clear=True):
            cfg = LLMBackendConfig.from_env()
        self.assertFalse(cfg.configured)
        self.assertFalse(backend_status(cfg)["configured"])
        with self.assertRaises(BackendUnavailable):
            build_chat_model(cfg, require_tools=True)

    def test_dashscope_vl_declares_tools_and_vision(self):
        with mock.patch.dict(
            os.environ,
            {
                "CSTF_LLM_BACKEND": "dashscope",
                "CSTF_LLM_API_KEY": "unit-key",
                "CSTF_LLM_MODEL": "qwen-vl-plus",
            },
            clear=True,
        ):
            cfg = LLMBackendConfig.from_env()
        self.assertTrue({"text", "tools", "vision"}.issubset(cfg.capabilities))
        self.assertTrue(backend_status(cfg)["configured"])

    def test_dashscope_qwen38_models_declare_vision_without_vl_suffix(self):
        """DashScope 多模态模型不应依赖模型名包含 ``vl`` 才声明视觉能力。"""
        for model in ("qwen3.8-flash", "qwen3.8-27b"):
            with self.subTest(model=model), mock.patch.dict(
                os.environ,
                {
                    "CSTF_LLM_BACKEND": "dashscope",
                    "CSTF_LLM_API_KEY": "unit-key",
                    "CSTF_LLM_MODEL": model,
                },
                clear=True,
            ):
                cfg = LLMBackendConfig.from_env()
            self.assertIn("vision", cfg.capabilities)

    def test_local_without_tool_capability_is_rejected_for_agent(self):
        with mock.patch.dict(
            os.environ,
            {
                "CSTF_LLM_BACKEND": "local",
                "CSTF_LLM_BASE_URL": "http://127.0.0.1:8000/v1",
                "CSTF_LLM_API_KEY": "local",
            },
            clear=True,
        ):
            cfg = LLMBackendConfig.from_env()
        self.assertEqual(cfg.capabilities, frozenset({"text"}))
        with self.assertRaises(BackendUnavailable):
            build_chat_model(cfg, require_tools=True)


if __name__ == "__main__":
    unittest.main()
