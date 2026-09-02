"""Streamlit 原生 AppTest：不依赖 Chromium 的离线交互验收。"""
from __future__ import annotations

import base64
from pathlib import Path
import sys

from streamlit.testing.v1 import AppTest


ROOT = Path(__file__).resolve().parents[2]
APP = ROOT / "TF-agent" / "app.py"
if str(ROOT / "TF-agent") not in sys.path:
    sys.path.insert(0, str(ROOT / "TF-agent"))

from job_store import JobStore  # noqa: E402
from task_timeline import TimelineStore  # noqa: E402
from conversation_store import ConversationStore  # noqa: E402


_PNG_1X1 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
)


def _run_app(tmp_path, monkeypatch):
    for key in (
        "DASHSCOPE_API_KEY",
        "CSTF_LLM_API_KEY",
        "QWEN_API_KEY",
        "CSTF_LLM_BACKEND",
        "CSTF_LLM_MODEL",
        "CSTF_LLM_BASE_URL",
        "EE_PROJECT",
        "GOOGLE_CLOUD_PROJECT",
        "EARTHENGINE_PROJECT",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "CSTF_ALLOW_RAW_SYSTEM_COMMAND",
    ):
        monkeypatch.setenv(key, "")
    monkeypatch.setenv("CSTF_ALLOW_RAW_SYSTEM_COMMAND", "0")
    monkeypatch.setenv("CSTF_CONVERSATION_DB_PATH", str(tmp_path / "conversations.sqlite3"))
    monkeypatch.setenv("CSTF_JOB_DB_PATH", str(tmp_path / "jobs.sqlite3"))
    monkeypatch.setenv("CSTF_TIMELINE_LEDGER_PATH", str(tmp_path / "timeline_ledger.json"))
    monkeypatch.setenv("CSTF_CHAT_PREVIEW_DIR", str(tmp_path / "chat-previews"))
    return AppTest.from_file(APP, default_timeout=60).run(timeout=60)


def _button(at, label: str):
    return next(button for button in at.button if button.label == label)


def test_root_shell_and_conversation_controls_render_without_credentials(tmp_path, monkeypatch):
    at = _run_app(tmp_path, monkeypatch)
    assert not at.exception
    assert any("智能分析助手" in item.value for item in at.markdown)
    dock_view = next(item for item in at.radio if item.label == "Agent 面板")
    dock_view.set_value("历史").run(timeout=60)
    assert not at.exception
    assert _button(at, "新会话")
    assert _button(at, "清空会话")
    # AppTest exposes rendered widgets before browser CSS is applied; visibility
    # of the history navigation-only view is covered by browser acceptance.
    assert any(item.label == "chat_input" for item in at.text_input)
    assert not any("开发模式：允许本轮直接执行聊天框系统命令" in item.label for item in at.checkbox)

    _button(at, "新会话").click().run(timeout=60)
    assert not at.exception
    assert any(item.label == "chat_input" for item in at.text_input)
    dock_view = next(item for item in at.radio if item.label == "Agent 面板")
    dock_view.set_value("历史").run(timeout=60)
    _button(at, "清空会话").click().run(timeout=60)
    assert not at.exception


def test_history_session_selection_opens_dialog_view(tmp_path, monkeypatch):
    """历史页选择已有会话后自动切换到对话视图。"""
    db_path = tmp_path / "conversations.sqlite3"
    store = ConversationStore(str(db_path))
    thread_id = store.create_thread()
    store.append_message(thread_id, "user", "历史会话测试")
    store.append_message(thread_id, "assistant", "已加载历史会话")

    at = _run_app(tmp_path, monkeypatch)
    assert not at.exception
    next(item for item in at.radio if item.label == "Agent 面板").set_value("历史").run(timeout=60)
    session_button = next(button for button in at.button if "历史会话测试" in button.label)
    session_button.click().run(timeout=60)
    assert not at.exception
    dock_view = next(item for item in at.radio if item.label == "Agent 面板")
    assert dock_view.value == "对话"
    assert any(item.label == "chat_input" for item in at.text_input)


def test_unconsented_attachment_is_rendered_as_local_message_preview(tmp_path, monkeypatch):
    """外发授权只控制模型载荷，不能阻止用户消息中的本地缩略图。"""
    at = _run_app(tmp_path, monkeypatch)
    initial_image_count = len(at.image)
    uploader = next(item for item in at.file_uploader if item.label == "chat_attach")
    uploader.set_value(("local-preview.png", _PNG_1X1, "image/png"))
    next(item for item in at.text_input if item.label == "chat_input").set_value("只做本地预览")
    _button(at, "➤").click().run(timeout=60)

    assert not at.exception
    assert len(at.image) == initial_image_count + 1
    assert any("local-preview.png" in caption for image in at.image for caption in image.captions)


def test_chat_uploader_accepts_multiple_files_and_clears_after_submit(tmp_path, monkeypatch):
    """一次可选择多张图片，提交完成后新的上传器不能保留旧选择。"""
    at = _run_app(tmp_path, monkeypatch)
    uploader = next(item for item in at.file_uploader if item.label == "chat_attach")
    assert uploader.accept_multiple_files is True
    uploader.set_value(
        [
            ("first.png", _PNG_1X1, "image/png"),
            ("second.png", _PNG_1X1, "image/png"),
        ]
    )
    next(item for item in at.text_input if item.label == "chat_input").set_value("比较两张图")
    _button(at, "➤").click().run(timeout=60)

    assert not at.exception
    current_uploader = next(item for item in at.file_uploader if item.label == "chat_attach")
    assert current_uploader.value in (None, [])
    captions = [caption for image in at.image for caption in image.captions]
    assert "first.png" in captions
    assert "second.png" in captions


def test_attachment_is_sent_by_default_without_consent_widget(tmp_path, monkeypatch):
    """恢复原 main：附件直接随当前模型请求发送，不再显示授权控件。"""
    at = _run_app(tmp_path, monkeypatch)
    assert not any(item.label == "附件外发授权（仅本轮）" for item in at.checkbox)
    uploader = next(item for item in at.file_uploader if item.label == "chat_attach")
    uploader.set_value(("authorized.png", _PNG_1X1, "image/png"))
    next(item for item in at.text_input if item.label == "chat_input").set_value("分析附件")
    _button(at, "➤").click().run(timeout=60)

    assert not at.exception
    assert not any("本轮未向外部模型发送附件内容" in item.value for item in at.warning)


def test_manual_deep_learning_button_creates_plan_before_execution(tmp_path, monkeypatch):
    at = _run_app(tmp_path, monkeypatch)
    _button(at, "开始模型提取").click().run(timeout=60)
    assert not at.exception
    visible_messages = [item.value for item in (*at.warning, *at.info, *at.success)]
    assert any("条件未满足" in message or "计划" in message for message in visible_messages)
    assert any("确认执行提取" == button.label for button in at.button)


def test_manual_index_button_creates_plan_before_execution(tmp_path, monkeypatch):
    at = _run_app(tmp_path, monkeypatch)
    mode = next(item for item in at.radio if item.label == "提取方式")
    mode.set_value("指数法").run(timeout=60)
    assert not at.exception
    _button(at, "开始指数法提取").click().run(timeout=60)
    assert not at.exception
    visible_messages = [item.value for item in (*at.warning, *at.info, *at.success)]
    assert any("指数法计划" in message or "条件" in message for message in visible_messages)
    assert any("确认执行指数法" == button.label for button in at.button)


def test_interrupted_inference_job_exposes_replan_gate(tmp_path, monkeypatch):
    """恢复账本只提供重新生成计划，不会自动重跑中断任务。"""
    jobs_path = tmp_path / "jobs.sqlite3"
    store = JobStore(str(jobs_path))
    store.create(task="task-recover", kind="dl", job_id="job-recover")
    store.claim("job-recover")

    at = _run_app(tmp_path, monkeypatch)
    assert not at.exception
    _button(at, "重新生成推理计划").click().run(timeout=60)
    assert not at.exception
    visible_messages = [item.value for item in (*at.warning, *at.info, *at.success)]
    assert any("重新生成推理计划" in message or "提取计划" in message for message in visible_messages)


def test_restored_timeline_exposes_report_confirmation_gate(tmp_path, monkeypatch):
    """历史时间线可回放，但报告生成仍必须经过独立确认。"""
    ledger = tmp_path / "timeline_ledger.json"
    timeline = TimelineStore(str(ledger))
    timeline.add(
        "task-replay",
        "VERIFY",
        status="SUCCEEDED",
        message="历史成果校验通过",
        details={"mode": "dl"},
    )
    timeline.save()

    at = _run_app(tmp_path, monkeypatch)
    assert not at.exception
    assert any("任务进度" in item.label for item in at.expander)
    _button(at, "📄 生成成果报告").click().run(timeout=60)
    assert not at.exception
    assert any("已生成成果报告计划" in item.value for item in at.info)
    assert _button(at, "确认生成成果报告")
    assert _button(at, "取消报告计划")
    # Planning is not execution: no report-generation success message appears
    # until the separate confirmation button is pressed.
    assert not any("成果报告已生成" in item.value for item in at.success)

    _button(at, "取消报告计划").click().run(timeout=60)
    assert not at.exception
    restored = TimelineStore(str(ledger))
    restored.load()
    assert any(event.status == "CANCELLED" and event.phase == "REPORT" for event in restored.events())
