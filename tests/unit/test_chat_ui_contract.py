# -*- coding: utf-8 -*-
"""聊天区布局契约：会话可见、授权控件不侵入输入区、宽度可调。"""
from __future__ import annotations

from pathlib import Path
import unittest


class TestChatUiContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = (Path(__file__).parents[2] / "TF-agent" / "app.py").read_text(encoding="utf-8")

    def test_chat_panel_has_resizable_width_and_session_list(self):
        self.assertIn("agent_chat_width_pct", self.source)
        self.assertIn("对话区宽度", self.source)
        self.assertIn("list_threads", self.source)
        self.assertIn("conversation_switch", self.source)

    def test_agent_dock_separates_chat_history_and_resizes_at_edge(self):
        """Agent Dock 保持常驻，宽度通过边缘分隔线调整。"""
        self.assertIn("agent_dock_view", self.source)
        self.assertIn("历史", self.source)
        self.assertIn("cstf-dock-resize-handle", self.source)
        self.assertNotIn('key="agent_dock_collapse"', self.source)

    def test_history_view_hides_chat_stream_and_composer(self):
        """历史页只做会话导航，不展示聊天流或发送控件。"""
        # Streamlit 会按浏览器语言翻译 aria-label，不能用英文 widget key 定位表单。
        self.assertIn(
            'div[data-testid="stColumn"]:has(.cstf-agent-view-history) [data-testid="stForm"],',
            self.source,
        )
        self.assertIn(
            'div[data-testid="stColumn"]:has(.cstf-agent-view-history) [data-testid="stForm"]:has(input[aria-label="chat_input"])',
            self.source,
        )
        self.assertIn(
            'div[data-testid="stColumn"]:has(.cstf-agent-view-history) [data-testid="stChatMessage"]',
            self.source,
        )
        self.assertIn(
            'div[data-testid="stColumn"]:has(.cstf-agent-view-history) > div[data-testid="stVerticalBlock"] > [data-testid="stLayoutWrapper"]:has([data-testid="stChatMessage"])',
            self.source,
        )
        self.assertIn("cstf-chat-stream-marker", self.source)
        self.assertIn('[data-testid="stVerticalBlockBorderWrapper"]:has(.cstf-chat-stream-marker)', self.source)

    def test_history_session_list_uses_remaining_dock_height(self):
        """会话列表填充剩余空间；空状态仅显示提示文本。"""
        self.assertIn('with st.container(height="stretch", border=True):', self.source)
        self.assertIn('if not _conversation_threads:', self.source)
        self.assertIn('[data-testid="stLayoutWrapper"]:has(.cstf-chat-stream-marker)', self.source)
        self.assertIn('st.caption("暂无历史会话")', self.source)
        self.assertIn('_conv_c1, _conv_c2 = st.columns(2)', self.source)

    def test_history_session_list_has_no_fixed_count_limit_and_scrolls(self):
        """历史会话不按固定条数截断，超过面板高度时只在列表框内滚动。"""
        self.assertIn("list_threads(\n                limit=None,", self.source)
        self.assertNotIn("list_threads(\n                limit=8,", self.source)
        self.assertIn("cstf-history-list-marker", self.source)
        self.assertIn('[data-testid="stLayoutWrapper"]:has(.cstf-history-list-marker)', self.source)
        self.assertIn("overflow-y: auto !important;", self.source)

    def test_history_list_frame_stays_fixed_while_items_scroll(self):
        """列表边框固定在滚动视口，不能随内部会话项一起滚动。"""
        frame_selector = (
            'div[data-testid="stColumn"]:has(.cstf-agent-view-history) '
            '[data-testid="stLayoutWrapper"]:has(.cstf-history-list-marker) {'
        )
        frame_css = self.source.split(frame_selector, 1)[1].split("}", 1)[0]
        self.assertIn("border: 1px solid", frame_css)

        inner_selector = (
            'div[data-testid="stColumn"]:has(.cstf-agent-view-history) '
            '[data-testid="stLayoutWrapper"]:has(.cstf-history-list-marker) '
            '> [data-testid="stVerticalBlock"] {'
        )
        inner_css = self.source.split(inner_selector, 1)[1].split("}", 1)[0]
        self.assertIn("border: 0 !important;", inner_css)

    def test_status_log_panel_is_below_map_and_adjustable(self):
        """状态/日志移到地图下方，并提供可收起与高度调节。"""
        self.assertIn('cstf-map-status-zone', self.source)
        self.assertIn('agent_status_panel_height', self.source)
        self.assertIn('agent_status_panel_collapsed', self.source)
        self.assertIn('cstf-status-edge-handle', self.source)

    def test_status_panel_preserves_fragment_refresh_fallback(self):
        """日志保持局部刷新，并为不支持 fragment 的环境保留整页刷新兜底。"""
        self.assertIn('st.fragment(run_every=2.5)(_pipeline_monitor_inner)', self.source)
        self.assertIn('not _PIPELINE_USE_FRAGMENT\n    and st.session_state.is_running', self.source)
        self.assertIn('cstf-status-edge-toggle', self.source)
        self.assertIn('cstf-dock-resize-handle', self.source)
        self.assertNotIn('cstf-map-status-title', self.source)
        self.assertIn('if _log_panel_slot is not None:', self.source)
        self.assertNotIn('_log_panel_slot = st.container()\n    st.markdown(\'<div class="cstf-log-panel-host-marker', self.source)
        self.assertIn('--cstf-status-panel-reserve', self.source)
        self.assertIn('mapPx', self.source)

    def test_resize_controls_do_not_render_sliders(self):
        """尺寸控制使用边缘拖拽，不再渲染两个可见滑块。"""
        self.assertNotIn('st.slider(\n            "状态区高度"', self.source)
        self.assertNotIn('st.slider(\n            "对话区宽度"', self.source)

    def test_chat_attachment_is_a_plus_button_in_compose_row(self):
        """附件选择器隐藏，+ 入口移动到消息输入行左侧。"""
        self.assertIn("inputRow.insertBefore(bar, inputRow.firstChild)", self.source)
        self.assertIn('input[aria-label="聊天输入"]', self.source)
        self.assertIn("cstf-chat-compose", self.source)
        self.assertIn("cstf-chat-input-row", self.source)
        self.assertIn("cstf-chat-send-column", self.source)
        self.assertIn('class="cstf-plus-btn"', self.source)
        self.assertIn("每个文件≤200MB", self.source)
        self.assertIn('data-tooltip="每个文件≤200MB · PNG / JPG / WebP / TIFF"', self.source)
        self.assertIn("content: attr(data-tooltip);", self.source)
        self.assertNotIn("附件仅用于本机会话预览", self.source)
        self.assertIn('data-testid="stFileUploader"] {', self.source)
        self.assertIn("flex-direction: row !important;", self.source)
        self.assertIn("order: 2 !important;", self.source)
        self.assertIn("fileInput.click();", self.source)
        self.assertNotIn('data-media-mode="local"', self.source)
        self.assertNotIn('data-media-mode="external"', self.source)
        self.assertNotIn("cstf-attach-choice", self.source)
        self.assertIn("fileInput.value = '';", self.source)
        self.assertNotIn("win.setTimeout(clearSelectedFileUi", self.source)
        self.assertIn("__cstfAttachmentLastEpoch", self.source)
        self.assertIn("attachmentEpochChanged", self.source)
        self.assertIn("__cstfAttachmentReconciler", self.source)
        self.assertIn("setInterval(reconcileAttachment", self.source)

    def test_attachment_preview_supports_all_formats_and_keeps_tooltip_static(self):
        """上传阶段显示预览；文件名不能污染固定的加号格式提示。"""
        self.assertIn("cstf-attach-preview", self.source)
        self.assertIn("URL.createObjectURL(file)", self.source)
        self.assertIn("image/tiff", self.source)
        self.assertIn("cstf-attach-preview-clear", self.source)
        self.assertIn("plusBtn.dataset.tooltip = defaultTooltip", self.source)
        self.assertNotIn("const described =", self.source)
        self.assertNotIn("const names = files.slice(0, 3)", self.source)

    def test_attachment_previews_are_arranged_in_one_horizontal_row(self):
        """多个附件预览应并排显示，超出宽度时仅预览条横向滚动。"""
        start = self.source.index(".cstf-attach-preview {")
        end = self.source.index(".cstf-attach-preview.is-visible", start)
        preview_css = self.source[start:end]
        self.assertIn("flex-direction: row !important;", preview_css)
        self.assertIn("flex-wrap: nowrap !important;", preview_css)
        self.assertIn("width: auto !important;", preview_css)
        self.assertIn("min-width: 0 !important;", preview_css)
        self.assertIn("overflow-x: auto !important;", preview_css)

    def test_attachment_previews_stay_inside_compose_box(self):
        """预览条应在聊天输入外框内展开，不能漂浮到外框上方。"""
        start = self.source.index(".cstf-attach-preview {")
        end = self.source.index(".cstf-attach-preview.is-visible", start)
        preview_css = self.source[start:end]
        self.assertIn("top: 8px !important;", preview_css)
        self.assertIn("bottom: auto !important;", preview_css)
        self.assertIn("left: 10px !important;", preview_css)
        self.assertIn("right: 10px !important;", preview_css)
        self.assertIn("min-width: 0 !important;", preview_css)
        self.assertIn("max-width: none !important;", preview_css)
        self.assertIn(".cstf-chat-compose .cstf-attach-bar", self.source)
        self.assertIn("position: static !important;", self.source)
        self.assertIn(".cstf-chat-compose:has(.cstf-attach-preview.is-visible)", self.source)
        self.assertIn("padding-top: 5.8rem !important;", self.source)

    def test_attachment_selection_accumulates_across_chooser_rounds(self):
        """连续打开选择器时，附件应累积到同一个原生 FileList。"""
        self.assertIn("selectedFiles", self.source)
        self.assertIn("new win.DataTransfer()", self.source)
        self.assertIn("transfer.items.add(file)", self.source)
        self.assertIn("fileInput.files = transfer.files", self.source)
        self.assertIn("selectedFilesSyncing", self.source)
        self.assertIn("dispatchEvent(new win.Event('change'", self.source)
        self.assertIn("mergeSelectedFiles", self.source)
        self.assertIn("syncAttach([])", self.source)

    def test_attachment_first_selection_does_not_dispatch_duplicate_change(self):
        """首轮原生 change 已包含文件，不应再合成一次相同变更。"""
        self.assertIn("const priorSelectionCount = selectedFiles.length", self.source)
        self.assertIn("const shouldNotify = priorSelectionCount > 0", self.source)
        self.assertIn("selectedFiles.length > priorSelectionCount", self.source)
        self.assertIn("assignSelectedFiles(selectedFiles, shouldNotify)", self.source)
        self.assertIn("fileInput.addEventListener('change', (event) =>", self.source)
        self.assertIn("event.stopImmediatePropagation()", self.source)
        self.assertIn("2026-08-23-attachment-v7", self.source)

    def test_attachment_submit_clear_survives_epoch_rotation_and_preview_sync_is_idempotent(self):
        """提交清空跨 uploader epoch 生效，轮询同步不能反复重建同一预览。"""
        self.assertIn("win.__cstfAttachmentClearRequested = true", self.source)
        self.assertIn("const mustRemainCleared =", self.source)
        self.assertIn("Boolean(\n                win.__cstfAttachmentClearRequested", self.source)
        self.assertIn("? (mustRemainCleared ? []", self.source)
        self.assertIn("let renderedPreviewSignature = null", self.source)
        self.assertIn("if (previewSignature === renderedPreviewSignature) return;", self.source)
        self.assertNotIn("delete win.__cstfAttachmentPendingEpoch", self.source)

    def test_attachment_file_identity_ignores_unstable_last_modified_timestamp(self):
        """浏览器重建 File 时的动态时间戳不能触发重复预览或重复附件。"""
        start = self.source.index("const fileIdentity =")
        end = self.source.index("const assignSelectedFiles", start)
        identity_code = self.source[start:end]
        self.assertIn("String(file?.type || '')", identity_code)
        self.assertNotIn("file?.lastModified", identity_code)

    def test_attachment_submit_keeps_native_file_until_server_rotates_uploader(self):
        """发送时可立即隐藏预览，但不能先清空原生 FileList。"""
        start = self.source.index("const handleSendClick =")
        end = self.source.index("const destroy =", start)
        submit_bridge = self.source[start:end]
        self.assertIn("renderAttachmentPreview([])", submit_bridge)
        self.assertIn("_attachment_uploader_epoch", self.source)
        self.assertNotIn("clearSelectedFileUi", submit_bridge)
        self.assertNotIn("fileInput.value = ''", submit_bridge)

    def test_uploaded_attachment_batch_is_deduplicated_before_submission(self):
        """前端重复 change 即使抵达服务端，也不能复制同一附件消息。"""
        self.assertIn("def _dedupe_uploaded_images", self.source)
        self.assertIn(
            "uploaded_images = _dedupe_uploaded_images(uploaded_images)",
            self.source,
        )

    def test_map_ready_warning_waits_until_fly_ack_is_known(self):
        """地图已收到定位确认时，不应保留发送前的未就绪提示。"""
        self.assertIn("_map_ready_warning = False", self.source)
        self.assertIn("if _ack is None and _map_ready_warning:", self.source)
        self.assertIn(
            '_globe_srv.map_protocol_state(\n                                channel_id=_map_channel_id',
            self.source,
        )
        self.assertIn(
            'channel_id=st.session_state.get("_map_channel_id")',
            self.source,
        )
        self.assertIn(
            '_globe_srv.wait_map_ack(\n                                _fly_payload.get("command_id", ""),',
            self.source,
        )

    def test_resize_handles_show_only_the_blue_drag_bar(self):
        """拖动命中区不能带浏览器焦点外框或额外边框。"""
        self.assertIn(".cstf-dock-resize-handle:focus-visible", self.source)
        self.assertIn(".cstf-status-edge-handle:focus-visible", self.source)
        self.assertIn(
            "outline: none !important;\n        border: 0 !important;\n        box-shadow: none !important;",
            self.source,
        )

    def test_status_drawer_has_no_streamlit_gap_row(self):
        """地图列的默认子项间距不能把状态抽屉推离地图边界。"""
        self.assertIn("gap: 0 !important;", self.source)
        self.assertIn("max-height: 0 !important;", self.source)
        self.assertIn("(mapRect.left + mapRect.right) / 2", self.source)
        self.assertIn("collapsed ? mapRect.bottom - toggleHeight : mapRect.bottom", self.source)

    def test_status_toggle_targets_real_button_not_help_tooltip_button(self):
        """状态三角桥接必须避开 Streamlit help 生成的提示按钮。"""
        self.assertIn("div.st-key-agent_status_panel_toggle button", self.source)

    def test_status_bridge_button_does_not_create_nested_help_control(self):
        """隐藏桥接按钮不应再创建会拦截点击的 Streamlit help 子按钮。"""
        self.assertNotIn(
            'key="agent_status_panel_toggle",\n            help="收起或展开状态区",',
            self.source,
        )

    def test_status_bridge_row_remains_rendered_without_reserving_layout_space(self):
        """桥接行不能 display:none，否则第二次点击无法触发 Streamlit 事件。"""
        self.assertIn("position: absolute !important;", self.source)
        self.assertIn("pointer-events: none !important;", self.source)
        self.assertIn(
            'div.st-key-agent_status_panel_toggle button {',
            self.source,
        )

    def test_status_toggle_rebinds_after_streamlit_rerender(self):
        """Streamlit 重绘后必须重新确保三角按钮的 click 监听器存在。"""
        self.assertIn("toggle.dataset.cstfStatusClickBound", self.source)
        self.assertIn('toggle.setAttribute(\n            "onclick"', self.source)

    def test_resize_handles_expose_accessible_size_values(self):
        self.assertIn('handle.setAttribute("aria-valuemin", "24")', self.source)
        self.assertIn('handle.setAttribute("aria-valuemax", "48")', self.source)
        self.assertIn('handle.setAttribute("aria-valuemin", "192")', self.source)
        self.assertIn('handle.setAttribute("aria-valuemax", "392")', self.source)

    def test_resized_columns_remain_responsive_after_viewport_changes(self):
        """拖动后列宽不能写死像素，否则窗口变化会把一列挤到视口外。"""
        self.assertIn("flex-wrap: nowrap !important;", self.source)
        self.assertIn('setImp(pair[0], "width", "calc("', self.source)
        self.assertIn('setImp(pair[0], "max-width", "calc("', self.source)
        self.assertNotIn(
            'setImp(pair[0], "flex", "0 0 " + pair[1] + "px")',
            self.source,
        )
        self.assertIn('handle.setAttribute("aria-valuenow"', self.source)

    def test_viewport_resize_keeps_root_surfaces_dark(self):
        """窗口快速缩放时，顶层布局重排不能露出浏览器默认白色背景。"""
        self.assertIn("html, body, .stApp, [data-testid=\"stApp\"]", self.source)
        self.assertIn("background-color: #0e0e0e !important;", self.source)

    def test_streamlit_text_input_wrappers_do_not_flash_white_on_resize(self):
        """Streamlit 输入控件的外层包装不能在重排时露出白底白框。"""
        self.assertIn('[data-testid="stTextInputRootElement"] {', self.source)
        self.assertIn('background-color: transparent !important;', self.source)
        self.assertIn('border: none !important;', self.source)

    def test_resize_keeps_streamlit_status_and_chat_surfaces_dark(self):
        """状态进度条、组合框与头像外层不能恢复 Streamlit 默认白色主题。"""
        self.assertIn('[data-testid="stProgress"] [role="progressbar"] > div:first-child {', self.source)
        self.assertIn('.react-aria-ComboBox {', self.source)
        self.assertIn('.react-aria-ComboBox > [role="group"] {', self.source)
        self.assertIn('[data-testid="stChatMessage"] > div:first-child {', self.source)

    def test_resize_edge_hit_areas_stay_above_embedded_content(self):
        """地图 iframe 与聊天内容不能遮挡整条拖拽边缘。"""
        self.assertIn("z-index: 1350;", self.source)
        self.assertIn("z-index: 1600;", self.source)

    def test_resize_drag_lifecycle_runs_in_parent_document_realm(self):
        """拖拽不能依赖会被 Streamlit 重绘回收的组件 iframe 监听器。"""
        self.assertIn("const parentResizePointerDown = String.raw`", self.source)
        self.assertIn("const parentResizeKeyDown = String.raw`", self.source)
        self.assertGreaterEqual(
            self.source.count('handle.setAttribute("onpointerdown", parentResizePointerDown);'),
            2,
        )
        self.assertIn('win.addEventListener("pointermove", move, true);', self.source)
        self.assertIn('win.addEventListener("pointerup", stop, true);', self.source)
        self.assertIn('overlay.className = "cstf-resize-capture";', self.source)
        self.assertIn('overlay.addEventListener("pointermove", move, true);', self.source)
        self.assertIn('overlay.addEventListener("mouseup", stop, true);', self.source)
        self.assertIn("overlay.remove();", self.source)
        self.assertNotIn('handle.addEventListener("pointermove"', self.source)

    def test_agent_width_resize_resynchronizes_status_boundary(self):
        """Agent 宽度变化后，状态区边缘必须立即跟随新的地图宽度。"""
        self.assertIn('statusHandle.style.width = mapRect.width + "px";', self.source)
        self.assertIn('statusHandle.style.left = mapRect.left + "px";', self.source)
        self.assertIn("applyDockWidth(pct);", self.source)
        self.assertIn("syncResizeGeometry();", self.source)

    def test_map_height_excludes_zero_height_command_iframes(self):
        """定位 postMessage 的辅助 iframe 不得被扩成地图高度。"""
        self.assertIn("const getPrimaryMapFrame = (mapCol) =>", self.source)
        self.assertIn('iframe[src*="/globe"]', self.source)
        self.assertIn('iframe[title*="streamlit_folium"]', self.source)
        self.assertIn("const mapFrame = getPrimaryMapFrame(mapCol);", self.source)

    def test_map_fly_retries_cancel_previous_command(self):
        """连续定位时，旧命令的延迟重试不能覆盖新定位。"""
        self.assertIn("__cstfFlyRetryTimers", self.source)
        self.assertIn("oldTimers.forEach((timerId) => win.clearTimeout(timerId));", self.source)
        self.assertIn("win.__cstfFlyRetryTimers.push(timerId);", self.source)

    def test_alerts_are_dismissible_and_do_not_reflow_workbench(self):
        """错误/警告通知浮动显示并支持关闭，避免挤压地图与 Agent。"""
        self.assertIn("cstf-dismissible-alert", self.source)
        self.assertIn("cstf-alert-close", self.source)
        self.assertIn('aria-label", "关闭通知"', self.source)
        self.assertIn("sessionStorage.getItem(noticeKey)", self.source)
        self.assertIn("sessionStorage.setItem(noticeKey, \"1\")", self.source)

    def test_history_view_prioritizes_conversation_space_over_monitor_log(self):
        """状态/日志在地图下方，与 Agent 历史页相互独立。"""
        self.assertIn('cstf-map-status-zone', self.source)
        self.assertIn('if _log_panel_slot is not None:', self.source)

    def test_chat_failure_is_persisted_as_an_assistant_reply(self):
        """连接失败也要留在会话流中，避免历史发送看起来像没有响应。"""
        self.assertIn('st.session_state.messages.append({"role": "assistant", "content": _error_reply})', self.source)

    def test_chat_messages_have_distinct_left_right_alignment(self):
        """用户消息右对齐、助手消息左对齐，且卡片宽度受控。"""
        self.assertIn('[data-testid="stChatMessage"]:has(.msg-role-user)', self.source)
        self.assertIn('flex-direction: row-reverse', self.source)
        self.assertIn('margin-left: auto', self.source)
        self.assertIn('[data-testid="stChatMessage"]:has(.msg-role-assistant)', self.source)
        self.assertIn('margin-right: auto', self.source)
        self.assertIn('max-width: 86%', self.source)

    def test_chat_stream_and_composer_have_explicit_size_contract(self):
        """消息滚动区占剩余高度，输入区固定收缩，避免再次出现空白或溢出。"""
        self.assertIn('min-height: 0 !important', self.source)
        self.assertIn('flex: 0 0 auto !important', self.source)
        self.assertIn('max-height: 250px !important', self.source)

    def test_message_cards_shrink_to_content_before_max_width(self):
        """短消息不应继承整列宽度，长消息仍受最大宽度约束。"""
        self.assertIn('width: fit-content !important', self.source)
        self.assertIn('min-width: 7rem !important', self.source)
        self.assertIn('max-width: 86% !important', self.source)

    def test_history_view_is_navigation_only_and_session_switch_opens_chat(self):
        """历史页仅展示记录，选中会话后自动返回对话视图。"""
        self.assertIn(
            'div[data-testid="stColumn"]:has(.cstf-agent-view-history) [data-testid="stForm"]:has(input[aria-label="chat_input"])',
            self.source,
        )
        self.assertIn(
            'div[data-testid="stColumn"]:has(.cstf-agent-view-history) [data-testid="stChatMessage"]',
            self.source,
        )
        self.assertIn(
            'div[data-testid="stColumn"]:has(.cstf-agent-view-history) > div[data-testid="stVerticalBlock"] > [data-testid="stLayoutWrapper"]:has([data-testid="stChatMessage"])',
            self.source,
        )
        self.assertIn('st.session_state.agent_dock_view = "对话"', self.source)

    def test_clear_session_selects_next_without_opening_chat(self):
        """清空当前会话后，应在历史页选中下一条会话。"""
        self.assertIn("next_thread_id_after_delete", self.source)
        self.assertIn("_next_thread_id = next_thread_id_after_delete", self.source)
        self.assertIn("load_messages(_next_thread_id)", self.source)
        self.assertIn('_conv_c1, _conv_c2 = st.columns(2)', self.source)
        self.assertNotIn("_conversation_stay_in_history", self.source)

    def test_clear_last_session_creates_new_chat_and_opens_dialog(self):
        """删除最后一条会话后，必须创建新会话并直接回到对话视图。"""
        start = self.source.index("_next_thread_id = next_thread_id_after_delete")
        end = self.source.index("st.rerun()", start)
        clear_block = self.source[start:end]
        self.assertIn("st.session_state._conversation_thread_id = (", clear_block)
        self.assertIn("st.session_state._conversation_store.create_thread()", clear_block)
        self.assertIn("st.session_state.messages = [_default_chat_message.copy()]", clear_block)
        self.assertIn("st.session_state._conversation_open_dialog = True", clear_block)

    def test_attachment_observer_uses_parent_document_realm(self):
        self.assertIn("win.MutationObserver", self.source)

    def test_chat_input_does_not_render_unrequested_consent_checkboxes(self):
        self.assertNotIn("允许将上传影像发送给外部模型", self.source)
        self.assertNotIn("允许发送精确空间元数据", self.source)
        self.assertNotIn('st.checkbox(\n            "允许将上传影像', self.source)
        self.assertNotIn('st.checkbox(\n            "允许发送精确空间', self.source)


if __name__ == "__main__":
    unittest.main()
