"""可见 UI 验收；只有显式 RUN_BROWSER_ACCEPTANCE=1 才运行。"""
from __future__ import annotations

import base64
import os
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
APP = ROOT / "TF-agent" / "app.py"
if str(ROOT / "TF-agent") not in sys.path:
    sys.path.insert(0, str(ROOT / "TF-agent"))

from task_timeline import TimelineStore  # noqa: E402


_PNG_1X1 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@pytest.mark.external
def test_chat_attachment_state_stays_clear_and_uses_only_custom_tooltip():
    """发送完成后旧文件名不能回流，加号也不能叠加浏览器原生 title。"""
    if os.environ.get("RUN_EXTERNAL_ACCEPTANCE") != "1" or os.environ.get("RUN_BROWSER_ACCEPTANCE") != "1":
        pytest.skip("browser acceptance is opt-in")
    playwright = pytest.importorskip("playwright.sync_api")
    port = _free_port()
    with tempfile.TemporaryDirectory(prefix="cstf-attachment-browser-") as temp_dir:
        temp = Path(temp_dir)
        upload = temp / "attachment-regression.png"
        upload.write_bytes(_PNG_1X1)
        second_upload = temp / "attachment-second-round.png"
        second_upload.write_bytes(_PNG_1X1)
        env = os.environ.copy()
        for key in (
            "DASHSCOPE_API_KEY", "CSTF_LLM_API_KEY", "QWEN_API_KEY",
            "CSTF_LLM_BACKEND", "CSTF_LLM_MODEL", "CSTF_LLM_BASE_URL",
            "EE_PROJECT", "GOOGLE_CLOUD_PROJECT", "EARTHENGINE_PROJECT",
            "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "CSTF_ALLOW_RAW_SYSTEM_COMMAND",
        ):
            env[key] = ""
        env["CSTF_CONVERSATION_DB_PATH"] = str(temp / "conversations.sqlite3")
        env["CSTF_JOB_DB_PATH"] = str(temp / "jobs.sqlite3")
        env["CSTF_TIMELINE_LEDGER_PATH"] = str(temp / "timeline.json")
        env["CSTF_CHAT_PREVIEW_DIR"] = str(temp / "previews")
        process = subprocess.Popen(
            [
                sys.executable, "-m", "streamlit", "run", str(APP),
                "--server.headless", "true", "--server.address", "127.0.0.1",
                "--server.port", str(port), "--server.fileWatcherType", "none",
            ],
            cwd=str(ROOT), env=env,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        try:
            url = f"http://127.0.0.1:{port}/"
            with playwright.sync_playwright() as browser_api:
                browser = browser_api.chromium.launch(headless=True, args=["--no-proxy-server"])
                page = browser.new_page()
                deadline = time.monotonic() + 35
                while time.monotonic() < deadline:
                    try:
                        page.goto(url, wait_until="domcontentloaded", timeout=3000)
                        break
                    except Exception:
                        time.sleep(0.25)
                else:
                    pytest.fail("Streamlit attachment page did not become ready")

                plus = page.locator(".cstf-plus-btn")
                plus.wait_for(state="visible", timeout=15000)
                assert plus.get_attribute("title") is None
                assert "每个文件≤200MB" in (plus.get_attribute("data-tooltip") or "")

                with page.expect_file_chooser() as chooser_info:
                    plus.click()
                chooser_info.value.set_files(str(upload))
                # afff61c 起加号气泡保持格式说明不变，选定文件名出现在预览卡片上。
                page.locator(f'.cstf-attach-preview-card[title="{upload.name}"]').wait_for(
                    state="visible", timeout=15000
                )
                page.get_by_role("textbox", name="chat_input").fill("附件状态清理回归")
                page.get_by_role("button", name="➤").click()
                page.get_by_text("附件状态清理回归", exact=True).wait_for(
                    state="visible", timeout=15000
                )

                def assert_compose_is_clear() -> None:
                    assert page.locator(".cstf-attach-bar").count() == 1
                    # afff61c 起原生 FileList 保留到提交后 rerun 轮换上传器
                    # epoch 才清空，因此轮询等待而不是在消息回显瞬间断言。
                    page.wait_for_function(
                        "() => {"
                        " const el = document.querySelector('input[type=\"file\"]');"
                        " return !!el && el.files.length === 0;"
                        " }",
                        timeout=15000,
                    )
                    page.locator(".cstf-attach-preview-card").wait_for(
                        state="detached", timeout=15000
                    )
                    assert upload.name not in (plus.get_attribute("data-tooltip") or "")
                    assert plus.get_attribute("title") is None

                assert_compose_is_clear()
                page.wait_for_timeout(4200)
                assert_compose_is_clear()

                with page.expect_file_chooser() as second_chooser_info:
                    plus.click()
                second_chooser_info.value.set_files(str(second_upload))
                page.locator(f'.cstf-attach-preview-card[title="{second_upload.name}"]').wait_for(
                    state="visible", timeout=15000
                )
                assert page.locator(f'.cstf-attach-preview-card[title="{upload.name}"]').count() == 0
                assert upload.name not in (plus.get_attribute("data-tooltip") or "")
                assert plus.get_attribute("title") is None

                page.get_by_role("textbox", name="chat_input").fill("第二轮附件状态清理回归")
                page.get_by_role("button", name="➤").click()
                page.get_by_text("第二轮附件状态清理回归", exact=True).wait_for(
                    state="visible", timeout=15000
                )
                assert_compose_is_clear()
                assert second_upload.name not in (plus.get_attribute("data-tooltip") or "")
                browser.close()
        except Exception as exc:
            if "Executable doesn't exist" in str(exc) or "executable doesn't exist" in str(exc).lower():
                pytest.skip("Playwright Chromium executable is not installed")
            raise
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=10)


@pytest.mark.external
def test_streamlit_root_and_copilot_are_visible_in_browser():
    if os.environ.get("RUN_EXTERNAL_ACCEPTANCE") != "1" or os.environ.get("RUN_BROWSER_ACCEPTANCE") != "1":
        pytest.skip("browser acceptance is opt-in")
    playwright = pytest.importorskip("playwright.sync_api")
    port = _free_port()
    with tempfile.TemporaryDirectory(prefix="cstf-browser-") as temp_dir:
        timeline_path = Path(temp_dir) / "timeline_ledger.json"
        timeline = TimelineStore(str(timeline_path))
        timeline.add(
            "task-browser-replay",
            "VERIFY",
            status="SUCCEEDED",
            message="浏览器验收的历史成果已校验",
            details={"mode": "dl"},
        )
        timeline.save()
        env = os.environ.copy()
        for key in (
            "DASHSCOPE_API_KEY", "CSTF_LLM_API_KEY", "QWEN_API_KEY",
            "CSTF_LLM_BACKEND", "CSTF_LLM_MODEL", "CSTF_LLM_BASE_URL",
            "EE_PROJECT", "GOOGLE_CLOUD_PROJECT", "EARTHENGINE_PROJECT",
            "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "CSTF_ALLOW_RAW_SYSTEM_COMMAND",
        ):
            env[key] = ""
        env["CSTF_CONVERSATION_DB_PATH"] = str(Path(temp_dir) / "conversations.sqlite3")
        env["CSTF_JOB_DB_PATH"] = str(Path(temp_dir) / "jobs.sqlite3")
        env["CSTF_TIMELINE_LEDGER_PATH"] = str(Path(temp_dir) / "timeline_ledger.json")
        process = subprocess.Popen(
            [sys.executable, "-m", "streamlit", "run", str(APP), "--server.headless", "true", "--server.address", "127.0.0.1", "--server.port", str(port), "--server.fileWatcherType", "none"],
            cwd=str(ROOT), env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        try:
            deadline = time.monotonic() + 35
            url = f"http://127.0.0.1:{port}/"
            with playwright.sync_playwright() as browser_api:
                browser = browser_api.chromium.launch(headless=True, args=["--no-proxy-server"])
                page = browser.new_page()
                while time.monotonic() < deadline:
                    try:
                        page.goto(url, wait_until="domcontentloaded", timeout=3000)
                        break
                    except Exception:
                        time.sleep(0.25)
                else:
                    pytest.fail("Streamlit browser page did not become ready")
                page.get_by_text("智能分析助手", exact=False).first.wait_for(state="visible", timeout=15000)
                assert page.locator('[data-testid="stAppViewContainer"]').is_visible()
                # 本地 globe 服务应至少渲染 AOI 工具栏；底图网络失败不应隐藏
                # 协议入口，用户仍可看到明确的地图交互边界。
                # Streamlit may attach/rebuild the component iframe after its
                # src is already present.  A dynamic frame locator waits for
                # the actual child document instead of sampling page.frames
                # during that navigation window.
                globe_frame = page.frame_locator('iframe[src*="127.0.0.1:8765/globe"]').first
                try:
                    globe_frame.locator("#aoiToolbar").wait_for(state="visible", timeout=30000)
                except Exception:
                    iframe_srcs = page.locator("iframe").evaluate_all(
                        "els => els.map(el => el.getAttribute('src') || '')"
                    )
                    pytest.fail(f"local globe iframe did not become available: {iframe_srcs}")
                for label in ("点选", "矩形", "多边形", "当前视图", "清除"):
                    button = globe_frame.get_by_role("button", name=label)
                    if not button.is_visible():
                        pytest.fail(
                            f"AOI button not visible: {label}; count={button.count()}, "
                            f"toolbar={globe_frame.locator('#aoiToolbar').count()}"
                        )
                # Cesium 相机需先渲染首帧，computeViewRectangle 才可用；
                # 因此先等 canvas 可见再触发「当前视图」。headless 下首帧
                # 渲染耗时波动较大，等待上限与工具栏一致。
                canvas = globe_frame.locator("#cesiumContainer canvas").first
                canvas.wait_for(state="visible", timeout=30000)
                globe_frame.get_by_role("button", name="当前视图").click()
                # AOI ACK 由 globe 页面异步 fetch 返回，与其他异步断言一致使用 wait_for。
                globe_frame.get_by_text("AOI 已选定，已同步", exact=True).wait_for(
                    state="visible", timeout=10000
                )
                canvas_box = canvas.bounding_box()
                assert canvas_box is not None

                # 点击工具按钮只能进入绘制模式，不能把按钮点击本身误当成地图选择。
                globe_frame.get_by_role("button", name="清除").click()
                globe_frame.get_by_role("button", name="点选").click()
                assert globe_frame.get_by_text("点选模式：点击地图选择一点", exact=True).is_visible()
                assert "active" in (globe_frame.locator("#aoiBtnClick").get_attribute("class") or "")
                page.mouse.click(canvas_box["x"] + canvas_box["width"] * 0.48,
                                 canvas_box["y"] + canvas_box["height"] * 0.48)
                globe_frame.get_by_text("AOI 已选定，已同步", exact=True).wait_for(
                    state="visible", timeout=10000
                )

                # 用真实 canvas 鼠标验证矩形 AOI：单击不得提交，只有超过最小距离的
                # 按下/拖拽/松开才提交。坐标取点由 Cesium camera.pickEllipsoid 完成。
                globe_frame.get_by_role("button", name="清除").click()
                globe_frame.get_by_role("button", name="矩形").click()
                assert globe_frame.get_by_text("矩形模式：按住鼠标拖拽框选", exact=True).is_visible()
                page.mouse.click(canvas_box["x"] + canvas_box["width"] * 0.45,
                                 canvas_box["y"] + canvas_box["height"] * 0.45)
                assert globe_frame.get_by_text("矩形模式：请按住鼠标拖拽框选", exact=True).is_visible()
                assert "active" in (globe_frame.locator("#aoiBtnRect").get_attribute("class") or "")
                page.mouse.move(canvas_box["x"] + canvas_box["width"] * 0.35,
                                canvas_box["y"] + canvas_box["height"] * 0.35)
                page.mouse.down()
                page.mouse.move(canvas_box["x"] + canvas_box["width"] * 0.55,
                                canvas_box["y"] + canvas_box["height"] * 0.55)
                page.mouse.up()
                globe_frame.get_by_text("AOI 已选定，已同步", exact=True).wait_for(
                    state="visible", timeout=10000
                )
                assert "active" not in (globe_frame.locator("#aoiBtnRect").get_attribute("class") or "")
                globe_frame.get_by_role("button", name="多边形").click()
                polygon_points = (
                    (canvas_box["x"] + canvas_box["width"] * 0.38,
                     canvas_box["y"] + canvas_box["height"] * 0.38),
                    (canvas_box["x"] + canvas_box["width"] * 0.56,
                     canvas_box["y"] + canvas_box["height"] * 0.40),
                    (canvas_box["x"] + canvas_box["width"] * 0.48,
                     canvas_box["y"] + canvas_box["height"] * 0.58),
                )
                for point in polygon_points:
                    page.mouse.click(point[0], point[1])
                page.mouse.click(polygon_points[0][0], polygon_points[0][1], button="right")
                globe_frame.get_by_text("AOI 已选定，已同步", exact=True).wait_for(
                    state="visible", timeout=10000
                )

                fly_message = {
                    "type": "CSTF_FLY",
                    "version": 1,
                    "command_id": "browser-fly-check",
                    "lon": 120.8,
                    "lat": 30.5,
                    "height": 280000,
                    "duration": 0,
                    "label": "浏览器地图验收",
                    "source": "browser",
                }
                page.evaluate(
                    """(message) => {
                        const iframe = [...document.querySelectorAll('iframe')]
                          .find(el => (el.getAttribute('src') || '').includes(':8765/globe'));
                        if (!iframe) throw new Error('globe iframe missing');
                        const origin = new URL(iframe.src, window.location.href).origin;
                        iframe.contentWindow.postMessage(message, origin);
                    }""",
                    fly_message,
                )
                globe_frame.get_by_text("已定位 浏览器地图验收", exact=False).wait_for(
                    state="visible", timeout=10000
                )
                globe_frame.get_by_role("button", name="清除").click()
                globe_frame.get_by_text("AOI 已清除", exact=True).wait_for(
                    state="visible", timeout=10000
                )
                # Agent 面板默认停在「对话」视图；会话操作按钮位于「历史」视图。
                # 浮动顶栏可能遮挡 radio 命中区，用原生 click 切换视图。
                page.get_by_role("radio", name="历史").evaluate("el => el.click()")
                page.get_by_role("button", name="新会话").wait_for(state="visible", timeout=10000)
                page.get_by_role("button", name="新会话").click()
                # 「新会话」点击后 rerun 会切回「对话」视图；等 rerun 完成
                #（对话 radio 重新选中）再进入「历史」，避免点击落在旧 DOM 上。
                page.wait_for_function(
                    """() => {
                        const group = document.querySelector('[aria-label="Agent 面板"]');
                        if (!group) return false;
                        const inputs = [...group.querySelectorAll('input[type="radio"]')];
                        return inputs.length > 0 && !!inputs[0] && inputs[0].checked;
                    }""",
                    timeout=10000,
                )
                page.get_by_role("radio", name="历史").evaluate("el => el.click()")
                page.get_by_role("button", name="清空会话").wait_for(state="visible", timeout=10000)
                # 会话已创建后按钮才解除禁用；等 disabled 属性解除再点击。
                page.wait_for_function(
                    """() => {
                        const btns = [...document.querySelectorAll('button')];
                        const el = btns.find(b => (b.textContent || '').includes('清空会话'));
                        return !!el && !el.disabled;
                    }""",
                    timeout=10000,
                )
                page.get_by_role("button", name="清空会话").click()
                page.get_by_role("textbox", name="chat_input").wait_for(state="visible", timeout=10000)
                page.get_by_role("button", name="开始模型提取").click()
                page.get_by_text("潮滩智能提取计划", exact=False).wait_for(state="visible", timeout=15000)
                confirm_button = page.get_by_role("button", name="确认执行提取")
                confirm_button.wait_for(state="visible", timeout=15000)
                assert confirm_button.is_visible()
                page.get_by_role("button", name="取消计划", exact=True).click()

                # 侧栏切换到指数法后仍沿用“计划 → 确认”门闩。
                page.get_by_text("提取参数", exact=True).click()
                page.get_by_text("指数法", exact=True).last.click()
                page.get_by_role("button", name="开始指数法提取").click()
                page.get_by_text("指数法计划暂不可执行", exact=True).wait_for(state="visible", timeout=15000)
                assert page.get_by_role("button", name="确认执行指数法").is_visible()
                page.get_by_role("button", name="取消指数法计划").click()

                # 进程恢复的时间线可以驱动报告入口，但生成仍需单独确认。
                page.get_by_text("任务进度", exact=False).first.click()
                page.get_by_role("button", name="📄 生成成果报告").click()
                page.get_by_text("已生成成果报告计划", exact=False).wait_for(state="visible", timeout=10000)
                assert page.get_by_role("button", name="确认生成成果报告").is_visible()
                assert page.get_by_role("button", name="取消报告计划").is_visible()
                page.get_by_role("button", name="取消报告计划").click()
                browser.close()
        except Exception as exc:
            if "Executable doesn't exist" in str(exc) or "executable doesn't exist" in str(exc).lower():
                pytest.skip("Playwright Chromium executable is not installed")
            raise
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=10)


@pytest.mark.external
def test_gateway_login_logout_and_websocket_auth_in_browser():
    if os.environ.get("RUN_EXTERNAL_ACCEPTANCE") != "1" or os.environ.get("RUN_BROWSER_ACCEPTANCE") != "1":
        pytest.skip("browser acceptance is opt-in")
    playwright = pytest.importorskip("playwright.sync_api")
    port = _free_port()
    public_url = f"http://127.0.0.1:{port}"
    env = os.environ.copy()
    env.update(
        {
            "CSTF_GATEWAY_HOST": "0.0.0.0",
            "CSTF_GATEWAY_PORT": str(port),
            "CSTF_PUBLIC_URL": public_url,
            "CSTF_GATEWAY_ACCESS_TOKEN": "browser-test-token",
            # No upstream is needed to verify the edge authentication boundary.
            "CSTF_STREAMLIT_UPSTREAM": "http://127.0.0.1:9",
            "CSTF_GLOBE_UPSTREAM": "http://127.0.0.1:9",
            "HTTP_PROXY": "",
            "HTTPS_PROXY": "",
            "ALL_PROXY": "",
        }
    )
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "cstf_gateway:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--log-level",
            "warning",
        ],
        cwd=str(ROOT / "TF-agent"),
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        with playwright.sync_playwright() as browser_api:
            browser = browser_api.chromium.launch(headless=True, args=["--no-proxy-server"])
            page = browser.new_page()
            deadline = time.monotonic() + 20
            while time.monotonic() < deadline:
                try:
                    page.goto(f"{public_url}/__auth/login", wait_until="domcontentloaded", timeout=3000)
                    break
                except Exception:
                    time.sleep(0.25)
            else:
                pytest.fail("Gateway login page did not become ready")

            page.get_by_label("Access token").fill("wrong-token")
            page.get_by_role("button", name="Sign in").click()
            page.locator("#status").wait_for(state="visible", timeout=5000)
            assert page.locator("#status").inner_text() == "登录失败"

            page.get_by_label("Access token").fill("browser-test-token")
            page.get_by_role("button", name="Sign in").click()
            page.wait_for_url(f"{public_url}/", timeout=10000)
            session = page.evaluate(
                """async () => {
                    const r = await fetch('/__auth/session');
                    return {status: r.status, body: await r.json()};
                }"""
            )
            assert session["status"] == 200
            csrf = session["body"]["csrf_token"]
            logout = page.evaluate(
                """async (csrf) => {
                    const r = await fetch('/__auth/logout', {
                        method: 'POST',
                        headers: {'X-CSTF-CSRF': csrf},
                    });
                    return {status: r.status, body: await r.json()};
                }""",
                csrf,
            )
            assert logout["status"] == 200
            assert logout["body"]["logged_out"] is True
            close_code = page.evaluate(
                """() => new Promise(resolve => {
                    const ws = new WebSocket(`ws://${location.host}/stream`);
                    const timer = setTimeout(() => { try { ws.close(); } finally { resolve(-1); } }, 5000);
                    ws.onclose = event => { clearTimeout(timer); resolve(event.code); };
                    ws.onerror = () => {};
                })"""
            )
            # Browsers expose a pre-accept server close as 1006; Starlette's
            # in-process client observes the configured 1008 policy code.
            assert close_code in (1006, 1008)
            browser.close()
    except Exception as exc:
        if "Executable doesn't exist" in str(exc) or "executable doesn't exist" in str(exc).lower():
            pytest.skip("Playwright Chromium executable is not installed")
        raise
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)


@pytest.mark.external
def test_authenticated_gateway_proxies_real_local_streamlit_upstream():
    """Exercise the authenticated edge with a real local Streamlit upstream."""
    if os.environ.get("RUN_EXTERNAL_ACCEPTANCE") != "1" or os.environ.get("RUN_BROWSER_ACCEPTANCE") != "1":
        pytest.skip("browser acceptance is opt-in")
    playwright = pytest.importorskip("playwright.sync_api")
    streamlit_port = _free_port()
    gateway_port = _free_port()
    public_url = f"http://127.0.0.1:{gateway_port}"
    with tempfile.TemporaryDirectory(prefix="cstf-gateway-browser-") as temp_dir:
        env = os.environ.copy()
        for key in (
            "DASHSCOPE_API_KEY", "CSTF_LLM_API_KEY", "QWEN_API_KEY",
            "CSTF_LLM_BACKEND", "CSTF_LLM_MODEL", "CSTF_LLM_BASE_URL",
            "EE_PROJECT", "GOOGLE_CLOUD_PROJECT", "EARTHENGINE_PROJECT",
            "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "CSTF_ALLOW_RAW_SYSTEM_COMMAND",
        ):
            env[key] = ""
        env["CSTF_CONVERSATION_DB_PATH"] = str(Path(temp_dir) / "conversations.sqlite3")
        env["CSTF_JOB_DB_PATH"] = str(Path(temp_dir) / "jobs.sqlite3")
        env["CSTF_TIMELINE_LEDGER_PATH"] = str(Path(temp_dir) / "timeline_ledger.json")
        streamlit = subprocess.Popen(
            [
                sys.executable, "-m", "streamlit", "run", str(APP),
                "--server.headless", "true", "--server.address", "127.0.0.1",
                "--server.port", str(streamlit_port), "--server.fileWatcherType", "none",
            ],
            cwd=str(ROOT), env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        gateway_env = env.copy()
        gateway_env.update(
            {
                "CSTF_GATEWAY_HOST": "0.0.0.0",
                "CSTF_GATEWAY_PORT": str(gateway_port),
                "CSTF_PUBLIC_URL": public_url,
                "CSTF_GATEWAY_ACCESS_TOKEN": "browser-local-token",
                "CSTF_STREAMLIT_UPSTREAM": f"http://127.0.0.1:{streamlit_port}",
                "CSTF_GLOBE_UPSTREAM": "http://127.0.0.1:9",
            }
        )
        gateway = subprocess.Popen(
            [
                sys.executable, "-m", "uvicorn", "cstf_gateway:app",
                "--host", "127.0.0.1", "--port", str(gateway_port), "--log-level", "warning",
            ],
            cwd=str(ROOT / "TF-agent"), env=gateway_env,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        try:
            with playwright.sync_playwright() as browser_api:
                browser = browser_api.chromium.launch(headless=True, args=["--no-proxy-server"])
                page = browser.new_page()
                deadline = time.monotonic() + 40
                while time.monotonic() < deadline:
                    try:
                        page.goto(f"{public_url}/__auth/login", wait_until="domcontentloaded", timeout=3000)
                        break
                    except Exception:
                        time.sleep(0.25)
                else:
                    pytest.fail("Gateway login page did not become ready")
                page.get_by_label("Access token").fill("browser-local-token")
                page.get_by_role("button", name="Sign in").click()
                page.wait_for_url(f"{public_url}/", timeout=10000)
                page.get_by_text("智能分析助手", exact=False).first.wait_for(state="visible", timeout=20000)
                assert page.locator('[data-testid="stAppViewContainer"]').is_visible()
                health = page.evaluate(
                    """async () => { const r = await fetch('/_stcore/health'); return r.status; }"""
                )
                assert health == 200
                browser.close()
        except Exception as exc:
            if "Executable doesn't exist" in str(exc) or "executable doesn't exist" in str(exc).lower():
                pytest.skip("Playwright Chromium executable is not installed")
            raise
        finally:
            for process in (gateway, streamlit):
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=10)
