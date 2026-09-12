# TF-agent

潮滩遥感解译与分析平台（Streamlit + 深度学习推理 + 遥感后处理 + LLM Copilot）。

## 仓库结构

```text
TF-agent/                 # 主应用（Streamlit、推理引擎、智能体）
research/jb/              # 研究原型脚本（M1–M5、E1 CLI 版）
tests/                    # E2E / UX 压力 / 单元测试
docs/                     # 文档与 QA 报告
cstf_ux.py                # 共享地学防御工具库
config/                   # 配置样例（ngrok 等）
```

## 快速开始

```powershell
conda activate tf-agent
cd TF-agent
python -m pip install -r requirements.txt
python -m pip install -r requirements-test.txt
cp .env.example .env    # 填入 DASHSCOPE_API_KEY
python -m streamlit run app.py --server.port 8501
```

### 同门 / 新机器拉取运行

1. 克隆仓库：`git clone https://github.com/KD-CHL/TF-agent.git && cd TF-agent`
2. 创建 Python 3.10 或 3.11 环境，并安装运行与测试依赖：`python -m pip install -r TF-agent/requirements.txt && python -m pip install -r TF-agent/requirements-test.txt`
3. 配置密钥：`copy TF-agent\.env.example TF-agent\.env`，填入 `DASHSCOPE_API_KEY`
4. 准备模型权重 `best_train_loss_model_resnet50.pth`（见下方说明），在侧栏「提取模型权重」中选择
5. 启动：`python -m streamlit run TF-agent/app.py --server.port 8501`

> 侧栏中的原始影像目录 / 输出目录 / 矢量约束等默认值为空，首次使用时按本机路径选择即可；
> 代码不会强制使用开发机的本地路径（`I:\GEE_data\20` 等仅为开发机示例，新机器自动留空）。

### 模型权重获取（必读）

`best_train_loss_model_resnet50.pth`（CDNet/ResNet50 潮滩分割权重）为推理必需文件，
因体积原因**不放入 Git 仓库**（`.gitignore` 已排除 `*.pth`）。获取方式：

- 方式一：找师兄/师姐拷贝该文件（约 200–400 MB），放到任意目录后在侧栏选择；
- 方式二：如仓库维护者已上传 GitHub Release，在 [Releases](https://github.com/KD-CHL/TF-agent/releases) 下载；
- 方式三：自行训练（`TF-agent/train_agent.py` 提供了训练入口）。

> 其它外部数据（AOI 矢量、潮滩数据集、M5 基线 SHP 等）同样按需准备，在侧栏对应输入框选择。


## 运行测试

```bash
conda activate tf-agent
python -m pytest \
  tests/smoke/test_app_boot.py \
  tests/smoke/test_streamlit_apptest.py \
  tests/unit/test_agent_commands.py \
  tests/unit/test_p0_hardening.py \
  tests/unit/test_workflow_orchestrator.py \
  tests/unit/test_agent_task_timeline.py \
  tests/unit/test_gateway_auth.py \
  tests/unit/test_local_api_auth.py \
  -q --tb=short -p no:cacheprovider
```

该核心命令只运行离线单元测试（包含 Gateway/本地 API 认证边界），不需要 DashScope、GEE、GPU 或外部数据。当前本地 Python 3.10 环境核心命令为 `204 passed`，单元测试命令 `python -m pytest tests/unit -q --tb=short -p no:cacheprovider` 为 `592 passed`；仓库全量 `python -m pytest -q` 为 `598 passed, 3 skipped`（仅保留第三方依赖 warning）。本轮受控矩阵已完成 DashScope 1 次/32 tokens、Playwright 根页面/计划门闩、侧栏深度学习/指数法计划、AOI 当前视图/矩形/多边形/清除及本地 CSTF_FLY 定位、Gateway 登录/登出/WebSocket 边界，以及认证 Gateway 到本地 Streamlit 上游的实际浏览器验收；GEE/GPU 因缺少 project 权限或真实权重保持 `SKIPPED`，远程 CI、Windows、公网地图、GEE/GPU 真实链路仍按 `docs/dev/AGENT_TECH_DEBT_TASKS.md` 单独记录，不以离线测试代替。

知识库可用 `python TF-agent/scripts/build_knowledge_base.py docs.jsonl --dry-run` 做离线输入校验，再通过 `CHROMA_RS_DB_PATH` 执行幂等入库；重型任务账本默认写入 `TF-agent/data/jobs.sqlite3`，进程重启会将未完成任务标记为 `INTERRUPTED`。

聊天上传的 PNG 预览只作为短期 UI 临时文件保存，新会话会按 7 天/200 个文件上限清理；可用 `CSTF_CHAT_PREVIEW_DIR` 指定缓存目录。JobStore 在事务锁内按活动 `plan_id` 去重，避免同一计划跨进程重复启动，终态计划仍可显式重跑。

浏览器验收是可选依赖：`python -m pip install -r TF-agent/requirements-browser.txt && python -m playwright install chromium`，然后按 [验收矩阵说明](tests/acceptance/README.md) 显式设置 `RUN_EXTERNAL_ACCEPTANCE=1` 和 `RUN_BROWSER_ACCEPTANCE=1`。

## 模块映射

| 编号 | 功能 | 研究脚本 (`research/jb/`) | 产品引擎 (`TF-agent/`) |
|------|------|---------------------------|------------------------|
| M4 | GEE 影像下载 | `M4.py` | `m4_engine.py` |
| M5 | 时空异常告警 | `M5.py` | `m5_engine.py` |
| E1 | 多源一致性诊断 | `E1.py` | `e1_engine.py` |

详细使用说明见 [TF-agent/README.md](TF-agent/README.md)。

## 远程仓库

```text
https://github.com/KD-CHL/TF-agent.git
```

## 文档

- [远程演示配置](TF-agent/REMOTE_DEMO.md)
- [Agent 验收矩阵](tests/acceptance/README.md)
- [QA 工作报告](docs/reports/导师汇报_潮滩系统测试与质量保障工作汇报.md)
