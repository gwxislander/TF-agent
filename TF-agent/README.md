# TF-agent（主应用）

基于 `Streamlit + 深度学习推理 + 遥感后处理 + LLM Copilot` 的潮滩遥感分析系统。  
支持批量 GeoTIFF 推理、时空后处理、地图可视化、智能体对话调度。

> 本目录为仓库主应用。研究原型脚本见上级目录 `research/jb/`。

---

## 1. 项目能力

- 批量推理遥感影像，输出单景掩膜 `*_mask.tif`
- 时空后处理融合，输出 `*_Final_p{prob}_c{cnt}.tif`
- 地图叠加展示与资产缓存管理
- 智能体对话（地图跳转、任务触发、知识库检索）
- TIFF 上传对话分析（含大文件与无效像素容错）

---

## 2. 项目结构

```text
TF-agent/
├─ app.py                 # Streamlit 主界面 + 调度 + 地图 + 聊天
├─ agent.py               # 智能体与多模态输入构建
├─ pre_engine.py          # 单景推理引擎
├─ post_engine.py         # 时空后处理与结果合成
├─ e1_engine.py           # 多源一致性诊断（封装 research/jb/E1.py）
├─ m5_engine.py           # 时空异常告警（封装 research/jb/M5.py）
├─ m4_engine.py           # GEE 影像下载
├─ YYnet.py               # CDNet 主模型定义
├─ backbone.py            # 主干网络
├─ modules.py             # 网络模块
├─ assets_registry.json   # 结果资产注册表（自动维护）
├─ requirements.txt
├─ scripts/               # 启动脚本（gateway、ngrok）
└─ .env                   # 本地密钥（勿提交）
```

---

## 3. 环境准备

推荐：

- Python `3.10`
- Conda 环境名：`tf-agent`

```powershell
conda activate tf-agent
cd TF-agent
python -m pip install -r requirements.txt
python -m pip install -r requirements-test.txt
```

---

## 4. `.env` 配置

在 `TF-agent/.env` 中配置（可复制 `.env.example`）：

```env
# 可选：百炼 Copilot Key；未配置时手动地图/任务仍可启动
DASHSCOPE_API_KEY=你的Key

# 模型
QWEN_CHAT_MODEL=qwen-vl-plus
QWEN_OPENAI_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1

# 可选：本地 OpenAI-compatible 后端（未声明 tools 时只提供纯文本问答）
# CSTF_LLM_BACKEND=local
# CSTF_LLM_BASE_URL=http://127.0.0.1:8000/v1
# CSTF_LLM_API_KEY=local

# TIFF 对话输入策略
YYNET_TIFF_MODE=auto
# 默认不发送精确空间元数据；仅在用户明确授权后由会话临时开启
YYNET_ATTACH_GEO_META=0
YYNET_TIFF_AUTO_PNG_MB=12
YYNET_VLM_MAX_SIDE=2048

# 知识库（可选；与 Agent 查询和维护 CLI 共用）
CHROMA_RS_DB_PATH=./rs_knowledge_db
CSTF_KB_EMBEDDING_MODEL=BAAI/bge-small-zh-v1.5

# 后台任务账本（进程重启会将未完成任务标记为 INTERRUPTED）
CSTF_JOB_DB_PATH=./data/jobs.sqlite3

# 代理（建议清空，避免 Connection error）
HTTP_PROXY=
HTTPS_PROXY=
ALL_PROXY=
GIT_HTTP_PROXY=
GIT_HTTPS_PROXY=
NO_PROXY=localhost,127.0.0.1,::1,dashscope.aliyuncs.com

# Windows OpenMP 兼容
KMP_DUPLICATE_LIB_OK=TRUE
```

### TIFF 策略说明

- `YYNET_TIFF_MODE=auto`：优先原样 TIFF，接口拒收时自动降级 PNG
- `YYNET_TIFF_MODE=native`：强制原样 TIFF（理论信息损失最小，但可能 400）
- `YYNET_TIFF_MODE=png`：始终转 PNG（兼容性最好）

---

### 矢量资源（内置）

仓库内 **`TF-agent/data/`** 已内置核心矢量资源，克隆后侧栏会自动指向这些路径：

| 文件 | 用途 |
|------|------|
| `data/china_costal.shp` | **全国省片 AOI 分区**，按 `name`（如 `zhejiang1`/`fujian1`）区分各省任务研究区 |
| `data/points_export.shp` | 指数法推理点位 |

> 侧栏默认值优先使用仓库内 `data/` 路径；若不存在则回退到开发机旧外部路径。同门克隆后无需手动配置这两个矢量。

**仍需自行准备的资源（体积大或按机器变化，未入库）**：

- **模型权重 `best_train_loss_model_resnet50.pth`**（CDNet/ResNet50，约 112MB）——推理必需，复制任意路径后，在侧栏「路径与模型环境」→「提取模型权重」选择；
- **水域约束 `max_water_extent23.shp`**（约 61MB，潮滩最大水域范围）——后处理约束，侧栏「路径与模型环境」选择；也可用 `research/jb/water-line/` 下的同名文件；
- **原始影像目录**（如 `I:\GEE_data\20`）与输出目录——按本机路径在侧栏配置；
- **官方潮滩成果矢量**（`china_tidal_flat_projected_YYYY.shp`，用于 M5/E1 基线）——需自行准备，见 `TF-agent/DATA/sqq_TF_20-25/` 示例。

---

## 5. 启动方式

```powershell
conda activate tf-agent
cd TF-agent
python -m streamlit run app.py --server.port 8501
```

访问：`http://localhost:8501`

---

## 6. 使用流程

1. 在侧边栏选择输入根目录（任务目录）。
2. 配置输出目录、模型权重、矢量约束（可选）。
3. 设置阈值：
   - `Probability (prob_th)`
   - `Absolute Count (min_cnt)`
4. 点击“运行模型”。
5. 在地图查看结果图层，或在聊天区让智能体解释。

---

## 7. 聊天与图片分析（当前实现）

- 用户与智能体消息样式已区分（便于识别）
- Agent Dock 默认进入“对话”视图；“历史”视图独立显示最近 8 条非空会话，可直接切换历史会话
- Agent Dock 支持“收起/展开”，并提供“对话区宽度”滑块，可在 24%–48% 间调整工作台右侧栏比例
- “新会话”和“清空会话”只在历史视图中显示，避免挤压当前对话
- 上传图片后输入框自动激活（无需再点）
- 发送后聊天记录会回显用户上传图片
- 仅上传图片不输入文本时，会自动使用默认提示词并给出提醒

默认提示词：

`请结合上传的遥感/地图影像进行专业解译，说明可能的地物、波段组合或异常现象。`

---

## 8. GeoTIFF 元信息与外部模型授权

默认不向外部模型发送精确空间范围、绝对路径或原始文件。当前对话 UI 不提供影像外发或精确空间元数据勾选项；两类授权均固定为关闭，避免历史会话状态意外扩大数据边界。未授权时只提供脱敏的本地上下文和文件名，不影响本地推理。

`YYNET_ATTACH_GEO_META=0` 仍是默认安全值；如果未来需要重新开放外部数据授权，应单独设计、评审并补充可撤销的授权入口。

会话历史也遵循同一边界：未授权时，重新拼入外部模型的历史消息会移除标注的 `bbox`、`centroid` 和 `map_center`；SQLite 持久化记录不会保存这些精确空间字段。该授权只对当前会话生效。

知识库检索返回的来源和正文在回填模型上下文前会再次移除本地路径、凭据和精确空间字段，并限制单条文档与总上下文长度；直接调用 `chat_with_vlm()` 也执行同样的文本边界检查。

若未来重新开放显式授权，允许字段清单将包括：

- `bands`
- `size`
- `dtype`
- `crs`
- `resolution`
- `nodata`
- `bounds`
- `compression`
- `tiled`
- `block_size`（有则显示）
- `finite_pixel_ratio`

---

## 9. 知识库维护（离线/幂等）

知识库输入采用 UTF-8 JSONL，每行至少包含 `document_id` 和 `content`，推荐同时提供 `source`、`title`、`published_at` 和 `checksum`；checksum 始终由正文 SHA-256 计算，若显式提供则必须与正文匹配：

```json
{"document_id":"paper-001","source":"论文或数据来源","title":"潮滩变化研究","published_at":"2024-01-01","content":"正文内容"}
```

先做不写盘的校验：

```bash
python scripts/build_knowledge_base.py docs.jsonl --dry-run
```

验收或多集合部署时可显式固定 embedding 模型和 collection，避免依赖当前 shell 环境：

```bash
python scripts/build_knowledge_base.py docs.jsonl --dry-run \
  --embedding-model local/bge-cache --collection remote_sensing_papers
```

实际构建或增量更新（同一 `document_id` 且 checksum 未变化时不会重复向量化）：

```bash
CHROMA_RS_DB_PATH=./rs_knowledge_db \
python scripts/build_knowledge_base.py docs.jsonl
```

首次非 dry-run 可能需要下载 `CSTF_KB_EMBEDDING_MODEL`。离线环境请预先将模型缓存到本机，并在无网验证时固定本地模型目录；CLI 不会在聊天请求中隐式写入知识库。manifest 位于知识库目录下，用于幂等更新和删除失效文档；发现损坏记录会保留 `.corrupt-*` 证据并停止增量写入。当前 `tf-agent` 环境已用临时 Chroma 库完成 1 条文档的真实 embedding 入库烟测。

---

## 10. 后台任务恢复

重型任务的 job 元数据写入 `CSTF_JOB_DB_PATH` 指定的 SQLite 账本。页面刷新不会丢失任务记录；若进程重启时仍有 `QUEUED`/`RUNNING` 任务，启动时会保留账本并将其标记为 `INTERRUPTED`，不会伪造成功或自动重复执行。请在时间线中核对输入和产物后重新确认重跑。

侧栏与 Agent 入口对推理、指数法、GEE、M5/E1、Workflow、AutoTune 和报告生成统一采用“计划 → 确认 → 执行 → 验证 → 登记 → 回复”语义。AutoTune 和报告按钮不会在首次点击时直接启动后台工作；确认前可取消，参数变化后旧计划会失效。报告只有在文件存在、非空校验通过且报告资产登记成功后才报告成功；GEE/M4 本地下载还要求每个请求场景实际生成非空 GeoTIFF，云端筛选成功但本地文件缺失时直接失败；M5/E1 输出校验失败时会明确标记为未完全通过，不登记或加载成果。

---

## 11. 常见问题（重点）

### 11.1 上传 TIFF 报 `Connection error` 或 400

常见原因：

- 代理污染（尤其是 `127.0.0.1:9`）
- 模型端不接受该 TIFF 编码
- 大 TIFF base64 后过大

建议：

1. 清空代理环境变量并重启终端。
2. 使用 `YYNET_TIFF_MODE=auto` 或 `png`。
3. 检查 `.env` 中 `DASHSCOPE_API_KEY` 与模型名。

### 11.2 为什么有些 TIFF 会被说“全黑/无效像素”

这通常是数据本身有效像素极少或全为 `NaN/Inf`。  
可看 `finite_pixel_ratio`：

- 接近 `0`：数据几乎不可解译
- 较大：可正常分析

### 11.3 VS Code 一直提示“同步更改”

这通常表示本地 `ahead` 远端。先看：

```powershell
git -C YYnet status --short --branch
```

如果显示 `ahead N`，直接推送即可：

```powershell
git -C YYnet push -u origin main
```

### 11.4 `push` 报 `Recv failure: Connection was reset`

优先排查代理：

```powershell
$env:HTTP_PROXY=""
$env:HTTPS_PROXY=""
$env:ALL_PROXY=""
git -c http.proxy= -c https.proxy= -C YYnet push -u origin main
```

---

## 12. 开发建议

- 将路径硬编码进一步收敛到 `.env` 或配置文件
- 为 `pre_engine/post_engine` 增加最小测试集
- `assets_registry.json` 已由 `asset_registry_schema.py` 做兼容历史格式的结构校验；读取会过滤坏记录，写入会拒绝非法字段类型。

---

## 13. 安全说明

- `.env` 含密钥，禁止提交到公共仓库
- 建议 `.gitignore` 包含：

```gitignore
.env
_chat_upload_tmp/
streamlit.out.log
streamlit.err.log
```
