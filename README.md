# ResearchFlow · 科研证据工作台

ResearchFlow 是面向传智杯「AI大模型创新应用挑战赛」的全流程科研 Agent 应用。用户提出研究问题，系统补齐论文、检索证据、生成对比，最后导出包含原文定位、引用关系与写作提纲的研究报告。[赛题范围与验收](docs/模块计划.md)

## 当前能做什么

- 浏览器批量上传PDF，单文件默认50MB；同一文件改名再次上传不会重复存储。
- 后台逐页解析，持久保存页码、文本块、原文坐标与进度。服务重启后重新排队未完成解析。
- 支持英文扫描件本地 OCR。识别文本可检索，并保留原始 PDF 页码。已入库的扫描件可重新识别。
- 根据正文词组指纹提示相似文献；从PDF正文提取arXiv版本信息，给出保留建议。该算法是词面相似基线，尚非语义向量去重。
- 多选文献时按篇检索并展示证据。片段保留原文、页码和块坐标。中文提供少量科研词汇映射。
- 对比任务从问题识别目标与子问题，支持 BERT 简称、arXiv 编号和完整标题。缺证据时补查。已支持的架构与训练目标结论附上双方原文，训练步数和耗时各保留自己的出处。
- 展示真实执行记录、保存任务，点击证据跳转原始PDF页码。
- 联动阅读支持目录、数字引用往返跳转、原版页面、版面结构候选、划选与按页或整篇翻译。译文可导出 Markdown。
- 主题追踪定期刷新 arXiv，结合本地文献标题、代码链接与赞踩反馈排序。
- 用参考文献原文生成局域引用图谱、PageRank、综述线索和后续研究线索。
- 按研究构想生成带引用的提纲和 Mermaid 图。CSV 支持数据表、柱状图、折线图、三维雷达图、图注和 Matplotlib 脚本。

默认关闭模型调用。训练硬件支持从原文提取明确写出的设备数量及适用范围；其余证据按字段展示，需按原页核对。

## 启动（Windows）

要求 Windows x64、Python 3.12、Node.js 和 npm。在项目目录打开 PowerShell：

```powershell
Set-Location -LiteralPath 'C:\Users\DELL\Documents\Codex\2026-11-09\wo\outputs\ResearchFlow'
powershell -NoProfile -ExecutionPolicy Bypass -File .\start.ps1 -LocalModel
```

浏览器打开 <http://127.0.0.1:8000>，接口文档在 <http://127.0.0.1:8000/docs>。终端保持打开，按 Ctrl+C 停止。换电脑时，把第一行路径改成解压后的项目目录。首次启动按锁文件安装前后端依赖。`-LocalModel` 使用通过校验的 llama.cpp 与 Qwen2.5-0.5B。当前机器已缓存，换电脑首次需联网下载约 423 MB。运行在 CPU，无需 API 密钥。省略 `-LocalModel` 时按 `.env` 配置模型，默认使用证据检索模式。

换电脑时，运行 `powershell -NoProfile -ExecutionPolicy Bypass -File .\setup-ocr.ps1` 下载约 4 MB 英文识别数据。本机已准备好。使用自定义数据目录时，可用 `RF_TESSDATA_DIR` 指向包含 `eng.traineddata` 的目录。

本地模式绑定回环地址。`docker compose up --build` 提供同源服务的容器部署配置。公开评审部署需配置 HTTPS 反代及模型服务，可设置 `RF_DEMO_PASSWORD` 并以 `reviewer` 登录。Docker 配置尚待在目标主机实构建和上线核验。

## 第一次体验

1. 上传 v7，等待「可检索」，再上传改名后的 v1。正文相似提示中可以查看对照原文、保留对照版本或用当前版本替换。另一份会进入「归档版本」，可以恢复，历史任务的原文链接仍可打开。
2. 勾选文献，输入 `What hardware and training schedule were used?`。
3. 点击“检索原文证据”，查看执行记录，点击“第 N 页”核对原文。
4. 上传扫描件，等待识别。已有扫描件显示“需要OCR”时，在原文阅读区点“重新进行 OCR”。检索后从带“OCR 识别片段”标记的结果打开对应页。
5. 上传541页教材，观察逐页进度，解析后跳转第541页或搜索关键词。
6. 只勾选 Transformer，展开「补齐论文并对比」，点「填入 BERT 架构示例」再运行。首次执行会连接 arXiv。查看跨论文结论，点击原文页码在联动阅读中标亮证据段落，再下载研究报告。

对比任务按用户问题选择支持的字段。启用本地模型后，模型生成检索词，并尝试从候选中摘录原句。硬件优先显示含设备数量和训练范围的原句，证据不足时保留候选。小模型适合流程验收，术语、复杂推理与长文翻译需要人工复核。每次 PDF 翻译最多 60 页。

原始数据目录：`C:\Users\DELL\Doubao\chats\2026-11-09\new-chat\pdf`。
这是公开替代数据集；自行生成的扫描件不能代表官方扫描件难度。不从配套 `.md` 文件导入答案。

## 使用其他模型

复制 `.env.example` 为 `.env`。默认 `RF_MODEL_MODE=disabled`，没有API费用。

连接其他本地模型服务或云端服务时，配置如下：

```dotenv
RF_MODEL_MODE=openai_compatible
RF_MODEL_BASE_URL=http://127.0.0.1:11434/v1
RF_MODEL_NAME=填写实际部署的模型名称
RF_MODEL_API_KEY=
```

上面的地址只是本地服务配置示例，不表示已安装或运行模型。接入前确认服务支持JSON文本输出和所需上下文。改完重启后端。前端不接收密钥；禁止将真实 `.env` 提交Git。

## 结构与验证

```text
backend/app/       配置、SQLite、PDF解析、检索、任务执行、HTTP API
backend/tests/     隔离数据的回归测试
backend/review_smoke.py  用真实PDF走查运行中的工作台
frontend/src/      React工作台
docs/              模块计划、接口约定、豆包任务书、验收报告
data/              上传文件和SQLite（自动生成，不提交）
.cache/local-model/ 本机的推理程序和模型权重（自动下载，不提交）
samples/           绘图用CSV样例
```

服务运行后，在另一个 PowerShell 进入项目目录，运行真实 PDF 验收。它会向当前服务的文献库导入样例并保存对比任务。

```powershell
.\.venv\Scripts\python.exe .\backend\review_smoke.py 'C:\Users\DELL\Doubao\chats\2026-11-09\new-chat\pdf\1706.03762v7.pdf'
```

只跑离线回归和前端编译时，使用下列命令。

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s backend/tests -v
npm.cmd --prefix frontend run build
```

后端精确依赖在 `backend/requirements-lock.txt`，前端精确依赖在 `frontend/package-lock.json`。本机实测结果见 [验收报告](docs/验收报告.md)。

## 参赛与开源说明

开发者：待填写参赛者姓名、学校、组别。AI辅助：Codex参与架构设计、基础实现与验证；其他Agent贡献按实际工作补充。

开源来源、许可证、当前限制及后续计划见 `THIRD_PARTY.md` 和 `docs/模块计划.md`。项目采用AGPL-3.0-or-later，许可证全文在 `LICENSE`。保留真实Git历史，不补造开发记录。
