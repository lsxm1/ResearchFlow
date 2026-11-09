# ResearchFlow · 科研证据工作台

ResearchFlow 面向传智杯「AI大模型创新应用挑战赛」，参赛目标是覆盖文献检索与导入、解析与精读、溯源问答、引用图谱、综述、写作和绘图的全流程科研 Agent。当前 v0.1 已实现本地文献与证据检索基础。六模块的完成状态、缺项和提交前验收见[赛题范围与验收](docs/模块计划.md)。

## 当前能做什么

- 浏览器批量上传PDF，单文件默认50MB；同一文件改名再次上传不会重复存储。
- 后台逐页解析，持久保存页码、文本块、原文坐标与进度。服务重启后重新排队未完成解析。
- 支持英文扫描件本地 OCR。识别文本可检索，并保留原始 PDF 页码。已入库的扫描件可重新识别。
- 根据正文词组指纹提示相似文献；从PDF正文提取arXiv版本信息，给出保留建议。该算法是词面相似基线，尚非语义向量去重。
- 多选文献时按篇检索并展示证据。片段保留原文、页码和块坐标。中文提供少量科研词汇映射。
- 展示真实执行记录、保存任务，点击证据跳转原始PDF页码。
- 可配置兼容 `/chat/completions` 的本地或远端模型：一次受限补充检索、基于证据生成结论、验证来源ID和原文引句。**未配置真实模型进行端到端验证**。

默认没有模型调用，只显示原文候选，不生成回答。候选不等于答案；精确引句校验也不能保证结论蕴含关系正确，仍需人工核对。

## 启动（Windows）

要求 Python 3.12、Node.js、npm。项目已在本机安装依赖；换电脑或依赖变化时再运行安装脚本。

在本目录打开PowerShell：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\setup.ps1
```

然后分别在两个终端运行：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\start-backend.ps1
```

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\start-frontend.ps1
```

浏览器打开 <http://127.0.0.1:5173>。接口文档 <http://127.0.0.1:8000/docs>。

换电脑时，运行 `powershell -NoProfile -ExecutionPolicy Bypass -File .\setup-ocr.ps1` 下载约 4 MB 英文识别数据。本机已准备好。使用自定义数据目录时，可用 `RF_TESSDATA_DIR` 指向包含 `eng.traineddata` 的目录。

两份服务只绑定本机。当前没有账户和公网访问保护，公网部署放在后续里程碑处理。

## 第一次体验

1. 上传v7，等待“可检索”；再上传改名后的v1，打开查看正文相似提示。
2. 勾选文献，输入 `What hardware and training schedule were used?`。
3. 点击“检索原文证据”，查看执行记录，点击“第 N 页”核对原文。
4. 上传扫描件，等待识别。已有扫描件显示“需要OCR”时，在原文阅读区点“重新进行 OCR”。检索后从带“OCR 识别片段”标记的结果打开对应页。
5. 上传541页教材，观察逐页进度，解析后跳转第541页或搜索关键词。

原始数据目录：`C:\Users\DELL\Doubao\chats\2026-11-09\new-chat\pdf`。
这是公开替代数据集；自行生成的扫描件不能代表官方扫描件难度。不从配套 `.md` 文件导入答案。

## 模型配置以后再决定

复制 `.env.example` 为 `.env`。默认 `RF_MODEL_MODE=disabled`，没有API费用。

将来无论是本地开源模型服务还是云端服务，只需配置：

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
frontend/src/      React工作台
docs/              模块计划、接口约定、豆包任务书、验收报告
data/              上传文件和SQLite（自动生成，不提交）
```

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s backend/tests -v
cd frontend
npm.cmd run build
```

后端精确依赖在 `backend/requirements-lock.txt`；前端精确依赖在 `frontend/package-lock.json`。
首次安装使用范围依赖，复现当前环境时后端使用 `pip install -r backend/requirements-lock.txt`，前端使用 `npm ci`。

## 参赛与开源说明

开发者：待填写参赛者姓名、学校、组别。AI辅助：Codex参与架构设计、基础实现与验证；其他Agent贡献按实际工作补充。

开源来源、许可证、当前限制及后续计划见 `THIRD_PARTY.md` 和 `docs/模块计划.md`。项目采用AGPL-3.0-or-later，许可证全文在 `LICENSE`。保留真实Git历史，不补造开发记录。
