# 开源复用记录

当前采用成熟库搭建，不直接复制或冒用某个完整产品。自定义部分主要为：证据数据模型、任务记录、正文版本提示、受限检索与引句校验的连接流程。

| 组件 | 来源 | 上游许可证 | 当前用途 |
|---|---|---|---|
| PyMuPDF | https://github.com/pymupdf/PyMuPDF | AGPL-3.0 / 商业双授权 | PDF文本块、坐标、文档元信息 |
| Tesseract 英文识别数据 | https://github.com/tesseract-ocr/tessdata_fast/tree/4.1.0 | Apache-2.0 | PyMuPDF 内置引擎使用的本地识别数据 |
| llama.cpp | https://github.com/ggml-org/llama.cpp/releases/tag/b12223 | MIT | 可选 Windows 本地推理服务 |
| Qwen2.5-0.5B Instruct 量化权重 | https://ollama.com/library/qwen2.5:0.5b | Apache-2.0 | 本地翻译、检索词生成和原文摘录 |
| FastAPI | https://github.com/fastapi/fastapi | MIT | HTTP API、类型校验、接口文档 |
| Uvicorn | https://github.com/Kludex/uvicorn | BSD-3-Clause | ASGI服务器 |
| HTTPX | https://github.com/encode/httpx | BSD-3-Clause | 可配置模型接口调用 |
| python-multipart | https://github.com/Kludex/python-multipart | Apache-2.0 | 文件上传 |
| python-dotenv | https://github.com/theskumar/python-dotenv | BSD-3-Clause | 本地环境配置 |
| React | https://github.com/facebook/react | MIT | 工作台界面 |
| Vite | https://github.com/vitejs/vite | MIT | 前端开发、构建 |
| TypeScript | https://github.com/microsoft/TypeScript | Apache-2.0 | 前端类型检查 |

鉴于采用PyMuPDF的开源授权，项目按AGPL-3.0-or-later提供源码，发布和网络部署时保留许可证及相应源码获取方式。依赖的实际许可证以各安装版本附带文件为准，完整依赖由锁文件记录。论文PDF拥有各自版权，不因代码许可证而变为AGPL；不默认打包进公开仓库。

## 已调研、尚未集成

- PDFMathTranslate：https://github.com/PDFMathTranslate/PDFMathTranslate 。参考其将正文翻译与公式、版式保护分开处理的思路。ResearchFlow 使用自身的原文页与译文对照实现。
- Docling：https://github.com/docling-project/docling 。可作为后续精细版面、表格和公式解析的候选。

- PDF.js：https://github.com/mozilla/pdf.js ，Apache-2.0。用于下一步原文高亮、标题导航、双向翻译。当前阅读器使用浏览器原生PDF显示。
- LangGraph：https://github.com/langchain-ai/langgraph ，MIT。候选编排框架。当前任务执行器为轻量自建流程，不能宣称已经接入LangGraph或多Agent。

引入代码时记录源仓库、版本/commit、修改内容，保留相应许可证。普通算法（BM25、词组指纹）并非本项目原创算法，创新点应围绕应用流程与证据交互描述。
