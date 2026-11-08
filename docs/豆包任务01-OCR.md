# 可直接转发给豆包：扫描PDF OCR接入

你在现有ResearchFlow项目中完成一个有边界的任务。先阅读README.md、docs/接口约定.md、backend/app/documents.py。不重建项目，不重写前端，不改模型配置。

项目绝对路径：`C:\Users\DELL\Documents\Codex\2026-11-09\wo\outputs\ResearchFlow`。

## 目标

给图像页接入本地开源OCR，让扫描版Transformer能被检索。优先评估PyMuPDF的Tesseract OCR接口；若本机条件不适合，说明原因再采用其他开源OCR。安装外部程序须明确告知，不默默下载大模型。没有OCR依赖时保留needs_ocr提示。

## 文件边界

- 可新增：`backend/app/ocr.py`、`backend/tests/test_ocr.py`、`docs/OCR安装说明.md`。
- 可最小修改：`backend/app/documents.py`、`backend/app/main.py`中的health字段、后端依赖清单；如必须加字段，先说明兼容迁移方案。
- 不修改：agent.py、retrieval.py、前端源码、现有API字段含义。
- 不删除、覆盖用户原始PDF；不把临时缓存和语言模型数据放进Git。

## 实现要求

1. 仅对无可靠原生文本且存在图像的页面OCR；原生文字页保持现有提取流程。
2. 输出文本块及bbox，保持1-based页码和现有chunks接口。OCR后能进入BM25检索。
3. 遵守MuPDF单工作线程限制；避免一次渲染整本书，限制分辨率和页面处理时间。
4. OCR失败要保存可解释状态，不能把空文本标为解析成功。已有needs_ocr文档应有显式重新解析入口，不依赖删除重传。
5. 提供确实可执行的安装与验证命令，更新开源来源及许可证记录。

## 验收

- 数据：`C:\Users\DELL\Doubao\chats\2026-11-09\new-chat\pdf\1706.03762v1_img.pdf`，不得读取旁边.md文件或其他原生PDF补答案。
- 能识别标题、检索到八块GPU训练相关段落，并回到正确PDF物理页。
- 原生v1/v7及541页教材解析不回退，现有测试仍通过。
- 卸载/禁用OCR依赖后，服务仍能启动并明确提示不可用。

完成后只汇报：改动文件、选型与许可证、验证命令和实际结果、耗时、已知限制。不要声称与官方扫描件等价或达到100%识别率。
