import asyncio
import base64
import hashlib
import hmac
import json
import logging
import os
from contextlib import asynccontextmanager, suppress
from uuid import uuid4
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from . import agent, comparison, discovery, documents, workspace
from .compare_plan import MAX_PAPERS
from .config import MAX_UPLOAD, ROOT, UPLOADS, model_config
from .db import connect, document, init_db, task

logging.basicConfig(level=logging.INFO)


@asynccontextmanager
async def lifespan(app):
    init_db()
    with connect() as con:
        # Tasks are not silently replayed after a crash (which could bill model calls twice).
        con.execute("UPDATE tasks SET status='failed', result=? WHERE status IN ('queued','running')",
                    (json.dumps({'mode':'error','claims':[], 'evidence':[], 'warnings':['服务重启中断了任务，请重新提交。']}, ensure_ascii=False),))
        pending = con.execute("SELECT id FROM documents WHERE status IN ('queued','parsing')").fetchall()
    documents.PDF_POOL.submit(documents.refresh_arxiv_titles)
    for row in pending:
        documents.PDF_POOL.submit(documents.parse_document, row['id'])
    scheduler = asyncio.create_task(discovery.run_scheduler())
    try:
        yield
    finally:
        scheduler.cancel()
        with suppress(asyncio.CancelledError):
            await scheduler


app = FastAPI(title='ResearchFlow API', version='0.1.0', lifespan=lifespan)
app.include_router(workspace.router)


@app.middleware('http')
async def review_access(request: Request, call_next):
    password = os.getenv('RF_DEMO_PASSWORD')
    if password:
        expected = b'Basic ' + base64.b64encode(f'reviewer:{password}'.encode())
        actual = request.headers.get('Authorization', '').encode('latin1')
        if not hmac.compare_digest(actual, expected):
            return PlainTextResponse('Reviewer sign-in required.', status_code=401,
                                     headers={'WWW-Authenticate': 'Basic realm="ResearchFlow", charset="UTF-8"'})
    return await call_next(request)


def get_document(doc_id):
    with connect() as con:
        row = con.execute('SELECT * FROM documents WHERE id=?', (doc_id,)).fetchone()
    if row is None:
        raise HTTPException(404, '文献不存在。')
    return row


@app.get('/api/health')
def health():
    cfg = model_config()
    configured = cfg['mode'] == 'openai_compatible' and bool(cfg['model'] and cfg['base_url'])
    return {'status': 'ok', 'version': '0.1.0', 'model_mode': 'configured' if configured else 'evidence_only',
            'model_name': cfg['model'] if configured else None, 'ocr_available': documents.ocr_available()}


@app.get('/api/documents')
def list_documents():
    with connect() as con:
        return [document(row) for row in con.execute('SELECT * FROM documents ORDER BY created_at DESC')]


@app.post('/api/documents', status_code=201)
async def upload(file: UploadFile = File(...)):
    doc_id = uuid4().hex
    path = UPLOADS / f'{doc_id}.pdf'
    digest = hashlib.sha256()
    size = 0
    try:
        with path.open('wb') as handle:
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                if size > MAX_UPLOAD:
                    raise HTTPException(413, f'文件超过{MAX_UPLOAD // (1024*1024)}MB限制。')
                digest.update(chunk)
                handle.write(chunk)
        try:
            row, duplicate = documents.register_pdf(path, file.filename or 'paper.pdf', digest.hexdigest())
        except ValueError as exc:
            raise HTTPException(415, str(exc)) from exc
        return {'document': document(row), 'exact_duplicate': duplicate}
    except Exception:
        path.unlink(missing_ok=True)
        raise
    finally:
        await file.close()


@app.get('/api/documents/{doc_id}')
def detail(doc_id: str):
    result = document(get_document(doc_id))
    result['duplicates'] = documents.duplicates(doc_id)
    return result


class VersionChoice(BaseModel):
    keep_id: str
    archive_id: str


@app.post('/api/documents/resolve-version')
def resolve_version(body: VersionChoice):
    if body.keep_id == body.archive_id:
        raise HTTPException(422, '请选择两个不同的文献版本。')
    with connect() as con:
        con.execute('BEGIN IMMEDIATE')
        pair = {row['id']: row for row in con.execute('SELECT * FROM documents WHERE id IN (?,?)',
                                                     (body.keep_id, body.archive_id))}
        if len(pair) != 2:
            raise HTTPException(404, '其中一份文献已移除。')
        kept, archived = pair[body.keep_id], pair[body.archive_id]
        if kept['archived'] or any(row['status'] not in ('ready', 'partial') for row in pair.values()):
            raise HTTPException(409, '文献状态已变化，请刷新后重新选择。')
        a, b = (set(json.loads(row['signature'])) for row in (kept, archived))
        if documents.body_similarity(a, b) < documents.DUPLICATE_THRESHOLD:
            raise HTTPException(409, '两份文献未达到正文相似阈值。')
        con.execute('UPDATE documents SET archived=1 WHERE id=?', (body.archive_id,))
    return {'kept_id': body.keep_id, 'archived_id': body.archive_id}


@app.post('/api/documents/{doc_id}/restore', status_code=204)
def restore_document(doc_id: str):
    with connect() as con:
        if not con.execute('UPDATE documents SET archived=0 WHERE id=?', (doc_id,)).rowcount:
            raise HTTPException(404, '文献不存在。')


@app.get('/api/documents/{doc_id}/file')
def pdf_file(doc_id: str):
    get_document(doc_id)
    path = UPLOADS / f'{doc_id}.pdf'
    if not path.is_file():
        raise HTTPException(404, '原始PDF文件缺失。')
    return FileResponse(path, media_type='application/pdf', filename='paper.pdf', content_disposition_type='inline')


@app.post('/api/documents/{doc_id}/retry-ocr', status_code=202)
def retry_ocr(doc_id: str):
    get_document(doc_id)
    if not documents.ocr_available():
        raise HTTPException(409, '请先运行 setup-ocr.ps1 准备英文OCR数据。')
    if not (UPLOADS / f'{doc_id}.pdf').is_file():
        raise HTTPException(404, '原始PDF文件缺失。')
    with connect() as con:
        changed = con.execute(
            "UPDATE documents SET status='queued', error=NULL WHERE id=? AND status IN ('needs_ocr','partial')",
            (doc_id,),
        ).rowcount
    if not changed:
        raise HTTPException(409, '只有仍有页面需要OCR的文献可以重新识别。')
    documents.PDF_POOL.submit(documents.parse_document, doc_id)
    return {'status': 'queued'}


@app.get('/api/documents/{doc_id}/pages/{page}')
def page_text(doc_id: str, page: int):
    row = get_document(doc_id)
    if page < 1 or page > row['page_count']:
        raise HTTPException(404, '页码不存在。')
    with connect() as con:
        result = con.execute('SELECT * FROM pages WHERE document_id=? AND page=?', (doc_id, page)).fetchone()
        if result is None:
            raise HTTPException(409, '该页尚未解析完成。')
        chunks = con.execute('SELECT id,text,bbox FROM chunks WHERE document_id=? AND page=?', (doc_id, page)).fetchall()
    return {**dict(result), 'blocks': [{**dict(c), 'bbox': json.loads(c['bbox'])} for c in chunks]}


@app.delete('/api/documents/{doc_id}', status_code=204)
def delete_document(doc_id: str):
    get_document(doc_id)
    with connect() as con:
        con.execute('DELETE FROM documents WHERE id=?', (doc_id,))
    # Windows may keep the file open while its current page finishes parsing.
    try:
        (UPLOADS / f'{doc_id}.pdf').unlink(missing_ok=True)
    except PermissionError:
        logging.warning('Deferred cleanup required for deleted PDF %s', doc_id)


class TaskRequest(BaseModel):
    question: str = Field(min_length=2, max_length=2000)
    document_ids: list[str] = Field(min_length=1, max_length=30)


class CompareRequest(TaskRequest):
    target_paper: str = Field(default='', max_length=240)


def enqueue_task(body: TaskRequest, runner, *args):
    ids = list(dict.fromkeys(body.document_ids))
    task_id = uuid4().hex
    with connect() as con:
        con.execute('BEGIN IMMEDIATE')
        for doc_id in ids:
            row = con.execute('SELECT status,archived FROM documents WHERE id=?', (doc_id,)).fetchone()
            if row is None:
                raise HTTPException(404, '文献不存在。')
            if row['archived']:
                raise HTTPException(409, '请先恢复归档版本，再将其加入新任务。')
            if row['status'] not in ('ready', 'partial'):
                raise HTTPException(409, '选中文献尚未就绪，或需要OCR。')
        con.execute('INSERT INTO tasks(id,question,document_ids) VALUES (?,?,?)',
                    (task_id, body.question.strip(), json.dumps(ids)))
    agent.TASK_POOL.submit(runner, task_id, *args)
    return {'id': task_id, 'status': 'queued'}


@app.post('/api/tasks', status_code=202)
def create_task(body: TaskRequest):
    return enqueue_task(body, agent.execute)


@app.post('/api/tasks/compare', status_code=202)
def create_comparison(body: CompareRequest):
    count = len(set(body.document_ids))
    if count > MAX_PAPERS:
        raise HTTPException(422, f'一次最多比较 {MAX_PAPERS} 篇已入库论文。')
    if count > 1 and body.target_paper.strip():
        raise HTTPException(422, '多篇对比直接使用勾选的论文，请清空目标论文栏。')
    if len(body.question.strip()) < 2:
        raise HTTPException(422, '请填写研究问题。')
    return enqueue_task(body, comparison.execute, body.target_paper.strip())


@app.get('/api/tasks')
def list_tasks():
    with connect() as con:
        return [task(row) for row in con.execute('SELECT * FROM tasks ORDER BY created_at DESC LIMIT 30')]


@app.get('/api/tasks/{task_id}')
def task_detail(task_id: str):
    with connect() as con:
        row = con.execute('SELECT * FROM tasks WHERE id=?', (task_id,)).fetchone()
    if row is None:
        raise HTTPException(404, '任务不存在。')
    return task(row)


@app.get('/api/tasks/{task_id}/report.md')
def task_report(task_id: str, request: Request):
    data = task_detail(task_id)
    if data['status'] != 'completed' or not data['result'].get('comparison'):
        raise HTTPException(409, '对比任务完成后可下载报告。')
    markdown = workspace.comparison_report(data, str(request.base_url).rstrip('/'))
    return PlainTextResponse(markdown, headers={'Content-Disposition': 'attachment; filename="researchflow-report.md"'})


frontend = ROOT / 'frontend' / 'dist'
if (frontend / 'index.html').is_file():
    app.mount('/', StaticFiles(directory=frontend, html=True), name='workbench')
