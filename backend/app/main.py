import hashlib
import json
import logging
import sqlite3
from contextlib import asynccontextmanager
from pathlib import Path
from uuid import uuid4
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from . import agent, documents
from .config import MAX_UPLOAD, UPLOADS, model_config
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
    for row in pending:
        documents.PDF_POOL.submit(documents.parse_document, row['id'])
    yield


app = FastAPI(title='ResearchFlow API', version='0.1.0', lifespan=lifespan)


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
            'model_name': cfg['model'] if configured else None, 'ocr_available': False}


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
        with path.open('rb') as handle:
            if b'%PDF-' not in handle.read(1024):
                raise HTTPException(415, '请上传有效PDF文件。')
        # Path components are never used for storage; both Windows and Unix names are stripped.
        filename = Path((file.filename or 'paper.pdf').replace('\\', '/')).name[:200]
        try:
            with connect() as con:
                con.execute('INSERT INTO documents(id,filename,sha256,title) VALUES (?,?,?,?)',
                            (doc_id, filename, digest.hexdigest(), filename))
        except sqlite3.IntegrityError:
            path.unlink(missing_ok=True)
            with connect() as con:
                existing = con.execute('SELECT * FROM documents WHERE sha256=?', (digest.hexdigest(),)).fetchone()
            if existing is None:
                raise
            return {'document': document(existing), 'exact_duplicate': True}
        documents.PDF_POOL.submit(documents.parse_document, doc_id)
        return {'document': document(get_document(doc_id)), 'exact_duplicate': False}
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


@app.get('/api/documents/{doc_id}/file')
def pdf_file(doc_id: str):
    get_document(doc_id)
    path = UPLOADS / f'{doc_id}.pdf'
    if not path.is_file():
        raise HTTPException(404, '原始PDF文件缺失。')
    return FileResponse(path, media_type='application/pdf', filename='paper.pdf', content_disposition_type='inline')


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


@app.post('/api/tasks', status_code=202)
def create_task(body: TaskRequest):
    ids = list(dict.fromkeys(body.document_ids))
    for doc_id in ids:
        row = get_document(doc_id)
        if row['status'] not in ('ready', 'partial'):
            raise HTTPException(409, '选中文献尚未就绪，或需要OCR。')
    task_id = uuid4().hex
    with connect() as con:
        con.execute('INSERT INTO tasks(id,question,document_ids) VALUES (?,?,?)',
                    (task_id, body.question.strip(), json.dumps(ids)))
    agent.TASK_POOL.submit(agent.execute, task_id)
    return {'id': task_id, 'status': 'queued'}


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
