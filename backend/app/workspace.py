"""Web entry points for the remaining contest modules."""
import json
import re
from concurrent.futures import TimeoutError
from typing import Literal
from uuid import uuid4

import httpx
from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import PlainTextResponse, Response
from pydantic import BaseModel, Field

from . import agent, arxiv, authoring, discovery, library_tools
from .comparison import local_paper
from .db import connect
from .documents import PDF_POOL

router = APIRouter(prefix='/api', tags=['research workspace'])


def paper(doc_id):
    with connect() as con:
        row = con.execute('SELECT * FROM documents WHERE id=?', (doc_id,)).fetchone()
    if row is None:
        raise HTTPException(404, '文献不存在。')
    if row['status'] not in ('ready', 'partial', 'needs_ocr'):
        raise HTTPException(409, '文献还在处理，请等待解析。')
    return row


def check_page(doc_id, page):
    row = paper(doc_id)
    if not 1 <= page <= row['page_count']:
        raise HTTPException(404, '页码不存在。')
    return row


def pdf_operation(function, *args):
    future = PDF_POOL.submit(function, *args)
    try:
        return future.result(timeout=25)
    except TimeoutError as exc:
        future.cancel()
        raise HTTPException(503, 'PDF 处理队列繁忙，请稍后重试。') from exc
    except (RuntimeError, ValueError, OSError) as exc:
        raise HTTPException(422, 'PDF 页面读取失败。') from exc


@router.get('/read/{doc_id}')
def reading_index(doc_id: str):
    paper(doc_id)
    row, pages = library_tools.paper_and_pages(doc_id)
    result = pdf_operation(library_tools.inspect_pdf, doc_id)
    result['outline'] = result['outline'] or library_tools.outline_from_text(pages)
    result['references'] = library_tools.bibliography(pages)[:180]
    abstract = re.search(r'(?im)^\s*abstract\s*\n([\s\S]{1,3500}?)(?=\n\s*(?:1\s+)?Introduction\b|\Z)', pages[0]['text']) if pages else None
    result['abstract'] = abstract[1].strip() if abstract else ''
    result['title'] = row['title']
    return result


@router.get('/read/{doc_id}/pages/{page}')
def reading_page(doc_id: str, page: int, language: Literal['Chinese', 'English'] = 'Chinese',
                 evidence_id: str | None = Query(None, max_length=100)):
    check_page(doc_id, page)
    with connect() as con:
        row = con.execute('SELECT text,source FROM pages WHERE document_id=? AND page=?', (doc_id, page)).fetchone()
        anchor = con.execute('SELECT bbox FROM chunks WHERE id=? AND document_id=? AND page=?',
                             (evidence_id, doc_id, page)).fetchone() if evidence_id else None
        if anchor is None and evidence_id:
            anchor = con.execute('SELECT bbox FROM table_anchors WHERE id=? AND document_id=? AND page=?',
                                 (evidence_id, doc_id, page)).fetchone()
    text = row['text'] if row else ''
    highlight = pdf_operation(library_tools.page_anchor, doc_id, page, json.loads(anchor['bbox'])) if anchor else None
    return {'text': text, 'source': row['source'] if row else 'native',
            'highlight': highlight,
            'translation': authoring.get_translation(doc_id, page, text, language),
            'citations': list(dict.fromkeys(re.findall(r'\[[1-9][0-9]{0,3}\]', text)))[:40]}


@router.get('/read/{doc_id}/pages/{page}/image')
def page_image(doc_id: str, page: int):
    check_page(doc_id, page)
    return Response(pdf_operation(library_tools.render_page, doc_id, page), media_type='image/png',
                    headers={'Cache-Control': 'private, max-age=300'})


@router.get('/read/{doc_id}/pages/{page}/structure')
def page_structure(doc_id: str, page: int):
    check_page(doc_id, page)
    return pdf_operation(library_tools.inspect_pdf, doc_id, page)


class TranslationRange(BaseModel):
    start: int = Field(ge=1)
    end: int = Field(ge=1)
    language: Literal['Chinese', 'English'] = 'Chinese'


def require_model():
    if not authoring.model_enabled():
        raise HTTPException(409, '请先在后端配置可用的模型服务。')


@router.post('/read/{doc_id}/translate', status_code=202)
def queue_translation(doc_id: str, body: TranslationRange):
    row = check_page(doc_id, body.start)
    if body.start > body.end or body.end > row['page_count'] or body.end - body.start >= 60:
        raise HTTPException(422, '请选择有效页码范围，每次最多翻译 60 页。')
    require_model()
    task_id = uuid4().hex
    with connect() as con:
        con.execute('INSERT INTO tasks(id,question,document_ids) VALUES (?,?,?)',
                    (task_id, f'翻译第 {body.start}–{body.end} 页 · {row["title"]}', json.dumps([doc_id])))
    agent.TASK_POOL.submit(authoring.translate_pages, task_id, doc_id, body.start, body.end, body.language)
    return {'id': task_id, 'status': 'queued'}


@router.get('/read/{doc_id}/translation.md')
def export_translation(doc_id: str, language: Literal['Chinese', 'English'] = 'Chinese'):
    row, pages = library_tools.paper_and_pages(paper(doc_id)['id'])
    lines = [f'# {row["title"]}', '']
    for page in pages:
        translated = authoring.get_translation(doc_id, page['page'], page['text'], language)
        if translated is not None:
            lines.extend((f'## 第 {page["page"]} 页', translated, ''))
    if len(lines) == 2:
        raise HTTPException(404, '暂无可导出的译文。')
    return PlainTextResponse('\n'.join(lines), headers={'Content-Disposition': 'attachment; filename="translation.md"'})


class TextTranslation(BaseModel):
    text: str = Field(min_length=1, max_length=8000)
    language: Literal['Chinese', 'English']


@router.post('/translate')
def translate(body: TextTranslation):
    require_model()
    try:
        return {'translation': authoring.translate_text(body.text, body.language)}
    except ValueError as exc:
        raise HTTPException(502, str(exc)) from exc
    except (httpx.HTTPError, KeyError, TypeError) as exc:
        raise HTTPException(502, '翻译服务暂不可用。') from exc


def search_arxiv(query):
    try:
        with httpx.Client(timeout=httpx.Timeout(18, connect=8), trust_env=False) as client:
            return arxiv.search_papers(client, query)
    except (ValueError, httpx.HTTPError) as exc:
        raise HTTPException(502, 'arXiv 检索未完成，请检查关键词或稍后重试。') from exc


@router.get('/literature/search')
def literature_search(query: str = Query(min_length=1, max_length=240)):
    return search_arxiv(query)


class PaperImport(BaseModel):
    arxiv_id: str = Field(max_length=30)


@router.post('/literature/import', status_code=202)
def literature_import(body: PaperImport):
    identifier = arxiv.paper_id(body.arxiv_id)
    if identifier is None:
        raise HTTPException(422, '请输入有效 arXiv 编号。')
    existing = local_paper(identifier)
    if existing:
        return {'id': existing['id'], 'reused': True}
    try:
        with httpx.Client(timeout=httpx.Timeout(20, connect=8), trust_env=False) as client:
            return {'id': arxiv.download_pdf(client, identifier), 'reused': False}
    except (ValueError, httpx.HTTPError) as exc:
        raise HTTPException(502, 'arXiv PDF 下载未完成，请稍后重试。') from exc


class TopicRequest(BaseModel):
    query: str = Field(min_length=1, max_length=240)
    interval_minutes: int = Field(default=60, ge=15, le=1440)


def topic_row(topic_id):
    with connect() as con:
        row = con.execute('SELECT * FROM tracked_topics WHERE id=?', (topic_id,)).fetchone()
    if row is None:
        raise HTTPException(404, '追踪主题不存在。')
    return row


@router.get('/literature/topics')
def topics():
    with connect() as con:
        rows = con.execute('SELECT * FROM tracked_topics ORDER BY query').fetchall()
    return [discovery.topic_view(row) for row in rows]


@router.post('/literature/topics', status_code=201)
def create_topic(body: TopicRequest):
    query = body.query.strip()
    if not query:
        raise HTTPException(422, '请输入追踪主题。')
    with connect() as con:
        con.execute('''INSERT INTO tracked_topics(id,query,interval_minutes,next_at) VALUES (?,?,?,0)
                       ON CONFLICT(query) DO UPDATE SET interval_minutes=excluded.interval_minutes''',
                    (uuid4().hex, query, body.interval_minutes))
        row = con.execute('SELECT * FROM tracked_topics WHERE query=?', (query,)).fetchone()
    return discovery.topic_view(row)


@router.post('/literature/topics/{topic_id}/refresh')
def refresh_topic(topic_id: str):
    topic_row(topic_id)
    discovery.refresh_topic(topic_id, force=True)
    return discovery.topic_view(topic_row(topic_id))


class Feedback(BaseModel):
    paper_id: str = Field(max_length=30)
    rating: Literal[-1, 0, 1]


@router.post('/literature/topics/{topic_id}/feedback')
def topic_feedback(topic_id: str, body: Feedback):
    row = topic_row(topic_id)
    if not any(item['id'] == body.paper_id for item in json.loads(row['results'])):
        raise HTTPException(404, '该论文不在主题的最新结果中。')
    with connect() as con:
        con.execute('''INSERT INTO topic_feedback VALUES (?,?,?) ON CONFLICT(topic_id,paper_id)
                       DO UPDATE SET rating=excluded.rating''', (topic_id, body.paper_id, body.rating))
    return discovery.topic_view(row)


@router.delete('/literature/topics/{topic_id}', status_code=204)
def delete_topic(topic_id: str):
    with connect() as con:
        con.execute('DELETE FROM tracked_topics WHERE id=?', (topic_id,))


class Selection(BaseModel):
    document_ids: list[str] = Field(min_length=1, max_length=24)


@router.post('/graph')
def graph(body: Selection):
    for doc_id in body.document_ids:
        paper(doc_id)
    return library_tools.citation_graph(body.document_ids)


class OutlineRequest(Selection):
    idea: str = Field(min_length=2, max_length=2500)


@router.post('/write/outline')
def write_outline(body: OutlineRequest):
    return authoring.outline(body.idea.strip(), [paper(doc_id) for doc_id in dict.fromkeys(body.document_ids)])


@router.post('/write/chart')
async def chart(file: UploadFile = File(...), kind: Literal['bar', 'line', 'radar'] = Form('bar'),
                label: str = Form('', max_length=120), value: str = Form('', max_length=120)):
    try:
        raw = await file.read(1024 * 1024 + 1)
        if len(raw) > 1024 * 1024:
            raise HTTPException(413, 'CSV 文件应小于 1 MB。')
        return authoring.chart_data(raw, kind, label, value)
    except (ValueError, UnicodeError) as exc:
        raise HTTPException(422, str(exc)) from exc
    finally:
        await file.close()


def comparison_report(task, base_url):
    result = task['result']
    table = result['comparison']
    sources = {source['id']: source for source in result['evidence']}
    def citation(source, quote):
        page = ' '.join(quote.splitlines())
        ocr = ' · OCR' if source.get('source') == 'ocr' else ''
        table = source.get('table')
        cell = ''
        if table:
            heading = ' · '.join(filter(None, (table.get('model'), table['heading'])))
            cell = f" · 表格第 {table['row']} 行第 {table['column']} 列（{heading}）"
        return f"{page} [{source['title']} · 第 {source['page']} 页{cell}{ocr}]({base_url}{source['file_url']})"

    lines = ['# ResearchFlow 研究报告', '', '## 研究问题', task['question'], '', '## 论文',
             *[f"- {item['title']} · {item['filename']}" for item in table['papers']]]
    if table.get('findings'):
        lines.extend(('', '## 跨论文结论'))
        for item in table['findings']:
            lines.append(f"- **{item['label']}**。{item['text']}")
            for ref in item['citations']:
                if ref['source_id'] in sources:
                    lines.append('  - ' + citation(sources[ref['source_id']], ref['quote']))
    lines.extend(('', '## 分字段证据'))
    for row in table['rows']:
        lines.extend(('', f"### {row['label']}"))
        for column, cell in zip(table['papers'], row['cells'], strict=True):
            source = sources.get(cell.get('source_id'))
            lines.append(f"- **{column['title']}**")
            for excerpt in cell.get('excerpts', []):
                if excerpt['status'] == 'conflict':
                    lines.append(f"  - **{excerpt['label']} · 证据冲突，待核对**")
                    for alternative in excerpt['alternatives']:
                        cited = sources[alternative['source_id']]
                        ref = citation(cited, alternative['quote'])
                        lines.append(f"    - **{alternative['label']} {alternative['display']}** · {ref}")
                    lines.append('    - 检索词 · ' + ' / '.join(excerpt['searched_queries']))
                    continue
                cited = sources.get(excerpt.get('source_id'))
                value = citation(cited, excerpt['quote']) if cited else '未检得足够证据。'
                measurement = f" {excerpt['display']}" if excerpt.get('display') else ''
                lines.append(f"  - **{excerpt['label']}{measurement}** · {value}")
                lines.append('    - 检索词 · ' + ' / '.join(excerpt['searched_queries']))
            if source is None and not cell.get('excerpts'):
                lines.append('  - 未检得匹配片段。')
            elif source is not None:
                if cell.get('text'):
                    lines.append(f"  - 模型摘要：{cell['text']}")
                label = '训练设定原文' if cell['status'] == 'verified_excerpt' else '原文候选'
                lines.append(f"  - {label} · " + citation(source, cell['quote']))
            if cell.get('searched_queries'):
                lines.append('  - 检索词：' + '；'.join(cell['searched_queries']))
    rows = []
    for item in table['papers']:
        try:
            rows.append(paper(item['id']))
        except HTTPException as exc:
            if exc.status_code != 404:
                raise
    if rows:
        graph = library_tools.citation_graph([row['id'] for row in rows])
        lines.extend(('', '## 引用关系'))
        for edge in graph['edges']:
            titles = {item['id']: item['title'] for item in table['papers']}
            url = f"{base_url}/api/documents/{edge['source']}/file#page={edge['page']}"
            lines.append(f"- {titles[edge['source']]} → {titles[edge['target']]}。[参考文献原页]({url})")
        if not graph['edges']:
            lines.append('当前范围内未发现可核对的引用边。')
        lines.extend(('', '## 写作提纲', authoring.outline(task['question'], rows)['markdown'].replace('/api/documents/', base_url + '/api/documents/')))
    lines.extend(('', '## 执行记录', *[f"- {event['tool']} · {event['message']}" for event in task['events']]))
    return '\n'.join(lines) + '\n'
