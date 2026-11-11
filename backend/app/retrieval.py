"""Dependency-light lexical BM25 baseline. Dense retrieval is a later milestone."""
import heapq
import json
import math
from collections import Counter
from contextlib import contextmanager
from contextvars import ContextVar
from .db import connect
from .documents import words

ALIASES = {
    '硬件': 'hardware gpu training', '显卡': 'gpu hardware', '训练': 'training',
    '超参数': 'hyperparameters adam beta optimizer', '参数': 'parameters',
    '注意力': 'attention', '模型': 'model', '结构': 'architecture',
    '批次': 'batch', '优化器': 'optimizer adam', '学习率': 'learning rate',
    '翻译': 'translation bleu', '数据集': 'dataset data', '编码器': 'encoder',
    'hardware': 'gpu nvidia processor accelerator', '硬件环境': 'gpu processor accelerator',
}
STOP = set('the a an is are was were of to and in for with on how what does do it this that used using use'.split())
FORMS = {'trained': 'train', 'training': 'train', 'trains': 'train', 'models': 'model',
         'gpus': 'gpu', 'parameters': 'parameter', 'hyperparameters': 'hyperparameter'}
CHUNK_SELECT = '''SELECT c.*, d.title, d.filename, p.source FROM chunks c JOIN documents d ON d.id=c.document_id
                  JOIN pages p ON p.document_id=c.document_id AND p.page=c.page'''
MAX_CACHED_CHUNKS = 4096
_task_chunks = ContextVar('task_chunks', default=None)


def terms(text):
    return [FORMS.get(token, token) for token in words(text)]


@contextmanager
def reuse_paper_chunks():
    """Reuse token counts only within the current comparison task."""
    scope = _task_chunks.set({})
    try:
        yield
    finally:
        _task_chunks.reset(scope)


def cached_paper_chunks(cache, doc_id):
    if doc_id not in cache:
        budget = MAX_CACHED_CHUNKS - sum(len(chunks) for chunks in cache.values() if chunks is not None)
        cache[doc_id] = None
        if budget:
            with connect() as con:
                count = con.execute('SELECT count(*) FROM (SELECT 1 FROM chunks WHERE document_id=? LIMIT ?)',
                                    (doc_id, budget + 1)).fetchone()[0]
                if count > budget:
                    return None
                rows = con.execute(CHUNK_SELECT + ' WHERE c.document_id IN (?)', (doc_id,)).fetchmany(budget + 1)
            if len(rows) <= budget:
                prepared = []
                for row in rows:
                    tokens = terms(row['text'])
                    prepared.append((row, Counter(tokens), len(tokens)))
                cache[doc_id] = prepared
    return cache[doc_id]


def retrieve(question: str, document_ids: list[str], limit: int = 6, *, max_page: int | None = None):
    expanded = question + ' ' + ' '.join(v for k, v in ALIASES.items() if k in question)
    query = set(terms(expanded)) - STOP
    if not query or not document_ids:
        return []
    cache = _task_chunks.get()
    if cache is not None and len(document_ids) == 1:
        prepared = cached_paper_chunks(cache, document_ids[0])
        if prepared is not None:
            def cached_rows():
                return (chunk for chunk in prepared if max_page is None or chunk[0]['page'] <= max_page)
            return rank_chunks(query, limit, cached_rows)

    placeholders = ','.join('?' for _ in document_ids)
    sql = CHUNK_SELECT + f' WHERE c.document_id IN ({placeholders})'
    params = list(document_ids)
    if max_page is not None:
        sql += ' AND c.page<=?'
        params.append(max_page)
    # Stream oversized papers and searches outside a comparison task.
    with connect() as con:
        def streamed_rows():
            for row in con.execute(sql, params):
                tokens = terms(row['text'])
                yield row, tokens, len(tokens)
        return rank_chunks(query, limit, streamed_rows)


def rank_chunks(query, limit, rows):
    df = Counter()
    total = lengths = 0
    for _, tokens, length in rows():
        total += 1
        lengths += length
        df.update(query.intersection(tokens))
    average = max(lengths / max(total, 1), 1)
    idfs = {token: math.log(1 + (total - df[token] + .5) / (df[token] + .5)) for token in query}
    heap = []
    serial = 0
    for row, tokens, length in rows():
        counts = tokens if isinstance(tokens, Counter) else Counter(tokens)
        score = 0.0
        for token in query.intersection(counts):
            frequency = counts[token]
            score += idfs[token] * frequency * 2.5 / (frequency + 1.5 * (.25 + .75 * length / average))
        # Favor explanatory passages over short section headings.
        if length < 8:
            score *= .3
        if score > 0:
            item = (score, serial, dict(row))
            serial += 1
            if len(heap) < limit * 4:
                heapq.heappush(heap, item)
            elif score > heap[0][0]:
                heapq.heapreplace(heap, item)
    results = []
    for score, _, row in sorted(heap, reverse=True):
        # Avoid returning overlapping splits of the same source block.
        if any(x['document_id'] == row['document_id'] and x['page'] == row['page'] and x['bbox'] == json.loads(row['bbox']) for x in results):
            continue
        results.append({'id': row['id'], 'document_id': row['document_id'], 'title': row['title'],
                        'filename': row['filename'], 'page': row['page'], 'text': row['text'],
                        'bbox': json.loads(row['bbox']), 'score': round(score, 3), 'source': row['source'],
                        'file_url': f"/api/documents/{row['document_id']}/file#page={row['page']}"})
        if len(results) >= limit:
            break
    return results
