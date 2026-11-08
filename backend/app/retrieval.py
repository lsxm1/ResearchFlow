"""Dependency-light lexical BM25 baseline. Dense retrieval is a later milestone."""
import heapq
import json
import math
from collections import Counter
from .db import connect
from .documents import words

ALIASES = {
    '硬件': 'hardware gpu training', '显卡': 'gpu hardware', '训练': 'training',
    '超参数': 'hyperparameters adam beta optimizer', '参数': 'parameters',
    '注意力': 'attention', '模型': 'model', '结构': 'architecture',
    '批次': 'batch', '优化器': 'optimizer adam', '学习率': 'learning rate',
    '翻译': 'translation bleu', '数据集': 'dataset data', '编码器': 'encoder',
    'hardware': 'gpu machine device', '硬件环境': 'gpu machine device',
}
STOP = set('the a an is are was were of to and in for with on how what does do it this that'.split())
FORMS = {'trained': 'train', 'training': 'train', 'trains': 'train', 'models': 'model',
         'gpus': 'gpu', 'parameters': 'parameter', 'hyperparameters': 'hyperparameter'}


def terms(text):
    return [FORMS.get(token, token) for token in words(text)]


def retrieve(question: str, document_ids: list[str], limit: int = 6):
    expanded = question + ' ' + ' '.join(v for k, v in ALIASES.items() if k in question)
    query = set(terms(expanded)) - STOP
    if not query or not document_ids:
        return []
    placeholders = ','.join('?' for _ in document_ids)
    # First pass computes corpus statistics without loading all book chunks at once.
    df = Counter()
    total = lengths = 0
    with connect() as con:
        sql = f'''SELECT c.*, d.title, d.filename FROM chunks c JOIN documents d ON d.id=c.document_id
                  WHERE c.document_id IN ({placeholders})'''
        for row in con.execute(sql, document_ids):
            tokens = terms(row['text'])
            total += 1
            lengths += len(tokens)
            df.update(query.intersection(tokens))
        average = lengths / max(total, 1)
        heap = []
        serial = 0
        for row in con.execute(sql, document_ids):
            tokens = terms(row['text'])
            counts = Counter(tokens)
            score = 0.0
            for token in query.intersection(counts):
                idf = math.log(1 + (total - df[token] + .5) / (df[token] + .5))
                frequency = counts[token]
                score += idf * frequency * 2.5 / (frequency + 1.5 * (.25 + .75 * len(tokens) / max(average, 1)))
            # A section heading can match perfectly but cannot answer a detail question.
            # Keep it as a candidate while favoring actual explanatory evidence.
            if len(tokens) < 8:
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
                        'bbox': json.loads(row['bbox']), 'score': round(score, 3),
                        'file_url': f"/api/documents/{row['document_id']}/file#page={row['page']}"})
        if len(results) >= limit:
            break
    return results
