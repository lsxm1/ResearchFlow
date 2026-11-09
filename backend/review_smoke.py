"""Exercise the reviewer path against a running ResearchFlow instance."""
import argparse
import os
import re
import time
from pathlib import Path

import httpx


def verify(client: httpx.Client, fixture: Path, empty: bool) -> None:
    def get(path):
        return client.get(path).raise_for_status()

    def post(path, **kwargs):
        return client.post(path, **kwargs).raise_for_status().json()

    def wait(path):
        deadline = time.monotonic() + 180
        while time.monotonic() < deadline:
            value = get(path).json()
            if value['status'] not in {'queued', 'parsing', 'running'}:
                return value
            time.sleep(.5)
        raise TimeoutError(f'Operation did not finish: {path}')

    health = get('/api/health').json()
    home = get('/').text
    assets = re.findall(r'(?:src|href)="(/assets/[^\"]+)"', home)
    assert 'ResearchFlow' in home and assets, 'Production frontend is not mounted.'
    for asset in assets:
        get(asset)
    if empty:
        assert get('/api/documents').json() == [], 'Expected an empty review library.'

    with fixture.open('rb') as handle:
        uploaded = post('/api/documents', files={'file': (fixture.name, handle, 'application/pdf')})
    base = uploaded['document']['id']
    assert wait(f'/api/documents/{base}')['status'] in {'ready', 'partial'}

    def compare(question):
        job = post('/api/tasks/compare', json={'question': question, 'document_ids': [base]})
        task = wait(f"/api/tasks/{job['id']}")
        assert task['status'] == 'completed', task.get('result')
        result = task['result']
        return task, {row['key']: row['cells'] for row in result['comparison']['rows']}, {
            source['id']: source for source in result['evidence']
        }

    def check_cell(cell, sources, page, facts):
        source = sources[cell['source_id']]
        assert cell['document_id'] == source['document_id'] and source['page'] == page, source
        assert cell['quote'] in source['text']
        quote = ' '.join(cell['quote'].split())
        assert all(fact in quote for fact in facts), quote
        path = f"/api/read/{source['document_id']}/pages/{page}"
        assert cell['quote'] in get(path).json()['text']
        return path

    task, rows, sources = compare('比较 arXiv:1810.04805 和当前论文的训练硬件与训练能耗。')
    if empty:
        assert any(event['tool'] == 'download_pdf' for event in task['events']), 'Expected automatic BERT import.'
    assert len(rows['hardware']) == 2
    expected = (
        (7, ('8 NVIDIA P100 GPUs',)),
        (13, ('BERTBASE', '4 Cloud TPUs', '16 TPU chips', 'BERTLARGE', '16 Cloud TPUs', '64 TPU chips')),
    )
    for cell, (page, facts) in zip(rows['hardware'], expected, strict=True):
        assert cell['status'] == 'verified_excerpt', cell
        path = check_cell(cell, sources, page, facts)
        assert get(path + '/image').content.startswith(b'\x89PNG')
    assert all(cell['status'] == 'missing' and len(cell['searched_queries']) == 2 for cell in rows['energy'])
    report = get(f"/api/tasks/{task['id']}/report.md").text
    assert all(item in report for item in ('## 引用关系', '## 写作提纲', '#page=', '64 TPU chips'))

    data_task, data_rows, data_sources = compare('比较 arXiv:1810.04805 和当前论文的训练数据。')
    transformer, bert = data_rows['data']
    check_cell(transformer, data_sources, 7, ('WMT 2014 English-German', '4.5 million'))
    check_cell(bert, data_sources, 5, ('BooksCorpus (800M words)', 'English Wikipedia (2,500M words)'))

    def structure(doc_id, page):
        return get(f'/api/read/{doc_id}/pages/{page}/structure').json()

    transformer_rows = [row for table in structure(base, 8)['tables'] for row in table['rows']]
    assert any({'Transformer (big)', '28.4', '41.8'} <= set(row) for row in transformer_rows)
    bert_tables = structure(bert['document_id'], 8)['tables']
    assert len(bert_tables) == 1 and len(bert_tables[0]['rows']) == 6
    assert ['No NSP', '83.9', '84.9', '86.5', '92.6', '87.9'] in bert_tables[0]['rows']
    assert any('warmup_steps' in formula['text'] for formula in structure(base, 7)['formula_candidates'])
    print(f"PASS homepage, import, evidence, report and PDF structure; model={health['model_mode']}, "
          f"hardware={task['id']}, data={data_task['id']}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('transformer_pdf', type=Path, help='Path to the original 1706.03762v7 PDF')
    parser.add_argument('--url', default='http://127.0.0.1:8000')
    parser.add_argument('--expect-empty', action='store_true')
    args = parser.parse_args()
    password = os.getenv('RF_REVIEW_PASSWORD')
    with httpx.Client(base_url=args.url, timeout=30, trust_env=False,
                      auth=('reviewer', password) if password else None) as http:
        verify(http, args.transformer_pdf, args.expect_empty)
