"""Exercise the reviewer path against a running ResearchFlow instance."""
import argparse
import os
import re
import time
from pathlib import Path

import httpx


def verify(client: httpx.Client, fixture: Path, empty: bool, gpt_fixtures: Path | None = None,
           bert_fixture: Path | None = None) -> None:
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

    def import_pdf(path):
        with path.open('rb') as handle:
            doc = post('/api/documents', files={'file': (path.name, handle, 'application/pdf')})['document']
        parsed = wait(f"/api/documents/{doc['id']}")
        assert parsed['status'] in {'ready', 'partial'}, parsed
        return parsed

    base = import_pdf(fixture)['id']
    if bert_fixture:
        bert = import_pdf(bert_fixture)
        assert bert['arxiv_id'] == '1810.04805', 'Use the original BERT PDF with its arXiv identifier.'

    def compare(question, document_ids=None):
        job = post('/api/tasks/compare', json={'question': question, 'document_ids': document_ids or [base]})
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

    architecture, _, sources = compare('BERT 继承了 Transformer Encoder 的什么，又改了什么？')
    if bert_fixture:
        assert architecture['document_ids'] == [base, bert['id']]
        assert not any(event['tool'] == 'download_pdf' for event in architecture['events'])
    elif empty:
        assert any(event['tool'] == 'download_pdf' for event in architecture['events']), 'Expected automatic BERT import.'
    findings = architecture['result']['comparison']['findings']
    assert [item['label'] for item in findings] == ['架构的沿袭与差异', '训练目标的差异']
    for finding, pages in zip(findings, ((3, 3, 3), (5, 13)), strict=True):
        assert {sources[ref['source_id']]['document_id'] for ref in finding['citations']} == set(architecture['document_ids'])
        for ref, page in zip(finding['citations'], pages, strict=True):
            source = sources[ref['source_id']]
            path = check_cell({'document_id': source['document_id'], **ref}, sources, page, ())
            highlight = get(path + '?evidence_id=' + source['id']).json()['highlight']
            assert highlight and all(0 <= value <= 1 for value in highlight) and min(highlight[2:]) > 0
            other_doc = next(doc for doc in architecture['document_ids'] if doc != source['document_id'])
            assert get(f'/api/read/{other_doc}/pages/{page}?evidence_id={source["id"]}').json()['highlight'] is None
    report = get(f"/api/tasks/{architecture['id']}/report.md").text
    assert all(item in report for item in ('## 跨论文结论', '## 引用关系', '## 写作提纲', '#page=3', '#page=13'))

    _, schedule, sources = compare('比较 BERT 和当前论文的训练步数与训练耗时。')
    for cell, page, steps, duration in zip(schedule['schedule'], (7, 13), ('300,000 steps', '1,000,000 steps'),
                                            ('3.5 days', '4 days'), strict=True):
        assert cell['status'] == 'verified_excerpt'
        for item, expected in zip(cell['excerpts'], (steps, duration), strict=True):
            check_cell({'document_id': cell['document_id'], **item}, sources, page, (expected,))
    assert len({item['source_id'] for item in schedule['schedule'][1]['excerpts']}) == 2

    task, rows, sources = compare('比较 arXiv:1810.04805 和当前论文的训练硬件与训练能耗。')
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
    print(f"PASS homepage, import, grounded findings, multi-source evidence, highlights, report and PDF structure; "
          f"model={health['model_mode']}, architecture={architecture['id']}, hardware={task['id']}, data={data_task['id']}")

    if gpt_fixtures:
        ids = []
        for version in (1, 2, 3):
            doc = import_pdf(gpt_fixtures / f'GPT-{version}.pdf')
            assert doc['status'] == 'ready'
            ids.append(doc['id'])
        gpt_task, gpt_rows, gpt_sources = compare('比较 GPT-1、GPT-2、GPT-3 的参数量、上下文窗口和 LN 位置。', ids)
        assert gpt_task['document_ids'] == ids
        assert not any(event['tool'] == 'download_pdf' for event in gpt_task['events'])
        expected = {
            'parameters': (None, ('parameter_count', 1_542_000_000, (4,)), ('parameter_count', 175_000_000_000, (1, 8))),
            'context': (('training_sequence', 512, (5,)), ('context_window', 1024, (4,)), ('context_window', 2048, (6, 8))),
            'layer_norm': (None, ('pre_norm_final', None, (4,)), ('pre_norm', None, (8,))),
        }
        for key, cases in expected.items():
            for doc_id, cell, case in zip(ids, gpt_rows[key], cases, strict=True):
                item, = cell['excerpts']
                if case is None:
                    assert item['status'] == 'missing' and len(item['searched_queries']) == (3 if key == 'parameters' else 2)
                    continue
                kind, value, pages = case
                assert item['kind'] == kind and item.get('value') == value, item
                page = gpt_sources[item['source_id']]['page']
                assert page in pages
                check_cell({'document_id': doc_id, **item}, gpt_sources, page, ())
        exact = gpt_rows['parameters'][1]['excerpts'][0]
        table_source = gpt_sources[exact['source_id']]
        assert exact['quote'] == '1542M' and table_source['table']['heading'] == 'Parameters'
        assert (table_source['table']['row'], table_source['table']['column']) == (5, 1)
        anchor = get(f"/api/read/{ids[1]}/pages/4?evidence_id={table_source['id']}").json()['highlight']
        assert anchor and 0 < anchor[2] < .15 and 0 < anchor[3] < .04, anchor
        assert get(f"/api/read/{ids[1]}/pages/3?evidence_id={table_source['id']}").json()['highlight'] is None
        findings = gpt_task['result']['comparison']['findings']
        assert len(findings) == 3
        assert {gpt_sources[ref['source_id']]['document_id'] for ref in findings[1]['citations']} == set(ids)
        assert '1,024 tokens' in findings[1]['text'] and '训练序列长度' in findings[1]['text']
        gpt_report = get(f"/api/tasks/{gpt_task['id']}/report.md").text
        assert re.search(r'175(?:\.0)?\s*B', gpt_report)
        assert all(term in gpt_report for term in ('1542M', '表格第 5 行第 1 列', '512 tokens', '未检得可直接核对'))
        print(f"PASS GPT three-paper comparison; task={gpt_task['id']}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('transformer_pdf', type=Path, help='Path to the original 1706.03762v7 PDF')
    parser.add_argument('--url', default='http://127.0.0.1:8000')
    parser.add_argument('--expect-empty', action='store_true')
    parser.add_argument('--bert-fixture', type=Path, help='Optional original 1810.04805 PDF to verify without an arXiv download')
    parser.add_argument('--gpt-fixtures', type=Path, help='Optional directory with GPT-1.pdf, GPT-2.pdf and GPT-3.pdf')
    args = parser.parse_args()
    password = os.getenv('RF_REVIEW_PASSWORD')
    with httpx.Client(base_url=args.url, timeout=30, trust_env=False,
                      auth=('reviewer', password) if password else None) as http:
        verify(http, args.transformer_pdf, args.expect_empty, args.gpt_fixtures, args.bert_fixture)
