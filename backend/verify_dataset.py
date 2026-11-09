"""Read-only source fixtures -> local development API. Writes a truthful QA report."""
import json
import sys
import time
from pathlib import Path
import httpx

base = Path(sys.argv[1])
root = Path(__file__).resolve().parents[1]
records = []


def search(client, document_ids):
    response = client.post('/api/tasks', json={
        'question': 'What hardware and training schedule were used?',
        'document_ids': document_ids,
    })
    response.raise_for_status()
    task_id = response.json()['id']
    for _ in range(100):
        task = client.get(f'/api/tasks/{task_id}').json()
        if task['status'] not in ('queued', 'running'):
            break
        time.sleep(.1)
    assert task['status'] == 'completed' and task['result']['evidence']
    return task['result']['evidence']


with httpx.Client(base_url='http://127.0.0.1:8000', timeout=90, trust_env=False) as client:
    health = client.get('/api/health').json()
    assert health['model_mode'] == 'evidence_only', 'Disable model calls before running this verification.'
    ocr_available = health['ocr_available']
    for filename, upload_name in [
        ('1706.03762v7.pdf', 'newer-renamed.pdf'),
        ('1706.03762v1.pdf', 'older-renamed.pdf'),
        ('1706.03762v1_img.pdf', 'scan.pdf'),
        ('UnderstandingDeepLearning_02_09_26_C.pdf', 'book.pdf'),
    ]:
        started = time.monotonic()
        with (base/filename).open('rb') as file:
            response = client.post('/api/documents', files={'file':(upload_name,file,'application/pdf')})
        response.raise_for_status()
        cached = response.json()['exact_duplicate']
        uploaded = response.json()['document']
        doc_id = uploaded['id']
        if filename.endswith('_img.pdf') and ocr_available and uploaded['status'] in ('needs_ocr', 'partial'):
            client.post(f'/api/documents/{doc_id}/retry-ocr').raise_for_status()
        deadline = time.monotonic() + 300
        seen_progress = []
        while time.monotonic() < deadline:
            response = client.get(f'/api/documents/{doc_id}')
            response.raise_for_status()
            doc = response.json()
            seen_progress.append(doc['processed_pages'])
            if doc['status'] not in ('queued','parsing'):
                break
            time.sleep(.3)
        assert doc['status'] in ('ready','partial','needs_ocr'), doc
        records.append({'fixture':filename, 'cached':cached, 'id':doc_id, 'status':doc['status'], 'pages':doc['page_count'],
                        'text_pages':doc['text_pages'], 'ocr_pages':doc['ocr_pages'], 'version':doc['version'],
                        'seconds':round(time.monotonic()-started,2), 'progress_samples':sorted(set(seen_progress)),
                        'duplicates':doc['duplicates']})
        print(json.dumps(records[-1],ensure_ascii=False),flush=True)
    assert records[0]['pages'] == 15 and records[0]['version'] == 7
    assert records[1]['pages'] == 15 and records[1]['version'] == 1
    assert any(x['document_id']==records[0]['id'] for x in records[1]['duplicates']), 'Renamed versions were not matched by body'
    scan = records[2]
    scan_ready = scan['status'] == 'ready'
    if ocr_available or scan_ready:
        assert scan['status'] == 'ready' and scan['text_pages'] == 15
        scanned_sources = search(client, [scan['id']])
        assert any(s['source'] == 'ocr' and s['page'] == 7 and 'P100' in s['text'] for s in scanned_sources)
    else:
        assert scan['status'] == 'needs_ocr' and scan['text_pages'] == 0
    assert records[3]['pages'] == 541 and records[3]['text_pages'] > 500
    ids = [records[0]['id'], records[1]['id']]
    sources = search(client, ids)
    for doc_id in ids:
        assert any(s['document_id'] == doc_id and 'P100' in s['text'] for s in sources), 'Expected GPU evidence in each version'
    for source in sources:
        page=client.get(f"/api/documents/{source['document_id']}/pages/{source['page']}").json()
        assert source['text'] in page['text']
    last=client.get(f"/api/documents/{records[3]['id']}/pages/541")
    assert last.status_code==200 and 'Index' in last.json()['text']

report = '# 真实数据验收 · 2026-11-09\n\n'
report += '通过本机HTTP API上传原始PDF，v1/v7使用无版本含义的文件名。未读取配套Markdown；未配置真实模型。\n\n'
report += '| 数据 | 页数 | 文本页 | 需OCR页 | 状态 | 本次检查耗时（秒） | 复用已入库文件 |\n|---|---:|---:|---:|---|---:|---|\n'
for record in records:
    report += f"| {record['fixture']} | {record['pages']} | {record['text_pages']} | {record['ocr_pages']} | {record['status']} | {record['seconds']} | {record['cached']} |\n"
report += '\n改名后的两个版本仍提示正文相似。硬件问题分别命中两篇的 P100 原文。541 页教材末页可读。\n'
report += '\n英文扫描件全部识别，P100 证据可跳转第 7 页。\n' if scan_ready else '\n英文 OCR 数据未配置，扫描件标为待识别。\n'
if '--no-report' not in sys.argv[2:]:
    (root/'docs'/'验收报告.md').write_text(report, encoding='utf-8')
print('Dataset verification passed.',flush=True)
