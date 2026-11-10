"""Behavioral regression tests. Isolated data, no paid APIs, no external fixtures."""
import json
import os
import sys
import time
import unittest
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'backend'))
os.environ['RF_DATA_DIR'] = str(ROOT / '.cache' / 'tests' / uuid4().hex)
os.environ['RF_MODEL_MODE'] = 'disabled'

import pymupdf
from fastapi.testclient import TestClient
from app.comparison import hardware_excerpt, summarize_cells
from app.documents import arxiv_page_title
from app.library_tools import reference_title_span
from app.main import app


def pdf_bytes(text, image=False):
    with pymupdf.open() as pdf:
        page = pdf.new_page()
        page.insert_text((50,80), text)
        if not image:
            return pdf.tobytes()
        png = page.get_pixmap().tobytes('png')
    with pymupdf.open() as scan:
        scan.new_page().insert_image(pymupdf.Rect(0,0,595,842), stream=png)
        return scan.tobytes()


def wait_for(client, path, field, terminal):
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        data = client.get(path).json()
        if data[field] in terminal:
            return data
        time.sleep(.05)
    raise AssertionError('Background operation timed out')


class CoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.native = pdf_bytes('Hardware: eight NVIDIA P100 GPUs. Training used Adam optimizer with beta1=0.9.')
        cls.scan = pdf_bytes('Scanned page: this content must not be silently treated as indexed.', image=True)
        cls.client = TestClient(app)
        cls.client.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls.client.__exit__(None, None, None)

    def upload_ready(self):
        response = self.client.post('/api/documents', files={'file': ('paper.pdf', self.native, 'application/pdf')})
        self.assertEqual(response.status_code, 201)
        doc_id = response.json()['document']['id']
        doc = wait_for(self.client, f'/api/documents/{doc_id}', 'status', {'ready','failed'})
        self.assertEqual(doc['status'], 'ready')
        return doc_id

    def test_renaming_cannot_bypass_exact_duplicate_detection(self):
        doc_id = self.upload_ready()
        response = self.client.post('/api/documents', files={'file': ('totally-renamed.pdf', self.native, 'application/pdf')})
        self.assertTrue(response.json()['exact_duplicate'])
        self.assertEqual(response.json()['document']['id'], doc_id)

    def test_version_choice_is_reversible_and_preserves_citation_links(self):
        text = 'Transformer training uses multi-head attention and an encoder decoder architecture.'
        versions = []
        for version in (7, 1):
            with pymupdf.open() as pdf:
                page = pdf.new_page()
                page.insert_text((50, 60), f'arXiv:1706.03762v{version}')
                page.insert_text((50, 120), text)
                content = pdf.tobytes()
            upload = self.client.post('/api/documents', files={'file': ('renamed.pdf', content)})
            doc_id = upload.raise_for_status().json()['document']['id']
            result = wait_for(self.client, f'/api/documents/{doc_id}', 'status', {'ready', 'failed'})
            self.assertEqual(result['version'], version)
            versions.append((doc_id, content))
        (new, _), (old, old_pdf) = versions
        old_url = f'/api/documents/{old}'
        pair = {'keep_id': new, 'archive_id': old}
        path = '/api/documents/resolve-version'

        unrelated = self.upload_ready()
        self.assertEqual(self.client.post(path, json={**pair, 'archive_id': unrelated}).status_code, 409)
        self.client.post(path, json=pair).raise_for_status()
        self.client.post(path, json=pair).raise_for_status()
        self.assertEqual(self.client.post(path, json={'keep_id': old, 'archive_id': new}).status_code, 409)
        self.assertTrue(self.client.get(old_url).json()['archived'])
        self.assertEqual(self.client.get(old_url + '/file').content, old_pdf)
        self.assertIn(text, self.client.get(old_url + '/pages/1').json()['text'])
        self.assertEqual(self.client.get(f'/api/documents/{new}').json()['duplicates'], [])
        self.assertEqual(self.client.post('/api/tasks', json={'question': 'Training setup?', 'document_ids': [old]}).status_code, 409)

        self.client.post(old_url + '/restore').raise_for_status()
        self.assertFalse(self.client.get(old_url).json()['archived'])
        self.client.post(path, json={'keep_id': old, 'archive_id': new}).raise_for_status()
        self.assertTrue(self.client.get(f'/api/documents/{new}').json()['archived'])

    def test_scanned_pdf_cannot_silently_answer_without_ocr(self):
        response = self.client.post('/api/documents', files={'file': ('scan.pdf', self.scan, 'application/pdf')})
        doc_id = response.json()['document']['id']
        result = wait_for(self.client, f'/api/documents/{doc_id}', 'status', {'needs_ocr','failed'})
        self.assertEqual(result['status'], 'needs_ocr')
        self.assertEqual(result['text_pages'], 0)
        self.assertEqual(result['ocr_pages'], 1)
        result = self.client.post('/api/tasks', json={'question':'GPU hardware?', 'document_ids':[doc_id]})
        self.assertEqual(result.status_code, 409)

    def test_retrieval_preserves_page_and_exact_original_text(self):
        doc_id = self.upload_ready()
        result = self.client.post('/api/tasks', json={'question':'NVIDIA P100 hardware', 'document_ids':[doc_id]})
        result = wait_for(self.client, f"/api/tasks/{result.json()['id']}", 'status', {'completed','failed'})
        self.assertEqual(result['status'], 'completed')
        self.assertEqual(result['result']['mode'], 'evidence_only')
        self.assertEqual(result['result']['claims'], [])
        source = result['result']['evidence'][0]
        self.assertEqual(source['document_id'], doc_id)
        original = self.client.get(f'/api/documents/{doc_id}/pages/1').json()
        self.assertIn(source['text'], original['text'])
        self.assertEqual(source['page'], 1)
        self.assertEqual(len(source['bbox']), 4)

    def test_comparison_keeps_step_and_duration_evidence_independently(self):
        base = self.upload_ready()
        with pymupdf.open() as pdf:
            pdf.set_metadata({'title': 'Schedule Evidence'})
            page = pdf.new_page()
            page.insert_text((50, 80), 'We train with a batch size of 256 sequences for 40,000 steps.')
            page = pdf.new_page()
            page.insert_text((50, 80), 'Each pre-training took 4 days to complete.')
            page.set_rotation(90)
            target_pdf = pdf.tobytes()
        target = self.client.post('/api/documents', files={'file': ('schedule.pdf', target_pdf)}).json()['document']['id']
        wait_for(self.client, f'/api/documents/{target}', 'status', {'ready', 'failed'})
        job = self.client.post('/api/tasks/compare', json={
            'question': 'Compare training steps and training duration.',
            'document_ids': [base], 'target_paper': 'Schedule Evidence',
        }).raise_for_status().json()
        result = wait_for(self.client, f"/api/tasks/{job['id']}", 'status', {'completed', 'failed'})['result']
        left, right = result['comparison']['rows'][0]['cells']
        self.assertEqual([item['status'] for item in left['excerpts']], ['missing', 'missing'])
        self.assertTrue(all(len(item['searched_queries']) == 2 for item in left['excerpts']))
        self.assertTrue(all(len(item['searched_queries']) == 1 for item in right['excerpts']))
        self.assertEqual(right['status'], 'verified_excerpt')
        sources = {source['id']: source for source in result['evidence']}
        steps, duration = [sources[item['source_id']] for item in right['excerpts']]
        self.assertEqual((steps['page'], duration['page']), (1, 2))
        reader = self.client.get(f'/api/read/{target}/pages/2', params={'evidence_id': duration['id']}).json()
        self.assertIn('4 days', reader['text'])
        self.assertGreater(reader['highlight'][3], reader['highlight'][2])
        wrong_page = self.client.get(f'/api/read/{target}/pages/1', params={'evidence_id': duration['id']}).json()
        self.assertIsNone(wrong_page['highlight'])

    def test_parameter_table_requires_scope_and_consistent_precision(self):
        from app.parameter_tables import table_maximum

        table = {'caption': 'Table 1. Architecture of our models.',
                 'rows': [['Parameters', 'Layers'], ['120M', '8'], ['1542M', '24']],
                 'cells': [[None, None], [[10, 20, 40, 30], None], [[10, 30, 40, 40], None]]}
        prose = {'value': 1_500_000_000, 'rounding_step': 100_000_000}
        value, quote, _, location = table_maximum(table, prose)
        self.assertEqual((value, quote, location['row'], location['column']), (1_542_000_000, '1542M', 3, 1))
        self.assertIsNone(table_maximum(table, {'value': 1_200_000_000, 'rounding_step': 100_000_000}))
        self.assertIsNone(table_maximum({**table, 'caption': 'Table 2. Baseline model size comparison.'}, prose))
        self.assertIsNone(table_maximum({**table, 'rows': [['Non-Embedding Params', 'Layers'], *table['rows'][1:]]}, None))

    def test_multi_paper_comparison_keeps_measurements_in_scope(self):
        papers = (
            'We train for 20 epochs on contiguous sequences of 256 tokens. A cited baseline has 900 million parameters.',
            'We introduce SampleLM, a 20-million-parameter model. All models use a context window of 512 tokens.',
            'We present SampleLM, a family ranging from 90M to 3B parameters. We increase the context size from 512 to 1024 tokens.',
        )
        ids = []
        for index, text in enumerate(papers):
            with pymupdf.open() as pdf:
                pdf.set_metadata({'title': f'Sample paper {index + 1}'})
                pdf.new_page().insert_textbox((50, 55, 550, 200), text)
                upload = self.client.post('/api/documents', files={'file': (f'sample-{index}.pdf', pdf.tobytes())})
            doc_id = upload.raise_for_status().json()['document']['id']
            self.assertEqual(wait_for(self.client, f'/api/documents/{doc_id}', 'status', {'ready', 'failed'})['status'], 'ready')
            ids.append(doc_id)
        body = {'question': '比较参数量和上下文窗口。', 'document_ids': ids}
        self.assertEqual(self.client.post('/api/tasks/compare', json={**body, 'target_paper': '1234.56789'}).status_code, 422)
        job = self.client.post('/api/tasks/compare', json=body).raise_for_status().json()
        task = wait_for(self.client, f"/api/tasks/{job['id']}", 'status', {'completed', 'failed'})
        self.assertEqual(task['status'], 'completed')
        self.assertEqual(task['document_ids'], ids)
        evidence = {item['id']: item for item in task['result']['evidence']}
        size, context = task['result']['comparison']['findings']
        self.assertEqual([evidence[item['source_id']]['document_id'] for item in size['citations']], ids[1:])
        self.assertIn('150.0 倍', size['text'])
        self.assertIn('未检得', size['text'])
        self.assertEqual([evidence[item['source_id']]['document_id'] for item in context['citations']], ids)
        self.assertIn('训练序列长度 256 tokens', context['text'])
        self.assertIn('窗口相差最多 512 tokens', context['text'])

    def test_natural_hardware_question_reaches_actual_gpu_evidence(self):
        doc_id = self.upload_ready()
        result = self.client.post('/api/tasks', json={'question':'What hardware and training schedule were used?', 'document_ids':[doc_id]})
        result = wait_for(self.client, f"/api/tasks/{result.json()['id']}", 'status', {'completed','failed'})
        self.assertTrue(any('P100' in e['text'] for e in result['result']['evidence']))

    def test_fabricated_citation_is_removed_from_model_output(self):
        doc_id = self.upload_ready()
        def fake_model(system, payload):
            if system.startswith('You are'):
                return {'search_query': ''}
            source = payload['evidence'][0]
            return {'claims': [
                {'text':'invented fact', 'source_id':'fake-id', 'quote':'invented quotation'},
                {'text':'hardware statement', 'source_id':source['id'], 'quote':source['text'][:40]},
                {'text':'wrong quote', 'source_id':source['id'], 'quote':'This quotation does not exist in the PDF.'},
            ], 'limitations': []}
        with patch('app.agent.model_config', return_value={'mode':'openai_compatible','base_url':'http://unused','model':'stub','key':''}), patch('app.agent.call_model', side_effect=fake_model):
            result = self.client.post('/api/tasks', json={'question':'NVIDIA hardware', 'document_ids':[doc_id]})
            result = wait_for(self.client, f"/api/tasks/{result.json()['id']}", 'status', {'completed','failed'})
        self.assertEqual(result['status'], 'completed')
        self.assertEqual(len(result['result']['claims']), 1)
        self.assertEqual(result['result']['claims'][0]['text'], 'hardware statement')
        self.assertTrue(result['result']['warnings'])

    def test_invalid_upload_and_unknown_document_are_rejected(self):
        result = self.client.post('/api/documents', files={'file':('fake.pdf',b'not a PDF','application/pdf')})
        self.assertEqual(result.status_code, 415)
        self.assertEqual(self.client.get('/api/documents/unknown/file').status_code, 404)

    def test_comparison_rejects_a_quote_from_the_other_paper(self):
        left = {'document_id': 'left', 'status': 'candidate', 'source_id': 'left-block',
                'quote': 'Training took 80,000 steps.'}
        right = {'document_id': 'right', 'status': 'candidate', 'source_id': 'right-block',
                 'quote': 'Training took 100,000 steps.'}
        rows = [{'key': 'schedule', 'cells': [left, right]}]
        response = {'cells': [
            {'field': 'schedule', 'document_id': 'left', 'source_id': 'right-block',
             'quote': right['quote'], 'text': '100,000 steps.'},
            {'field': 'schedule', 'document_id': 'right', 'source_id': 'right-block',
             'quote': right['quote'], 'text': '100,000 steps.'},
        ]}
        with patch('app.agent.call_model', return_value=response):
            self.assertEqual(summarize_cells('test', 'Compare the hardware', rows), 1)
        self.assertEqual(left['status'], 'candidate')
        self.assertEqual(right['status'], 'cited_summary')

    def test_compact_extractor_rejects_altered_number(self):
        cell = {'document_id': 'left', 'status': 'candidate', 'source_id': 'left-block',
                'quote': 'Training used 80,000 steps.'}
        rows = [{'key': 'schedule', 'cells': [cell]}]
        with patch.dict(os.environ, {'RF_MODEL_PROFILE': 'compact'}), patch('app.agent.chat_completion', return_value=('90,000 steps', 'stop')):
            self.assertEqual(summarize_cells('test', 'Compare training steps', rows), 0)
        self.assertEqual(cell['status'], 'candidate')

    def test_hardware_excerpt_keeps_each_experiment_and_unit(self):
        text = ('Training of ModelSmall was performed on 4\nCloud TPUs in Pod mode (16 TPU chips total).3 '
                'Training of ModelLarge was performed on 16 Cloud TPUs (64 TPU chips total). '
                'Fine-tuning took an hour on 1 Cloud TPU.')
        self.assertEqual(hardware_excerpt(text), text[:text.index('. Fine-tuning')])
        self.assertIsNone(hardware_excerpt('Fine-tuning was performed on 1 Cloud TPU.'))

    def test_arxiv_title_excludes_venue_and_preserves_wrapped_title(self):
        with pymupdf.open() as pdf:
            page = pdf.new_page()
            page.insert_text((70, 35), 'Published as a conference paper at ICLR 2020', fontsize=10)
            page.insert_text((70, 90), 'Learning from a Large', fontsize=17)
            page.insert_text((70, 110), 'Collection of Papers', fontsize=17)
            page.insert_text((70, 133), 'Author Name', fontsize=14)
            self.assertEqual(arxiv_page_title(page), 'Learning from a Large Collection of Papers')

    def test_reference_span_preserves_wrapped_title_and_ligature(self):
        original = 'Earlier reference. Reformer: The Eﬃ-\ncient Transformer. Following reference.'
        start, end = reference_title_span(original, 'reformer the efficient transformer')
        self.assertEqual(original[start:end], 'Reformer: The Eﬃ-\ncient Transformer')


if __name__ == '__main__':
    unittest.main()
