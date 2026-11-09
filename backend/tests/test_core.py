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
