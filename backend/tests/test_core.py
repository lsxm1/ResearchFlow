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

import httpx
import pymupdf
from fastapi.testclient import TestClient
from app.comparison import hardware_excerpt, summarize_cells
from app.documents import arxiv_page_title
from app.library_tools import reference_title_span
from app.main import app
from app.training_experiments import training_measurements
from app.training_tables import grid_counts, ruled_counts


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
        value, quote, _, location, orientation = table_maximum(table, prose)
        self.assertEqual((value, quote, location['row'], location['column']), (1_542_000_000, '1542M', 3, 1))
        self.assertEqual(orientation, 'column')
        self.assertIsNone(table_maximum(table, {'value': 1_200_000_000, 'rounding_step': 100_000_000}))
        self.assertIsNone(table_maximum({**table, 'caption': 'Table 2. Baseline model size comparison.'}, prose))
        self.assertIsNone(table_maximum({**table, 'rows': [['Non-Embedding Params', 'Layers'], *table['rows'][1:]]}, None))

    def test_parameter_table_reads_model_scales_across_rows(self):
        from app.parameter_tables import table_maximum

        bbox = [50, 20, 80, 30]
        table = {'caption': 'Table 1 | Key model parameters.',
                 'rows': [['Parameters', '2B', '7B'], ['Layers', '18', '28']],
                 'cells': [[None, [20, 20, 50, 30], bbox], [None, None, None]]}
        prose = {'value': 7_000_000_000, 'rounding_step': 1_000_000_000}
        value, quote, cell, location, orientation = table_maximum(table, prose)
        self.assertEqual((value, quote, cell, orientation), (7_000_000_000, '7B', bbox, 'row'))
        self.assertEqual((location['row'], location['column']), (1, 3))
        self.assertIsNone(table_maximum(table, None))
        ambiguous = {**table, 'rows': [['Parameters', '2B', '7B'], ['7B', '18', '28']],
                     'cells': [table['cells'][0], [[0, 30, 20, 40], None, None]]}
        self.assertIsNone(table_maximum(ambiguous, prose))

    def test_parameter_table_keeps_only_owned_models_in_a_mixed_comparison(self):
        from app.parameter_tables import table_maximum

        table = {'caption': 'Table 2: Comparison of ModelX and other models.',
                 'rows': [['Model', 'Params'], ['Baseline', '360M'], ['ModelXL', '380M'],
                          ['ModelX-small', '14M'], ['ModelX-large', '335M']],
                 'cells': [[None, [30, 10 * row, 50, 10 * row + 10]] for row in range(5)]}
        self.assertIsNone(table_maximum(table, None))
        value, quote, bbox, location, _ = table_maximum(table, None, 'ModelX')
        self.assertEqual((value, quote, bbox), (335_000_000, '335M', table['cells'][4][1]))
        self.assertEqual((location['model'], location['row'], location['column']), ('ModelX-large', 5, 2))
        prose = {'value': 333_000_000, 'rounding_step': 1_000_000, 'variant': 'modelxlarge'}
        self.assertEqual(table_maximum(table, prose, 'ModelX')[0], 335_000_000)
        self.assertIsNone(table_maximum(table, {**prose, 'variant': 'modelxsmall'}, 'ModelX'))

    def test_multi_paper_comparison_keeps_measurements_in_scope(self):
        papers = (
            'We train for 20 epochs on contiguous sequences of 256 tokens. A cited baseline has 900 million parameters.',
            'We introduce SampleLM, a 20-million-parameter model. All models use a context window of 512 tokens.',
            'We develop and release SampleLM, a family ranging in scale from 90M to 3B parameters. We increase the context size from 512 to 1024 tokens.',
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

    def test_parameter_claims_keep_model_ownership_and_units(self):
        from app.comparison_facts import FACETS, identify

        facet = FACETS['parameters'][0]
        cases = (
            ('Our largest (11 billion parameter) model had the best results.', 11_000_000_000, '11 B'),
            ('We have built a larger version of ModelX with 1.5 billion parameters. '
             'A cited baseline uses 11B parameters.', 1_500_000_000, '1.5 B'),
            ('We design two sparse models, one with 395 billion and 1.6 trillion parameters, respectively. '
             'The smaller one uses 6.3T FLOPs per sequence.', 1_600_000_000_000, '1.6 T'),
        )
        for excerpt, value, display in cases:
            with self.subTest(excerpt=excerpt):
                fact = identify(excerpt, facet)
                self.assertEqual((fact['value'], fact['display']), (value, display))
        self.assertIsNone(identify('We design two baseline models, one with 395B and 1.6T parameters.', facet))
        self.assertIsNone(identify('We design two models, one with 395B and 1.6T FLOPs per sequence.', facet))

    def test_parameter_comparison_uses_largest_totals_and_lists_active_counts_separately(self):
        papers = (
            ('We train a small model with 125 million parameters.',
             'Our largest model has 1.5 billion total parameters and 320 million parameters per token.'),
            ('Our largest model has 3 billion parameters.',),
            ('We introduce a sparse model with 48 billion active parameters.',),
        )
        ids = []
        for index, passages in enumerate(papers):
            with pymupdf.open() as pdf:
                pdf.set_metadata({'title': f'Parameter comparison {index + 1}'})
                for passage in passages:
                    pdf.new_page().insert_text((50, 80), passage)
                upload = self.client.post('/api/documents', files={'file': (f'parameters-{index}.pdf', pdf.tobytes())})
            doc_id = upload.raise_for_status().json()['document']['id']
            self.assertEqual(wait_for(self.client, f'/api/documents/{doc_id}', 'status', {'ready', 'failed'})['status'], 'ready')
            ids.append(doc_id)
        job = self.client.post('/api/tasks/compare', json={
            'question': '比较最大模型的总参数和激活参数。', 'document_ids': ids,
        }).raise_for_status().json()
        task = wait_for(self.client, f"/api/tasks/{job['id']}", 'status', {'completed', 'failed'})
        self.assertEqual(task['status'], 'completed')
        row, = task['result']['comparison']['rows']
        self.assertEqual([[(item['kind'], item['value']) for item in cell['excerpts']] for cell in row['cells']], [
            [('parameter_count', 1_500_000_000), ('active_parameter_count', 320_000_000)],
            [('parameter_count', 3_000_000_000)],
            [('active_parameter_count', 48_000_000_000)],
        ])
        sources = {source['id']: source for source in task['result']['evidence']}
        total = row['cells'][0]['excerpts'][0]
        self.assertEqual(sources[total['source_id']]['page'], 2)
        finding, = task['result']['comparison']['findings']
        self.assertIn('的 2.0 倍', finding['text'])
        self.assertIn('激活参数量', finding['text'])

    def test_parameter_comparison_keeps_requested_variant_or_leaves_a_gap(self):
        papers = {
            'Alpha': ['We introduce the Alpha. AlphaBASE contains 110M parameters.',
                      'AlphaLARGE contains 340M parameters.'],
            'Beta': ['We introduce the Beta. BetaLARGE contains 400M parameters.'],
        }
        ids = []
        for name, passages in papers.items():
            with pymupdf.open() as pdf:
                pdf.set_metadata({'title': f'{name}: Language models'})
                for passage in passages:
                    pdf.new_page().insert_text((50, 80), passage)
                upload = self.client.post('/api/documents', files={'file': (f'{name}.pdf', pdf.tobytes())})
            doc_id = upload.raise_for_status().json()['document']['id']
            self.assertEqual(wait_for(self.client, f'/api/documents/{doc_id}', 'status', {'ready', 'failed'})['status'], 'ready')
            ids.append(doc_id)

        def compare(question):
            job = self.client.post('/api/tasks/compare', json={
                'question': question, 'document_ids': ids,
            }).raise_for_status().json()
            task = wait_for(self.client, f"/api/tasks/{job['id']}", 'status', {'completed', 'failed'})
            self.assertEqual(task['status'], 'completed')
            row, = task['result']['comparison']['rows']
            return row['cells']

        alpha, beta = compare('比较两篇论文 Base 版的参数量')
        self.assertEqual(alpha['excerpts'][0]['value'], 110_000_000)
        self.assertEqual(beta['status'], 'missing')
        self.assertNotIn('source_id', beta)
        alpha, beta = compare('比较 Alpha-Base 和 Beta-Large 的参数量')
        self.assertEqual([cell['excerpts'][0]['value'] for cell in (alpha, beta)], [110_000_000, 400_000_000])
        alpha, beta = compare('比较最大模型的参数量')
        self.assertEqual([cell['excerpts'][0]['value'] for cell in (alpha, beta)], [340_000_000, 400_000_000])

    def test_named_parameter_evidence_resolves_variants_and_withholds_conflicts(self):
        from app.comparison import named_parameter_evidence

        source = {'source': 'native', 'text': 'Other-ModelX-Large has 900M parameters. '
                  'ModelXBASE contains 110M parameters and ModelX-Large contains 340M parameters. '
                  'ModelX-Large has 450M parameters per layer.'}
        paper = {'id': 'paper-x'}
        with patch('app.comparison.retrieve', return_value=[source]):
            _, fact = named_parameter_evidence(paper, 'ModelX', 'query')
            self.assertEqual(fact['value'], 340_000_000)
        conflicting = {**source, 'text': 'ModelXLARGE contains 350M parameters.'}
        with patch('app.comparison.retrieve', return_value=[source, conflicting]):
            self.assertIsNone(named_parameter_evidence(paper, 'ModelX', 'query'))

    def test_natural_hardware_question_reaches_actual_gpu_evidence(self):
        doc_id = self.upload_ready()
        result = self.client.post('/api/tasks', json={'question':'What hardware and training schedule were used?', 'document_ids':[doc_id]})
        result = wait_for(self.client, f"/api/tasks/{result.json()['id']}", 'status', {'completed','failed'})
        self.assertTrue(any('P100' in e['text'] for e in result['result']['evidence']))

    def test_adam_question_keeps_cited_values_and_leaves_unknown_values_open(self):
        doc_id = self.upload_ready()
        job = self.client.post('/api/tasks', json={
            'question': '这篇论文的 Adam β₁ 和 β₂ 分别是多少？', 'document_ids': [doc_id],
        }).raise_for_status().json()
        task = wait_for(self.client, f"/api/tasks/{job['id']}", 'status', {'completed', 'failed'})
        self.assertEqual(task['status'], 'completed')
        result = task['result']
        claim, = result['claims']
        self.assertEqual(claim['text'], 'Adam β1 = 0.9')
        self.assertTrue(any('缺少' in warning and 'β2' in warning for warning in result['warnings']))
        source = next(item for item in result['evidence'] if item['id'] == claim['source_id'])
        self.assertIn(claim['quote'], source['text'])
        reader = self.client.get(f"/api/read/{doc_id}/pages/{source['page']}", params={'evidence_id': source['id']})
        self.assertIsNotNone(reader.raise_for_status().json()['highlight'])

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

    def test_question_preserves_citable_evidence_on_model_failure(self):
        doc_id = self.upload_ready()
        config = {'mode': 'openai_compatible', 'base_url': 'http://unused', 'model': 'stub', 'key': ''}
        cases = (
            ([httpx.ConnectTimeout('model is offline')], '模型请求失败'),
            ([{'search_query': 'P100 GPU'}, {'claims': None}], '模型未能生成可读取的答案'),
        )
        for replies, warning in cases:
            with (
                self.subTest(warning=warning),
                patch('app.agent.model_config', return_value=config),
                patch('app.agent.call_model', side_effect=replies) as model,
            ):
                job = self.client.post('/api/tasks', json={'question': 'NVIDIA hardware', 'document_ids': [doc_id]})
                task = wait_for(self.client, f"/api/tasks/{job.raise_for_status().json()['id']}", 'status', {'completed', 'failed'})
                self.assertEqual(model.call_count, len(replies))
                self.assertEqual(task['status'], 'completed')
                result = task['result']
                self.assertEqual(result['mode'], 'evidence_only')
                self.assertEqual(result['claims'], [])
                self.assertTrue(any(warning in message for message in result['warnings']))
                source = next(e for e in result['evidence'] if 'P100' in e['text'])
                reader = self.client.get(f"/api/read/{doc_id}/pages/{source['page']}", params={'evidence_id': source['id']})
                self.assertIsNotNone(reader.raise_for_status().json()['highlight'])

    def test_model_hardware_counts_must_match_quoted_device_units(self):
        gpu_quote = 'Hardware: eight NVIDIA P100 GPUs.'
        tpu_quote = 'BERT uses 4 Cloud TPUs (16 TPU chips total).'
        pdf = pdf_bytes(f'{gpu_quote} {tpu_quote}')
        upload = self.client.post('/api/documents', files={'file': ('devices.pdf', pdf)}).raise_for_status().json()
        doc_id = upload['document']['id']
        doc = wait_for(self.client, f'/api/documents/{doc_id}', 'status', {'ready', 'failed'})
        self.assertEqual(doc['status'], 'ready')

        def fake_model(system, payload):
            if system.startswith('You are'):
                return {'search_query': ''}
            source = next(item for item in payload['evidence'] if gpu_quote in item['text'] and tpu_quote in item['text'])
            return {'claims': [
                {'text': '训练使用了 8 块 NVIDIA P100 GPU。', 'source_id': source['id'], 'quote': gpu_quote},
                {'text': '训练使用了 12 块 NVIDIA P100 GPU。', 'source_id': source['id'], 'quote': gpu_quote},
                {'text': 'BERT 使用 16 个 TPU 芯片。', 'source_id': source['id'], 'quote': tpu_quote},
                {'text': 'BERT 使用 16 个 Cloud TPU。', 'source_id': source['id'], 'quote': tpu_quote},
            ]}

        config = {'mode': 'openai_compatible', 'base_url': 'http://unused', 'model': 'stub', 'key': ''}
        with (
            patch('app.agent.model_config', return_value=config),
            patch('app.agent.call_model', side_effect=fake_model),
        ):
            job = self.client.post('/api/tasks', json={'question': 'What GPU and TPU hardware was used?', 'document_ids': [doc_id]})
            task = wait_for(self.client, f"/api/tasks/{job.raise_for_status().json()['id']}", 'status', {'completed', 'failed'})
        self.assertEqual(task['status'], 'completed')
        self.assertEqual([claim['text'] for claim in task['result']['claims']],
                         ['训练使用了 8 块 NVIDIA P100 GPU。', 'BERT 使用 16 个 TPU 芯片。'])
        self.assertTrue(any('GPU 或 TPU 数量' in message for message in task['result']['warnings']))

    def test_invalid_upload_and_unknown_document_are_rejected(self):
        result = self.client.post('/api/documents', files={'file':('fake.pdf',b'not a PDF','application/pdf')})
        self.assertEqual(result.status_code, 415)
        self.assertEqual(self.client.get('/api/documents/unknown/file').status_code, 404)

    def test_second_followup_reaches_the_answer_and_records_validation(self):
        doc_id = self.upload_ready()
        quote = 'Hardware: eight NVIDIA P100 GPUs.'

        def source(source_id, text):
            return {'id': source_id, 'text': text, 'document_id': doc_id}

        initial = [source(f'initial-{index}', 'Background evidence from the original search.') for index in range(6)]
        followup = [source(f'followup-{index}', 'The experiment describes a training setup.') for index in range(6)]
        decisive = source('decisive', quote)
        plans = []

        def fake_model(system, payload):
            evidence_ids = {item['id'] for item in payload['evidence']}
            if system.startswith('You are'):
                plans.append(evidence_ids)
                if len(plans) == 1:
                    return {'sufficient': False, 'search_query': 'training setup'}
                return {'sufficient': False, 'search_query': 'NVIDIA P100 GPU'}
            self.assertIn(decisive['id'], evidence_ids)
            return {'claims': [
                {'source_id': decisive['id'], 'text': '训练使用 8 块 NVIDIA P100 GPU。', 'quote': quote},
                {'source_id': decisive['id'], 'text': '训练使用 12 块 NVIDIA P100 GPU。', 'quote': quote},
                {'source_id': decisive['id'], 'text': '观察到训练硬件。', 'quote': 'A fabricated quote of adequate length.'},
            ]}

        config = {'mode': 'openai_compatible', 'base_url': 'http://unused', 'model': 'stub', 'key': ''}
        with (
            patch('app.agent.model_config', return_value=config),
            patch('app.agent.collect_evidence', side_effect=[initial, followup, [decisive]]) as search,
            patch('app.agent.call_model', side_effect=fake_model),
        ):
            job = self.client.post('/api/tasks', json={'question': 'What hardware was used?', 'document_ids': [doc_id]})
            task = wait_for(self.client, f"/api/tasks/{job.raise_for_status().json()['id']}", 'status', {'completed', 'failed'})

        self.assertEqual(task['status'], 'completed')
        self.assertEqual(search.call_count, 3)
        self.assertNotIn('followup-0', plans[0])
        self.assertIn('followup-0', plans[1])
        self.assertEqual([event['step'] for event in task['events'] if event['step'].startswith('observe_')],
                         ['observe_1', 'observe_2'])
        self.assertEqual(task['result']['mode'], 'model_assisted')
        claim, = task['result']['claims']
        self.assertEqual(claim['source_id'], decisive['id'])
        self.assertIn(decisive, task['result']['evidence'])
        self.assertEqual(task['result']['validation'],
                         {'generated': 3, 'checked': 3, 'accepted': 1, 'rejected': 2,
                          'rejected_quote': 1, 'rejected_hardware': 1})

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

    def test_named_training_measurements_keep_variant_and_shared_scope(self):
        transformer = ('We trained our models on 8 NVIDIA P100 GPUs. '
                       'The base models were trained for 100,000 steps (12 hours). '
                       'The big models were trained for 300,000 steps (3.5 days).')
        self.assertEqual(list(training_measurements(transformer, 'Transformer', 'big', 'duration')),
                         [(5040, '3.5 days')])
        self.assertEqual(list(training_measurements(transformer, 'Transformer', 'base', 'steps')),
                         [(100000, '100,000 steps')])
        bert = ('Training of BERTBASE was performed on 4 Cloud TPUs (16 TPU chips total).13 '
                'Training of BERTLARGE was performed on 16 Cloud TPUs (64 TPU chips total). '
                'Each pre-training took 4 days to complete. We train for 1,000,000 steps.')
        self.assertEqual(list(training_measurements(bert, 'BERT', 'large', 'hardware'))[0][0],
                         {(16, 'tpu', ''), (64, 'tpu', 'chip')})
        self.assertEqual(list(training_measurements(bert, 'BERT', 'large', 'duration')), [(5760, '4 days')])
        self.assertEqual(list(training_measurements(bert, 'BERT', 'large', 'steps')), [])


    def test_training_table_uses_owned_row_and_device_count_heading(self):
        table = {
            'rows': [['Model Size', 'GPU Count', 'GPT-3 GPUs', 'Total GPU hours'],
                     ['410 M', '32', '8', '2,540'], ['1.0 B', '64', '16', '4,830']],
            'cells': [[[0, 0, 1, 1]] * 4, [[0, 1, 1, 2], [1, 1, 2, 2], [2, 1, 3, 2], [3, 1, 4, 2]],
                      [[0, 2, 1, 3]] * 4],
        }
        caption = 'models in the pythia suite and number of gpus used during training'
        self.assertEqual(list(grid_counts(table, caption, 'Pythia', '410m')),
                         [('gpu', '32', [1, 1, 2, 2], 2, 2, 'GPU Count')])
        self.assertEqual(list(grid_counts(table, caption, 'Other', '410m')), [])
        table['rows'][0][1] = 'GPU hours'
        self.assertEqual(list(grid_counts(table, caption, 'Pythia', '410m')), [])

        with pymupdf.open() as pdf:
            page = pdf.new_page()
            for x, y, text in ((50, 70, 'Model Size'), (112, 70, 'GPU Count'), (180, 70, 'GPT-3 GPUs'),
                               (55, 91, '410 M'), (122, 91, '32'), (192, 91, '8')):
                page.insert_text((x, y), text)
            repaired, = ruled_counts(page, {'bbox': [64, 62, 238, 96]}, caption, 'Pythia', '410m')
            unit, value, bbox, row, column, heading = repaired
            self.assertEqual((unit, value, row, column, heading), ('gpu', '32', 2, 2, 'GPU Count'))
            self.assertTrue(112 <= bbox[0] < bbox[2] < 180)


if __name__ == '__main__':
    unittest.main()
