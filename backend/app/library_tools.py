"""Shared reading, bibliography, and citation graph tools."""
import re

import pymupdf

from .config import UPLOADS
from .db import connect
from .documents import words

REFERENCE_HEADING = re.compile(r'^\s*(?:\d+[.\s]+)?(?:references|bibliography|参考文献)\s*$', re.I | re.M)
APPENDIX = re.compile(r'^\s*(?:appendix\b|[A-Z]\.\d+\s+)', re.I | re.M)
NUMBERED_ENTRY = re.compile(r'^\s*\[(\d+)\]\s*', re.M)


def paper_and_pages(doc_id):
    with connect() as con:
        paper = con.execute('SELECT * FROM documents WHERE id=?', (doc_id,)).fetchone()
        if paper is None:
            raise LookupError('文献不存在。')
        pages = con.execute('SELECT page,text FROM pages WHERE document_id=? ORDER BY page', (doc_id,)).fetchall()
    return paper, pages


def bibliography(pages):
    """Find the reference section and keep each entry's physical page."""
    found = False
    entries = []
    earlier_text = []
    for row in pages:
        text, page = row['text'], row['page']
        if not found:
            heading = REFERENCE_HEADING.search(text)
            if not heading:
                earlier_text.append((page, text))
                continue
            earlier_text.append((page, text[:heading.start()]))
            text = text[heading.end():]
            found = True
        end = APPENDIX.search(text)
        if end:
            text = text[:end.start()]
        matches = list(NUMBERED_ENTRY.finditer(text))
        if matches:
            for index, match in enumerate(matches):
                stop = matches[index+1].start() if index + 1 < len(matches) else len(text)
                label = f'[{match[1]}]'
                first = next((p for p, body in earlier_text if label in body), None)
                entries.append({'label': label, 'page': page, 'text': text[match.start():stop].strip(), 'citation_page': first})
        elif text.strip():
            # Author-year bibliographies still preserve verbatim page-level evidence.
            entries.append({'label': '', 'page': page, 'text': text.strip(), 'citation_page': None})
        if end:
            break
    return entries


def outline_from_text(pages):
    outline = []
    for row in pages:
        lines = [line.strip() for line in row['text'].splitlines() if line.strip()]
        for index, line in enumerate(lines):
            match = re.fullmatch(r'(\d{1,2}(?:\.\d{1,2}){0,2})\.?\s*([A-Z][\w ,\-:]{2,90})?', line)
            if not match:
                continue
            title = match[2] or (lines[index+1] if index+1 < len(lines) else '')
            if re.fullmatch(r'[A-Z][\w ,\-:]{2,90}', title):
                outline.append({'title': f'{match[1]} {title}', 'page': row['page'], 'level': match[1].count('.')+1})
    return outline[:250]


def inspect_pdf(doc_id, page=None):
    """Run on the existing single PyMuPDF worker."""
    with pymupdf.open(UPLOADS / f'{doc_id}.pdf') as pdf:
        if page is None:
            metadata = pdf.metadata or {}
            return {'metadata': {key: metadata.get(key, '') for key in ('title', 'author', 'subject')},
                    'outline': [{'level': level, 'title': title, 'page': number} for level, title, number in pdf.get_toc() if number > 0]}
        source = pdf[page-1]
        tables = [{'bbox': list(table.bbox), 'rows': table.extract()} for table in source.find_tables().tables]
        figures = [{'bbox': list(item['bbox'])} for item in source.get_image_info()]
        formulas = [{'bbox': list(block[:4]), 'text': block[4]} for block in source.get_text('blocks')
                    if block[6] == 0 and re.search(r'[∑∫√≈≤≥∂∏]|\\(?:frac|sum|int)\b', block[4])]
        return {'tables': tables, 'figures': figures, 'formula_candidates': formulas}


def render_page(doc_id, page):
    with pymupdf.open(UPLOADS / f'{doc_id}.pdf') as pdf:
        source = pdf[page-1]
        scale = min(2.0, 1200 / max(source.rect.width, source.rect.height, 1))
        return source.get_pixmap(matrix=pymupdf.Matrix(scale, scale), alpha=False).tobytes('png')


def normalize(text):
    return ' '.join(words(re.sub(r'(?<=\w)-\s*\n\s*(?=\w)', '', text)))


def citation_graph(doc_ids):
    records = [paper_and_pages(doc_id) for doc_id in dict.fromkeys(doc_ids)]
    nodes = [{'id': paper['id'], 'title': paper['title'], 'filename': paper['filename']} for paper, _ in records]
    edges = []
    future_work = []
    for source, pages in records:
        refs = [(entry, normalize(entry['text'])) for entry in bibliography(pages)]
        for target, _ in records:
            if source['id'] == target['id'] or (source['arxiv_id'] and source['arxiv_id'] == target['arxiv_id']):
                continue
            title = normalize(target['title'])
            for entry, normalized in refs:
                title_hit = len(title.split()) >= 4 and f' {title} ' in f' {normalized} '
                id_hit = target['arxiv_id'] and re.search(r'(?<!\d)' + re.escape(target['arxiv_id']) + r'(?!\d)', entry['text'])
                if title_hit or id_hit:
                    edges.append({'source': source['id'], 'target': target['id'], 'page': entry['page'], 'quote': entry['text']})
                    break
        for page in pages:
            if len(future_work) >= 12:
                break
            match = re.search(r'\b(?:future work|future research|leave .{1,90} for future|plan to investigate)\b', page['text'], re.I)
            if match:
                start = max(0, match.start()-80)
                future_work.append({'document_id': source['id'], 'page': page['page'], 'quote': page['text'][start:start+400].strip()})
    if nodes:
        neighbors = {node['id']: [] for node in nodes}
        for edge in edges:
            neighbors[edge['source']].append(edge['target'])
        rank = {key: 1 / len(nodes) for key in neighbors}
        for _ in range(25):
            dangling = sum(rank[key] for key, values in neighbors.items() if not values)
            new_rank = {key: (.15 + .85 * dangling) / len(nodes) for key in rank}
            for key, targets in neighbors.items():
                for target in targets:
                    new_rank[target] += .85 * rank[key] / len(targets)
            rank = new_rank
        for node in nodes:
            node['rank'] = round(rank[node['id']], 5)
        nodes.sort(key=lambda node: -node['rank'])
    titles = {node['id']: node['title'] for node in nodes}
    for item in future_work:
        item['related'] = [{'document_id': edge['source'], 'title': titles[edge['source']], 'page': edge['page']}
                           for edge in edges if edge['target'] == item['document_id']]
    overview = [f"{titles[edge['source']]} 在参考文献中引用了 {titles[edge['target']]}。" for edge in edges[:12]]
    return {'nodes': nodes, 'edges': edges, 'overview': overview, 'future_work': future_work}
