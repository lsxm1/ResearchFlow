"""Shared reading, bibliography, and citation graph tools."""
import re

import pymupdf

from .comparison_facts import MODEL_SIZE, PARAMETER_UNIT, normalized_evidence
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


TABLE_CAPTION = re.compile(r'\s*Table\s+\d+(?:\.\d+)*\s*[:.|]', re.I)


def table_record(table, method, caption=None, *, with_cells=False):
    aligned = [(row, geometry.cells) for row, geometry in zip(table.extract(), table.rows, strict=True)
               if method == 'grid' or any(row)]
    record = {'bbox': list(table.bbox), 'rows': [row for row, _ in aligned], 'method': method}
    if with_cells:
        record.update(caption=caption[4].strip() if caption else '',
                      cells=[[list(cell) if cell is not None else None for cell in geometry]
                             for _, geometry in aligned])
    return record


def nearest_caption(table, captions):
    left, top, right, bottom = table.bbox
    candidates = [(max(top - block[3], block[1] - bottom), block) for block in captions
                  if block[0] < right and block[2] > left]
    return min((item for item in candidates if 0 <= item[0] <= 26), default=(0, None),
               key=lambda item: item[0])[1]


def page_tables(page, blocks, *, with_cells=False):
    captions = [block for block in blocks if TABLE_CAPTION.match(block[4])]
    tables = [table_record(table, 'grid', nearest_caption(table, captions) if with_cells else None,
                           with_cells=with_cells) for table in page.find_tables().tables]
    if tables:
        return tables
    # LaTeX booktabs tables often use horizontal rules and place the caption below.
    rules = {}
    for path in page.get_drawings():
        for item in path['items']:
            if item[0] != 'l':
                continue
            _, left, right = item
            if abs(left.y - right.y) < .5 and abs(right.x - left.x) >= 45:
                key = tuple(sorted((round(left.x, 1), round(right.x, 1))))
                rules.setdefault(key, set()).add(round(left.y, 1))

    for caption in captions:
        candidates = []
        middle = (page.rect.x0 + page.rect.x1) / 2
        for (left, right), lines in rules.items():
            # A short caption may end before the table's right edge.
            same_column = abs(left - caption[0]) <= 12 and (right <= middle or min(left, caption[0]) >= middle)
            if len(lines) < 2 or left < caption[0] - 12 or (right > caption[2] + 12 and not same_column):
                continue
            # A caption between two tables may be next to either. Pick the nearest
            # adjacent rule group in its own column, and keep both directions possible.
            for group in ([y for y in lines if caption[1] - 250 <= y <= caption[1]],
                          [y for y in lines if caption[3] <= y <= caption[3] + 250]):
                if len(group) < 2:
                    continue
                top, bottom = min(group), max(group)
                gap = max(top - caption[3], caption[1] - bottom)
                if 0 <= gap <= 25:
                    # Prefer a preceding table when a following table is similarly close.
                    score = gap + (8 if top >= caption[3] else 0)
                    candidates.append((score, pymupdf.Rect(left-1, top-1, right+1, bottom+1)))
        if candidates:
            _, clip = min(candidates, key=lambda item: item[0])
            previous_count = len(tables)
            for table in page.find_tables(clip=clip & page.rect, strategy='text').tables:
                record = table_record(table, 'ruled_text', caption, with_cells=with_cells)
                if len(record['rows']) >= 3 and 2 <= table.col_count <= 18:
                    tables.append(record)
            if len(tables) > previous_count:
                continue

        # For unruled tables only, inspect nearby blocks within the caption's column.
        region = []
        bottom = caption[3]
        for block in blocks:
            if block[1] < caption[3] - 1 or block == caption:
                continue
            if block[0] < caption[0] - 12 or block[2] > caption[2] + 12:
                continue
            if block[1] - bottom > (42 if not region else 22) or len(region) >= 30:
                break
            region.append(block)
            bottom = max(bottom, block[3])
        if len(region) < 3:
            continue
        clip = pymupdf.Rect(min(b[0] for b in region)-2, min(b[1] for b in region)-2,
                            max(b[2] for b in region)+2, bottom+2) & page.rect
        for table in page.find_tables(clip=clip, strategy='text').tables:
            record = table_record(table, 'caption_text', caption, with_cells=with_cells)
            rows = record['rows']
            numeric_rows = sum(any(re.fullmatch(r'[+−-]?\d+(?:\.\d+)?%?', cell or '') for cell in row) for row in rows)
            if numeric_rows >= 2 and len(rows) >= 3 and 2 <= table.col_count <= 18:
                tables.append(record)
    return tables


def grouped_parameter_table(page, table):
    """Recover parameter rows when the PDF grid groups several models into a single cell."""
    if table['method'] != 'grid' or not re.search(r'\b(?:model configurations?|configurations of)\b',
                                                  normalized_evidence(table['caption'])):
        return None
    if any(len(row) != 2 or not all(row) for row in table['cells']):
        return None
    left, top, right, bottom = table['bbox']
    words = page.get_text('words', clip=pymupdf.Rect(left, top - 18, right, bottom))
    header = sorted((word for word in words if (word[1] + word[3]) / 2 < top), key=lambda word: word[0])
    if len(header) < 3 or [word[4].casefold() for word in header[:2]] != ['model', 'parameters']:
        return None
    if any(abs(word[1] - header[0][1]) > 2 for word in header[1:3]):
        return None
    rows, cells = [['Model', 'Parameters']], [[None, list(header[1][:4])]]
    families = set()
    for model_box, value_box in table['cells']:
        group = [word for word in words if model_box[1] <= (word[1] + word[3]) / 2 < model_box[3]]
        names = [word for word in group if word[2] <= model_box[2]]
        sizes = [word for word in names if re.fullmatch(MODEL_SIZE, word[4], re.I)]
        labels = [word for word in names if word not in sizes]
        counts = [word for word in group if value_box[0] <= word[0] and word[2] < header[2][0]]
        if len(labels) != 1 or len(sizes) < 2 or len(sizes) != len(counts):
            return None
        label, = labels
        family = label[4]
        if family.casefold() in families or not re.fullmatch(r'[a-z][a-z0-9-]+', family, re.I):
            return None
        families.add(family.casefold())
        for size, count in zip(sorted(sizes, key=lambda w: w[1]), sorted(counts, key=lambda w: w[1]), strict=True):
            if size[0] < label[2] or abs(size[1] - count[1]) > 2 or not re.fullmatch(
                    rf'\d+(?:\.\d+)?(?:{PARAMETER_UNIT})', count[4], re.I):
                return None
            rows.append([f'{family}-{size[4]}', count[4]])
            cells.append([list(size[:4]), list(count[:4])])
    return {**table, 'rows': rows, 'cells': cells, 'method': 'grouped_config'}


def merged_parameter_table(page, table):
    """Recover the parameter column when the PDF grid merges adjacent columns."""
    if table['method'] != 'grid' or not table['rows'] or len(table['rows'][0]) != 2:
        return None
    model, _ = table['rows'][0]
    if (model or '').strip().casefold() != 'model' or not all(table['cells'][0]):
        return None
    heading_box = pymupdf.Rect(table['cells'][0][1])
    headings = sorted(page.get_text('words', clip=heading_box), key=lambda word: word[0])
    if len(headings) < 2 or headings[0][4].casefold() != 'parameters' or abs(headings[0][1] - headings[1][1]) > 2:
        return None

    rows = [['Model', 'Parameters']]
    cells = [[table['cells'][0][0], list(headings[0][:4])]]
    column_right = headings[1][0]
    for row, boxes in zip(table['rows'][1:], table['cells'][1:], strict=True):
        if not any((cell or '').strip() for cell in row):
            break
        if len(boxes) != 2 or not all(boxes):
            return None
        model_box, merged_box = map(pymupdf.Rect, boxes)
        parameter_box = pymupdf.Rect(merged_box.x0, merged_box.y0, column_right, merged_box.y1)
        models = sorted(page.get_text('words', clip=model_box), key=lambda word: word[1])
        counts = sorted(page.get_text('words', clip=parameter_box), key=lambda word: word[1])
        if not models or len(models) != len(counts):
            return None
        if any(b[1] - a[1] <= 2 for a, b in zip(models, models[1:])):
            return None
        for name, count in zip(models, counts, strict=True):
            if abs(name[1] - count[1]) > 2 or not re.fullmatch(rf'\d+(?:\.\d+)?(?:{PARAMETER_UNIT})', count[4], re.I):
                return None
            rows.append([name[4], count[4]])
            cells.append([list(name[:4]), list(count[:4])])
    return {**table, 'rows': rows, 'cells': cells, 'method': 'merged_parameter_columns'}


def parameter_table_pages(doc_id, pages):
    """Read only candidate pages on the single PyMuPDF worker."""
    result = []
    with pymupdf.open(UPLOADS / f'{doc_id}.pdf') as pdf:
        for number in pages:
            page = pdf[number - 1]
            blocks = sorted((block for block in page.get_text('blocks') if block[6] == 0), key=lambda block: block[1])
            tables = page_tables(page, blocks, with_cells=True)
            grouped = [record for table in tables if (record := grouped_parameter_table(page, table))]
            merged = [record for table in tables if (record := merged_parameter_table(page, table))]
            result.append((number, [*tables, *grouped, *merged]))
    return result


def inspect_pdf(doc_id, page=None):
    """Run on the existing single PyMuPDF worker."""
    with pymupdf.open(UPLOADS / f'{doc_id}.pdf') as pdf:
        if page is None:
            metadata = pdf.metadata or {}
            return {'metadata': {key: metadata.get(key, '') for key in ('title', 'author', 'subject')},
                    'outline': [{'level': level, 'title': title, 'page': number} for level, title, number in pdf.get_toc() if number > 0]}
        source = pdf[page-1]
        blocks = sorted((block for block in source.get_text('blocks') if block[6] == 0), key=lambda block: block[1])
        tables = page_tables(source, blocks)
        figures = [{'bbox': list(item['bbox'])} for item in source.get_image_info()]
        formulas = [{'bbox': list(block[:4]), 'text': block[4]} for block in blocks
                    if re.search(r'[∑∫√≈≤≥∂∏βϵελ]|\\(?:frac|sum|int)\b|=\s*\w+−\d|[·=].*\n\(\d+\)\s*$', block[4])]
        return {'tables': tables, 'figures': figures, 'formula_candidates': formulas}


def render_page(doc_id, page):
    with pymupdf.open(UPLOADS / f'{doc_id}.pdf') as pdf:
        source = pdf[page-1]
        scale = min(2.0, 1200 / max(source.rect.width, source.rect.height, 1))
        return source.get_pixmap(matrix=pymupdf.Matrix(scale, scale), alpha=False).tobytes('png')


def page_anchor(doc_id, page, bbox):
    with pymupdf.open(UPLOADS / f'{doc_id}.pdf') as pdf:
        source = pdf[page - 1]
        bounds = source.rect
        rect = (pymupdf.Rect(bbox) * source.rotation_matrix) & bounds
        if rect.is_empty:
            return None
        return [(rect.x0 - bounds.x0) / bounds.width, (rect.y0 - bounds.y0) / bounds.height,
                rect.width / bounds.width, rect.height / bounds.height]


def normalize(text):
    return ' '.join(words(re.sub(r'(?<=\w)-\s*\n\s*(?=\w)', '', text)))


def reference_title_span(text, title):
    """Locate normalized title tokens without losing original PDF text offsets."""
    wanted = title.split()
    tokens = [(word, match.start(), match.end())
              for match in re.finditer(r'\w+(?:-\s*\n\s*\w+)*', text)
              for word in normalize(match[0]).split()]
    found = [word for word, _, _ in tokens]
    for index in range(len(found) - len(wanted) + 1):
        if found[index:index + len(wanted)] == wanted:
            return tokens[index][1], tokens[index + len(wanted) - 1][2]
    return None


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
            title_usable = len(title.split()) >= 4 and not re.match(
                r'published as a conference paper|journal of machine learning research \d', title
            )
            for entry, normalized in refs:
                title_hit = title_usable and f' {title} ' in f' {normalized} '
                id_hit = target['arxiv_id'] and re.search(r'(?<!\d)' + re.escape(target['arxiv_id']) + r'(?!\d)', entry['text'])
                if not (id_hit or title_hit):
                    continue
                span = id_hit.span() if id_hit else reference_title_span(entry['text'], title)
                if span:
                    start, end = span
                    edges.append({'source': source['id'], 'target': target['id'], 'page': entry['page'],
                                  'quote': entry['text'][max(0, start-130):end+130].strip()})
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
