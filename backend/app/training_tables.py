"""Read device counts only from a training table's owned model row and count column."""
import logging
import re

import pymupdf

from .comparison_facts import model_key, normalized_evidence
from .config import UPLOADS
from .db import connect
from .documents import PDF_POOL
from .library_tools import page_tables, table_cell_evidence

log = logging.getLogger(__name__)
MODEL_HEADER = re.compile(r'model(?: (?:name|size|variant))?')
COUNT_HEADER = re.compile(r'(?:(gpu|tpu)s? count|(?:#\s*|number of )(gpu|tpu)s|(?:gpus|tpus))')
COUNT = re.compile(r'(?:[1-9]\d*|[1-9]\d{0,2}(?:,\d{3})+)')
TRAINING_CAPTION = re.compile(r'\b(?:pre[- ]?training|trained|[gt]pus? used during)\b')


def matching_row(label, caption, family, size):
    key = model_key(normalized_evidence(label))
    owner = re.search(rf'(?<!\w){re.escape(family.casefold())}(?!\w)', caption)
    return key == model_key(family) + size or (key == size and owner is not None)


def device_unit(heading):
    match = COUNT_HEADER.fullmatch(normalized_evidence(heading))
    if not match:
        return None
    return match[1] or match[2] or match[0].removesuffix('s')


def grid_counts(table, caption, family, size):
    rows = table['rows']
    if len(rows) < 2:
        return
    headings = [normalized_evidence(cell or '') for cell in rows[0]]
    models = [i for i, heading in enumerate(headings) if MODEL_HEADER.fullmatch(heading)]
    columns = [(i, unit) for i, heading in enumerate(headings) if (unit := device_unit(heading))]
    if len(models) != 1 or len(columns) != 1:
        return
    model_column, = models
    (column, unit), = columns
    for index, row in enumerate(rows[1:], 1):
        label = row[model_column] or ''
        quote = (row[column] or '').strip()
        bbox = table['cells'][index][column]
        if bbox and COUNT.fullmatch(quote) and matching_row(label, caption, family, size):
            yield unit, quote, bbox, index + 1, column + 1, rows[0][column].strip()


def word_lines(words):
    lines = []
    for word in sorted(words, key=lambda word: (word[1], word[0])):
        if not lines or word[1] - lines[-1][0][1] > 2.5:
            lines.append([])
        lines[-1].append(word)
    return [sorted(line, key=lambda word: word[0]) for line in lines]


def ruled_counts(page, table, caption, family, size):
    # The text strategy can clip a right-aligned label or split a heading at a
    # column boundary. Recover these two columns from the original word boxes.
    left, top, right, bottom = table['bbox']
    clip = pymupdf.Rect(left - 32, top - 5, right + 32, bottom + 4) & page.rect
    lines = word_lines(page.get_text('words', clip=clip))
    if not lines:
        return
    header, *data = lines
    words = [word[4].casefold() for word in header]
    for index in range(1, len(header) - 1):
        if not MODEL_HEADER.fullmatch(' '.join(words[:index])):
            continue
        heading = ' '.join(word[4] for word in header[index:index+2])
        if (unit := device_unit(heading)) is None:
            continue
        count_left = header[index][0]
        count_right = header[index+1][2]
        for row, line in enumerate(data, 2):
            label = ' '.join(word[4] for word in line if word[2] < count_left)
            counts = [word for word in line if word[0] >= count_left and word[2] <= count_right]
            if len(counts) == 1 and matching_row(label, caption, family, size):
                word, = counts
                if COUNT.fullmatch(word[4]):
                    yield unit, word[4], list(word[:4]), row, 2, heading
        return


def candidate_pages(doc_id):
    with connect() as con:
        return [row['page'] for row in con.execute(
            """SELECT page FROM pages WHERE document_id=? AND source='native'
               AND instr(lower(text),'table')>0
               AND (instr(lower(text),'gpu')>0 OR instr(lower(text),'tpu')>0)
               ORDER BY (lower(text) LIKE '%gpu count%' OR lower(text) LIKE '%tpu count%') DESC, page
               LIMIT 6""", (doc_id,))]


def read_counts(doc_id, pages, family, size):
    results = []
    with pymupdf.open(UPLOADS / f'{doc_id}.pdf') as pdf:
        for number in pages:
            page = pdf[number - 1]
            blocks = sorted((block for block in page.get_text('blocks') if block[6] == 0), key=lambda block: block[1])
            for table in page_tables(page, blocks, with_cells=True):
                caption = normalized_evidence(table['caption'])
                if not TRAINING_CAPTION.search(caption):
                    continue
                matches = list(grid_counts(table, caption, family, size))
                if not matches and table['method'] == 'ruled_text':
                    matches = list(ruled_counts(page, table, caption, family, size))
                results.extend((number, table['caption'], *match) for match in matches)
    return results


def table_hardware(paper, family, size):
    pages = candidate_pages(paper['id'])
    if not pages:
        return
    future = PDF_POOL.submit(read_counts, paper['id'], pages, family, size)
    try:
        results = future.result(timeout=15)
    except (OSError, RuntimeError, TimeoutError, ValueError):
        future.cancel()
        log.warning('Training table parsing failed for %s', paper['id'], exc_info=True)
        return
    for page, caption, unit, quote, bbox, row, column, heading in results:
        metadata = {'caption': caption, 'row': row, 'column': column, 'heading': heading,
                    'model': f'{family}-{size}'}
        value = frozenset({(int(quote.replace(',', '')), unit, '')})
        yield value, table_cell_evidence(paper, page, quote, bbox, metadata), f'{quote} {unit.upper()}'
