"""Locate a paper's maximum or requested parameter count in a labeled table."""
from decimal import Decimal
import logging
import re

from .comparison_facts import PARAMETER_SCALES, PARAMETER_UNIT, model_key, normalized_evidence, variant_matches
from .db import connect
from .documents import PDF_POOL
from .library_tools import parameter_table_pages, table_cell_evidence

log = logging.getLogger(__name__)
HEADER = r'(?:#\s*|total\s+(?:number of\s+)?|number of\s+)?(?:parameters|params|nparams)'
PARAMETER_HEADER = re.compile(rf'{HEADER}(?:\s*\((?P<unit>{PARAMETER_UNIT})\))?', re.I)
COUNT = re.compile(rf'(?P<amount>(?:\d{{1,3}}(?:,\d{{3}})+|\d+)(?:\.\d+)?)(?:\s*(?P<unit>{PARAMETER_UNIT}))?', re.I)


def candidate_pages(doc_id):
    with connect() as con:
        rows = con.execute("""SELECT page,text FROM pages WHERE document_id=? AND source='native'
                              AND instr(lower(text),'table')>0
                              AND (instr(lower(text),'parameters')>0 OR instr(lower(text),'params')>0)
                              ORDER BY page""", (doc_id,))
        ranked = [(not any(PARAMETER_HEADER.fullmatch(line.strip()) for line in row['text'].splitlines()), row['page'])
                  for row in rows]
    return [page for _, page in sorted(ranked)[:6]]


def owned_model_family(paper):
    title = re.match(r'([a-z][a-z0-9-]{2,29})(\s+transformers?)?\s*:', paper['title'], re.I)
    if not title:
        return None
    family = title[1]
    name = re.escape(family.casefold())
    full_name = rf'{name}\s+transformers?' if title[2] else name
    naming = re.compile(
        rf'\b(?:we (?:call|name) our (?:approach|model|method)|which we call|'
        rf'we (?:introduce|present|propose) [^.!?;]{{0,100}}\b(?:model|architecture|method) (?:called|named))'
        rf'\s+{name}(?:\d{{1,2}})?(?![\w-])|'
        rf'\bwe [^.!?;]{{0,100}}\b(?:design(?:ing)?|develop(?:ing)?) [\w -]{{2,48}}'
        rf'\({name}\)\s+(?:architecture|model|method)\b|'
        rf'\bwe (?:introduce|[^.!?;]{{0,100}}\bwith the introduction of)\s+(?:the\s+)?{full_name}(?![\w-])'
    )
    with connect() as con:
        pages = con.execute("SELECT text FROM pages WHERE document_id=? AND source='native' AND page<=4",
                            (paper['id'],)).fetchall()
    return family if any(naming.search(normalized_evidence(page['text'])) for page in pages) else None


def model_rows(rows, family):
    columns = [i for i, heading in enumerate(rows[0]) if (heading or '').strip().casefold() == 'model']
    if len(columns) != 1:
        return None
    column = columns[0]
    variant = re.compile(rf'{re.escape(family)}(?:[-‐‑][a-z0-9.]+)*', re.I)
    return column, [i for i, row in enumerate(rows[1:], 1) if variant.fullmatch((row[column] or '').strip())]


def axis_maximum(table, coordinates, header, *, consistent=False):
    values = []
    for row, column in coordinates:
        cell = (table['rows'][row][column] or '').strip()
        match = COUNT.fullmatch(' '.join(cell.split()))
        bbox = table['cells'][row][column]
        if not match or not bbox:
            return None
        unit = match['unit'] or header['unit']
        value = Decimal(match['amount'].replace(',', '')) * (PARAMETER_SCALES[unit.casefold()] if unit else 1)
        if value <= 0 or value != int(value):
            return None
        if not unit and value < 10_000:
            return None  # Small bare values may use an unstated scale.
        values.append((int(value), row, column, cell, bbox))
    if consistent and len({item[0] for item in values}) > 1:
        return None
    return max(values, key=lambda item: item[0], default=None)


def table_maximum(table, prose, family=None, variants=()):
    caption = ' '.join(table['caption'].split())
    rows = table['rows']
    if len(rows) < 2:
        return None
    comparison = table.get('method') in {'grouped_config', 'merged_parameter_columns'} or re.search(
        r'\b(?:benchmark|baselines?|comparison|accuracy|performance)\b', caption, re.I)
    if comparison:
        if not family or not re.search(rf'(?<!\w){re.escape(family)}(?!\w)', caption, re.I):
            return None
        if re.search(r'\bablations?\b', caption, re.I) or not (scope := model_rows(rows, family)):
            return None
        model_column, data_rows = scope
    else:
        if not re.search(r'\b(?:architecture|architectural|hyper[- ]?parameters|model sizes?|key model parameters)\b', caption, re.I):
            return None
        if prose is None and not re.search(r'\bour\b.{0,40}\bmodels?\b|\bmodels? (?:used|trained) in this work\b', caption, re.I):
            return None
        data_rows = range(1, len(rows))
        if variants:
            columns = [i for i, heading in enumerate(rows[0]) if (heading or '').strip().casefold() == 'model']
            if len(columns) != 1:
                return None
            model_column = columns[0]
    if variants:
        data_rows = [row for row in data_rows
                     if variant_matches(rows[row][model_column] or '', family, variants)]
        if len({model_key(rows[row][model_column]) for row in data_rows}) != 1:
            return None
    axes = [('column', rows[0], lambda column: ((row, column) for row in data_rows))]
    if not comparison and not variants:
        axes.append(('row', [row[0] for row in rows], lambda row: ((row, col) for col in range(1, len(rows[row])))))
    matches = []
    for orientation, headings, coordinates in axes:
        headers = [(index, heading, match) for index, heading in enumerate(headings)
                   if (match := PARAMETER_HEADER.fullmatch(' '.join((heading or '').split())))]
        if len(headers) == 1:
            index, heading, header = headers[0]
            if maximum := axis_maximum(table, coordinates(index), header, consistent=bool(variants)):
                matches.append((maximum, orientation, heading.strip()))
    if len(matches) != 1:
        return None
    (value, row, column, cell, bbox), orientation, heading = matches[0]
    location = {'caption': caption, 'row': row + 1, 'column': column + 1, 'heading': heading}
    if comparison or variants:
        location['model'] = rows[row][model_column].strip()
    if prose and disagrees(value, prose):
        model = location.get('model', '')
        if not model or not variant_matches(model, family, (prose.get('variant'),)):
            return None
    return value, cell, bbox, location, orientation


def disagrees(value, prose):
    return abs(value - prose['value']) * 2 > prose['rounding_step']


def find_parameter_cell(paper, prose, family, variants=()):
    pages = candidate_pages(paper['id'])
    if not pages:
        return None
    future = PDF_POOL.submit(parameter_table_pages, paper['id'], pages)
    try:
        tables_by_page = future.result(timeout=15)
    except (OSError, RuntimeError, TimeoutError, ValueError):
        future.cancel()
        log.warning('Parameter table parsing failed for %s', paper['id'], exc_info=True)
        return None
    matches = [(page, *match) for page, tables in tables_by_page for table in tables
               if (match := table_maximum(table, prose, family, variants))]
    if not matches:
        return None
    if variants and len({(model_key(item[4]['model']), item[1]) for item in matches}) > 1:
        return None
    configs = [item for item in matches if 'model' not in item[4]]
    if len({item[1] for item in configs}) > 1:
        return None
    page, value, quote, bbox, table, orientation = configs[0] if configs else max(matches, key=lambda item: item[1])
    model = table.get('model')
    if model and any(item[4]['model'].casefold() == model.casefold() and item[1] != value for item in matches):
        return None
    evidence = table_cell_evidence(paper, page, quote, bbox, table)
    label = '配置表报告的最大模型规模' if orientation == 'row' else '模型配置表最大参数量'
    if variants:
        label = f'{model} 总参数量'
    elif model:
        label = '本论文模型在对照表中的最大参数量'
    return evidence, {'kind': 'parameter_count', 'label': label, 'value': value, 'display': quote,
                      'conflict': prose is not None and disagrees(value, prose)}
