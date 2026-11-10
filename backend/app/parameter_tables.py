"""Locate a paper's maximum parameter count in an explicitly labeled table."""
from decimal import Decimal
import hashlib
import json
import logging
import re

from .comparison_facts import PARAMETER_SCALES
from .db import connect
from .documents import PDF_POOL
from .library_tools import parameter_table_pages

log = logging.getLogger(__name__)
HEADER = r'(?:#\s*|total\s+(?:number of\s+)?|number of\s+)?(?:parameters|params|nparams)'
UNIT = r'billion|million|b|m'
PARAMETER_HEADER = re.compile(rf'{HEADER}(?:\s*\((?P<unit>{UNIT})\))?', re.I)
COUNT = re.compile(rf'(?P<amount>(?:\d{{1,3}}(?:,\d{{3}})+|\d+)(?:\.\d+)?)(?:\s*(?P<unit>{UNIT}))?', re.I)


def candidate_pages(doc_id):
    with connect() as con:
        rows = con.execute("""SELECT page,text FROM pages WHERE document_id=? AND source='native'
                              AND instr(lower(text),'table')>0
                              AND (instr(lower(text),'parameters')>0 OR instr(lower(text),'params')>0)
                              ORDER BY page""", (doc_id,))
        ranked = [(not any(PARAMETER_HEADER.fullmatch(line.strip()) for line in row['text'].splitlines()), row['page'])
                  for row in rows]
    return [page for _, page in sorted(ranked)[:6]]


def axis_maximum(table, coordinates, header):
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
    return max(values, key=lambda item: item[0], default=None)


def table_maximum(table, prose):
    caption = ' '.join(table['caption'].split())
    if not re.search(r'\b(?:architecture|architectural|hyper[- ]?parameters|model sizes?|key model parameters)\b', caption, re.I):
        return None
    if re.search(r'\b(?:benchmark|baselines?|comparison|accuracy|performance)\b', caption, re.I):
        return None
    if prose is None and not re.search(r'\bour\b.{0,40}\bmodels?\b|\bmodels? (?:used|trained) in this work\b', caption, re.I):
        return None
    rows = table['rows']
    if len(rows) < 2:
        return None
    axes = (
        ('column', rows[0], lambda column: ((row, column) for row in range(1, len(rows)))),
        ('row', [row[0] for row in rows], lambda row: ((row, column) for column in range(1, len(rows[row])))),
    )
    matches = []
    for orientation, headings, coordinates in axes:
        headers = [(index, heading, match) for index, heading in enumerate(headings)
                   if (match := PARAMETER_HEADER.fullmatch(' '.join((heading or '').split())))]
        if len(headers) == 1:
            index, heading, header = headers[0]
            if maximum := axis_maximum(table, coordinates(index), header):
                matches.append((maximum, orientation, heading.strip()))
    if len(matches) != 1:
        return None
    (value, row, column, cell, bbox), orientation, heading = matches[0]
    if prose and abs(value - prose['value']) * 2 > prose['rounding_step']:
        return None
    location = {'caption': caption, 'row': row + 1, 'column': column + 1, 'heading': heading}
    return value, cell, bbox, location, orientation


def find_parameter_cell(paper, prose):
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
               if (match := table_maximum(table, prose))]
    if not matches or len({item[1] for item in matches}) != 1:
        return None
    page, value, quote, bbox, table, orientation = matches[0]
    doc_id = paper['id']
    location = json.dumps([page, bbox, table, quote], ensure_ascii=False, sort_keys=True)
    digest = hashlib.sha256(location.encode()).hexdigest()[:20]
    source_id = f'{doc_id}:{page}:table:{digest}'
    with connect() as con:
        con.execute('INSERT OR IGNORE INTO table_anchors(id,document_id,page,bbox) VALUES (?,?,?,?)',
                    (source_id, doc_id, page, json.dumps(bbox)))
    evidence = {'id': source_id, 'document_id': doc_id, 'title': paper['title'], 'filename': paper['filename'],
                'page': page, 'text': quote, 'bbox': bbox, 'score': 0, 'source': 'native', 'table': table,
                'file_url': f'/api/documents/{doc_id}/file#page={page}'}
    label = '配置表报告的最大模型规模' if orientation == 'row' else '模型配置表最大参数量'
    return evidence, {'kind': 'parameter_count', 'label': label, 'value': value, 'display': quote}
