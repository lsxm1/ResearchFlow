"""Bind training measurements to explicitly named variants within a source block."""
from decimal import Decimal
import re

from .comparison_facts import AMOUNT, FACETS, TIME, model_key, normalized_evidence
from .qa_facts import HARDWARE_QUANTITY, hardware_quantities
from .retrieval import retrieve
from .training_tables import table_hardware

SIZE = r'xxlarge|xlarge|large|medium|small|base|big|\d+(?:\.\d+)?[bmt]'
NAMED = re.compile(rf'(?<![\w-])([a-z][a-z0-9-]+?)(?:[ -]|\s+\()({SIZE})(?![a-z0-9-])', re.ASCII)
SENTENCE = re.compile(r'(?:(?<=[.!?])|(?<=[a-z)][.!?])\d{1,2})\s+(?=[a-z])')
TRAINING = re.compile(r'\b(?:pre-?)?train(?:ed|ing|s)?\b')
SHARED_HARDWARE = re.compile(r'\bwe trained (?:all )?our models on\b')
SHARED_DURATION = re.compile(r'\beach (?:pre-?)?training took\b')
STEPS = re.compile(rf'\bfor (?:a total of )?({AMOUNT})( million)? steps\b')
DURATION = re.compile(rf'\b(?:pre-?training|training) took (?:about )?({TIME})\b|'
                      rf'\bsteps(?: or |\s*\()({TIME})\b')
METRICS = {
    'hardware': ('训练硬件', 'training models GPU TPU accelerator', 'trained models NVIDIA Cloud TPU chips'),
    **{facet.key: (facet.label, facet.query, facet.retry) for facet in FACETS['schedule']},
}


def requested_training_variants(question):
    normalized = normalized_evidence(re.sub(r'《[^》]+》', '', question))
    return tuple(dict.fromkeys((model_key(family), size) for family, size in NAMED.findall(normalized)
                               if family not in {'the', 'our', 'and', 'each', 'both', 'for', 'with', 'two'}))


def training_scope(paper, variants):
    if paper['arxiv_id'] == '1706.03762' or paper['title'].strip().casefold() == 'attention is all you need':
        family = 'Transformer'
    else:
        title_prefix = re.match(r'^([a-z][a-z0-9-]{2,29})\s*:', paper['title'], re.I)
        family = title_prefix[1] if title_prefix else None
    sizes = [size for name, size in variants if family and name == model_key(family)]
    return (family, sizes[0]) if len(sizes) == 1 else None


def training_measurements(text, family, size, metric):
    name = re.escape(family.casefold())
    variants = rf'\b{name}(?:[ -]?|\s+\()(?P<named>{SIZE})\b'
    normalized = normalized_evidence(text)
    declares_family = re.search(rf'\bwe (?:pre-?)?train(?:ed)? (?:our |the )?{name} models?\b', normalized)
    if family == 'Transformer' or declares_family:
        variants += rf'|\b(?:the|our) (?P<local>{SIZE}) models?\b'
    variant = re.compile(variants)
    block_sizes = {match[match.lastgroup] for match in variant.finditer(normalized)}
    seen = set()
    for clause in SENTENCE.split(normalized):
        named = {match[match.lastgroup] for match in variant.finditer(clause)}
        shared = size in block_sizes and len(block_sizes) > 1 and not named and (
            metric == 'hardware' and SHARED_HARDWARE.search(clause)
            or metric == 'duration' and size in seen and len(seen) > 1 and SHARED_DURATION.search(clause)
        )
        seen.update(named)
        if not TRAINING.search(clause) or (named != {size} and not shared):
            continue
        if metric == 'hardware' and (quantities := hardware_quantities(clause)):
            display = '；'.join(m.group().strip() for m in HARDWARE_QUANTITY.finditer(clause))
            yield frozenset(quantities), display
        elif metric == 'steps' and (match := STEPS.search(clause)):
            value = Decimal(match[1].replace(',', '')) * (1_000_000 if match[2] else 1)
            yield value, f'{value:,} steps'
        elif metric == 'duration' and (match := DURATION.search(clause)):
            display = match[1] or match[2]
            amount, unit = display.split()
            yield Decimal(amount.replace(',', '')) * {'day': 1440, 'hour': 60, 'minute': 1}[unit.rstrip('s')], display


def experiment_excerpts(paper, scope, field, evidence, on_lookup, model_query=''):
    items = []
    for metric in ('hardware',) if field == 'hardware' else ('steps', 'duration'):
        label, primary, retry = METRICS[metric]
        item = {'label': label, 'status': 'missing', 'searched_queries': []}
        items.append(item)
        if not scope:
            continue
        family, size = scope
        item['label'] = f'{family}-{size} · {label}'
        values = {}
        for index, keywords in enumerate((f'{primary} {model_query}'.strip(), retry)):
            query = f'{family}{size} {size} {keywords}'
            item['searched_queries'].append(query)
            if index:
                on_lookup('retry_evidence', f'缺少「{item["label"]}」的明确证据，换词补查。')
            for source in retrieve(query, [paper['id']], limit=16):
                if source['source'] != 'native':
                    continue
                for value, display in training_measurements(source['text'], family, size, metric):
                    values.setdefault(value, (source, display))
            if values:
                break
        if metric == 'hardware':
            on_lookup('search_training_table', f'检查 {family}-{size} 对应训练表格的设备数量列。')
            item['searched_queries'].append(f'{family}-{size} 训练表格 GPU Count / TPU Count')
            for value, source, display in table_hardware(paper, family, size):
                current = values.get(value)
                if (current and 'table' in current[0]) or any(value < known for known in values):
                    continue
                values[value] = (source, display)
        alternatives = []
        for source, display in values.values():
            evidence[source['id']] = source
            alternatives.append({'label': item['label'], 'source_id': source['id'], 'quote': source['text'].strip(),
                                 'display': display})
        if len(alternatives) == 1:
            item.update(alternatives[0], status='verified_excerpt')
        elif alternatives:
            item.update(status='conflict', alternatives=alternatives)
    return items
