"""Recognize explicit comparison facts while retaining their verbatim source blocks."""
from dataclasses import dataclass
from decimal import Decimal
import re
import unicodedata


@dataclass(frozen=True)
class Facet:
    key: str
    label: str
    query: str
    retry: str
    patterns: tuple[tuple[str, str], ...]


ORIGINAL_ENCODER = (
    'encoder_layers',
    r'\bthe encoder is composed of a stack\b.{0,200}\bmulti-head self-attention\b.{0,120}\bfeed-forward network\b',
)
DERIVED_ENCODER = (
    'derived_encoder',
    r'\bbidirectional transformer encoder based on the original implementation described in vaswani et al\. \(2017\)',
)
AMOUNT = r'\d[\d,]*(?:\.\d+)?'
TIME = rf'{AMOUNT}\s+(?:hours?|days?|minutes?)'
PARAMETER_SCALES = {'trillion': 10**12, 'billion': 10**9, 'million': 10**6,
                    't': 10**12, 'b': 10**9, 'm': 10**6}
PARAMETER_UNIT = '|'.join(PARAMETER_SCALES)
PARAMETER_COUNT = rf'(?P<amount>{AMOUNT})\s*[-–]?\s*(?P<unit>{PARAMETER_UNIT})'
SERIES_COUNT = rf'{AMOUNT}\s*(?:{PARAMETER_UNIT})'
MODEL_RANGE_START = rf'{SERIES_COUNT}(?:\s+parameters)?'
MODEL_SERIES = rf'(?P<series>{SERIES_COUNT}(?:\s*(?:,\s*(?:and\s+)?|and\s+){SERIES_COUNT})+)'
MODEL_ANNOUNCEMENT = r'\bwe (?:train(?:ed)?|introduce|present|(?:develop and )?release)\b'
TOKENS = r'(?P<amount>\d{1,3}(?:,\d{3})+|\d+)'
FACETS = {
    'architecture': (
        Facet('encoder', '编码器设计',
              'encoder composed stack multi-head self-attention feed-forward bidirectional Transformer encoder',
              'encoder layers multi-head self-attention position-wise feed-forward original implementation Vaswani',
              (ORIGINAL_ENCODER, DERIVED_ENCODER)),
        Facet('backbone', '整体架构',
              'model architecture Transformer encoder decoder bidirectional original implementation',
              'overall architecture stacked self-attention both encoder decoder bidirectional Transformer encoder',
              (('encoder_decoder', r'\btransformer follows this overall architecture\b.{0,180}\bboth the encoder and decoder\b'),
               DERIVED_ENCODER)),
    ),
    'objective': (
        Facet('prediction', '训练时预测什么',
              'training loss masked LM likelihood next sentence prediction decoder predicted next-token probabilities',
              'decoder softmax next-token predictions masked language model next sentence pretraining loss',
              (('next_token', r'\bdecoder output to predicted next[- ]token probabilities\b'),
               ('masked_and_next_sentence', r'\btraining loss is the sum of\b.{0,90}\bmasked (?:lm|language model)\b.{0,80}\bnext sentence prediction\b'))),
    ),
    'schedule': (
        Facet('steps', '训练步数',
              'train trained for total steps batch size tokens per batch',
              'we train with batch size tokens per batch total number of training steps',
              (('step_count', rf'\b(?:we (?:pre[- ]?)?train(?:ed)?|models? (?:was|were) trained)\b[^.!?]{{0,145}}'
                rf'\bfor (?:a total of )?{AMOUNT}(?: million)? steps\b'),)),
        Facet('duration', '训练耗时',
              'training took days hours to complete wall clock training time',
              'pre-training took days to complete trained models steps hours duration',
              (('elapsed_time', rf'\b(?:pre[- ]?training|training) took (?:about )?{TIME}\b|'
                rf'\btrained\b[^.!?]{{0,140}}\bsteps(?: or |\s*\(){TIME}\b'),)),
    ),
    'parameters': (
        Facet('parameter_count', '论文报告的参数量',
              'we train largest model parameters trainable billion million parameter transformer',
              'model size largest model parameter count billion total parameters',
              (('parameter_count', rf'\bour largest\s*\({PARAMETER_COUNT}[- ]+parameters?\)\s+model\b'),
               ('parameter_count', rf'\bour (?:largest )?model\b[^.!?]{{0,65}}\b(?:is a|has|with)\s+{PARAMETER_COUNT}[- ]+parameters?\b'),
               ('parameter_count', rf'\bwe (?:train(?:ed)?|introduce|present)\b[^.!?]{{0,120}}\bwith {PARAMETER_COUNT}\s+parameters\b'),
               ('parameter_count', rf'\bwe (?:have )?(?:build|built)\s+a larger version of\b'
                rf'[^.!?]{{0,70}}\bwith {PARAMETER_COUNT}\s+parameters\b'),
               ('parameter_count', rf'{MODEL_ANNOUNCEMENT}[^.!?]{{0,230}}'
                rf'\b(?:ranging(?: in (?:size|scale))? from|from) {MODEL_RANGE_START} to {PARAMETER_COUNT}\s+parameters\b'),
               ('parameter_count', rf'\bwe (?:introduce|present|release|train)\s+(?!(?:the|a|an|results|models?)\b)'
                rf'[\w -]{{2,40}},\s+(?:a|an)\s+{PARAMETER_COUNT}[- ]+parameters?\b'),
               ('parameter_count', rf'\bwe (?:introduce|present|release|train)\b[^.!?]{{0,75}}'
                rf'\b(?:series|family|suite)\s*:\s*{MODEL_SERIES}\s+parameters?\b'),
               ('parameter_count', rf'{MODEL_ANNOUNCEMENT}\s+\w+ sizes of (?:our )?models\s*'
                rf'\({MODEL_SERIES}\s+parameters\)'),
               ('parameter_count', rf'\bwe (?:design(?:ed)?|build|built)\s+'
                rf'(?![^,.!?]{{0,65}}\b(?:baselines?|reference|comparison)\b)'
                rf'[^,.!?]{{0,65}}\bmodels,\s+(?:one )?with {MODEL_SERIES}\s+parameters\b'))),
    ),
    'context': (
        Facet('context_size', '上下文窗口或训练序列',
              'context window context size nctx tokens all models sequences training',
              'model context window length maximum tokens sampled contiguous sequences',
              (('context_window', rf'\b(?:all (?:our )?models use|we (?:also )?(?:use|increase))\b[^.!?]{{0,85}}'
                rf'\bcontext (?:window|size|length)\s+(?:of|to|from \d[\d,]* to)\s+(?:n\s*ctx\s*=\s*)?{TOKENS}\s+tokens?\b'),
               ('context_window', rf'\bmodel[’\x27]s context window\s*\(n\s*ctx\s*=\s*{TOKENS}\)'),
               ('training_sequence', rf'\bwe (?:pre[- ]?)?train(?:ed)?\b[^.!?]{{0,150}}\bsequences? of {TOKENS}\s+tokens?\b'))),
    ),
    'layer_norm': (
        Facet('norm_position', '层归一化放置方式',
              'layer normalization moved input sub-block pre-normalization same architecture',
              'layer norm pre-normalization input sub-block final self-attention residual',
              (('pre_norm_final', r'\blayer normalization\b.{0,95}\bmoved to the input of each sub-block\b'
                                  r'.{0,145}\badditional layer normalization was added after the final self-?attention block\b'),
               ('pre_norm', r'\bwe use the same model and architecture\b.{0,125}\bpre[- ]?normalization\b'),
               ('pre_norm', r'\b(?:we|our model) (?:use|uses|apply|applies) (?:a )?pre[- ]?(?:layer )?norm(?:alization)?\b'))),
    ),
}


def parameter_measurement(amount, unit):
    value = Decimal(amount.replace(',', ''))
    scale = PARAMETER_SCALES[unit]
    return {'value': int(value * scale), 'display': f'{value:g} {unit[0].upper()}',
            'rounding_step': int(Decimal(10) ** value.as_tuple().exponent * scale)}


def normalized_evidence(text):
    text = re.sub(r'(?<=\w)-[ \t]*\r?\n[ \t]*(?=\w)', '', text)
    return ' '.join(unicodedata.normalize('NFKC', text).split()).casefold()


def named_parameter_claims(text, family):
    name = re.escape(family.casefold())
    model = rf'(?P<model>{name}(?:[- ]?(?:small|base|medium|large|xlarge|xxlarge))?)'
    patterns = (
        rf'(?<![\w-]){model}\s+(?:contains|has)\s+{PARAMETER_COUNT}\s+parameters?(?=\s*(?:[.;!?)]|and\b|$))',
        rf'\bwe (?:begin by )?train(?:ed|ing)?\s+{model}\s+following (?:the )?[\w-]+ architecture'
        rf'\s*\((?:[^()]{{0,100}},\s*)?{PARAMETER_COUNT}\s+parameters\s*\)',
    )
    normalized = normalized_evidence(text)
    for pattern in patterns:
        for match in re.finditer(pattern, normalized):
            variant = re.sub(r'[- ]', '', match['model'])
            yield variant, parameter_measurement(match['amount'], match['unit'])


def identify(text, facet):
    normalized = normalized_evidence(text)
    for kind, pattern in facet.patterns:
        match = re.search(pattern, normalized)
        if not match:
            continue
        facts = {'kind': kind}
        if kind in FACT_LABELS:
            facts['label'] = FACT_LABELS[kind]
        if series := match.groupdict().get('series'):
            facts.update(max((parameter_measurement(item['amount'], item['unit'])
                              for item in re.finditer(PARAMETER_COUNT, series)), key=lambda item: item['value']))
        elif amount := match.groupdict().get('amount'):
            value = Decimal(amount.replace(',', ''))
            unit = match.groupdict().get('unit')
            if unit:
                facts.update(parameter_measurement(amount, unit))
            else:
                facts.update(value=int(value), display=f'{int(value):,} tokens')
        return facts
    return None


FACT_LABELS = {
    'parameter_count': '报告参数量',
    'context_window': '上下文窗口',
    'training_sequence': '训练序列长度',
    'pre_norm': '采用预归一化',
    'pre_norm_final': '子块输入处归一化，最后一个自注意力块之后另有归一化',
}


def measured_findings(papers, rows):
    findings = []
    for row in rows:
        key = row['key']
        if key not in ('parameters', 'context', 'layer_norm'):
            continue
        excerpts, observations, missing = [], [], []
        for paper, cell in zip(papers, row['cells'], strict=True):
            title = f"「{paper['title']}」"
            item = cell['excerpts'][0]
            if item['status'] != 'verified_excerpt':
                missing.append(title)
                continue
            excerpts.append(item)
            observations.append(f"{title}{item['label']} {item.get('display', '')}".rstrip())
        if len(excerpts) < 2:
            continue
        text = '，'.join(observations) + '。'
        windows = [item['value'] for item in excerpts if item['kind'] == 'context_window']
        if len(windows) >= 2 and max(windows) > min(windows):
            text += f'已明确报告的窗口相差最多 {max(windows) - min(windows):,} tokens。'
        if any(item['kind'] == 'training_sequence' for item in excerpts):
            text += '训练序列长度与模型的上下文窗口分别列示。'
        if key == 'parameters':
            counts = [item['value'] for item in excerpts]
            if max(counts) > min(counts) > 0:
                text += f'已核对数值的最大值约为最小值的 {max(counts) / min(counts):.1f} 倍。'
        if missing:
            text += '、'.join(missing) + '未检得可直接核对的对应证据。'
        findings.append(finding(row['label'] + '对照', text, excerpts))
    return findings


def grounded_findings(papers, rows):
    """Build findings from verified excerpts and keep gaps in the comparison."""
    by_field = {row['key']: row['cells'] for row in rows}
    findings = measured_findings(papers, rows)
    if len(papers) != 2:
        return findings
    for original, derivative in ((0, 1), (1, 0)):
        if papers[original]['arxiv_id'] != '1706.03762':
            continue
        architecture = by_field.get('architecture')
        if architecture:
            left, right = (architecture[index]['excerpts'] for index in (original, derivative))
            if {item.get('kind') for item in left} >= {'encoder_layers', 'encoder_decoder'}:
                derived = next((item for item in right if item.get('kind') == 'derived_encoder'), None)
                if derived:
                    findings.append(finding(
                        '架构的沿袭与差异',
                        '原版 Transformer 的编码器由多头自注意力和逐位置前馈网络组成，整体还包含解码器。'
                        '另一篇论文明确说明其双向 Transformer 编码器沿用了原版实现。',
                        [*left, derived],
                    ))
        objective = by_field.get('objective')
        if objective and papers[derivative]['arxiv_id'] == '1810.04805':
            left, right = (objective[index]['excerpts'][0] for index in (original, derivative))
            if left.get('kind') == 'next_token' and right.get('kind') == 'masked_and_next_sentence':
                findings.append(finding(
                    '训练目标的差异',
                    '原版 Transformer 从解码器输出预测下一个 token。BERT 的预训练损失结合掩码语言模型和下一句预测。',
                    [left, right],
                ))
    return findings


def finding(label, text, excerpts):
    return {'label': label, 'text': text,
            'citations': [{'source_id': item['source_id'], 'quote': item['quote']} for item in excerpts]}
