"""Recognize explicit comparison facts while retaining their verbatim source blocks."""
from dataclasses import dataclass
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
}


def identify(text, facet):
    text = re.sub(r'(?<=\w)-[ \t]*\r?\n[ \t]*(?=\w)', '', text)
    normalized = ' '.join(unicodedata.normalize('NFKC', text).split()).casefold()
    return next((kind for kind, pattern in facet.patterns if re.search(pattern, normalized)), None)


def grounded_findings(papers, rows):
    """State only comparisons whose required facts were found on both sides."""
    by_field = {row['key']: row['cells'] for row in rows}
    findings = []
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
