"""Extract directly reported values for narrowly scoped paper questions."""
import re
import unicodedata
from decimal import Decimal

from .documents import words
from .retrieval import retrieve

ADAM = re.compile(r'(?<![a-z0-9_])adam(?![a-z0-9_])')
BETA = re.compile(r'(?<![a-z0-9_])(?:beta|β)(?:[\s_-]*(\d+))?(?![a-z0-9_])')
QUESTION_WORDS = set(
    'what is are were was did does the this paper use used set to for in during training '
    'optimizer hyperparameter hyperparameters parameter parameters coefficient coefficients '
    'value values and respectively of report reports reported its'.split()
)
QUESTION_PHRASES = re.compile(
    r'这篇论文|该论文|本文|文中|优化器|超参数|训练时|训练|使用|采用|用了|设置|设为|取值|'
    r'分别|参数|系数|多少|什么|请问|等于|为|值|的|和|与|及|是|用'
)
ADAM_SETTINGS = re.compile(
    r'\b(?:we\s+(?:use[ds]?|adopt(?:ed|s)?|employ(?:ed|s)?)|training\s+used)'
    r'\s+(?:the\s+)?adam(?:\s+optimizer)?(?:\s*\[\d+\])?\s+with\b'
    r'(?P<settings>(?:[^.!?;]|\.(?=\d)){0,260})',
    re.I,
)
BETA_ASSIGNMENT = re.compile(
    r'(?<!\w)(?:beta|β)[\s_]*(?P<index>[0-9₀-₉]+)\s*=\s*'
    r'(?P<value>0(?:\.\d+)?|\.\d+)(?![\w%]|[.,]\d|\s*(?:[/±]|[-–−]\s*[.\d]|to\s+\d))',
    re.I,
)
SETTINGS_END = re.compile(r'\b(?:while|whereas|versus|but|adamw?|sgd|rmsprop|adagrad)\b', re.I)


def requested_adam_betas(question):
    normalized = unicodedata.normalize('NFKC', question).casefold()
    betas = BETA.findall(normalized)
    if not betas or not ADAM.search(normalized):
        return ()
    remaining = QUESTION_PHRASES.sub(' ', ADAM.sub(' ', BETA.sub(' ', normalized)))
    if set(words(remaining)) - QUESTION_WORDS:
        return ()  # Leave questions about specific experiments or other settings to general Q&A.
    return tuple(dict.fromkeys(index for index in betas if index)) or ('1', '2')


def adam_beta_evidence(document_id, indices):
    matches = {index: {} for index in indices}
    sources = {}
    for source in retrieve('Adam optimizer', [document_id], limit=16):
        if source['source'] != 'native':
            continue
        for clause in ADAM_SETTINGS.finditer(source['text']):
            settings = clause['settings']
            boundary = SETTINGS_END.search(settings)
            stop = boundary.start() if boundary else len(settings)
            for assignment in BETA_ASSIGNMENT.finditer(settings, 0, stop):
                index = unicodedata.normalize('NFKC', assignment['index'])
                if index not in matches:
                    continue
                value = assignment['value']
                end = clause.start('settings') + assignment.end()
                quote = source['text'][clause.start():end]
                claim = {'text': f'Adam β{index} = {value}', 'source_id': source['id'], 'quote': quote}
                matches[index].setdefault(Decimal(value), claim)
                sources[source['id']] = source

    claims = []
    warnings = []
    for index, values in matches.items():
        if len(values) == 1:
            claims.extend(values.values())
        elif values:
            warnings.append(f'找到的训练段落对 Adam β{index} 报告了不同数值，请核对原文。')
        else:
            warnings.append(f'在找到的训练段落中，缺少能核对 Adam β{index} 数值的证据。')
    return claims, list(sources.values()), warnings
