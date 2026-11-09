"""Resolve exact titles and import PDFs from arXiv's fixed public endpoints."""
import hashlib
import re
import threading
import time
from uuid import uuid4
from xml.etree import ElementTree

import httpx

from .config import MAX_UPLOAD, UPLOADS
from .documents import register_pdf, words

ATOM = '{http://www.w3.org/2005/Atom}'
ARXIV_ID = re.compile(r'[0-9]{4}\.[0-9]{4,5}(?:v[1-9][0-9]*)?')
_search_lock = threading.Lock()
_last_search = 0.0


def paper_id(value: str):
    value = re.sub(r'^arxiv:\s*', '', value.strip(), flags=re.I)
    return value if ARXIV_ID.fullmatch(value) else None


def resolve_title(client: httpx.Client, title: str) -> tuple[str, str]:
    """Only import a title match. Broad terms can identify more than one paper."""
    global _last_search
    query = ' '.join(re.findall(r'[\w-]+', title))
    if not query:
        raise ValueError('请输入论文完整标题或 arXiv 编号。')
    with _search_lock:
        time.sleep(max(0, 3 - (time.monotonic() - _last_search)))
        try:
            response = client.get('https://export.arxiv.org/api/query', params={
                'search_query': f'ti:"{query}"', 'max_results': 5,
            })
            response.raise_for_status()
        finally:
            _last_search = time.monotonic()
    feed = ElementTree.fromstring(response.content)
    matches = {}
    for entry in feed.findall(f'{ATOM}entry'):
        name = entry.findtext(f'{ATOM}title', '')
        identifier = paper_id(entry.findtext(f'{ATOM}id', '').rsplit('/', 1)[-1])
        if identifier and words(name) == words(title):
            matches[identifier] = ' '.join(name.split())
    if len(matches) != 1:
        raise ValueError('arXiv 没有返回唯一的完整标题匹配，请使用论文的 arXiv 编号。')
    return matches.popitem()


def download_pdf(client: httpx.Client, identifier: str, title: str | None = None):
    if not ARXIV_ID.fullmatch(identifier):
        raise ValueError('无效的 arXiv 编号。')
    path = UPLOADS / f'{uuid4().hex}.pdf'
    checksum = hashlib.sha256()
    length = 0
    try:
        with client.stream('GET', f'https://arxiv.org/pdf/{identifier}') as response:
            response.raise_for_status()
            with path.open('wb') as handle:
                for block in response.iter_bytes(64 * 1024):
                    length += len(block)
                    if length > MAX_UPLOAD:
                        raise ValueError('下载的 PDF 超过上传大小限制。')
                    checksum.update(block)
                    handle.write(block)
        base_id, _, version = identifier.partition('v')
        row, _ = register_pdf(path, f'{identifier}.pdf', checksum.hexdigest(), title=title,
                              arxiv_id=base_id, version=int(version) if version else None)
        return row['id']
    except Exception:
        path.unlink(missing_ok=True)
        raise
