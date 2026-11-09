"""PDF ingestion: bounded page-at-a-time extraction, persistent progress and source boxes."""
import hashlib
import json
import logging
import math
import re
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import pymupdf
from .config import TESSDATA, UPLOADS
from .db import connect

log = logging.getLogger(__name__)
# MuPDF is not safe to use concurrently from threads. One PDF worker serializes access.
PDF_POOL = ThreadPoolExecutor(max_workers=1, thread_name_prefix='pdf')


def register_pdf(path: Path, filename: str, checksum: str, *, title=None, arxiv_id=None, version=None):
    """Register a staged PDF and start the same parser for uploads and downloads."""
    with path.open('rb') as handle:
        if b'%PDF-' not in handle.read(1024):
            raise ValueError('文件内容不是有效 PDF。')
    name = Path(filename.replace('\\', '/')).name[:200] or 'paper.pdf'
    with connect() as con:
        con.execute('''INSERT INTO documents(id,filename,sha256,title,arxiv_id,version) VALUES (?,?,?,?,?,?)
                       ON CONFLICT(sha256) DO UPDATE SET title=COALESCE(?,documents.title),
                         arxiv_id=COALESCE(excluded.arxiv_id,documents.arxiv_id),
                         version=COALESCE(excluded.version,documents.version)''',
                    (path.stem, name, checksum, title or name, arxiv_id, version, title))
        row = con.execute('SELECT * FROM documents WHERE sha256=?', (checksum,)).fetchone()
    duplicate = row['id'] != path.stem
    if duplicate:
        path.unlink(missing_ok=True)
    else:
        PDF_POOL.submit(parse_document, row['id'])
    return row, duplicate


def words(text: str):
    return re.findall(r'[a-z0-9]+|[\u4e00-\u9fff]', unicodedata.normalize('NFKC', text).casefold())


def fingerprint(text: str) -> set[int]:
    tokens = words(text)
    return {int.from_bytes(hashlib.blake2b(' '.join(tokens[i:i+4]).encode(), digest_size=8).digest(), 'big')
            for i in range(max(0, len(tokens) - 3))}


def ocr_available():
    return (TESSDATA / 'eng.traineddata').is_file()


def extract_blocks(page, use_ocr):
    blocks = [block for block in page.get_text('blocks', sort=True) if block[6] == 0]
    text = '\n'.join(block[4] for block in blocks)
    needs_ocr = len(text.strip()) < 20 and bool(page.get_image_info())
    if needs_ocr and use_ocr:
        # Keep the raster under four megapixels, including unusually large PDF pages.
        area = max(page.rect.width * page.rect.height, 1)
        dpi = min(150, int(72 * math.sqrt(4_000_000 / area)))
        if dpi >= 36:
            try:
                textpage = page.get_textpage_ocr(language='eng', dpi=dpi, full=True, tessdata=str(TESSDATA))
                recognized = page.get_text('blocks', textpage=textpage, sort=True)
                recognized = [block for block in recognized if block[6] == 0]
                recognized_text = '\n'.join(block[4] for block in recognized)
                if len(recognized_text.strip()) >= 20:
                    return recognized, recognized_text, 'ocr', False
            except RuntimeError:
                log.warning('OCR failed on PDF page %s', page.number + 1, exc_info=True)
    return blocks, text, 'native', needs_ocr


def arxiv_page_title(page):
    """Read the prominent title lines rather than the first line of the PDF."""
    lines = []
    for block in page.get_text('dict', flags=pymupdf.TEXTFLAGS_TEXT)['blocks']:
        for line in block['lines']:
            left, top, _, _ = line['bbox']
            if line['dir'][0] < .9 or top > page.rect.height * .30:
                continue
            text = ''.join(span['text'] for span in line['spans']).strip()
            if words(text):
                lines.append((round(top, 1), left, max(span['size'] for span in line['spans']), text))
    if not lines:
        return ''
    largest = max(line[2] for line in lines)
    prominent = sorted(line for line in lines if line[2] >= largest * .9)
    title_lines = []
    previous_top = prominent[0][0]
    for top, _, _, text in prominent:
        if top - previous_top > largest * 1.7:
            break
        title_lines.append(text)
        previous_top = top
    title = re.sub(r'(?<=\w)-\n(?=\w)', '', '\n'.join(title_lines))
    return ' '.join(title.split())[:240] if len(words(title)) >= 3 else ''


def refresh_arxiv_titles():
    """Upgrade older first-line titles without rebuilding the index or overwriting PDF metadata."""
    with connect() as con:
        rows = con.execute("""SELECT d.id, d.title, p.text FROM documents d
                            JOIN pages p ON p.document_id=d.id AND p.page=1
                            WHERE d.arxiv_id IS NOT NULL AND d.status IN ('ready','partial')""").fetchall()
    for row in rows:
        first_line = next((line.strip() for line in row['text'].splitlines() if line.strip()), '')
        if row['title'] != first_line:
            continue
        try:
            with pymupdf.open(UPLOADS / f"{row['id']}.pdf") as pdf:
                title = '' if (pdf.metadata or {}).get('title', '').strip() else arxiv_page_title(pdf[0])
            if title and title != row['title']:
                with connect() as con:
                    con.execute('UPDATE documents SET title=? WHERE id=? AND title=?',
                                (title, row['id'], row['title']))
        except (RuntimeError, ValueError, OSError):
            log.warning('Could not refresh title for %s', row['id'])


def parse_document(doc_id: str):
    try:
        with connect() as con:
            row = con.execute('SELECT * FROM documents WHERE id=?', (doc_id,)).fetchone()
            if row is None:
                return
            con.execute("UPDATE documents SET status='parsing', error=NULL, processed_pages=0, text_pages=0, ocr_pages=0 WHERE id=?", (doc_id,))
            con.execute('DELETE FROM chunks WHERE document_id=?', (doc_id,))
            con.execute('DELETE FROM pages WHERE document_id=?', (doc_id,))
        signature = set()
        text_pages = 0
        ocr_pages = 0
        use_ocr = ocr_available()
        with pymupdf.open(UPLOADS / f'{doc_id}.pdf') as pdf:
            if pdf.needs_pass:
                raise ValueError('加密PDF暂不支持，请先解密后上传。')
            count = len(pdf)
            if not 0 < count <= 2000:
                raise ValueError('当前版本支持1—2000页PDF。')
            with connect() as con:
                con.execute('UPDATE documents SET page_count=? WHERE id=?', (count, doc_id))
            for index in range(count):
                page = pdf[index]
                blocks, text, source, needs_ocr = extract_blocks(page, use_ocr)
                text_pages += int(len(text.strip()) >= 20)
                ocr_pages += int(needs_ocr)
                # Keep at most 4096 bottom hashes, rather than holding an entire book in memory.
                signature.update(fingerprint(text))
                signature = set(sorted(signature)[:4096])
                with connect() as con:
                    if not con.execute('SELECT 1 FROM documents WHERE id=?', (doc_id,)).fetchone():
                        return
                    con.execute('INSERT INTO pages(document_id,page,text,width,height,source) VALUES (?,?,?,?,?,?)',
                                (doc_id, index+1, text, page.rect.width, page.rect.height, source))
                    for b_index, block in enumerate(blocks):
                        if not block[4].strip():
                            continue
                        # Bound context length. Each split retains the original source block box.
                        for start in range(0, len(block[4]), 1100):
                            chunk_id = f'{doc_id}:{index+1}:{b_index}:{start}'
                            con.execute('INSERT INTO chunks VALUES (?,?,?,?,?)',
                                        (chunk_id, doc_id, index+1, block[4][start:start+1400], json.dumps(list(block[:4]))))
                    con.execute('UPDATE documents SET processed_pages=?, text_pages=?, ocr_pages=? WHERE id=?', (index+1, text_pages, ocr_pages, doc_id))
                    if index == 0:
                        match = re.search(r'arXiv:\s*(\d{4}\.\d{4,5})v(\d+)', text, re.I)
                        # Use PDF metadata / text, never the uploaded filename, for version hints.
                        title = (pdf.metadata or {}).get('title', '').strip()
                        if not title and (row['arxiv_id'] or match):
                            title = arxiv_page_title(page)
                        if not title:
                            lines = [line.strip() for line in text.splitlines() if line.strip()]
                            title = lines[0] if lines else row['filename']
                        con.execute('''UPDATE documents SET
                                      title=CASE WHEN arxiv_id IS NOT NULL AND title!=filename THEN title ELSE ? END,
                                      arxiv_id=COALESCE(arxiv_id,?), version=COALESCE(version,?) WHERE id=?''',
                                    (title[:240], match[1] if match else None, int(match[2]) if match else None, doc_id))
            status = 'needs_ocr' if text_pages == 0 else ('partial' if ocr_pages else 'ready')
            with connect() as con:
                con.execute('UPDATE documents SET status=?, signature=? WHERE id=?',
                            (status, json.dumps(sorted(signature)), doc_id))
    except Exception as exc:
        log.exception('PDF extraction failed for %s', doc_id)
        # Do not expose server filesystem paths or internal tracebacks to the browser.
        message = str(exc) if isinstance(exc, ValueError) else 'PDF解析失败，请检查文件是否损坏。'
        with connect() as con:
            con.execute("UPDATE documents SET status='failed', error=? WHERE id=?", (message, doc_id))


def duplicates(doc_id: str):
    with connect() as con:
        current = con.execute('SELECT * FROM documents WHERE id=?', (doc_id,)).fetchone()
        if current is None:
            return []
        other = con.execute('SELECT * FROM documents WHERE id!=?', (doc_id,)).fetchall()
    a = set(json.loads(current['signature']))
    found = []
    for row in other:
        b = set(json.loads(row['signature']))
        if not a or not b:
            continue
        # Bottom-k sketch estimate using the shared sample threshold.
        threshold = min(max(a), max(b))
        aa, bb = {x for x in a if x <= threshold}, {x for x in b if x <= threshold}
        similarity = len(aa & bb) / max(1, len(aa | bb))
        if similarity < 0.4:
            continue
        recommendation = '正文高度相似，请比较后选择保留版本。'
        if row['arxiv_id'] and row['arxiv_id'] == current['arxiv_id'] and row['version'] and current['version']:
            recommendation = '已有版本较新，建议保留已有版本。' if row['version'] > current['version'] else '当前版本较新，可保留当前版本。'
        found.append({'document_id': row['id'], 'title': row['title'], 'filename': row['filename'],
                      'similarity': round(similarity, 3), 'method': 'body_word_shingles', 'recommendation': recommendation})
    return sorted(found, key=lambda x: -x['similarity'])
