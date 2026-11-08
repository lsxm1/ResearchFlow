import json
import sqlite3
from contextlib import contextmanager
from .config import DB


@contextmanager
def connect():
    con = sqlite3.connect(DB, timeout=30)
    con.row_factory = sqlite3.Row
    con.execute('PRAGMA foreign_keys=ON')
    try:
        yield con
        con.commit()
    except Exception:
        con.rollback()
        raise
    finally:
        con.close()


def init_db():
    with connect() as con:
        con.execute('PRAGMA journal_mode=WAL')
        con.executescript('''
        CREATE TABLE IF NOT EXISTS documents (
            id TEXT PRIMARY KEY, filename TEXT NOT NULL, sha256 TEXT UNIQUE NOT NULL,
            title TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'queued',
            page_count INTEGER NOT NULL DEFAULT 0, processed_pages INTEGER NOT NULL DEFAULT 0,
            text_pages INTEGER NOT NULL DEFAULT 0, ocr_pages INTEGER NOT NULL DEFAULT 0, arxiv_id TEXT, version INTEGER,
            signature TEXT NOT NULL DEFAULT '[]', error TEXT,
            created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
        );
        CREATE TABLE IF NOT EXISTS pages (
            document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
            page INTEGER NOT NULL, text TEXT NOT NULL, width REAL NOT NULL, height REAL NOT NULL,
            PRIMARY KEY(document_id, page)
        );
        CREATE TABLE IF NOT EXISTS chunks (
            id TEXT PRIMARY KEY, document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
            page INTEGER NOT NULL, text TEXT NOT NULL, bbox TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS chunks_document ON chunks(document_id);
        CREATE TABLE IF NOT EXISTS tasks (
            id TEXT PRIMARY KEY, question TEXT NOT NULL, document_ids TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'queued', events TEXT NOT NULL DEFAULT '[]',
            result TEXT, created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
        );
        ''')


def document(row):
    if row is None:
        return None
    data = dict(row)
    data.pop('signature', None)
    data.pop('sha256', None)
    data['file_url'] = f"/api/documents/{data['id']}/file"
    return data


def task(row):
    data = dict(row)
    for key in ('document_ids', 'events', 'result'):
        data[key] = json.loads(data[key]) if data[key] else None
    return data
