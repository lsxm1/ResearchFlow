"""Persisted arXiv topics and user feedback, with scheduled refresh."""
import asyncio
import json
import time

import httpx

from .arxiv import search_papers
from .db import connect
from .documents import words


def topic_view(row):
    data = dict(row)
    results = json.loads(data['results'])
    with connect() as con:
        ratings = {item['paper_id']: item['rating'] for item in con.execute(
            'SELECT paper_id,rating FROM topic_feedback WHERE topic_id=?', (data['id'],))}
        library = set(words(' '.join(item['title'] for item in con.execute('SELECT title FROM documents'))))
    for paper in results:
        overlap = len(set(words(paper['title'])) & library)
        paper['rating'] = ratings.get(paper['id'], 0)
        paper['score'] = overlap + (2 if paper['code_url'] else 0) + 5 * paper['rating']
        paper['reason'] = '匹配追踪主题'
        if overlap:
            paper['reason'] += f'，与库内标题共有 {overlap} 个词'
        if paper['code_url']:
            paper['reason'] += '，摘要包含 GitHub 链接'
        if paper['rating']:
            paper['reason'] += '，已纳入你的反馈'
    data['results'] = sorted(results, key=lambda paper: -paper['score'])
    return data


def refresh_topic(topic_id, force=False):
    now = time.time()
    with connect() as con:
        con.execute('BEGIN IMMEDIATE')
        row = con.execute('SELECT * FROM tracked_topics WHERE id=?', (topic_id,)).fetchone()
        if row is None or (not force and row['next_at'] > now):
            return
        con.execute('UPDATE tracked_topics SET next_at=? WHERE id=?', (now + row['interval_minutes'] * 60, topic_id))
    try:
        with httpx.Client(timeout=httpx.Timeout(18, connect=8), trust_env=False) as client:
            papers = search_papers(client, row['query'])
    except Exception:
        with connect() as con:
            con.execute('UPDATE tracked_topics SET error=? WHERE id=?', ('arXiv 刷新失败，可稍后重试。', topic_id))
        return
    with connect() as con:
        con.execute('UPDATE tracked_topics SET results=?,updated_at=?,error=NULL WHERE id=?',
                    (json.dumps(papers, ensure_ascii=False), time.time(), topic_id))


def refresh_due():
    with connect() as con:
        due = [row['id'] for row in con.execute('SELECT id FROM tracked_topics WHERE next_at<=? ORDER BY next_at LIMIT 8', (time.time(),))]
    for topic_id in due:
        refresh_topic(topic_id)


async def run_scheduler():
    while True:
        await asyncio.to_thread(refresh_due)
        await asyncio.sleep(60)
