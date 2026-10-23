import sqlite3
from dataclasses import astuple, dataclass, fields
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS papers (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    abstract TEXT NOT NULL,
    authors TEXT NOT NULL,
    categories TEXT NOT NULL,
    primary_category TEXT NOT NULL,
    published TEXT NOT NULL,
    announced TEXT,
    fetched_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);
CREATE INDEX IF NOT EXISTS papers_published ON papers (published);
"""


@dataclass(frozen=True)
class Paper:
    id: str #unversioned: "2610.00844", or "hep-th/9711200" for old-style IDs
    title: str
    abstract: str
    authors: str # comma-separated
    categories: str # space-separated, as listed by arXiv
    primary_category: str
    published: str # v1 submission time, ISO 8601 UTC, so it sorts as text


COLUMNS = [f.name for f in fields(Paper)]


def connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.executescript(SCHEMA)
    return conn


def insert_papers(conn: sqlite3.Connection, papers: list[Paper]) -> int:
    """Insert papers, ignoring IDs already stored. Returns how many were new."""
    sql = (
        f"INSERT OR IGNORE INTO papers ({', '.join(COLUMNS)}) "
        f"VALUES ({', '.join('?' * len(COLUMNS))})"
    )
    with conn:
        cur = conn.executemany(sql, [astuple(p) for p in papers])
    return cur.rowcount


def mark_announced(conn: sqlite3.Connection, ids: list[str], day: str) -> int:
    """Record the announcement date, keeping the first one if a paper is seen again."""
    with conn:
        cur = conn.executemany(
            "UPDATE papers SET announced = ? WHERE id = ? AND announced IS NULL",
            [(day, i) for i in ids],
        )
    return cur.rowcount
