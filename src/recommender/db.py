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

-- Built once by scripts/build_pools.py and never changed, so every method is evaluated
-- on the same papers. Batches don't overlap, so a paper is in at most one pool.
CREATE TABLE IF NOT EXISTS pools (
    paper_id TEXT PRIMARY KEY REFERENCES papers (id),
    batch TEXT NOT NULL,
    split TEXT NOT NULL CHECK (split IN ('train', 'test')),
    source TEXT NOT NULL CHECK (source IN ('bm25', 'dense', 'both', 'random'))
);

CREATE TABLE IF NOT EXISTS queue (
    position INTEGER PRIMARY KEY,
    paper_id TEXT NOT NULL REFERENCES pools (paper_id),
    is_repeat INTEGER NOT NULL
);

-- Append-only: undoing a label adds a new row, and the latest row for a position counts.
-- label is 1 (relevant), 0 (not relevant) or NULL (skipped).
CREATE TABLE IF NOT EXISTS labels (
    position INTEGER NOT NULL REFERENCES queue (position),
    label INTEGER CHECK (label IN (0, 1)),
    labelled_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);
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

def get_papers(conn: sqlite3.Connection, ids: list[str] | None = None) -> list[Paper]:
    """Papers with the given IDs (in no particular order), or all papers by publication time."""
    sql = f"SELECT {', '.join(COLUMNS)} FROM papers"
    if ids is None:
        rows = conn.execute(sql + " ORDER BY published, id")
    else:
        rows = conn.execute(sql + f" WHERE id IN ({', '.join('?' * len(ids))})", ids)
    return [Paper(*row) for row in rows]

def papers_published(conn: sqlite3.Connection, start: str, end: str) -> list[Paper]:
    """Papers first submitted in [start, end), compared as ISO 8601 text."""
    rows = conn.execute(
        f"SELECT {', '.join(COLUMNS)} FROM papers "
        "WHERE published >= ? AND published < ? ORDER BY published, id",
        (start, end),
    )
    return [Paper(*row) for row in rows]

def mark_announced(conn: sqlite3.Connection, ids: list[str], day: str) -> int:
    """Record the announcement date, keeping the first one if a paper is seen again."""
    with conn:
        cur = conn.executemany(
            "UPDATE papers SET announced = ? WHERE id = ? AND announced IS NULL",
            [(day, i) for i in ids],
        )
    return cur.rowcount

def pools_exist(conn: sqlite3.Connection) -> bool:
    return conn.execute("SELECT EXISTS (SELECT 1 FROM pools)").fetchone()[0] == 1

def save_pools(
    conn: sqlite3.Connection,
    pools: list[tuple[str, str, dict[str, str]]],
    queue: list[tuple[str, bool]],
) -> None:
    """Store (batch, split, {paper_id: source}) pools and the labelling queue together."""
    with conn:
        conn.executemany(
            "INSERT INTO pools (paper_id, batch, split, source) VALUES (?, ?, ?, ?)",
            [(pid, batch, split, src) for batch, split, pool in pools for pid, src in pool.items()],
        )
        conn.executemany(
            "INSERT INTO queue (position, paper_id, is_repeat) VALUES (?, ?, ?)",
            [(pos, pid, int(rep)) for pos, (pid, rep) in enumerate(queue)],
        )

def add_label(conn: sqlite3.Connection, position: int, label: int | None) -> None:
    with conn:
        conn.execute("INSERT INTO labels (position, label) VALUES (?, ?)", (position, label))

def next_unlabelled(conn: sqlite3.Connection) -> int | None:
    row = conn.execute(
        "SELECT MIN(position) FROM queue WHERE position NOT IN (SELECT position FROM labels)"
    ).fetchone()
    return row[0]

def last_labelled(conn: sqlite3.Connection) -> int | None:
    row = conn.execute("SELECT position FROM labels ORDER BY rowid DESC LIMIT 1").fetchone()
    return row[0] if row else None

def queue_paper(conn: sqlite3.Connection, position: int) -> Paper:
    row = conn.execute(
        f"SELECT {', '.join('p.' + c for c in COLUMNS)} FROM queue q "
        "JOIN papers p ON p.id = q.paper_id WHERE q.position = ?",
        (position,),
    ).fetchone()
    return Paper(*row)

def queue_progress(conn: sqlite3.Connection) -> tuple[int, int]:
    done = conn.execute("SELECT COUNT(DISTINCT position) FROM labels").fetchone()[0]
    total = conn.execute("SELECT COUNT(*) FROM queue").fetchone()[0]
    return done, total

# The latest label per queue position, decided by insertion order (rowid) rather than
# timestamp, since two labels can land in the same second when undoing quickly.
LATEST = """
SELECT l.position, l.label FROM labels l
WHERE l.rowid = (SELECT MAX(rowid) FROM labels WHERE position = l.position)
"""

def pool_labels(conn: sqlite3.Connection) -> list[tuple[str, str, str, str, int | None]]:
    """(batch, split, paper_id, source, label) for every pool paper, from its first showing.

    label is None if the paper is unlabelled or was skipped; either way it must be left
    out of evaluation, not counted as non-relevant.
    """
    return conn.execute(
        f"""
        SELECT p.batch, p.split, p.paper_id, p.source, latest.label
        FROM pools p
        JOIN queue q ON q.paper_id = p.paper_id AND q.is_repeat = 0
        LEFT JOIN ({LATEST}) latest ON latest.position = q.position
        ORDER BY p.batch, p.paper_id
        """
    ).fetchall()

def repeat_pairs(conn: sqlite3.Connection) -> list[tuple[int, int]]:
    """(first label, repeat label) for repeated papers where both answers are a 0 or 1."""
    return conn.execute(
        f"""
        SELECT first.label, again.label
        FROM queue q1
        JOIN queue q2 ON q2.paper_id = q1.paper_id AND q1.is_repeat = 0 AND q2.is_repeat = 1
        JOIN ({LATEST}) first ON first.position = q1.position
        JOIN ({LATEST}) again ON again.position = q2.position
        WHERE first.label IS NOT NULL AND again.label IS NOT NULL
        """
    ).fetchall()
