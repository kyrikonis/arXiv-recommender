from dataclasses import replace

from recommender.db import (
    Paper,
    connect,
    get_papers,
    insert_papers,
    mark_announced,
    papers_published,
)

PAPER = Paper(
    id="2610.00989",
    title="Effects of Poisson Isocurvature Perturbations on Induced Gravitational Waves",
    abstract="A sudden transition...",
    authors="Zi-An Zhou, Cheng-Jun Fang, Zong-Kuan Guo",
    categories="astro-ph.CO gr-qc",
    primary_category="astro-ph.CO",
    published="2026-10-01T03:23:32Z",
)
OTHER = replace(PAPER, id="2610.03798", title="Dynamical thin shells in black hole ringdown")

def test_insert_counts_only_new(tmp_path):
    conn = connect(tmp_path / "papers.db")
    assert insert_papers(conn, [PAPER, OTHER]) == 2
    assert insert_papers(conn, [PAPER, OTHER]) == 0
    assert conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0] == 2

def test_duplicate_within_batch_stored_once(tmp_path):
    # The same paper arrives from several categories in one fetch.
    conn = connect(tmp_path / "papers.db")
    assert insert_papers(conn, [PAPER, PAPER]) == 1

def test_first_stored_version_wins(tmp_path):
    conn = connect(tmp_path / "papers.db")
    insert_papers(conn, [PAPER])
    insert_papers(conn, [replace(PAPER, title="Revised title")])
    title = conn.execute("SELECT title FROM papers WHERE id = ?", (PAPER.id,)).fetchone()[0]
    assert title == PAPER.title

def test_mark_announced_keeps_first_date(tmp_path):
    conn = connect(tmp_path / "papers.db")
    insert_papers(conn, [PAPER, OTHER])
    assert mark_announced(conn, [PAPER.id], "2026-10-02") == 1
    assert mark_announced(conn, [PAPER.id, OTHER.id, "9999.99999"], "2026-10-05") == 1
    rows = dict(conn.execute("SELECT id, announced FROM papers"))
    assert rows == {PAPER.id: "2026-10-02", OTHER.id: "2026-10-05"}

def test_get_papers_round_trips_fields(tmp_path):
    conn = connect(tmp_path / "papers.db")
    insert_papers(conn, [PAPER, OTHER])
    assert get_papers(conn, [OTHER.id]) == [OTHER]
    assert get_papers(conn) == [PAPER, OTHER]

def test_papers_published_end_is_exclusive(tmp_path):
    conn = connect(tmp_path / "papers.db")
    early = replace(PAPER, id="a", published="2026-10-01T23:59:59Z")
    midnight = replace(PAPER, id="b", published="2026-10-02T00:00:00Z")
    insert_papers(conn, [early, midnight])
    assert [p.id for p in papers_published(conn, "2026-10-01", "2026-10-02")] == ["a"]
    assert [p.id for p in papers_published(conn, "2026-10-02", "2026-10-03")] == ["b"]
