from datetime import UTC, date, datetime

import numpy as np
import pytest

from recommender.db import (
    Paper,
    add_label,
    connect,
    insert_papers,
    last_labelled,
    next_unlabelled,
    pool_labels,
    queue_progress,
    repeat_pairs,
    save_pools,
)
from recommender.pools import batches, build_pool, build_queue, cutoffs, iso, split_batches

NO_HOLIDAYS = frozenset()

def utc(*args) -> datetime:
    return datetime(*args, tzinfo=UTC)

def test_cutoffs_are_weekdays_at_14_eastern_including_across_dst():
    cuts = cutoffs(date(2026, 10, 30), date(2026, 11, 2), NO_HOLIDAYS)
    # Fri 30 Oct is still EDT (UTC-4); US clocks go back on Sun 1 Nov, so Mon 2 Nov is EST.
    assert [iso(c) for c in cuts] == ["2026-10-30T18:00:00Z", "2026-11-02T19:00:00Z"]

def test_weekend_batch_runs_friday_to_monday():
    b = batches(date(2026, 10, 1), date(2026, 10, 13), NO_HOLIDAYS)
    monday = next(x for x in b if x.name == "2026-10-05")
    assert monday.opens == utc(2026, 10, 2, 18)
    assert monday.closes == utc(2026, 10, 5, 18)

def test_holiday_merges_neighbouring_batches():
    b = batches(date(2026, 10, 1), date(2026, 10, 20), frozenset({date(2026, 10, 12)}))
    names = [x.name for x in b]
    assert "2026-10-12" not in names
    tuesday = b[names.index("2026-10-13")]
    assert tuesday.opens == utc(2026, 10, 9, 18) # Friday's cutoff

def test_partial_batches_at_window_edges_are_dropped():
    # The window starts Thu 1 Oct 00:00 UTC, inside the batch that closes that afternoon.
    b = batches(date(2026, 10, 1), date(2026, 10, 3), NO_HOLIDAYS)
    assert [x.name for x in b] == ["2026-10-02"]

def test_batches_tile_time_without_gaps():
    b = batches(date(2026, 5, 1), date(2026, 10, 23), frozenset({date(2026, 7, 3)}))
    assert all(x.closes == y.opens for x, y in zip(b, b[1:]))

def test_split_keeps_every_train_batch_before_every_test_batch():
    b = batches(date(2026, 5, 1), date(2026, 10, 23), NO_HOLIDAYS)
    train, test = split_batches(b, 15, 12)
    assert len(train) == 12 and len(test) == 15
    assert test == b[-15:]
    assert max(t.closes for t in train) <= min(t.opens for t in test)
    assert len(set(train)) == 12

def test_split_needs_enough_batches():
    b = batches(date(2026, 10, 1), date(2026, 10, 10), NO_HOLIDAYS)
    with pytest.raises(ValueError, match="need"):
        split_batches(b, 15, 12)

def test_pool_has_each_retrievers_top_k_and_random_fill():
    ids = [f"p{i}" for i in range(100)]
    bm25 = np.arange(100, dtype=float) # p99 best
    dense = -np.arange(100, dtype=float) # p0 best
    dense[99] = 1000 # p99 is also dense's top paper
    pool = build_pool(ids, bm25, dense, np.random.default_rng(0), size=40, top_k=15)
    assert len(pool) == 40
    assert pool["p99"] == "both"
    assert {f"p{i}" for i in range(85, 100)} <= pool.keys()
    assert {f"p{i}" for i in range(14)} <= pool.keys()
    # Dense's top 15 is p99 (shared with BM25) plus p0..p13, so 29 distinct, 11 random.
    counts = {s: list(pool.values()).count(s) for s in ("bm25", "dense", "both", "random")}
    assert counts == {"bm25": 14, "dense": 14, "both": 1, "random": 11}

def test_pool_is_reproducible_from_the_seed():
    ids = [f"p{i}" for i in range(200)]
    scores = np.random.default_rng(1).random((2, 200))
    a = build_pool(ids, scores[0], scores[1], np.random.default_rng(7))
    b = build_pool(ids, scores[0], scores[1], np.random.default_rng(7))
    assert a == b

def test_small_batch_pool_takes_everything():
    ids = ["a", "b", "c"]
    pool = build_pool(ids, np.zeros(3), np.zeros(3), np.random.default_rng(0), size=40, top_k=15)
    assert sorted(pool) == ids

def test_queue_keeps_pools_whole_and_repeats_come_a_pool_later():
    pools = [[f"{p}-{i}" for i in range(40)] for p in "abcdefghij"]
    queue = build_queue(pools, np.random.default_rng(0), repeat_fraction=0.05)
    originals = [pid for pid, rep in queue if not rep]
    assert sorted(originals) == sorted(pid for pool in pools for pid in pool)
    # Without the repeats, each pool's papers are contiguous and pools keep their order.
    assert [pid[0] for pid in originals] == [p for p in "abcdefghij" for _ in range(40)]
    repeats = [(i, pid) for i, (pid, rep) in enumerate(queue) if rep]
    assert len(repeats) == 20
    first = {pid: i for i, (pid, rep) in enumerate(queue) if not rep}
    assert all(i - first[pid] > 40 for i, pid in repeats)

PAPER = Paper("2610.00001", "T", "A", "X", "gr-qc", "gr-qc", "2026-10-01T00:00:00Z")

def labelled_db(tmp_path):
    conn = connect(tmp_path / "papers.db")
    other = Paper("2610.00002", "T2", "A2", "X", "gr-qc", "gr-qc", "2026-10-01T00:00:01Z")
    insert_papers(conn, [PAPER, other])
    pools = [("2026-10-01", "test", {PAPER.id: "bm25", other.id: "random"})]
    save_pools(conn, pools, [(PAPER.id, False), (other.id, False), (PAPER.id, True)])
    return conn

def test_latest_label_wins_and_skip_counts_as_answered(tmp_path):
    conn = labelled_db(tmp_path)
    add_label(conn, 0, 0)
    add_label(conn, 0, 1) # undo, then relabel as relevant
    add_label(conn, 1, None) # skipped
    assert next_unlabelled(conn) == 2
    assert last_labelled(conn) == 1
    assert queue_progress(conn) == (2, 3)
    labels = {pid: label for _, _, pid, _, label in pool_labels(conn)}
    assert labels == {PAPER.id: 1, "2610.00002": None}

def test_repeat_label_does_not_overwrite_the_first(tmp_path):
    conn = labelled_db(tmp_path)
    add_label(conn, 0, 1)
    add_label(conn, 1, 0)
    add_label(conn, 2, 0) # the hidden repeat disagrees
    labels = {pid: label for _, _, pid, _, label in pool_labels(conn)}
    assert labels[PAPER.id] == 1
    assert repeat_pairs(conn) == [(1, 0)]
