import logging
import sqlite3
from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from recommender.config import Config, load_profile
from recommender.db import Paper, get_papers, labelled_rows, papers_published
from recommender.pools import batches, iso
from recommender.retrieve import (
    BM25,
    bm25_profile_scores,
    dense_profile_scores,
    load_embeddings,
    paper_text,
)

log = logging.getLogger(__name__)

def doc_text(p: Paper) -> str:
    return paper_text(p.title, p.abstract, " ")

class ProfileScorer:
    """Both retrieval scores for any set of papers, against a fixed profile.

    Holds the fitted BM25 and the loaded embeddings, which are expensive to rebuild.
    """

    def __init__(self, conn: sqlite3.Connection, cfg: Config, corpus_end: str):
        areas = load_profile()
        seeds = get_papers(conn, [s for a in areas for s in a.seeds])
        self.queries = [doc_text(p) for p in seeds] + [a.description for a in areas]
        corpus = papers_published(conn, "0000", corpus_end)
        self.bm25 = BM25([doc_text(p) for p in corpus])
        log.info("BM25 fitted on %d papers before %s", len(corpus), corpus_end)
        ids, self.vectors = load_embeddings(cfg.data_dir / "embeddings")
        self.row = {pid: i for i, pid in enumerate(ids)}
        _, interests = load_embeddings(cfg.data_dir / "interests")
        self.profile = np.vstack([self.embeddings([p.id for p in seeds]), interests])

    def embeddings(self, ids: list[str]) -> np.ndarray:
        return self.vectors[[self.row[pid] for pid in ids]]

    def scores(self, papers: list[Paper]) -> tuple[np.ndarray, np.ndarray]:
        """(BM25, dense) profile scores, standardised over exactly these papers for BM25."""
        bm25 = bm25_profile_scores(self.bm25, self.queries, [doc_text(p) for p in papers])
        dense = dense_profile_scores(self.profile, self.embeddings([p.id for p in papers]))
        return bm25, dense

@dataclass(frozen=True)
class Pool:
    batch: str
    papers: list[Paper]
    labels: np.ndarray
    bm25: np.ndarray
    dense: np.ndarray
    embeddings: np.ndarray

    @property
    def features(self) -> np.ndarray:
        """SPECTER2 embedding plus the two retrieval scores, one row per paper."""
        return np.hstack([self.embeddings, self.bm25[:, None], self.dense[:, None]])

    @property
    def texts(self) -> list[str]:
        return [doc_text(p) for p in self.papers]

def first_test_open(conn: sqlite3.Connection, cfg: Config) -> str:
    first_test = conn.execute("SELECT MIN(batch) FROM pools WHERE split = 'test'").fetchone()[0]
    by_name = {b.name: b for b in batches(cfg.history_start, cfg.history_end, cfg.holidays)}
    return iso(by_name[first_test].opens)

def load_pools(conn: sqlite3.Connection, cfg: Config, split: str) -> list[Pool]:
    """Labelled pools for one split, in time order, with features for every paper."""
    by_batch = defaultdict(dict)
    for batch, pid, label in labelled_rows(conn, split):
        by_batch[batch][pid] = label
    all_batches = {b.name: b for b in batches(cfg.history_start, cfg.history_end, cfg.holidays)}
    # BM25 statistics come from before the test period for both splits, as when the
    # pools were built, so nothing from the test period shapes any feature.
    scorer = ProfileScorer(conn, cfg, first_test_open(conn, cfg))

    pools = []
    for name in sorted(by_batch):
        batch = all_batches[name]
        # Retrieval scores are standardised over the whole batch, as in production where
        # the reranker sees one day's candidates, not just the 40 that were labelled.
        candidates = papers_published(conn, iso(batch.opens), iso(batch.closes))
        bm25, dense = scorer.scores(candidates)
        index = {p.id: i for i, p in enumerate(candidates)}
        ids = sorted(by_batch[name])
        rows = [index[pid] for pid in ids]
        pools.append(
            Pool(
                batch=name,
                papers=[candidates[i] for i in rows],
                labels=np.array([by_batch[name][pid] for pid in ids]),
                bm25=bm25[rows],
                dense=dense[rows],
                embeddings=scorer.embeddings(ids),
            )
        )
    return pools
