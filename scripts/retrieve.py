"""Show the top papers for one day by BM25 and by SPECTER2, as a sanity check.

    uv run scripts/retrieve.py --day 2026-10-15
"""

import argparse
import logging
import time
from datetime import date, timedelta

import numpy as np

from recommender.config import load_config, load_profile
from recommender.db import connect, get_papers, papers_published
from recommender.retrieve import (
    BM25,
    bm25_profile_scores,
    dense_profile_scores,
    load_embeddings,
    paper_text,
)

log = logging.getLogger("retrieve")

def show(title: str, papers, scores, k: int) -> None:
    print(f"\n{title}")
    for i in scores.argsort()[::-1][:k]:
        p = papers[i]
        print(f"  {scores[i]:6.3f}  {p.id:<12} [{p.primary_category}] {p.title[:90]}")

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--day", type=date.fromisoformat, required=True)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--how", choices=["max", "mean"], default="max")
    args = parser.parse_args()

    cfg = load_config()
    areas = load_profile()
    conn = connect(cfg.db_path)
    day = args.day.isoformat()
    candidates = papers_published(conn, day, (args.day + timedelta(days=1)).isoformat())
    seeds = get_papers(conn, [s for a in areas for s in a.seeds])
    log.info("%d candidates on %s, %d seeds, %d areas", len(candidates), day, len(seeds), len(areas))

    # BM25 statistics come only from papers published before the day, so nothing from the day itself or later shapes the scores.
    start = time.perf_counter()
    corpus = papers_published(conn, "0000", day)
    bm25 = BM25([paper_text(p.title, p.abstract, " ") for p in corpus])
    log.info("BM25 fitted on %d papers in %.1fs", len(corpus), time.perf_counter() - start)
    queries = [paper_text(p.title, p.abstract, " ") for p in seeds] + [a.description for a in areas]
    docs = [paper_text(p.title, p.abstract, " ") for p in candidates]
    show("BM25", candidates, bm25_profile_scores(bm25, queries, docs, args.how), args.k)

    ids, vectors = load_embeddings(cfg.data_dir / "embeddings")
    row = {pid: i for i, pid in enumerate(ids)}
    _, interests = load_embeddings(cfg.data_dir / "interests")
    profile = np.vstack([vectors[[row[p.id] for p in seeds]], interests])
    docs_vec = vectors[[row[p.id] for p in candidates]]
    show("SPECTER2", candidates, dense_profile_scores(profile, docs_vec, args.how), args.k)

    # Brute-force cost of scoring the whole corpus, to see whether an ANN index is needed.
    start = time.perf_counter()
    dense_profile_scores(profile, vectors, args.how)
    log.info(
        "scoring all %d papers against %d profile items took %.1f ms",
        len(vectors),
        len(profile),
        (time.perf_counter() - start) * 1000,
    )

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
