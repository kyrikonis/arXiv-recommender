"""Build the fixed labelling pools for the train and test batches, once.

    uv run scripts/build_pools.py
"""

import logging
import sys
from collections import Counter

import numpy as np

from recommender.config import load_config, load_profile
from recommender.db import connect, get_papers, papers_published, pools_exist, save_pools
from recommender.pools import (
    TEST_BATCHES,
    TRAIN_BATCHES,
    batches,
    build_pool,
    build_queue,
    iso,
    split_batches,
)
from recommender.retrieve import (
    BM25,
    bm25_profile_scores,
    dense_profile_scores,
    load_embeddings,
    paper_text,
)

log = logging.getLogger("build_pools")

def main() -> None:
    cfg = load_config()
    conn = connect(cfg.db_path)
    # Labels refer to pool papers, so rebuilding would silently invalidate them.
    if pools_exist(conn):
        log.error("pools already exist in %s; they are built once and never rebuilt", cfg.db_path)
        sys.exit(1)

    all_batches = batches(cfg.history_start, cfg.history_end, cfg.holidays)
    train, test = split_batches(all_batches, TEST_BATCHES, TRAIN_BATCHES)
    log.info(
        "%d complete batches; train %s..%s, test %s..%s",
        len(all_batches), train[0].name, train[-1].name, test[0].name, test[-1].name,
    )

    areas = load_profile()
    seeds = get_papers(conn, [s for a in areas for s in a.seeds])
    queries = [paper_text(p.title, p.abstract, " ") for p in seeds] + [a.description for a in areas]
    ids, vectors = load_embeddings(cfg.data_dir / "embeddings")
    row = {pid: i for i, pid in enumerate(ids)}
    _, interests = load_embeddings(cfg.data_dir / "interests")
    profile = np.vstack([vectors[[row[p.id] for p in seeds]], interests])

    # One BM25 for every pool, fitted on everything submitted before the test period. Fitting
    # on "papers before each batch" left the May pools with almost no corpus to estimate
    # word rarity from. This still lets nothing from the test period shape any score.
    corpus = papers_published(conn, "0000", iso(test[0].opens))
    bm25 = BM25([paper_text(p.title, p.abstract, " ") for p in corpus])
    log.info("BM25 fitted on %d papers submitted before %s", len(corpus), iso(test[0].opens))

    rng = np.random.default_rng(cfg.seed)
    # Alternate train and test pools in the queue so any drift in what I count as
    # relevant, as labelling goes on, affects both splits alike.
    order = [b for pair in zip(train, test) for b in pair] + test[len(train):]
    pools = []
    for batch in order:
        split = "test" if batch in test else "train"
        candidates = papers_published(conn, iso(batch.opens), iso(batch.closes))
        docs = [paper_text(p.title, p.abstract, " ") for p in candidates]
        pool = build_pool(
            [p.id for p in candidates],
            bm25_profile_scores(bm25, queries, docs),
            dense_profile_scores(profile, vectors[[row[p.id] for p in candidates]]),
            rng,
        )
        pools.append((batch.name, split, pool))
        log.info(
            "%s %-5s %4d candidates -> %s", batch.name, split, len(candidates),
            dict(Counter(pool.values())),
        )

    queue = build_queue([list(pool) for _, _, pool in pools], rng)
    save_pools(conn, pools, queue)
    repeats = sum(rep for _, rep in queue)
    log.info("saved %d pools; queue of %d (%d repeats)", len(pools), len(queue), repeats)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
