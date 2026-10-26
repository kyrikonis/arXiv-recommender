"""Fetch the seed papers in profile.toml and check none overlaps the historical window.

    uv run scripts/fetch_seeds.py
"""

import logging
import sys

from recommender.config import load_config, load_profile
from recommender.db import connect, get_papers, insert_papers
from recommender.ingest import Arxiv, fetch_by_ids

log = logging.getLogger("fetch_seeds")

def main() -> None:
    cfg = load_config()
    seeds = [s for area in load_profile() for s in area.seeds]
    conn = connect(cfg.db_path)

    known = {p.id for p in get_papers(conn, seeds)}
    missing = [s for s in seeds if s not in known]
    if missing:
        added = insert_papers(conn, fetch_by_ids(Arxiv(), missing))
        log.info("fetched %d of %d missing seeds", added, len(missing))

    papers = {p.id: p for p in get_papers(conn, seeds)}
    problems = [f"{s}: not found on arXiv" for s in seeds if s not in papers]
    # Seeds inside the window could end up in training or test data: that is leakage.
    cutoff = cfg.history_start.isoformat()
    problems += [
        f"{p.id}: published {p.published[:10]}, not before {cutoff}"
        for p in papers.values()
        if p.published >= cutoff
    ]
    for problem in problems:
        log.error(problem)
    if problems:
        sys.exit(1)
    log.info("all %d seeds stored and published before %s", len(seeds), cutoff)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    main()
