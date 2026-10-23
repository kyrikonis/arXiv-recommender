"""Fetch the papers in the latest arXiv announcement for the configured categories.

uv run --env-file .env scripts/fetch_daily.py
"""

import logging

from recommender.config import load_config
from recommender.db import connect, insert_papers, mark_announced
from recommender.ingest import Arxiv, fetch_announced, fetch_by_ids

log = logging.getLogger("fetch_daily")


def main() -> None:
    cfg = load_config()
    client = Arxiv()
    day, ids = fetch_announced(client, cfg.categories)
    if not ids:
        log.info("no new papers in the feeds (no announcement today?)")
        return

    papers = fetch_by_ids(client, ids)
    if len(papers) < len(ids):
        log.warning("feeds listed %d papers but the API returned %d", len(ids), len(papers))

    conn = connect(cfg.db_path)
    added = insert_papers(conn, papers)
    marked = mark_announced(conn, [p.id for p in papers], day)
    log.info("listing %s: %d papers, %d new, %d marked announced", day, len(papers), added, marked)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    main()
