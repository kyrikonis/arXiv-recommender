"""Fetch historical papers for labelling, one category and one week at a time.

uv run --env-file .env scripts/fetch_history.py --start 2026-05-01 --end 2026-10-23
"""

import argparse
import logging
from datetime import date, timedelta

from recommender.config import load_config
from recommender.db import connect, insert_papers
from recommender.ingest import Arxiv, fetch_range

log = logging.getLogger("fetch_history")

def weeks(start: date, end: date):
    while start < end:
        stop = min(start + timedelta(days=7), end)
        yield start, stop
        start = stop

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=date.fromisoformat, required=True)
    parser.add_argument("--end", type=date.fromisoformat, required=True, help="exclusive")
    parser.add_argument("--categories", nargs="+", help="defaults to config.toml")
    args = parser.parse_args()

    cfg = load_config()
    conn = connect(cfg.db_path)
    client = Arxiv()
    for category in args.categories or cfg.categories:
        for start, stop in weeks(args.start, args.end):
            papers = fetch_range(client, category, start, stop)
            added = insert_papers(conn, papers)
            log.info("%s %s to %s: %d fetched, %d new", category, start, stop, len(papers), added)

    total = conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0]
    log.info("done, %d papers in %s", total, cfg.db_path)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    main()