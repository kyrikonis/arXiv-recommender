"""Build today's digest: fetch the latest announcement, rank it with the fine-tuned
cross-encoder on CPU, and write <out>/<day>.json. Run by the GitHub Actions workflow.

    uv run scripts/run_daily.py --out digests
"""

import argparse
import json
import logging
from datetime import date
from pathlib import Path

from recommender.config import ROOT, load_config, load_profile
from recommender.digest import build_digest, recent, save_digest
from recommender.features import doc_text
from recommender.ingest import Arxiv, fetch_announced, fetch_by_ids
from recommender.rerank import (
    CROSS_ENCODER,
    load_cross,
    predict_cross_ensemble,
    profile_query,
)

log = logging.getLogger("run_daily")

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--models", type=Path, default=ROOT / "models" / "cross")
    args = parser.parse_args()

    cfg = load_config()
    client = Arxiv()
    day, ids = fetch_announced(client, cfg.categories)
    if not ids:
        log.info("no papers in the feeds; no announcement to digest")
        return
    # The feeds keep showing the last listing until the next one, so a rerun (or a run on
    # a day without an announcement) would otherwise rebuild a digest that already exists.
    if (args.out / f"{day}.json").exists():
        log.info("digest for %s already exists", day)
        return

    papers = recent(fetch_by_ids(client, ids), date.fromisoformat(day))
    log.info("listing %s: %d announced, %d submitted recently", day, len(ids), len(papers))

    config = json.loads((args.models / "config.json").read_text())
    models = [load_cross(str(args.models / f"seed{s}"), "cpu") for s in range(config["seeds"])]
    query = profile_query([a.description for a in load_profile()])
    scores = predict_cross_ensemble(models, query, [doc_text(p) for p in papers], "cpu")

    model = f"{CROSS_ENCODER}, fine-tuned, {len(models)}-seed ensemble"
    digest = build_digest(date.fromisoformat(day), papers, scores, cfg.digest_size, model)
    path = save_digest(digest, args.out)
    for p in digest["papers"]:
        log.info("%7.3f %s %s", p["score"], p["id"], p["title"][:80])
    log.info("wrote %s", path)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    for noisy in ("httpx", "transformers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    main()
