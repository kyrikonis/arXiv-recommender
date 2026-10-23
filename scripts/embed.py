"""Embed every paper not yet embedded, and the interest descriptions, with SPECTER2.

    uv run scripts/embed.py
"""
import logging
import time
import numpy as np
import torch
from recommender.config import load_config, load_profile
from recommender.db import connect, get_papers
from recommender.retrieve import (
    embed,
    load_embeddings,
    load_specter,
    paper_text,
    save_embeddings,
)

log = logging.getLogger("embed")

def main() -> None:
    cfg = load_config()
    areas = load_profile()
    conn = connect(cfg.db_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer, model = load_specter(device)

    papers_path = cfg.data_dir / "embeddings"
    ids, vectors = load_embeddings(papers_path)
    done = set(ids)
    todo = [p for p in get_papers(conn) if p.id not in done]
    if todo:
        log.info("embedding %d papers on %s", len(todo), device)
        start = time.perf_counter()
        texts = [paper_text(p.title, p.abstract, tokenizer.sep_token) for p in todo]
        new = embed(texts, tokenizer, model, "proximity")
        elapsed = time.perf_counter() - start
        log.info("%.0fs, %.0f papers/s", elapsed, len(todo) / elapsed)
        save_embeddings(papers_path, ids + [p.id for p in todo], np.concatenate([vectors, new]))
    else:
        log.info("all %d papers already embedded", len(ids))

    # Always redone: there are only a handful, and descriptions may have been edited
    interests = embed([a.description for a in areas], tokenizer, model, "adhoc_query")
    save_embeddings(cfg.data_dir / "interests", [a.name for a in areas], interests)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("adapters").setLevel(logging.WARNING)
    main()
