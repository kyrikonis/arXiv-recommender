import json
from datetime import date, timedelta
from pathlib import Path

import httpx
import numpy as np

from recommender.db import Paper

# Old papers newly cross-listed into one of my categories show up in a day's listing (a
# 1999 paper did on 2026-10-23); a digest of new work should leave them out.
MAX_AGE_DAYS = 14

def recent(papers: list[Paper], day: date, max_age_days: int = MAX_AGE_DAYS) -> list[Paper]:
    cutoff = (day - timedelta(days=max_age_days)).isoformat()
    return [p for p in papers if p.published >= cutoff]

def build_digest(day: date, papers: list[Paper], scores: np.ndarray, size: int, model: str) -> dict:
    """The top `size` papers by score, best first, as a JSON-ready dict."""
    top = np.argsort(-scores, kind="stable")[:size]
    return {
        "day": day.isoformat(),
        "model": model,
        "candidates": len(papers),
        "papers": [
            {
                "id": papers[i].id,
                "title": papers[i].title,
                "authors": papers[i].authors,
                "categories": papers[i].categories,
                "abstract": papers[i].abstract,
                "published": papers[i].published,
                "score": float(scores[i]),
                "url": f"https://arxiv.org/abs/{papers[i].id}",
            }
            for i in top
        ],
    }

def save_digest(digest: dict, out_dir: Path) -> Path:
    """Write <day>.json and latest.json, which the API serves when no day is asked for."""
    out_dir.mkdir(parents=True, exist_ok=True)
    text = json.dumps(digest, indent=2, ensure_ascii=False)
    path = out_dir / f"{digest['day']}.json"
    path.write_text(text, encoding="utf-8")
    (out_dir / "latest.json").write_text(text, encoding="utf-8")
    return path

def read_digest(source: str, day: date | None = None) -> dict | None:
    """A digest from a local folder or a base URL (the digests branch on GitHub), or None."""
    name = f"{day.isoformat()}.json" if day else "latest.json"
    if source.startswith(("http://", "https://")):
        resp = httpx.get(f"{source.rstrip('/')}/{name}", timeout=30, follow_redirects=True)
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        return resp.json()
    path = Path(source) / name
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
