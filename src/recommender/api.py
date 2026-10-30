"""Serve the daily digests and record feedback on them.

    uv run uvicorn recommender.api:app

DIGESTS sets where digests are read from: a folder, or by default the digests branch of the
GitHub repo, which the daily workflow writes to.
"""

import html
import os
from datetime import date
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from recommender.config import ROOT, load_config
from recommender.db import add_feedback, connect, export_feedback
from recommender.digest import read_digest

DEFAULT_DIGESTS = "https://raw.githubusercontent.com/kyrikonis/arXiv-recommender/digests"

class Feedback(BaseModel):
    paper_id: str
    day: date
    relevant: bool

def create_app(digests: str, db_path: Path, feedback_csv: Path) -> FastAPI:
    app = FastAPI(title="arXiv digest")

    def get_digest(day: date | None) -> dict:
        digest = read_digest(digests, day)
        if digest is None:
            raise HTTPException(404, f"no digest for {day or 'latest'}")
        return digest

    @app.get("/digest")
    def latest() -> dict:
        return get_digest(None)

    @app.get("/digest/{day}")
    def by_day(day: date) -> dict:
        return get_digest(day)

    @app.post("/feedback")
    def feedback(fb: Feedback) -> dict:
        # Only papers that were actually served can get feedback, so a typo can't become
        # a training label for some unrelated paper.
        if fb.paper_id not in {p["id"] for p in get_digest(fb.day)["papers"]}:
            raise HTTPException(404, f"{fb.paper_id} is not in the {fb.day} digest")
        conn = connect(db_path)
        add_feedback(conn, fb.paper_id, fb.day.isoformat(), int(fb.relevant))
        export_feedback(conn, feedback_csv)
        conn.close()
        return {"saved": True}

    @app.get("/", response_class=HTMLResponse)
    def page() -> str:
        return render(get_digest(None))

    return app

def render(digest: dict) -> str:
    items = []
    for p in digest["papers"]:
        pid = html.escape(p["id"])
        items.append(
            f"<li><a href='{html.escape(p['url'])}'>{html.escape(p['title'])}</a>"
            f"<div class='meta'>{html.escape(p['categories'])}</div>"
            f"<p>{html.escape(p['abstract'][:400])}...</p>"
            f"<button onclick=\"send('{pid}', true, this)\">relevant</button> "
            f"<button onclick=\"send('{pid}', false, this)\">not relevant</button></li>"
        )
    day = html.escape(digest["day"])
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>arXiv digest {day}</title>
<style>body{{font-family:sans-serif;max-width:46rem;margin:2rem auto;padding:0 1rem}}
li{{margin-bottom:1.5rem}} .meta{{color:#666;font-size:.85rem}}</style></head>
<body><h1>arXiv digest, {day}</h1>
<p>Top {len(digest['papers'])} of {digest['candidates']} papers.</p>
<ol>{''.join(items)}</ol>
<script>
async function send(id, relevant, button) {{
  const r = await fetch('/feedback', {{method: 'POST', headers: {{'Content-Type': 'application/json'}},
    body: JSON.stringify({{paper_id: id, day: '{day}', relevant: relevant}})}});
  button.parentElement.querySelectorAll('button').forEach(b => b.disabled = true);
  button.textContent += r.ok ? ' (saved)' : ' (failed)';
}}
</script></body></html>"""

app = create_app(
    os.environ.get("DIGESTS", DEFAULT_DIGESTS),
    load_config().db_path,
    ROOT / "labels" / "feedback.csv",
)
