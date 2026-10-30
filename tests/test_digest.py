import csv
from datetime import date

import numpy as np
from fastapi.testclient import TestClient

from recommender.api import create_app
from recommender.db import Paper
from recommender.digest import build_digest, read_digest, recent, save_digest

DAY = date(2026, 10, 23)

def paper(i: int, published: str = "2026-10-22T10:00:00Z") -> Paper:
    return Paper(f"2610.0000{i}", f"Title {i}", "Abstract", "A. Author", "gr-qc", "gr-qc", published)

def test_recent_drops_old_cross_lists():
    papers = [paper(1), paper(2, "1999-12-03T17:04:09Z"), paper(3, "2026-10-09T00:00:00Z")]
    assert [p.id for p in recent(papers, DAY)] == ["2610.00001", "2610.00003"]

def test_digest_takes_the_top_scores_in_order():
    papers = [paper(i) for i in range(7)]
    scores = np.array([0.1, 0.9, 0.3, 0.8, 0.2, 0.7, 0.0])
    digest = build_digest(DAY, papers, scores, size=5, model="m")
    assert [p["id"][-1] for p in digest["papers"]] == ["1", "3", "5", "2", "4"]
    assert digest["candidates"] == 7 and digest["day"] == "2026-10-23"

def test_digest_with_fewer_papers_than_slots():
    digest = build_digest(DAY, [paper(1)], np.array([0.5]), size=5, model="m")
    assert len(digest["papers"]) == 1

def test_saved_digest_is_readable_by_day_and_as_latest(tmp_path):
    digest = build_digest(DAY, [paper(1)], np.array([0.5]), size=5, model="m")
    save_digest(digest, tmp_path)
    assert read_digest(str(tmp_path), DAY) == digest
    assert read_digest(str(tmp_path)) == digest
    assert read_digest(str(tmp_path), date(2026, 10, 22)) is None

def client(tmp_path):
    digest = build_digest(DAY, [paper(1), paper(2)], np.array([0.9, 0.1]), size=5, model="m")
    save_digest(digest, tmp_path / "digests")
    app = create_app(str(tmp_path / "digests"), tmp_path / "papers.db", tmp_path / "feedback.csv")
    return TestClient(app)

def test_api_serves_latest_and_by_day(tmp_path):
    c = client(tmp_path)
    assert c.get("/digest").json()["day"] == "2026-10-23"
    assert c.get("/digest/2026-10-23").status_code == 200
    assert c.get("/digest/2026-10-22").status_code == 404
    assert c.get("/digest/not-a-date").status_code == 422
    assert "Title 1" in c.get("/").text

def test_feedback_is_stored_and_exported(tmp_path):
    c = client(tmp_path)
    body = {"paper_id": "2610.00001", "day": "2026-10-23", "relevant": True}
    assert c.post("/feedback", json=body).json() == {"saved": True}
    c.post("/feedback", json={**body, "relevant": False})
    with (tmp_path / "feedback.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    assert [(r["paper_id"], r["digest_day"], r["label"]) for r in rows] == [
        ("2610.00001", "2026-10-23", "1"),
        ("2610.00001", "2026-10-23", "0"),
    ]

def test_feedback_only_for_papers_in_that_digest(tmp_path):
    c = client(tmp_path)
    body = {"paper_id": "2610.09999", "day": "2026-10-23", "relevant": True}
    assert c.post("/feedback", json=body).status_code == 404
    assert c.post("/feedback", json={**body, "day": "2026-10-22"}).status_code == 404
    assert not (tmp_path / "feedback.csv").exists()
