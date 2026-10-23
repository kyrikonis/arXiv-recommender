# arXiv recommender

A personal recommender that ranks each day's new arXiv papers in machine learning, NLP,
quantum computing and astrophysics against my interests and shows me the top five. I've
built this mainly for personal use as I enjoy reading science papers, but also to learn how a small ML system fits together end to end, and to compare ranking methods with a proper temporal evaluation.

Work in progress.

## Running it

Requires [uv](https://docs.astral.sh/uv/).

```
uv sync
uv run pytest
```

Copy `.env.example` to `.env` (the contact email in it is optional), then fetch papers
into `data/papers.db`:

```
uv run --env-file .env scripts/fetch_history.py --start 2026-05-01 --end 2026-10-23
uv run --env-file .env scripts/fetch_daily.py
```

Both scripts keep to arXiv's limit of one request every three seconds and can be re-run
safely; papers already stored are skipped.

## Results

No results yet.
