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

Then fetch the seed papers in `profile.toml`, embed everything, and look at one day's top
papers:

```
uv run scripts/fetch_seeds.py
uv run scripts/embed.py
uv run scripts/retrieve.py --day 2026-10-15
```

Build the labelling pools (once) and label them; progress is saved after every key, and
the pools and labels are written to `labels/*.csv` on exit:

```
uv run scripts/build_pools.py
uv run scripts/label.py
uv run scripts/label.py --stats
```

The CSVs are committed. On a fresh clone, fetch the papers and then load my labels
instead of building new pools:

```
uv run scripts/labels_csv.py import
```

## How it works

The profile (`profile.toml`) is a short description and a handful of seed papers for each
of four interest areas. A paper's retrieval score is its highest similarity to any one
profile item, so a strong match to one area isn't diluted by the others.

- **BM25**, written out on top of scikit-learn's `CountVectorizer`. Each seed and
  description is a separate query; scores are standardised per query before taking the
  maximum, because raw BM25 grows with query length.
- **SPECTER2** embeddings (`allenai/specter2_base` with the proximity adapter for papers
  and the ad-hoc query adapter for the descriptions), compared by cosine similarity.

There is no approximate-nearest-neighbour index. On 2026-10-23, scoring all 86,937 stored
papers against the 28 profile items with a plain NumPy matrix product took 12.7 ms on a
Ryzen 5 9600X, so an index like FAISS would add complexity without saving anything at
this scale. Embedding the same papers took 224 s on an RTX 4070 Super.

### Labels

Papers are grouped into the batches arXiv announces together: everything submitted
between two weekday 14:00 US Eastern deadlines, with holidays merging neighbouring
batches. The last 15 complete batches before 2026-10-22 are the test set and 12 batches
spread over May to September are for training, so every training paper was submitted
before every test paper.

For each of these 27 batches I label a fixed pool of 40 papers: BM25's top 15, SPECTER2's
top 15, and random papers from the batch to make up the rest. The pools are built once
and never rebuilt, and every method is later scored on the same ones. The labelling tool
shows them shuffled, without saying which method picked them. About 5% of papers come
back a second time later in the queue, to measure how consistent my own labels are.

A paper counts as relevant only if it would earn one of the day's five digest slots: a
headline result, a readable review, or work likely to matter widely, not just anything on
my topics. I first labelled 401 papers with the looser "is this in my interests" rule;
about half came out relevant, including 40% of the randomly sampled papers, which would
leave nothing for a ranker to separate. Those labels are kept in
`labels/labels_broad.csv` but not used, and I started again with the stricter rule.

## Results

No results yet.
