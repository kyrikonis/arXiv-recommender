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

There is no approximate-nearest-neighbour index. Scoring all 86,937 stored
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

A paper counts as relevant only if it would earn one of the day's five digest slots ("if
this were in today's digest, would I be glad it was there?"); being on one of my topics
isn't enough. I first labelled around 400 papers based on whether I'd be generally interested;
about half came out relevant, including 40% of the randomly sampled papers, which would
leave nothing for a ranker to separate. Those labels are kept in
`labels/labels_broad.csv` but not used, and I started again with the stricter rule.

My final manual labels cover 1,079 of the 1,080 pool papers: 79 of 479
training papers and 68 of 600 test papers are relevant. On the 54 hidden repeats I gave
the same answer 85% of the time, a Cohen's kappa of 0.47. Part of every label is my own
inconsistency on borderline papers, which caps how closely any model can match them.

### Rerankers

Each reranker orders a pool's 40 papers, and each is a baseline for the next:

- **Logistic regression** on the 768-dimensional SPECTER2 embedding plus the two
  retrieval scores.
- **A small MLP** (one hidden layer) on the same features, averaged over 5 seeds.
- **A cross-encoder**, `cross-encoder/ms-marco-MiniLM-L6-v2`, reading my four interest
  descriptions and the paper's title and abstract together. It is reported both as
  downloaded and fine-tuned on my labels, to separate what pretraining gives from what
  the labels add.

Hyperparameters are chosen by expanding-window cross-validation on the 12 training pools
(each of the last six is scored by a model trained on every pool before it), so the test
labels are not read until the final evaluation. The scores for every configuration tried
are in `results/cv_*.md`.

```
uv run scripts/train.py --model baselines
uv run scripts/train.py --model logreg
uv run scripts/train.py --model mlp
uv run scripts/train.py --model cross
uv run scripts/evaluate.py
```

### Daily digest and feedback

A GitHub Actions workflow (`.github/workflows/daily.yml`) runs at 06:00 UTC, Monday to
Friday, after arXiv's feeds switch to the new listing. It fetches the announcement, drops
papers first submitted more than 14 days earlier (old papers newly cross-listed), scores
every remaining paper with the fine-tuned cross-encoder on CPU, and commits the top five
as JSON to the `digests` branch. The model weights aren't in the repo; the workflow
downloads them from the `models-v1` release. On 2026-10-30 a run on my PC's CPU scored that
day's 1,162 recent papers and took 153 s including the arXiv requests.

The digest scores every paper of the day rather than a BM25/SPECTER2 shortlist, because
neither retriever did better than random within the evaluation pools. That is a different
setting from the evaluation, which measured ranking within 40-paper pools.

A small FastAPI app reads the digests and records feedback in a separate table, exported
to `labels/feedback.csv` and never mixed into the evaluation labels:

```
uv run uvicorn recommender.api:app      # http://localhost:8000
```

or in Docker, keeping the database and labels on the host:

```
docker build -t arxiv-digest .
docker run -p 8000:8000 -v ./data:/app/data -v ./labels:/app/labels arxiv-digest
```

`GET /` shows the latest digest with relevant / not relevant buttons; `GET /digest`,
`GET /digest/{day}` and `POST /feedback` are the JSON endpoints. Set `DIGESTS` to a local
folder to serve digests from disk instead of the GitHub branch.

## Results

Scored once on the test set on 2026-10-30: 15 announcement batches from 2026-10-01 to
2026-10-22, 600 labelled papers of which 68 are relevant, with models trained on 479
labelled papers (79 relevant) from May to September. Each method reranks the same 40-paper
pool per day; numbers are means over the 15 days with 95% bootstrap intervals. The full
report, including P@10, nDCG@5 and every day's scores, is in `results/results.md`, and
`uv run scripts/evaluate.py` regenerates it.

| Method | nDCG@10 | P@5 | MRR |
|---|---|---|---|
| Random order | 0.183 (0.168-0.197) | 0.114 (0.089-0.140) | 0.281 (0.240-0.321) |
| BM25 profile score | 0.281 (0.179-0.386) | 0.133 (0.067-0.213) | 0.378 (0.218-0.558) |
| SPECTER2 profile score | 0.158 (0.076-0.249) | 0.120 (0.053-0.200) | 0.284 (0.145-0.453) |
| Logistic regression | 0.319 (0.220-0.420) | 0.173 (0.093-0.267) | 0.397 (0.240-0.577) |
| MLP | 0.341 (0.252-0.436) | 0.213 (0.120-0.307) | 0.378 (0.224-0.552) |
| Cross-encoder, pretrained only | 0.172 (0.082-0.276) | 0.147 (0.067-0.240) | 0.263 (0.148-0.408) |
| Cross-encoder, fine-tuned | 0.396 (0.283-0.512) | 0.307 (0.200-0.427) | 0.526 (0.351-0.708) |

Paired comparisons on nDCG@10, chosen before the test set was scored, with an exact
sign-flip permutation test over the 15 days and Holm's correction for the seven of them:

- The fine-tuned cross-encoder beats random order (+0.213, better on 13 of 15 days,
  adjusted p = 0.010) and the same model before fine-tuning (+0.224, adjusted p = 0.015).
  Fine-tuning on my labels is what made it useful: as downloaded, it ranks no better than
  random (p = 0.84).
- The MLP beats random order (+0.158, adjusted p = 0.023). Logistic regression is ahead of
  random on 12 of 15 days (+0.136) but misses significance after the correction (adjusted
  p = 0.060).
- The fine-tuned cross-encoder is ahead of logistic regression on average (+0.077) but the
  interval spans zero (adjusted p = 0.45); 15 test days can't separate the learned models.

In practical terms, the fine-tuned cross-encoder puts on average 1.5 relevant papers in a
5-paper digest, against 0.6 for random order.

On the CPU (Ryzen 5 9600X), reranking a 40-paper pool takes under 1 ms for logistic
regression and the MLP and about 3.1 s for the three-model cross-encoder ensemble.
Embedding 40 new papers with SPECTER2, which the other learned models need first, takes
about 7.3 s.

A learning curve (retraining on the most recent 3, 6, 9 and 12 training pools) is in
`results/results.md`. The fine-tuned cross-encoder goes from 0.297 to 0.395 nDCG@10, but
not monotonically, and every interval overlaps, so it doesn't show whether more labels
would help.

## What I learned

- My first labels ("is this in my interests?") made about half of all papers relevant,
  including 40% of random ones, so nothing could beat random order. Deciding what counts
  as relevant mattered more than any model choice.
- Topic similarity alone is weak inside a pool of on-topic papers: the SPECTER2 profile
  score is no better than random here. What separates the papers I'd want from the rest
  had to be learned from my labels.
- Cross-validation scores were optimistic. The fine-tuned cross-encoder scored 0.515
  nDCG@10 in cross-validation and 0.396 on test; the pretrained cross-encoder looked
  strong in cross-validation (0.383) and was no better than random on test. With five or
  six small validation pools, picking the best of many configurations flatters the
  winner, which is why only the untouched test set counts.
- Several grids had their best value on the edge and needed widening; checking that
  before trusting a tuned hyperparameter changed which settings were chosen.

## Limitations

- **Small test set.** 15 days and 68 relevant papers give wide intervals; the learned
  models can't be told apart from each other with confidence.
- **One labeller, moderately consistent.** On 54 hidden repeats I agreed with my own
  earlier answer 85% of the time (Cohen's kappa 0.47). Some of what the models get
  "wrong" is noise in my own labels.
- **Pooling.** Only 40 papers per day are labelled: BM25's top 15, SPECTER2's top 15 and a
  random sample. Relevant papers outside the pool are never seen, so these numbers measure
  ranking within the pool, not recall over the whole day.
- **Approximate days.** Test days are submission batches derived from arXiv's 14:00 ET
  deadline. Cross-lists of older papers, which arXiv announces later, aren't placed on the
  day they were actually announced.
- **Latency is measured on one machine and depends on what is already computed.** The
  linear model and MLP rerank in under a millisecond only because embeddings already
  exist; for new papers they first need SPECTER2 (7.3 s per 40 papers on CPU). The
  cross-encoder reads raw text, so end to end it is not the slow option, despite its
  3.1 s reranking time.
