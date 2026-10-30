# Results

Generated 2026-10-30 by `uv run scripts/evaluate.py`.

- Test set: 15 announcement batches, 2026-10-01 to 2026-10-22; 600 labelled papers, 68 relevant.
- Training data: 12 batches from 2026-05-04 to 2026-09-30; 479 labelled papers, 79 relevant. Every training paper was submitted before every test paper.
- Labels: one labeller. Agreement with my own earlier answer on 54 hidden repeats: 85%, Cohen's kappa 0.47.
- Each method reranks the same fixed 40-paper pool per test day. Metrics are averaged over days; brackets are 95% bootstrap intervals over the 15 days (10,000 resamples).

## Ranking quality

| Method | ndcg@10 | ndcg@5 | p@5 | p@10 | mrr |
|---|---|---|---|---|---|
| Random order | 0.183 (0.168-0.197) | 0.129 (0.113-0.148) | 0.114 (0.089-0.140) | 0.113 (0.089-0.139) | 0.281 (0.240-0.321) |
| BM25 score | 0.281 (0.179-0.386) | 0.181 (0.089-0.275) | 0.133 (0.067-0.213) | 0.160 (0.107-0.213) | 0.378 (0.218-0.558) |
| SPECTER2 score | 0.158 (0.076-0.249) | 0.135 (0.055-0.227) | 0.120 (0.053-0.200) | 0.087 (0.047-0.133) | 0.284 (0.145-0.453) |
| Logistic regression | 0.319 (0.220-0.420) | 0.197 (0.104-0.294) | 0.173 (0.093-0.267) | 0.220 (0.147-0.293) | 0.397 (0.240-0.577) |
| MLP | 0.341 (0.252-0.436) | 0.216 (0.118-0.323) | 0.213 (0.120-0.307) | 0.220 (0.167-0.273) | 0.378 (0.224-0.552) |
| Cross-encoder, pretrained | 0.172 (0.082-0.276) | 0.155 (0.061-0.267) | 0.147 (0.067-0.240) | 0.100 (0.053-0.147) | 0.263 (0.148-0.408) |
| Cross-encoder, fine-tuned | 0.396 (0.283-0.512) | 0.339 (0.215-0.479) | 0.307 (0.200-0.427) | 0.233 (0.160-0.307) | 0.526 (0.351-0.708) |

## Paired comparisons (ndcg@10)

Per-day differences, A minus B. p is an exact two-sided sign-flip permutation test over all 2^15 sign patterns; Holm adjusts for the 7 comparisons, which were fixed before the test set was scored.

| A | B | mean difference (95% CI) | days A better / worse | p | p (Holm) |
|---|---|---|---|---|---|
| Logistic regression | Random order | 0.136 (0.044-0.232) | 12 / 3 | 0.0150 | 0.0601 |
| MLP | Random order | 0.158 (0.071-0.252) | 11 / 4 | 0.0046 | 0.0232 |
| Cross-encoder, pretrained | Random order | -0.011 (-0.097-0.092) | 6 / 9 | 0.8381 | 0.8381 |
| Cross-encoder, fine-tuned | Random order | 0.213 (0.111-0.323) | 13 / 2 | 0.0015 | 0.0103 |
| Logistic regression | SPECTER2 score | 0.161 (0.036-0.290) | 10 / 2 | 0.0312 | 0.0938 |
| Cross-encoder, fine-tuned | Cross-encoder, pretrained | 0.224 (0.105-0.340) | 13 / 1 | 0.0026 | 0.0154 |
| Cross-encoder, fine-tuned | Logistic regression | 0.077 (-0.039-0.188) | 10 / 4 | 0.2251 | 0.4502 |

## Latency (CPU)

AMD64 Family 26 Model 68 Stepping 0, AuthenticAMD, 6 PyTorch threads. Median over the test pools after one warm-up call. Reranker times assume the paper embeddings and retrieval scores already exist; those costs are listed separately below.

| Step | ms |
|---|---|
| Logistic regression: rerank one 40-paper pool | 0.2 |
| MLP: rerank one 40-paper pool | 0.5 |
| Cross-encoder, pretrained: rerank one 40-paper pool | 1058.1 |
| Cross-encoder, fine-tuned: rerank one 40-paper pool | 3173.9 |
| SPECTER2 embedding of 40 new papers | 7071.9 |
| BM25 + dense profile scores for a whole batch (median 765 papers) | 53.2 |

## Learning curve (ndcg@10 on the test set)

Each learned model retrained on the most recent k training pools with its chosen settings. Descriptive only: nothing was chosen from these numbers. The cross-encoder is retrained on the GPU here, where run-to-run nondeterminism can move its scores slightly from the saved model's.

| Training pools | papers (relevant) | Logistic regression | MLP | Cross-encoder, fine-tuned |
|---|---|---|---|---|
| 3 | 120 (16) | 0.217 (0.117-0.336) | 0.200 (0.103-0.314) | 0.297 (0.185-0.417) |
| 6 | 240 (32) | 0.333 (0.235-0.436) | 0.291 (0.185-0.398) | 0.260 (0.167-0.359) |
| 9 | 359 (52) | 0.311 (0.204-0.422) | 0.265 (0.182-0.352) | 0.346 (0.212-0.482) |
| 12 | 479 (79) | 0.319 (0.218-0.418) | 0.341 (0.253-0.438) | 0.395 (0.284-0.508) |

## ndcg@10 per test day

| Day | papers (relevant) | Random order | BM25 score | SPECTER2 score | Logistic regression | MLP | Cross-encoder, pretrained | Cross-encoder, fine-tuned |
|---|---|---|---|---|---|---|---|---|
| 2026-10-01 | 40 (9) | 0.238 | 0.000 | 0.068 | 0.470 | 0.297 | 0.091 | 0.391 |
| 2026-10-02 | 40 (5) | 0.191 | 0.446 | 0.214 | 0.219 | 0.753 | 0.316 | 0.491 |
| 2026-10-05 | 40 (6) | 0.203 | 0.287 | 0.000 | 0.404 | 0.188 | 0.226 | 0.541 |
| 2026-10-06 | 40 (4) | 0.178 | 0.248 | 0.000 | 0.638 | 0.409 | 0.000 | 0.151 |
| 2026-10-07 | 40 (1) | 0.120 | 0.387 | 0.000 | 0.000 | 0.333 | 0.000 | 0.000 |
| 2026-10-08 | 40 (7) | 0.221 | 0.137 | 0.171 | 0.504 | 0.560 | 0.393 | 0.810 |
| 2026-10-09 | 40 (5) | 0.193 | 0.419 | 0.131 | 0.219 | 0.146 | 0.121 | 0.301 |
| 2026-10-13 | 40 (6) | 0.201 | 0.195 | 0.000 | 0.521 | 0.585 | 0.342 | 0.619 |
| 2026-10-14 | 40 (4) | 0.179 | 0.363 | 0.559 | 0.633 | 0.498 | 0.000 | 0.414 |
| 2026-10-15 | 40 (4) | 0.171 | 0.390 | 0.459 | 0.269 | 0.398 | 0.000 | 0.195 |
| 2026-10-16 | 40 (5) | 0.195 | 0.107 | 0.170 | 0.244 | 0.277 | 0.335 | 0.539 |
| 2026-10-19 | 40 (2) | 0.147 | 0.000 | 0.264 | 0.264 | 0.237 | 0.651 | 0.307 |
| 2026-10-20 | 40 (3) | 0.161 | 0.753 | 0.000 | 0.296 | 0.148 | 0.000 | 0.765 |
| 2026-10-21 | 40 (2) | 0.155 | 0.000 | 0.000 | 0.000 | 0.185 | 0.000 | 0.185 |
| 2026-10-22 | 40 (5) | 0.191 | 0.486 | 0.339 | 0.107 | 0.102 | 0.102 | 0.229 |
