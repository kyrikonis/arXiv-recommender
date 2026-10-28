# Cross-validation: baselines

Generated 2026-10-28 by `scripts/train.py --model baselines`. Expanding-window folds over the 12 training pools: each of pools 7-12 is scored by a model trained on every pool before it. Training pools only; no test labels were used.

| config | mrr | ndcg@5 | p@5 | ndcg@10 | p@10 |
|---|---|---|---|---|---|
| method=random (mean of 200) | 0.347 | 0.170 | 0.134 | 0.214 | 0.131 |
| method=dense | 0.198 | 0.121 | 0.133 | 0.210 | 0.167 |
| method=bm25 | 0.217 | 0.072 | 0.067 | 0.155 | 0.117 |
