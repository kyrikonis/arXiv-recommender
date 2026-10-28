# Cross-validation: mlp

Generated 2026-10-28 by `scripts/train.py --model mlp`. Expanding-window folds over the 12 training pools: each of pools 7-12 is scored by a model trained on every pool before it. Training pools only; no test labels were used.

| config | mrr | ndcg@5 | p@5 | ndcg@10 | p@10 |
|---|---|---|---|---|---|
| hidden=64, dropout=0.5, weight_decay=0.01, epochs=20 | 0.400 | 0.183 | 0.133 | 0.253 | 0.167 |
| hidden=64, dropout=0.2, weight_decay=0.01, epochs=20 | 0.408 | 0.187 | 0.133 | 0.251 | 0.167 |
| hidden=64, dropout=0.2, weight_decay=0.0001, epochs=20 | 0.406 | 0.187 | 0.133 | 0.250 | 0.167 |
| hidden=256, dropout=0.5, weight_decay=0.0001, epochs=20 | 0.433 | 0.180 | 0.133 | 0.250 | 0.167 |
| hidden=256, dropout=0.5, weight_decay=0.01, epochs=20 | 0.433 | 0.180 | 0.133 | 0.250 | 0.167 |
| hidden=64, dropout=0.2, weight_decay=0.0001, epochs=60 | 0.383 | 0.209 | 0.167 | 0.245 | 0.167 |
| hidden=64, dropout=0.2, weight_decay=0.01, epochs=60 | 0.384 | 0.209 | 0.167 | 0.245 | 0.167 |
| hidden=64, dropout=0.5, weight_decay=0.0001, epochs=20 | 0.400 | 0.169 | 0.133 | 0.243 | 0.167 |
| hidden=64, dropout=0.5, weight_decay=0.0001, epochs=60 | 0.383 | 0.200 | 0.167 | 0.241 | 0.167 |
| hidden=256, dropout=0.2, weight_decay=0.0001, epochs=20 | 0.333 | 0.189 | 0.167 | 0.238 | 0.167 |
| hidden=256, dropout=0.2, weight_decay=0.01, epochs=20 | 0.333 | 0.189 | 0.167 | 0.238 | 0.167 |
| hidden=64, dropout=0.5, weight_decay=0.01, epochs=60 | 0.383 | 0.195 | 0.167 | 0.238 | 0.167 |
| hidden=256, dropout=0.2, weight_decay=0.0001, epochs=60 | 0.333 | 0.184 | 0.167 | 0.234 | 0.167 |
| hidden=256, dropout=0.2, weight_decay=0.01, epochs=60 | 0.333 | 0.184 | 0.167 | 0.234 | 0.167 |
| hidden=256, dropout=0.5, weight_decay=0.01, epochs=60 | 0.250 | 0.132 | 0.133 | 0.213 | 0.167 |
| hidden=256, dropout=0.5, weight_decay=0.0001, epochs=60 | 0.248 | 0.132 | 0.133 | 0.186 | 0.150 |
