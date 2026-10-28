# Cross-validation: logreg

Generated 2026-10-28 by `scripts/train.py --model logreg`. Expanding-window folds over the 12 training pools: each of pools 7-12 is scored by a model trained on every pool before it. Training pools only; no test labels were used.

| config | mrr | ndcg@5 | p@5 | ndcg@10 | p@10 |
|---|---|---|---|---|---|
| C=0.0001, class_weight=balanced | 0.514 | 0.256 | 0.167 | 0.332 | 0.183 |
| C=3e-05, class_weight=None | 0.514 | 0.256 | 0.167 | 0.319 | 0.167 |
| C=3e-05, class_weight=balanced | 0.514 | 0.256 | 0.167 | 0.319 | 0.167 |
| C=0.0001, class_weight=None | 0.514 | 0.256 | 0.167 | 0.319 | 0.167 |
| C=1e-05, class_weight=None | 0.514 | 0.256 | 0.167 | 0.319 | 0.167 |
| C=1e-05, class_weight=balanced | 0.514 | 0.256 | 0.167 | 0.319 | 0.167 |
| C=0.0003, class_weight=None | 0.512 | 0.256 | 0.167 | 0.318 | 0.167 |
| C=0.1, class_weight=None | 0.407 | 0.200 | 0.167 | 0.284 | 0.200 |
| C=0.03, class_weight=None | 0.306 | 0.227 | 0.233 | 0.279 | 0.200 |
| C=0.1, class_weight=balanced | 0.380 | 0.192 | 0.167 | 0.276 | 0.200 |
| C=0.03, class_weight=balanced | 0.301 | 0.197 | 0.200 | 0.273 | 0.200 |
| C=0.001, class_weight=balanced | 0.308 | 0.182 | 0.167 | 0.265 | 0.183 |
| C=0.003, class_weight=None | 0.308 | 0.182 | 0.167 | 0.264 | 0.183 |
| C=0.01, class_weight=balanced | 0.292 | 0.192 | 0.200 | 0.260 | 0.183 |
| C=0.0003, class_weight=balanced | 0.267 | 0.174 | 0.167 | 0.253 | 0.183 |
| C=0.003, class_weight=balanced | 0.333 | 0.187 | 0.167 | 0.252 | 0.167 |
| C=0.01, class_weight=None | 0.325 | 0.178 | 0.167 | 0.251 | 0.167 |
| C=0.3, class_weight=None | 0.371 | 0.195 | 0.167 | 0.242 | 0.183 |
| C=1.0, class_weight=None | 0.364 | 0.171 | 0.133 | 0.241 | 0.183 |
| C=1.0, class_weight=balanced | 0.364 | 0.171 | 0.133 | 0.240 | 0.183 |
| C=0.001, class_weight=None | 0.284 | 0.176 | 0.167 | 0.240 | 0.167 |
| C=0.3, class_weight=balanced | 0.371 | 0.192 | 0.167 | 0.239 | 0.183 |
