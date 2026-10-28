from collections.abc import Callable, Sequence

import numpy as np

K_VALUES = (5, 10)

def ranked_labels(labels: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """Labels in the order a method ranks them, best first.

    Ties keep the input order (a stable sort), so equal scores can't be broken in a way
    that quietly favours one method.
    """
    return labels[np.argsort(-scores, kind="stable")]

def ndcg_at_k(ranked: Sequence[int], k: int) -> float | None:
    """nDCG@k for binary labels, or None when nothing is relevant and it's undefined."""
    ranked = np.asarray(ranked, dtype=float)
    ideal = np.sort(ranked)[::-1][:k]
    if ideal.sum() == 0:
        return None
    discounts = 1 / np.log2(np.arange(2, k + 2))
    dcg = (ranked[:k] * discounts[: len(ranked[:k])]).sum()
    idcg = (ideal * discounts[: len(ideal)]).sum()
    return float(dcg / idcg)

def precision_at_k(ranked: Sequence[int], k: int) -> float:
    return float(np.sum(ranked[:k]) / k)

def mrr(ranked: Sequence[int]) -> float | None:
    """Reciprocal rank of the first relevant paper, or None when nothing is relevant."""
    hits = np.flatnonzero(np.asarray(ranked) == 1)
    return float(1 / (hits[0] + 1)) if len(hits) else None

def pool_metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, float | None]:
    ranked = ranked_labels(labels, scores)
    metrics = {"mrr": mrr(ranked)}
    for k in K_VALUES:
        metrics[f"ndcg@{k}"] = ndcg_at_k(ranked, k)
        metrics[f"p@{k}"] = precision_at_k(ranked, k)
    return metrics

def mean_metrics(per_pool: list[dict[str, float | None]]) -> dict[str, float]:
    """Average each metric over the pools where it is defined."""
    out = {}
    for name in per_pool[0]:
        values = [m[name] for m in per_pool if m[name] is not None]
        out[name] = float(np.mean(values)) if values else float("nan")
    return out

def expanding_folds(n_pools: int, first_validation: int) -> list[tuple[list[int], int]]:
    """(training pool indexes, validation pool index), each validating on a later pool."""
    return [(list(range(v)), v) for v in range(first_validation, n_pools)]

def cross_validate(
    n_pools: int,
    first_validation: int,
    fit_predict: Callable[[list[int], int], np.ndarray],
    labels: list[np.ndarray],
) -> dict[str, float]:
    """Mean metrics over expanding-window folds. fit_predict(train_idx, val_idx) returns
    scores for the validation pool's papers, in the same order as labels[val_idx]."""
    per_pool = [
        pool_metrics(labels[v], fit_predict(train, v))
        for train, v in expanding_folds(n_pools, first_validation)
    ]
    return mean_metrics(per_pool)
