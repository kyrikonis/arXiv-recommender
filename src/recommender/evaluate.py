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

def expected_random_metrics(
    labels: np.ndarray, rng: np.random.Generator, draws: int = 1000
) -> dict[str, float]:
    """A random ordering's expected metrics for one pool, averaged over many shuffles."""
    per_draw = [pool_metrics(labels, rng.random(len(labels))) for _ in range(draws)]
    return mean_metrics(per_draw)

def bootstrap_ci(
    values: np.ndarray, rng: np.random.Generator, n: int = 10_000, level: float = 0.95
) -> tuple[float, float]:
    """Percentile bootstrap interval for the mean, resampling days with replacement."""
    values = np.asarray(values, dtype=float)
    means = values[rng.integers(0, len(values), size=(n, len(values)))].mean(axis=1)
    tail = (1 - level) / 2 * 100
    return float(np.percentile(means, tail)), float(np.percentile(means, 100 - tail))

def sign_flip_p(differences: np.ndarray) -> float:
    """Exact two-sided paired permutation test on per-day differences.

    If two methods were equally good, each day's difference would be as likely positive as
    negative. Enumerating all 2^n sign patterns gives the exact share that produce a mean
    difference at least as extreme as the one observed.
    """
    d = np.asarray(differences, dtype=float)
    if len(d) > 20:
        raise ValueError("exact enumeration is only practical for up to 20 days")
    patterns = 1 - 2 * ((np.arange(2 ** len(d))[:, None] >> np.arange(len(d))) & 1)
    null_means = np.abs((patterns * d).mean(axis=1))
    # The small tolerance stops floating-point noise from excluding the observed pattern.
    return float(np.mean(null_means >= abs(d.mean()) - 1e-12))

def holm(p_values: list[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values, in the input order."""
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values))
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p_values) - rank) * p_values[i])
        adjusted[i] = min(running, 1.0)
    return adjusted.tolist()
