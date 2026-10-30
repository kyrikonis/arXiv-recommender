"""Score every method once on the test pools and write results/results.md.

Needs the models saved by train.py (logreg, mlp, cross). Retrains only for the learning
curve, with the settings those models were chosen with.

    uv run scripts/evaluate.py
"""

import json
import logging
import platform
import time
from datetime import date

import joblib
import numpy as np
import torch
from sklearn.metrics import cohen_kappa_score

from recommender.config import ROOT, load_config, load_profile
from recommender.db import connect, papers_published, repeat_pairs
from recommender.evaluate import (
    bootstrap_ci,
    expected_random_metrics,
    holm,
    pool_metrics,
    sign_flip_p,
)
from recommender.features import ProfileScorer, first_test_open, load_pools
from recommender.pools import batches, iso
from recommender.rerank import (
    CROSS_ENCODER,
    fit_cross,
    fit_logreg,
    fit_mlp,
    load_cross,
    load_mlp,
    predict_cross,
    predict_cross_ensemble,
    predict_mlp,
    profile_query,
)
from recommender.retrieve import embed, load_specter, paper_text

log = logging.getLogger("evaluate")

MODELS = ROOT / "models"
METRICS = ["ndcg@10", "ndcg@5", "p@5", "p@10", "mrr"]
PRIMARY = "ndcg@10"
RANDOM = "Random order"
# Decided before looking at any test result, and corrected together with Holm's method.
COMPARISONS = [
    ("Logistic regression", RANDOM),
    ("MLP", RANDOM),
    ("Cross-encoder, pretrained", RANDOM),
    ("Cross-encoder, fine-tuned", RANDOM),
    ("Logistic regression", "SPECTER2 score"),
    ("Cross-encoder, fine-tuned", "Cross-encoder, pretrained"),
    ("Cross-encoder, fine-tuned", "Logistic regression"),
]
CURVE_SIZES = [3, 6, 9, 12]

def per_day_scores(methods, pools) -> tuple[dict[str, list[dict]], dict[str, float]]:
    """Metrics for every method on every pool, plus each method's median ms per pool."""
    results, latency = {}, {}
    for name, score in methods.items():
        score(pools[0]) # warm-up: first calls pay for lazy initialisation
        days, times = [], []
        for pool in pools:
            start = time.perf_counter()
            scores = score(pool)
            times.append((time.perf_counter() - start) * 1000)
            days.append(pool_metrics(pool.labels, scores))
        results[name], latency[name] = days, float(np.median(times))
        log.info("%-28s %s %.3f", name, PRIMARY, np.mean([d[PRIMARY] for d in days]))
    return results, latency

def column(days: list[dict], metric: str) -> np.ndarray:
    values = [d[metric] for d in days]
    # Every test pool has a relevant paper, so no metric should be undefined.
    if any(v is None for v in values):
        raise ValueError(f"{metric} undefined on some test day")
    return np.array(values)

def learning_curve(train, test, cfg, device) -> dict[str, dict[int, np.ndarray]]:
    """Per-day test nDCG@10 for each learned model trained on the most recent k pools."""
    logreg_params = joblib.load(MODELS / "logreg.joblib").named_steps["logisticregression"]
    mlp_config = torch.load(MODELS / "mlp.pt", weights_only=False)["config"]
    cross_config = json.loads((MODELS / "cross" / "config.json").read_text())
    query = profile_query([a.description for a in load_profile()])
    curve = {"Logistic regression": {}, "MLP": {}, "Cross-encoder, fine-tuned": {}}

    for k in CURVE_SIZES:
        pools = train[-k:]
        X = np.concatenate([p.features for p in pools])
        y = np.concatenate([p.labels for p in pools])
        texts = [t for p in pools for t in p.texts]

        logreg = fit_logreg(X, y, logreg_params.C, logreg_params.class_weight)
        mlps = [fit_mlp(X, y, seed=cfg.seed + s, **mlp_config) for s in range(5)]
        crosses = []
        for s in range(cross_config["seeds"]):
            snapshot = {}
            def keep(epoch, model, tokenizer, snapshot=snapshot):
                if epoch == cross_config["epochs"]:
                    snapshot["scores"] = [
                        predict_cross(model, tokenizer, query, p.texts, device) for p in test
                    ]
            fit_cross(
                query,
                texts,
                y,
                seed=cfg.seed + s,
                lr=cross_config["lr"],
                epochs=cross_config["max_epochs"],
                device=device,
                after_epoch=keep,
            )
            crosses.append(snapshot["scores"])

        cross_scores = [np.mean([c[i] for c in crosses], axis=0) for i in range(len(test))]
        for name, scores in (
            ("Logistic regression", [logreg.decision_function(p.features) for p in test]),
            ("MLP", [predict_mlp(mlps, p.features) for p in test]),
            ("Cross-encoder, fine-tuned", cross_scores),
        ):
            curve[name][k] = np.array(
                [pool_metrics(p.labels, s)[PRIMARY] for p, s in zip(test, scores)]
            )
        log.info("learning curve k=%d done", k)
    return curve

def cpu_costs(conn, cfg, test) -> dict[str, float]:
    """Median CPU ms for the steps that happen before reranking on a real new day."""
    tokenizer, model = load_specter("cpu")
    times = []
    for pool in test[:3]:
        texts = [paper_text(p.title, p.abstract, tokenizer.sep_token) for p in pool.papers]
        start = time.perf_counter()
        embed(texts, tokenizer, model, "proximity", batch_size=8)
        times.append((time.perf_counter() - start) * 1000)
    costs = {"SPECTER2 embedding of 40 new papers": float(np.median(times))}

    scorer = ProfileScorer(conn, cfg, first_test_open(conn, cfg))
    by_name = {b.name: b for b in batches(cfg.history_start, cfg.history_end, cfg.holidays)}
    times, sizes = [], []
    for pool in test[:5]:
        batch = by_name[pool.batch]
        candidates = papers_published(conn, iso(batch.opens), iso(batch.closes))
        start = time.perf_counter()
        scorer.scores(candidates)
        times.append((time.perf_counter() - start) * 1000)
        sizes.append(len(candidates))
    step = f"BM25 + dense profile scores for a whole batch (median {int(np.median(sizes))} papers)"
    costs[step] = float(np.median(times))
    return costs

def fmt_ci(values: np.ndarray, rng) -> str:
    low, high = bootstrap_ci(values, rng)
    return f"{values.mean():.3f} ({low:.3f}-{high:.3f})"

def write_report(results, latency, costs, curve, train, test, conn, cfg, rng) -> str:
    n_test = sum(len(p.labels) for p in test)
    rel_test = int(sum(p.labels.sum() for p in test))
    n_train = sum(len(p.labels) for p in train)
    rel_train = int(sum(p.labels.sum() for p in train))
    pairs = repeat_pairs(conn)
    agree = np.mean([a == b for a, b in pairs])
    kappa = cohen_kappa_score(*zip(*pairs))
    lines = [
        "# Results",
        "",
        f"Generated {date.today()} by `uv run scripts/evaluate.py`.",
        "",
        f"- Test set: {len(test)} announcement batches, {test[0].batch} to {test[-1].batch}; "
        f"{n_test} labelled papers, {rel_test} relevant.",
        f"- Training data: {len(train)} batches from {train[0].batch} to {train[-1].batch}; "
        f"{n_train} labelled papers, {rel_train} relevant. Every training paper was submitted "
        "before every test paper.",
        f"- Labels: one labeller. Agreement with my own earlier answer on {len(pairs)} hidden "
        f"repeats: {agree:.0%}, Cohen's kappa {kappa:.2f}.",
        "- Each method reranks the same fixed 40-paper pool per test day. Metrics are averaged "
        "over days; brackets are 95% bootstrap intervals over the "
        f"{len(test)} days (10,000 resamples).",
        "",
        "## Ranking quality",
        "",
        "| Method | " + " | ".join(METRICS) + " |",
        "|---|" + "---|" * len(METRICS),
    ]
    for name, days in results.items():
        cells = [fmt_ci(column(days, m), rng) for m in METRICS]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")

    diffs = [column(results[a], PRIMARY) - column(results[b], PRIMARY) for a, b in COMPARISONS]
    raw = [sign_flip_p(d) for d in diffs]
    lines += [
        "",
        f"## Paired comparisons ({PRIMARY})",
        "",
        "Per-day differences, A minus B. p is an exact two-sided sign-flip permutation test "
        f"over all 2^{len(test)} sign patterns; Holm adjusts for the {len(COMPARISONS)} "
        "comparisons, which were fixed before the test set was scored.",
        "",
        "| A | B | mean difference (95% CI) | days A better / worse | p | p (Holm) |",
        "|---|---|---|---|---|---|",
    ]
    for (a, b), d, p, p_holm in zip(COMPARISONS, diffs, raw, holm(raw)):
        better, worse = int((d > 0).sum()), int((d < 0).sum())
        lines.append(
            f"| {a} | {b} | {fmt_ci(d, rng)} | {better} / {worse} | {p:.4f} | {p_holm:.4f} |"
        )

    lines += [
        "",
        "## Latency (CPU)",
        "",
        f"{platform.processor() or platform.machine()}, {torch.get_num_threads()} PyTorch "
        "threads. Median over the test pools after one warm-up call. Reranker times assume "
        "the paper embeddings and retrieval scores already exist; those costs are listed "
        "separately below.",
        "",
        "| Step | ms |",
        "|---|---|",
    ]
    for name, ms in latency.items():
        if name not in (RANDOM, "BM25 score", "SPECTER2 score"):
            lines.append(f"| {name}: rerank one 40-paper pool | {ms:.1f} |")
    for name, ms in costs.items():
        lines.append(f"| {name} | {ms:.1f} |")

    lines += [
        "",
        f"## Learning curve ({PRIMARY} on the test set)",
        "",
        "Each learned model retrained on the most recent k training pools with its chosen "
        "settings. Descriptive only: nothing was chosen from these numbers. The cross-encoder "
        "is retrained on the GPU here, where run-to-run nondeterminism can move its scores "
        "slightly from the saved model's.",
        "",
        "| Training pools | papers (relevant) | " + " | ".join(curve) + " |",
        "|---|---|" + "---|" * len(curve),
    ]
    for k in CURVE_SIZES:
        pools = train[-k:]
        size = f"{sum(len(p.labels) for p in pools)} ({int(sum(p.labels.sum() for p in pools))})"
        cells = [fmt_ci(curve[name][k], rng) for name in curve]
        lines.append(f"| {k} | {size} | " + " | ".join(cells) + " |")

    lines += [
        "",
        f"## {PRIMARY} per test day",
        "",
        "| Day | papers (relevant) | " + " | ".join(results) + " |",
        "|---|---|" + "---|" * len(results),
    ]
    for i, pool in enumerate(test):
        cells = [f"{days[i][PRIMARY]:.3f}" for days in results.values()]
        lines.append(
            f"| {pool.batch} | {len(pool.labels)} ({int(pool.labels.sum())}) | "
            + " | ".join(cells) + " |"
        )
    return "\n".join(lines) + "\n"

def main() -> None:
    cfg = load_config()
    conn = connect(cfg.db_path)
    train = load_pools(conn, cfg, "train")
    test = load_pools(conn, cfg, "test")
    query = profile_query([a.description for a in load_profile()])

    logreg = joblib.load(MODELS / "logreg.joblib")
    mlp = load_mlp(MODELS / "mlp.pt")
    pretrained = load_cross(CROSS_ENCODER, "cpu")
    cross_config = json.loads((MODELS / "cross" / "config.json").read_text())
    fine_tuned = [
        load_cross(str(MODELS / "cross" / f"seed{s}"), "cpu") for s in range(cross_config["seeds"])
    ]

    random_rng = np.random.default_rng(cfg.seed)
    results = {RANDOM: [expected_random_metrics(p.labels, random_rng) for p in test]}
    methods = {
        "BM25 score": lambda p: p.bm25,
        "SPECTER2 score": lambda p: p.dense,
        "Logistic regression": lambda p: logreg.decision_function(p.features),
        "MLP": lambda p: predict_mlp(mlp, p.features),
        "Cross-encoder, pretrained": lambda p: predict_cross(*pretrained, query, p.texts, "cpu"),
        "Cross-encoder, fine-tuned": lambda p: predict_cross_ensemble(
            fine_tuned, query, p.texts, "cpu"
        ),
    }
    scored, latency = per_day_scores(methods, test)
    results.update(scored)

    costs = cpu_costs(conn, cfg, test)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    curve = learning_curve(train, test, cfg, device)

    report = write_report(
        results, latency, costs, curve, train, test, conn, cfg, np.random.default_rng(cfg.seed)
    )
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "results.md").write_text(report)
    log.info("wrote results/results.md")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    for noisy in ("httpx", "transformers", "adapters"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    main()
