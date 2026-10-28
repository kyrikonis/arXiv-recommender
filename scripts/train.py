"""Tune a reranker by expanding-window cross-validation on the training pools, then refit
it on all of them and save it. Never reads test labels.

    uv run scripts/train.py --model baselines
    uv run scripts/train.py --model logreg
    uv run scripts/train.py --model mlp
    uv run scripts/train.py --model cross
"""

import argparse
import itertools
import logging
import shutil
from datetime import date

import joblib
import numpy as np
import torch

from recommender.config import ROOT, load_config, load_profile
from recommender.db import connect
from recommender.evaluate import cross_validate
from recommender.features import load_pools
from recommender.rerank import (
    CROSS_ENCODER,
    fit_cross,
    fit_logreg,
    fit_mlp,
    load_cross,
    predict_cross,
    predict_mlp,
    profile_query,
)

log = logging.getLogger("train")

FIRST_VALIDATION = 6 # validate on pools 7..12, each trained on all pools before it
MLP_SEEDS = 5
CROSS_SEEDS = 3
CROSS_MAX_EPOCHS = 6
MODELS = ROOT / "models"
RESULTS = ROOT / "results"

def stack(pools, idx, attr):
    return np.concatenate([getattr(pools[i], attr) for i in idx])

def write_cv(model: str, rows: list[tuple[dict, dict]], n_pools: int) -> None:
    """One table row per configuration, best mean nDCG@10 first."""
    rows = sorted(rows, key=lambda r: -np.nan_to_num(r[1]["ndcg@10"], nan=-1))
    metrics = list(rows[0][1])
    lines = [
        f"# Cross-validation: {model}",
        "",
        f"Generated {date.today()} by `scripts/train.py --model {model}`. Expanding-window "
        f"folds over the {n_pools} training pools: each of pools {FIRST_VALIDATION + 1}-"
        f"{n_pools} is scored by a model trained on every pool before it. Training pools "
        "only; no test labels were used.",
        "",
        "| config | " + " | ".join(metrics) + " |",
        "|---|" + "---|" * len(metrics),
    ]
    for config, means in rows:
        name = ", ".join(f"{k}={v}" for k, v in config.items()) or "-"
        lines.append(f"| {name} | " + " | ".join(f"{means[m]:.3f}" for m in metrics) + " |")
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"cv_{model}.md").write_text("\n".join(lines) + "\n")

def run_baselines(pools, labels, seed):
    n = len(pools)
    rows = []
    for source in ("bm25", "dense"):
        means = cross_validate(n, FIRST_VALIDATION, lambda tr, v: getattr(pools[v], source), labels)
        rows.append(({"method": source}, means))
    # A random ranking's expected score, averaged over many shuffles so it isn't one lucky draw.
    rng = np.random.default_rng(seed)
    draws = [
        cross_validate(n, FIRST_VALIDATION, lambda tr, v: rng.random(len(labels[v])), labels)
        for _ in range(200)
    ]
    means = {m: float(np.mean([d[m] for d in draws])) for m in draws[0]}
    rows.append(({"method": "random (mean of 200)"}, means))
    return rows

def run_logreg(pools, labels):
    # Extended down to 1e-5 after the first run picked 0.0003, the edge of the original grid.
    Cs = [0.00001, 0.00003, 0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0]
    grid = [
        {"C": C, "class_weight": cw} for C, cw in itertools.product(Cs, [None, "balanced"])
    ]
    rows = []
    for config in grid:
        def fit_predict(tr, v):
            model = fit_logreg(stack(pools, tr, "features"), stack(pools, tr, "labels"), **config)
            return model.decision_function(pools[v].features)
        rows.append((config, cross_validate(len(pools), FIRST_VALIDATION, fit_predict, labels)))
    return rows

def refit_logreg(pools, config):
    idx = range(len(pools))
    model = fit_logreg(stack(pools, idx, "features"), stack(pools, idx, "labels"), **config)
    joblib.dump(model, MODELS / "logreg.joblib")

def mlp_ensemble(X, y, config, seed):
    return [fit_mlp(X, y, seed=seed + s, **config) for s in range(MLP_SEEDS)]

def run_mlp(pools, labels, seed):
    grid = [
        {"hidden": h, "dropout": d, "weight_decay": wd, "epochs": e}
        for h, d, wd, e in itertools.product([64, 256], [0.2, 0.5], [1e-4, 1e-2], [20, 60])
    ]
    rows = []
    for config in grid:
        def fit_predict(tr, v):
            X, y = stack(pools, tr, "features"), stack(pools, tr, "labels")
            return predict_mlp(mlp_ensemble(X, y, config, seed), pools[v].features)
        rows.append((config, cross_validate(len(pools), FIRST_VALIDATION, fit_predict, labels)))
    return rows

def refit_mlp(pools, config, seed):
    idx = range(len(pools))
    models = mlp_ensemble(stack(pools, idx, "features"), stack(pools, idx, "labels"), config, seed)
    torch.save(
        {
            "config": config,
            "states": [m.state_dict() for m, _, _ in models],
            "means": [mean for _, mean, _ in models],
            "stds": [std for _, _, std in models],
        },
        MODELS / "mlp.pt",
    )

def run_cross(pools, labels, seed, device):
    query = profile_query([a.description for a in load_profile()])
    texts = [p.texts for p in pools]
    # The pretrained model before any fine-tuning, so the table shows whether training on
    # my labels helped at all.
    model, tokenizer = load_cross(CROSS_ENCODER, device)
    means = cross_validate(
        len(pools),
        FIRST_VALIDATION,
        lambda tr, v: predict_cross(model, tokenizer, query, texts[v], device),
        labels,
    )
    rows = [({"lr": 0, "epochs": 0}, means)]
    # The grid was widened twice because the best setting sat on its edge: 1e-4 after
    # 3e-5 won, then 3e-4 and up to 6 epochs after 1e-4 with 4 epochs won.
    for lr in (1e-5, 3e-5, 1e-4, 3e-4):
        # scores[epoch][fold] collects every seed's validation scores, averaged afterwards
        # into the ensemble's prediction, like the MLP's.
        scores = {e: {} for e in range(1, CROSS_MAX_EPOCHS + 1)}
        for s in range(CROSS_SEEDS):
            for v in range(FIRST_VALIDATION, len(pools)):
                tr = range(v)
                def record(epoch, model, tokenizer, v=v):
                    pred = predict_cross(model, tokenizer, query, texts[v], device)
                    scores[epoch].setdefault(v, []).append(pred)
                fit_cross(
                    query,
                    [t for i in tr for t in texts[i]],
                    stack(pools, tr, "labels"),
                    seed=seed + s,
                    lr=lr,
                    epochs=CROSS_MAX_EPOCHS,
                    device=device,
                    after_epoch=record,
                )
                log.info("cross lr=%g seed=%d fold=%d done", lr, s, v)
        for epoch, by_fold in scores.items():
            means = cross_validate(
                len(pools), FIRST_VALIDATION, lambda tr, v: np.mean(by_fold[v], axis=0), labels
            )
            rows.append(({"lr": lr, "epochs": epoch}, means))
    return rows

def refit_cross(pools, config, seed, device):
    query = profile_query([a.description for a in load_profile()])
    texts = [t for p in pools for t in p.texts]
    y = np.concatenate([p.labels for p in pools])
    # Clear old seeds so a smaller ensemble can't be mixed with a previous run's models.
    shutil.rmtree(MODELS / "cross", ignore_errors=True)
    if config["epochs"] == 0:
        model, tokenizer = load_cross(CROSS_ENCODER, device)
        model.save_pretrained(MODELS / "cross" / "seed0")
        tokenizer.save_pretrained(MODELS / "cross" / "seed0")
        return
    for s in range(CROSS_SEEDS):
        out = MODELS / "cross" / f"seed{s}"
        # Same schedule as in cross-validation (planned over the max epochs), saved at the
        # chosen epoch, so the final model is trained exactly as the one that was scored.
        def save_at(epoch, model, tokenizer):
            if epoch == config["epochs"]:
                model.save_pretrained(out)
                tokenizer.save_pretrained(out)
        fit_cross(
            query,
            texts,
            y,
            seed=seed + s,
            lr=config["lr"],
            epochs=CROSS_MAX_EPOCHS,
            device=device,
            after_epoch=save_at,
        )

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["baselines", "logreg", "mlp", "cross"], required=True)
    args = parser.parse_args()

    cfg = load_config()
    conn = connect(cfg.db_path)
    pools = load_pools(conn, cfg, "train")
    labels = [p.labels for p in pools]
    log.info(
        "%d training pools, %d papers, %d relevant",
        len(pools), sum(len(x) for x in labels), sum(x.sum() for x in labels),
    )
    MODELS.mkdir(exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.model == "baselines":
        rows = run_baselines(pools, labels, cfg.seed)
    elif args.model == "logreg":
        rows = run_logreg(pools, labels)
    elif args.model == "mlp":
        rows = run_mlp(pools, labels, cfg.seed)
    else:
        rows = run_cross(pools, labels, cfg.seed, device)
    write_cv(args.model, rows, len(pools))

    best_config, best = max(rows, key=lambda r: np.nan_to_num(r[1]["ndcg@10"], nan=-1))
    log.info("best %s: %s", best_config, {k: round(v, 3) for k, v in best.items()})
    if args.model == "logreg":
        refit_logreg(pools, best_config)
    elif args.model == "mlp":
        refit_mlp(pools, best_config, cfg.seed)
    elif args.model == "cross":
        refit_cross(pools, best_config, cfg.seed, device)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    for noisy in ("httpx", "transformers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    main()
