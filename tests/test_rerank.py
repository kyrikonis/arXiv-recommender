import numpy as np

from recommender.db import Paper, add_label, connect, insert_papers, labelled_rows, save_pools
from recommender.evaluate import pool_metrics
from recommender.features import Pool
from recommender.rerank import fit_logreg, fit_mlp, predict_mlp

def toy_data(seed: int = 0, n: int = 200):
    # Relevance depends on the first feature, so any working model should learn it.
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 10))
    y = (X[:, 0] + 0.3 * rng.normal(size=n) > 1).astype(float)
    return X, y

def test_labelled_rows_never_return_the_other_split_or_skips(tmp_path):
    conn = connect(tmp_path / "papers.db")
    papers = [Paper(f"2610.0000{i}", "T", "A", "X", "gr-qc", "gr-qc", "2026-10-01") for i in range(3)]
    insert_papers(conn, papers)
    pools = [
        ("2026-09-30", "train", {papers[0].id: "bm25", papers[1].id: "random"}),
        ("2026-10-01", "test", {papers[2].id: "dense"}),
    ]
    queue = [(p.id, False) for p in papers]
    save_pools(conn, pools, queue)
    add_label(conn, 0, 1)
    add_label(conn, 1, None) # skipped
    add_label(conn, 2, 0)
    assert labelled_rows(conn, "train") == [("2026-09-30", papers[0].id, 1)]
    assert labelled_rows(conn, "test") == [("2026-10-01", papers[2].id, 0)]

def test_pool_features_are_embedding_then_retrieval_scores():
    pool = Pool(
        batch="2026-10-01",
        papers=[],
        labels=np.array([1, 0]),
        bm25=np.array([2.0, -1.0]),
        dense=np.array([0.9, 0.8]),
        embeddings=np.zeros((2, 768)),
    )
    X = pool.features
    assert X.shape == (2, 770)
    assert X[:, -2].tolist() == [2.0, -1.0]
    assert X[:, -1].tolist() == [0.9, 0.8]

def test_logreg_learns_the_signal():
    X, y = toy_data()
    Xv, yv = toy_data(seed=1)
    model = fit_logreg(X, y, C=1.0, class_weight="balanced")
    assert pool_metrics(yv, model.decision_function(Xv))["ndcg@10"] > 0.9

def test_mlp_learns_the_signal_and_is_reproducible():
    X, y = toy_data()
    Xv, yv = toy_data(seed=1)
    config = dict(hidden=32, dropout=0.2, weight_decay=1e-4, epochs=30)
    a = predict_mlp([fit_mlp(X, y, seed=3, **config)], Xv)
    b = predict_mlp([fit_mlp(X, y, seed=3, **config)], Xv)
    c = predict_mlp([fit_mlp(X, y, seed=4, **config)], Xv)
    assert pool_metrics(yv, a)["ndcg@10"] > 0.9
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)
