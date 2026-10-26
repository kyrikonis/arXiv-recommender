import math

import numpy as np
import pytest

from recommender.retrieve import (
    BM25,
    combine,
    in_length_order,
    load_embeddings,
    save_embeddings,
    zscore_rows,
)

CORPUS = ["quantum error correction", "quantum computing hardware", "galaxy formation"]

def test_bm25_matches_hand_computation():
    bm25 = BM25(CORPUS, k1=1.5, b=0.75)
    scores = bm25.scores(["quantum error"], CORPUS)[0]

    # Worked by hand: 3 docs, "quantum" in 2, "error" in 1; lengths 3, 3, 2 (avg 8/3).
    idf_quantum = math.log((3 - 2 + 0.5) / (2 + 0.5) + 1)
    idf_error = math.log((3 - 1 + 0.5) / (1 + 0.5) + 1)
    # A 3-word doc with the term once: tf * (k1 + 1) / (tf + k1 * (1 - b + b * 3 / (8/3)))
    term = 1 * 2.5 / (1 + 1.5 * (0.25 + 0.75 * 3 / (8 / 3)))
    assert scores[0] == pytest.approx(term * (idf_quantum + idf_error))
    assert scores[1] == pytest.approx(term * idf_quantum)
    assert scores[2] == 0

def test_bm25_counts_repeated_query_terms_once_and_ignores_unknown_terms():
    bm25 = BM25(CORPUS)
    base = bm25.scores(["quantum"], CORPUS)
    assert np.allclose(bm25.scores(["quantum quantum neutrino"], CORPUS), base)

def test_bm25_prefers_shorter_doc_for_same_match():
    bm25 = BM25(CORPUS)
    docs = ["quantum", "quantum computing hardware architecture design"]
    short, long = bm25.scores(["quantum"], docs)[0]
    assert short > long

def test_zscore_rows():
    x = np.array([[1.0, 2.0, 3.0], [5.0, 5.0, 5.0]])
    z = zscore_rows(x)
    assert z[0].mean() == pytest.approx(0)
    assert z[0].std() == pytest.approx(1)
    assert np.all(z[1] == 0)

def test_max_keeps_a_strong_single_match_that_mean_dilutes():
    # Rows are two profile items from different areas, columns are docs A and B.
    # A matches the first item strongly and the second not at all; B is lukewarm about both.
    scores = np.array([[0.9, 0.55], [0.1, 0.55]])
    assert combine(scores, "max").argmax() == 0
    assert combine(scores, "mean").argmax() == 1

def test_combine_rejects_unknown_method():
    with pytest.raises(ValueError):
        combine(np.zeros((1, 1)), "median")

@pytest.mark.parametrize("batch_size", [1, 2, 64])
def test_in_length_order_restores_original_order(batch_size):
    texts = ["ccc", "a", "eeeee", "bb", "dddd"]

    def encode(batch):
        return np.array([[len(t)] for t in batch], dtype=np.float32)

    out = in_length_order(texts, encode, batch_size)
    assert out[:, 0].tolist() == [3, 1, 5, 2, 4]

def test_embeddings_round_trip(tmp_path):
    vectors = np.arange(6, dtype=np.float32).reshape(2, 3)
    save_embeddings(tmp_path / "emb", ["2501.00663", "quant-ph/9508027"], vectors)
    ids, loaded = load_embeddings(tmp_path / "emb")
    assert ids == ["2501.00663", "quant-ph/9508027"]
    assert np.array_equal(loaded, vectors)

def test_load_embeddings_detects_misaligned_files(tmp_path):
    save_embeddings(tmp_path / "emb", ["a", "b"], np.zeros((2, 3)))
    (tmp_path / "emb.ids").write_text("a")
    with pytest.raises(ValueError, match="1 ids but 2 vectors"):
        load_embeddings(tmp_path / "emb")
