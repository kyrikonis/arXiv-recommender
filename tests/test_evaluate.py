import math

import numpy as np
import pytest

from recommender.evaluate import (
    bootstrap_ci,
    cross_validate,
    expanding_folds,
    expected_random_metrics,
    holm,
    mean_metrics,
    mrr,
    ndcg_at_k,
    pool_metrics,
    precision_at_k,
    ranked_labels,
    sign_flip_p,
)

def test_ndcg_matches_hand_computation():
    # Relevant papers at ranks 2 and 4. DCG = 1/log2(3) + 1/log2(5);
    # the ideal puts them at ranks 1 and 2: IDCG = 1/log2(2) + 1/log2(3).
    expected = (1 / math.log2(3) + 1 / math.log2(5)) / (1 + 1 / math.log2(3))
    assert ndcg_at_k([0, 1, 0, 1], 10) == pytest.approx(expected)

def test_ndcg_is_one_for_a_perfect_ranking_and_cut_at_k():
    assert ndcg_at_k([1, 1, 0, 0], 10) == pytest.approx(1.0)
    # A relevant paper beyond k counts for nothing in DCG, but the ideal ranking still
    # has it inside the top k, so the score drops below 1.
    assert ndcg_at_k([1, 0, 0, 1], 2) == pytest.approx(1 / (1 + 1 / math.log2(3)))

def test_metrics_undefined_without_relevant_papers():
    assert ndcg_at_k([0, 0, 0], 10) is None
    assert mrr([0, 0, 0]) is None
    assert precision_at_k([0, 0, 0], 5) == 0

def test_precision_divides_by_k_even_for_short_lists():
    assert precision_at_k([1, 0, 1], 5) == pytest.approx(2 / 5)

def test_mrr_uses_the_first_relevant_paper():
    assert mrr([0, 0, 1, 1]) == pytest.approx(1 / 3)

def test_ranking_sorts_by_score_and_keeps_input_order_on_ties():
    labels = np.array([1, 0, 0, 1])
    assert ranked_labels(labels, np.array([0.1, 0.9, 0.5, 0.5])).tolist() == [0, 0, 1, 1]
    assert ranked_labels(labels, np.zeros(4)).tolist() == [1, 0, 0, 1]

def test_mean_skips_undefined_pools():
    pools = [
        pool_metrics(np.array([1, 0]), np.array([1.0, 0.0])),
        pool_metrics(np.array([0, 0]), np.array([1.0, 0.0])),
    ]
    means = mean_metrics(pools)
    assert means["ndcg@10"] == pytest.approx(1.0) # only the first pool is defined
    assert means["p@5"] == pytest.approx((1 / 5 + 0) / 2) # defined for both

def test_folds_only_validate_on_later_pools():
    folds = expanding_folds(12, 6)
    assert [v for _, v in folds] == [6, 7, 8, 9, 10, 11]
    assert all(max(train) < v and train == list(range(v)) for train, v in folds)

def test_cross_validate_passes_folds_and_scores_validation_pool():
    labels = [np.array([0, 1]), np.array([1, 0]), np.array([0, 1])]
    calls = []

    def fit_predict(train, v):
        calls.append((train, v))
        return labels[v].astype(float) # a perfect ranker
    means = cross_validate(3, 1, fit_predict, labels)
    assert calls == [([0], 1), ([0, 1], 2)]
    assert means["ndcg@10"] == pytest.approx(1.0)
    assert means["mrr"] == pytest.approx(1.0)

def test_bootstrap_ci_of_constant_is_that_constant_and_brackets_the_mean():
    rng = np.random.default_rng(0)
    assert bootstrap_ci(np.full(15, 0.4), rng) == pytest.approx((0.4, 0.4))
    values = rng.random(15)
    low, high = bootstrap_ci(values, rng)
    assert low < values.mean() < high

def test_sign_flip_p_is_exact():
    # All 5 differences positive: only the all-plus and all-minus patterns are as extreme.
    assert sign_flip_p(np.array([0.1, 0.2, 0.3, 0.4, 0.5])) == pytest.approx(2 / 32)
    assert sign_flip_p(np.zeros(5)) == pytest.approx(1.0)
    assert sign_flip_p(np.array([0.3, -0.3])) == pytest.approx(1.0)

def test_holm_adjusts_in_order_and_stays_monotone():
    # Sorted 0.01, 0.02, 0.04 get multiplied by 3, 2, 1 -> 0.03, 0.04, 0.04.
    assert holm([0.04, 0.01, 0.02]) == pytest.approx([0.04, 0.03, 0.04])
    assert holm([0.5, 0.9]) == pytest.approx([1.0, 1.0])

def test_expected_random_metrics_match_the_closed_form():
    # One relevant paper among two: it is first half the time.
    means = expected_random_metrics(np.array([1, 0]), np.random.default_rng(0), draws=4000)
    assert means["mrr"] == pytest.approx(0.75, abs=0.02)
    assert means["ndcg@10"] == pytest.approx((1 + 1 / math.log2(3)) / 2, abs=0.02)
