import math
import random

import numpy as np
import pytest

from judgepack.pairwise import bradley_terry, sample_pairs
from judgepack.stats import (
    average_ranks,
    bootstrap_ci,
    fit_logistic,
    paired_bootstrap,
    predict_logistic,
    r_precision,
    rank_correlation_within,
    recall_at,
    residualise_within,
    wilson_interval,
    within_group_auc,
    within_group_percentile,
)


def test_average_ranks_share_ties():
    assert list(average_ranks([10, 20, 20, 30])) == [1, 2.5, 2.5, 4]


def test_auc_perfect_reversed_and_ties():
    g = np.array(["a"] * 4)
    y = np.array([1, 1, 0, 0])
    assert within_group_auc(np.array([4, 3, 2, 1]), y, g) == 1.0
    assert within_group_auc(np.array([1, 2, 3, 4]), y, g) == 0.0
    assert within_group_auc(np.array([1, 1, 1, 1]), y, g) == 0.5


def test_auc_only_compares_within_event():
    # Event b's projects all score higher than event a's, but within each event
    # the winner is ranked top. A pooled AUC would be confused; within-event is perfect.
    scores = np.array([1.0, 0.5, 0.1, 10.0, 9.0, 8.0])
    labels = np.array([1, 0, 0, 1, 0, 0])
    groups = np.array(["a", "a", "a", "b", "b", "b"])
    assert within_group_auc(scores, labels, groups) == 1.0


def test_auc_ignores_events_without_both_classes_and_nans():
    scores = np.array([3.0, 1.0, 5.0, np.nan])
    labels = np.array([1, 0, 1, 0])
    groups = np.array(["a", "a", "b", "a"])
    assert within_group_auc(scores, labels, groups) == 1.0
    assert math.isnan(within_group_auc(np.array([1.0]), np.array([1]), np.array(["a"])))


def test_recall_and_r_precision():
    scores = np.array([5, 4, 3, 2, 1, 0], dtype=float)
    labels = np.array([1, 0, 1, 0, 0, 0])
    groups = np.array(["a"] * 6)
    assert recall_at(scores, labels, groups, 0.0) == 0.0
    assert recall_at(scores, labels, groups, 0.5) == 1.0   # top 3 holds both winners
    assert recall_at(scores, labels, groups, 1 / 6) == 0.5
    assert r_precision(scores, labels, groups) == 0.5      # top 2 holds one winner


def test_percentiles_are_within_event():
    p = within_group_percentile(np.array([1.0, 2.0, 100.0, 200.0]), np.array(["a", "a", "b", "b"]))
    assert list(p) == [0.25, 0.75, 0.25, 0.75]


def test_rank_correlation_and_residualise():
    x = np.arange(10, dtype=float)
    g = np.array(["a"] * 10)
    assert rank_correlation_within(x, x * 3 + 1, g) == pytest.approx(1.0)
    resid, slope = residualise_within(2 * x + 5, x, g)
    assert slope == pytest.approx(2.0)
    assert np.allclose(resid, 0)


def test_bootstrap_interval_contains_point_and_is_reproducible():
    rng = np.random.default_rng(1)
    scores = rng.normal(size=60)
    labels = (scores + rng.normal(scale=1.0, size=60) > 0.8).astype(int)
    groups = np.array(["a", "b", "c"] * 20)
    a = bootstrap_ci(within_group_auc, scores, labels, groups, n_boot=300, seed=3)
    b = bootstrap_ci(within_group_auc, scores, labels, groups, n_boot=300, seed=3)
    assert a == b
    point, lo, hi = a
    assert lo <= point <= hi


def test_paired_bootstrap_detects_a_clearly_better_ranking():
    rng = np.random.default_rng(0)
    labels = np.array([1] * 20 + [0] * 60)
    groups = np.array(["a", "b"] * 40)
    good = labels + rng.normal(scale=0.3, size=80)
    noise = rng.normal(size=80)
    r = paired_bootstrap(within_group_auc, good, noise, labels, groups, n_boot=500, seed=0)
    assert r["diff"] > 0.3 and r["lo"] > 0 and r["p_a_better"] > 0.99


def test_wilson_interval():
    p, lo, hi = wilson_interval(50, 100)
    assert p == 0.5 and lo < 0.5 < hi
    assert all(math.isnan(v) for v in wilson_interval(0, 0))


def test_logistic_regression_learns_the_signal():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, 2))
    y = (X[:, 0] * 2 + rng.normal(scale=0.5, size=400) > 0).astype(float)
    w = fit_logistic(X, y, l2=0.1)
    assert w[1] > 1.0 and abs(w[2]) < 0.5
    assert predict_logistic(np.array([[3.0, 0.0]]), w)[0] > 0.9


def test_bradley_terry_recovers_order():
    ids = ["a", "b", "c", "d"]
    true = {"a": 4.0, "b": 2.0, "c": 1.0, "d": 0.5}
    rng = random.Random(0)
    games = []
    for _ in range(400):
        i, j = rng.sample(ids, 2)
        p = true[i] / (true[i] + true[j])
        games.append((i, j) if rng.random() < p else (j, i))
    s = bradley_terry(ids, games)
    assert sorted(ids, key=lambda k: -s[k]) == ["a", "b", "c", "d"]


def test_bradley_terry_stays_finite_for_unbeaten():
    s = bradley_terry(["a", "b"], [("a", "b")] * 5)
    assert all(math.isfinite(v) for v in s.values()) and s["a"] > s["b"]


def test_sample_pairs_covers_everyone_once_per_round():
    ids = [f"p{i}" for i in range(11)]
    pairs = sample_pairs(ids, 3, random.Random(1))
    seen = {x for pair in pairs for x in pair}
    assert seen == set(ids)
    assert all(a < b for a, b in pairs)
    assert len(sample_pairs(["a", "b", "c"], 4, random.Random(0))) == 3  # small events: all pairs
