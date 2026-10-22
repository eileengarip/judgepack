"""Statistics used by the evaluation. Plain numpy, so every number can be traced by hand.

The central idea is the within-event comparison: winners are chosen relative to the other
projects at the same hackathon, so we only ever compare a winner with a non-winner from
the same event.
"""

from __future__ import annotations

import math
from typing import Callable

import numpy as np

Array = np.ndarray


def average_ranks(values: Array) -> Array:
    """Ranks starting at 1, ties sharing their average rank."""
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values))
    sorted_vals = values[order]
    i = 0
    while i < len(values):
        j = i
        while j + 1 < len(values) and sorted_vals[j + 1] == sorted_vals[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return ranks


def within_group_auc(scores: Array, labels: Array, groups: Array) -> float:
    """Probability that a random winner outscores a random non-winner from the same event.

    Ties count as half. 0.5 is chance; 1.0 means every winner beat every non-winner.
    """
    scores, labels, groups = np.asarray(scores, float), np.asarray(labels, int), np.asarray(groups)
    ok = np.isfinite(scores)
    scores, labels, groups = scores[ok], labels[ok], groups[ok]
    concordant, pairs = 0.0, 0
    for g in np.unique(groups):
        m = groups == g
        pos, neg = scores[m & (labels == 1)], scores[m & (labels == 0)]
        if len(pos) == 0 or len(neg) == 0:
            continue
        diff = pos[:, None] - neg[None, :]
        concordant += float((diff > 0).sum()) + 0.5 * float((diff == 0).sum())
        pairs += diff.size
    return concordant / pairs if pairs else math.nan


def within_group_concordance(a: Array, b: Array, groups: Array) -> float:
    """For pairs in the same group that b separates, the share that a puts in the same order.

    b is the reference (e.g. the judges' scores): pairs b scores equally are skipped, and pairs
    a scores equally count as half. 0.5 is chance; 1.0 means a ordered every pair exactly as b did.
    """
    a, b, groups = np.asarray(a, float), np.asarray(b, float), np.asarray(groups)
    agree, pairs = 0.0, 0
    for g in np.unique(groups):
        m = groups == g
        da = a[m][:, None] - a[m][None, :]
        db = b[m][:, None] - b[m][None, :]
        upper = np.triu(np.ones(da.shape, dtype=bool), k=1) & (db != 0)
        if not upper.any():
            continue
        prod = np.sign(da[upper]) * np.sign(db[upper])
        agree += float((prod > 0).sum()) + 0.5 * float((prod == 0).sum())
        pairs += int(upper.sum())
    return agree / pairs if pairs else math.nan


def within_group_pairs(b: Array, groups: Array) -> int:
    """How many within-group pairs b actually separates (the sample size behind a concordance)."""
    b, groups = np.asarray(b, float), np.asarray(groups)
    total = 0
    for g in np.unique(groups):
        v = b[groups == g]
        d = v[:, None] - v[None, :]
        total += int((np.triu(np.ones(d.shape, dtype=bool), k=1) & (d != 0)).sum())
    return total


def within_group_percentile(scores: Array, groups: Array) -> Array:
    """Each score as a percentile within its own event (higher score, higher percentile)."""
    scores, groups = np.asarray(scores, float), np.asarray(groups)
    out = np.full(len(scores), np.nan)
    for g in np.unique(groups):
        m = np.flatnonzero((groups == g) & np.isfinite(scores))
        if len(m):
            out[m] = (average_ranks(scores[m]) - 0.5) / len(m)
    return out


def zscore_within(scores: Array, groups: Array) -> Array:
    scores, groups = np.asarray(scores, float), np.asarray(groups)
    out = np.zeros(len(scores))
    for g in np.unique(groups):
        m = groups == g
        sd = scores[m].std()
        out[m] = (scores[m] - scores[m].mean()) / sd if sd > 0 else 0.0
    return out


def recall_at(scores: Array, labels: Array, groups: Array, fraction: float) -> float:
    """Share of winners inside the top `fraction` of their own event's ranking."""
    scores, labels, groups = np.asarray(scores, float), np.asarray(labels, int), np.asarray(groups)
    ok = np.isfinite(scores)
    scores, labels, groups = scores[ok], labels[ok], groups[ok]
    hit, total = 0, 0
    for g in np.unique(groups):
        m = np.flatnonzero(groups == g)
        n = len(m)
        rank_from_top = n + 1 - average_ranks(scores[m])  # 1 = best, ties share
        winners = labels[m] == 1
        hit += int((rank_from_top[winners] <= fraction * n + 1e-9).sum())
        total += int(winners.sum())
    return hit / total if total else math.nan


def recall_curve(scores: Array, labels: Array, groups: Array, steps: int = 20) -> list[tuple[float, float]]:
    return [(f, recall_at(scores, labels, groups, f)) for f in np.linspace(0, 1, steps + 1)]


def r_precision(scores: Array, labels: Array, groups: Array, seed: int = 0) -> float:
    """If each event's top-w projects were chosen (w = its number of winners), the share that won."""
    scores, labels, groups = np.asarray(scores, float), np.asarray(labels, int), np.asarray(groups)
    ok = np.isfinite(scores)
    scores, labels, groups = scores[ok], labels[ok], groups[ok]
    rng = np.random.default_rng(seed)
    hit, total = 0, 0
    for g in np.unique(groups):
        m = np.flatnonzero(groups == g)
        w = int(labels[m].sum())
        if w == 0:
            continue
        jitter = rng.random(len(m)) * 1e-9  # break exact ties at random, reproducibly
        top = m[np.argsort(-(scores[m] + jitter))[:w]]
        hit += int(labels[top].sum())
        total += w
    return hit / total if total else math.nan


def rank_correlation_within(x: Array, y: Array, groups: Array) -> float:
    """Spearman-style correlation of within-event percentiles."""
    px, py = within_group_percentile(x, groups), within_group_percentile(y, groups)
    ok = np.isfinite(px) & np.isfinite(py)
    if ok.sum() < 3 or np.std(px[ok]) == 0 or np.std(py[ok]) == 0:
        return math.nan
    return float(np.corrcoef(px[ok], py[ok])[0, 1])


def residualise_within(y: Array, x: Array, groups: Array) -> tuple[Array, float]:
    """Remove what x linearly explains about y, within events. Returns (residuals, slope)."""
    y, x, groups = np.asarray(y, float), np.asarray(x, float), np.asarray(groups)
    yd, xd = y.copy(), x.copy()
    for g in np.unique(groups):
        m = groups == g
        yd[m] -= y[m].mean()
        xd[m] -= x[m].mean()
    denom = float((xd * xd).sum())
    slope = float((xd * yd).sum() / denom) if denom > 0 else 0.0
    return yd - slope * xd, slope


def stratified_indices(groups: Array, rng: np.random.Generator) -> Array:
    """Bootstrap resample: draw projects with replacement inside each event, keeping event sizes."""
    groups = np.asarray(groups)
    parts = [rng.choice(np.flatnonzero(groups == g), size=int((groups == g).sum()), replace=True)
             for g in np.unique(groups)]
    return np.concatenate(parts)


Stat = Callable[[Array, Array, Array], float]


def bootstrap_ci(stat: Stat, scores: Array, labels: Array, groups: Array,
                 n_boot: int = 2000, seed: int = 0, alpha: float = 0.05) -> tuple[float, float, float]:
    """Point estimate and percentile bootstrap interval. labels may be 0/1 outcomes or reference scores."""
    scores, labels, groups = np.asarray(scores, float), np.asarray(labels, float), np.asarray(groups)
    point = stat(scores, labels, groups)
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n_boot):
        ix = stratified_indices(groups, rng)
        v = stat(scores[ix], labels[ix], groups[ix])
        if np.isfinite(v):
            values.append(v)
    if len(values) < n_boot * 0.5:
        return point, math.nan, math.nan
    lo, hi = np.quantile(values, [alpha / 2, 1 - alpha / 2])
    return point, float(lo), float(hi)


def paired_bootstrap(stat: Stat, scores_a: Array, scores_b: Array, labels: Array, groups: Array,
                     n_boot: int = 2000, seed: int = 0, alpha: float = 0.05) -> dict[str, float]:
    """Difference stat(a) - stat(b) on the same resamples, with an interval and P(a > b)."""
    a, b = np.asarray(scores_a, float), np.asarray(scores_b, float)
    labels, groups = np.asarray(labels, float), np.asarray(groups)
    point = stat(a, labels, groups) - stat(b, labels, groups)
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        ix = stratified_indices(groups, rng)
        d = stat(a[ix], labels[ix], groups[ix]) - stat(b[ix], labels[ix], groups[ix])
        if np.isfinite(d):
            diffs.append(d)
    if not diffs:
        return {"diff": point, "lo": math.nan, "hi": math.nan, "p_a_better": math.nan}
    diffs_arr = np.asarray(diffs)
    lo, hi = np.quantile(diffs_arr, [alpha / 2, 1 - alpha / 2])
    return {"diff": float(point), "lo": float(lo), "hi": float(hi), "p_a_better": float((diffs_arr > 0).mean())}


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    if n == 0:
        return math.nan, math.nan, math.nan
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return p, max(0.0, centre - half), min(1.0, centre + half)


def _sigmoid(z: Array) -> Array:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def fit_logistic(X: Array, y: Array, l2: float = 1.0, max_iter: int = 100) -> Array:
    """L2-regularised logistic regression by Newton's method. Returns [intercept, weights...]."""
    X, y = np.asarray(X, float), np.asarray(y, float)
    Xb = np.hstack([np.ones((len(X), 1)), X])
    w = np.zeros(Xb.shape[1])
    penalty = np.eye(Xb.shape[1]) * l2
    penalty[0, 0] = 0.0  # don't shrink the intercept
    for _ in range(max_iter):
        p = _sigmoid(Xb @ w)
        grad = Xb.T @ (p - y) + penalty @ w
        hess = Xb.T @ (Xb * (p * (1 - p))[:, None]) + penalty + np.eye(len(w)) * 1e-9
        step = np.linalg.solve(hess, grad)
        w -= step
        if np.max(np.abs(step)) < 1e-8:
            break
    return w


def predict_logistic(X: Array, w: Array) -> Array:
    X = np.asarray(X, float)
    return _sigmoid(np.hstack([np.ones((len(X), 1)), X]) @ w)
