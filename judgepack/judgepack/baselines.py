"""Baselines the AI has to beat.

If an AI judge can't beat "longer write-ups win" or a tiny model of obvious signals
(is there a demo video? a code repo?), it isn't adding judgement, just restating them.
"""

from __future__ import annotations

import math

import numpy as np

from .ingest import Submission
from .stats import fit_logistic, predict_logistic

FEATURES = [
    ("log_words", "Write-up length (log of word count)"),
    ("has_video", "Links a demo video"),
    ("has_repo", "Links a code repository"),
    ("n_tech", "Number of technologies listed"),
]


def feature_matrix(subs: list[Submission]) -> np.ndarray:
    return np.array(
        [[math.log1p(s.word_count), float(s.has_video), float(s.has_repo), float(len(s.built_with))] for s in subs],
        dtype=float,
    ).reshape(len(subs), len(FEATURES))


def _standardise(train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu, sd = train.mean(axis=0), train.std(axis=0)
    sd[sd == 0] = 1.0
    return (train - mu) / sd, (test - mu) / sd


def out_of_fold_logistic(X: np.ndarray, y: np.ndarray, groups: np.ndarray, l2: float = 1.0) -> tuple[np.ndarray, str]:
    """Predictions where no project is scored by a model that saw its own label.

    With two or more events: train on the other events, predict the held-out one
    (leave-one-event-out), which mimics using last year's results on this year's entries.
    With a single event: leave-one-project-out.
    """
    y = np.asarray(y, float)
    groups = np.asarray(groups)
    preds = np.full(len(y), np.nan)
    events = np.unique(groups)
    if len(events) >= 2:
        folds = [groups == e for e in events]
        scheme = "leave-one-event-out"
    else:
        folds = [np.arange(len(y)) == i for i in range(len(y))]
        scheme = "leave-one-project-out"
    for test in folds:
        train = ~test
        if train.sum() < 3 or y[train].min() == y[train].max():
            continue  # a fold needs both winners and non-winners to learn from
        Xtr, Xte = _standardise(X[train], X[test])
        w = fit_logistic(Xtr, y[train], l2=l2)
        preds[test] = predict_logistic(Xte, w)
    return preds, scheme


def signal_weights(X: np.ndarray, y: np.ndarray, l2: float = 1.0) -> list[tuple[str, float]]:
    """Standardised coefficients from one model fit on everything: which simple signals go with winning."""
    if len(np.unique(y)) < 2:
        return []
    Xs, _ = _standardise(X, X)
    w = fit_logistic(Xs, y, l2=l2)
    return [(label, float(coef)) for (_, label), coef in zip(FEATURES, w[1:])]
