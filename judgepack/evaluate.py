"""Compare the model's judgements with what actually happened, against honest baselines.

Everything here is a pure function of the model outputs and the labels, so the numbers
in the report can be recomputed from results.json alone.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np

from .baselines import feature_matrix, out_of_fold_logistic, signal_weights
from .config import Config, EvaluationConfig
from .ingest import Submission
from .llm import Usage
from .pairwise import PairwiseResult
from .schemas import Brief
from .score import ProjectScore
from .stats import (
    bootstrap_ci,
    paired_bootstrap,
    r_precision,
    rank_correlation_within,
    recall_at,
    recall_curve,
    residualise_within,
    within_group_auc,
    within_group_percentile,
    wilson_interval,
    zscore_within,
)
from .util import norm


class EvaluationError(ValueError):
    """There isn't enough labelled data to evaluate."""


@dataclass
class ModelOutputs:
    briefs: dict[str, Brief] = field(default_factory=dict)
    scores: dict[str, ProjectScore] = field(default_factory=dict)
    pairwise: PairwiseResult | None = None
    errors: list[str] = field(default_factory=list)


def counted_prizes(sub: Submission, ev: EvaluationConfig) -> list[str]:
    prizes = [p for p in sub.prizes if not any(norm(x) and norm(x) in norm(p) for x in ev.exclude_prizes)]
    if ev.win_prizes == "any":
        return prizes
    return [p for p in prizes if any(norm(w) and norm(w) in norm(p) for w in ev.win_prizes)]


def is_win(sub: Submission, ev: EvaluationConfig) -> bool:
    return bool(counted_prizes(sub, ev))


def describe_win_label(ev: EvaluationConfig) -> str:
    base = "any prize" if ev.win_prizes == "any" else "a prize matching " + ", ".join(f"'{w}'" for w in ev.win_prizes)
    if ev.exclude_prizes:
        base += ", not counting " + ", ".join(f"'{x}'" for x in ev.exclude_prizes) + " prizes"
    return base


def _ci(t: tuple[float, float, float]) -> list[float | None]:
    return [None if (v is None or (isinstance(v, float) and math.isnan(v))) else round(float(v), 4) for v in t]


def _num(v: float) -> float | None:
    return None if v is None or (isinstance(v, float) and math.isnan(v)) else round(float(v), 4)


def _pct(v: float | None) -> str:
    return "n/a" if v is None else f"{v * 100:.0f}%"


def compute_results(
    subs: list[Submission],
    cfg: Config,
    outputs: ModelOutputs,
    *,
    usage: Usage | None = None,
    provider: str = "",
    model: str = "",
    mock: bool = False,
    holdout: list[str] | tuple[str, ...] = (),
    n_boot: int | None = None,
    demo: bool = False,
) -> dict[str, Any]:
    ev = cfg.evaluation
    n_boot = n_boot or ev.bootstrap
    seed = ev.seed

    scored = [s for s in subs if s.id in outputs.scores]
    events_all: list[str] = []
    for s in subs:
        if s.event not in events_all:
            events_all.append(s.event)
    # Match --holdout names loosely ("AdaHack 2026" finds "Adahack 2026").
    holdout_names = [h for h in holdout]
    holdout = [e for e in events_all if any(norm(h) == norm(e) for h in holdout_names)]
    unknown_holdout = [h for h in holdout_names if not any(norm(h) == norm(e) for e in events_all)]

    per_event_info = []
    usable_events = []
    for e in events_all:
        members = [s for s in scored if s.event == e]
        wins = sum(is_win(s, ev) for s in members)
        reason = ""
        if not members:
            reason = "no scored projects"
        elif wins == 0:
            reason = "no winners matched"
        elif wins == len(members):
            reason = "every project won"
        else:
            usable_events.append(e)
        per_event_info.append({"event": e, "n": len(members), "winners": wins, "excluded": reason})

    evalset = [s for s in scored if s.event in usable_events]
    if not evalset:
        raise EvaluationError(
            "No event has both winners and non-winners among the scored projects. "
            "Check that the winners file matched (run `judgepack inspect`)."
        )

    groups = np.array([s.event for s in evalset])
    y = np.array([int(is_win(s, ev)) for s in evalset])
    ai = np.array([outputs.scores[s.id].overall for s in evalset])
    words = np.array([s.word_count for s in evalset], dtype=float)
    X = feature_matrix(evalset)
    simple_pred, scheme = out_of_fold_logistic(X, y, groups)
    combo_pred, _ = out_of_fold_logistic(np.hstack([X, zscore_within(ai, groups)[:, None]]), y, groups)

    bt = None
    if outputs.pairwise is not None:
        bt = np.array([outputs.pairwise.strengths.get(s.id, np.nan) for s in evalset])

    method_defs: list[tuple[str, str, np.ndarray, bool, str]] = [
        ("ai", "AI rubric score", ai, True, "Mean of the model's weighted rubric scores across samples."),
    ]
    if bt is not None:
        method_defs.append(("pairwise", "AI pairwise comparisons", bt, True,
                            "Bradley-Terry strength from head-to-head judgements, both orders."))
    method_defs += [
        ("length", "Write-up length", words, False, "Word count of the write-up. Longer ranks higher."),
        ("simple", "Simple signals (learned)", simple_pred, False,
         f"Logistic regression on length, demo video, code repo and number of technologies ({scheme})."),
        ("simple_ai", "Simple signals + AI score", combo_pred, True,
         f"The same model with the AI score added as a fifth signal ({scheme})."),
    ]

    methods = []
    for key, label, s, is_ai, how in method_defs:
        if not np.isfinite(s).any():
            continue
        methods.append(
            {
                "key": key,
                "label": label,
                "is_ai": is_ai,
                "how": how,
                "auc": _ci(bootstrap_ci(within_group_auc, s, y, groups, n_boot, seed)),
                "recall_25": _ci(bootstrap_ci(lambda a, b, c: recall_at(a, b, c, 0.25), s, y, groups, n_boot, seed)),
                "recall_50": _ci(bootstrap_ci(lambda a, b, c: recall_at(a, b, c, 0.5), s, y, groups, n_boot, seed)),
                "r_precision": _num(r_precision(s, y, groups, seed)),
                "curve": [[round(f, 3), _num(r)] for f, r in recall_curve(s, y, groups)],
            }
        )
    arrays = {d[0]: d[2] for d in method_defs}

    comparisons = []
    for a, b, question in [
        ("ai", "length", "Is the AI better than write-up length?"),
        ("ai", "simple", "Is the AI better than simple signals?"),
        ("simple_ai", "simple", "Does adding the AI score improve on simple signals?"),
        ("pairwise", "ai", "Are pairwise comparisons better than rubric scores?"),
    ]:
        if a in arrays and b in arrays and np.isfinite(arrays[a]).any() and np.isfinite(arrays[b]).any():
            r = paired_bootstrap(within_group_auc, arrays[a], arrays[b], y, groups, n_boot, seed)
            comparisons.append({"a": a, "b": b, "question": question, **{k: _num(v) for k, v in r.items()}})

    # Length: does the AI just reward long write-ups?
    resid, slope = residualise_within(ai, np.log1p(words), groups)
    length_info = {
        "ai_length_corr": _num(rank_correlation_within(ai, words, groups)),
        "length_auc": next((m["auc"] for m in methods if m["key"] == "length"), None),
        "ai_adjusted_auc": _ci(bootstrap_ci(within_group_auc, resid, y, groups, n_boot, seed)),
        "slope_per_log_word": _num(slope),
    }

    # Consistency between independent samples of the same prompt.
    all_scores = [outputs.scores[s.id] for s in scored]
    consistency: dict[str, Any] = {
        "samples": cfg.model.samples,
        "mean_sd": _num(float(np.mean([p.overall_sd for p in all_scores]))) if all_scores else None,
        "unstable": sum(p.unstable for p in all_scores),
        "thin_evidence": sum(p.thin_evidence for p in all_scores),
        "winners_flagged": sum(1 for s in scored if is_win(s, ev) and outputs.scores[s.id].needs_human_look),
        "winners_total": sum(1 for s in scored if is_win(s, ev)),
        "test_retest": None,
    }
    multi = [s for s in scored if len(outputs.scores[s.id].sample_overalls) >= 2]
    if len(multi) >= 3:
        g2 = np.array([s.event for s in multi])
        s0 = np.array([outputs.scores[s.id].sample_overalls[0] for s in multi])
        s1 = np.array([outputs.scores[s.id].sample_overalls[1] for s in multi])
        consistency["test_retest"] = _num(rank_correlation_within(s0, s1, g2))

    pairwise_info = None
    if outputs.pairwise is not None and outputs.pairwise.judgements:
        pw = outputs.pairwise
        n = len(pw.judgements)
        agree, both = pw.order_consistency()
        pairwise_info = {
            "judgements": n,
            "first_position_rate": _ci(wilson_interval(pw.first_position_wins, n)),
            "order_consistency": _ci(wilson_interval(agree, both)),
            "pairs_both_orders": both,
            "rubric_corr": _num(rank_correlation_within(ai, bt, groups)) if bt is not None else None,
        }

    challenges = _challenge_results(subs, cfg, outputs.briefs)

    percentiles = within_group_percentile(ai, groups)
    pct_by_id = {s.id: float(p) for s, p in zip(evalset, percentiles)}
    projects = []
    for s in scored:
        sc = outputs.scores[s.id]
        projects.append(
            {
                "id": s.id,
                "event": s.event,
                "title": s.title,
                "won": is_win(s, ev),
                "prizes": s.prizes,
                "ai": round(sc.overall, 3),
                "ai_sd": round(sc.overall_sd, 3),
                "percentile": _num(pct_by_id.get(s.id, math.nan)),
                "pairwise": _num(outputs.pairwise.strengths.get(s.id, math.nan)) if outputs.pairwise else None,
                "words": s.word_count,
                "thin_evidence": sc.thin_evidence,
                "unstable": sc.unstable,
                "in_evaluation": s.event in usable_events,
            }
        )

    for info in per_event_info:
        e = info["event"]
        info["holdout"] = e in holdout
        if e in usable_events:
            m = groups == e
            info["ai_auc"] = _num(within_group_auc(ai[m], y[m], groups[m]))
            info["length_auc"] = _num(within_group_auc(words[m], y[m], groups[m]))

    holdout_info = None
    hold = [e for e in holdout if e in usable_events]
    if holdout_names:
        if hold:
            hm = np.isin(groups, hold)
            holdout_info = {
                "events": hold,
                "ai_auc": _ci(bootstrap_ci(within_group_auc, ai[hm], y[hm], groups[hm], n_boot, seed)),
                "length_auc": _ci(bootstrap_ci(within_group_auc, words[hm], y[hm], groups[hm], n_boot, seed)),
                "other_ai_auc": _ci(bootstrap_ci(within_group_auc, ai[~hm], y[~hm], groups[~hm], n_boot, seed))
                if (~hm).any() else None,
            }
        else:
            holdout_info = {"events": [], "missing": unknown_holdout or list(holdout_names)}

    usage_info = None
    if usage is not None:
        usage_info = usage.to_dict()
        cost = usage.cost(cfg.model.price_input_per_million, cfg.model.price_output_per_million)
        usage_info["cost_usd"] = _num(cost) if cost is not None else None
        usage_info["cost_per_project_usd"] = _num(cost / len(subs)) if cost is not None and subs else None

    ai_method = next(m for m in methods if m["key"] == "ai")
    results: dict[str, Any] = {
        "meta": {
            "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
            "event_name": cfg.event_name,
            "provider": provider,
            "model": model,
            "mock": mock,
            "demo": demo,
            "events": usable_events,
            "events_all": events_all,
            "n_projects": len(evalset),
            "n_winners": int(y.sum()),
            "win_label": describe_win_label(ev),
            "samples": cfg.model.samples,
            "bootstrap": n_boot,
            "baseline_scheme": scheme,
            "pairwise": outputs.pairwise is not None,
            "errors": outputs.errors,
        },
        "methods": methods,
        "comparisons": comparisons,
        "length": length_info,
        "consistency": consistency,
        "pairwise": pairwise_info,
        "challenges": challenges,
        "per_event": per_event_info,
        "holdout": holdout_info,
        "signal_weights": [{"signal": s, "weight": round(w, 3)} for s, w in signal_weights(X, y)],
        "usage": usage_info,
        "projects": projects,
        "redacted_projects": sum(1 for s in subs if s.redactions),
        "truncated_projects": sum(1 for s in subs if s.truncated),
    }
    results["headline"] = _headline(results, ai_method)
    results["caveats"] = _caveats(results, holdout_names)
    if unknown_holdout:
        results["caveats"].append(f"Held-out event(s) not found: {', '.join(unknown_holdout)}. Check the spelling.")
    return results


def _challenge_results(all_subs: list[Submission], cfg: Config, briefs: dict[str, Brief]) -> list[dict[str, Any]]:
    out = []
    events: list[str] = []
    for s in all_subs:
        if s.event not in events:
            events.append(s.event)
    for event in events:
        out += _event_challenge_results(event, [s for s in all_subs if s.event == event], cfg, briefs)
    return out


def _event_challenge_results(event: str, subs: list[Submission], cfg: Config,
                             briefs: dict[str, Brief]) -> list[dict[str, Any]]:
    out = []
    any_opt_in = any(s.opted_in_raw for s in subs)
    for c in cfg.briefed_challenges(event):
        fits = {}
        for s in subs:
            b = briefs.get(s.id)
            if b:
                f = next((x.fit for x in b.challenge_fits if x.challenge_id == c.id), "none")
                fits[s.id] = f
        assessed = len(fits)
        winners = [s for s in subs if any(c.matches(p) for p in s.prizes)]
        entry: dict[str, Any] = {
            "id": c.id,
            "event": event,
            "name": c.name,
            "sponsor": c.sponsor,
            "assessed": assessed,
            "strong": sum(1 for f in fits.values() if f == "strong"),
            "possible": sum(1 for f in fits.values() if f == "possible"),
            "winners": [{"title": s.title, "event": s.event, "fit": fits.get(s.id, "not assessed")} for s in winners],
        }
        if any_opt_in:
            opted = [s for s in subs if c.matches(s.opted_in_raw)]
            entry["opted_in"] = len(opted)
            entry["opted_in_no_fit"] = [s.title for s in opted if fits.get(s.id) == "none"]
            entry["strong_not_opted"] = [
                s.title for s in subs if fits.get(s.id) == "strong" and not c.matches(s.opted_in_raw)
            ]
        out.append(entry)
    return out


def _headline(results: dict[str, Any], ai_method: dict[str, Any]) -> dict[str, Any]:
    auc = ai_method["auc"]
    length = next((m for m in results["methods"] if m["key"] == "length"), None)
    comp = next((c for c in results["comparisons"] if c["a"] == "ai" and c["b"] == "length"), None)
    sentence = (
        f"Given a winner and a non-winner from the same event, the AI ranked the winner higher "
        f"{_pct(auc[0])} of the time"
        + (f" (95% interval {_pct(auc[1])} to {_pct(auc[2])})" if auc[1] is not None else "")
        + "."
    )
    if length:
        sentence += f" Write-up length alone managed {_pct(length['auc'][0])}."
    verdict = "unknown"
    if comp and comp["lo"] is not None:
        if comp["lo"] > 0:
            verdict = "better"
        elif comp["hi"] < 0:
            verdict = "worse"
        else:
            verdict = "inconclusive"
    verdict_text = {
        "better": "The AI beat the length baseline, and the interval for the difference excludes zero.",
        "worse": "The AI did worse than simply ranking by write-up length.",
        "inconclusive": "With this much data, the AI and the length baseline can't be told apart: "
        "the interval for their difference includes zero.",
        "unknown": "",
    }[verdict]
    return {
        "auc": auc,
        "recall_50": ai_method["recall_50"],
        "sentence": sentence,
        "verdict": verdict,
        "verdict_text": verdict_text,
    }


def _caveats(results: dict[str, Any], holdout: list[str] | tuple[str, ...]) -> list[str]:
    meta = results["meta"]
    notes = []
    if meta.get("demo"):
        notes.append(
            "This is the built-in demo: the projects and winners were invented to show how the report works, "
            "so these numbers say nothing about real judging."
        )
    if meta["mock"]:
        notes.append(
            "These numbers come from the mock provider, which uses simple text rules, not an AI model. "
            "They show what the report looks like and must not be quoted as results."
        )
    if meta["n_winners"] < 15:
        notes.append(
            f"There are only {meta['n_winners']} winners, so the intervals are wide. "
            "Treat any difference smaller than the intervals as noise."
        )
    notes.append(
        "Winners are chosen after live demos and pitches, which the tool never sees. "
        "Perfect agreement is impossible and isn't the aim; the question is whether it helps judges prepare and triage."
    )
    if results["redacted_projects"]:
        notes.append(
            f"{results['redacted_projects']} write-up(s) mentioned their own result (for example 'we won...'). "
            "Those sentences were removed before the model saw them."
        )
    if not holdout:
        notes.append(
            "Older Devpost pages may be in the model's training data. The cleanest test is an event held after "
            "the model's training cutoff: mark it with --holdout and compare."
        )
    length = results["length"]
    if length["length_auc"] and length["length_auc"][0] is not None and length["length_auc"][0] >= 0.6:
        notes.append(
            "Longer write-ups won more often here, so length is a strong baseline. The AI only adds value "
            "where it beats it."
        )
    tr = results["consistency"]["test_retest"]
    if tr is not None and tr < 0.7:
        notes.append("Scores varied noticeably between runs; rely on the averaged score, not any single sample.")
    if meta["errors"]:
        notes.append(f"{len(meta['errors'])} model call(s) failed; affected projects are missing from some analyses.")
    return notes
