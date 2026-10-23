"""Judging sheets: matching sheets to projects, totals checks and the within-challenge comparison."""

import csv
import math

import numpy as np
import pytest

from judgepack.config import config_from_dict
from judgepack.ingest import IngestError, load_submissions
from judgepack.judging import compute_judging_results, load_judging
from judgepack.score import ProjectScore
from judgepack.stats import within_group_concordance, within_group_pairs

CFG = {
    "event": {"name": "Hack"},
    "criteria": [
        {"id": "originality", "name": "Originality", "description": "Is it original?"},
        {"id": "theme", "name": "Theme", "description": "Does it fit {theme}?"},
        {"id": "quality", "name": "Quality", "description": "Is it good?", "weight": 2},
    ],
    "events": {
        "Hack 2026": {
            "theme": "Sustainability",
            "challenges": [
                {"id": "ppl", "name": "Greener by Postcode", "prize_names": ["PPL", "Postcode Lottery"], "brief": "Go green."},
                {"id": "optiver", "name": "Carbon portfolio", "prize_names": ["Optiver"], "brief": "Credits."},
                {"id": "compsoc", "name": "CompSoc challenge", "prize_names": ["CompSoc"]},
            ],
        }
    },
    "judging_sheet": [
        {"column": "teamwork", "max": 10},
        {"column": "originality", "max": 10, "criterion": "originality"},
        {"column": "theme", "max": 10, "criterion": "theme"},
        {"column": "presentation", "max": 10},
        {"column": "quality", "max": 20, "criterion": "quality"},
    ],
}


def write(path, header, rows):
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)
    return path


@pytest.fixture
def setup(tmp_path):
    cfg = config_from_dict(CFG)
    sub_csv = write(
        tmp_path / "hack-2026.csv",
        ["Project Title", "About The Project", "Opt-In Prizes", "What Is Your Table Number?", "Project Status"],
        [
            ["PeatPulse", "Peat monitoring " * 20, "CompSoc", "1", "Submitted (Gallery/Visible)"],
            ["Greener", "Street greening " * 30, "Postcode Lottery", "1", "Submitted (Gallery/Visible)"],
            ["Carbon Lab", "Carbon credits " * 40, "Optiver", "1", "Submitted (Gallery/Visible)"],
            ["OurPatch", "Grow your street " * 50, "Postcode Lottery", "7", "Submitted (Gallery/Visible)"],
            ["Borrow", "Borrow things " * 10, "", "11", "Submitted (Gallery/Visible)"],
            ["EcoBalance", "Balance " * 60, "Optiver", "23 Travel", "Submitted (Gallery/Visible)"],
            ["Spam", "test", "Optiver", "3", "Submitted (Hidden)"],
        ],
    )
    subs = load_submissions([sub_csv]).submissions
    for s in subs:
        s.event = cfg.canonical_event(s.event)
    return cfg, subs, tmp_path


HEADER = ["round", "team", "challenge", "teamwork", "originality", "theme", "presentation", "quality", "total"]


def test_matches_by_table_then_opt_in_and_flags_problems(setup):
    cfg, subs, tmp = setup
    path = write(tmp / "judging.csv", HEADER, [
        [1, 1, "COMPSOC", 3.5, 4, 10, 8, 19, 44.5],   # three projects at table 1: opt-in decides
        [1, 1, "PPL", 5, 6, 9, 8, 13, 41],
        [1, 1, "OPTIVER", 0, 4, 5, 2, 8, 19],
        [1, 7, "PPL", 9, 7, 9, 8, 16, 49],
        [2, 7, "PPL", 10, 8, 10, 9, 18, 56],           # typed total is wrong (should be 55)
        [1, 11, "PPL", 7, 5, 9, 6, 12, 39],            # only project at its table, but didn't opt in
        [1, 23, "OPTIVER", 5, 5, 8, 6, 10, 34],        # "23 Travel" is table 23
        [1, 3, "OPTIVER", 5, 5, 8, 6, 10, 34],         # table 3's only submission was hidden
        [1, 9, "MYSTERY", 5, 5, 8, 6, 10, 34],         # not a challenge
        [1, 7, "PPL", 1, 1, 1, 1, 1, 5],               # duplicate sheet
    ])
    data = load_judging(path, subs, cfg, "hack 2026")
    got = {(e.title, e.challenge_id): sorted(e.rounds) for e in data.entries}
    assert got == {
        ("PeatPulse", "compsoc"): [1], ("Greener", "ppl"): [1], ("Carbon Lab", "optiver"): [1],
        ("OurPatch", "ppl"): [1, 2], ("Borrow", "ppl"): [1], ("EcoBalance", "optiver"): [1],
    }
    assert data.event == "Hack 2026"
    assert len(data.total_mismatches) == 1 and "56" in data.total_mismatches[0]
    assert any("Borrow" in n for n in data.not_opted_in)
    assert any("table number 3" in u for u in data.unmatched)
    assert any("MYSTERY" in u for u in data.unmatched)
    assert any("second round 1" in w for w in data.warnings)
    oursheet = next(e for e in data.entries if e.title == "OurPatch").rounds[1]
    assert oursheet.values["teamwork"] == 9  # the first sheet was kept, not the duplicate


def test_ambiguous_table_needs_a_teams_file(setup):
    cfg, subs, tmp = setup
    path = write(tmp / "judging.csv", HEADER, [[1, 1, "CompSoc challenge", 1, 1, 1, 1, 1, 5]])
    for s in subs:
        if s.title == "Greener":
            s.opted_in_raw = "CompSoc"  # now two projects at table 1 opted into CompSoc
    data = load_judging(path, subs, cfg, "Hack 2026")
    assert not data.entries and "teams file" in data.unmatched[0]
    teams = write(tmp / "teams.csv", ["team", "challenge", "project"], [[1, "CompSoc", "PeatPulse"]])
    data = load_judging(path, subs, cfg, "Hack 2026", teams)
    assert [e.title for e in data.entries] == ["PeatPulse"]


def test_out_of_range_and_missing_columns(setup):
    cfg, subs, tmp = setup
    path = write(tmp / "j.csv", HEADER, [[1, 7, "PPL", 12, 7, 9, 8, 16, ""]])
    data = load_judging(path, subs, cfg, "Hack 2026")
    assert data.entries[0].rounds[1].values["teamwork"] is None
    assert any("outside 0-10" in w for w in data.warnings)
    bad = write(tmp / "bad.csv", ["round", "team", "challenge", "originality"], [[1, 7, "PPL", 5]])
    with pytest.raises(IngestError, match="missing column"):
        load_judging(bad, subs, cfg, "Hack 2026")


def fake_score(pid, o, t, q):
    return ProjectScore(project_id=pid, overall=(o + t + 2 * q) / 4, overall_sd=0.0, sample_overalls=[1.0],
                        criterion_means={"originality": o, "theme": t, "quality": q}, evidence={},
                        evidence_levels={}, biggest_uncertainty="", thin_evidence=False)


def test_comparison_results_end_to_end(setup):
    cfg, subs, tmp = setup
    path = write(tmp / "judging.csv", HEADER, [
        [1, 1, "PPL", 5, 6, 9, 8, 13, 41],
        [1, 7, "PPL", 9, 7, 9, 8, 16, 49],
        [2, 7, "PPL", 10, 8, 10, 9, 18, 55],
        [1, 11, "PPL", 7, 5, 9, 6, 12, 39],
        [2, 11, "PPL", 7, 5, 9, 6, 14, 41],
        [1, 1, "OPTIVER", 0, 4, 5, 2, 8, 19],
        [1, 23, "OPTIVER", 5, 5, 8, 6, 10, 34],
    ])
    data = load_judging(path, subs, cfg, "Hack 2026")
    by_title = {s.title: s.id for s in subs}
    scores = {
        by_title["OurPatch"]: fake_score(by_title["OurPatch"], 4, 5, 5),
        by_title["Greener"]: fake_score(by_title["Greener"], 3, 4, 3),
        by_title["Borrow"]: fake_score(by_title["Borrow"], 2, 4, 2),
        by_title["Carbon Lab"]: fake_score(by_title["Carbon Lab"], 2, 2, 2),
        by_title["EcoBalance"]: fake_score(by_title["EcoBalance"], 3, 4, 3),
    }
    r = compute_judging_results(data, subs, scores, cfg, n_boot=200, mock=True)
    ai = next(m for m in r["methods"] if m["key"] == "ai")
    assert ai["value"][0] == 1.0  # the fake AI orders every within-challenge pair as the judges did
    assert ai["pairs"] == 4        # 3 PPL pairs + 1 Optiver pair; never across challenges
    assert r["finalists"]["n"] == 2 and r["finalists"]["judges"]["pairs"] == 1
    assert r["reach_final"]["finalists"] == 2
    assert r["meta"]["compared_columns"] == ["originality", "theme", "quality"]
    assert r["meta"]["left_out_columns"] == ["teamwork", "presentation"]
    assert {c["column"] for c in r["per_criterion"]} == {"originality", "theme", "quality"}
    assert any("mock provider" in c for c in r["caveats"])


def test_theme_column_dropped_when_theme_unknown(setup):
    cfg, subs, tmp = setup
    cfg.events[0].theme = ""
    assert [c.id for c in cfg.criteria_for("Hack 2026")] == ["originality", "quality"]
    path = write(tmp / "j.csv", HEADER, [[1, 7, "PPL", 9, 7, 9, 8, 16, 49], [1, 1, "PPL", 5, 6, 9, 8, 13, 41]])
    data = load_judging(path, subs, cfg, "Hack 2026")
    ids = {s.title: s.id for s in subs}
    scores = {ids["OurPatch"]: fake_score(ids["OurPatch"], 4, 0, 5), ids["Greener"]: fake_score(ids["Greener"], 3, 0, 3)}
    r = compute_judging_results(data, subs, scores, cfg, n_boot=100)
    assert r["meta"]["compared_columns"] == ["originality", "quality"]


def test_concordance_skips_reference_ties_and_counts_pairs():
    a = np.array([1, 2, 3, 1, 2])
    b = np.array([10, 20, 20, 5, 1])
    g = np.array(["x", "x", "x", "y", "y"])
    # x: pairs (1,2) and (1,3) are separated by b and ordered the same; (2,3) is a b-tie, skipped.
    # y: b says first > second, a says the opposite.
    assert within_group_pairs(b, g) == 3
    assert within_group_concordance(a, b, g) == pytest.approx(2 / 3)
    assert math.isnan(within_group_concordance([1, 2], [3, 3], ["x", "x"]))
