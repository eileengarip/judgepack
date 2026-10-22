"""End to end on the fictional demo data with the offline mock provider."""

import json
from importlib import resources
from pathlib import Path

import pytest

from judgepack.cli import main
from judgepack.config import ConfigError, config_from_dict, load_config
from judgepack.ingest import load_submissions
from judgepack.prompts import brief_prompt, render_submission

DATA = Path(str(resources.files("judgepack") / "demo_data"))


def test_demo_runs_end_to_end(tmp_path):
    out = tmp_path / "out"
    assert main(["demo", "--out", str(out), "--cache", str(tmp_path / "cache")]) == 0
    for name in ["evaluation.html", "results.json", "scores.csv", "briefs.json", "judging.json",
                 "judges-demo-hack-2025.html", "judge-pack-demo-hack-2024.html", "triage-demo-hack-2025.html"]:
        assert (out / name).exists(), name

    results = json.loads((out / "results.json").read_text())
    assert results["meta"]["mock"] is True
    assert results["meta"]["n_projects"] == 24 and results["meta"]["n_winners"] == 8
    keys = [m["key"] for m in results["methods"]]
    assert keys[:3] == ["ai", "pairwise", "length"]
    auc = results["methods"][0]["auc"]
    assert 0 <= auc[1] <= auc[0] <= auc[2] <= 1
    assert any("mock provider" in c for c in results["caveats"])
    assert results["redacted_projects"] == 1

    html = (out / "evaluation.html").read_text()
    assert "Mock output: not real results" in html and "<svg" in html
    pack = (out / "judge-pack-demo-hack-2024.html").read_text()
    assert "test.person@example.com" not in pack  # personal columns never reach outputs

    # Second run is served entirely from the cache.
    assert main(["demo", "--out", str(out), "--cache", str(tmp_path / "cache")]) == 0
    again = json.loads((out / "results.json").read_text())
    assert again["usage"]["total"]["calls"] == 0
    assert again["methods"][0]["auc"] == results["methods"][0]["auc"]


def test_model_never_sees_outcomes_or_opt_ins():
    cfg = load_config(DATA / "demo.yaml")
    subs = load_submissions(sorted(DATA.glob("demo-hack-*.csv"))).submissions
    draft = next(s for s in subs if s.title == "Draft Buddy")
    draft.prizes = ["2nd Place Overall"]
    assert "2nd place" not in brief_prompt(cfg, draft).lower()
    assert "won" not in render_submission(draft).lower().split()

    clinic = next(s for s in subs if s.title == "Clinic Queue")
    assert clinic.opted_in_raw == "Best Social Impact"
    assert clinic.opted_in_raw not in render_submission(clinic)  # fit is judged independently of opt-in


def test_holdout_names_match_loosely_and_unknown_ones_are_reported(tmp_path):
    from judgepack.cli import run_models
    from judgepack.evaluate import compute_results
    from judgepack.ingest import load_winners
    from judgepack.llm import LLM, MockProvider

    cfg = load_config(DATA / "demo.yaml")
    subs = load_submissions(sorted(DATA.glob("demo-hack-*.csv"))).submissions
    load_winners(DATA / "winners.csv", subs)
    llm = LLM(MockProvider(), cache_dir=tmp_path)
    outputs = run_models(subs, cfg, llm, briefs=False, scores=True, pairwise=False)
    r = compute_results(subs, cfg, outputs, mock=True, holdout=["demo hack 2025", "Nonexistent 2030"], n_boot=200)
    assert r["holdout"]["events"] == ["Demo Hack 2025"]
    assert any("Nonexistent 2030" in c for c in r["caveats"])
    assert [e["event"] for e in r["per_event"] if e["holdout"]] == ["Demo Hack 2025"]


def test_inspect_makes_no_calls(tmp_path, capsys):
    code = main(["inspect", "-s", *map(str, sorted(DATA.glob("demo-hack-*.csv"))),
                 "-w", str(DATA / "winners.csv"), "-c", str(DATA / "demo.yaml")])
    assert code == 0
    text = capsys.readouterr().out
    assert "personal data, never read" in text and "No model calls were made." in text


def test_config_validation():
    with pytest.raises(ConfigError, match="criterion"):
        config_from_dict({"event": {"name": "X"}, "criteria": []})
    with pytest.raises(ConfigError, match="twice"):
        config_from_dict({"event": {"name": "X"}, "criteria": [
            {"id": "a", "name": "A", "description": "d"}, {"id": "a", "name": "B", "description": "d"}]})
    cfg = config_from_dict({"event": {"name": "X"}, "criteria": [{"id": "a", "name": "A", "description": "d"}],
                            "evaluation": {"win_prizes": "overall", "exclude_prizes": "beginner"}})
    assert cfg.evaluation.win_prizes == ["overall"] and cfg.evaluation.exclude_prizes == ["beginner"]


def test_adahack_template_loads():
    root = Path(__file__).resolve().parents[1]
    cfg = load_config(root / "config" / "adahack.yaml")
    assert cfg.event_name == "AdaHack" and cfg.model.provider == "gemini"
    # 2026 has its theme and briefed challenges; a year not in the config drops the theme criterion.
    assert cfg.canonical_event("Adahack 2026") == "AdaHack 2026"
    assert "Sustainability" in next(c for c in cfg.criteria_for("Adahack 2026") if c.id == "theme").description
    assert [c.id for c in cfg.criteria_for("Adahack 2023")] == ["originality", "quality"]
    assert "Education" in next(c for c in cfg.criteria_for("Adahack 2024") if c.id == "theme").description
    assert {c.id for c in cfg.briefed_challenges("AdaHack 2026")} == {"jane-street", "g-research", "optiver", "ppl", "hoppers", "compsoc"}
    assert cfg.challenge_for_label("AdaHack 2026", "JANE STREET").id == "jane-street"
    assert cfg.challenge_for_label("AdaHack 2026", "G-RESEARCH").id == "g-research"
    assert cfg.challenge_for_label("AdaHack 2026", "PPL").id == "ppl"
    assert [c.criterion for c in cfg.sheet_for("AdaHack 2026")] == [None, "originality", "theme", None, "quality"]
    # 2025 had its own theme, challenges and sheet; Devpost's 2025 opt-in labels map to the right challenge.
    assert "Innovation" in next(c for c in cfg.criteria_for("Adahack 2025") if c.id == "theme").description
    assert sum(c.max for c in cfg.sheet_for("AdaHack 2025")) == 55
    assert sum(c.max for c in cfg.sheet_for("AdaHack 2026")) == 60
    assert sum(c.max for c in cfg.sheet_for("AdaHack 2024")) == 50
    assert cfg.sheet_for("AdaHack 2023") == []
    ch24 = cfg.challenges_for("AdaHack 2024")
    for label, want in [("HandsOn Challnge", "handson"), ("Hoppers Beginner Challenge Winners", "hoppers-beginner"),
                        ("Formula Student Challenge", "formula-student"), ("BlackRock Challenge", "blackrock"),
                        ("CompSoc Challenge", "compsoc"), ("HackathonsUK Challenge", "hackathons-uk")]:
        assert [c.id for c in ch24 if c.matches(label)] == [want], label
    ch25 = cfg.challenges_for("AdaHack 2025")
    hits = lambda label: [c.id for c in ch25 if c.matches(label)]
    assert hits("Hoppers + Optiver Challenge Winners") == ["hoppers-optiver"]
    assert hits("Hoppers Challenge Winners") == ["hoppers"]
    assert hits("People's Postcode Lottery Challenge Winners") == ["ppl"]
