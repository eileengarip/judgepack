"""Command line interface.

  judgepack inspect   check how a Devpost export will be read (no model calls)
  judgepack brief     write the judge briefing pack (and optionally organiser triage)
  judgepack evaluate  compare the AI with past winners and/or the judges' score sheets
  judgepack demo      run everything on fictional data with the offline mock provider
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from importlib import resources
from pathlib import Path

from . import __version__
from .brief import generate_briefs
from .config import DEFAULT_MODELS, Config, ConfigError, load_config
from .evaluate import EvaluationError, ModelOutputs, compute_results, is_win
from .ingest import IngestError, IngestResult, Submission, WinnerReport, load_submissions, load_winners
from .judging import JudgingData, compute_judging_results, load_judging
from .llm import LLM, LLMError, QuotaExhausted, make_provider
from .pairwise import run_pairwise
from .report import render_evaluation, render_judge_pack, render_judging, render_triage
from .score import score_projects
from .util import slug


def parse_map(items: list[str]) -> dict[str, str]:
    out = {}
    for item in items or []:
        if "=" not in item:
            raise IngestError(f"--map expects FIELD=COLUMN, got '{item}'")
        fld, col = item.split("=", 1)
        out[fld.strip()] = col.strip().strip('"').strip("'")
    return out


def load_subs(args: argparse.Namespace, cfg: Config | None) -> IngestResult:
    ing = load_submissions(args.submissions, parse_map(args.map), cfg.max_description_chars if cfg else 12000,
                           include_hidden=getattr(args, "include_hidden", False))
    if cfg:
        for s in ing.submissions:
            s.event = cfg.canonical_event(s.event)
    return ing


def judging_event(args: argparse.Namespace, ing: IngestResult) -> str:
    if getattr(args, "judging_event", None):
        return args.judging_event
    event = sorted(ing.events)[-1]
    if len(ing.events) > 1:
        print(f"\nJudging sheets assumed to be from {event}. Use --judging-event to choose another year.")
    return event


def print_judging(data: JudgingData) -> None:
    print(f"\nJudging sheets ({data.event}): {data.rows} row(s) -> {len(data.entries)} judged entr(y/ies) "
          f"across {len({e.challenge_id for e in data.entries})} challenge(s), "
          f"{sum(1 for e in data.entries if e.finalist)} finalist(s).")
    for e in sorted(data.entries, key=lambda x: (x.challenge_name, x.team)):
        rounds = "+".join(str(r) for r in sorted(e.rounds))
        flag = "" if e.opted_in else "   (didn't opt in on Devpost)"
        print(f"    {e.challenge_name:<24} team {e.team:<3} -> {e.title}  [round {rounds}]{flag}")
    sections = [
        ("NOT MATCHED", data.unmatched), ("total doesn't match the marks", data.total_mismatches),
        ("opted in but no sheet", data.opted_but_unjudged), ("warning", data.warnings),
    ]
    for label, items in sections:
        for item in items:
            print(f"  {label}: {item}")
    if data.challenges_without_sheets:
        print(f"  no sheets at all for: {', '.join(data.challenges_without_sheets)}")
    cols = [c.column for c in data.comparable_columns]
    other = [c.column for c in data.columns if not c.criterion]
    print(f"  AI will be compared on: {', '.join(cols)}" + (f" (not {', '.join(other)}: can't be seen in a write-up)" if other else ""))


def apply_overrides(cfg: Config, args: argparse.Namespace) -> None:
    provider = getattr(args, "provider", None)
    model = getattr(args, "model", None)
    if provider:
        if provider != cfg.model.provider and not model:
            cfg.model.name = DEFAULT_MODELS[provider]
        cfg.model.provider = provider
    if model:
        cfg.model.name = model
    if getattr(args, "workers", None):
        cfg.model.workers = args.workers
    if getattr(args, "samples", None):
        cfg.model.samples = max(1, args.samples)
    if getattr(args, "rpm", None):
        cfg.model.requests_per_minute = args.rpm


def build_llm(cfg: Config, args: argparse.Namespace) -> LLM:
    provider = make_provider(cfg.model)
    return LLM(
        provider,
        cache_dir=args.cache,
        requests_per_minute=cfg.model.requests_per_minute,
        max_retries=cfg.model.max_retries,
    )


def model_label(llm: LLM) -> str:
    if llm.provider.is_mock:
        return "mock provider (no AI model)"
    return f"{llm.provider.name} {llm.provider.model}"


def print_ingest(ing: IngestResult, cfg: Config | None = None) -> None:
    print(f"Read {len(ing.files)} file(s), {len(ing.submissions)} projects across {len(ing.events)} event(s).")
    for f in ing.files:
        print(f"\n  {Path(f.path).name}: {f.kept} of {f.rows} rows kept (event if no event column: \"{f.event_default}\")")
        for fld, col in f.mapping.items():
            print(f"    {fld:<12} <- \"{col}\"")
        if f.personal_columns:
            print(f"    personal data, never read: {', '.join(f.personal_columns)}")
        if f.ignored_columns:
            print(f"    not used: {', '.join(f.ignored_columns)}")
        for reason in f.skipped:
            print(f"    skipped {reason}")
        if f.hidden:
            print(f"    left out {len(f.hidden)} hidden or draft submission(s): {', '.join(f.hidden)}")
        for note in f.merged:
            print(f"    duplicate: {note}")
    if ing.redacted:
        print(f"\n  {len(ing.redacted)} write-up(s) mentioned their own result; those sentences were removed:")
        for s in ing.redacted:
            print(f"    - {s.title} ({s.event})")
    if ing.truncated:
        print(f"  {len(ing.truncated)} long write-up(s) were shortened for the model.")
    if cfg:
        for event in ing.events:
            spec = cfg.event_spec(event)
            criteria = ", ".join(c.name for c in cfg.criteria_for(event))
            print(f"\n  {event}: theme {spec.theme or 'not set'}; AI criteria: {criteria}")
            if any(c.needs_theme for c in cfg.criteria) and not spec.theme:
                print("    (no theme in the config, so the theme criterion is left out for this year)")
            challenges = cfg.challenges_for(event)
            if not challenges:
                continue
            members = [s for s in ing.submissions if s.event == event]
            has_opt = any(s.opted_in_raw for s in members)
            for c in challenges:
                n = sum(1 for s in members if c.matches(s.opted_in_raw)) if has_opt else 0
                brief = "brief set" if c.brief else "NO BRIEF: fit won't be assessed"
                opted = f"{n} opted in" if has_opt else "opt-ins not in export"
                print(f"    {c.name:<34} {opted:<16} {brief}")


def print_winners(wr: WinnerReport, subs: list[Submission], cfg: Config | None) -> None:
    print(f"\nWinners file: {wr.matched} of {wr.rows} rows matched, {wr.winning_projects} winning project(s).")
    for f in wr.fuzzy:
        print(f"  close match, please check: {f}")
    for u in wr.unmatched:
        print(f"  NOT MATCHED {u}")
    if cfg:
        counted = sum(1 for s in subs if is_win(s, cfg.evaluation))
        print(f"  {counted} project(s) count as a win for the evaluation.")


def confirm(cfg: Config, llm: LLM, calls: int, args: argparse.Namespace) -> bool:
    if llm.provider.is_mock or getattr(args, "yes", False) or not sys.stdin.isatty():
        return True
    answer = input(
        f"\nAbout to make up to {calls:,} calls to {model_label(llm)} "
        f"(about {calls / max(cfg.model.requests_per_minute, 0.1):.0f} min at {cfg.model.requests_per_minute:g}/min). "
        "Cached calls are free. Continue? [y/N] "
    )
    return answer.strip().lower() in ("y", "yes")


def run_models(subs: list[Submission], cfg: Config, llm: LLM, *, briefs: bool, scores: bool, pairwise: bool) -> ModelOutputs:
    out = ModelOutputs()
    workers = cfg.model.workers
    try:
        if briefs:
            out.briefs, errs = generate_briefs(subs, cfg, llm, workers)
            out.errors += errs
        if scores:
            out.scores, errs = score_projects(subs, cfg, llm, workers)
            out.errors += errs
        if pairwise:
            out.pairwise, errs = run_pairwise(subs, cfg, llm, workers)
            out.errors += errs
    except QuotaExhausted:
        t = llm.usage.total()
        print(f"\nStopped early: {t.calls:,} new replies this run, {t.cache_hits:,} already in the cache.", file=sys.stderr)
        raise
    for err in out.errors[:10]:
        print(f"  warning: {err}", file=sys.stderr)
    if len(out.errors) > 10:
        print(f"  ...and {len(out.errors) - 10} more warnings", file=sys.stderr)
    return out


def write_scores_csv(path: Path, subs: list[Submission], outputs: ModelOutputs, cfg: Config, labelled: bool) -> None:
    crit = [c.id for c in cfg.criteria]  # every criterion; a year without a theme leaves that column blank
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        header = ["event", "project", "table", "ai_score", "ai_spread", *crit, "thin_evidence", "unstable", "words"]
        if outputs.pairwise:
            header.append("pairwise_strength")
        if labelled:
            header += ["won", "prizes"]
        w.writerow(header)
        for s in subs:
            sc = outputs.scores.get(s.id)
            if sc is None:
                continue
            row = [s.event, s.title, s.table, f"{sc.overall:.3f}", f"{sc.overall_sd:.3f}",
                   *[f"{sc.criterion_means[c]:.2f}" if c in sc.criterion_means else "" for c in crit],
                   sc.thin_evidence, sc.unstable, s.word_count]
            if outputs.pairwise:
                row.append(f"{outputs.pairwise.strengths.get(s.id, float('nan')):.3f}")
            if labelled:
                row += [is_win(s, cfg.evaluation), "; ".join(s.prizes)]
            w.writerow(row)


def write_outputs(out_dir: Path, subs: list[Submission], cfg: Config, llm: LLM, outputs: ModelOutputs,
                  results: dict | None = None, demo: bool = False, judging: dict | None = None) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    generated = datetime.now().strftime("%d %b %Y, %H:%M")
    label = model_label(llm)
    written: list[Path] = []
    events: list[str] = []
    for s in subs:
        if s.event not in events:
            events.append(s.event)
    for event in events:
        members = [s for s in subs if s.event == event]
        if outputs.briefs:
            p = out_dir / f"judge-pack-{slug(event)}.html"
            p.write_text(render_judge_pack(event, members, outputs.briefs, cfg, model_label=label,
                                           mock=llm.provider.is_mock, generated=generated, demo=demo), encoding="utf-8")
            written.append(p)
        if outputs.scores:
            p = out_dir / f"triage-{slug(event)}.html"
            p.write_text(render_triage(event, members, outputs.scores, outputs.briefs, cfg, model_label=label,
                                       mock=llm.provider.is_mock, generated=generated, demo=demo), encoding="utf-8")
            written.append(p)
    if outputs.briefs:
        p = out_dir / "briefs.json"
        p.write_text(json.dumps({k: v.model_dump() for k, v in outputs.briefs.items()}, indent=2, ensure_ascii=False),
                     encoding="utf-8")
        written.append(p)
    if outputs.scores:
        p = out_dir / "scores.csv"
        write_scores_csv(p, subs, outputs, cfg, labelled=results is not None)
        written.append(p)
    if results is not None:
        p = out_dir / "results.json"
        p.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
        written.append(p)
        p = out_dir / "evaluation.html"
        p.write_text(render_evaluation(results), encoding="utf-8")
        written.append(p)
    if judging is not None:
        p = out_dir / "judging.json"
        p.write_text(json.dumps(judging, indent=2, ensure_ascii=False), encoding="utf-8")
        written.append(p)
        p = out_dir / f"judges-{slug(judging['meta']['event'])}.html"
        p.write_text(render_judging(judging), encoding="utf-8")
        written.append(p)
    return written


def _finish(written: list[Path], llm: LLM) -> None:
    t = llm.usage.total()
    print(f"\nModel calls: {t.calls:,} new, {t.cache_hits:,} from cache, {t.failures:,} retried.")
    print("Wrote:")
    for p in written:
        print(f"  {p}")


def cmd_inspect(args: argparse.Namespace) -> int:
    cfg = load_config(args.config) if args.config else None
    ing = load_subs(args, cfg)
    print_ingest(ing, cfg)
    if args.winners:
        print_winners(load_winners(args.winners, ing.submissions), ing.submissions, cfg)
    if args.judging:
        if cfg is None:
            raise ConfigError("Reading judging sheets needs the config (-c), which says what each column means.")
        print_judging(load_judging(args.judging, ing.submissions, cfg, judging_event(args, ing), args.teams))
    print("\nNo model calls were made.")
    return 0


def cmd_brief(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    apply_overrides(cfg, args)
    ing = load_subs(args, cfg)
    print_ingest(ing, cfg)
    llm = build_llm(cfg, args)
    n = len(ing.submissions)
    calls = n + (n * cfg.model.samples if args.with_scores else 0)
    if not confirm(cfg, llm, calls, args):
        print("Cancelled.")
        return 1
    outputs = run_models(ing.submissions, cfg, llm, briefs=True, scores=args.with_scores, pairwise=False)
    _finish(write_outputs(args.out, ing.submissions, cfg, llm, outputs), llm)
    return 0


def _evaluate(cfg: Config, subs: list[Submission], llm: LLM, args: argparse.Namespace, *, pairwise: bool,
              holdout: list[str], out_dir: Path, demo: bool = False, winners: bool = True,
              judging: JudgingData | None = None) -> int:
    n = len(subs)
    calls = n + n * cfg.model.samples + (n * cfg.pairwise.opponents_per_project if pairwise else 0)
    if not confirm(cfg, llm, calls, args):
        print("Cancelled.")
        return 1
    outputs = run_models(subs, cfg, llm, briefs=not getattr(args, "no_briefs", False), scores=True, pairwise=pairwise)
    n_boot = getattr(args, "bootstrap", None)
    results = None
    if winners:
        results = compute_results(
            subs, cfg, outputs, usage=llm.usage, provider=llm.provider.name, model=llm.provider.model,
            mock=llm.provider.is_mock, holdout=holdout, n_boot=n_boot, demo=demo,
        )
    judged = None
    if judging is not None:
        judged = compute_judging_results(judging, subs, outputs.scores, cfg, pairwise=outputs.pairwise, n_boot=n_boot,
                                         mock=llm.provider.is_mock, demo=demo, provider=llm.provider.name,
                                         model=llm.provider.model)
    written = write_outputs(out_dir, subs, cfg, llm, outputs, results, demo=demo, judging=judged)
    if results:
        print("\n" + results["headline"]["sentence"])
        if results["headline"]["verdict_text"]:
            print(results["headline"]["verdict_text"])
    if judged:
        print("\n" + judged["headline"]["sentence"])
        if judged["headline"]["ceiling"]:
            print(judged["headline"]["ceiling"])
    if llm.provider.is_mock:
        print("(Mock provider: these numbers are not real results.)")
    if demo:
        print("(Demo data: the projects and winners are invented, so these numbers mean nothing.)")
    _finish(written, llm)
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    if not args.winners and not args.judging:
        raise EvaluationError("Give a winners file (-w), judging sheets (--judging), or both.")
    cfg = load_config(args.config)
    apply_overrides(cfg, args)
    ing = load_subs(args, cfg)
    print_ingest(ing, cfg)
    subs = ing.submissions
    if args.winners:
        wr = load_winners(args.winners, subs)
        print_winners(wr, subs, cfg)
        if wr.winning_projects == 0:
            raise EvaluationError("None of the winners matched a project. Fix the winners file and run `judgepack inspect`.")
    judging = None
    if args.judging:
        judging = load_judging(args.judging, subs, cfg, judging_event(args, ing), args.teams)
        print_judging(judging)
        if not judging.entries:
            raise EvaluationError("No judging sheet rows matched a project. Run `judgepack inspect --judging ...` to see why.")
        if not args.winners:
            subs = [s for s in subs if s.event == judging.event]  # only that year's projects need scoring
    llm = build_llm(cfg, args)
    return _evaluate(cfg, subs, llm, args, pairwise=args.pairwise, holdout=args.holdout or [],
                     out_dir=args.out, winners=bool(args.winners), judging=judging)


def cmd_demo(args: argparse.Namespace) -> int:
    data = Path(str(resources.files("judgepack") / "demo_data"))
    cfg = load_config(data / "demo.yaml")
    apply_overrides(cfg, args)
    ing = load_submissions(sorted(data.glob("demo-hack-*.csv")), None, cfg.max_description_chars)
    print("Demo: two fictional hackathons, invented projects, winners and judging sheets.\n")
    print_ingest(ing, cfg)
    wr = load_winners(data / "winners.csv", ing.submissions)
    print_winners(wr, ing.submissions, cfg)
    judging = load_judging(data / "judging.csv", ing.submissions, cfg, "Demo Hack 2025")
    print_judging(judging)
    llm = build_llm(cfg, args)
    return _evaluate(cfg, ing.submissions, llm, args, pairwise=True, holdout=["Demo Hack 2025"], out_dir=args.out,
                     demo=True, judging=judging)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="judgepack", description="AI briefing packs and triage for hackathon judges.")
    parser.add_argument("--version", action="version", version=f"judgepack {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    def data_args(p: argparse.ArgumentParser, config_required: bool = True) -> None:
        p.add_argument("--submissions", "-s", nargs="+", type=Path, required=True,
                       help="Devpost export CSV(s): one per event, or one file with an 'event' column")
        p.add_argument("--config", "-c", type=Path, required=config_required, help="event config YAML")
        p.add_argument("--map", action="append", default=[], metavar="FIELD=COLUMN",
                       help='name a column yourself, e.g. --map description="About The Project"')
        p.add_argument("--include-hidden", action="store_true",
                       help="keep submissions marked Hidden on Devpost (normally tests, spam or duplicates)")

    def judging_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("--judging", "-j", type=Path, help="typed-up judging sheets CSV")
        p.add_argument("--judging-event", help="which year the sheets are from (default: the latest)")
        p.add_argument("--teams", type=Path, help="optional CSV of team,challenge,project to settle unclear matches")

    def model_args(p: argparse.ArgumentParser, out_default: str = "out") -> None:
        p.add_argument("--out", "-o", type=Path, default=Path(out_default), help="output folder")
        p.add_argument("--provider", choices=sorted(DEFAULT_MODELS), help="override the config's provider")
        p.add_argument("--model", help="override the config's model name")
        p.add_argument("--workers", type=int, help="parallel requests")
        p.add_argument("--samples", type=int, help="scores per project (overrides the config; 1 is cheapest)")
        p.add_argument("--rpm", type=float, help="requests per minute (overrides the config; raise it on a paid key)")
        p.add_argument("--cache", type=Path, default=Path(".judgepack_cache"), help="cache folder")
        p.add_argument("--yes", "-y", action="store_true", help="don't ask before making model calls")

    p = sub.add_parser("inspect", help="check how files will be read; makes no model calls")
    data_args(p, config_required=False)
    p.add_argument("--winners", "-w", type=Path, help="winners CSV (project, prize, optional event)")
    judging_args(p)
    p.set_defaults(func=cmd_inspect)

    p = sub.add_parser("brief", help="write the judge briefing pack")
    data_args(p)
    model_args(p)
    p.add_argument("--with-scores", action="store_true", help="also write the organiser triage table")
    p.set_defaults(func=cmd_brief)

    p = sub.add_parser("evaluate", help="compare the AI with past winners and/or the judges' score sheets")
    data_args(p)
    model_args(p)
    p.add_argument("--winners", "-w", type=Path, help="winners CSV (project, prize, optional event)")
    judging_args(p)
    p.add_argument("--pairwise", action="store_true", help="also run pairwise comparisons (more calls)")
    p.add_argument("--holdout", nargs="+", metavar="EVENT", help="events held after the model's training cutoff")
    p.add_argument("--bootstrap", type=int, help="number of bootstrap resamples (default from config)")
    p.add_argument("--no-briefs", action="store_true", help="skip briefs (and challenge-matching results)")
    p.set_defaults(func=cmd_evaluate)

    p = sub.add_parser("demo", help="run everything on fictional data (offline by default)")
    model_args(p, out_default="out/demo")
    p.set_defaults(func=cmd_demo, provider="mock")
    return parser


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")  # emoji in a project title must not crash a Windows console
        except (AttributeError, ValueError):
            pass
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args) or 0)
    except (ConfigError, IngestError, EvaluationError, LLMError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nStopped. Finished calls are cached, so running the same command again carries on from here.",
              file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
