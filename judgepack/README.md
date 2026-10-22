# JudgePack

![tests](https://github.com/eileengarip/judgepack/actions/workflows/tests.yml/badge.svg)

AI briefing packs for hackathon judges, plus an honest test of whether the AI's judgement matches the real judges.

JudgePack reads a hackathon's Devpost submissions and produces:

- **A judge briefing pack.** One card per team: what the project is, three questions worth asking at the demo, claims to check, and which sponsor challenges it fits. It deliberately contains no scores.
- **An organiser triage table.** Rubric scores averaged over several independent samples, with flags for write-ups too thin to judge and scores the model wasn't sure about.
- **Two evaluation reports.** One compares the AI's rankings with past winners. The other compares them with the judges' own score sheets, and with how much the judges agreed with each other. Both use simple baselines and confidence intervals.

## Why

I helped run AdaHack, Scotland's hackathon for gender minorities in tech: first on the website and the Devpost submission platform, then as head of sponsorship. In 2026 it had more than 300 applications for 100 places.

Judging is the bottleneck. Judges are volunteers with a couple of hours, they meet each team cold, and they have to work out on the spot which projects fit which sponsor challenge. Many teams are at their first hackathon and don't always opt in to the prizes they'd be eligible for.

The obvious AI answer is "let the model score the projects". I think that's the wrong product. Judges should decide; the tool should help them prepare, stay consistent and spend their limited time on the right conversations. So the main output is a briefing, not a ranking, and the scores exist mainly so the tool can be tested.

## Results

> **Fill this in from your own run.** Never publish numbers produced by the mock provider.

### Against the judges' score sheets (AdaHack 2026)

AdaHack 2026's paper judging sheets were typed up: 40 sheets, 24 projects across 6 sponsor challenges. Each challenge had its own judges, so teams are only compared within a challenge. Only the marks a write-up can show are used (originality, theme, project quality).

| | Pairs put in the same order as the judges (95% interval) |
|---|---|
| AI rubric score vs round 1 judges | _run it_ |
| Write-up length vs round 1 judges (baseline) | 51% (29% to 73%), 39 pairs |
| Round 1 judges vs round 2 judges, finalists only (the fair bar) | 71% (44% to 100%), 14 pairs |
| AI vs round 2 judges, finalists only | _run it_ |

The last two rows matter most: different judges scoring the same finalists agreed on the order 71% of the time, so that's roughly the most anyone can expect of the AI. Write-up length predicted nothing in 2026, so the AI can't score well just by rewarding long write-ups.

### Against past winners

| | Winner ranked above a non-winner (95% interval) | Winners in top half |
|---|---|---|
| AI rubric score | _run it_ | _run it_ |
| AI pairwise comparisons | _run it_ | _run it_ |
| Write-up length (baseline) | _run it_ | _run it_ |
| Simple signals, learned (baseline) | _run it_ | _run it_ |

Data: _N_ submissions from AdaHack _years_, _W_ winners. Model: _model name_. Held-out event (after the model's training cutoff): _event_.

**Data notes.** The evaluation uses AdaHack 2024 to 2026, the years with known challenge briefs and winners (2026 also has the judges' score sheets). 2022 and 2023 are left out: 2023's winners were never recorded, and neither year's briefs were found. Submissions marked Hidden on Devpost (tests, spam and duplicates) are left out, and the same project submitted once per challenge is counted once.

## What judges said

> **Fill this in after the review.** _N_ judges from AdaHack 2026 compared the briefing for each team they judged with the demo they saw.

- Briefings with a factual error: _X of Y_
- Teams where the judge would have asked at least one suggested question: _X of Y_
- Claims under "Check in the demo" that turned out shaky: _X of Y_
- What I changed because of it: _..._

## Quick start

Needs Python 3.10 or newer.

```bash
git clone https://github.com/eileengarip/judgepack.git
cd judgepack
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[gemini,dev]"

judgepack demo                     # fictional data, offline, no API key needed
```

Open `out/demo/evaluation.html` and `out/demo/judge-pack-demo-hack-2025.html`. The demo uses a rule-based mock instead of a model, so the pages carry a "not real results" banner.

### On real data

1. **Export submissions** from the Devpost organiser dashboard as CSV, one file per event (for example `adahack-2024.csv`). The file name becomes the event name unless there's an `event` column.
2. **List the winners** in a CSV with columns `event`, `project` and `prize`, one row per prize.
3. **Type up the judging sheets** (optional, one year), one row per sheet column of scores:
   ```
   round,team,challenge,teamwork,originality,theme,presentation,quality,total
   1,7,PPL,9,7,9,8,16,49
   ```
   Teams are matched to projects by table number, and by challenge opt-in when several projects share a table.
4. **Fill in the config.** `config/adahack.yaml` holds the rubric and, for each year, its theme, challenge briefs and paper judging sheet (the sheet changed between years: 2025 marked innovation and learning out of 55, 2026 originality and theme out of 60). 2024, 2025 and 2026 are filled in.
5. **Check how the files will be read.** This makes no model calls:
   ```bash
   judgepack inspect -s data/adahack-*.csv -w data/winners.csv -c config/adahack.yaml \
       --judging data/judging.csv --judging-event "AdaHack 2026"
   ```
   It shows which column became which field, which columns were ignored as personal data, which write-ups mentioned their own result, which winners or sheets it couldn't match, and any sheet totals that don't add up.
6. **Add an API key.** Create one in [Google AI Studio](https://aistudio.google.com/), then:
   ```bash
   export GEMINI_API_KEY="..."        # Windows PowerShell: $env:GEMINI_API_KEY="..."
   ```
7. **Evaluate:**
   ```bash
   judgepack evaluate -s data/adahack-2024.csv data/adahack-2025.csv data/adahack-2026.csv \
       -w data/winners.csv --judging data/judging.csv --judging-event "AdaHack 2026" \
       -c config/adahack.yaml --pairwise --holdout "AdaHack 2026"
   ```
8. **Make a briefing pack** for an event:
   ```bash
   judgepack brief -s adahack-2026.csv -c config/adahack.yaml --with-scores
   ```

Every model reply is cached, so re-running is free, gives identical numbers, and picks up where an interrupted run stopped.

**Quotas.** Free-tier keys have a daily limit per model, and it can be small; check yours at [aistudio.google.com/rate-limit](https://aistudio.google.com/rate-limit). If the daily quota runs out, the run stops at once and says so. Run the same command after it resets (midnight Pacific time) and it carries on. To use fewer calls, `--no-briefs` skips the briefing packs (the evaluation only needs scores) and `--samples 1` scores each project once instead of three times. On a paid key, `--rpm 60` speeds things up; the full AdaHack run (324 calls) costs about $1–4 at October 2026 prices, and paid-tier prompts aren't used to improve Google's products.

## How it works

```
Devpost CSV ──► ingest ──► briefs ─────────────► judge pack
  (personal data     │         └─ challenge fits ─► organiser triage
   dropped, results  ├──► rubric scores × 3 samples ─┐
   redacted)         └──► pairwise, both orders ─────┴─► evaluation vs real winners
                                                          and baselines
                                    judging sheets ──► AI vs judges, and judges vs judges
```

The reasoning behind each decision is in [DESIGN.md](DESIGN.md). In short:

- **Assist, don't decide.** Judges get briefings, not scores. Scores stay in the organiser view, because seeing a number first anchors people.
- **Honest evaluation.** Winners are compared only with non-winners from the same event, and judges' scores only within the same challenge. Every result has a bootstrap interval. The AI has to beat write-up length and a small model of obvious signals; if it can't, the report says so.
- **A fair bar.** Finalists were scored by two different sets of judges, so the report shows how much humans agree with each other. The AI is held to that, not to perfection.
- **The model can't read the answer.** Prize opt-ins and outcomes are never shown to it, and sentences like "Update: we won 2nd place!" are removed first.
- **It knows when it doesn't know.** Each score cites its evidence; thin write-ups and unstable scores are flagged for a human.

## Privacy

Only project fields are read from the export: title, tagline, write-up, technologies, links, opt-ins and table number. Names, emails and anything else that looks personal are dropped when the file is loaded and never reach the model, the cache or a report. On the Gemini free tier, Google may use prompts to improve its products, so only send write-ups that are already public on the Devpost gallery, or use a paid key.

## Limitations

- Winners are chosen after live demos and pitches, which the tool never sees. It can't and shouldn't match the judges perfectly.
- A few years of one hackathon means a few dozen winners, so intervals are wide.
- Older Devpost pages may be in a model's training data. Use `--holdout` for events after the model's cutoff and compare.
- Teamwork and presentation are on the judging sheet but can't be judged from a write-up, so the AI isn't compared on them.
- Judging sheets exist for 2026 only, and each challenge had only a handful of entries: the judge comparison rests on a few dozen pairs.
- It hasn't been used during a live event yet. Reviewing briefings after an event checks whether they're accurate, but not whether they change how judges work on the day.

## Project layout

```
judgepack/
  ingest.py     reading exports, privacy, outcome redaction, matching winners
  prompts.py    every prompt, in one place
  llm.py        Gemini, Claude and mock providers; caching, retries, rate limits
  brief.py      judge briefings and challenge fit
  score.py      rubric scoring with self-consistency
  pairwise.py   head-to-head comparisons and the Bradley-Terry model
  baselines.py  the baselines the AI has to beat
  stats.py      within-event AUC, bootstrap, logistic regression (numpy only)
  evaluate.py   turns outputs and winners into results.json
  judging.py    reads the paper judging sheets and compares the AI with the judges
  report.py     the HTML pages
config/adahack.yaml   AdaHack's rubric, and each year's theme, challenge briefs and judging sheet
tests/                pytest suite, runs offline
```

## Tests

```bash
pytest
```

The suite runs offline: the mock provider stands in for the model, and the Gemini and Claude request builders are checked against stubbed clients.
