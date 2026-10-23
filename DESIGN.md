# Design notes

Why JudgePack works the way it does. Each section is a decision, the reason for it, and what it costs.

## 1. Help judges decide; don't decide for them

The tempting version of this project is "an AI that scores hackathon projects". I didn't build that, for three reasons.

- **Judges see things the model can't.** Winners are decided after live demos and pitches. The model only has the write-up.
- **Accountability.** Participants accept a decision from a panel of people. "The model ranked you 14th" is not something an organiser should ever have to say, least of all at a hackathon for gender minorities, where many people are at their first event.
- **The real bottleneck is preparation.** Judges are volunteers with a couple of hours. Most of their time goes on working out what each project is. A briefing that does that for them frees time for the conversation, which is the part only a person can do.

So the main output is the **judge pack**: what the project is, three questions to ask, claims to check, and challenge fit. Scores exist for organiser triage and so the tool can be tested.

## 2. Judges never see the AI's scores first

The judge pack contains no scores, on purpose. If a judge sees "AI score: 2.1" before meeting a team, that number pulls their own judgement towards it (anchoring). The judging would get less independent, which is the opposite of the goal.

Scores live in a separate **organiser triage** page with a warning at the top. Organisers can use it to spot projects that need a closer look, or to check that no strong project was overlooked.

## 3. What the model sees, and what it never sees

The model sees the title, tagline, technologies, whether there's a code repo or demo video, and the write-up. It never sees:

- **Personal data.** Ingest keeps a whitelist of project fields. Everything else, including names and emails, is dropped when the file loads. It's a whitelist rather than a blacklist so a new column in next year's export can't leak by accident.
- **Outcomes.** Teams often edit their Devpost page after the event ("Update: we won 2nd place!"). Left in, that hands the model the answer. Sentences where a team reports its own result are removed first, and the number removed is reported. The patterns are deliberately narrow so "in the first place" or "the winner of each round" survive; the tests pin down both sides.
- **Prize opt-ins.** The model judges challenge fit from the project alone. That makes the opt-in data useful as a check: the triage page lists teams that opted in with no clear fit, and strong fits that *didn't* opt in. First-time teams often don't know to opt in, so the second list is a fairness feature.

## 4. It should know when it doesn't know

Every criterion score comes with:

- **evidence**: the facts from the write-up it relied on, or what's missing;
- **evidence_level**: strong, weak or none. With no evidence, the score is capped at 3.

If half or more of the criteria have no evidence, the project is flagged **thin write-up**. The tool says "judge this one in person" instead of guessing low. A short write-up isn't a bad project.

The JSON fields are ordered so the model writes its evidence *before* its score, and its reason *before* its verdict. Models generate left to right, so this makes the number follow from the reasoning rather than the reverse.

## 5. Several samples, not one

Each project is scored three times, independently, at temperature 0.4, and the scores are averaged. Two reasons:

- **Stability.** One sample can be an unlucky draw. The average moves less.
- **Uncertainty for free.** If the three samples disagree by half a point or more, the model wasn't sure, so the project is flagged **unstable**. That flag is more useful than any single score.

The report shows how well two independent runs agree (rank correlation). Temperature 0 would make the samples identical and throw that signal away.

## 6. Scores versus comparisons

People, and models, are often more consistent at "which of these two is better?" than at "rate this out of 5". So there's a second method: show the model two projects and ask which experienced judges would rate higher.

Two things make this work:

- **Both orders.** Research on LLM judges (for example Zheng et al., 2023, on MT-Bench) has found that models favour whichever answer comes first, and longer answers. Every pair is shown as A-then-B and B-then-A. That cancels the first-position preference and measures it: the report shows how often the model picked whichever came first (a fair judge: 50%) and how often it gave the same answer in both orders.
- **Bradley-Terry.** Each project gets a strength *p*, and the chance that *i* beats *j* is *p_i / (p_i + p_j)*. The strengths are fitted with the MM algorithm (Hunter, 2004): repeatedly set each *p_i* to its wins divided by the sum, over its comparisons, of 1 / (*p_i + p_j*). A small prior (one win and one loss against an average opponent) keeps the estimate finite for a project that won or lost everything.

Each project meets about four others rather than all of them, which keeps the number of calls linear in the number of projects.

## 7. Evaluating honestly

This is the part I care most about. An evaluation that can only say "it works" isn't an evaluation.

**What counts as a win.** By default any prize, except prizes about who the team is rather than the project (Best Beginner Team). The label definition is in the config and printed on the report, because it changes the numbers.

**Compare within an event.** Winners are picked relative to the other projects in the room. So the main metric takes every pair of one winner and one non-winner *from the same event* and asks how often the method ranks the winner higher. That's a within-event AUC: 50% is chance, 100% means every winner beat every non-winner. It also has a plain-English reading, which matters when explaining it to organisers.

**Intervals, always.** A few years of one hackathon gives a few dozen winners. Every number gets a 95% bootstrap interval: resample projects with replacement inside each event, recompute, repeat 2,000 times. Comparisons between two methods use the *same* resamples for both, so the interval is for the difference itself. If it includes zero, the report says the two can't be told apart. It doesn't round that up to a win.

**Baselines the AI has to beat.**

- **Write-up length.** LLM judges are known to favour longer text, and longer write-ups may also genuinely win more often. If the AI can't beat word count, it isn't adding judgement.
- **Simple signals, learned.** A logistic regression on length, demo video, code repo and number of technologies, trained leave-one-event-out (train on other years, predict this one) so no project is scored by a model that saw its own result.
- **Simple signals + AI.** The same model with the AI score added. Does the AI add anything *on top of* the obvious signals?

**Is it just length?** The report regresses the AI score on log word count within events and checks whether what's left still separates winners. If it doesn't, much of what the AI "sees" may be length.

**Training-data leakage.** Older Devpost pages may be in the model's training data. An event held after the model's training cutoff is the cleanest test. Mark it with `--holdout` and the report compares it with the rest. A large drop on the held-out event would suggest memory rather than judgement.

**Shortlisting.** The curve shows, for any shortlist size, what share of eventual winners it would have held. That turns the result into an organiser's question: how many projects need a close look?

## 8. Comparing with the judges' score sheets

Winners are a blunt signal: one bit per project. AdaHack 2026's paper sheets give a score for every team in every challenge, so they allow a much better test.

**Only compare like with like.** Each sponsor challenge had its own judges, who may score more generously or harshly than others. So every comparison is made within a challenge: for each pair of teams in the same challenge, does the AI put them in the same order as the judges? That's a concordance. 50% is chance, and pairs the judges scored equally are skipped.

**Only compare what a write-up can show.** The sheet has five marks. Teamwork and presentation only exist in the room, so they're left out; originality, theme and project quality are compared, using the sheet's own weights (quality is out of 20, the others out of 10). The judges' full totals are reported separately for reference. The sheet itself changed between years (2025 had one "innovation" mark and a "learning" mark), so each year's sheet is set separately in the config.

**Hold the AI to a human bar, not to perfection.** Finalists were scored again in round 2 by different judges. How often those two sets of judges agreed on the order of finalists is the fairest ceiling there is. On 2026's 14 finalist pairs they agreed 71% of the time, which says that even careful human judging is far from deterministic.

**Matching sheets to projects is part of the work.** Sheets say "team 7", not a project name. The tool matches by table number, and when several projects share a table (one team submitting a project per challenge) it uses the challenge the project opted into on Devpost. It reports every row it couldn't match, totals that don't add up, and teams judged for a challenge they never opted into, rather than guessing. On the 2026 sheets it matched all 40 rows and found two totals that didn't add up.

## 9. Messy real exports

Real Devpost exports weren't as tidy as the demo. AdaHack's own exports turned up: a table-number column asked as a free-text question ("Used to be table 6, now 23"); test entries and late online submissions that organisers had hidden; the same project submitted once per challenge; and a decorative 🏆 in a feature list that the outcome filter first mistook for a team announcing its win. Each is handled explicitly: hidden entries are left out, duplicate submissions are merged (same title, or write-ups at least 90% identical), the last number in a table answer is used, and trophy emoji only count beside a word like "won". Each of these has a test.

## 10. Engineering choices

- **Gemini by default, Claude supported.** Gemini has a free tier, which matters for a student project and a volunteer-run event. Providers sit behind one small interface, so swapping is a config change. A **mock provider** (simple text rules, clearly labelled) lets the whole pipeline run in tests and demos without a key.
- **Validate everything.** Every reply is checked against a pydantic schema and a semantic check (every criterion present, scores 1 to 5). A bad reply is sent back with the error and retried, so one malformed answer doesn't sink a run.
- **Cache everything.** Replies are cached on disk, keyed by the exact prompt and settings. Re-running is free and gives identical numbers, an interrupted run resumes, and changing the rubric naturally invalidates the right entries.
- **Rate limiting and backoff.** Free-tier keys have per-minute and per-day limits. Calls are spaced evenly. A per-minute 429 waits as long as the server asks (or backs off exponentially if it doesn't say), but a used-up daily quota stops the whole run at once, because waiting a minute can't fix it. The first real run taught me that: it spent four hours retrying 315 calls that could never succeed. Bad keys and unknown model names also fail immediately with a clear message.
- **numpy only for the statistics.** AUC, bootstrap, logistic regression and Bradley-Terry are each a few lines, so every number in the report can be traced by hand.

## 11. Cost

The report counts calls and tokens from the API's own usage figures, and turns them into a cost if prices are set in the config. One evaluation makes roughly 1 + samples + opponents calls per project; the exact count is printed before anything runs, and cached calls cost nothing.

## 12. What I'd do next

- **Use it live at AdaHack 2027.** Give judges the pack before the demos, then ask whether it saved time and changed what they asked. Reviewing briefings after an event checks whether they're accurate, but only a live trial shows whether they help.
- **Keep the judging sheets every year.** With only 2026 typed up, the judge comparison rests on a few dozen pairs. Two or three more years would make it far more certain, and a shared spreadsheet on the day would save the typing.
- **Calibrate the rubric anchors** on past events, since clear anchors are the biggest lever on consistency.
- **Look for bias.** Check whether scores differ systematically by project type (hardware versus web, say) in ways judges' decisions don't.
