"""HTML outputs: the judge pack, the organiser triage table and the evaluation report.

Each is one self-contained file with no external requests. They follow the reader's light
or dark setting and print cleanly.
"""

from __future__ import annotations

from html import escape
from typing import Any

from . import __version__, charts
from .config import Config
from .ingest import Submission
from .schemas import Brief
from .score import ProjectScore
from .util import slug

LIGHT = {
    "page": "#f9f9f7", "surface": "#fcfcfb", "surface-2": "#f1f0ec", "ink": "#0b0b0b", "ink-2": "#52514e",
    "muted": "#898781", "grid": "#e1e0d9", "axis": "#c3c2b7", "border": "rgba(11,11,11,0.10)",
    "series-1": "#2a78d6", "series-2": "#eb6834", "series-3": "#1baf7a", "series-4": "#eda100",
    "accent": "#2a78d6", "accent-ink": "#1c5cab", "accent-wash": "rgba(42,120,214,0.10)",
    "warning": "#fab219", "good-ink": "#006300",
}
DARK = {
    "page": "#0d0d0d", "surface": "#1a1a19", "surface-2": "#242422", "ink": "#f0efec", "ink-2": "#c3c2b7",
    "muted": "#898781", "grid": "#2c2c2a", "axis": "#383835", "border": "rgba(255,255,255,0.10)",
    "series-1": "#3987e5", "series-2": "#d95926", "series-3": "#199e70", "series-4": "#c98500",
    "accent": "#3987e5", "accent-ink": "#86b6ef", "accent-wash": "rgba(57,135,229,0.16)",
    "warning": "#fab219", "good-ink": "#0ca30c",
}


def _tokens(d: dict[str, str]) -> str:
    return "".join(f"--{k}:{v};" for k, v in d.items())


CSS = """
:root{color-scheme:light;/*LIGHT*/}
@media (prefers-color-scheme: dark){:root:where(:not([data-theme="light"])){color-scheme:dark;/*DARK*/}}
:root[data-theme="dark"]{color-scheme:dark;/*DARK*/}
*{box-sizing:border-box}
body{margin:0;background:var(--page);color:var(--ink);font:15px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif;-webkit-font-smoothing:antialiased}
main{max-width:980px;margin:0 auto;padding:40px 16px 72px}
h1{font-size:30px;line-height:1.2;font-weight:650;margin:4px 0 8px;letter-spacing:-0.01em}
h2{font-size:20px;line-height:1.3;font-weight:620;margin:40px 0 10px}
h3{font-size:13px;line-height:1.3;font-weight:620;margin:18px 0 6px;color:var(--ink-2)}
p{margin:8px 0}
a{color:var(--accent-ink);text-underline-offset:2px}
.eyebrow{font-size:13px;color:var(--ink-2);font-weight:560;margin:0}
.meta{font-size:13px;color:var(--ink-2);margin:6px 0 0}
.muted{color:var(--ink-2)}
.small{font-size:13px}
.lede{font-size:17px;max-width:68ch}
.prose{max-width:72ch}
.card{background:var(--surface);border:1px solid var(--border);border-radius:14px;padding:20px 22px;margin:16px 0}
.callout{background:var(--surface);border:1px solid var(--border);border-radius:14px;padding:16px 20px;margin:20px 0}
.callout ul{margin:6px 0 0;padding-left:20px}
.banner{display:flex;gap:12px;align-items:flex-start;background:var(--surface);border:1px solid var(--border);border-left:4px solid var(--warning);border-radius:10px;padding:12px 16px;margin:20px 0}
.banner .icon{flex:none;font-weight:700;width:22px;height:22px;border-radius:50%;background:var(--warning);color:#0b0b0b;display:grid;place-items:center;font-size:13px;margin-top:1px}
.banner strong{display:block}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:12px;margin:20px 0}
.tile{background:var(--surface);border:1px solid var(--border);border-radius:14px;padding:16px 18px}
.tile .label{font-size:13px;color:var(--ink-2)}
.tile .value{font-size:34px;font-weight:620;line-height:1.15;margin-top:4px}
.tile.hero .value{font-size:52px}
.tile .sub{font-size:13px;color:var(--ink-2);margin-top:2px}
table{width:100%;border-collapse:collapse;font-size:14px}
th{text-align:left;font-size:13px;font-weight:600;color:var(--ink-2);padding:8px 10px;border-bottom:1px solid var(--axis);vertical-align:bottom}
td{padding:9px 10px;border-bottom:1px solid var(--grid);vertical-align:top}
td.num,th.num{text-align:right;font-variant-numeric:tabular-nums}
td.num{white-space:nowrap}
td.proj{min-width:150px}
td.why{min-width:210px}
.table-wrap{overflow-x:auto;background:var(--surface);border:1px solid var(--border);border-radius:14px;padding:6px 8px;margin:14px 0}
.chips{display:flex;flex-wrap:wrap;gap:6px;margin:8px 0 0}
.chip{display:inline-block;padding:2px 9px;border-radius:999px;font-size:12px;background:var(--surface-2);color:var(--ink-2);border:1px solid var(--border)}
.pill{display:inline-block;padding:1px 9px;border-radius:999px;font-size:12px;font-weight:600;white-space:nowrap}
.fit-strong{background:var(--accent-wash);color:var(--accent-ink)}
.fit-possible{border:1px solid var(--axis);color:var(--ink-2)}
.flag{display:inline-block;font-size:12px;padding:1px 8px 1px 6px;border-radius:6px;background:var(--surface-2);border-left:3px solid var(--warning);margin:0 4px 4px 0;white-space:nowrap}
.won{display:inline-block;font-size:12px;font-weight:600;color:var(--good-ink)}
details{margin:8px 0 0}
summary{cursor:pointer;color:var(--accent-ink);font-size:14px}
figure{margin:12px 0;background:var(--surface);border:1px solid var(--border);border-radius:14px;padding:16px 18px}
figcaption{font-size:13px;color:var(--ink-2);margin-top:6px}
.legend{display:flex;flex-wrap:wrap;gap:6px 18px;list-style:none;padding:0;margin:0 0 8px;font-size:13px;color:var(--ink-2)}
.legend .key{display:inline-block;width:18px;height:3px;border-radius:2px;vertical-align:middle;margin-right:6px}
.legend .key.dot{width:10px;height:10px;border-radius:50%}
.k1{background:var(--series-1)}.k2{background:var(--series-2)}.k3{background:var(--series-3)}.k4{background:var(--series-4)}.kmuted{background:var(--muted)}
svg.chart{width:100%;height:auto;display:block;font-family:inherit}
svg .grid{stroke:var(--grid);stroke-width:1}
svg .axis{stroke:var(--axis);stroke-width:1}
svg .ref{stroke:var(--muted);stroke-width:1}
svg .ref-strong{stroke:var(--axis);stroke-width:1.5}
svg .ref-label{fill:var(--muted);font-size:12px}
svg .tick{fill:var(--muted);font-size:12px;font-variant-numeric:tabular-nums}
svg .axis-title{fill:var(--ink-2);font-size:12px}
svg .row-label{font-size:13px}
svg .row-label.emph{fill:var(--ink);font-weight:600}
svg .row-label.base{fill:var(--ink-2)}
svg .line{fill:none;stroke-width:2;stroke-linejoin:round;stroke-linecap:round}
svg .s1{stroke:var(--series-1)}svg .s2{stroke:var(--series-2)}svg .s3{stroke:var(--series-3)}svg .s4{stroke:var(--series-4)}
svg .ci{stroke-width:2;stroke-linecap:round}
svg .ci.emph{stroke:var(--series-1)}svg .ci.base{stroke:var(--muted)}
svg .dot{stroke:var(--surface);stroke-width:2}
svg .dot.emph{fill:var(--series-1)}svg .dot.base{fill:var(--muted)}
svg .dot-win{fill:var(--series-1);stroke:var(--surface);stroke-width:2}
svg .dot-other{fill:var(--muted);fill-opacity:.55;stroke:var(--surface);stroke-width:2}
svg .hit{fill:var(--ink);fill-opacity:0;stroke:none}
svg .hit:hover{fill-opacity:.06}
.index td:first-child{white-space:nowrap;color:var(--ink-2)}
.project h2{margin:2px 0 4px;font-size:22px}
.project .one-line{font-size:16px;margin:2px 0 6px}
.project .links{font-size:13px;color:var(--ink-2)}
.project .links a{margin-right:12px}
.project .table-no{font-size:12px;font-weight:600;color:var(--ink-2)}
.cols{display:grid;grid-template-columns:1fr 1fr;gap:6px 28px}
.cols ol,.cols ul{margin:4px 0;padding-left:20px}
.cols li{margin:3px 0}
.fits{border-top:1px solid var(--grid);margin-top:14px;padding-top:4px}
.fit-row{display:flex;gap:10px;align-items:baseline;margin:6px 0}
.fit-row .pill{flex:none}
.foot{font-size:12px;color:var(--muted);margin-top:12px}
th[data-sort]{cursor:pointer;user-select:none}
th[data-sort]:hover{color:var(--ink)}
th[aria-sort="ascending"]::after{content:" \\2191"}
th[aria-sort="descending"]::after{content:" \\2193"}
.printbtn{font:inherit;font-size:13px;padding:6px 12px;border-radius:8px;border:1px solid var(--border);background:var(--surface);color:var(--ink);cursor:pointer}
footer.page-foot{margin-top:48px;font-size:12px;color:var(--muted)}
@media (max-width:720px){.cols{grid-template-columns:1fr}h1{font-size:25px}.tile.hero .value{font-size:42px}}
@media print{
  body{background:#fff;color:#000}
  main{padding:0;max-width:none}
  .no-print{display:none!important}
  .card,figure,.tile,.table-wrap,.callout{break-inside:avoid;border-color:#ccc}
  .project{break-inside:avoid;page-break-inside:avoid}
  a{color:#000}
}
""".replace("/*LIGHT*/", _tokens(LIGHT)).replace("/*DARK*/", _tokens(DARK))

SORT_SCRIPT = """
document.querySelectorAll('table.sortable').forEach(function (table) {
  var heads = table.querySelectorAll('th[data-sort]');
  heads.forEach(function (th, col) {
    th.addEventListener('click', function () {
      var idx = Array.prototype.indexOf.call(th.parentNode.children, th);
      var dir = th.getAttribute('aria-sort') === 'descending' ? 'ascending' : 'descending';
      heads.forEach(function (h) { h.removeAttribute('aria-sort'); });
      th.setAttribute('aria-sort', dir);
      var body = table.tBodies[0];
      var rows = Array.prototype.slice.call(body.rows);
      var numeric = th.getAttribute('data-sort') === 'num';
      rows.sort(function (a, b) {
        var x = a.cells[idx].getAttribute('data-v') || a.cells[idx].textContent;
        var y = b.cells[idx].getAttribute('data-v') || b.cells[idx].textContent;
        var c = numeric ? (parseFloat(x) - parseFloat(y)) : x.localeCompare(y);
        return dir === 'ascending' ? c : -c;
      });
      rows.forEach(function (r) { body.appendChild(r); });
    });
  });
});
"""


def e(value: object) -> str:
    return escape(str(value), quote=True)


def page(title: str, body: str, script: str = "") -> str:
    js = f"<script>{script}</script>" if script else ""
    return (
        "<!doctype html>\n<html lang=\"en-GB\">\n<head>\n<meta charset=\"utf-8\">\n"
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n"
        f"<title>{e(title)}</title>\n<style>{CSS}</style>\n</head>\n<body>\n<main>\n{body}\n"
        f"<footer class=\"page-foot\">Made with JudgePack {e(__version__)}.</footer>\n</main>\n{js}\n</body>\n</html>\n"
    )


def mock_banner() -> str:
    return (
        '<div class="banner" role="note"><span class="icon" aria-hidden="true">!</span><div>'
        "<strong>Mock output: not real results</strong>"
        "This was produced by the offline mock provider, which uses simple text rules instead of an AI model. "
        "It shows the layout only. Run with a real provider before reading anything into it.</div></div>"
    )


def demo_banner() -> str:
    return (
        '<div class="banner" role="note"><span class="icon" aria-hidden="true">!</span><div>'
        "<strong>Fictional demo data</strong>"
        "Every project, sponsor and winner here was invented to show how JudgePack works. "
        "Numbers computed on it say nothing about real judging.</div></div>"
    )


def pct(v: float | None, digits: int = 0) -> str:
    return "n/a" if v is None else f"{v * 100:.{digits}f}%"


def ci_text(t: list | None) -> str:
    if not t or t[0] is None:
        return "n/a"
    if t[1] is None:
        return pct(t[0])
    return f"{pct(t[0])} <span class=\"muted\">({pct(t[1])} to {pct(t[2])})</span>"


# --------------------------------------------------------------------------- judge pack


def _fit_pill(fit: str) -> str:
    label = {"strong": "Strong fit", "possible": "Possible fit"}.get(fit, "No clear fit")
    return f'<span class="pill fit-{e(fit)}">{label}</span>'


def _links(sub: Submission) -> str:
    links = []
    if sub.url:
        links.append(f'<a href="{e(sub.url)}">Devpost</a>')
    code = next((l for l in sub.links if any(h in l.lower() for h in ("github.com", "gitlab.com", "bitbucket.org"))), "")
    if code:
        links.append(f'<a href="{e(code)}">Code</a>')
    video = sub.video or next((l for l in sub.links if any(h in l.lower() for h in ("youtube.com", "youtu.be", "vimeo.com"))), "")
    if video:
        links.append(f'<a href="{e(video)}">Video</a>')
    others = [l for l in sub.links if l not in (code, video)]
    for i, l in enumerate(others[:2], 1):
        links.append(f'<a href="{e(l)}">Link {i}</a>')
    return f'<div class="links">{"".join(links)}</div>' if links else ""


def _project_card(sub: Submission, brief: Brief | None, cfg: Config) -> str:
    anchor = f"p-{slug(sub.id)}"
    table = f'<div class="table-no">Table {e(sub.table)}</div>' if sub.table else ""
    if brief is None:
        return (
            f'<article class="card project" id="{anchor}">{table}<h2>{e(sub.title)}</h2>'
            '<p class="muted">No briefing: the model call failed for this project. Read the write-up on Devpost.</p>'
            f"{_links(sub)}</article>"
        )

    def items(xs: list[str], empty: str, tag: str = "ul") -> str:
        if not xs:
            return f'<p class="muted small">{empty}</p>'
        return f"<{tag}>" + "".join(f"<li>{e(x)}</li>" for x in xs) + f"</{tag}>"

    tech = "".join(f'<span class="chip">{e(t)}</span>' for t in sub.built_with)
    tech_html = f'<div class="chips">{tech}</div>' if tech else ""
    fits_html = ""
    briefed = cfg.briefed_challenges(sub.event)
    if briefed:
        rows, none = [], []
        names = {c.id: c.name for c in briefed}
        for f in brief.challenge_fits:
            if f.fit == "none":
                none.append(names.get(f.challenge_id, f.challenge_id))
            else:
                rows.append(
                    f'<div class="fit-row">{_fit_pill(f.fit)}<div><strong>{e(names.get(f.challenge_id, f.challenge_id))}</strong>'
                    f' <span class="muted">{e(f.reason)}</span></div></div>'
                )
        if none:
            rows.append(f'<p class="muted small">No clear fit: {e(", ".join(none))}.</p>')
        fits_html = '<section class="fits"><h3>Sponsor challenges</h3>' + "".join(rows) + "</section>"

    notes = [f"Write-up: {sub.word_count:,} words."]
    if sub.redactions:
        notes.append("A sentence reporting the team's result was removed before the model read it.")
    if sub.truncated:
        notes.append("The write-up was long and only the first part was read.")
    return f"""<article class="card project" id="{anchor}">
{table}<h2>{e(sub.title)}</h2>
<p class="one-line">{e(brief.one_line)}</p>
{_links(sub)}
<div class="cols">
<section><h3>What it is</h3><p>{e(brief.summary)}</p><p class="muted">{e(brief.how_it_works)}</p>{tech_html}
<h3>Strengths</h3>{items(brief.strengths, "None stood out from the write-up alone.")}</section>
<section><h3>Ask the team</h3>{items(brief.questions_for_team, "No questions generated.", "ol")}
<h3>Check in the demo</h3>{items(brief.unverified_claims, "Nothing flagged.")}</section>
</div>
{fits_html}
<p class="foot">{e(" ".join(notes))}</p>
</article>"""


def render_judge_pack(event: str, subs: list[Submission], briefs: dict[str, Brief], cfg: Config, *,
                      model_label: str, mock: bool, generated: str, demo: bool = False) -> str:
    rows = []
    names = {c.id: c.name for c in cfg.briefed_challenges(event)}
    for s in subs:
        b = briefs.get(s.id)
        strong = [names.get(f.challenge_id, f.challenge_id) for f in (b.challenge_fits if b else []) if f.fit == "strong"]
        chips = " ".join('<span class="chip">' + e(n) + "</span>" for n in strong)
        one_line = e(b.one_line) if b else '<span class="muted">No briefing</span>'
        rows.append(
            f"<tr><td>{e(s.table) if s.table else ''}</td>"
            f'<td><a href="#p-{slug(s.id)}">{e(s.title)}</a></td>'
            f"<td>{one_line}</td><td>{chips}</td></tr>"
        )
    cards = "\n".join(_project_card(s, briefs.get(s.id), cfg) for s in subs)
    body = f"""
<p class="eyebrow">{e(event)}</p>
<h1>Judge briefing pack</h1>
<p class="meta">{len(subs)} projects · generated {e(generated)} · {e(model_label)}</p>
{demo_banner() if demo else ''}{mock_banner() if mock else ''}
<div class="callout">
<strong>How to use this pack</strong>
<ul>
<li>Skim the one-liner and the three questions before you reach a team, then spend your time on the demo.</li>
<li>"Check in the demo" lists claims the write-up makes without support. Ask to see them.</li>
<li>This pack doesn't score or rank anyone. The decisions are yours. It was written by an AI from the Devpost write-up alone, so it can be wrong.</li>
</ul>
</div>
<p class="no-print"><button class="printbtn" onclick="window.print()">Print or save as PDF</button></p>
<h2>All projects</h2>
<div class="table-wrap"><table class="index"><thead><tr><th>Table</th><th>Project</th><th>What it is</th><th>Strong challenge fits</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table></div>
<h2>Projects</h2>
{cards}
"""
    return page(f"{event}: judge briefing pack", body)


# --------------------------------------------------------------------------- organiser triage


def render_triage(event: str, subs: list[Submission], scores: dict[str, ProjectScore], briefs: dict[str, Brief],
                  cfg: Config, *, model_label: str, mock: bool, generated: str, demo: bool = False) -> str:
    scored = [s for s in subs if s.id in scores]
    scored.sort(key=lambda s: -scores[s.id].overall)
    criteria = cfg.criteria_for(event)
    crit_heads = "".join(f'<th class="num" data-sort="num" title="{e(c.description)}">{e(c.name)}</th>' for c in criteria)
    rows = []
    flagged = 0
    for rank, s in enumerate(scored, 1):
        sc = scores[s.id]
        flags = []
        if sc.thin_evidence:
            flags.append('<span class="flag" title="Half or more criteria had no usable evidence in the write-up">Thin write-up</span>')
        if sc.unstable:
            flags.append('<span class="flag" title="Independent samples disagreed by half a point or more">Unstable score</span>')
        flagged += bool(flags)
        crit = "".join(
            f'<td class="num" data-v="{sc.criterion_means[c.id]:.3f}" title="{e(sc.evidence.get(c.id, ""))}">'
            f"{sc.criterion_means[c.id]:.1f}</td>"
            for c in criteria
            if c.id in sc.criterion_means
        )
        rows.append(
            f'<tr><td class="num" data-v="{rank}">{rank}</td><td class="proj" data-v="{e(s.title.lower())}"><strong>{e(s.title)}</strong>'
            f"{'<br><span class=muted>Table ' + e(s.table) + '</span>' if s.table else ''}</td>"
            f'<td class="num" data-v="{sc.overall:.4f}"><strong>{sc.overall:.2f}</strong>'
            f'<br><span class="muted small">± {sc.overall_sd:.2f}</span></td>{crit}'
            f'<td data-v="{len(flags)}">{"".join(flags)}</td><td class="small muted why">{e(sc.biggest_uncertainty)}</td></tr>'
        )

    opt_html = ""
    briefed = cfg.briefed_challenges(event)
    if briefed and briefs:
        has_opt = any(s.opted_in_raw for s in subs)
        blocks = []
        for c in briefed:
            fits = {s.id: next((f.fit for f in briefs[s.id].challenge_fits if f.challenge_id == c.id), "none")
                    for s in subs if s.id in briefs}
            strong = [s.title for s in subs if fits.get(s.id) == "strong"]
            line = f"<p><strong>{e(c.name)}</strong>: strong fit for {len(strong)} project(s)"
            line += (": " + e(", ".join(strong)) if strong else "") + ".</p>"
            if has_opt:
                opted = [s for s in subs if c.matches(s.opted_in_raw)]
                weak = [s.title for s in opted if fits.get(s.id) == "none"]
                missing = [s.title for s in subs if fits.get(s.id) == "strong" and not c.matches(s.opted_in_raw)]
                if weak:
                    line += f'<p class="small"><span class="flag">Opted in, no clear fit</span> {e(", ".join(weak))}</p>'
                if missing:
                    line += f'<p class="small"><span class="flag">Strong fit, didn\'t opt in</span> {e(", ".join(missing))}</p>'
            blocks.append(line)
        note = "" if has_opt else '<p class="muted small">The export has no prize opt-in column, so opt-in mismatches can\'t be checked.</p>'
        opt_html = "<h2>Sponsor challenges</h2><div class=\"card\">" + note + "".join(blocks) + "</div>"

    body = f"""
<p class="eyebrow">{e(event)}</p>
<h1>Organiser triage</h1>
<p class="meta">{len(scored)} projects · generated {e(generated)} · {e(model_label)} · {cfg.model.samples} sample(s) per project</p>
{demo_banner() if demo else ''}{mock_banner() if mock else ''}
<div class="banner" role="note"><span class="icon" aria-hidden="true">!</span><div><strong>For organisers only</strong>
Don't show these scores to judges before they've reached their own view. Seeing a number first pulls people towards it
(anchoring), which would make the judging less independent, not more.</div></div>
<div class="tiles">
<div class="tile"><div class="label">Projects scored</div><div class="value">{len(scored)}</div></div>
<div class="tile"><div class="label">Need a human look</div><div class="value">{flagged}</div><div class="sub">thin write-up or unstable scores</div></div>
</div>
<p class="prose muted small">Scores are the mean of {cfg.model.samples} independent samples on a 1 to 5 scale; ± is the spread between samples.
Hover a criterion score to see the evidence the model cited. Click a column heading to sort.</p>
<div class="table-wrap"><table class="sortable"><thead><tr><th class="num" data-sort="num">#</th><th data-sort="text">Project</th>
<th class="num" data-sort="num">AI score</th>{crit_heads}<th data-sort="num">Flags</th><th>What the demo needs to show</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table></div>
{opt_html}
"""
    return page(f"{event}: organiser triage", body, SORT_SCRIPT)


# --------------------------------------------------------------------------- evaluation report

SLOTS = {"ai": 1, "pairwise": 2, "length": 3, "simple": 4}


def _verdict_row(c: dict[str, Any]) -> str:
    if c.get("lo") is None:
        return "Not enough data"
    if c["lo"] > 0:
        return "Yes, and the interval excludes zero"
    if c["hi"] < 0:
        return "No, it is worse"
    return "Can't tell with this much data"


def render_evaluation(results: dict[str, Any]) -> str:
    m = results["meta"]
    methods = results["methods"]
    by_key = {x["key"]: x for x in methods}
    head = results["headline"]
    cons = results["consistency"]
    length = results["length"]

    tiles = [
        f'<div class="tile hero"><div class="label">Winner ranked above a non-winner</div>'
        f'<div class="value">{pct(head["auc"][0])}</div>'
        f'<div class="sub">95% interval {pct(head["auc"][1])} to {pct(head["auc"][2])} · chance is 50%</div></div>',
        f'<div class="tile"><div class="label">Winners in the AI\'s top half</div>'
        f'<div class="value">{pct(head["recall_50"][0])}</div><div class="sub">within each event</div></div>',
    ]
    if "length" in by_key:
        tiles.append(
            f'<div class="tile"><div class="label">Write-up length baseline</div>'
            f'<div class="value">{pct(by_key["length"]["auc"][0])}</div><div class="sub">the number to beat</div></div>'
        )
    if cons.get("test_retest") is not None:
        tiles.append(
            f'<div class="tile"><div class="label">Agreement between runs</div>'
            f'<div class="value">{cons["test_retest"]:.2f}</div><div class="sub">rank correlation, 1 = identical</div></div>'
        )

    forest = charts.forest_chart(
        [{"label": x["label"], "point": x["auc"][0], "lo": x["auc"][1], "hi": x["auc"][2], "emph": x["is_ai"]} for x in methods],
        "How often each method ranks a winner above a non-winner",
    )
    method_rows = "".join(
        f"<tr><td><strong>{e(x['label'])}</strong><br><span class=\"muted small\">{e(x['how'])}</span></td>"
        f'<td class="num">{ci_text(x["auc"])}</td><td class="num">{pct(x["recall_25"][0])}</td>'
        f'<td class="num">{pct(x["recall_50"][0])}</td><td class="num">{pct(x["r_precision"])}</td></tr>'
        for x in methods
    )

    curve_series = [
        {"label": by_key[k]["label"], "slot": SLOTS[k], "points": by_key[k]["curve"]} for k in SLOTS if k in by_key
    ]
    legend = "".join(f'<li><span class="key k{s["slot"]}"></span>{e(s["label"])}</li>' for s in curve_series)
    legend += '<li><span class="key kmuted"></span>Random order</li>'
    curve = charts.recall_chart(curve_series, "Winners captured by shortlists of increasing size")
    fractions = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]
    curve_head = "".join(f'<th class="num">Top {int(f * 100)}%</th>' for f in fractions)
    curve_rows = ""
    for s in curve_series:
        lookup = {round(x, 3): y for x, y in s["points"]}
        curve_rows += f"<tr><td>{e(s['label'])}</td>" + "".join(
            f'<td class="num">{pct(lookup.get(round(f, 3)))}</td>' for f in fractions
        ) + "</tr>"

    ai50 = by_key["ai"]["recall_50"][0]
    shortlist_text = (
        f"If organisers took the top half of each event by AI score for a closer look, {pct(ai50)} of the eventual "
        "winners would have been on that list."
        if ai50 is not None
        else ""
    )

    scatter_pts = [
        {"x": p["words"], "y": p["ai"], "won": p["won"], "label": p["title"]} for p in results["projects"] if p["in_evaluation"]
    ]
    scatter = charts.scatter_chart(scatter_pts, "AI score against write-up length")
    adj = length["ai_adjusted_auc"]
    if adj and adj[1] is not None and adj[1] > 0.5:
        adj_text = "Even after removing what length explains, what's left of the AI score still separates winners from non-winners."
    elif adj and adj[2] is not None and adj[2] < 0.5:
        adj_text = "Once length is removed, what's left of the AI score points the wrong way."
    else:
        adj_text = "Once length is accounted for, there's no clear signal left at this sample size, so much of what the AI picks up may be length."
    corr = length["ai_length_corr"]
    corr_text = "n/a" if corr is None else f"{corr:.2f}"

    comp_rows = ""
    for c in results["comparisons"]:
        diff = "n/a" if c["diff"] is None else "{:+.0f} pts".format(c["diff"] * 100)
        if c["lo"] is not None:
            diff += ' <span class="muted">({:+.0f} to {:+.0f})</span>'.format(c["lo"] * 100, c["hi"] * 100)
        comp_rows += (
            f"<tr><td>{e(c['question'])}</td><td class=\"num\">{diff}</td>"
            f"<td class=\"num\">{pct(c['p_a_better'])}</td><td>{_verdict_row(c)}</td></tr>"
        )

    def fmt2(v: float | None) -> str:
        return "n/a" if v is None else f"{v:.2f}"

    pw_html = ""
    pw = results.get("pairwise")
    if pw:
        rubric_corr = fmt2(pw["rubric_corr"])
        pw_html = f"""
<h2>Pairwise judging</h2>
<p class="prose">Each pair was shown twice, once in each order. A fair judge picks the first-shown project about half the time,
and gives the same answer whichever order it sees.</p>
<div class="table-wrap"><table><tbody>
<tr><td>Judgements made</td><td class="num">{pw['judgements']:,}</td></tr>
<tr><td>Chose whichever project was shown first</td><td class="num">{ci_text(pw['first_position_rate'])}</td></tr>
<tr><td>Same answer in both orders</td><td class="num">{ci_text(pw['order_consistency'])}</td></tr>
<tr><td>Rank agreement with rubric scores</td><td class="num">{rubric_corr}</td></tr>
</tbody></table></div>"""

    ch_rows = ""
    for c in results["challenges"]:
        winners = "; ".join(f"{w['title']} ({w['fit']})" for w in c["winners"]) or "no winner matched"
        extra = ""
        if "opted_in" in c:
            extra = f"<br><span class=\"muted small\">{c['opted_in']} opted in · {len(c['opted_in_no_fit'])} opted in with no clear fit · {len(c['strong_not_opted'])} strong fit but didn't opt in</span>"
        ch_rows += (
            f"<tr><td><strong>{e(c['name'])}</strong>{(' <span class=muted>(' + e(c['sponsor']) + ')</span>') if c['sponsor'] else ''}{extra}</td>"
            f'<td class="num">{c["strong"]} / {c["assessed"]}</td><td class="num">{c["possible"]}</td><td>{e(winners)}</td></tr>'
        )
    ch_html = (
        f"""<h2>Sponsor challenges</h2>
<p class="prose">For each challenge, how many projects the AI rated a strong fit, and how it rated the projects that actually won that prize.
A useful matcher rates the real winner as a strong or possible fit while keeping the strong list short.</p>
<div class="table-wrap"><table><thead><tr><th>Challenge</th><th class="num">Rated strong</th><th class="num">Possible</th><th>Actual winner (AI's fit)</th></tr></thead>
<tbody>{ch_rows}</tbody></table></div>"""
        if results["challenges"]
        else ""
    )

    ev_rows = ""
    for r in results["per_event"]:
        status = r["excluded"] or ("held out" if r.get("holdout") else "")
        ev_rows += (
            f"<tr><td>{e(r['event'])}</td><td class=\"num\">{r['n']}</td><td class=\"num\">{r['winners']}</td>"
            f"<td class=\"num\">{pct(r.get('ai_auc'))}</td><td class=\"num\">{pct(r.get('length_auc'))}</td>"
            f"<td class=\"muted small\">{e(status)}</td></tr>"
        )
    hold_html = ""
    h = results.get("holdout")
    if h and h.get("events"):
        hold_html = (
            f'<div class="card"><strong>Held-out events ({e(", ".join(h["events"]))})</strong>'
            f"<p>AI: {ci_text(h['ai_auc'])} · write-up length: {ci_text(h['length_auc'])}"
            + (f" · AI on the other events: {ci_text(h['other_ai_auc'])}" if h.get("other_ai_auc") else "")
            + "</p><p class=\"muted small\">If the AI does much better on older events than on held-out ones, "
            "it may be remembering pages from its training data rather than judging.</p></div>"
        )
    elif h and h.get("missing"):
        hold_html = f'<p class="muted small">Held-out event(s) not found among usable events: {e(", ".join(h["missing"]))}.</p>'

    sig_rows = "".join(
        f"<tr><td>{e(s['signal'])}</td><td class=\"num\">{s['weight']:+.2f}</td></tr>" for s in results.get("signal_weights", [])
    )
    usage = results.get("usage")
    usage_html = ""
    if usage:
        t = usage["total"]
        cost = usage.get("cost_usd")
        cost_line = (
            f"<tr><td>Estimated cost</td><td class=\"num\">${cost:.2f} (${usage['cost_per_project_usd']:.3f} per project)</td></tr>"
            if cost is not None
            else "<tr><td>Estimated cost</td><td class=\"num muted\">add prices to the config to see this</td></tr>"
        )
        usage_html = f"""<h2>Cost of this run</h2>
<div class="table-wrap"><table><tbody>
<tr><td>Model calls</td><td class="num">{t['calls']:,}</td></tr>
<tr><td>Answered from cache</td><td class="num">{t['cache_hits']:,}</td></tr>
<tr><td>Unusable replies or errors (retried)</td><td class="num">{t['failures']:,}</td></tr>
<tr><td>Tokens in / out</td><td class="num">{t['input_tokens']:,} / {t['output_tokens']:,}</td></tr>
{cost_line}
</tbody></table></div>
<p class="muted small">Cached calls cost nothing, so re-running gives identical results for free.</p>"""

    caveats = "".join(f"<li>{e(c)}</li>" for c in results["caveats"])
    if m.get("mock"):
        model_line = "mock provider (no AI model)"
    else:
        model_line = f"{m['provider']} · {m['model']}" if m.get("provider") else ""
    test_retest = fmt2(cons["test_retest"])
    mean_sd = fmt2(cons["mean_sd"])
    signals_html = ""
    if sig_rows:
        signals_html = (
            "<h2>Which simple signals went with winning?</h2>"
            '<p class="prose muted small">Standardised weights from one logistic regression on all projects. '
            "Positive means more of it went with winning. With this few winners, read the signs, not the sizes.</p>"
            f'<div class="table-wrap"><table><tbody>{sig_rows}</tbody></table></div>'
        )
    banner = (demo_banner() if m.get("demo") else "") + (mock_banner() if m["mock"] else "")

    body = f"""
<p class="eyebrow">{e(m['event_name'])} · evaluation</p>
<h1>How well does the AI's judgement match the real winners?</h1>
<p class="meta">{m['n_projects']} projects across {len(m['events'])} event(s) · {m['n_winners']} winners · a win means {e(m['win_label'])}
· {e(model_line)} · {m['samples']} sample(s) per project · generated {e(m['generated'])}</p>
{banner}
<div class="tiles">{''.join(tiles)}</div>
<p class="lede">{e(head['sentence'])}</p>
<p class="prose">{e(head['verdict_text'])}</p>

<h2>How each method compares</h2>
<p class="prose">Each method ranks the projects. The score is how often a real winner comes out above a non-winner from the
same hackathon: 50% is a coin toss, 100% means every winner beat every non-winner. Bars show 95% bootstrap intervals.</p>
<figure>{forest}<figcaption>AI-based methods in colour, baselines in grey.</figcaption></figure>
<div class="table-wrap"><table><thead><tr><th>Method</th><th class="num">Winner above non-winner</th>
<th class="num">Winners in top 25%</th><th class="num">Winners in top 50%</th><th class="num">Top picks that won</th></tr></thead>
<tbody>{method_rows}</tbody></table></div>
<p class="muted small">"Top picks that won": if each event's top <em>w</em> projects were picked, where <em>w</em> is its number of winners, the share that actually won.</p>

<h2>Shortlisting</h2>
<p class="prose">{e(shortlist_text)} The further a line sits above the diagonal, the fewer projects judges would need to
look at closely to keep the winners in view.</p>
<figure><ul class="legend">{legend}</ul>{curve}
<details><summary>Show the numbers</summary><div class="table-wrap"><table><thead><tr><th>Method</th>{curve_head}</tr></thead>
<tbody>{curve_rows}</tbody></table></div></details></figure>

<h2>Is it just rewarding long write-ups?</h2>
<p class="prose">AI judges are known to favour longer answers. Within each event, the AI's scores and write-up length have a rank
correlation of <strong>{corr_text}</strong>. Ranking by length alone gives {ci_text(length['length_auc'])}.
After removing what length explains from the AI score, the remainder gives {ci_text(adj)}. {e(adj_text)}</p>
<figure><ul class="legend"><li><span class="key dot k1"></span>Won a prize</li><li><span class="key dot kmuted"></span>Didn't win</li></ul>{scatter}</figure>

<h2>Head-to-head</h2>
<p class="prose">The same bootstrap resamples are used for both methods in each row, so the interval is for the difference itself.</p>
<div class="table-wrap"><table><thead><tr><th>Question</th><th class="num">Difference</th><th class="num">Resamples where it's better</th><th>Answer</th></tr></thead>
<tbody>{comp_rows}</tbody></table></div>

<h2>Consistency</h2>
<div class="table-wrap"><table><tbody>
<tr><td>Independent samples per project</td><td class="num">{cons['samples']}</td></tr>
<tr><td>Rank agreement between two runs</td><td class="num">{test_retest}</td></tr>
<tr><td>Average spread between samples (1 to 5 scale)</td><td class="num">{mean_sd}</td></tr>
<tr><td>Projects with unstable scores</td><td class="num">{cons['unstable']}</td></tr>
<tr><td>Projects whose write-up was too thin to judge</td><td class="num">{cons['thin_evidence']}</td></tr>
<tr><td>Winners the tool flagged for a human look</td><td class="num">{cons['winners_flagged']} of {cons['winners_total']}</td></tr>
</tbody></table></div>
{pw_html}
{ch_html}

<h2>By event</h2>
<div class="table-wrap"><table><thead><tr><th>Event</th><th class="num">Projects</th><th class="num">Winners</th>
<th class="num">AI</th><th class="num">Length</th><th>Note</th></tr></thead><tbody>{ev_rows}</tbody></table></div>
{hold_html}

{signals_html}

{usage_html}

<h2>Caveats</h2>
<ul class="prose">{caveats}</ul>

<h2>How the numbers are calculated</h2>
<div class="prose small muted">
<p><strong>Winner above non-winner</strong> is a within-event AUC: every pair of one winner and one non-winner from the same event is
compared, ties count half. Comparing only within an event matters because winners are chosen relative to the other projects in the room.</p>
<p><strong>Intervals</strong> come from {m['bootstrap']:,} bootstrap resamples, drawing projects with replacement inside each event.</p>
<p><strong>Baselines</strong> never see the label of the project they score: the learned baselines use {e(m['baseline_scheme'])} cross-validation.</p>
<p><strong>Length adjustment</strong> regresses the AI score on log word count within events and evaluates the residual.</p>
<p><strong>Leakage</strong>: prize opt-ins and outcomes are never shown to the model, and sentences where a team reports its own result are removed.</p>
</div>
"""
    return page(f"{m['event_name']}: evaluation", body)


# --------------------------------------------------------------------------- judges' scores report


def _pairs_cell(d: dict | None) -> str:
    if not d or not d.get("value") or d["value"][0] is None:
        return "n/a"
    return f"{ci_text(d['value'])} <span class=\"muted small\">· {d['pairs']} pairs</span>"


def render_judging(r: dict[str, Any]) -> str:
    m = r["meta"]
    head = r["headline"]
    methods = r["methods"]
    by_key = {x["key"]: x for x in methods}
    fin = r["finalists"]
    notes = r["data_notes"]
    cols = ", ".join(m["compared_columns"])
    banner = (demo_banner() if m.get("demo") else "") + (mock_banner() if m["mock"] else "")
    model_line = "mock provider (no AI model)" if m["mock"] else f"{m['provider']} · {m['model']}"

    ai_v = by_key["ai"]["value"]
    tiles = [
        f'<div class="tile hero"><div class="label">Same order as the round 1 judges</div>'
        f'<div class="value">{pct(ai_v[0])}</div><div class="sub">95% interval {pct(ai_v[1])} to {pct(ai_v[2])} · '
        f'chance is 50% · {by_key["ai"]["pairs"]} pairs</div></div>',
        f'<div class="tile"><div class="label">Write-up length baseline</div>'
        f'<div class="value">{pct(by_key["length"]["value"][0])}</div><div class="sub">the number to beat</div></div>',
    ]
    if fin.get("n"):
        tiles.append(
            f'<div class="tile"><div class="label">Two sets of judges agreed</div>'
            f'<div class="value">{pct(fin["judges"]["value"][0])}</div>'
            f'<div class="sub">finalists only · {fin["judges"]["pairs"]} pairs</div></div>'
        )

    rows = [{"label": x["label"], "point": x["value"][0], "lo": x["value"][1], "hi": x["value"][2], "emph": x["is_ai"]}
            for x in methods]
    forest = charts.forest_chart(rows, "How often each method orders two teams the way the judges did",
                                 axis_title="Pairs put in the same order as the round 1 judges (50% = chance)")
    method_rows = "".join(
        f"<tr><td><strong>{e(x['label'])}</strong></td><td class=\"num\">{_pairs_cell(x)}</td></tr>" for x in methods
    )
    vs = r["ai_vs_length"]
    if vs.get("lo") is None:
        vs_text = "There isn't enough data to compare the AI with the length baseline."
    elif vs["lo"] > 0:
        vs_text = "The AI beat write-up length, and the interval for the difference excludes zero."
    elif vs["hi"] < 0:
        vs_text = "The AI did worse than simply ranking by write-up length."
    else:
        vs_text = ("With this much data, the AI and write-up length can't be told apart: the interval for their "
                   "difference includes zero.")
    diff = "n/a" if vs.get("diff") is None else "{:+.0f} points ({:+.0f} to {:+.0f})".format(
        vs["diff"] * 100, (vs["lo"] or 0) * 100, (vs["hi"] or 0) * 100)

    crit_rows = "".join(
        f"<tr><td>{e(c['criterion'])} <span class=\"muted small\">(out of {c['max']:g} on the sheet)</span></td>"
        f"<td class=\"num\">{_pairs_cell(c)}</td></tr>"
        for c in r["per_criterion"]
    )

    fin_html = ""
    if fin.get("n"):
        fin_html = f"""
<h2>The fair bar: how much do judges agree with each other?</h2>
<p class="prose">Finalists were scored twice, by different judges in round 1 and round 2. Their agreement shows how
consistent human judging is, which is the most anyone can expect of the AI. Only pairs of finalists in the same challenge count.</p>
<div class="table-wrap"><table><tbody>
<tr><td><strong>Round 1 judges vs round 2 judges</strong></td><td class="num">{_pairs_cell(fin['judges'])}</td></tr>
<tr><td>AI vs round 2 judges</td><td class="num">{_pairs_cell(fin['ai_round2'])}</td></tr>
<tr><td>AI vs round 1 judges</td><td class="num">{_pairs_cell(fin['ai_round1'])}</td></tr>
<tr><td>Write-up length vs round 2 judges</td><td class="num">{_pairs_cell(fin['length_round2'])}</td></tr>
</tbody></table></div>
<p class="muted small">{fin['n']} finalist entries. Finalists are all fairly strong, so they're harder to tell apart than
the field as a whole: expect lower agreement here, for judges and AI alike.</p>"""

    reach = r.get("reach_final")
    reach_html = ""
    if reach:
        pw_row = (f"<tr><td>AI pairwise comparisons</td><td class=\"num\">{ci_text(reach['pairwise'])}</td></tr>"
                  if reach.get("pairwise") else "")
        reach_html = f"""
<h2>Picking the finalists</h2>
<p class="prose">Within each challenge, how often a team that reached round 2 was ranked above one that didn't
({reach['finalists']} of {reach['entries']} entries reached the final).</p>
<div class="table-wrap"><table><tbody>
<tr><td><strong>AI rubric score</strong></td><td class="num">{ci_text(reach['ai'])}</td></tr>{pw_row}
<tr><td>Write-up length</td><td class="num">{ci_text(reach['length'])}</td></tr>
<tr><td>The judges' own round 1 totals</td><td class="num">{ci_text(reach['judges_round1_total'])}</td></tr>
</tbody></table></div>
<p class="muted small">The last row shows how closely finalists followed the round 1 scores, for reference.</p>"""

    ch_rows = "".join(
        f"<tr><td>{e(c['challenge'])}</td><td class=\"num\">{c['entries']}</td><td class=\"num\">{c['finalists']}</td>"
        f"<td class=\"num\">{c['pairs']}</td><td class=\"num\">{pct(c['ai'])}</td><td class=\"num\">{pct(c['length'])}</td></tr>"
        for c in r["per_challenge"]
    )

    detail_rows = "".join(
        f"<tr><td>{e(x['challenge'])}</td><td class=\"num\">{x['team']}</td><td>{e(x['title'])}"
        f"{' <span class=won>finalist</span>' if x['finalist'] else ''}"
        f"{'' if x['opted_in'] else ' <span class=flag>not opted in</span>'}</td>"
        f"<td class=\"num\">{'' if x['judges_round1'] is None else format(x['judges_round1'], 'g')}</td>"
        f"<td class=\"num\">{'' if x['judges_round2'] is None else format(x['judges_round2'], 'g')}</td>"
        f"<td class=\"num\">{'' if x['ai'] is None else format(x['ai'], '.2f')}</td>"
        f"<td class=\"num\">{x['words']:,}</td></tr>"
        for x in sorted(r["rows"], key=lambda x: (x["challenge"], -(x["judges_round1"] or 0)))
    )

    def note_list(title: str, items: list[str]) -> str:
        if not items:
            return ""
        return f"<h3>{e(title)}</h3><ul class=\"small\">" + "".join(f"<li>{e(i)}</li>" for i in items) + "</ul>"

    notes_html = "".join([
        note_list("Sheet rows that couldn't be matched to a project", notes["unmatched"]),
        note_list("Totals that don't match the marks", notes["total_mismatches"]),
        note_list("Judged for a challenge the team didn't opt into on Devpost", notes["not_opted_in"]),
        note_list("Opted into a challenge but no sheet was found", notes["opted_but_unjudged"]),
        note_list("Challenges with no sheets at all", notes["challenges_without_sheets"]),
        note_list("Other warnings", notes["warnings"]),
    ]) or '<p class="muted">No problems found in the sheets.</p>'
    caveats = "".join(f"<li>{e(c)}</li>" for c in r["caveats"])
    lo = m["left_out_columns"]
    left_out = "nothing" if not lo else lo[0] if len(lo) == 1 else ", ".join(lo[:-1]) + " and " + lo[-1]

    body = f"""
<p class="eyebrow">{e(m['event'])} · judges' scores</p>
<h1>Does the AI order teams the way the judges did?</h1>
<p class="meta">{m['sheets']} sheet rows · {m['projects']} projects · {m['challenges']} challenges · {m['finalists']} finalist entries
· {e(model_line)} · {m['samples']} sample(s) per project</p>
{banner}
<div class="tiles">{''.join(tiles)}</div>
<p class="lede">{e(head['sentence'])}</p>
<p class="prose">{e(head['ceiling'])}</p>

<h2>How the comparison works</h2>
<p class="prose">Each sponsor challenge had its own judges, so teams are only compared with others in the same challenge.
For every pair of teams in a challenge, the question is whether the method puts them in the same order as the judges.
Only the marks a write-up can show are used ({e(cols)}, out of {m['compared_max']:g}); {e(left_out)} can only be seen in
person, so they're left out. The AI's criterion scores are combined using the sheet's own weights.</p>

<h2>Round 1: every team</h2>
<figure>{forest}<figcaption>AI-based methods in colour, baseline in grey. Bars are 95% bootstrap intervals.</figcaption></figure>
<div class="table-wrap"><table><thead><tr><th>Method</th><th class="num">Same order as the judges</th></tr></thead>
<tbody>{method_rows}
<tr><td>{e(r['full_sheet']['label'])}</td><td class="num">{_pairs_cell(r['full_sheet'])}</td></tr>
</tbody></table></div>
<p class="prose"><strong>AI minus write-up length:</strong> {diff}. {e(vs_text)}</p>

<h2>Criterion by criterion</h2>
<p class="prose">Each AI criterion compared with the matching mark on the sheet.</p>
<div class="table-wrap"><table><thead><tr><th>Criterion</th><th class="num">Same order as the judges</th></tr></thead>
<tbody>{crit_rows}</tbody></table></div>
{fin_html}
{reach_html}

<h2>By challenge</h2>
<div class="table-wrap"><table><thead><tr><th>Challenge</th><th class="num">Entries</th><th class="num">Finalists</th>
<th class="num">Pairs</th><th class="num">AI</th><th class="num">Length</th></tr></thead><tbody>{ch_rows}</tbody></table></div>

<details><summary>Every judged entry</summary>
<div class="table-wrap"><table><thead><tr><th>Challenge</th><th class="num">Team</th><th>Project</th>
<th class="num">Judges R1</th><th class="num">Judges R2</th><th class="num">AI</th><th class="num">Words</th></tr></thead>
<tbody>{detail_rows}</tbody></table></div>
<p class="muted small">Judges' columns are the comparable marks only (out of {m['compared_max']:g}). AI is its weighted rubric score (1 to 5).</p>
</details>

<h2>Data checks</h2>
{notes_html}

<h2>Caveats</h2>
<ul class="prose">{caveats}</ul>

<h2>How the numbers are calculated</h2>
<div class="prose small muted">
<p><strong>Same order</strong> is a concordance: for each pair of entries in the same challenge whose judges' scores differ,
does the method rank them the same way? Ties in the method's scores count half.</p>
<p><strong>Intervals</strong> come from {m['bootstrap']:,} bootstrap resamples, drawing entries with replacement within each challenge.</p>
<p><strong>The AI never saw the sheets.</strong> It scored each write-up on its own, with prize opt-ins and results hidden; the sheets were only used afterwards, for the comparison.</p>
</div>
"""
    return page(f"{m['event']}: AI vs judges", body)
