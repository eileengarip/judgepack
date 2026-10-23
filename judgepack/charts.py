"""Small inline-SVG charts for the evaluation report.

Colours come from CSS variables, so the charts follow light and dark themes. Every mark
has a hover tooltip (an SVG <title> on a generous hit area), and every chart is paired with
a table in the report, so nothing depends on colour or hovering alone.
"""

from __future__ import annotations

import math
from html import escape


def _pct(v: float) -> str:
    return f"{v * 100:.0f}%"


def _e(text: object) -> str:
    return escape(str(text), quote=True)


def recall_chart(series: list[dict], title: str) -> str:
    """Shortlist curves. series: [{"label", "slot", "points": [[x, y], ...]}] with x, y in [0, 1]."""
    W, H = 900, 420
    left, right, top, bottom = 60, 20, 14, 52
    pw, ph = W - left - right, H - top - bottom

    def X(v: float) -> float:
        return left + v * pw

    def Y(v: float) -> float:
        return top + (1 - v) * ph

    out = [f'<svg class="chart" viewBox="0 0 {W} {H}" role="img" aria-label="{_e(title)}">']
    for v in (0, 0.25, 0.5, 0.75, 1):
        out.append(f'<line class="grid" x1="{left}" x2="{W - right}" y1="{Y(v):.1f}" y2="{Y(v):.1f}"/>')
        out.append(f'<text class="tick" x="{left - 8}" y="{Y(v) + 4:.1f}" text-anchor="end">{_pct(v)}</text>')
        out.append(f'<text class="tick" x="{X(v):.1f}" y="{top + ph + 20}" text-anchor="middle">{_pct(v)}</text>')
    out.append(f'<line class="axis" x1="{left}" x2="{W - right}" y1="{Y(0):.1f}" y2="{Y(0):.1f}"/>')
    out.append(f'<line class="ref" x1="{X(0):.1f}" y1="{Y(0):.1f}" x2="{X(1):.1f}" y2="{Y(1):.1f}"/>')
    out.append(f'<text class="ref-label" x="{X(0.7):.1f}" y="{Y(0.62):.1f}">Random order</text>')
    out.append(
        f'<text class="axis-title" x="{left + pw / 2:.1f}" y="{H - 8}" text-anchor="middle">'
        "Shortlist size: share of each event's projects, best-ranked first</text>"
    )
    out.append(
        f'<text class="axis-title" transform="translate(16 {top + ph / 2:.1f}) rotate(-90)" '
        'text-anchor="middle">Winners on the shortlist</text>'
    )
    for s in series:
        pts = [(x, y) for x, y in s["points"] if y is not None]
        if not pts:
            continue
        d = "M" + " L".join(f"{X(x):.1f},{Y(y):.1f}" for x, y in pts)
        out.append(f'<path class="line s{s["slot"]}" d="{d}"/>')
    for s in series:
        for x, y in s["points"]:
            if y is None:
                continue
            tip = f"{s['label']}: a shortlist of the top {_pct(x)} holds {_pct(y)} of winners"
            out.append(f'<circle class="hit" cx="{X(x):.1f}" cy="{Y(y):.1f}" r="10"><title>{_e(tip)}</title></circle>')
    out.append("</svg>")
    return "".join(out)


def forest_chart(rows: list[dict], title: str,
                 axis_title: str = "Winner ranked above a non-winner from the same event (50% = chance)") -> str:
    """Point estimates with intervals. rows: [{"label", "point", "lo", "hi", "emph"}], values in [0, 1]."""
    rows = [r for r in rows if r.get("point") is not None]
    W = 900
    left, right, top, bottom = 230, 24, 10, 46
    row_h = 34
    H = top + bottom + row_h * max(1, len(rows))
    lows = [r["lo"] for r in rows if r.get("lo") is not None] + [r["point"] for r in rows]
    d_lo = min(0.3, math.floor((min(lows) - 0.05) * 10) / 10) if lows else 0.3
    d_lo = max(0.0, d_lo)
    d_hi = 1.0
    pw = W - left - right

    def X(v: float) -> float:
        return left + (v - d_lo) / (d_hi - d_lo) * pw

    out = [f'<svg class="chart" viewBox="0 0 {W} {H}" role="img" aria-label="{_e(title)}">']
    tick = d_lo
    while tick <= d_hi + 1e-9:
        x = X(tick)
        out.append(f'<line class="grid" x1="{x:.1f}" x2="{x:.1f}" y1="{top}" y2="{H - bottom}"/>')
        out.append(f'<text class="tick" x="{x:.1f}" y="{H - bottom + 18}" text-anchor="middle">{_pct(tick)}</text>')
        tick = round(tick + 0.1, 10)
    out.append(f'<line class="ref-strong" x1="{X(0.5):.1f}" x2="{X(0.5):.1f}" y1="{top}" y2="{H - bottom}"/>')
    out.append(
        f'<text class="axis-title" x="{left + pw / 2:.1f}" y="{H - 8}" text-anchor="middle">{_e(axis_title)}</text>'
    )
    for i, r in enumerate(rows):
        cy = top + row_h * i + row_h / 2
        cls = "emph" if r.get("emph") else "base"
        out.append(f'<text class="row-label {cls}" x="{left - 14}" y="{cy + 4:.1f}" text-anchor="end">{_e(r["label"])}</text>')
        if r.get("lo") is not None and r.get("hi") is not None:
            out.append(f'<line class="ci {cls}" x1="{X(r["lo"]):.1f}" x2="{X(r["hi"]):.1f}" y1="{cy:.1f}" y2="{cy:.1f}"/>')
            interval = f" (95% interval {_pct(r['lo'])} to {_pct(r['hi'])})"
        else:
            interval = ""
        out.append(f'<circle class="dot {cls}" cx="{X(r["point"]):.1f}" cy="{cy:.1f}" r="5.5"/>')
        tip = f"{r['label']}: {_pct(r['point'])}{interval}"
        out.append(
            f'<rect class="hit" x="{left - 200}" y="{cy - row_h / 2:.1f}" width="{W - left + 200 - right}" '
            f'height="{row_h}"><title>{_e(tip)}</title></rect>'
        )
    out.append("</svg>")
    return "".join(out)


def scatter_chart(points: list[dict], title: str) -> str:
    """Write-up length (log scale) against AI score. points: [{"x", "y", "won", "label"}]."""
    pts = [p for p in points if p.get("x") and p.get("y") is not None]
    W, H = 900, 400
    left, right, top, bottom = 56, 20, 14, 52
    pw, ph = W - left - right, H - top - bottom
    if not pts:
        return ""
    xs = [max(1.0, float(p["x"])) for p in pts]
    lx_lo, lx_hi = math.log10(min(xs) * 0.85), math.log10(max(xs) * 1.15)
    if lx_hi - lx_lo < 0.3:
        lx_lo, lx_hi = lx_lo - 0.15, lx_hi + 0.15

    def X(v: float) -> float:
        return left + (math.log10(max(1.0, v)) - lx_lo) / (lx_hi - lx_lo) * pw

    def Y(v: float) -> float:
        return top + (5 - v) / 4 * ph

    out = [f'<svg class="chart" viewBox="0 0 {W} {H}" role="img" aria-label="{_e(title)}">']
    for v in (1, 2, 3, 4, 5):
        out.append(f'<line class="grid" x1="{left}" x2="{W - right}" y1="{Y(v):.1f}" y2="{Y(v):.1f}"/>')
        out.append(f'<text class="tick" x="{left - 8}" y="{Y(v) + 4:.1f}" text-anchor="end">{v}</text>')
    for t in (25, 50, 100, 200, 400, 800, 1600, 3200):
        if lx_lo <= math.log10(t) <= lx_hi:
            out.append(f'<text class="tick" x="{X(t):.1f}" y="{top + ph + 20}" text-anchor="middle">{t:,}</text>')
    out.append(
        f'<text class="axis-title" x="{left + pw / 2:.1f}" y="{H - 8}" text-anchor="middle">'
        "Write-up length in words (log scale)</text>"
    )
    out.append(
        f'<text class="axis-title" transform="translate(16 {top + ph / 2:.1f}) rotate(-90)" '
        'text-anchor="middle">AI rubric score (1 to 5)</text>'
    )
    for won in (False, True):  # winners drawn last, on top
        for p in pts:
            if bool(p["won"]) != won:
                continue
            cls = "dot-win" if won else "dot-other"
            out.append(f'<circle class="{cls}" cx="{X(p["x"]):.1f}" cy="{Y(p["y"]):.1f}" r="5"/>')
    for p in pts:
        tip = f"{p['label']}: {int(p['x']):,} words, AI score {p['y']:.2f}{' (won)' if p['won'] else ''}"
        out.append(f'<circle class="hit" cx="{X(p["x"]):.1f}" cy="{Y(p["y"]):.1f}" r="10"><title>{_e(tip)}</title></circle>')
    out.append("</svg>")
    return "".join(out)
