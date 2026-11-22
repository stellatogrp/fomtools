"""The report card of a code-generation run, as a self-contained SVG.

`report_svg` draws what a finished run measured: the time per solve of the NumPy
reference, the translated and the optimized solver (log scale), the rounds of the
code optimization phase and why each was kept or reverted, the profile of the
shipped solver, and the acceptance check against the tolerance. Each part is
left out when its data is missing. `write_report_svg` writes the card as
``report.svg`` next to the generated solver.

Standard library only (string templating), so the picture needs no plotting
package and is the same on every machine: the same input gives the same bytes.
All text is escaped. The card has its own background and border, and follows
``prefers-color-scheme`` with ``theme="auto"``; ``"light"`` and ``"dark"`` fix it.
"""

from __future__ import annotations

import html
import math
import re
import unicodedata
from pathlib import Path

from fomtools.codegen.progress import format_time

REPORT_SVG_NAME = "report.svg"

_W = 760  # card width; the height follows the content
_PAD = 28
_X0, _X1 = 168, 468  # the plots of the time and round sections
_VX = 488  # the value column
_RIGHT = _W - _PAD

_SANS = "system-ui, -apple-system, 'Segoe UI', Helvetica, Arial, sans-serif"
_MONO = "ui-monospace, SFMono-Regular, Menlo, Consolas, 'Liberation Mono', monospace"

# Colors by role, (light, dark). The three series (every pair of them) and the
# five profile colors (neighbors in the bar) keep a color difference of at least
# 8 (OKLab x 100) under simulated protanopia, deuteranopia and tritanopia on both
# surfaces, and every mark also has a label. Text never takes a series color.
_COLORS = {
    "bg": ("#fcfcfb", "#1a1a19"),
    "border": ("#d9d8d0", "#3a3a37"),
    "ink": ("#0b0b0b", "#f0efec"),
    "ink2": ("#52514e", "#c3c2b7"),
    "muted": ("#7b7974", "#98968f"),
    "grid": ("#e6e5df", "#2c2c2a"),
    "axis": ("#c3c2b7", "#4a4a46"),
    "track": ("#eeede8", "#2a2a28"),
    "numpy": ("#4788cc", "#4f8fd6"),
    "translated": ("#de7b48", "#d9773f"),
    "optimized": ("#35ab7c", "#3aa585"),
    "p4": ("#dba41c", "#b8891a"),
    "p5": ("#d98bb0", "#cf6f99"),
    "other": ("#a9a7a0", "#6b6a65"),
    "bad": ("#d03b3b", "#e66767"),
}
_PROFILE_COLORS = ("numpy", "translated", "optimized", "p4", "p5")
_PROFILE_TOP = 5

# Characters XML 1.0 cannot carry, and whitespace runs.
_XML_INVALID = re.compile("[^\t\n\r\x20-퟿-�\U00010000-\U0010ffff]")
_SPACES = re.compile(r"\s+")


# -- text and numbers --------------------------------------------------------
def _clean(text) -> str:
    """``text`` on one line, without the characters XML cannot carry."""
    return _SPACES.sub(" ", _XML_INVALID.sub("?", str(text))).strip()


def _esc(text) -> str:
    return html.escape(_clean(text), quote=True)


def _n(value: float) -> str:
    """A coordinate: two decimals at most, no trailing zeros."""
    text = f"{value:.2f}".rstrip("0").rstrip(".")
    return "0" if text in ("-0", "") else text


def _char_w(ch: str, mono: bool) -> float:
    """A cautious (too wide rather than too narrow) width of ``ch``, in em."""
    if unicodedata.east_asian_width(ch) in "WF" or ord(ch) > 0x2E7F:
        return 1.1
    if mono:
        return 0.62
    if ch in "iljtfrI.,:;'|!()[] ":
        return 0.36
    if ch in "mwMW@%":
        return 0.95
    return 0.72 if ch.isupper() or ch.isdigit() else 0.6


def _width(text: str, size: float, mono: bool = False) -> float:
    return size * sum(_char_w(ch, mono) for ch in text)


def _fit(text, size: float, max_width: float, mono: bool = False) -> str:
    """``text`` cut with an ellipsis so that it fits ``max_width`` pixels."""
    text = _clean(text)
    if _width(text, size, mono) <= max_width:
        return text
    while text and _width(text + "…", size, mono) > max_width:
        text = text[:-1]
    return text.rstrip() + "…"


def _sci(value: float) -> str:
    """``2.3e-13``, ``1e-6``: two significant digits, a plain exponent."""
    if not value:
        return "0"
    if not math.isfinite(value):
        return "∞"
    mantissa, exponent = f"{value:.1e}".split("e")
    mantissa = mantissa[:-2] if mantissa.endswith(".0") else mantissa
    return f"{mantissa}e{int(exponent)}"


def _times(ratio: float) -> str:
    """A speedup factor: ``1.07×``, ``12.3×``, ``758×``, ``1.2e9×``."""
    if ratio >= 1.0e6 or ratio < 0.01:
        return f"{_sci(ratio)}×"
    if ratio >= 99.95:
        return f"{ratio:,.0f}×"
    return f"{ratio:.1f}×" if ratio >= 9.995 else f"{ratio:.2f}×"


def _positive(value) -> float | None:
    """``value`` as a positive finite float, else ``None``."""
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) and value > 0 else None


# -- drawing -----------------------------------------------------------------
def _text(x, y, text, size=12, fill="ink", anchor=None, mono=False, weight=None) -> str:
    attrs = f'x="{_n(x)}" y="{_n(y)}" font-size="{_n(size)}" class="fc-{fill}"'
    if anchor:
        attrs += f' text-anchor="{anchor}"'
    if mono:
        attrs += f' font-family="{_MONO}"'
    if weight:
        attrs += f' font-weight="{weight}"'
    return f"<text {attrs}>{_esc(text)}</text>"


def _line(x1, y1, x2, y2, stroke="grid", width=1.0) -> str:
    return (
        f'<line x1="{_n(x1)}" y1="{_n(y1)}" x2="{_n(x2)}" y2="{_n(y2)}" '
        f'class="fc-s-{stroke}" stroke-width="{_n(width)}"/>'
    )


def _bar(x, y, width, height, fill, tip) -> str:
    """A bar from ``x``: square at the baseline, rounded at the data end."""
    r = min(3.0, height / 2, width)
    d = (
        f"M{_n(x)} {_n(y)}h{_n(width - r)}a{_n(r)} {_n(r)} 0 0 1 {_n(r)} {_n(r)}"
        f"v{_n(height - 2 * r)}a{_n(r)} {_n(r)} 0 0 1 {_n(-r)} {_n(r)}h{_n(r - width)}z"
    )
    return (
        f'<path class="fc-{fill} fc-bar" data-length="{_n(width)}" d="{d}">'
        f"<title>{_esc(tip)}</title></path>"
    )


def _cross(x, y, r, stroke, width=2.0) -> str:
    d = f"M{_n(x - r)} {_n(y - r)}l{_n(2 * r)} {_n(2 * r)}M{_n(x - r)} {_n(y + r)}l{_n(2 * r)} {_n(-2 * r)}"
    return (
        f'<path d="{d}" fill="none" class="fc-s-{stroke}" stroke-width="{_n(width)}" '
        'stroke-linecap="round"/>'
    )


def _check(x, y, stroke="optimized") -> str:
    """A check mark in a 12 px box whose left middle is ``(x, y)``."""
    d = f"M{_n(x + 1.5)} {_n(y + 0.5)}l{_n(3)} {_n(3)}l{_n(6)} {_n(-7)}"
    return (
        f'<path d="{d}" fill="none" class="fc-s-{stroke}" stroke-width="2" '
        'stroke-linecap="round" stroke-linejoin="round"/>'
    )


def _mark(kind: str, x, y) -> str:
    """The mark of a round: a different shape for each outcome, not only a color."""
    if kind == "translated":
        return (
            f'<circle class="fc-translated fc-mark-translated" cx="{_n(x)}" cy="{_n(y)}" r="4.5"/>'
        )
    if kind == "kept":
        return f'<circle class="fc-optimized fc-mark-kept" cx="{_n(x)}" cy="{_n(y)}" r="4.5"/>'
    if kind == "failed":
        return f'<g class="fc-mark-failed">{_cross(x, y, 3.5, "bad")}</g>'
    if kind == "unchanged":
        return f'<g class="fc-mark-unchanged">{_line(x - 4, y, x + 4, y, "muted", 2)}</g>'
    return (  # reverted: a hollow ring
        f'<circle class="fc-bg fc-s-ink2 fc-mark-reverted" cx="{_n(x)}" cy="{_n(y)}" r="3.75" '
        'stroke-width="1.5"/>'
    )


def _heading(y, title, note="") -> list[str]:
    out = [_text(_PAD, y, title, 12.5, "ink", weight=600)]
    if note:
        out.append(_text(_RIGHT, y, note, 10.5, "muted", "end"))
    return out


def _style(theme: str) -> str:
    def rules(index: int) -> str:
        fills = "".join(f".fc-{k}{{fill:{v[index]}}}" for k, v in _COLORS.items())
        strokes = "".join(f".fc-s-{k}{{stroke:{v[index]}}}" for k, v in _COLORS.items())
        return fills + strokes

    if theme == "light":
        return rules(0)
    if theme == "dark":
        return rules(1)
    return rules(0) + "@media (prefers-color-scheme:dark){" + rules(1) + "}"


# -- sections: each returns its elements and its height -----------------------
def _time_rows(timing, optimized: bool) -> list[tuple]:
    """``(label, color, time, stats, note)`` of the bars that have a time."""
    if timing is None:
        return []
    python = _positive(getattr(timing, "python_us", None))
    first = _positive(getattr(timing, "c_initial_us", None))
    final = _positive(getattr(timing, "c_final_us", None)) if optimized else None
    rows = []
    if python:
        rows.append(("NumPy reference", "numpy", python, None, ""))
    if first:
        note = f"{_times(python / first)} vs NumPy" if python else ""
        rows.append(("Translated solver", "translated", first, timing.c_initial_stats, note))
    if final:
        note = f"{_times(first / final)} vs translated" if first else ""
        rows.append(("Optimized solver", "optimized", final, timing.c_final_stats, note))
    return rows


def _spread(stats) -> tuple[float, float] | None:
    low, high = _positive(getattr(stats, "min", None)), _positive(getattr(stats, "max", None))
    return (low, high) if low and high and high >= low else None


def _time_section(y: float, rows: list[tuple]) -> tuple[list[str], float]:
    spreads = [_spread(r[3]) for r in rows]
    values = [r[2] for r in rows] + [v for s in spreads if s for v in s]
    low = math.floor(math.log10(min(values)))
    if math.log10(min(values)) - low < 0.2:
        low -= 1  # the shortest bar keeps a visible length
    high = max(math.ceil(math.log10(max(values))), low + 1)

    def x(value: float) -> float:
        return _X0 + (_X1 - _X0) * (math.log10(value) - low) / (high - low)

    pitch = 38 if any(spreads) else 30
    top = y + 14
    bottom = top + pitch * len(rows)
    labels = {k: format_time(10.0**k, digits=1) for k in range(low, high + 1)}
    widest = max(_width(label, 10, True) for label in labels.values()) + 10
    step = math.ceil(widest * (high - low) / (_X1 - _X0))  # labels do not touch
    lines = max(1, math.ceil((high - low) / 24))  # nor do gridlines
    out = _heading(
        y, "Time per solve", "log scale: each gridline is 10×" if lines == 1 else "log scale"
    )
    for k, label in labels.items():
        labelled = (k - low) % step == 0
        if not labelled and (k - low) % lines:
            continue
        xk = _X0 + (_X1 - _X0) * (k - low) / (high - low)
        end = bottom + (4 if labelled else 0)
        out.append(_line(xk, top, xk, end, "axis" if k == low else "grid"))
        if labelled:
            out.append(_text(xk, bottom + 16, label, 10, "muted", "middle", mono=True))
    for i, ((label, color, value, stats, note), spread) in enumerate(zip(rows, spreads)):
        mid = top + pitch * i + (15 if spread else pitch / 2)
        out.append(_text(_PAD, mid + 4, label, 12.5, "ink"))
        shown = format_time(value)
        out.append(_bar(_X0, mid - 6, x(value) - _X0, 12, color, f"{label}: {shown} per solve"))
        if spread:
            a, b = x(spread[0]), x(spread[1])
            out.append(_line(a, mid, b, mid, "ink", 1.25))
            out += [_line(v, mid - 3.5, v, mid + 3.5, "ink", 1.25) for v in (a, b)]
            runs = getattr(stats, "runs", None)
            text = (
                f"{format_time(spread[0], like=value, digits=4, unit=False)} to "
                f"{format_time(spread[1], like=value, digits=4)}"
                + (f", {runs} runs" if runs else "")
            )
            out.append(_text(_VX, mid + 19, text, 10.5, "muted", mono=True))
        out.append(_text(_VX, mid + 4.5, shown, 13.5, "ink", mono=True, weight=600))
        if note:
            out.append(_text(_RIGHT, mid + 4.5, note, 11.5, "ink2", "end"))
    return out, bottom + 24 - y


def _round_outcome(r) -> tuple[str, str]:
    """``(mark, label)`` of a round, as the terminal progress words it."""
    gain = getattr(r, "gain", None)
    if not r.changed:
        return "unchanged", "no further change"
    if not r.correct:
        return "failed", "reverted: failed verification"
    if not r.kept:
        if _positive(r.us_after) is None:
            return "reverted", "reverted: not timed"
        return "reverted", "reverted: inside noise" if gain and gain > 0 else "reverted: not faster"
    if _positive(r.us_after) is None:
        return "kept", "kept, not timed"
    if gain is None or not math.isfinite(gain):
        return "kept", "kept"
    return "kept", f"kept, {-100 * gain:+.1f}%".replace("-", "−")


def _rounds_section(y: float, rounds: list) -> tuple[list[str], float]:
    start = next((v for v in (_positive(r.us_before) for r in rounds) if v), None)
    times = [v for r in rounds for v in (_positive(r.us_before), _positive(r.us_after)) if v]
    out = _heading(
        y,
        "Optimization rounds",
        "linear scale, not from zero; the line is the best so far" if times else "",
    )
    low, high = (min(times), max(times)) if times else (1.0, 1.0)
    span = (high - low) or 0.2 * high
    low, high = low - 0.12 * span, high + 0.12 * span

    def x(value: float) -> float:
        return _X0 + (_X1 - _X0) * (value - low) / (high - low)

    pitch = 26
    top = y + 14
    rows = [("Translated", "translated", start, "")] if start else []
    for r in rounds:
        kind, label = _round_outcome(r)
        rows.append((f"Round {r.index}", kind, _positive(r.us_after), label))
    bottom = top + pitch * len(rows)
    marks, path, best = [], "", None
    for i, (label, kind, value, outcome) in enumerate(rows):
        mid = top + pitch * i + pitch / 2
        out.append(_text(_PAD, mid + 4, label, 12.5, "ink"))
        if times:
            out.append(_line(_X0, mid, _X1, mid, "grid"))
        if value and kind in ("translated", "kept"):
            path += f"M{_n(x(value))} {_n(mid)}" if best is None else f"V{_n(mid)}H{_n(x(value))}"
            best = value
        elif best is not None:
            path += f"V{_n(mid)}"
        if value:
            marks.append(_mark(kind, x(value), mid))
        marks.append(_mark(kind, _VX + 5, mid))  # the key of the outcome text
        shown = format_time(value) if value else ""
        if shown:
            out.append(_text(_VX + 16, mid + 4, shown, 12, "ink", mono=True))
        room = _RIGHT - _VX - 16 - (_width(shown, 12, True) + 10 if shown else 0)
        if outcome:
            out.append(_text(_RIGHT, mid + 4, _fit(outcome, 11.5, room), 11.5, "ink2", "end"))
    if path:
        out.append(
            f'<path d="{path}" fill="none" class="fc-s-ink2" stroke-width="1.5" '
            'stroke-linejoin="round"/>'
        )
    out += marks
    if not times:
        return out, bottom + 6 - y
    out.append(_line(_X0, bottom, _X1, bottom, "axis"))
    ends = sorted({min(times), max(times)})
    for i, value in enumerate(ends):
        out.append(_line(x(value), bottom, x(value), bottom + 4, "axis"))
        anchor = "middle" if len(ends) == 1 else ("start", "end")[i]
        out.append(_text(x(value), bottom + 16, format_time(value), 10, "muted", anchor, mono=True))
    return out, bottom + 24 - y


def _profile_section(y: float, profile: list) -> tuple[list[str], float]:
    regions = []
    for r in profile:
        share = _positive(getattr(r, "pct", None))
        if share:
            regions.append((share, _clean(getattr(r, "name", ""))))
    regions.sort(key=lambda item: (-item[0], item[1]))
    total = sum(share for share, _ in regions)
    if not total:
        return [], 0.0
    parts = [(share, name, color) for (share, name), color in zip(regions, _PROFILE_COLORS)]
    rest = regions[_PROFILE_TOP:]
    if rest:
        plural = "s" if len(rest) > 1 else ""
        parts.append((sum(s for s, _ in rest), f"{len(rest)} other region{plural}", "other"))

    out = _heading(y, "Where the time goes", "profile of the shipped solver, share of the total")
    top, height, gap = y + 14, 12, 2
    width = _RIGHT - _PAD - gap * (len(parts) - 1)
    clip = f"fc-clip-{_n(y)}"
    out.append(
        f'<clipPath id="{clip}"><rect x="{_PAD}" y="{_n(top)}" width="{_RIGHT - _PAD}" '
        f'height="{height}" rx="4"/></clipPath>'
    )
    left = float(_PAD)
    segments = []
    for share, name, color in parts:
        w = width * share / total
        segments.append(
            f'<rect class="fc-{color} fc-segment" x="{_n(left)}" y="{_n(top)}" width="{_n(w)}" '
            f'height="{height}"><title>{_esc(name)}: {100 * share / total:.1f}%</title></rect>'
        )
        left += w + gap
    out.append(f'<g clip-path="url(#{clip})">' + "".join(segments) + "</g>")
    columns, pitch = 2, 20
    column = (_RIGHT - _PAD) / columns
    for i, (share, name, color) in enumerate(parts):
        cx = _PAD + column * (i % columns)
        cy = top + height + 20 + pitch * (i // columns)
        out.append(
            f'<rect class="fc-{color}" x="{_n(cx)}" y="{_n(cy - 9)}" width="10" height="10" rx="2"/>'
        )
        share_text = f"{100 * share / total:.1f}%"
        out.append(_text(cx + 16, cy, _fit(name, 11.5, column - 96, True), 11.5, "ink", mono=True))
        out.append(_text(cx + column - 24, cy, share_text, 11.5, "ink2", "end", mono=True))
    rows = (len(parts) + columns - 1) // columns
    return out, top + height + 20 + pitch * (rows - 1) + 14 - y


def _verify_section(y: float, verify) -> tuple[list[str], float]:
    samples, error, tolerance = (tuple(verify) + (None, None))[:3]
    error = float(error) if error is not None and math.isfinite(float(error)) else None
    error = None if error is None else abs(error)
    tolerance = _positive(tolerance)
    scale = tolerance is not None and error is not None
    out = _heading(y, "Acceptance check", "log scale: one tick per decade" if scale else "")
    line = y + 24
    out.append(_check(_PAD, line - 4))
    facts = []
    if error is not None:
        facts.append(f"worst error {_sci(error)}")
    if tolerance:
        facts.append(f"tolerance {_sci(tolerance)}")
    if scale and 0 < error < tolerance:
        facts.append(f"margin {_times(tolerance / error)}")
    facts = " · ".join(facts)
    room = _RIGHT - _PAD - 18 - (_width(facts, 11.5, True) + 16 if facts else 0)
    plural = "" if samples == 1 else "s"
    said = f"Verified on {_clean(samples)} sample{plural}"
    if _width(said + " against the NumPy reference", 12.5) <= room:
        said += " against the NumPy reference"
    out.append(_text(_PAD + 18, line, _fit(said, 12.5, room), 12.5))
    if facts:
        out.append(_text(_RIGHT, line, facts, 11.5, "ink2", "end", mono=True))
    if not scale:
        return out, line + 14 - y

    # Decades from below the error to one past the tolerance, at least eight.
    high = math.ceil(math.log10(tolerance)) + 1
    largest = max(error, tolerance)
    high = max(high, math.ceil(math.log10(largest)) + 1)
    low = min(math.floor(math.log10(error)) - 1 if error else high - 14, high - 8)
    x0, x1 = _PAD + 8, _RIGHT - 120
    track = line + 34

    def x(value: float) -> float:
        position = (math.log10(value) - low) / (high - low) if value else 0.0
        return x0 + (x1 - x0) * min(1.0, max(0.0, position))

    out.append(
        f'<rect class="fc-track" x="{_n(x0 - 6)}" y="{_n(track - 4)}" width="{_n(x1 - x0 + 12)}" '
        'height="8" rx="4"/>'
    )
    step = 1 if high - low <= 24 else math.ceil((high - low) / 24)
    for k in range(low, high + 1, step):
        xk = x0 + (x1 - x0) * (k - low) / (high - low)
        out.append(_line(xk, track - 2, xk, track + 2, "axis"))
    xt, xe = x(tolerance), x(error)
    out.append(_line(xt, track - 11, xt, track + 11, "ink2", 2))
    out.append(_text(xt + 7, track - 9, "tolerance", 11, "ink2"))
    out.append(
        f'<circle class="fc-optimized fc-s-bg fc-error" cx="{_n(xe)}" cy="{_n(track)}" r="6" '
        f'stroke-width="2"><title>worst error {_esc(_sci(error))}</title></circle>'
    )
    label = "worst error" + (" (zero)" if not error else "")
    right = xe + _width(label, 11) / 2 > x1
    left = xe - _width(label, 11) / 2 < _PAD
    anchor = "end" if right else "start" if left else "middle"
    out.append(
        _text(
            _RIGHT - 120 if right else _PAD if left else xe, track + 22, label, 11, "ink2", anchor
        )
    )
    return out, track + 30 - y


# -- the card ------------------------------------------------------------------
def report_svg(
    name: str,
    target: str | None = None,
    dtype: str | None = None,
    timing=None,
    rounds=(),
    verify: tuple | None = None,
    theme: str = "auto",
) -> str:
    """The report card of a finished code-generation run, as an SVG string.

    Parameters
    ----------
    name : str
        The solver's name.
    target, dtype : str, optional
        The target (``"embedded_c"``) and the floating-point type.
    timing : TimingReport, optional
        The measured times and the profile. Without it the time and profile
        parts are left out.
    rounds : sequence of OptimizationRound
        The rounds of the code optimization phase. A single round that was kept
        and timed adds nothing to the bars, so it is not drawn.
    verify : tuple, optional
        ``(samples, worst max-abs error, tolerance)`` of the passing verify, as
        `fomtools.codegen.progress.verify_numbers` returns it.
    theme : {"auto", "light", "dark"}
        ``"auto"`` follows ``prefers-color-scheme``.

    Returns
    -------
    str
        A standalone SVG document. The same arguments give the same string.
    """
    rounds = list(rounds or ())
    optimized = (
        any(r.kept for r in rounds)
        if rounds
        else bool(
            timing is not None
            and _positive(timing.c_final_us)
            and _positive(timing.c_initial_us)
            and timing.c_final_us != timing.c_initial_us
        )
    )
    bars = _time_rows(timing, optimized)
    final = bars[-1] if bars and bars[-1][1] != "numpy" else None
    python = bars[0][2] if bars and bars[0][1] == "numpy" else None
    speedup = python / final[2] if python and final else None

    # The headline, also the accessible name.
    facts = []
    if final:
        facts.append(f"{format_time(final[2])} per solve")
    if speedup:
        facts.append(f"{_times(speedup)} the speed of the NumPy reference")
    if verify:
        samples, error = verify[0], verify[1]
        facts.append(
            f"verified on {samples} samples with worst error {_sci(abs(error))}"
            + (f" (tolerance {_sci(verify[2])})" if len(verify) > 2 and verify[2] else "")
        )
    kept = sum(1 for r in rounds if r.kept)
    if rounds:
        facts.append(f"{kept} of {len(rounds)} optimization rounds kept")
    label = f"{_clean(name)}: " + ("; ".join(facts) if facts else "generated solver")

    body: list[str] = []
    hero = speedup is not None and speedup >= 1.005
    body.append(
        _text(_PAD, 42, _fit(name, 19, _RIGHT - _PAD - (190 if hero else 0)), 19, "ink", weight=600)
    )
    subtitle = "  ·  ".join(_clean(part) for part in (target, dtype) if part)
    if subtitle:
        body.append(_text(_PAD, 62, _fit(subtitle, 12, 420, True), 12, "ink2", mono=True))
    if hero:
        body.append(_text(_RIGHT, 44, _times(speedup), 26, "ink", "end", weight=600))
        body.append(_text(_RIGHT, 62, "faster than NumPy", 11.5, "ink2", "end"))
    y = 78.0

    sections = []
    if bars:
        sections.append(lambda at: _time_section(at, bars))
    single = len(rounds) == 1 and rounds[0].kept and _positive(rounds[0].us_after)
    if rounds and not single:
        sections.append(lambda at: _rounds_section(at, rounds))
    if timing is not None and getattr(timing, "profile", None):
        sections.append(lambda at: _profile_section(at, timing.profile))
    if verify:
        sections.append(lambda at: _verify_section(at, verify))
    for section in sections:
        elements, height = section(y + 22)
        if not elements:
            continue
        body.append(_line(_PAD, y, _RIGHT, y, "grid"))
        body += elements
        y += 22 + height + 8
    if len(body) <= 3 and not sections:
        body.append(
            _text(_PAD, y + 12, "No timings or verification numbers were recorded.", 12, "ink2")
        )
        y += 24
    height = math.ceil(y + 10)

    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {_W} {height}" width="{_W}" '
        f'height="{height}" role="img" aria-label="{_esc(label)}" font-family="{_SANS}">'
        f"<title>{_esc(label)}</title><style>{_style(theme)}</style>"
        f'<rect class="fc-bg fc-s-border" x="0.5" y="0.5" width="{_W - 1}" height="{height - 1}" '
        'rx="10"/>' + "".join(body) + "</svg>\n"
    )


def result_report_svg(result, theme: str = "auto") -> str:
    """`report_svg` of a `fomtools.codegen.generate.CodegenResult`."""
    target = getattr(result, "target", None)
    return report_svg(
        getattr(getattr(result, "ir", None), "name", None) or "solver",
        target=getattr(target, "name", None),
        dtype=getattr(getattr(target, "config", None), "dtype", None),
        timing=getattr(result, "timing", None),
        rounds=getattr(result, "rounds", None) or (),
        verify=getattr(result, "verify", None),
        theme=theme,
    )


def remove_report_svg(out_dir: str | Path) -> None:
    """Drop the card of an earlier run: the directory is about to change."""
    (Path(out_dir) / REPORT_SVG_NAME).unlink(missing_ok=True)


def write_report_svg(out_dir: str | Path, svg: str) -> Path:
    """Write ``svg`` as ``report.svg`` in ``out_dir`` and return its path."""
    path = Path(out_dir) / REPORT_SVG_NAME
    path.write_text(svg, encoding="utf-8")
    return path
