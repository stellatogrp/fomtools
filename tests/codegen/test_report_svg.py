"""The report card of a run (`report_svg`), the shared time formatter, the file
next to the generated solver, and the notebook display of a `CodegenResult`."""

import base64
import math
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from fomtools.codegen import StubAgentDriver, generate_code
from fomtools.codegen.agent.driver import AgentResult
from fomtools.codegen.generate import CodegenResult
from fomtools.codegen.generate import OptimizationRound as Round
from fomtools.codegen.harness import BenchStats, ProfRegion, TimingReport
from fomtools.codegen.manifest import check_manifest
from fomtools.codegen.progress import format_time
from fomtools.codegen.report_svg import REPORT_SVG_NAME, _width, report_svg
from tests.codegen.conftest import build_ista
from tests.codegen.test_optimization_rounds import _run

GOLDEN = Path(__file__).parent / "golden" / "ista" / "embedded_c" / "fom_solver.c"
NS = "{http://www.w3.org/2000/svg}"
PROFILE = [
    ProfRegion("soft_threshold", 20246.0, 1000, 51.21),
    ProfRegion("state_update", 19288.0, 1000, 48.79),
]
ROUNDS = [
    Round(1, True, True, True, 1.753, 1.52, 0.133, noise=0.01),
    Round(2, True, True, False, 1.52, 1.55, -0.02, noise=0.015),
    Round(3, True, True, False, 1.52, 1.51, 0.007, noise=0.02),
    Round(4, True, False, False),
    Round(5, False, True, False),
]
TIMING = TimingReport(
    1328.25,
    1.753,
    1.52,
    PROFILE,
    BenchStats(5, 1.753, 1.749, 1.79, 0.004),
    BenchStats(5, 1.52, 1.51, 1.56, 0.01),
)
FULL = dict(
    name="lasso",
    target="embedded_c",
    dtype="float64",
    timing=TIMING,
    rounds=ROUNDS,
    verify=(60, 4.4e-16, 1e-6),
)


def _texts(svg: str) -> str:
    return "\n".join(t.text or "" for t in ET.fromstring(svg).iter(f"{NS}text"))


# -- the shared time formatter -------------------------------------------------
@pytest.mark.parametrize(
    "us, text",
    [
        (0.04, "40.0 ns"),
        (0.9996, "1.00 µs"),
        (1.753, "1.75 µs"),
        (40.0, "40.0 µs"),
        (999.96, "1.00 ms"),
        (1328.25, "1.33 ms"),
        (2.05e6, "2.05 s"),
        (1.2345e10, "12,345 s"),
        (0, "0 s"),
        (None, "-"),
        (float("nan"), "-"),
        (float("inf"), "-"),
    ],
)
def test_format_time(us, text):
    assert format_time(us) == text


def test_format_time_in_the_unit_of_another():
    assert format_time(0.004, like=1.753, digits=4, unit=False) == "0.004"
    assert format_time(1.79, like=1.753, digits=4) == "1.790 µs"
    assert format_time(1.0e-3, digits=1) == "1 ns" and format_time(1.0e4, digits=1) == "10 ms"
    assert format_time(None, missing="n/a") == "n/a"
    assert "1.33 ms" in TIMING.summary() and "(1.749 to 1.790 µs, 5 runs)" in TIMING.summary()


# -- the card ------------------------------------------------------------------
def test_full_card_is_wellformed_accessible_and_complete():
    svg = report_svg(**FULL)
    root = ET.fromstring(svg)
    assert root.tag == f"{NS}svg" and root.get("role") == "img"
    label = root.get("aria-label")
    assert root.find(f"{NS}title").text == label
    for fact in ("lasso", "1.52 µs per solve", "874×", "60 samples", "4.4e-16", "1 of 5"):
        assert fact in label
    text = _texts(svg)
    for expected in (
        "lasso",
        "embedded_c",
        "float64",
        "NumPy reference",
        "1.33 ms",
        "Translated solver",
        "1.75 µs",
        "758× vs NumPy",
        "Optimized solver",
        "1.52 µs",
        "1.15× vs translated",
        "1.749 to 1.790 µs, 5 runs",
        "log scale",
        "Round 5",
        "kept, −13.3%",
        "reverted: not faster",
        "reverted: inside noise",
        "reverted: failed verification",
        "no further change",
        "linear scale, not from zero",
        "soft_threshold",
        "51.2%",
        "Verified on 60 samples",
        "worst error 4.4e-16 · tolerance 1e-6 · margin 2.3e9×",
    ):
        assert expected in text, expected
    assert "prefers-color-scheme:dark" in svg


@pytest.mark.parametrize(
    "missing, absent",
    [
        (dict(timing=None), ["Time per solve", "Where the time goes"]),
        (dict(rounds=[]), ["Optimization rounds"]),
        (dict(rounds=ROUNDS[1:]), ["Optimized solver"]),  # nothing kept
        (dict(rounds=ROUNDS[:1]), ["Optimization rounds"]),  # one kept round: the bars say it
        (dict(verify=None), ["Acceptance check"]),
        (dict(verify=(8, 1e-9, None)), ["tolerance"]),
        (dict(timing=TimingReport(None, 1.7, 1.5)), ["NumPy", "Where the time goes"]),
        (dict(timing=TimingReport(900.0, None, None)), ["Translated solver"]),
        (dict(timing=TimingReport(), rounds=[Round(1, True, True, True)]), ["Time per solve"]),
        (dict(target=None, dtype=None), ["embedded_c", "float64"]),
        (
            dict(timing=None, rounds=[], verify=None),
            ["Time per solve", "Optimization rounds", "Where the time goes", "Acceptance check"],
        ),
    ],
)
def test_missing_data_leaves_the_part_out(missing, absent):
    svg = report_svg(**{**FULL, **missing})
    text = _texts(ET.tostring(ET.fromstring(svg), encoding="unicode"))
    assert "lasso" in text
    for label in absent:
        assert label not in text, label
    assert "None" not in svg and "nan" not in svg


def test_untrusted_text_is_escaped():
    name = "<script>alert('x')</script> & \"quoted\" \x00\x1b\nnext"
    region = ProfRegion("</text><rect onload='x'/>", 1.0, 1, 100.0)
    svg = report_svg(name, target="<b>", dtype="&amp;", timing=TimingReport(profile=[region]))
    root = ET.fromstring(svg)  # still well-formed
    assert not list(root.iter(f"{NS}script")) and "onload" not in [
        k for e in root.iter() for k in e.attrib
    ]
    assert "<script>alert('x')</script> & \"quoted\" ?? next" in root.get("aria-label")
    assert "</text><rect onload='x'/>" in _texts(svg)


def test_same_input_gives_the_same_bytes():
    assert report_svg(**FULL) == report_svg(**FULL)
    assert report_svg(theme="dark", **FULL) != report_svg(theme="light", **FULL)
    assert "@media" not in report_svg(theme="light", **FULL)


def test_kept_and_reverted_rounds_differ_by_shape_and_label():
    root = ET.fromstring(report_svg(**FULL))
    marks = {
        kind: [e for e in root.iter() if f"fc-mark-{kind}" in (e.get("class") or "").split()]
        for kind in ("kept", "reverted", "failed", "unchanged")
    }
    # One mark on the axis and one next to the label; a failed or unchanged round has no time.
    assert [len(marks[k]) for k in ("kept", "reverted", "failed", "unchanged")] == [2, 4, 1, 1]
    kept, reverted = marks["kept"][0], marks["reverted"][0]
    assert kept.tag == reverted.tag == f"{NS}circle"
    assert kept.get("stroke-width") is None  # a filled disc
    assert reverted.get("stroke-width") and "fc-bg" in reverted.get("class")  # a hollow ring
    assert marks["failed"][0].find(f"{NS}path") is not None  # a cross
    assert marks["unchanged"][0].find(f"{NS}line") is not None  # a dash
    text = _texts(report_svg(**FULL))
    assert text.count("kept,") == 1 and text.count("reverted:") == 3


EXTREME = [
    dict(timing=TimingReport(2.5e6, 0.04, 0.038, PROFILE), rounds=ROUNDS[:1]),
    dict(timing=TimingReport(1.0, 1.0, 1.0)),
    dict(timing=TimingReport(1e-9, 1e12, 1e-9), rounds=[Round(1, True, True, True, 1e12, 1e-9)]),
    dict(verify=(10**9, 0.0, 1e-6)),
    dict(verify=(1, 1e-300, 1e300)),
    dict(verify=(1, 5.0, 1e-6)),  # an error above the tolerance still draws
    dict(name="W" * 400, target="x" * 200, dtype="y" * 200),
    dict(name="解" * 200, timing=TimingReport(profile=[ProfRegion("名" * 99, 1.0, 1, 3.0)] * 9)),
    dict(rounds=[Round(i, True, True, i % 2 == 0, 9.0, 9.0 - i, 0.1) for i in range(1, 13)]),
    dict(timing=TimingReport(float("nan"), float("inf"), -1.0, [ProfRegion("a", 1.0, 1, 0.0)])),
]


@pytest.mark.parametrize("case", EXTREME + [{}])
def test_geometry_stays_inside_the_card(case):
    svg = report_svg(**{**FULL, **case})
    assert not re.search(r"\b(nan|inf)\b", svg)
    root = ET.fromstring(svg)
    width, height = (float(v) for v in root.get("viewBox").split()[2:])

    def inside(x, y=None):
        assert -0.01 <= x <= width + 0.01, x
        assert y is None or -0.01 <= y <= height + 0.01, y

    for e in root.iter():
        tag = e.tag.removeprefix(NS)
        if tag == "rect":
            assert float(e.get("width")) >= 0 and float(e.get("height")) >= 0
            inside(float(e.get("x")), float(e.get("y")))
            inside(
                float(e.get("x")) + float(e.get("width")),
                float(e.get("y")) + float(e.get("height")),
            )
        elif tag == "line":
            inside(float(e.get("x1")), float(e.get("y1")))
            inside(float(e.get("x2")), float(e.get("y2")))
        elif tag == "circle":
            inside(float(e.get("cx")) - float(e.get("r")), float(e.get("cy")) - float(e.get("r")))
            inside(float(e.get("cx")) + float(e.get("r")), float(e.get("cy")) + float(e.get("r")))
        elif tag == "path":
            numbers = [float(v) for v in re.findall(r"-?\d+(?:\.\d+)?", e.get("d"))]
            assert all(math.isfinite(v) and abs(v) <= max(width, height) for v in numbers)
        elif tag == "text":
            # The module's own width estimate is cautious (wider than the glyphs).
            x, size = float(e.get("x")), float(e.get("font-size"))
            extent = _width(e.text or "", size, mono="monospace" in (e.get("font-family") or ""))
            anchor = e.get("text-anchor") or "start"
            left = x - {"start": 0, "middle": extent / 2, "end": extent}[anchor]
            inside(left, float(e.get("y")))
            inside(left + extent)
    for bar in root.iter(f"{NS}path"):
        if "fc-bar" in (bar.get("class") or ""):
            assert float(bar.get("data-length")) >= 0


def test_bar_lengths_follow_the_log_of_the_time():
    def lengths(python, first, final):
        svg = report_svg("x", timing=TimingReport(python, first, final))
        return [
            float(p.get("data-length"))
            for p in ET.fromstring(svg).iter(f"{NS}path")
            if "fc-bar" in (p.get("class") or "")
        ]

    numpy, translated, optimized = lengths(1.0e4, 10.0, 1.0)
    assert numpy > translated > optimized > 0
    # Equal ratios are equal lengths: one decade from 10 to 1, three from 1e4 to 10.
    assert numpy - translated == pytest.approx(3 * (translated - optimized), rel=1e-3)
    previous = 0.0
    for python in (2.0, 20.0, 3.0e3, 4.0e6, 5.0e9):  # 1000x and more: nothing vanishes
        numpy, translated, optimized = lengths(python, 1.5, 1.0)
        assert numpy > previous and translated > optimized > 5
        previous = 0.0 if python < 20 else previous  # the axis grows by whole decades
    assert "log scale" in _texts(report_svg("x", timing=TimingReport(1.0e4, 10.0, 1.0)))


# -- the file next to the solver -------------------------------------------------
def _stub_run(out_dir, driver=None, **kwargs):
    case = build_ista(max_iters=50)
    return generate_code(
        case["verifier"],
        case["next_state"],
        out_dir,
        config=case["config"],
        driver=driver or StubAgentDriver(GOLDEN),
        name="ista",
        profile=False,
        **kwargs,
    )


@pytest.mark.requires_compiler
def test_a_run_writes_the_card_and_a_rerun_replaces_it(tmp_path, requires_compiler):
    result = _stub_run(tmp_path, code_optimization=True)
    card = tmp_path / REPORT_SVG_NAME
    assert result.report_svg == card and card.is_file()
    assert result.verify and result.verify[0] > 0 and result.verify[2] == 1e-6
    root = ET.fromstring(card.read_text(encoding="utf-8"))
    assert f"verified on {result.verify[0]} samples" in root.get("aria-label")
    report = (tmp_path / "OPTIMIZATIONS.md").read_text()
    assert report.startswith(
        f"# Optimization report\n\n![Report card of the run]({REPORT_SVG_NAME})"
    )
    assert "## What the agent changed" in report  # the text sections stay
    check_manifest(tmp_path)  # the card is not a recorded source: nothing to trip

    # A translation-only rerun under another name replaces the card.
    card.write_text("stale")
    case = build_ista(max_iters=50)
    generate_code(
        case["verifier"],
        case["next_state"],
        tmp_path,
        config=case["config"],
        driver=StubAgentDriver(GOLDEN),
        name="second",
        profile=False,
    )
    assert "second" in ET.fromstring(card.read_text(encoding="utf-8")).get("aria-label")
    check_manifest(tmp_path)

    # A failed run leaves no card of the earlier one.
    failed = _stub_run(tmp_path, driver=StubAgentDriver("this is not C;\n"))
    assert not failed.success and failed.report_svg is None and not card.exists()
    assert "FAILED" in failed._repr_html_() and "<img" not in failed._repr_html_()


@pytest.mark.requires_compiler
def test_the_card_shows_the_rounds_of_a_run(tmp_path, monkeypatch, requires_compiler):
    times = {"translated": 100.0, "r1": 50.0, "r2": 50.5}
    result, _, _ = _run(tmp_path, monkeypatch, ["r1", "r2", "r3"], times)
    assert [r.kept for r in result.rounds] == [True, False]
    text = _texts(result.report_svg.read_text(encoding="utf-8"))
    assert "kept, −50.0%" in text and "reverted: not faster" in text
    assert "Optimized solver" in text and "50.0 µs" in text and "2.00× vs translated" in text


# -- the notebook display --------------------------------------------------------
def _result(success=True, **kwargs):
    translate = AgentResult(success=success, log="", agent_log="", turns_used=1)
    return CodegenResult(ir=None, out_dir=Path("out"), target=None, translate=translate, **kwargs)


def test_notebook_display_of_a_result():
    result = _result(timing=TIMING, rounds=ROUNDS, verify=(60, 4.4e-16, 1e-6))
    shown = result._repr_html_()
    data = re.search(r'src="data:image/svg\+xml;base64,([^"]+)"', shown).group(1)
    svg = base64.b64decode(data).decode("utf-8")
    assert ET.fromstring(svg).get("role") == "img" and "Round 5" in _texts(svg)
    assert "verified on 60 samples" in shown and "1.52 µs per solve" in shown
    assert "<style" not in shown  # the card's styles stay inside the image
    assert repr(result).startswith("CodegenResult(ir=None, out_dir=")  # the dataclass repr


def test_notebook_display_never_raises():
    failed = _result(success=False, timing=TIMING)
    assert "FAILED" in failed._repr_html_() and "<img" not in failed._repr_html_()
    assert "FAILED" in failed.summary()
    assert "<img" in _result()._repr_html_()  # nothing measured: still a card
    broken = _result(timing="not a timing", rounds=[object()], verify=("x",))
    assert isinstance(broken._repr_html_(), str)
    escaped = _result(success=False)
    escaped.out_dir = Path("<b>&")
    assert "<b>" not in escaped._repr_html_()
