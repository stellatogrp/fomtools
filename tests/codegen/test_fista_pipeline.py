"""End-to-end code generation for FISTA written as an ordinary loop.

FISTA used to be the awkward case: its momentum changes every iteration, so it
could not be recorded as a finished tape and had to be restructured into a build
function over a declared schedule. It no longer does. These tests generate the
loop a user writes to verify with, unchanged.

The trade is size, not capability: because the momentum genuinely varies, the run
does not roll back up and each iteration gets its own block. Declaring the same
method as a schedule gets one block plus a table instead — see
``test_scheduled_pipeline.py``.
"""

from pathlib import Path

import numpy as np

from fomtools.codegen import ReferenceOracle, StubAgentDriver
from fomtools.codegen.extract_unrolled import extract_unrolled_ir
from fomtools.codegen.spec import render_markdown
from fomtools.codegen.targets.embedded_c import EmbeddedCTarget
from tests.codegen.conftest import build_fista_run, build_ista, fista_loop_reference

GOLDEN = Path(__file__).parent / "golden" / "fista" / "embedded_c" / "fom_solver.c"

#: Matches the golden solver body, which names one block per recorded iteration.
STEPS = 6


def _ir(K=STEPS):
    case = build_fista_run(K=K)
    return case, extract_unrolled_ir(case["verifier"], case["iterates"], name="fista")


# --- the recording ----------------------------------------------------------


def test_recorded_fista_is_textbook_fista():
    """Guards an off-by-one nothing else could catch.

    The recording must apply momentum from the *second* iteration, not the third.
    Extraction self-checks against the recorded tape, so a lagged loop reproduces
    itself perfectly and looks fine; only an independent NumPy FISTA notices.
    """
    case, ir = _ir()
    data = case["data"]

    got = ReferenceOracle(ir).solve(param_values={"p0": data["b"]}).solution
    assert np.allclose(got, fista_loop_reference(data, data["b"], STEPS)[-1], atol=1e-12)


def test_recorded_fista_is_not_ista():
    """The anti-degradation assertion, on the recorded-run path."""
    case, ir = _ir()
    data = case["data"]

    got = ReferenceOracle(ir).solve(param_values={"p0": data["b"]}).solution
    plain = fista_loop_reference(data, data["b"], STEPS, momentum=False)[-1]
    assert np.abs(got - plain).max() > 1e-2


def test_recorded_fista_does_not_roll():
    """Its momentum genuinely varies, so one iteration cannot stand for the run."""
    _, ir = _ir()
    assert ir.unrolled is True
    assert len(ir.body) == STEPS


def test_momentum_appears_as_a_two_iterate_lookback():
    """Nothing declared it: extrapolating over two iterates is what the tape says.

    Later blocks read the two preceding temporaries. The first two are the
    exception, because there are not yet two to read.
    """
    _, ir = _ir()
    for step in ir.body[2:]:
        tmps = [t.source for t in step.linear_op.terms if t.source.startswith("tmp:")]
        assert len(tmps) == 2, (step.name, tmps)


def test_the_parameter_block_is_shared_across_iterations():
    """Every iteration applies the same parameter coefficient, so dedup merges it."""
    _, ir = _ir()
    param_blocks = {
        t.matrix for step in ir.body for t in step.linear_op.terms if t.source == "param:p0"
    }
    assert len(param_blocks) == 1


# --- emission ---------------------------------------------------------------


def test_spec_describes_a_single_pass():
    _, ir = _ir()
    md = render_markdown(ir)
    assert "to be implemented as a single pass" in md
    assert "Do **not** write a loop over iterations" in md
    # No schedule was declared, so there is no table to document.
    assert "## Iteration-varying coefficients" not in md
    assert "fom_sched" not in md


def test_golden_passes_static_checks():
    target = EmbeddedCTarget(build_ista()["config"])
    assert target.static_checks(GOLDEN.read_text()) == []


def test_codegen_verifies_the_recorded_loop(tmp_path, requires_compiler):
    """The whole point: verify with a loop, then generate that same loop."""
    case = build_fista_run(K=STEPS)
    result = case["verifier"].codegen(
        tmp_path,
        iterates=case["iterates"],
        config=case["config"],
        driver=StubAgentDriver(GOLDEN),
        name="fista",
    )

    assert result.success, result.translate.log
    assert "VERIFY OK" in result.translate.log
    assert result.ir.unrolled is True
    assert len(result.ir.body) == STEPS
    assert "#define FOM_UNROLLED 1" in (tmp_path / "include" / "fom_workspace.h").read_text()
