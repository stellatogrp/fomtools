"""End-to-end code generation for a method with a *declared* varying coefficient.

Runs the full pipeline — probe, extract, scaffold, build, verify — with the
deterministic stub driver, then checks the compiled solver against a plain NumPy
loop over the *whole* generated horizon, not just the one iteration that was
recorded. That horizon is the point of declaring a schedule: the same method
recorded as an ordinary loop generates one block per iteration (see
``test_fista_pipeline.py``), whereas this emits one block and a table.

The fixture is the accelerated lasso, but nothing here depends on which method it
is; its coefficient is supplied by ``scheduled_example``, not by the package.
"""

import importlib.util
from pathlib import Path

import numpy as np

from fomtools.codegen import ReferenceOracle, StubAgentDriver, generate_code
from fomtools.codegen.harness import build_lib
from fomtools.codegen.spec import load, render_markdown, save
from fomtools.codegen.targets.embedded_c import EmbeddedCTarget
from tests.codegen.conftest import build_scheduled, fista_reference

GOLDEN = Path(__file__).parent / "golden" / "scheduled" / "embedded_c" / "fom_solver.c"


def _generate(tmp_path, max_iters=200, **kwargs):
    case = build_scheduled(max_iters=max_iters, **kwargs)
    result = generate_code(
        case["build"],
        None,
        tmp_path,
        config=case["config"],
        driver=StubAgentDriver(GOLDEN),
        name="scheduled",
        schedules=case["schedules"],
    )
    return case, result


def _load_wrapper(out_dir):
    spec = importlib.util.spec_from_file_location(
        "scheduled_wrapper", out_dir / "python" / "fom_solver.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --- the emitted table ------------------------------------------------------


def test_golden_passes_static_checks():
    """The golden must read the table through the clamping accessor."""
    target = EmbeddedCTarget(build_scheduled()["config"])
    assert target.static_checks(GOLDEN.read_text()) == []


def test_direct_table_indexing_is_rejected():
    """Indexing FOM_SCHED_* can run past the table, since the bound is runtime."""
    target = EmbeddedCTarget(build_scheduled()["config"])
    bad = GOLDEN.read_text().replace("fom_sched_gamma(it)", "FOM_SCHED_gamma[it]")
    assert any("indexed directly" in v for v in target.static_checks(bad))


def test_schedule_table_is_emitted(tmp_path):
    case, result = _generate(tmp_path, max_iters=64)

    header = (tmp_path / "include" / "fom_data.h").read_text()
    source = (tmp_path / "src" / "fom_data.c").read_text()

    assert "#define FOM_SCHED_GAMMA_LEN 64" in header
    assert "fom_float fom_sched_gamma(fom_int it)" in header
    assert "const fom_float FOM_SCHED_gamma[64]" in source
    # The emitted values are the schedule the IR carries.
    assert len(result.ir.schedules["gamma"]) == 64


def test_one_block_covers_any_horizon(tmp_path):
    """What declaring a schedule buys: the body does not grow with the horizon."""
    _, short = _generate(tmp_path / "short", max_iters=16)
    _, long_ = _generate(tmp_path / "long", max_iters=2000)

    assert len(short.ir.body) == len(long_.ir.body) == 1
    assert len(short.ir.schedules["gamma"]) == 16
    assert len(long_.ir.schedules["gamma"]) == 2000


# --- correctness ------------------------------------------------------------


def test_stub_pipeline_verifies(tmp_path, requires_compiler):
    _, result = _generate(tmp_path)
    assert result.translate.success, result.translate.log
    assert "VERIFY OK" in result.translate.log
    assert (tmp_path / "src" / "fom_solver.c").exists()


def test_generated_solver_matches_the_numpy_loop(tmp_path, requires_compiler):
    """The point of the whole exercise, over the full 200-iteration horizon.

    The recording is one iteration long; the generated solver runs 200. Nothing
    about the coefficient is inferred from the recorded horizon, so this is the
    check that matters.
    """
    case, result = _generate(tmp_path, max_iters=200)
    assert result.translate.success, result.translate.log

    build_lib(tmp_path)
    solver = _load_wrapper(tmp_path).Solver()

    rng = np.random.default_rng(7)
    for _ in range(3):
        b_val = case["data"]["b"] + rng.uniform(-0.5, 0.5, size=case["data"]["m"])
        out = solver.solve(p0=b_val)
        truth = fista_reference(case["data"], b_val, n_iters=200)
        assert np.allclose(out["x"], truth[-1], atol=1e-9), (out["x"], truth[-1])


def test_the_coefficient_survives_lowering(tmp_path, requires_compiler):
    """Guards the original silent degradation.

    Dropping the schedule would freeze the coefficient at its recorded value,
    which for this method is zero — turning it into plain ISTA. Both converge to
    the same solution, so they are compared over a short horizon where the
    acceleration still shows.
    """
    case, result = _generate(tmp_path, max_iters=6)
    assert result.translate.success, result.translate.log

    b_val = case["data"]["b"]
    rolled = ReferenceOracle(result.ir).solve(param_values={"p0": b_val}).solution
    accelerated = fista_reference(case["data"], b_val, n_iters=6)[-1]
    plain = fista_reference(case["data"], b_val, n_iters=6, momentum=False)[-1]

    assert np.abs(rolled - accelerated).max() < 1e-12
    assert np.abs(rolled - plain).max() > 1e-2


# --- persistence and the agent contract -------------------------------------


def test_scheduled_ir_survives_a_spec_roundtrip(tmp_path):
    _, result = _generate(tmp_path / "gen", max_iters=32)

    save(result.ir, tmp_path / "spec")
    loaded = load(tmp_path / "spec")

    assert set(loaded.schedules) == {"gamma"}
    assert np.allclose(loaded.schedules["gamma"], result.ir.schedules["gamma"])
    scaled = {t.matrix: t.scale for t in loaded.body[0].linear_op.terms if t.scale is not None}
    assert scaled == {"t0_state__d_gamma": "gamma"}


def test_spec_markdown_documents_the_schedule(tmp_path):
    _, result = _generate(tmp_path, max_iters=16)
    md = render_markdown(result.ir)

    assert "## Iteration-varying coefficients" in md
    assert "fom_sched_gamma(it)" in md
    assert "0-based" in md
