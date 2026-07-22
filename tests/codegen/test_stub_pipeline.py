"""End-to-end pipeline test using the LLM-free StubAgentDriver.

Exercises extract -> scaffold -> (stub drops golden solver) -> make verify, plus
the static-invariant checks on the golden source. Requires a C compiler.
"""

from pathlib import Path

import pytest

from fomtools.codegen import StubAgentDriver, generate_code
from fomtools.codegen.targets.embedded_c import EmbeddedCTarget
from tests.codegen.conftest import build_ista

GOLDEN = Path(__file__).parent / "golden" / "ista" / "embedded_c" / "fom_solver.c"


def test_ista_stub_pipeline_verifies(tmp_path, requires_compiler):
    case = build_ista(max_iters=200)
    driver = StubAgentDriver(GOLDEN)
    result = generate_code(
        case["verifier"],
        case["next_state"],
        tmp_path,
        config=case["config"],
        driver=driver,
        name="ista",
    )
    assert result.translate.success, result.translate.log
    assert "VERIFY OK" in result.translate.log
    # The solver source was written into the workspace.
    assert (tmp_path / "src" / "fom_solver.c").exists()


def test_golden_passes_static_checks():
    # No compiler needed: the embedded golden must be malloc- and division-free.
    target = EmbeddedCTarget(build_ista()["config"])
    violations = target.static_checks(GOLDEN.read_text())
    assert violations == [], violations


def test_stub_pipeline_inf_norm_change(tmp_path, requires_compiler):
    case = build_ista(max_iters=1000, tol=1e-8)
    driver = StubAgentDriver(GOLDEN)
    result = generate_code(
        case["verifier"],
        case["next_state"],
        tmp_path,
        config=case["config"],
        driver=driver,
        name="ista",
    )
    assert result.translate.success, result.translate.log


def test_failing_solver_reports_failure(tmp_path, requires_compiler):
    # A solver that returns the (wrong) initial state should fail verification.
    bad = """
    #include "fom_solver.h"
    #include "fom_data.h"
    void fom_set_default_settings(FomSettings *s){ s->max_iter=FOM_MAX_ITER; s->tol=FOM_TOL; }
    void fom_setup(FomWorkspace *w, const FomSettings *s){ int i; w->settings=*s;
        for(i=0;i<FOM_STATE_DIM;i++) w->state[i]=FOM_STATE_INIT[i]; }
    void fom_update_params(FomWorkspace *w, const FomParams *p){ w->params=*p; }
    fom_int fom_solve(FomWorkspace *w){ int i;
        for(i=0;i<FOM_STATE_DIM;i++) w->solution.x[i]=FOM_STATE_INIT[i];
        w->solution.status=FOM_MAX_ITER_REACHED; return FOM_MAX_ITER_REACHED; }
    void fom_cleanup(FomWorkspace *w){ (void)w; }
    """
    case = build_ista(max_iters=50)
    driver = StubAgentDriver(bad)
    result = generate_code(
        case["verifier"],
        case["next_state"],
        tmp_path,
        config=case["config"],
        driver=driver,
        name="ista",
    )
    assert not result.translate.success


def test_no_driver_without_env_raises(tmp_path):
    case = build_ista()
    with pytest.raises(RuntimeError, match="No coding-agent driver"):
        generate_code(case["verifier"], case["next_state"], tmp_path, config=case["config"])


def test_restore_hook_reverts_agent_tampering(tmp_path, requires_compiler):
    """If the agent clobbers a deterministic file (e.g. the Makefile), the
    restore hook reverts it before verify, so the build still succeeds."""

    class TamperingStub(StubAgentDriver):
        def run(self, task):
            (task.workdir / "Makefile").write_text("garbage:\n\techo broken\n")
            (task.workdir / "include" / "fom_data.h").write_text("/* clobbered */\n")
            return super().run(task)

    case = build_ista(max_iters=100)
    result = generate_code(
        case["verifier"],
        case["next_state"],
        tmp_path,
        config=case["config"],
        driver=TamperingStub(GOLDEN),
        name="ista",
    )
    assert result.translate.success, result.translate.log
    assert "VERIFY OK" in result.translate.log
    assert "garbage" not in (tmp_path / "Makefile").read_text()
    assert "clobbered" not in (tmp_path / "include" / "fom_data.h").read_text()
