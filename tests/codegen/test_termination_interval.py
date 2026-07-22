"""Tests for the OSQP-style ``check_termination`` interval.

``TerminationSpec.check_termination`` controls how often the INF_NORM_CHANGE
criterion is evaluated: every N iterations (residual skipped in between), with
0 disabling early exit entirely. The oracle, the emitted header macros, and
the solver body must agree on the semantics.
"""

import numpy as np
import pytest

from fomtools.codegen import spec as spec_mod
from fomtools.codegen.config import CodegenConfig, TerminationKind, TerminationSpec
from fomtools.codegen.examples import ista_example
from fomtools.codegen.extract import extract_ir
from fomtools.codegen.oracle import ReferenceOracle
from fomtools.codegen.targets.embedded_c import EmbeddedCTarget


def _ista_ir(term):
    ex = ista_example()
    config = CodegenConfig(termination=term)
    return extract_ir(ex["verifier"], ex["next_state"], config=config, name="ista"), config


def test_check_termination_validation():
    TerminationSpec(check_termination=0)  # disabled is allowed
    TerminationSpec(
        kind=TerminationKind.INF_NORM_CHANGE, max_iters=10, tol=1e-6, check_termination=25
    )
    with pytest.raises(ValueError, match="check_termination"):
        TerminationSpec(check_termination=-1)


def test_workspace_header_macros(tmp_path):
    term = TerminationSpec(
        kind=TerminationKind.INF_NORM_CHANGE, max_iters=100, tol=1e-6, check_termination=8
    )
    ir, config = _ista_ir(term)
    header = EmbeddedCTarget(config)._workspace_header(ir)
    assert "#define FOM_CHECK_TERMINATION 8" in header
    assert "#define FOM_USE_TOL 1" in header

    # check_termination=0 disables checking: FOM_USE_TOL goes to 0 and the
    # interval macro is clamped so `% FOM_CHECK_TERMINATION` stays well-defined.
    term0 = TerminationSpec(
        kind=TerminationKind.INF_NORM_CHANGE, max_iters=100, tol=1e-6, check_termination=0
    )
    ir0, config0 = _ista_ir(term0)
    header0 = EmbeddedCTarget(config0)._workspace_header(ir0)
    assert "#define FOM_USE_TOL 0" in header0
    assert "#define FOM_CHECK_TERMINATION 1" in header0


def test_oracle_checks_only_on_interval():
    b_val = np.zeros(4)

    def iters_with(check):
        term = TerminationSpec(
            kind=TerminationKind.INF_NORM_CHANGE,
            max_iters=500,
            tol=1e-6,
            check_termination=check,
        )
        ir, _ = _ista_ir(term)
        return ReferenceOracle(ir).solve(param_values={"p0": b_val})

    res1 = iters_with(1)
    res8 = iters_with(8)
    assert res1.iters < 500, "ista should converge before max_iters"
    # With interval 8 the exit can only happen on a multiple of 8, at or after
    # the every-iteration exit point.
    assert res8.iters % 8 == 0
    assert res1.iters <= res8.iters < res1.iters + 8

    res0 = iters_with(0)
    assert res0.iters == 500, "check_termination=0 must disable early exit"


def test_check_termination_spec_roundtrip(tmp_path):
    term = TerminationSpec(
        kind=TerminationKind.INF_NORM_CHANGE, max_iters=50, tol=1e-4, check_termination=25
    )
    ir, _ = _ista_ir(term)
    spec_mod.save(ir, tmp_path)
    loaded = spec_mod.load(tmp_path)
    assert loaded.termination.check_termination == 25

    # Old spec.json files without the field default to 1.
    import json

    raw = json.loads((tmp_path / "spec.json").read_text())
    del raw["termination"]["check_termination"]
    (tmp_path / "spec.json").write_text(json.dumps(raw))
    assert spec_mod.load(tmp_path).termination.check_termination == 1
