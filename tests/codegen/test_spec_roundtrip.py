"""Tests that FomIR serialization is lossless."""

import json
from dataclasses import replace

import numpy as np

from fomtools.codegen import spec
from fomtools.codegen.extract import extract_ir
from fomtools.codegen.oracle import ReferenceOracle
from tests.codegen.conftest import build_dr, build_ista, build_spd_cg


def _assert_ir_equal(a, b):
    assert a.name == b.name
    assert a.dtype == b.dtype
    assert a.unrolled == b.unrolled
    assert a.state.dim == b.state.dim
    assert a.state.slots == b.state.slots
    assert np.allclose(a.state.init, b.state.init)
    assert a.termination == b.termination
    assert [p.name for p in a.params] == [p.name for p in b.params]
    assert set(a.matrices) == set(b.matrices)
    for name, md in a.matrices.items():
        assert np.allclose(md.dense(), b.matrices[name].dense())
        assert (md.sparse_pattern is None) == (b.matrices[name].sparse_pattern is None)
    assert len(a.body) == len(b.body)
    for sa, sb in zip(a.body, b.body):
        assert sa.name == sb.name
        assert sa.out_dim == sb.out_dim
        assert sa.activation == sb.activation
        assert sa.linear_op == sb.linear_op
    assert a.state_update == b.state_update


def test_ista_roundtrip(tmp_path):
    case = build_ista()
    ir = extract_ir(case["verifier"], case["next_state"], config=case["config"], name="ista")
    spec.save(ir, tmp_path)
    loaded = spec.load(tmp_path)
    _assert_ir_equal(ir, loaded)


def test_dr_roundtrip_preserves_sparse_flag(tmp_path):
    case = build_dr()
    ir = extract_ir(case["verifier"], case["next_state"], config=case["config"], name="dr")
    spec.save(ir, tmp_path)
    loaded = spec.load(tmp_path)
    _assert_ir_equal(ir, loaded)


def test_roundtrip_preserves_solve(tmp_path):
    case = build_ista(max_iters=100)
    ir = extract_ir(case["verifier"], case["next_state"], config=case["config"])
    spec.save(ir, tmp_path)
    loaded = spec.load(tmp_path)
    b_val = case["data"]["b"]
    r1 = ReferenceOracle(ir).solve(param_values={"p0": b_val})
    r2 = ReferenceOracle(loaded).solve(param_values={"p0": b_val})
    assert np.allclose(r1.solution, r2.solution)


def test_cg_factorization_roundtrip(tmp_path):
    case = build_spd_cg()
    ir = extract_ir(case["verifier"], case["next_state"], config=case["config"], name="cgqp")
    fac = ir.factorizations["t0"]
    assert fac.method == "cg"
    spec.save(ir, tmp_path)
    loaded = spec.load(tmp_path)
    _assert_ir_equal(ir, loaded)
    lfac = loaded.factorizations["t0"]
    assert lfac.method == "cg"
    assert lfac.cg_tol == fac.cg_tol
    assert lfac.cg_max_iter == fac.cg_max_iter
    assert np.allclose(lfac.A.to_dense(), fac.A.to_dense())
    assert np.allclose(lfac.minv, fac.minv)
    # Both reconstructions solve identically.
    qval = np.full(case["data"]["n"], 0.3)
    r1 = ReferenceOracle(ir).solve(param_values={"p0": qval})
    r2 = ReferenceOracle(loaded).solve(param_values={"p0": qval})
    assert np.allclose(r1.solution, r2.solution)


def test_unrolled_flag_roundtrips(tmp_path):
    case = build_ista()
    ir = extract_ir(case["verifier"], case["next_state"], config=case["config"])
    ir = replace(ir, unrolled=True)
    spec.save(ir, tmp_path)
    loaded = spec.load(tmp_path)
    assert loaded.unrolled is True
    _assert_ir_equal(ir, loaded)


def test_spec_written_before_unrolled_emission_still_loads(tmp_path):
    """A spec.json with no ``unrolled`` key loads as a rolled IR."""
    case = build_ista()
    ir = extract_ir(case["verifier"], case["next_state"], config=case["config"])
    spec.save(ir, tmp_path)

    path = tmp_path / "spec.json"
    payload = json.loads(path.read_text())
    del payload["unrolled"]
    path.write_text(json.dumps(payload))

    loaded = spec.load(tmp_path)
    assert loaded.unrolled is False
    _assert_ir_equal(ir, loaded)


def test_markdown_renders(tmp_path):
    case = build_dr()
    ir = extract_ir(case["verifier"], case["next_state"], config=case["config"], name="dr")
    md = spec.render_markdown(ir)
    assert "FOM specification" in md
    assert "Iteration body" in md
    assert "State update" in md


def test_direct_factorization_keeps_its_system_matrix(tmp_path):
    # The CUDA target factorizes A itself (cuDSS), so a direct factorization keeps
    # it: the A it solves, through a save/load round trip.
    case = build_dr()
    ir = extract_ir(case["verifier"], case["next_state"], config=case["config"], name="dr")
    spec.save(ir, tmp_path)
    loaded = spec.load(tmp_path)
    for name, fac in ir.factorizations.items():
        assert fac.method in ("ldlt", "lu") and fac.A is not None
        b = np.arange(1.0, fac.dim + 1)
        assert np.allclose(fac.A.to_scipy() @ fac.solve(b), b)
        assert np.allclose(
            loaded.factorizations[name].A.to_scipy().toarray(), fac.A.to_scipy().toarray()
        )
