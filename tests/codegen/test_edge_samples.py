"""Deterministic edge-case parameter samples in the end-to-end acceptance check.

Uniform draws from the interior of the parameter box almost never reach the
points where a projection saturates or a soft-thresholding switches branch. The
verify data therefore also holds the bounds, the midpoint, zero, mixed vertices
and face points, each answered by the oracle like a random sample.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from fomtools.codegen import StubAgentDriver, generate_code
from fomtools.codegen.extract import extract_ir
from fomtools.codegen.ir import ParamSpec
from fomtools.codegen.manifest import MANIFEST_NAME
from fomtools.codegen.oracle import MAX_EDGE_SAMPLES, edge_param_values
from fomtools.codegen.targets.embedded_c import EmbeddedCTarget
from tests.codegen.conftest import build_boxqp, gate

BOXQP_GOLDEN = Path(__file__).parent / "golden" / "boxqp" / "embedded_c" / "fom_solver.c"
LOWER_CLIP = "            if (acc < FOM_MAT_t0_lo[i]) acc = FOM_MAT_t0_lo[i];\n"


def _param(name, lb, ub):
    lb, ub = np.asarray(lb, dtype=float), np.asarray(ub, dtype=float)
    return ParamSpec(name=name, dim=lb.size, lb=lb, ub=ub)


def _stack(values, params):
    return np.array([np.concatenate([v[p.name] for p in params]) for v in values])


def _contains(rows, vec):
    return any(np.array_equal(row, vec) for row in rows)


# -- the sample set ------------------------------------------------------------
def test_edge_samples_hold_the_bounds_the_midpoint_and_zero():
    params = [_param("p0", [-1.0, -2.0, 0.0], [3.0, 2.0, 0.0]), _param("p1", [-0.5] * 6, [1.5] * 6)]
    lb = np.concatenate([p.lb for p in params])
    ub = np.concatenate([p.ub for p in params])
    values = edge_param_values(params, seed=0)
    rows = _stack(values, params)

    assert _contains(rows, lb) and _contains(rows, ub)
    assert _contains(rows, 0.5 * (lb + ub)) and _contains(rows, np.zeros(9))
    assert np.all(rows >= lb) and np.all(rows <= ub)
    assert len(rows) == MAX_EDGE_SAMPLES == len(np.unique(rows, axis=0))
    assert [v["p0"].shape for v in values] == [(3,)] * len(values)

    # Beyond those four: vertices, and interior points pinned on a face.
    on_bound = (rows[4:] == lb) | (rows[4:] == ub)
    vertices = on_bound.all(axis=1)
    assert vertices.sum() == 2 and (~vertices).sum() == 2
    assert np.all(on_bound[~vertices].sum(axis=1) >= 9 // 4)

    again = _stack(edge_param_values(params, seed=0), params)
    assert np.array_equal(rows, again)


def test_zero_is_left_out_when_outside_the_box():
    params = [_param("p0", [1.0, -1.0], [2.0, 1.0])]
    rows = _stack(edge_param_values(params), params)
    assert not _contains(rows, np.zeros(2))
    assert np.all(rows >= params[0].lb) and np.all(rows <= params[0].ub)


def test_an_infinite_bound_stands_in_as_one():
    # The convention of the random samples: -1 / +1 in place of -inf / +inf.
    params = [_param("p0", [-np.inf, 0.0], [np.inf, np.inf])]
    rows = _stack(edge_param_values(params), params)
    assert np.array_equal(rows[:4], [[-1.0, 0.0], [1.0, 1.0], [0.0, 0.5], [0.0, 0.0]])
    assert np.all(np.isfinite(rows))


@pytest.mark.parametrize("dim", [1, 2, 4000])
def test_the_count_is_bounded_whatever_the_dimension(dim):
    params = [_param("p0", -np.ones(dim), np.ones(dim))]
    rows = _stack(edge_param_values(params), params)
    assert 3 <= len(rows) <= MAX_EDGE_SAMPLES
    assert len(np.unique(rows, axis=0)) == len(rows)


def test_a_degenerate_box_gives_no_repeated_sample():
    fixed = [_param("p0", [0.3, -1.0], [0.3, -1.0])]
    assert len(edge_param_values(fixed)) == 1
    # One coordinate: two vertices and the midpoint. The mixed vertices and the
    # face points (their one coordinate pinned) are those vertices again.
    scalar = [_param("p0", [1.0], [2.0])]
    assert np.array_equal(_stack(edge_param_values(scalar), scalar), [[1.0], [2.0], [1.5]])
    line = [_param("p0", [0.3, 1.0], [0.3, 2.0])]
    rows = _stack(edge_param_values(line), line)
    assert 3 <= len(rows) == len(np.unique(rows, axis=0))
    # A value the sample set already holds is not added again.
    kept = edge_param_values(fixed, existing=[{"p0": np.array([0.3, -1.0])}])
    assert kept == []
    assert edge_param_values([]) == []


# -- in the verify data ----------------------------------------------------------
def _boxqp(**config):
    case = build_boxqp()
    case["config"] = case["config"].model_copy(update=config)
    ir = extract_ir(case["verifier"], case["next_state"], config=case["config"], name="boxqp")
    return case, ir


def test_the_knob_restores_the_random_samples_exactly():
    case, ir = _boxqp(verify_edge_samples=False)
    off = EmbeddedCTarget(case["config"])._sample_params(ir)
    # What the check drew before the edge samples existed.
    rng = np.random.default_rng(case["config"].seed)
    p = ir.params[0]
    before = [rng.uniform(p.lb, p.ub) for _ in range(case["config"].num_verify_samples)]
    assert len(off) == 8
    assert all(np.array_equal(s["params"]["p0"], b) for s, b in zip(off, before))

    case, ir = _boxqp()
    assert case["config"].verify_edge_samples  # on by default
    on = EmbeddedCTarget(case["config"])._sample_params(ir)
    # The random samples keep their values and their indices; the edges follow.
    assert 8 < len(on) <= 8 + MAX_EDGE_SAMPLES
    for s, t in zip(off, on):
        assert np.array_equal(s["params"]["p0"], t["params"]["p0"])
        assert np.array_equal(s["solution"], t["solution"])
    rows = np.array([s["params"]["p0"] for s in on])
    assert _contains(rows[8:], p.lb) and _contains(rows[8:], p.ub)
    assert np.all(rows >= p.lb) and np.all(rows <= p.ub)

    header = EmbeddedCTarget(case["config"])._verify_data(ir)
    assert f"#define FOM_NUM_SAMPLES {len(on)}" in header
    assert f"FOM_VERIFY_EXPECTED[{len(on)}][FOM_STATE_DIM]" in header


def _generate(tmp_path, solver, **config):
    case, _ = _boxqp(**config)
    return generate_code(
        case["verifier"],
        case["next_state"],
        tmp_path,
        config=case["config"],
        driver=StubAgentDriver(solver),
        name="boxqp",
        **gate(case),
    )


def test_the_reported_count_and_the_manifest_include_the_edge_samples(tmp_path, requires_compiler):
    result = _generate(tmp_path, BOXQP_GOLDEN)
    assert result.success, result.translate.log
    case, ir = _boxqp()
    n = len(EmbeddedCTarget(case["config"])._sample_params(ir))
    assert n > 8
    assert f"worst max-abs error over {n} samples" in result.translate.log
    assert json.loads((tmp_path / MANIFEST_NAME).read_text())["verify"]["samples"] == n


def test_a_branch_broken_only_at_a_vertex_is_rejected(tmp_path, requires_compiler):
    # The box-QP golden without its lower clip. On this example no random
    # parameter drives a coordinate to its lower bound, so the eight interior
    # samples accept it; the all-upper-bound vertex does, and rejects it.
    golden = BOXQP_GOLDEN.read_text()
    assert LOWER_CLIP in golden
    broken = golden.replace(LOWER_CLIP, "")

    off = _generate(tmp_path / "off", broken, verify_edge_samples=False)
    assert off.success and "VERIFY OK" in off.translate.log, off.translate.log

    on = _generate(tmp_path / "on", broken)
    assert not on.success
    assert "samples exceed tol" in on.translate.log and "VERIFY OK" not in on.translate.log
    assert not (tmp_path / "on" / MANIFEST_NAME).exists()
    # Each failing sample is an edge sample (index 8 and up).
    failed = [int(ln.split()[1][:-1]) for ln in on.translate.log.splitlines() if "> tol" in ln]
    assert failed and min(failed) >= 8
