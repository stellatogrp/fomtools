"""The codegen tutorials show recorded timings unless the reader asks for a live run."""

import json

import pytest

from docs.tutorials.helper_functions import codegen_benchmarks as bench

RECORDED = {
    "python_us": 100.0,
    "c_initial_us": 2.0,
    "c_final_us": 1.0,
    "codegen_speedup": 50.0,
    "iteration_speedup": 2.0,
    "total_speedup": 100.0,
}


@pytest.fixture
def timing_file(tmp_path):
    path = tmp_path / "timings.json"
    path.write_text(json.dumps({"examples": {"lasso": RECORDED, "nnls": RECORDED}}))
    return path


@pytest.fixture
def no_agent(monkeypatch):
    """Fail the test if a live run would build an agent."""

    def fail(*args, **kwargs):
        raise AssertionError("live=False must not build a coding agent")

    monkeypatch.setattr(bench, "get_default_driver", fail)
    monkeypatch.setattr(bench, "find_compiler", fail)


def test_lasso_benchmark_recorded_without_agent(timing_file, no_agent):
    out = bench.run_lasso_codegen_benchmark(
        config=None,
        live=False,
        next_state=None,
        out_dir=None,
        python_us=0.0,
        state=None,
        timing_file=timing_file,
        verifier=None,
        x_param=None,
    )
    assert out["live"] is False
    assert out["total_speedup"] == RECORDED["total_speedup"]


def test_benchmark_recorded_without_agent(timing_file, no_agent):
    out = bench.run_codegen_benchmark(
        config=None,
        live=False,
        example="nnls",
        name="nnls",
        next_state=None,
        out_dir=None,
        params=None,
        python_us=0.0,
        state=None,
        timing_file=timing_file,
        verifier=None,
    )
    assert out["live"] is False
    assert out["codegen_us"] == RECORDED["c_initial_us"]
