"""Benchmark helpers used by the code generation tutorials."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from fomtools.codegen.agent import make_agent_driver
from fomtools.codegen.harness import find_compiler


def run_lasso_codegen_benchmark(
    *,
    config: Any,
    next_state: Any,
    out_dir: Path,
    python_us: float,
    state: Any,
    timing_file: Path,
    verifier: Any,
    x_param: Any,
) -> dict[str, Any]:
    """Generate and time the Lasso solver, or return the published timings.

    A live code generation run requires a configured coding agent and a C compiler.
    Documentation builds use the recorded benchmark row instead.
    """
    if os.environ.get("FOMTOOLS_CODEGEN_AGENT") != "1":
        recorded = json.loads(timing_file.read_text())["examples"]["lasso"]
        return {
            "python_us": recorded["python_us"],
            "codegen_us": recorded["c_initial_us"],
            "iteration_us": recorded["c_final_us"],
            "codegen_speedup": recorded["codegen_speedup"],
            "iteration_speedup": recorded["iteration_speedup"],
            "total_speedup": recorded["total_speedup"],
            "solver_path": None,
            "live": False,
        }

    if find_compiler() is None:
        raise RuntimeError("No C compiler found. Install cc, gcc, or clang to run codegen timing.")

    result = verifier.codegen_rolled(
        out_dir,
        next_state=next_state,
        config=config,
        driver=make_agent_driver(config=config),
        state=[state],
        params=[x_param],
        name="lasso_codegen_speedup",
        code_optimization=True,
        profile=True,
    )
    if not result.translate.success:
        raise RuntimeError(result.translate.log[-4000:])

    if (
        result.timing is None
        or result.timing.c_initial_us is None
        or result.timing.c_final_us is None
    ):
        raise RuntimeError("Code generation finished without the requested timing report.")

    codegen_us = result.timing.c_initial_us
    iteration_us = result.timing.c_final_us
    return {
        "python_us": python_us,
        "codegen_us": codegen_us,
        "iteration_us": iteration_us,
        "codegen_speedup": python_us / codegen_us,
        "iteration_speedup": codegen_us / iteration_us,
        "total_speedup": python_us / iteration_us,
        "solver_path": out_dir / result.target.solver_source_path(),
        "live": True,
    }


def run_codegen_benchmark(
    *,
    config: Any,
    example: str,
    name: str,
    next_state: Any,
    out_dir: Path,
    params: list[Any] | None,
    python_us: float,
    state: Any,
    timing_file: Path,
    verifier: Any,
) -> dict[str, Any]:
    """Generate and time a tutorial solver, or return its published timings."""
    if os.environ.get("FOMTOOLS_CODEGEN_AGENT") != "1":
        recorded = json.loads(timing_file.read_text())["examples"][example]
        return {
            "python_us": recorded["python_us"],
            "codegen_us": recorded["c_initial_us"],
            "iteration_us": recorded["c_final_us"],
            "codegen_speedup": recorded["codegen_speedup"],
            "iteration_speedup": recorded["iteration_speedup"],
            "total_speedup": recorded["total_speedup"],
            "solver_path": None,
            "live": False,
        }

    if find_compiler() is None:
        raise RuntimeError("No C compiler found. Install cc, gcc, or clang to run codegen timing.")

    kwargs = {} if params is None else {"params": params}
    result = verifier.codegen_rolled(
        out_dir,
        next_state=next_state,
        config=config,
        driver=make_agent_driver(config=config),
        state=state,
        name=name,
        code_optimization=True,
        profile=True,
        **kwargs,
    )
    if not result.translate.success:
        raise RuntimeError(result.translate.log[-4000:])

    if (
        result.timing is None
        or result.timing.c_initial_us is None
        or result.timing.c_final_us is None
    ):
        raise RuntimeError("Code generation finished without the requested timing report.")

    codegen_us = result.timing.c_initial_us
    iteration_us = result.timing.c_final_us
    return {
        "python_us": python_us,
        "codegen_us": codegen_us,
        "iteration_us": iteration_us,
        "codegen_speedup": python_us / codegen_us,
        "iteration_speedup": codegen_us / iteration_us,
        "total_speedup": python_us / iteration_us,
        "solver_path": out_dir / result.target.solver_source_path(),
        "live": True,
    }
