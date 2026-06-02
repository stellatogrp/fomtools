"""Serialize a :class:`~fomtools.codegen.ir.FomIR` to disk and back.

The IR is split into a human-readable ``spec.json`` (structure) plus a binary
``data.npz`` (all numeric arrays). The LLM coding agent reads ``spec.json`` (and
the rendered ``SPEC.md``) to understand the algorithm; the verify harness reads
``data.npz`` for the exact coefficients. Round-tripping is lossless.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from fomtools.codegen.config import TerminationKind, TerminationSpec
from fomtools.codegen.ir import (
    ActivationKind,
    ActivationOp,
    FomIR,
    IterationStep,
    LinearOp,
    MatrixData,
    ParamSpec,
    StateSpec,
    Term,
)


def _linop_to_dict(lin: LinearOp) -> dict:
    return {
        "out_dim": lin.out_dim,
        "terms": [{"source": t.source, "matrix": t.matrix} for t in lin.terms],
        "const": lin.const,
    }


def _activation_to_dict(act: ActivationOp) -> dict:
    return {
        "kind": act.kind.value,
        "lambd": act.lambd,
        "lower": act.lower,
        "upper": act.upper,
        "lower_scalar": act.lower_scalar,
        "upper_scalar": act.upper_scalar,
    }


def ir_to_dict(ir: FomIR) -> dict:
    """Structure-only dict (numeric arrays live in the companion npz)."""
    return {
        "name": ir.name,
        "dtype": ir.dtype,
        "termination": {
            "kind": ir.termination.kind.value,
            "max_iters": ir.termination.max_iters,
            "tol": ir.termination.tol,
        },
        "params": [{"name": p.name, "dim": p.dim} for p in ir.params],
        "state": {
            "dim": ir.state.dim,
            "slots": [list(s) for s in ir.state.slots],
        },
        "body": [
            {
                "name": s.name,
                "out_dim": s.out_dim,
                "linear_op": _linop_to_dict(s.linear_op),
                "activation": _activation_to_dict(s.activation),
            }
            for s in ir.body
        ],
        "state_update": _linop_to_dict(ir.state_update),
        "matrices": {
            name: {"shape": list(md.shape), "sparse": md.sparse_pattern is not None}
            for name, md in ir.matrices.items()
        },
    }


def save(ir: FomIR, directory: str | Path) -> tuple[Path, Path]:
    """Write ``spec.json`` + ``data.npz`` into ``directory``. Returns both paths."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    spec_path = directory / "spec.json"
    data_path = directory / "data.npz"

    spec_path.write_text(json.dumps(ir_to_dict(ir), indent=2))

    arrays: dict[str, np.ndarray] = {}
    for name, md in ir.matrices.items():
        arrays[f"mat::{name}"] = md.array
        if md.sparse_pattern is not None:
            rows, cols = md.sparse_pattern
            arrays[f"sp_rows::{name}"] = np.asarray(rows)
            arrays[f"sp_cols::{name}"] = np.asarray(cols)
    arrays["state_init"] = ir.state.init
    for p in ir.params:
        arrays[f"param_lb::{p.name}"] = p.lb
        arrays[f"param_ub::{p.name}"] = p.ub
    np.savez(data_path, **arrays)
    return spec_path, data_path


def load(directory: str | Path) -> FomIR:
    """Reconstruct a :class:`FomIR` from ``spec.json`` + ``data.npz``."""
    directory = Path(directory)
    spec = json.loads((directory / "spec.json").read_text())
    npz = np.load(directory / "data.npz")

    matrices: dict[str, MatrixData] = {}
    for name in spec["matrices"]:
        array = npz[f"mat::{name}"]
        pattern = None
        if spec["matrices"][name]["sparse"]:
            pattern = (npz[f"sp_rows::{name}"], npz[f"sp_cols::{name}"])
        matrices[name] = MatrixData(name=name, array=array, sparse_pattern=pattern)

    params = tuple(
        ParamSpec(
            name=p["name"],
            dim=p["dim"],
            lb=npz[f"param_lb::{p['name']}"],
            ub=npz[f"param_ub::{p['name']}"],
        )
        for p in spec["params"]
    )

    state = StateSpec(
        dim=spec["state"]["dim"],
        slots=tuple(tuple(s) for s in spec["state"]["slots"]),
        init=npz["state_init"],
    )

    body = tuple(_dict_to_step(s) for s in spec["body"])
    state_update = _dict_to_linop(spec["state_update"])
    term = TerminationSpec(
        kind=TerminationKind(spec["termination"]["kind"]),
        max_iters=spec["termination"]["max_iters"],
        tol=spec["termination"]["tol"],
    )

    return FomIR(
        params=params,
        state=state,
        body=body,
        state_update=state_update,
        termination=term,
        matrices=matrices,
        dtype=spec["dtype"],
        name=spec["name"],
    )


def _dict_to_linop(d: dict) -> LinearOp:
    return LinearOp(
        out_dim=d["out_dim"],
        terms=tuple(Term(source=t["source"], matrix=t["matrix"]) for t in d["terms"]),
        const=d["const"],
    )


def _dict_to_step(d: dict) -> IterationStep:
    a = d["activation"]
    act = ActivationOp(
        kind=ActivationKind(a["kind"]),
        lambd=a["lambd"],
        lower=a["lower"],
        upper=a["upper"],
        lower_scalar=a["lower_scalar"],
        upper_scalar=a["upper_scalar"],
    )
    return IterationStep(
        name=d["name"],
        out_dim=d["out_dim"],
        linear_op=_dict_to_linop(d["linear_op"]),
        activation=act,
    )


def render_markdown(ir: FomIR) -> str:
    """Render a human-readable specification for the coding agent."""
    lines = [
        f"# FOM specification: `{ir.name}`",
        "",
        "One iteration of a first-order method, to be implemented as a fixed loop.",
        "",
        f"- dtype: `{ir.dtype}`",
        f"- state dimension: {ir.state.dim}",
        f"- termination: {ir.termination.kind.value} "
        f"(max_iters={ir.termination.max_iters}, tol={ir.termination.tol})",
        "",
        "## Parameters (supplied at solve time)",
    ]
    for p in ir.params:
        lines.append(f"- `{p.name}`: dimension {p.dim}")
    if not ir.params:
        lines.append("- (none)")

    lines += ["", "## State layout", ""]
    for slot_name, dim in ir.state.slots:
        lines.append(f"- `{slot_name}`: dimension {dim}")

    lines += ["", "## Iteration body", ""]
    lines.append("Each step computes `tmp = activation(linear_op)`:")
    lines.append("")
    for step in ir.body:
        lines.append(f"### `{step.name}` (dim {step.out_dim})")
        lines.append(f"- linear_op = {_linop_str(step.linear_op)}")
        lines.append(f"- activation = {_activation_str(step.activation)}")
        lines.append("")

    lines.append("## State update")
    lines.append(f"- `state <- {_linop_str(ir.state_update)}`")
    lines.append("")
    lines.append("Residual = `max_abs(new_state - state)`.")
    lines.append("")
    lines.append("All matrices/consts are provided numerically in `data.npz` / `fom_data.c`.")
    return "\n".join(lines)


def _linop_str(lin: LinearOp) -> str:
    parts = []
    if lin.const is not None:
        parts.append(lin.const)
    for t in lin.terms:
        parts.append(f"{t.matrix} @ {t.source}")
    body = " + ".join(parts) if parts else "0"
    return body


def _activation_str(act: ActivationOp) -> str:
    if act.kind == ActivationKind.SOFT_THRESHOLD:
        return f"soft_threshold(., lambda={act.lambd})"
    if act.kind == ActivationKind.SATURATED_LINEAR:
        return f"clip(., {act.lower_scalar}, {act.upper_scalar})"
    return act.kind.value
