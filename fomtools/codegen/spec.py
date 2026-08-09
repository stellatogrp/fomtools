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
from fomtools.codegen.factorize import CSC, Factorization
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
        "terms": [{"source": t.source, "matrix": t.matrix, "scale": t.scale} for t in lin.terms],
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


def _step_to_dict(s: IterationStep) -> dict:
    return {
        "name": s.name,
        "out_dim": s.out_dim,
        "linear_op": _linop_to_dict(s.linear_op),
        "activation": _activation_to_dict(s.activation),
        "solve": s.solve,
    }


def _fac_meta(fac: Factorization) -> dict:
    meta = {"method": fac.method, "dim": fac.dim}
    if fac.method == "cg":
        meta["cg_tol"] = fac.cg_tol
        meta["cg_max_iter"] = fac.cg_max_iter
    return meta


def ir_to_dict(ir: FomIR) -> dict:
    """Structure-only dict (numeric arrays live in the companion npz)."""
    return {
        "name": ir.name,
        "dtype": ir.dtype,
        "factorizations": {name: _fac_meta(fac) for name, fac in ir.factorizations.items()},
        "termination": {
            "kind": ir.termination.kind.value,
            "max_iters": ir.termination.max_iters,
            "tol": ir.termination.tol,
            "check_termination": ir.termination.check_termination,
        },
        "params": [{"name": p.name, "dim": p.dim} for p in ir.params],
        "state": {
            "dim": ir.state.dim,
            "slots": [list(s) for s in ir.state.slots],
        },
        "body": [_step_to_dict(s) for s in ir.body],
        "state_update": _linop_to_dict(ir.state_update),
        "matrices": {
            name: {"shape": list(md.shape), "sparse": md.sparse_pattern is not None}
            for name, md in ir.matrices.items()
        },
        "schedules": {name: {"len": int(len(values))} for name, values in ir.schedules.items()},
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
    for name, values in ir.schedules.items():
        arrays[f"sched::{name}"] = np.asarray(values, dtype=float)
    arrays["state_init"] = ir.state.init
    for p in ir.params:
        arrays[f"param_lb::{p.name}"] = p.lb
        arrays[f"param_ub::{p.name}"] = p.ub
    for fname, fac in ir.factorizations.items():
        arrays[f"fac::{fname}::perm"] = fac.perm
        if fac.method == "cg":
            _save_csc(arrays, f"fac::{fname}::A", fac.A)
            arrays[f"fac::{fname}::minv"] = fac.minv
            continue
        _save_csc(arrays, f"fac::{fname}::L", fac.L)
        if fac.method == "ldlt":
            arrays[f"fac::{fname}::dinv"] = fac.dinv
        else:
            _save_csc(arrays, f"fac::{fname}::U", fac.U)
            arrays[f"fac::{fname}::uinv"] = fac.uinv
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
        check_termination=spec["termination"].get("check_termination", 1),
    )

    factorizations: dict[str, Factorization] = {}
    for fname, meta in spec.get("factorizations", {}).items():
        perm = npz[f"fac::{fname}::perm"]
        if meta["method"] == "cg":
            factorizations[fname] = Factorization(
                method="cg",
                dim=meta["dim"],
                perm=perm,
                A=_load_csc(npz, f"fac::{fname}::A"),
                minv=npz[f"fac::{fname}::minv"],
                cg_tol=meta["cg_tol"],
                cg_max_iter=meta["cg_max_iter"],
            )
            continue
        L = _load_csc(npz, f"fac::{fname}::L")
        if meta["method"] == "ldlt":
            factorizations[fname] = Factorization(
                method="ldlt",
                dim=meta["dim"],
                perm=perm,
                L=L,
                dinv=npz[f"fac::{fname}::dinv"],
            )
        else:
            factorizations[fname] = Factorization(
                method="lu",
                dim=meta["dim"],
                perm=perm,
                L=L,
                U=_load_csc(npz, f"fac::{fname}::U"),
                uinv=npz[f"fac::{fname}::uinv"],
            )

    return FomIR(
        params=params,
        state=state,
        body=body,
        state_update=state_update,
        termination=term,
        matrices=matrices,
        factorizations=factorizations,
        schedules={name: npz[f"sched::{name}"] for name in spec.get("schedules", {})},
        dtype=spec["dtype"],
        name=spec["name"],
    )


def _save_csc(arrays: dict, prefix: str, csc: CSC) -> None:
    arrays[f"{prefix}::indptr"] = csc.indptr
    arrays[f"{prefix}::indices"] = csc.indices
    arrays[f"{prefix}::data"] = csc.data
    arrays[f"{prefix}::shape"] = np.array([csc.n_rows, csc.n_cols])


def _load_csc(npz, prefix: str) -> CSC:
    shape = npz[f"{prefix}::shape"]
    return CSC(
        n_rows=int(shape[0]),
        n_cols=int(shape[1]),
        indptr=npz[f"{prefix}::indptr"],
        indices=npz[f"{prefix}::indices"],
        data=npz[f"{prefix}::data"],
    )


def _dict_to_linop(d: dict) -> LinearOp:
    return LinearOp(
        out_dim=d["out_dim"],
        terms=tuple(
            Term(source=t["source"], matrix=t["matrix"], scale=t.get("scale")) for t in d["terms"]
        ),
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
        solve=d.get("solve"),
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
        f"(max_iters={ir.termination.max_iters}, tol={ir.termination.tol}, "
        f"check_termination={ir.termination.check_termination})",
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

    if ir.schedules:
        lines += ["", "## Iteration-varying coefficients", ""]
        lines.append("Some terms below are scaled by a coefficient that changes with the")
        lines.append("iteration. Each is tabulated in `fom_data.c`, one value per iteration,")
        lines.append("and read through the accessor declared in `fom_data.h`:")
        lines.append("")
        for name, values in sorted(ir.schedules.items()):
            head = ", ".join(f"{v:.6g}" for v in values[:4])
            lines.append(f"- `fom_sched_{name}(it)` — {len(values)} values, starting {head}, ...")
        lines.append("")
        lines.append("`it` is the **0-based** loop counter. Always call the accessor rather")
        lines.append("than indexing `FOM_SCHED_*` directly: the loop bound is the runtime")
        lines.append("`work->settings.max_iter`, which a caller may raise above the table")
        lines.append("length, and the accessor clamps. Hoist the value into a local at the")
        lines.append("top of each iteration; it does not change within one iteration.")

    lines += ["", "## Iteration body", ""]
    lines.append("Each step computes `tmp = activation(linear_op)`, except affine")
    lines.append("steps which *solve* `M tmp = linear_op`. Always call the provided")
    lines.append("`fom_lin_solve_<name>` routine in `fom_linsolve.c`; never form or use")
    lines.append("a matrix inverse. The routine uses either a precomputed sparse direct")
    lines.append("factorization (LDL^T / LU) or, for CG, preconditioned conjugate")
    lines.append("gradient (sparse matrix-vector products, no factorization) — the")
    lines.append("calling convention is identical either way.")
    lines.append("")
    lines.append("Compute every `matrix @ source` via the accumulating helpers in")
    lines.append("`fom_ops.c`: zero the output, then call `fom_apply_<matrix>(source, out)`")
    lines.append("for each term (these are sparse CSR where the matrix is structured, so")
    lines.append("do not write your own dense matrix-vector loops):")
    lines.append("")
    for step in ir.body:
        lines.append(f"### `{step.name}` (dim {step.out_dim})")
        lines.append(f"- linear_op (rhs) = {_linop_str(step.linear_op)}")
        if step.solve is not None:
            fac = ir.factorizations[step.solve]
            kind = (
                "CG, preconditioned conjugate gradient"
                if fac.method == "cg"
                else f"{fac.method.upper()} factorization"
            )
            lines.append(
                f"- solve: `{step.name} = fom_lin_solve_{step.solve}(rhs)` "
                f"({kind}, dim {fac.dim}, nnz {fac.nnz})"
            )
        else:
            lines.append(f"- activation = {_activation_str(step.activation)}")
        lines.append("")

    lines.append("## State update")
    lines.append(f"- `state <- {_linop_str(ir.state_update)}`")
    lines.append("")
    lines.append("Residual = `max_abs(new_state - state)`.")
    lines.append("")
    if ir.termination.kind == TerminationKind.INF_NORM_CHANGE and ir.termination.check_termination:
        lines.append(
            "Termination: evaluate the residual and exit early (status `FOM_SOLVED`) only "
            "on iterations where `check > 0 && (it + 1) % check == 0` and "
            "`residual <= work->settings.tol`, with `check = work->settings.check_termination`; "
            "on other iterations skip the residual entirely (plain state copy). The check "
            "interval is a RUNTIME setting: initialize `settings->check_termination` to "
            "`FOM_CHECK_TERMINATION` in `fom_set_default_settings`, then read it from "
            "`work->settings.check_termination` in the loop (guarding `% check` with "
            "`check > 0`) so callers can tune it without regenerating. `FOM_USE_TOL`, "
            "`FOM_CHECK_TERMINATION`, and `FOM_TOL` are defined in `fom_workspace.h`."
        )
        lines.append("")
    lines.append("All matrices/consts are provided numerically in `data.npz` / `fom_data.c`.")
    return "\n".join(lines)


def _linop_str(lin: LinearOp) -> str:
    parts = []
    if lin.const is not None:
        parts.append(lin.const)
    for t in lin.terms:
        if t.scale is None:
            parts.append(f"{t.matrix} @ {t.source}")
        else:
            parts.append(f"fom_sched_{t.scale}(it) * ({t.matrix} @ {t.source})")
    body = " + ".join(parts) if parts else "0"
    return body


def _activation_str(act: ActivationOp) -> str:
    if act.kind == ActivationKind.SOFT_THRESHOLD:
        return f"soft_threshold(., lambda={act.lambd})"
    if act.kind == ActivationKind.SATURATED_LINEAR:
        lo = act.lower if act.lower is not None else act.lower_scalar
        hi = act.upper if act.upper is not None else act.upper_scalar
        return f"clip(., lower={lo}, upper={hi})  (elementwise; bounds are vectors)"
    return act.kind.value
