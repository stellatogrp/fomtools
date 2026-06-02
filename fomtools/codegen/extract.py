"""Extract a :class:`~fomtools.codegen.ir.FomIR` from a built ``Verifier``.

The verifier records an *unrolled* chain of steps (one block per iteration).
Code generation needs the opposite: a single, stationary iteration *body* that a
generated loop can repeat.  We recover it from the dependency cone of a
user-supplied ``next_state`` expression — the iterate after exactly one
iteration — which works uniformly whether an iteration is a single step (ISTA)
or several (Douglas–Rachford's affine-solve + projection).

Leaf roles are resolved structurally: ``Verifier.add_iterate`` appends to
``verifier.iterates`` while ``Verifier.add_param`` does not, so any external leaf
in ``verifier.iterates`` is *state* and any other external leaf is a *parameter*.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as spa

from fomtools.codegen.config import CodegenConfig
from fomtools.codegen.ir import (
    ActivationKind,
    ActivationOp,
    FomIR,
    IterationStep,
    LinearOp,
    MatrixData,
    NonStationaryFOMError,
    ParamSpec,
    StateSpec,
    Term,
)
from fomtools.linearexpression import LinearExpression
from fomtools.steps import Affine, Relu, SaturatedLinear, SoftThreshold


def _densify(coeff) -> np.ndarray:
    """Return a 2-D dense float array for a (possibly sparse) coefficient."""
    if spa.issparse(coeff):
        return np.asarray(coeff.todense(), dtype=float)
    return np.atleast_2d(np.asarray(coeff, dtype=float))


def _row_slice_expr(expr: LinearExpression, rows) -> LinearExpression:
    """Return a new ``LinearExpression`` keeping only the given output rows.

    ``LinearExpression.__getitem__`` does not accept a list of indices, so we
    build the sliced expression directly (mirroring its slice handling).
    """
    rows = list(rows)
    new_dd = {}
    for leaf, coeff in expr.decomposition_dict.items():
        if spa.issparse(coeff):
            new_dd[leaf] = coeff.tocsr()[rows]
        else:
            new_dd[leaf] = np.atleast_2d(np.asarray(coeff)[rows])
    return LinearExpression(
        decomposition_dict=new_dd, const=np.atleast_1d(np.asarray(expr.const)[rows])
    )


def eval_expr(expr: LinearExpression, values: dict[LinearExpression, np.ndarray]) -> np.ndarray:
    """Numerically evaluate ``expr`` given values for every leaf it references.

    ``out = expr.const + sum_leaf coeff_leaf @ values[leaf]``.
    """
    out = np.array(expr.const, dtype=float).reshape(-1).copy()
    for leaf, coeff in expr.decomposition_dict.items():
        out = out + np.asarray(coeff @ values[leaf]).reshape(-1)
    return out


def fresh_leaf_of(step) -> LinearExpression:
    """The single fresh leaf a step introduces."""
    leaves = step.lhs_leaves
    if len(leaves) != 1:
        raise NonStationaryFOMError(
            f"{type(step).__name__} introduces {len(leaves)} leaves; "
            "code generation supports exactly one fresh leaf per step."
        )
    return leaves[0]


def _minv_dense(lhs_mat) -> np.ndarray:
    mat = (
        np.asarray(lhs_mat.todense(), dtype=float)
        if spa.issparse(lhs_mat)
        else np.asarray(lhs_mat, dtype=float)
    )
    return np.linalg.inv(mat)


def _step_source_activation(step) -> tuple[LinearExpression, ActivationOp, int]:
    """Reduce a step to (linear source expression, elementwise activation, out_dim).

    The affine operator's inverse is folded into the linear coefficients, so its
    activation is the identity (no division survives).
    """
    if isinstance(step, Affine):
        minv = _minv_dense(step.lhs_mat)
        source = minv @ step.rhs
        return source, ActivationOp(ActivationKind.IDENTITY), source.ndim
    if isinstance(step, SoftThreshold):
        return (
            step.rhs,
            ActivationOp(ActivationKind.SOFT_THRESHOLD, lambd=float(step.lambd)),
            step.rhs.ndim,
        )
    if isinstance(step, Relu):
        proj = sorted(step.proj_indices)
        if proj == list(range(step.rhs.ndim)):
            return step.rhs, ActivationOp(ActivationKind.RELU), step.rhs.ndim
        return _row_slice_expr(step.rhs, proj), ActivationOp(ActivationKind.RELU), len(proj)
    if isinstance(step, SaturatedLinear):
        lower, upper = _satlin_bounds(step)
        return (
            step.rhs,
            ActivationOp(
                ActivationKind.SATURATED_LINEAR,
                lower_scalar=lower,
                upper_scalar=upper,
            ),
            step.rhs.ndim,
        )
    raise NonStationaryFOMError(f"Unsupported step type for code generation: {type(step).__name__}")


def _satlin_bounds(step: SaturatedLinear) -> tuple[float, float]:
    """Constant clip bounds for a saturated-linear step (MVP: scalars only)."""
    if isinstance(step.l, LinearExpression) or isinstance(step.u, LinearExpression):
        raise NonStationaryFOMError(
            "saturated_linear with parameter-dependent (LinearExpression) bounds "
            "is not yet supported by code generation."
        )
    lower = np.asarray(step.l, dtype=float)
    upper = np.asarray(step.u, dtype=float)
    if lower.size > 1 or upper.size > 1:
        raise NonStationaryFOMError(
            "saturated_linear with vector clip bounds is not yet supported; use scalar bounds."
        )
    return float(lower.reshape(-1)[0]), float(upper.reshape(-1)[0])


def _eval_step_fresh(step, values: dict[LinearExpression, np.ndarray]) -> np.ndarray:
    """Faithfully evaluate a step's fresh-leaf value (independent of the IR fold).

    Used by the oracle trust-check: the affine case solves the *original* system
    rather than reusing the precomputed inverse.
    """
    rhs_val = eval_expr(step.rhs, values)
    if isinstance(step, Affine):
        mat = (
            np.asarray(step.lhs_mat.todense(), dtype=float)
            if spa.issparse(step.lhs_mat)
            else np.asarray(step.lhs_mat, dtype=float)
        )
        return np.linalg.solve(mat, rhs_val)
    if isinstance(step, SoftThreshold):
        return np.sign(rhs_val) * np.maximum(np.abs(rhs_val) - step.lambd, 0.0)
    if isinstance(step, Relu):
        proj = sorted(step.proj_indices)
        if proj == list(range(step.rhs.ndim)):
            return np.maximum(0.0, rhs_val)
        return np.maximum(0.0, rhs_val[proj])
    if isinstance(step, SaturatedLinear):
        lower, upper = _satlin_bounds(step)
        return np.minimum(np.maximum(rhs_val, lower), upper)
    raise NonStationaryFOMError(f"Unsupported step type: {type(step).__name__}")


def _dependency_cone(steps, next_state_leaves: set[LinearExpression]):
    """Return the body steps that ``next_state`` transitively depends on.

    Walks backward from the ``next_state`` leaves through each producing step's
    rhs, collecting every step needed to compute one iteration. Returns the
    steps ordered by their position in ``steps``.
    """
    producer: dict[LinearExpression, object] = {}
    for step in steps:
        producer[fresh_leaf_of(step)] = step

    needed: set[int] = set()
    step_index = {id(step): i for i, step in enumerate(steps)}
    worklist = list(next_state_leaves)
    seen_leaves: set[LinearExpression] = set()
    while worklist:
        leaf = worklist.pop()
        if leaf in seen_leaves:
            continue
        seen_leaves.add(leaf)
        step = producer.get(leaf)
        if step is None:
            continue  # external leaf (state or param)
        needed.add(step_index[id(step)])
        worklist.extend(step.rhs.decomposition_dict.keys())

    return [steps[i] for i in sorted(needed)]


def _as_list(x) -> list:
    if x is None:
        return []
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x]


def resolve_structure(verifier, next_state, state=None, params=None):
    """Resolve the iteration body, state leaves, and parameter leaves.

    Shared by :func:`extract_ir` and the reference oracle so both agree on what
    constitutes one iteration. Returns ``(next_state_exprs, body_steps,
    body_fresh, state_leaves, param_leaves)``.
    """
    next_state_exprs = _as_list(next_state)
    if not next_state_exprs:
        raise ValueError("next_state must be provided (the iterate after one iteration).")

    body_steps = _dependency_cone(verifier.steps, _next_state_leaves(next_state_exprs))
    if not body_steps:
        raise NonStationaryFOMError(
            "next_state does not depend on any recorded step; nothing to generate."
        )

    body_fresh = [fresh_leaf_of(s) for s in body_steps]
    body_fresh_set = set(body_fresh)

    external: list[LinearExpression] = []
    seen: set[LinearExpression] = set()
    for step in body_steps:
        for leaf in step.rhs.decomposition_dict.keys():
            if leaf not in body_fresh_set and leaf not in seen:
                seen.add(leaf)
                external.append(leaf)
    for expr in next_state_exprs:
        for leaf in expr.decomposition_dict.keys():
            if leaf not in body_fresh_set and leaf not in seen:
                seen.add(leaf)
                external.append(leaf)

    iterate_set = set(verifier.iterates)
    if state is None:
        state_leaves = [leaf for leaf in external if leaf in iterate_set]
    else:
        state_leaves = _as_list(state)
    state_set = set(state_leaves)
    if params is None:
        param_leaves = [leaf for leaf in external if leaf not in state_set]
    else:
        param_leaves = _as_list(params)

    _validate_classification(external, state_set, set(param_leaves))
    return next_state_exprs, body_steps, body_fresh, state_leaves, param_leaves


def extract_ir(
    verifier,
    next_state,
    state=None,
    params=None,
    config: CodegenConfig | None = None,
    name: str = "fom",
) -> FomIR:
    """Build a :class:`FomIR` for one iteration of the FOM recorded in ``verifier``.

    Parameters
    ----------
    verifier:
        A ``Verifier`` whose ``step`` API was used to build at least one
        iteration of the algorithm.
    next_state:
        The iterate ``LinearExpression`` (or list thereof) produced after exactly
        one iteration. Its concatenation defines the loop-carried state update.
    state:
        Optional explicit list of initial-state leaves. When ``None``, inferred
        as the external leaves that appear in ``verifier.iterates``.
    params:
        Optional explicit list of parameter leaves. When ``None``, inferred as
        the external leaves not in ``verifier.iterates``.
    config:
        Code-generation configuration (supplies the termination spec / dtype).
    name:
        Human-friendly FOM name used for file naming.

    Returns
    -------
    FomIR
        The single-iteration intermediate representation.
    """
    config = config or CodegenConfig()
    next_state_exprs, body_steps, body_fresh, state_leaves, param_leaves = resolve_structure(
        verifier, next_state, state, params
    )

    # --- symbolic environment -------------------------------------------------
    state_layout = []  # (leaf, slot_name, dim, offset)
    offset = 0
    for j, leaf in enumerate(state_leaves):
        state_layout.append((leaf, f"s{j}", leaf.ndim, offset))
        offset += leaf.ndim
    state_dim = offset

    param_map = {leaf: (f"p{i}", leaf.ndim) for i, leaf in enumerate(param_leaves)}

    matrices: dict[str, MatrixData] = {}
    body: list[IterationStep] = []
    tmp_map: dict[LinearExpression, tuple[str, int]] = {}

    for i, step in enumerate(body_steps):
        source, activation, out_dim = _step_source_activation(step)
        prefix = f"t{i}"
        lin = _build_linear_op(source, prefix, state_layout, param_map, tmp_map, matrices, out_dim)
        # populate activation clip-bound matrices if needed (scalars handled inline)
        body.append(
            IterationStep(name=prefix, out_dim=out_dim, linear_op=lin, activation=activation)
        )
        tmp_map[body_fresh[i]] = (prefix, out_dim)

    # --- state update ---------------------------------------------------------
    state_update = _build_state_update(
        next_state_exprs, state_layout, param_map, tmp_map, matrices, state_dim
    )
    if state_update.out_dim != state_dim:
        raise NonStationaryFOMError(
            f"next_state dimension ({state_update.out_dim}) does not match the "
            f"inferred state dimension ({state_dim})."
        )

    # --- initial state + param specs -----------------------------------------
    init = _initial_state(state_layout)
    state_spec = StateSpec(
        dim=state_dim,
        slots=tuple((slot_name, dim) for (_, slot_name, dim, _) in state_layout),
        init=init,
    )
    param_specs = tuple(
        ParamSpec(
            name=param_map[leaf][0],
            dim=leaf.ndim,
            lb=np.asarray(leaf.lb, dtype=float).reshape(-1),
            ub=np.asarray(leaf.ub, dtype=float).reshape(-1),
        )
        for leaf in param_leaves
    )

    return FomIR(
        params=param_specs,
        state=state_spec,
        body=tuple(body),
        state_update=state_update,
        termination=config.termination,
        matrices=matrices,
        dtype=config.dtype,
        name=name,
    )


def _next_state_leaves(next_state_exprs) -> set[LinearExpression]:
    leaves: set[LinearExpression] = set()
    for expr in next_state_exprs:
        leaves.update(expr.decomposition_dict.keys())
    return leaves


def _validate_classification(external, state_set, param_set) -> None:
    for leaf in external:
        in_state = leaf in state_set
        in_param = leaf in param_set
        if in_state and in_param:
            raise NonStationaryFOMError("A leaf was classified as both state and parameter.")
        if not in_state and not in_param:
            raise NonStationaryFOMError(
                "Could not classify a leaf as state or parameter. Pass explicit "
                "state=/params= to extract_ir()."
            )


def _initial_state(state_layout) -> np.ndarray:
    init = np.zeros(sum(dim for (_, _, dim, _) in state_layout))
    pos = 0
    for leaf, _, dim, _ in state_layout:
        lb = np.asarray(leaf.lb, dtype=float).reshape(-1)
        ub = np.asarray(leaf.ub, dtype=float).reshape(-1)
        # When the verification fixes the initial iterate (lb == ub finite), use
        # it as the starting point; otherwise default to zero.
        fixed = np.isfinite(lb) & np.isfinite(ub) & np.isclose(lb, ub)
        init[pos : pos + dim] = np.where(fixed, lb, 0.0)
        pos += dim
    return init


def _build_linear_op(expr, prefix, state_layout, param_map, tmp_map, matrices, out_dim) -> LinearOp:
    terms: list[Term] = []

    # state block: horizontal concat over state slots (zero-fill missing leaves)
    state_blocks = []
    any_state = False
    for leaf, _, dim, _ in state_layout:
        coeff = expr.decomposition_dict.get(leaf)
        if coeff is None:
            state_blocks.append(np.zeros((out_dim, dim)))
        else:
            state_blocks.append(_densify(coeff))
            any_state = True
    if any_state:
        mat = np.hstack(state_blocks) if state_blocks else np.zeros((out_dim, 0))
        mname = f"{prefix}_state"
        matrices[mname] = MatrixData(name=mname, array=mat)
        terms.append(Term(source="state", matrix=mname))

    # parameter blocks
    for leaf, (pname, _) in param_map.items():
        coeff = expr.decomposition_dict.get(leaf)
        if coeff is None:
            continue
        dense = _densify(coeff)
        if not np.any(dense):
            continue
        mname = f"{prefix}_p_{pname}"
        matrices[mname] = MatrixData.from_coeff(mname, coeff)
        terms.append(Term(source=f"param:{pname}", matrix=mname))

    # temporary blocks (earlier body steps)
    for leaf, (tname, _) in tmp_map.items():
        coeff = expr.decomposition_dict.get(leaf)
        if coeff is None:
            continue
        dense = _densify(coeff)
        if not np.any(dense):
            continue
        mname = f"{prefix}_t_{tname}"
        matrices[mname] = MatrixData.from_coeff(mname, coeff)
        terms.append(Term(source=f"tmp:{tname}", matrix=mname))

    const_name = None
    const_vec = np.asarray(expr.const, dtype=float).reshape(-1)
    if np.any(const_vec):
        const_name = f"{prefix}_c"
        matrices[const_name] = MatrixData(name=const_name, array=const_vec)

    return LinearOp(out_dim=out_dim, terms=tuple(terms), const=const_name)


def _build_state_update(
    next_state_exprs, state_layout, param_map, tmp_map, matrices, state_dim
) -> LinearOp:
    """Concatenate the next_state expressions into one linear op over the
    current state + temporaries (+ params, for completeness)."""
    # Stack the per-expression coefficient blocks vertically.
    total = sum(e.ndim for e in next_state_exprs)
    terms: list[Term] = []

    # state
    state_blocks = []
    any_state = False
    for leaf, _, dim, _ in state_layout:
        rows = []
        for e in next_state_exprs:
            coeff = e.decomposition_dict.get(leaf)
            rows.append(_densify(coeff) if coeff is not None else np.zeros((e.ndim, dim)))
            if coeff is not None:
                any_state = True
        state_blocks.append(np.vstack(rows))
    if any_state:
        mat = np.hstack(state_blocks) if state_blocks else np.zeros((total, 0))
        matrices["su_state"] = MatrixData(name="su_state", array=mat)
        terms.append(Term(source="state", matrix="su_state"))

    # tmps
    for leaf, (tname, tdim) in tmp_map.items():
        rows = []
        present = False
        for e in next_state_exprs:
            coeff = e.decomposition_dict.get(leaf)
            if coeff is not None:
                rows.append(_densify(coeff))
                present = True
            else:
                rows.append(np.zeros((e.ndim, tdim)))
        if present:
            mat = np.vstack(rows)
            mname = f"su_t_{tname}"
            matrices[mname] = MatrixData(name=mname, array=mat)
            terms.append(Term(source=f"tmp:{tname}", matrix=mname))

    # params (rare in a state update but supported)
    for leaf, (pname, pdim) in param_map.items():
        rows = []
        present = False
        for e in next_state_exprs:
            coeff = e.decomposition_dict.get(leaf)
            if coeff is not None:
                rows.append(_densify(coeff))
                present = True
            else:
                rows.append(np.zeros((e.ndim, pdim)))
        if present:
            mat = np.vstack(rows)
            mname = f"su_p_{pname}"
            matrices[mname] = MatrixData(name=mname, array=mat)
            terms.append(Term(source=f"param:{pname}", matrix=mname))

    const_parts = [np.asarray(e.const, dtype=float).reshape(-1) for e in next_state_exprs]
    const_vec = np.concatenate(const_parts) if const_parts else np.zeros(0)
    const_name = None
    if np.any(const_vec):
        const_name = "su_c"
        matrices[const_name] = MatrixData(name=const_name, array=const_vec)

    return LinearOp(out_dim=total, terms=tuple(terms), const=const_name)
