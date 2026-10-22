# Constraints

Constraints are equalities and inequalities over linear expressions, used to
encode assumptions on the problem data and the initial conditions.

::: fomtools.constraint.Constraint
::: fomtools.constraint.EqualityConstraint
::: fomtools.constraint.InequalityConstraint

## Objective atoms

`ft.norm` and `ft.sum_squares` build the objective passed to
[`Verifier.maximize`](verifier.md#fomtools.Verifier.maximize), for example
`vp.maximize(ft.norm(z - z_prev, np.inf))`.

::: fomtools.constraint.norm
::: fomtools.constraint.sum_squares
