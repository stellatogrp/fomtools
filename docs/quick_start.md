# Quick start

The guide below walks through a verification example with FOMTools. It covers linear expressions, steps, proxable
functions, objectives, and the fixed-point encoding. For a more detailed breakdown of FOMTools'
capabilities, check out the [User guide](user_guide/index.md).

## The problem: verifying FISTA

Suppose you want to verify the performance of FISTA on a LASSO problem

$$
\begin{array}{ll}
\text{minimize} & \frac{1}{2}\|Ds - x\|_2^2 + \lambda \|s\|_1,
\end{array}
$$

where $D$ is a fixed data matrix and $x$ is a parameter that is only known to lie in some
uncertainty set, for instance sensor noise or a range of possible measurements. Starting from
an iterate $s^0$, FISTA repeats a gradient step on the smooth part followed by a
soft-thresholding step for the $\ell_1$-norm, plus a momentum term that mixes the current and
previous iterate. Its momentum follows the FISTA schedule $t_k$:

<span id="fista-updates"></span>

$$
w^0 = s^0, \qquad t_0 = 1,
$$

$$
\begin{array}{rcl}
s^{k+1} &=& \mathbf{prox}_{\eta\lambda\|\cdot\|_1}\left(w^k - \eta D^T(Dw^k - x)\right), \\
t_{k+1} &=& \frac{1}{2}\left(1 + \sqrt{1 + 4t_k^2}\right), \\
w^{k+1} &=& s^{k+1} + \frac{t_k - 1}{t_{k+1}}\left(s^{k+1} - s^k\right),
\end{array}
$$

for $k = 0, 1, \dots$, where $\eta$ is the step size.

With FOMTools, we can verify the worst-case value of a residual such as
$\|s^{k+1} - s^k\|_\infty$ after $K$ iterations of FISTA, over all admissible $x$. FOMTools lets you
write the FISTA iteration almost exactly as you would write a concrete implementation, then
maximize the residual over the declared parameter set. The rest of this guide builds up that
code.

In this example, FOMTools sets up the following worst-case residual problem:

$$
\begin{array}{ll}
\text{maximize} & \|s^K - s^{K-1}\|_\infty \\
\text{subject to} & x \in [0, 1]^m, \quad s^0 = 0, \\
& (s^{k+1}, w^{k+1}, t_{k+1}) \text{ given by the FISTA update}, \quad k = 0, \dots, K - 1,
\end{array}
$$

with variables $x$ and $s^0, \dots, s^K$.

The sections below unpack this example.

```python
import numpy as np
import fomtools as ft

# problem data
m, n, lambd, K = 5, 8, 0.1, 5
D = np.random.default_rng(0).normal(size=(m, n))
eta = 1 / np.linalg.norm(D, 2) ** 2  # step size 1 / L
vp = ft.Verifier()  # MIP with HiGHS by default

# parameter and initial iterate
x = vp.parameter(m, lb=0, ub=1)
s = w = vp.iterate(n, lb=0, ub=0)
t = 1.0
results = []

for k in range(K):
    # FISTA update
    s_prev, s = s, vp.soft_threshold(w - eta * D.T @ (D @ w - x), lambd=lambd * eta)
    t_next = 0.5 * (1 + np.sqrt(1 + 4 * t**2))
    w, t = s + (t - 1) / t_next * (s - s_prev), t_next

    # worst-case residual after k + 1 iterations
    vp.maximize(ft.norm(s - s_prev, np.inf))
    results.append(vp.solve())
```

## Setting up the problem

`vp.parameter(m, lb=0, ub=1)` creates the parameter $x$ and its uncertainty set $[0, 1]^m$.
`vp.iterate(n, lb=0, ub=0)` creates the initial iterate $s^0$ and its uncertainty set. Here the
equal lower and upper bounds fix $s^0 = 0$, and $w^0 = s^0$. Scalar bounds apply to every
entry; pass arrays for entry-wise bounds. See the [`Verifier` API](api/verifier.md) for all the
options.

```python
x = vp.parameter(m, lb=0, ub=1)
s = w = vp.iterate(n, lb=0, ub=0)
t = 1.0
```

`s`, `w`, and `t` are ordinary Python variables. The `Verifier` records every step, so the
loop only keeps the iterates the next update and the objective need: $s^{k+1}$, $s^k$, and
$w^k$. The momentum coefficient $t_k$ is a plain number. Keep the iterates in a list only when
you need the whole trajectory, for example for [code generation](user_guide/index.md#code-generation).

## Choosing a backend

The complete example uses the default MIP backend:

```python
vp = ft.Verifier()  # same as ft.Verifier(method="mip", solver="highs")
```

`method="mip"` selects the exact mixed-integer encoding, solved with the open-source HiGHS
solver by default. If you have a Gurobi license, pass `solver="gurobi"` to use Gurobi instead
(see [Get started](get_started.md#optional-dependencies)). MIP encodes the verification problem exactly and returns the true
worst-case value. Change `method` to `"simulator"` for a fast empirical analysis without a formal
worst-case guarantee, or to `"sdp"` for an [experimental semidefinite relaxation](user_guide/sdp.md). The subsequent algorithm code and objective are unchanged when you switch backends.
The solvers are silent by default; pass `verbose=True` to `ft.Verifier` or to `vp.solve` to see their logs.

## Steps and proxable functions

One FISTA iteration consists of the gradient step, the soft-thresholding step, and the momentum
update:

```python
s_prev, s = s, vp.soft_threshold(w - eta * D.T @ (D @ w - x), lambd=lambd * eta)
t_next = 0.5 * (1 + np.sqrt(1 + 4 * t**2))
w, t = s + (t - 1) / t_next * (s - s_prev), t_next
```

Simply write out these steps as in the [original FISTA updates](#fista-updates). Here $\eta$ is the gradient step size.
`vp.soft_threshold` is the proximal operator of $\lambda\eta\|\cdot\|_1$: `lambd=lambd * eta` supplies the
threshold, and the method returns a new expression for $s^{k+1}$. Affine operations such as the
gradient step and the momentum update need no special call. FOMTools also supports many other
convex functions. To explore the other native steps of FOMTools, check out the [User guide](user_guide/index.md) and the [step API](api/steps.md).

## Objectives

After each update, the example asks for the worst-case fixed-point residual and solves the
resulting verification problem:

```python
vp.maximize(ft.norm(s - s_prev, np.inf))
results.append(vp.solve())
```

`s - s_prev` is the symbolic residual $s^{k+1} - s^k$, and `ft.norm(..., np.inf)` its
$\ell_\infty$ norm. Use `ft.norm(..., 1)` for the $\ell_1$ norm, or `ft.sum_squares(...)` for the
squared $\ell_2$ norm, which makes the MIP quadratic and needs `solver="gurobi"`. `vp.solve()` creates and solves the backend-specific optimization problem
and returns the worst-case value. After a MIP solve, `x.value` is the worst-case parameter.

Solving at every iteration gives the worst-case residual as a function of $k$, and it makes the
later solves easier. Each solve warm-starts the next one. After the solve at iteration $k$,
FOMTools also chains the certified bounds $r_j$ on $\|s^j - s^{j-1}\|_\infty$ through the
triangle inequality,

$$
\|s^k - s^0\|_\infty \leq \sum_{j=1}^k r_j,
$$

and tightens the bounds of $s^k$ with it before the next step is added. The worst case is
unchanged, but the mixed-integer problems at larger $k$ get smaller big-M constants. This happens
whenever the objective is `ft.norm(a - b, np.inf)`; pass `tighten_residuals=False` to
`ft.Verifier` to turn it off. If you only need the bound at $K$, set the objective and solve once
after the loop.

For a distance-to-solution metric $\|s^{k+1} - s^\star\|_\infty$, use the fixed point
of FISTA. At a fixed point, the momentum term vanishes, so it is encoded by the following
proximal-gradient map
$$
T(s, x) = \mathbf{prox}_{\eta\lambda\|\cdot\|_1}\left(s - \eta D^T(Ds - x)\right):
$$

```python
s_star = vp.fixed_point(
    n_dim=n,
    operator=lambda s: vp.soft_threshold(s - eta * D.T @ (D @ s - x), lambd=lambd * eta),
    lb=-100,
    ub=100,
)
```

Then use it in the objective:

```python
vp.maximize(ft.norm(s - s_star, np.inf))
```

This creates a new iterate $s^\star$ constrained to satisfy $s^\star = T(s^\star, x)$.
