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
import fomtools as ft
import numpy as np

# problem parameters and setup
lambd, eta, K = 0.1, 0.01, 10
m, n = 10, 15
D = np.random.normal(size=(m, n))
VP = ft.Verifier(method="mip")  # HiGHS by default

# initialize VP parameters + iterates
x = VP.add_param(m, lb=0, ub=1)
s = [None for _ in range(K + 1)]
s[0] = VP.add_iterate(n, lb=0, ub=0)
wk = s[0]
t_k = 1
results = []

for k in range(K):
    # core algorithm logic
    grad = D.T @ (D @ wk - x)
    s[k + 1] = VP.step.soft_threshold(wk - eta * grad, lambd=lambd * eta)
    t_kplus1 = 0.5 * (1 + np.sqrt(1 + 4 * t_k ** 2))
    w_kplus1 = s[k + 1] + (t_k - 1) / t_kplus1 * (s[k + 1] - s[k])

    # solving the VP
    VP.set_objective_norm(s[k + 1] - s[k], np.inf)
    results.append(VP.solve())

    # setting up for next loop iteration
    wk, t_k = w_kplus1, t_kplus1
```

## Setting up the problem

`VP.add_param(m, lb=0, ub=1)` creates the parameter $x$ and its uncertainty set. `VP.add_iterate(n, lb=0, ub=0)` creates
the first iterate $s^0$ and its uncertainty set. Here the equal lower and upper bounds fix $s^0=0$. The list
`s` stores every FISTA iterate, so the residual between consecutive iterates can later be used
as an objective. Meanwhile, `wk` and `t_k` are ordinary Python variables for the
extrapolated point $w^k$ and the momentum coefficient $t_k$. See the
[`Verifier` API](api/verifier.md) for all the ways to declare parameters and iterates.

```python
# initialize VP parameters + iterates
x = VP.add_param(m, lb=0, ub=1)
s = [None for _ in range(K + 1)]
s[0] = VP.add_iterate(n, lb=0, ub=0)
wk = s[0]
t_k = 1
results = []
```

## Choosing a backend

The verifier is created with the MIP backend in the complete example:

```python
VP = ft.Verifier(method="mip")
```

`method="mip"` selects the exact mixed-integer encoding, solved with the open-source HiGHS
solver by default. If you have a Gurobi license, pass `solver="gurobi"` to use Gurobi instead
(see [Get started](get_started.md#optional-dependencies)). MIP encodes the verification problem exactly and returns the true
worst-case value. Change `method` to `"simulator"` for a fast empirical analysis without a formal
worst-case guarantee, or to `"sdp"` for an [experimental semidefinite relaxation](user_guide/sdp.md). The subsequent algorithm code and objective are unchanged when you switch backends.

## Steps and proxable functions

One FISTA iteration consists of the gradient, the soft-thresholding step, and the momentum
update:

```python
grad = D.T @ (D @ wk - x)
s[k + 1] = VP.step.soft_threshold(wk - eta * grad, lambd=lambd * eta)
t_kplus1 = 0.5 * (1 + np.sqrt(1 + 4 * t_k ** 2))
w_kplus1 = s[k + 1] + (t_k - 1) / t_kplus1 * (s[k + 1] - s[k])
```
Simply write out these steps as in the [original FISTA updates](#fista-updates). Here $\eta$ is the gradient step size. The `soft_threshold` call in the FISTA update is a specialized proximal operator: `lambd=lambd * eta` supplies the threshold used by soft-thresholding, and the method returns a
new expression for `s[k + 1]`. This is a convenient spelling of the proximal operator of
$\lambda\|\cdot\|_1$. FOMTools also supports many other convex functions. To explore the other native steps of FOMTools, check out the [User guide](user_guide/index.md) and the [step API](api/steps.md).

## Objectives

After each update, the example asks for the worst-case fixed-point residual and solves the
resulting verification problem:

```python
VP.set_objective_norm(s[k + 1] - s[k], np.inf)
results.append(VP.solve())
```

`s[k + 1] - s[k]` is the symbolic residual to maximize, and `np.inf` tells
[`set_objective_norm`](api/verifier.md#fomtools.Verifier.set_objective_norm) to use its $\ell_\infty$ norm. You can also use `1` for the $\ell_1$ norm
or `2` for the squared $\ell_2$ norm. `VP.solve()` then creates and solves the backend-specific
optimization problem, returning the worst-case value.

For a distance-to-solution metric $\|s^{k+1} - s^\star\|_2^2$, use the fixed point
of FISTA. At a fixed point, the momentum term vanishes, so it is encoded by the following
proximal-gradient map
$$
T(s, x) = \mathbf{prox}_{\eta\lambda\|\cdot\|_1}\left(s - \eta D^T(Ds - x)\right):

```python
s_star = VP.fixed_point(
    n_dim=n,
    operator=lambda s: VP.step.soft_threshold(
        s - eta * D.T @ (D @ s - x), lambd=lambd * eta),
    lb=-100,
    ub=100,
)
```

Then use it in the objective:

```python
VP.set_objective_norm(s[k + 1] - s_star, 2)
```

This creates a new iterate $s^\star$ constrained to satisfy $s^\star = T(s^\star, x)$.
