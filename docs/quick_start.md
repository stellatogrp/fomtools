# Quick start

The guide below walks through a verification example with FOMTools. It covers linear expressions, steps, proxable
functions, objectives, and the fixed-point encoding. For a more detailed breakdown of FOMTools'
capabilities, check out the [User guide](user_guide/index.md).

## The problem: verifying FISTA

Suppose you want to verify the performance of FISTA on a LASSO problem

\[
\min_{s}\; \frac{1}{2}\lVert Ds - x \rVert_2^2 + \lambda \lVert s \rVert_1
\]

where $D$ is a fixed data matrix and $x$ is a parameter that is only known to lie in some
uncertainty set, for instance sensor noise or a range of possible measurements. Starting from
an iterate $s^0$, FISTA repeats a gradient step on the smooth part followed by a
soft-thresholding step for the $\ell_1$-norm, plus a momentum term that mixes the current and
previous iterate.

<span id="fista-updates"></span>

\[
\text{Initialize:}\qquad w^0 = s^0, \qquad \beta_0 = 1.
\]

\[
\begin{aligned}
s^{k+1}
&= \operatorname{prox}_{\eta\lambda\lVert \cdot \rVert_1}
\!\left(w^k - \eta D^\top(Dw^k - x)\right), \\[0.8em]
\beta_{k+1}
&= \frac{1 + \left(1 + 4\beta_k^2\right)^{1/2}}{2}, \\[0.8em]
w^{k+1}
&= s^{k+1} + \frac{\beta_k - 1}{\beta_{k+1}}\left(s^{k+1} - s^k\right).
\end{aligned}
\]

With FOMTools, we can verify the worst-case value of a residual such as
$\|s^{k+1} - s^k\|$ after $K$ iterations of FISTA, over all admissible $x$. FOMTools lets you
write the FISTA iteration almost exactly as you would write a concrete implementation, then
maximize the residual over the declared parameter set. The rest of this guide builds up that
code.

In this example, FOMTools sets up the following worst-case residual problem:

\[
\begin{aligned}
\underset{x,\,s}{\operatorname{maximize}}\quad
& \lVert s^K - s^{K-1} \rVert_\infty \\
\text{subject to}\quad
& x \in [0, 1]^m, \qquad s^0 = 0, \\
& s^{k+1} = \operatorname{FISTA}(s^k), \qquad k = 0, \ldots, K - 1.
\end{aligned}
\]

The sections below unpack this example.

```python
import fomtools as ft
import numpy as np

# problem parameters and setup
lambd, eta, K = 0.1, 0.01, 10
m, n = 10, 15
D = np.random.normal(size=(m, n))
VP = ft.Verifier(method="mip", solver="gurobi")

# initialize VP parameters + iterates
x = VP.add_param(m, lb=0, ub=1)
s = [None for _ in range(K + 1)]
s[0] = VP.add_iterate(n, lb=0, ub=0)
wk = s[0]
beta_k = 1
results = []

for k in range(K):
    # core algorithm logic
    grad = D.T @ (D @ wk - x)
    s[k + 1] = VP.step.soft_threshold(wk - eta * grad, lambd=lambd * eta)
    beta_kplus1 = 0.5 * (1 + np.sqrt(1 + 4 * beta_k ** 2))
    w_kplus1 = s[k + 1] + (beta_k - 1) / beta_kplus1 * (s[k + 1] - s[k])

    # solving the VP
    VP.set_objective_norm(s[k + 1] - s[k], np.inf)
    results.append(VP.solve())

    # setting up for next loop iteration
    wk, beta_k = w_kplus1, beta_kplus1
```

## [Setting up the problem](api/verifier.md)

`VP.add_param(m, lb=0, ub=1)` creates the parameter $x$ and its uncertainty set. `VP.add_iterate(n, lb=0, ub=0)` creates
the first iterate $s^0$ and its uncertainty set. Here the equal lower and upper bounds fix $s^0=0$. The list
`s` stores every FISTA iterate, so the residual between consecutive iterates can later be used
as an objective. Meanwhile, `wk` and `beta_k` are ordinary Python variables for the
extrapolated point $w^k$ and momentum coefficient used in the FISTA step.

```python
# initialize VP parameters + iterates
x = VP.add_param(m, lb=0, ub=1)
s = [None for _ in range(K + 1)]
s[0] = VP.add_iterate(n, lb=0, ub=0)
wk = s[0]
beta_k = 1
results = []
```

## [Choosing a backend](api/verifier.md)

The verifier is created with the MIP backend in the complete example:

```python
VP = ft.Verifier(method="mip", solver="gurobi")
```

`method="mip"` selects the exact mixed-integer encoding and `solver="gurobi"` chooses the
solver used to optimize it. MIP encodes the verification problem exactly and returns the true
worst-case value. Change `method` to `"sdp"` for a semidefinite relaxation or to `"simulator"` for a fast empirical analysis without a formal
worst-case guarantee. The subsequent algorithm code and objective are unchanged when you switch backends.

## [Steps and Proxable functions](api/steps.md)

One FISTA iteration consists of the gradient, the soft-thresholding step, and the momentum
update:

```python
grad = D.T @ (D @ wk - x)
s[k + 1] = VP.step.soft_threshold(wk - eta * grad, lambd=lambd * eta)
beta_kplus1 = 0.5 * (1 + np.sqrt(1 + 4 * beta_k ** 2))
w_kplus1 = s[k + 1] + (beta_k - 1) / beta_kplus1 * (s[k + 1] - s[k])
```
Simply write out these steps as in the [original FISTA updates](#fista-updates). Here $\eta$ is the gradient step size. The `soft_threshold` call in the FISTA update is a specialized proximal operator: `lambd=lambd * eta` supplies the threshold used by soft-thresholding, and the method returns a
new expression for `s[k + 1]`. This is a convenient spelling of the proximal operator of
$\lambda\lVert\cdot\rVert_1$. FOMTools also supports many other convex functions. To explore the other native steps of FOMTools, check out the [User guide](user_guide/index.md).


## [Objectives](api/verifier.md#fomtools.Verifier.set_objective_norm)

After each update, the example asks for the worst-case fixed-point residual and solves the
resulting verification problem:

```python
VP.set_objective_norm(s[k + 1] - s[k], np.inf)
results.append(VP.solve())
```

`s[k + 1] - s[k]` is the symbolic residual to maximize, and `np.inf` tells
`set_objective_norm` to use its $\ell_\infty$ norm. You can also use `1` for the $\ell_1$ norm
or `2` for the squared $\ell_2$ norm. `VP.solve()` then creates and solves the backend-specific
optimization problem, returning the worst-case value.


For a distance-to-solution metric $\lVert s^{k+1} - s^\star \rVert_2^2$, use the fixed point
of FISTA. At a fixed point, the momentum term vanishes, so it is encoded by the following
proximal-gradient map:

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

This creates the fixed point $s^\star$ of the operator $T$, satisfying
$s^\star = T(s^\star, x)$.
