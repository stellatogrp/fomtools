import marimo

__generated_with = "0.14.16"
app = marimo.App()


@app.cell
def _():
    import matplotlib.pyplot as plt
    import numpy as np

    from fomtools.verifier import Verifier

    np.set_printoptions(precision=5)  # Print few decimal places
    np.set_printoptions(suppress=True)  # Suppress scientific notation
    return Verifier, np, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    In this tutorial we show how to use our package to verify projected gradient descent for the nonnegative least-squares problem:

    \begin{array}{ll}
        \text{minimize} & (1/2) \lVert Az - x \rVert^2 \\ \text{subject to} & z \ge 0,
    \end{array}


    where $A \in {\bf R}^{m \times n}$, $m \ge n$, $z \in {\bf R}^n$ is the decision variable, and $x \in {\bf R}^m$ parameterizes the family.

    For the first order method, we choose projected gradient descent with step size $\eta > 0$, with iterates given by

    \begin{array}{ll}
        z^{k+1} = {\bf ReLU}(z^{k} - \eta A^T(Az^{k} - x)),
    \end{array}

    where $\bf ReLU(v) = \max\{v, 0\}$ element-wise.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    In this example, we choose $m = 5, n = 3$, randomly sample $A$, and parameterize $x$ as a unit box.
    We also choose $\eta = 0.5/L$, where $L$ is the maximum eigenvalue of $A^T A$.
    We initialize the algorithm with $z^0 = 0$.
    """
    )
    return


@app.cell
def _(np):
    np.random.seed(0)
    m, n = 5, 3
    A = np.random.normal(size=(m, n))
    x_lb, x_ub = 10, 11
    L = np.max(np.linalg.eigvalsh(A.T @ A))  # symmetric routine → real eigenvalues
    eta = 0.5 / L
    K = 10
    return A, K, eta, m, n, x_lb, x_ub


@app.cell
def _(A, K, Verifier, eta, m, n, x_lb, x_ub):
    solver_params = {
        "OutputFlag": 0,
    }

    VP = Verifier(params=solver_params)
    x_param = VP.add_param(m, lb=x_lb, ub=x_ub)

    # `iterate_trajectory` returns a fixed-length, list-like container of
    # iterates. It behaves like the old `z = [None for _ in range(K + 1)]`
    # idiom, but chooses the right representation per backend: on MIP/simulator
    # it collapses affine updates (fewer variables), while on SDP it keeps each
    # iterate as a separate leaf so the moment matrix stays sparse. Here every
    # update is a ReLU (already a fresh variable), so the policy has no effect.
    z = VP.iterate_trajectory(K + 1)  # collapse_expr="auto"
    z[0] = VP.add_iterate(n, lb=1, ub=1)

    all_residuals = []
    for k in range(1, K + 1):
        z[k] = VP.step.relu(z[k - 1] - eta * A.T @ (A @ z[k - 1] - x_param))
        VP.set_infinity_norm_objective([z[k] - z[k - 1]])
        res = VP.solve()
        all_residuals.append(res)
    return (all_residuals,)


@app.cell
def _(K, all_residuals, plt):
    plt.plot(range(1, K + 1), all_residuals, label="VP")
    plt.yscale("log")
    plt.ylabel("Worst-case fixed-point residual")
    plt.xlabel(r"$K$")
    plt.legend()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Backend-aware collapsing with `iterate_trajectory`

    Writing an affine update such as `z[k] = M @ z[k-1] + c` produces a
    *collapsed* linear expression: it references existing variables and mints no
    new one. This is ideal for the MIP and simulator backends. For the **SDP**
    backend, however, collapsing couples many variables in a single dense moment
    matrix; keeping each iterate as its own leaf (linked by an equality) yields a
    much sparser matrix that decomposes into many small PSD cones and is often
    substantially faster to solve.

    `verifier.iterate_trajectory(K + 1, collapse_expr="auto")` picks the right
    behavior automatically: collapse on MIP/simulator, keep a separate leaf on
    SDP. Pass `collapse_expr=True`/`False` (on the trajectory or per slot via
    `z.assign(k, expr, collapse_expr=...)`) to force it. Note the policy only
    applies to *collapsed* expressions — a value that is already a fresh variable
    (a `relu`/`satlin`/`soft_threshold` output on MIP, or a `relu` on SDP) is
    stored as-is regardless.

    The cell below illustrates the SDP effect on a soft-threshold (ISTA) chain,
    where the collapsed and separated formulations are genuinely different SDP
    relaxations.
    """
    )
    return


@app.cell
def _(np):
    import time

    from fomtools.verifier import Verifier as _V

    def _sdp_ista_bound(collapse_expr, K=2, seed=0):
        m2, n2 = 4, 3
        t, lambd = 0.05, 0.1
        np.random.seed(seed)
        A2 = np.random.normal(size=(m2, n2))
        vp = _V(method="sdp", solver="clarabel", obbt_solver=None)
        xp = vp.add_param(m2, lb=-1, ub=1)
        zt = vp.iterate_trajectory(K + 1, collapse_expr=collapse_expr)
        zt[0] = vp.add_iterate(n2, lb=-0.5, ub=0.5)
        for kk in range(1, K + 1):
            zt[kk] = vp.step.soft_threshold(
                zt[kk - 1] - t * A2.T @ (A2 @ zt[kk - 1] - xp), lambd=lambd * t
            )
        vp.set_objective_norm(zt[K] - zt[K - 1], 2)
        t0 = time.perf_counter()
        bound = vp.solve()
        return bound, time.perf_counter() - t0

    try:
        b_collapse, s_collapse = _sdp_ista_bound(True)
        b_auto, s_auto = _sdp_ista_bound("auto")  # -> no-collapse on SDP
        print(f"SDP collapsed    : bound={b_collapse:.4f}  solve={s_collapse:.3f}s")
        print(f"SDP no-collapse  : bound={b_auto:.4f}  solve={s_auto:.3f}s")
        print("no-collapse is a sparser (>=) relaxation; timing favors it as K/n grow")
    except Exception as exc:  # missing SDP solver in this environment
        print(f"(skipping SDP comparison: {exc})")
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
