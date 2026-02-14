import marimo

__generated_with = "0.14.16"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


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
    In this tutorial, we illustrate the sparse coding example in the Section 4.1 of the paper: [Exact Verification of First-Order Methods via Mixed-Integer Linear Programming](https://arxiv.org/abs/2412.11330).

    Consider the following problem of recovering the signal $z^\star$: from its noisy measurement $x = D z^\star + \varepsilon \in {\bf R}^p$:

    \begin{array}{ll}
        \text{minimize} & (1/2) \lVert Dz - x \rVert_2^2 + \lambda \lVert z \rVert_1,
    \end{array}

    where $D \in {\bf R}^{p \times n}$ is the dictionary matrix, $\varepsilon \in {\bf R}^p$, $\lambda > 0$ is a sparsity-inducing regularization parameter, $z \in {\bf R}^n$ is the decision variable, and $x \in {\bf R}^p$ is the problem parameter.

    This problem can be solved with ISTA and its accelerated variant FISTA.
    We provide the definition of ISTA iteration below:

    \begin{array}{ll}
        s^{k+1} = {\cal T}_{\lambda \eta} ( (I - \eta D^T D) s^{k} + \eta D^T x),
    \end{array}

    where ${\cal T}_{\lambda}$ is a soft-thresholding operator applied element-wisely.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    For the strongly convex example, we choose $p=20$ and $n=15$, and for the nonstrongly convex example, we choose $p = 15$, $n = 20$, and $\lambda = 10^{-1}$.
    To construct $D$, we randomly sample from i.i.d. Gaussian ${\cal N}(0, 1/p)$ the $20$% nonzeros of $D$ and normalize its columns.
    The true signal $z^\star$ is constructed by sampling each entry from i.i.d. Gaussian ${\cal N}(0, 1)$ with probability $0.1$ and setting $0$ otherwise. We sample $m = 100$ of those random sparse signals $z^\star$.
    Finally, we let the parameter set $X$ to be the tightest hypercube containing all $x$'s.
    """
    )
    return


@app.cell
def _(np):
    np.random.seed(0)
    p, n, lambd = 15, 20, 1e-1
    D_value = np.random.normal(size=(p, n))
    D_mask = np.random.binomial(n=1, p=0.2, size=(p, n))
    D = np.multiply(D_value, D_mask)

    # signal z_star
    num_star = 100
    z_star_value = np.random.normal(size=(n, num_star))
    z_star_mask = np.random.binomial(n=1, p=0.1, size=(n, num_star))
    z_star = np.multiply(z_star_value, z_star_mask)

    # noise err
    err = np.random.normal(scale=1e-2, size=(p, num_star))

    # parameter x = D z_star + err
    x = D @ z_star + err

    # parameter set X = [x_lb, x_ub]
    x_lb, x_ub = np.min(x, axis=1), np.max(x, axis=1)
    return D, lambd, n, p, x_lb, x_ub


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    For the algorithm, we also choose $\eta = 1/L$, where $L$ is the maximum eigenvalue of $D^T D$.
    We initialize the algorithm with $S^0 = \{ s^0 \}$ where $s^0$ is the solution of unregularized problem ($\lambda = 0$) for a random sample.
    """
    )
    return


@app.cell
def _(D, np):
    L = np.real(np.max(np.linalg.eigvals(D.T @ D)))
    eta = 1 / L
    K = 5
    return K, eta


@app.cell
def _(D, K, Verifier, eta, lambd, n, np, p, x_lb, x_ub):
    # from fomtools.proxable_functions import NormL1
    # regularizer = lambd * NormL1
    # regularizer = NormL1(lambd=lambd)

    solver_params = {
        'OutputFlag': 0,
    }

    VP = Verifier(solver='highs', params=solver_params)
    x_param = VP.add_param(p, lb=x_lb, ub=x_ub)

    s0 = np.zeros(n) # initialized at zero
    s = [None for _ in range(K + 1)]
    s[0] = s0

    all_residuals = []
    for k in range(1, K+1):
        s[k] = VP.step.soft_threshold(s[k-1] - eta * D.T @ (D @ s[k-1] - x_param), lambd=lambd*eta)
        # s[k] = VP.prox(function=regularizer, rhs=s[k-1] - eta * D.T @ (D @ s[k-1] - x_param), gamma=eta)
        VP.set_infinity_norm_objective([s[k] - s[k-1]])
        res = VP.solve()
        all_residuals.append(res)
    return (all_residuals,)


@app.cell
def _(K, all_residuals, plt):
    plt.plot(range(1, K+1), all_residuals, label='VP')
    plt.suptitle('sparse coding example')
    plt.yscale('log')
    plt.ylabel('Worst-case fixed-point residual')
    plt.xlabel(r'$K$')
    plt.legend()
    return


if __name__ == "__main__":
    app.run()
