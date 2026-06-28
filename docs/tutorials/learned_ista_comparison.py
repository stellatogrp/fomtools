import marimo

__generated_with = "0.22.5"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    import jax
    import jax.numpy as jnp
    import matplotlib.pyplot as plt
    import numpy as np
    import optax

    from fomtools.verifier import Verifier

    jax.config.update("jax_enable_x64", True)
    np.set_printoptions(precision=5, suppress=True)

    return Verifier, jax, jnp, np, optax, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Learned ISTA for Lasso

    We consider the Lasso problem:

    $$\min_s \;\tfrac{1}{2}\|Ds - x\|^2 + \lambda\|s\|_1, \quad D \in \mathbb{R}^{m \times n},$$

    solved with ISTA using per-iteration step-sizes $\gamma_k$:

    $$s^{k+1} = \mathcal{T}_{\gamma_k\lambda}\!\left(s^k - \gamma_k D^\top(Ds^k - x)\right).$$

    **Vanilla ISTA** uses the fixed step $\gamma_k = 1/L$. **Learned ISTA** optimizes $\{\gamma_k\}$ via L2O to minimize
    the worst-case KKT residual $\|H_k(s^k - s^{k-1})\|$ where $H_k = \tfrac{1}{\gamma_k}I - D^\top D$.
    """)
    return


@app.cell
def _(np):
    np.random.seed(0)
    m, n, lambd = 10, 15, 1e-1
    s0 = np.zeros(n)

    D_value = np.random.normal(size=(m, n))
    D_mask = np.random.binomial(n=1, p=0.2, size=(m, n))
    D = np.multiply(D_value, D_mask)

    z_star = np.multiply(
        np.random.normal(size=(n, 500)),
        np.random.binomial(n=1, p=0.1, size=(n, 500)),
    )
    x_cloud = D @ z_star + np.random.normal(scale=1e-2, size=(m, 500))
    x_lb, x_ub = np.min(x_cloud, axis=1), np.max(x_cloud, axis=1)

    M = D.T @ D
    eig = np.linalg.eigvalsh(M)
    L = float(eig[-1])
    eta = 1.0 / L
    EYE = np.eye(n)

    def H_of(g):
        return (1.0 / g) * EYE - M

    N_box = 4000
    np.random.seed(1)
    X_box = np.random.uniform(x_lb, x_ub, size=(N_box, m))

    print(f"L = {L:.4f},  eta = 1/L = {eta:.4f}")

    return D, EYE, H_of, K, L, M, N_box, X_box, eta, lambd, m, n, s0, x_lb, x_ub


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We verify performance up to iteration budget $K$.
    """)
    return


@app.cell
def _():
    K = 10
    return (K,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Vanilla ISTA

    The vanilla schedule uses the constant step $\gamma_k = 1/L$ with coupled threshold $\theta_k = \gamma_k \lambda$.
    """)
    return


@app.cell
def _(K, eta, lambd, np):
    gammas_vanilla = np.full(K, eta)
    thetas_vanilla = lambd * gammas_vanilla
    return gammas_vanilla, thetas_vanilla


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## L2O: Learning Time-Varying Step-Sizes

    We learn $\gamma_k \in (0, \gamma_{\max})$ by minimizing the CVaR (worst 5%) of the squared KKT residuals
    over box-uniform samples of $x$, using Adam with gradient clipping.
    Allowing $\gamma_k > 2/L$ lets the optimizer discover long-step patterns unreachable by the fixed step.
    """)
    return


@app.cell
def _(D, K, L, M, N_box, X_box, eta, jax, jnp, lambd, n, np, optax):
    GAMMA_MAX = 10.0 / L
    _CVAR_Q = max(1, int(0.05 * N_box))

    _Dj = jnp.asarray(D)
    _Mj = jnp.asarray(M)
    _Ij = jnp.eye(n)
    _Xtr = jnp.asarray(X_box)

    def _soft(v, t):
        return jnp.sign(v) * jnp.maximum(jnp.abs(v) - t, 0.0)

    def _resid_perk(a, x):
        gam = GAMMA_MAX * jax.nn.sigmoid(a)
        s_prev = jnp.zeros(n)
        out = []
        for k in range(K):
            s = _soft(s_prev - gam[k] * (_Dj.T @ (_Dj @ s_prev - x)), lambd * gam[k])
            r = ((1.0 / gam[k]) * _Ij - _Mj) @ (s - s_prev)
            out.append(jnp.sum(r ** 2))
            s_prev = s
        return jnp.stack(out)

    def _loss(a, X):
        R_mat = jax.vmap(lambda x: _resid_perk(a, x))(X)
        per_k = jnp.mean(jnp.sort(R_mat, axis=0)[-_CVAR_Q:, :], axis=0)
        return jnp.sum(per_k)

    _a = jnp.full(K, float(np.log((eta / GAMMA_MAX) / (1 - eta / GAMMA_MAX))))
    _opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(5e-3))
    _opt_state = _opt.init(_a)
    _loss_grad = jax.jit(jax.value_and_grad(_loss))

    print("Training learned ISTA ...")
    for _step in range(6000):
        _loss_val, _grads = _loss_grad(_a, _Xtr)
        _updates, _opt_state = _opt.update(_grads, _opt_state)
        _a = optax.apply_updates(_a, _updates)
        if _step % 750 == 0 or _step == 5999:
            print(f"  step {_step:4d}  loss {float(_loss_val):.6f}")

    gammas_learned = np.asarray(GAMMA_MAX * jax.nn.sigmoid(_a))
    thetas_learned = lambd * gammas_learned
    print(f"  learned gamma_k (x L): {np.array2string(gammas_learned * L, precision=2)}")

    return gammas_learned, thetas_learned


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Simulation

    We simulate both schedules over $N$ box-uniform samples and report the sample-maximum KKT residual at each iteration.
    """)
    return


@app.cell
def _(D, H_of, K, Verifier, X_box, gammas_vanilla, m, np, s0, thetas_vanilla, x_lb, x_ub):
    _vp = Verifier(method="simulator")
    _x = _vp.add_param(ndim=m, lb=x_lb, ub=x_ub, samples=X_box)
    _s = [s0]
    for _k in range(1, K + 1):
        _g, _th = gammas_vanilla[_k - 1], thetas_vanilla[_k - 1]
        _sk = _vp.step.soft_threshold(_s[_k - 1] - _g * (D.T @ (D @ _s[_k - 1] - _x)), lambd=_th)
        _s.append(_sk)
        _vp.set_objective_norm([H_of(_g) @ (_sk - _s[_k - 1])], ord=np.inf)
    sam_v = np.asarray(_vp.solve()).max(axis=0)
    return (sam_v,)


@app.cell
def _(D, H_of, K, Verifier, X_box, gammas_learned, m, np, s0, thetas_learned, x_lb, x_ub):
    _vp = Verifier(method="simulator")
    _x = _vp.add_param(ndim=m, lb=x_lb, ub=x_ub, samples=X_box)
    _s = [s0]
    for _k in range(1, K + 1):
        _g, _th = gammas_learned[_k - 1], thetas_learned[_k - 1]
        _sk = _vp.step.soft_threshold(_s[_k - 1] - _g * (D.T @ (D @ _s[_k - 1] - _x)), lambd=_th)
        _s.append(_sk)
        _vp.set_objective_norm([H_of(_g) @ (_sk - _s[_k - 1])], ord=np.inf)
    sam_l = np.asarray(_vp.solve()).max(axis=0)
    return (sam_l,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Results

    The learned step-sizes yield a smaller sample-maximum KKT residual than vanilla ISTA.
    The bar chart shows the learned $\gamma_k$ relative to $1/L$; values above $2/L$ (dashed line) exceed the standard stability threshold.
    """)
    return


@app.cell
def _(K, L, gammas_learned, gammas_vanilla, np, plt, sam_l, sam_v):
    ks = np.arange(1, K + 1)
    BART_RED = (0.5020, 0.0000, 0.2510)

    _fig, _axes = plt.subplots(
        2, 2, figsize=(13, 8), sharex="col", sharey="row",
        gridspec_kw={"height_ratios": [2, 1]},
    )

    for _ax, _sam, _title in [
        (_axes[0, 0], sam_v, "ISTA"),
        (_axes[0, 1], sam_l, "Learned ISTA"),
    ]:
        _ax.plot(ks, _sam, "-", color="C3", lw=2, label="Sample Maximum")
        _ax.set_yscale("log")
        _ax.set_title(_title)
        _ax.grid(True, which="major", alpha=0.25)

    for _ax, _gammas in [(_axes[1, 0], gammas_vanilla), (_axes[1, 1], gammas_learned)]:
        _ax.bar(ks, _gammas * L, width=0.72, facecolor=(*BART_RED, 0.20), edgecolor=BART_RED, linewidth=2.0)
        _ax.axhline(2.0, color="0.35", ls="--", lw=2.0)
        _ax.set_xlabel(r"$k$")
        _ax.set_xticks(ks[1::2])
        _ax.grid(True, axis="y", alpha=0.25)

    _axes[0, 0].set_ylabel("Sample-max KKT residual")
    _axes[1, 0].set_ylabel(r"Step size ($\times L$)")
    _axes[0, 0].legend(fontsize=11)
    _fig.tight_layout()
    plt.show()
    return


if __name__ == "__main__":
    app.run()
