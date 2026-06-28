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
    # Learned FISTA for Lasso

    We consider the Lasso problem:

    $$\min_s \;\tfrac{1}{2}\|Ds - x\|^2 + \lambda\|s\|_1, \quad D \in \mathbb{R}^{m \times n},$$

    solved with FISTA using per-iteration step-sizes $\gamma_k$ and momentum parameters $\beta_k$:

    $$s^k = \mathcal{T}_{\gamma_k\lambda}\!\left(y^{k-1} - \gamma_k D^\top(Dy^{k-1} - x)\right), \qquad y^k = s^k + \beta_k(s^k - s^{k-1}).$$

    **Vanilla FISTA** uses $\gamma_k = 1/L$ and the standard Beck–Teboulle $t_k$ momentum schedule.
    **Learned FISTA** jointly optimizes $\{\gamma_k, \beta_k\}$ via L2O to minimize
    the worst-case KKT residual $\|H_k(s^k - y^{k-1})\|$ where $H_k = \tfrac{1}{\gamma_k}I - D^\top D$.
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

    return D, EYE, H_of, L, M, N_box, X_box, eta, lambd, m, n, s0, x_lb, x_ub


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
    ## Vanilla FISTA

    The vanilla schedule uses $\gamma_k = 1/L$ and the Beck–Teboulle $t_k$ momentum sequence.
    """)
    return


@app.cell
def _(K, eta, lambd, np):
    gammas_vanilla = np.full(K, eta)
    thetas_vanilla = lambd * gammas_vanilla

    _t = np.ones(K + 1)
    for _k in range(1, K):
        _t[_k + 1] = (1.0 + np.sqrt(1.0 + 4.0 * _t[_k] ** 2)) / 2.0
    betas_vanilla = np.array([(_t[k] - 1.0) / _t[k + 1] for k in range(K)])

    return betas_vanilla, gammas_vanilla, thetas_vanilla


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## L2O: Jointly Learning Step-Sizes and Momentum

    We learn $2K$ parameters — $\gamma_k \in (0, \gamma_{\max})$ and $\beta_k \in (0, \beta_{\max})$ — by minimizing the CVaR (worst 5%)
    of the squared KKT residuals over box-uniform samples of $x$, using Adam.
    Allowing $\gamma_k > 2/L$ and $\beta_k > 1$ lets the optimizer discover aggressive momentum and long-step patterns unreachable by vanilla FISTA.
    """)
    return


@app.cell
def _(D, K, L, M, N_box, X_box, betas_vanilla, eta, jax, jnp, lambd, n, np, optax):
    GAMMA_MAX = 10.0 / L
    BETA_MAX = 1.5
    _CVAR_Q = max(1, int(0.05 * N_box))

    _Dj = jnp.asarray(D)
    _Mj = jnp.asarray(M)
    _Ij = jnp.eye(n)
    _Xtr = jnp.asarray(X_box)

    def _soft(v, t):
        return jnp.sign(v) * jnp.maximum(jnp.abs(v) - t, 0.0)

    def _resid_perk(params, x):
        gammas = GAMMA_MAX * jax.nn.sigmoid(params[:K])
        betas = BETA_MAX * jax.nn.sigmoid(params[K:])
        s_prev = jnp.zeros(n)
        y = jnp.zeros(n)
        out = []
        for k in range(K):
            g = gammas[k]
            s = _soft(y - g * (_Dj.T @ (_Dj @ y - x)), lambd * g)
            r = ((1.0 / g) * _Ij - _Mj) @ (s - y)
            out.append(jnp.sum(r ** 2))
            y_new = s + betas[k] * (s - s_prev)
            s_prev = s
            y = y_new
        return jnp.stack(out)

    def _loss(params, X):
        R_mat = jax.vmap(lambda x: _resid_perk(params, x))(X)
        per_k = jnp.mean(jnp.sort(R_mat, axis=0)[-_CVAR_Q:, :], axis=0)
        return jnp.sum(per_k)

    _a_init = float(np.log((eta / GAMMA_MAX) / (1.0 - eta / GAMMA_MAX)))
    _b_clipped = np.clip(betas_vanilla, 1e-6, BETA_MAX - 1e-6)
    _b_init = np.log((_b_clipped / BETA_MAX) / (1.0 - _b_clipped / BETA_MAX))
    _params = jnp.concatenate([jnp.full(K, _a_init), jnp.asarray(_b_init)])

    _opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(5e-3))
    _opt_state = _opt.init(_params)
    _loss_grad = jax.jit(jax.value_and_grad(_loss))

    print("Training learned FISTA ...")
    for _step in range(6000):
        _loss_val, _grads = _loss_grad(_params, _Xtr)
        _updates, _opt_state = _opt.update(_grads, _opt_state)
        _params = optax.apply_updates(_params, _updates)
        if _step % 750 == 0 or _step == 5999:
            print(f"  step {_step:4d}  loss {float(_loss_val):.6f}")

    gammas_learned = np.asarray(GAMMA_MAX * jax.nn.sigmoid(_params[:K]))
    betas_learned = np.asarray(BETA_MAX * jax.nn.sigmoid(_params[K:]))
    thetas_learned = lambd * gammas_learned
    print(f"  learned gamma_k (x L): {np.array2string(gammas_learned * L, precision=2)}")
    print(f"  learned beta_k:        {np.array2string(betas_learned, precision=2)}")

    return betas_learned, gammas_learned, thetas_learned


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Simulation

    We simulate both schedules over $N$ box-uniform samples and report the sample-maximum KKT residual at each iteration.
    """)
    return


@app.cell
def _(D, H_of, K, Verifier, X_box, betas_vanilla, gammas_vanilla, m, np, s0, thetas_vanilla, x_lb, x_ub):
    _vp = Verifier(method="simulator")
    _x = _vp.add_param(ndim=m, lb=x_lb, ub=x_ub, samples=X_box)
    _s = [s0]
    _y = [s0]
    for _k in range(1, K + 1):
        _g, _th, _beta = gammas_vanilla[_k - 1], thetas_vanilla[_k - 1], float(betas_vanilla[_k - 1])
        _sk = _vp.step.soft_threshold(_y[_k - 1] - _g * (D.T @ (D @ _y[_k - 1] - _x)), lambd=_th)
        _s.append(_sk)
        _vp.set_objective_norm([H_of(_g) @ (_sk - _y[_k - 1])], ord=np.inf)
        _y.append(_sk + _beta * (_sk - _s[_k - 1]))
    sam_v = np.asarray(_vp.solve()).max(axis=0)
    return (sam_v,)


@app.cell
def _(D, H_of, K, Verifier, X_box, betas_learned, gammas_learned, m, np, s0, thetas_learned, x_lb, x_ub):
    _vp = Verifier(method="simulator")
    _x = _vp.add_param(ndim=m, lb=x_lb, ub=x_ub, samples=X_box)
    _s = [s0]
    _y = [s0]
    for _k in range(1, K + 1):
        _g, _th, _beta = gammas_learned[_k - 1], thetas_learned[_k - 1], float(betas_learned[_k - 1])
        _sk = _vp.step.soft_threshold(_y[_k - 1] - _g * (D.T @ (D @ _y[_k - 1] - _x)), lambd=_th)
        _s.append(_sk)
        _vp.set_objective_norm([H_of(_g) @ (_sk - _y[_k - 1])], ord=np.inf)
        _y.append(_sk + _beta * (_sk - _s[_k - 1]))
    sam_l = np.asarray(_vp.solve()).max(axis=0)
    return (sam_l,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Results

    The learned schedules yield a smaller sample-maximum KKT residual than vanilla FISTA.
    The bar charts show step-sizes relative to $1/L$ and momentum values; values above the dashed lines
    ($2/L$ and $1$ respectively) exceed the standard stability thresholds.
    """)
    return


@app.cell
def _(K, L, betas_learned, betas_vanilla, gammas_learned, gammas_vanilla, np, plt, sam_l, sam_v):
    ks = np.arange(1, K + 1)
    BART_RED = (0.5020, 0.0000, 0.2510)

    _fig, _axes = plt.subplots(
        3, 2, figsize=(13, 11), sharex="col", sharey="row",
        gridspec_kw={"height_ratios": [3, 1, 1]},
    )

    for _ax, _sam, _title in [
        (_axes[0, 0], sam_v, "FISTA"),
        (_axes[0, 1], sam_l, "Learned FISTA"),
    ]:
        _ax.plot(ks, _sam, "-", color="C3", lw=2, label="Sample Maximum")
        _ax.set_yscale("log")
        _ax.set_title(_title)
        _ax.grid(True, which="major", alpha=0.25)

    for _ax, _gammas in [(_axes[1, 0], gammas_vanilla), (_axes[1, 1], gammas_learned)]:
        _ax.bar(ks, _gammas * L, width=0.72, facecolor=(*BART_RED, 0.20), edgecolor=BART_RED, linewidth=2.0)
        _ax.axhline(2.0, color="0.35", ls="--", lw=2.0)
        _ax.set_xticks(ks[1::2])
        _ax.grid(True, axis="y", alpha=0.25)

    for _ax, _betas in [(_axes[2, 0], betas_vanilla), (_axes[2, 1], betas_learned)]:
        _ax.bar(ks, _betas, width=0.72, facecolor=(*BART_RED, 0.20), edgecolor=BART_RED, linewidth=2.0)
        _ax.axhline(1.0, color="0.35", ls="--", lw=2.0)
        _ax.set_xlabel(r"$k$")
        _ax.set_xticks(ks[1::2])
        _ax.grid(True, axis="y", alpha=0.25)

    _axes[0, 0].set_ylabel("Sample-max KKT residual")
    _axes[1, 0].set_ylabel(r"Step size ($\times L$)")
    _axes[2, 0].set_ylabel(r"Momentum $\beta_k$")
    _axes[0, 0].legend(fontsize=11)
    _fig.tight_layout()
    plt.show()
    return


if __name__ == "__main__":
    app.run()
