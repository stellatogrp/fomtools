r"""
Vanilla FISTA vs. learned FISTA with time-varying step-sizes and momentum (L2O), on a Lasso
problem, verified with the MIP framework from the mip_fom_verify paper.

Goal of the figure
-------------------
Show that the MIP verification machinery (``Verifier(method="mip")``) verifies not only
the fixed-schedule FISTA but also a *learned* FISTA whose per-iteration step-sizes AND
momentum parameters are jointly tuned (L2O).  The learned optimizer should show a clearly
smaller worst-case scaled fixed-point residual.

Lasso:
    minimize  (1/2) || D s - x ||^2  +  lambda || s ||_1,    D in R^{m x n}, s0 = 0
FISTA iteration (per-iteration step-size gamma_k, threshold theta_k = gamma_k*lambda,
momentum beta_k):
    s^k = T_{theta_k}( y^{k-1} - gamma_k * D^T (D y^{k-1} - x) )
    y^k = s^k + beta_k * (s^k - s^{k-1})
Vanilla:  gamma_k = 1/L,  beta_k = (t_k - 1) / t_{k+1}  (standard t-schedule).
Learned:  gamma_k, beta_k jointly time-varying (L2O).

y-axis: worst-case scaled fixed-point residual (KKT residual at s^k):
    r_k = H_k ( s^k - y^{k-1} ),   H_k = (1/gamma_k) I - D^T D
where y^{k-1} is the extrapolated point from which the gradient step produced s^k.
||r_k|| is the exact Lasso KKT residual at s^k (zero iff s^k = s*).
x-axis: number of iterations K = 1..10.

Two curves per subplot:
    - MIP    : exact worst-case bound over the parameter box  (Gurobi, the tight curve)
    - Sample : max scaled residual over N sampled instances    (lower bound on worst case)
    - PEP    : dimension-free SDP bound (looser; does not exploit D's spectrum)

Run:
    source .venv/bin/activate
    python tests/examples/learned_fista_comparison.py
"""

import os

import cvxpy as cp
import gurobipy as gp
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
from PEPit import PEP
from PEPit.functions import ConvexFunction, SmoothStronglyConvexFunction
from PEPit.primitive_steps import proximal_step

from fomtools.verifier import Verifier

jax.config.update("jax_enable_x64", True)
gp.setParam("OutputFlag", 0)


np.random.seed(0)
m, n, lambd = 10, 15, 1e-1
K = 10
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
mu = float(max(eig[0], 0.0))
eta = 1.0 / L
EYE = np.eye(n)


def H_of(g):
    return (1.0 / g) * EYE - M


# ──────────────────────────────────────────────────────────────────────────────
# Vanilla FISTA schedules
# ──────────────────────────────────────────────────────────────────────────────
gammas_vanilla = np.full(K, eta)
thetas_vanilla = lambd * gammas_vanilla


def _vanilla_betas(K):
    """Standard FISTA momentum from the t_k sequence (Beck & Teboulle 2009)."""
    t = np.ones(K + 1)
    for k in range(1, K):
        t[k + 1] = (1.0 + np.sqrt(1.0 + 4.0 * t[k] ** 2)) / 2.0
    betas = np.zeros(K)
    for k in range(K):
        betas[k] = (t[k] - 1.0) / t[k + 1]
    return betas


betas_vanilla = _vanilla_betas(K)

# ──────────────────────────────────────────────────────────────────────────────
# Box samples for L2O training and the worst-case Sample curve
# ──────────────────────────────────────────────────────────────────────────────
N_box = 4000
np.random.seed(1)
X_box = np.random.uniform(x_lb, x_ub, size=(N_box, m))

# PEP radius: max_x ||s0 - s*(x)||  over box samples
_scp = cp.Variable(n)
_xcp = cp.Parameter(m)
_lasso = cp.Problem(cp.Minimize(0.5 * cp.sum_squares(D @ _scp - _xcp) + lambd * cp.norm1(_scp)))
R = 0.0
for _xi in X_box[:400]:
    _xcp.value = _xi
    _lasso.solve(solver=cp.CLARABEL)
    R = max(R, float(np.linalg.norm(_scp.value)))
print(f"PEP radius R = {R:.4f},  L = {L:.4f},  mu = {mu:.4g}", flush=True)

# ──────────────────────────────────────────────────────────────────────────────
# L2O: jointly learn time-varying step-sizes gamma_k and momentum beta_k.
#   gamma_k in (0, GAMMA_MAX),  beta_k in (0, BETA_MAX).
#   Uncapping gamma beyond 2/L and beta beyond 1 lets the optimizer discover
#   silver-stepsize / aggressive-momentum patterns unreachable by vanilla FISTA.
#   loss = CVaR over worst CVAR_FRAC box samples, summed across iterations.
# ──────────────────────────────────────────────────────────────────────────────
TRAIN_OBJECTIVE = "cvar"
CVAR_FRAC = 0.05
GAMMA_MAX = 10.0 / L
BETA_MAX = 1.5
CVAR_Q = max(1, int(CVAR_FRAC * N_box))
Dj, Mj, Ij = jnp.asarray(D), jnp.asarray(M), jnp.eye(n)
Xtr = jnp.asarray(X_box)


def _soft(v, t):
    return jnp.sign(v) * jnp.maximum(jnp.abs(v) - t, 0.0)


def _resid_perk(params, x):
    """FISTA residuals for all K iterations on a single sample x."""
    gammas = GAMMA_MAX * jax.nn.sigmoid(params[:K])
    betas = BETA_MAX * jax.nn.sigmoid(params[K:])
    s_prev = jnp.zeros(n)
    y = jnp.zeros(n)  # y^0 = s^0 = 0
    out = []
    for k in range(K):
        g = gammas[k]
        beta = betas[k]
        s = _soft(y - g * (Dj.T @ (Dj @ y - x)), lambd * g)
        r = ((1.0 / g) * Ij - Mj) @ (s - y)  # H_k (s^k - y^{k-1})
        out.append(jnp.sum(r ** 2))
        y_new = s + beta * (s - s_prev)
        s_prev = s
        y = y_new
    return jnp.stack(out)


def _loss(params, X):
    residuals = jax.vmap(lambda x: _resid_perk(params, x))(X)  # (N, K)
    if TRAIN_OBJECTIVE == "cvar":
        per_k = jnp.mean(jnp.sort(residuals, axis=0)[-CVAR_Q:, :], axis=0)
    else:
        per_k = jnp.mean(residuals, axis=0)
    return jnp.sum(per_k)


# Initialise params: gammas -> eta, betas -> vanilla schedule
a_init = float(np.log((eta / GAMMA_MAX) / (1.0 - eta / GAMMA_MAX)))
betas_clipped = np.clip(betas_vanilla, 1e-6, BETA_MAX - 1e-6)
b_init = np.log((betas_clipped / BETA_MAX) / (1.0 - betas_clipped / BETA_MAX))

params = jnp.concatenate([jnp.full(K, a_init), jnp.asarray(b_init)])
opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(5e-3))
opt_state = opt.init(params)
loss_grad = jax.jit(jax.value_and_grad(_loss))

print(f"Training learned FISTA (gamma_k + beta_k, objective={TRAIN_OBJECTIVE}) ...", flush=True)
for step in range(6000):
    loss_val, grads = loss_grad(params, Xtr)
    updates, opt_state = opt.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    if step % 750 == 0 or step == 5999:
        print(f"  step {step:4d}  loss {float(loss_val):.6f}", flush=True)

gammas_learned = np.asarray(GAMMA_MAX * jax.nn.sigmoid(params[:K]))
betas_learned = np.asarray(BETA_MAX * jax.nn.sigmoid(params[K:]))
thetas_learned = lambd * gammas_learned
print(f"  learned gamma_k (x L): {np.array2string(gammas_learned * L, precision=2)}", flush=True)
print(f"  learned beta_k:        {np.array2string(betas_learned, precision=2)}", flush=True)

# ──────────────────────────────────────────────────────────────────────────────
# Verification: MIP (exact worst case), Simulator (sample worst case), PEP (SDP)
# ──────────────────────────────────────────────────────────────────────────────
GRB = {"OutputFlag": 0, "LogToConsole": 0}


def verify_schedule(gammas, thetas, betas, tag):
    """Worst-case scaled KKT residual ||H_k(s^k - y^{k-1})||_inf, k=1..K."""

    def build(vp, x, set_obj):
        s = [s0]       # s[k] = s^k;  s[0] = s^0 = 0 (numpy)
        y = [s0]       # y[k] = y^k;  y[0] = y^0 = s^0
        objs = []
        for k in range(1, K + 1):
            g, th, beta = gammas[k - 1], thetas[k - 1], float(betas[k - 1])
            # Proximal-gradient step at y^{k-1}
            s_k = vp.step.soft_threshold(y[k - 1] - g * (D.T @ (D @ y[k - 1] - x)), lambd=th)
            s.append(s_k)
            obj = [H_of(g) @ (s_k - y[k - 1])]  # H_k(s^k - y^{k-1})
            objs.append(obj)
            if set_obj:
                vp.set_objective_norm(obj, ord=np.inf)
            # Momentum extrapolation: y^k = s^k + beta*(s^k - s^{k-1})
            y_k = s_k + beta * (s_k - s[k - 1])
            y.append(y_k)
        return objs

    # --- MIP: exact worst-case over the box -----------------------------------
    print(f"[{tag}] MIP ...", flush=True)
    vp = Verifier(method="mip", solver="gurobi", obbt_solver="gurobi", params=GRB)
    objs = build(vp, vp.add_param(m, lb=x_lb, ub=x_ub), set_obj=False)
    mip = []
    for k in range(K):
        vp.set_objective_norm(objs[k], ord=np.inf)
        mip.append(vp.solve())
        print(f"   k={k + 1:2d}  MIP={mip[-1]:.6f}", flush=True)

    # --- Simulator: max residual over N_box box-uniform samples ---------------
    print(f"[{tag}] Simulator ...", flush=True)
    vps = Verifier(method="simulator")
    build(vps, vps.add_param(ndim=m, lb=x_lb, ub=x_ub, samples=X_box), set_obj=True)
    sample = np.asarray(vps.solve()).max(axis=0)

    # --- PEP: dimension-free SDP bound on the same scaled KKT residual --------
    # Uses SmoothStronglyConvexFunction (does not exploit D's exact spectrum),
    # so it is looser than the MIP -- especially for learned long/short steps.
    # KKT residual in PEP terms:
    #   r_k = (1/gamma_k)(s^k - y^{k-1}) - (grad_f(s^k) - grad_f(y^{k-1}))
    # which equals H_k(s^k - y^{k-1}) when f is quadratic with Hessian M.
    print(f"[{tag}] PEP ...", flush=True)
    pep = []
    for k in range(1, K + 1):
        problem = PEP()
        f = problem.declare_function(SmoothStronglyConvexFunction, L=L, mu=mu)
        gfun = problem.declare_function(ConvexFunction)
        xstar = (f + gfun).stationary_point()
        sp = [problem.set_initial_point()]  # sp[j] = s^j
        yp = [sp[0]]                        # yp[j] = y^j;  y^0 = s^0
        problem.set_initial_condition((sp[0] - xstar) ** 2 <= R ** 2)
        for j in range(k):
            # Proximal-gradient step at y^j
            snext, _, _ = proximal_step(yp[j] - gammas[j] * f.gradient(yp[j]), gfun, gammas[j])
            sp.append(snext)
            # Momentum: y^{j+1} = s^{j+1} + beta_{j+1}*(s^{j+1} - s^j)
            ynext = snext + float(betas[j]) * (snext - sp[j])
            yp.append(ynext)
        gk = gammas[k - 1]
        scaled = (1.0 / gk) * (sp[k] - yp[k - 1]) - (f.gradient(sp[k]) - f.gradient(yp[k - 1]))
        problem.set_performance_metric(scaled ** 2)
        val = problem.solve(solver="CLARABEL", verbose=0)
        pep.append(np.nan if (val is None or not np.isfinite(val)) else np.sqrt(max(val, 0.0)))
        print(f"   k={k:2d}  PEP={pep[-1]:.4f}", flush=True)
    return np.asarray(mip), np.asarray(sample), np.asarray(pep)


CACHE = os.path.join(os.path.dirname(__file__), "learned_fista_comparison_data.npz")
mip_v, sam_v, pep_v = verify_schedule(gammas_vanilla, thetas_vanilla, betas_vanilla, "FISTA")
mip_l, sam_l, pep_l = verify_schedule(gammas_learned, thetas_learned, betas_learned, "Learned FISTA")
np.savez(
    CACHE,
    mip_v=mip_v,
    sam_v=sam_v,
    pep_v=pep_v,
    mip_l=mip_l,
    sam_l=sam_l,
    pep_l=pep_l,
    gammas_learned=gammas_learned,
    betas_learned=betas_learned,
)

print(
    "\nordering Sample<=MIP  FISTA:",
    bool(np.all(sam_v <= mip_v * (1 + 1e-4))),
    " Learned:",
    bool(np.all(sam_l <= mip_l * (1 + 1e-4))),
)
print(
    "MIP improvement (vanilla / learned) per k:",
    np.array2string(mip_v / np.maximum(mip_l, 1e-12), precision=2),
)

# ──────────────────────────────────────────────────────────────────────────────
# Figure: 3-row subplots (residual, step sizes, momentum)  x  2 columns
# ──────────────────────────────────────────────────────────────────────────────
plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue"],
        "font.size": 28,
    }
)
plt.rcParams["text.latex.preamble"] = (
    r"\usepackage[scaled]{helvet}\renewcommand{\familydefault}{\sfdefault}"
)

ks = np.arange(1, K + 1)
BART_RED = (0.5020, 0.0000, 0.2510)


def bars_gamma(ax, gammas):
    ax.bar(
        ks,
        gammas * L,
        width=0.72,
        facecolor=(*BART_RED, 0.20),
        edgecolor=BART_RED,
        linewidth=2.0,
    )
    ax.axhline(2.0, color="0.35", ls="--", lw=2.5)
    ax.text(
        0.985,
        2.0,
        r"$2/L$",
        transform=ax.get_yaxis_transform(),
        va="bottom",
        ha="right",
        fontsize=22,
        color="0.35",
    )
    ax.set_xticks(ks[1::2])
    ax.grid(True, axis="y", alpha=0.25)


def bars_beta(ax, betas):
    ax.bar(
        ks,
        betas,
        width=0.72,
        facecolor=(*BART_RED, 0.20),
        edgecolor=BART_RED,
        linewidth=2.0,
    )
    ax.axhline(1.0, color="0.35", ls="--", lw=2.5)
    ax.text(
        0.985,
        1.0,
        r"$1$",
        transform=ax.get_yaxis_transform(),
        va="bottom",
        ha="right",
        fontsize=22,
        color="0.35",
    )
    ax.set_xlabel(r"$k$")
    ax.set_xticks(ks[1::2])
    ax.grid(True, axis="y", alpha=0.25)


def make_figure(show_pep, out_path):
    fig, axes = plt.subplots(
        3,
        2,
        figsize=(16, 14),
        sharex="col",
        sharey="row",
        gridspec_kw={"height_ratios": [3, 1, 1]},
    )

    def draw(ax, mip, sample, pep, title):
        if show_pep and np.isfinite(pep).any():
            ax.plot(ks, pep, "-", color="C2", lw=3, label=r"PEP")
        ax.plot(ks, mip, "-", color="C0", lw=3, label=r"Verification Problem")
        ax.plot(ks, sample, "-", color="C3", lw=3, label=r"Sample Maximum")
        ax.set_yscale("log")
        ax.set_title(title)
        ax.grid(True, which="major", alpha=0.25)

    draw(axes[0, 0], mip_v, sam_v, pep_v, r"FISTA")
    draw(axes[0, 1], mip_l, sam_l, pep_l, r"Learned FISTA")
    bars_gamma(axes[1, 0], gammas_vanilla)
    bars_gamma(axes[1, 1], gammas_learned)
    bars_beta(axes[2, 0], betas_vanilla)
    bars_beta(axes[2, 1], betas_learned)
    axes[0, 0].set_ylabel(r"Worst-case scaled fixed-point residual", fontsize=23)
    axes[1, 0].set_ylabel(r"Step size ($\times L$)")
    axes[2, 0].set_ylabel(r"Momentum $\beta_k$")
    axes[0, 0].legend(fontsize=22, framealpha=0.9, loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure to {out_path}")


_here = os.path.dirname(__file__)
make_figure(True, os.path.join(_here, "learned_fista_comparison_pep.pdf"))
make_figure(False, os.path.join(_here, "learned_fista_comparison_nopep.pdf"))
