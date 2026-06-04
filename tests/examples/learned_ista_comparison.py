r"""
Vanilla ISTA vs. learned ISTA with time-varying step-sizes (L2O), on a Lasso problem,
verified with the MIP framework from the mip_fom_verify paper.

Goal of the figure
-------------------
Show that the MIP verification machinery (``Verifier(method="mip")``) verifies not only
the fixed-step ISTA but also a *learned* ISTA whose per-iteration step-sizes are tuned
(time-varying, silver-stepsize style -- NO momentum / no acceleration variant).  The
learned optimizer should show a clearly smaller worst-case scaled fixed-point residual.

Lasso:
    minimize  (1/2) || D s - x ||^2  +  lambda || s ||_1,    D in R^{m x n}, s0 = 0
ISTA iteration (per-iteration step-size gamma_k; threshold coupled theta_k = gamma_k*lambda
so every step is a bona-fide proximal-gradient step for the ORIGINAL Lasso):
    s^{k+1} = T_{gamma_k*lambda}( s^k - gamma_k * D^T (D s^k - x) )
Vanilla:  gamma_k = 1/L  (L = max eig of D^T D).  Learned: gamma_k time-varying (L2O).

y-axis: worst-case scaled fixed-point residual, scaled to match the KKT/optimality
conditions (paper Appendix lemma).  At iterate s_k:
    r_k = H_k ( s^k - s^{k-1} ),   H_k = (1/gamma_k) I - D^T D,
where gamma_k is the ACTUAL step that produced s^k (the last step).  ||r_k|| is then the
exact Lasso KKT residual at s^k (zero iff s^k = s*).  Same definition for both variants:
vanilla gamma_k = 1/L (constant H), learned gamma_k time-varying (H_k varies).
x-axis: number of iterations K = 1..10  (Gurobi MIP does not scale to huge K).

Two curves per subplot:
    - MIP    : exact worst-case bound over the parameter box  (Gurobi, the tight curve)
    - Sample : max scaled residual over N sampled instances    (lower bound on worst case)

Run:
    source .venv/bin/activate
    python tests/examples/learned_ista_comparison.py
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
gp.setParam("OutputFlag", 0)  # silence Gurobi + OBBT subproblem logs

# ──────────────────────────────────────────────────────────────────────────────
# Problem setup  (m x n = 10 x 15, sparse Gaussian dictionary, lambda = 1e-1)
# ──────────────────────────────────────────────────────────────────────────────
np.random.seed(0)
m, n, lambd = 10, 15, 1e-1
K = 10
s0 = np.zeros(n)

D_value = np.random.normal(size=(m, n))
D_mask = np.random.binomial(n=1, p=0.2, size=(m, n))
D = np.multiply(D_value, D_mask)

# Parameter box: tightest hyperrectangle containing a sparse-coding sample cloud
# x = D z* + noise, z* ~ 10%-sparse Gaussian.  The MIP verifies over THIS box.
z_star = np.multiply(
    np.random.normal(size=(n, 500)),
    np.random.binomial(n=1, p=0.1, size=(n, 500)),
)
x_cloud = D @ z_star + np.random.normal(scale=1e-2, size=(m, 500))
x_lb, x_ub = np.min(x_cloud, axis=1), np.max(x_cloud, axis=1)

M = D.T @ D
eig = np.linalg.eigvalsh(M)
L = float(eig[-1])
mu = float(max(eig[0], 0.0))  # min curvature (0 here: m < n => D^T D rank-deficient)
eta = 1.0 / L  # vanilla step
EYE = np.eye(n)


# KKT scaling at iterate k uses the ACTUAL step gamma_k that produced it:
#   H_k = (1/gamma_k) I - D^T D,   and  ||H_k (s^k - s^{k-1})||  is the Lasso KKT residual
#   at s^k  (paper Appendix: H = -(D^T D - eta^{-1} I)).  Same rule for both variants:
#   vanilla -> gamma_k = 1/L (constant H);  learned -> gamma_k = learned step (time-varying).
def H_of(g):
    return (1.0 / g) * EYE - M


gammas_vanilla = np.full(K, eta)
thetas_vanilla = lambd * gammas_vanilla

# Samples for L2O training AND for the worst-case-Sample line: drawn UNIFORMLY from the
# verification box (NOT the sparse manifold), so the learned step-sizes are optimized over
# exactly the set the MIP maximizes over, and the Sample curve is a tight lower bound.
N_box = 4000
np.random.seed(1)
X_box = np.random.uniform(x_lb, x_ub, size=(N_box, m))

# PEP radius R = max_x ||s0 - s*(x)|| over the (sampled) verified set, for the PEP initial
# condition ||s0 - s*||^2 <= R^2.  s*(x) is the Lasso solution; estimated on box samples.
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
# L2O: learn time-varying step-sizes gamma_k (no momentum, threshold coupled).
#   Steps are UNCAPPED beyond the usual 2/L stability limit (gamma_k in (0, GAMMA_MAX/L)),
#   so training can discover long/short "silver" step patterns that accelerate the
#   worst-case rate -- a constant step <= 2/L cannot.
#   loss = sum_k reduce_over_box_samples( || H_k (s^k - s^{k-1}) ||^2 ),  where the reduction
#   is set by TRAIN_OBJECTIVE:
#       "average" -> mean over samples (matches the average residual)
#       "cvar"    -> mean of the worst CVAR_FRAC samples per iteration (matches the WORST-case
#                    residual the MIP reports; leaves the worst case tighter than "average").
#   Coupled threshold keeps it a genuine KKT residual (no shrink-to-zero degeneracy) -> drives
#   the iterates to s*.  Init at vanilla gamma = 1/L.
# ──────────────────────────────────────────────────────────────────────────────
TRAIN_OBJECTIVE = "cvar"  # <-- switch: "cvar" or "average"
CVAR_FRAC = 0.05  # tail fraction used when TRAIN_OBJECTIVE="cvar"
GAMMA_MAX = 10.0 / L
CVAR_Q = max(1, int(CVAR_FRAC * N_box))
Dj, Mj, Ij = jnp.asarray(D), jnp.asarray(M), jnp.eye(n)
Xtr = jnp.asarray(X_box)


def _soft(v, t):
    return jnp.sign(v) * jnp.maximum(jnp.abs(v) - t, 0.0)


def _resid_perk(a, x):
    gam = GAMMA_MAX * jax.nn.sigmoid(a)  # gamma_k in (0, GAMMA_MAX)
    s_prev = jnp.zeros(n)
    out = []
    for k in range(K):
        s = _soft(s_prev - gam[k] * (Dj.T @ (Dj @ s_prev - x)), lambd * gam[k])
        r = ((1.0 / gam[k]) * Ij - Mj) @ (s - s_prev)  # H_k (s^k - s^{k-1}), H_k from gamma_k
        out.append(jnp.sum(r**2))
        s_prev = s
    return jnp.stack(out)  # (K,)


def _loss(a, X):
    R = jax.vmap(lambda x: _resid_perk(a, x))(X)  # (N, K)  per-sample, per-iteration
    if TRAIN_OBJECTIVE == "cvar":
        per_k = jnp.mean(jnp.sort(R, axis=0)[-CVAR_Q:, :], axis=0)  # worst CVAR_FRAC per iteration
    else:  # "average"
        per_k = jnp.mean(R, axis=0)
    return jnp.sum(per_k)  # summed over iterations (deep supervision)


a = jnp.full(K, float(np.log((eta / GAMMA_MAX) / (1 - eta / GAMMA_MAX))))  # sigmoid -> eta
opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(5e-3))
opt_state = opt.init(a)
loss_grad = jax.jit(jax.value_and_grad(_loss))

print(f"Training learned time-varying step-sizes (objective={TRAIN_OBJECTIVE}) ...", flush=True)
for step in range(6000):
    loss_val, grads = loss_grad(a, Xtr)
    updates, opt_state = opt.update(grads, opt_state)
    a = optax.apply_updates(a, updates)
    if step % 750 == 0 or step == 5999:
        print(f"  step {step:4d}  loss {float(loss_val):.6f}", flush=True)

gammas_learned = np.asarray(GAMMA_MAX * jax.nn.sigmoid(a))
thetas_learned = lambd * gammas_learned  # coupled threshold
print(f"  learned gamma_k (x L): {np.array2string(gammas_learned * L, precision=2)}", flush=True)

# ──────────────────────────────────────────────────────────────────────────────
# Verification: MIP (exact worst case over the box) and Simulator (sample worst case)
# ──────────────────────────────────────────────────────────────────────────────
GRB = {"OutputFlag": 0, "LogToConsole": 0}


def verify_schedule(gammas, thetas, tag):
    """Worst-case scaled KKT (fixed-point) residual, k=1..K, via MIP and Simulator.
    r_k = H_k ( s^k - s^{k-1} ),  H_k = (1/gamma_k) I - D^T D,  ||r_k||_inf.
    H_k uses the actual step gamma_k that produced s^k (the last step) -- the genuine
    Lasso KKT residual at s^k; same definition for the static and learned variants."""

    def build(vp, x, set_obj):  # ISTA chain + per-k residual objective
        s = [s0]  # s0 = 0 (numpy); simulator binds each objective
        objs = []  # to the most-recent step, so set inline there
        for k in range(1, K + 1):
            g, th = gammas[k - 1], thetas[k - 1]
            s_k = vp.step.soft_threshold(s[k - 1] - g * (D.T @ (D @ s[k - 1] - x)), lambd=th)
            s.append(s_k)
            obj = [H_of(g) @ (s_k - s[k - 1])]  # H_k (s^k - s^{k-1}),  H_k from gamma_k
            objs.append(obj)
            if set_obj:
                vp.set_objective_norm(obj, ord=np.inf)
        return objs

    # --- MIP: exact worst-case over the box (build chain, set objective k, solve) -----
    print(f"[{tag}] MIP ...", flush=True)
    vp = Verifier(method="mip", solver="gurobi", obbt_solver="gurobi", params=GRB)
    objs = build(vp, vp.add_param(m, lb=x_lb, ub=x_ub), set_obj=False)
    mip = []
    for k in range(K):
        vp.set_objective_norm(objs[k], ord=np.inf)
        mip.append(vp.solve())
        print(f"   k={k + 1:2d}  MIP={mip[-1]:.6f}", flush=True)

    # --- Simulator: max scaled residual over the N_box box-uniform samples -----------
    print(f"[{tag}] Simulator ...", flush=True)
    vps = Verifier(method="simulator")
    build(vps, vps.add_param(ndim=m, lb=x_lb, ub=x_ub, samples=X_box), set_obj=True)
    sample = np.asarray(vps.solve()).max(axis=0)

    # --- PEP: SDP worst-case bound on the SAME scaled KKT residual, using the actual gamma_k.
    # Dimension-free (worst case over all L-smooth convex f with the given step sequence): it
    # does NOT know D's spectrum, so it is loose vs the MIP -- especially for the long learned
    # steps.  Quadratic f model makes grad f(s^k)-grad f(s^{k-1}) = M(s^k-s^{k-1}) exact, so
    # H_k(s^k-s^{k-1}) = (1/gamma_k)(s^k-s^{k-1}) - (grad f(s^k)-grad f(s^{k-1})).
    print(f"[{tag}] PEP ...", flush=True)
    pep = []
    for k in range(1, K + 1):
        problem = PEP()
        # f = problem.declare_function(SmoothStronglyConvexQuadraticFunction, L=L, mu=mu)
        f = problem.declare_function(SmoothStronglyConvexFunction, L=L, mu=mu)
        gfun = problem.declare_function(ConvexFunction)
        xstar = (f + gfun).stationary_point()
        sp = [problem.set_initial_point()]
        problem.set_initial_condition((sp[0] - xstar) ** 2 <= R**2)
        for j in range(k):
            y = sp[j] - gammas[j] * f.gradient(sp[j])
            snext, _, _ = proximal_step(
                y, gfun, gammas[j]
            )  # prox step = gamma_j (coupled threshold)
            sp.append(snext)
        gk = gammas[k - 1]
        scaled = (1.0 / gk) * (sp[k] - sp[k - 1]) - (f.gradient(sp[k]) - f.gradient(sp[k - 1]))
        problem.set_performance_metric(scaled**2)
        val = problem.solve(solver="CLARABEL", verbose=0)
        pep.append(np.nan if (val is None or not np.isfinite(val)) else np.sqrt(max(val, 0.0)))
        print(f"   k={k:2d}  PEP={pep[-1]:.4f}", flush=True)
    return np.asarray(mip), np.asarray(sample), np.asarray(pep)


CACHE = os.path.join(os.path.dirname(__file__), "learned_ista_comparison_data.npz")
mip_v, sam_v, pep_v = verify_schedule(gammas_vanilla, thetas_vanilla, "ISTA")
mip_l, sam_l, pep_l = verify_schedule(gammas_learned, thetas_learned, "Learned ISTA")
np.savez(
    CACHE,
    mip_v=mip_v,
    sam_v=sam_v,
    pep_v=pep_v,
    mip_l=mip_l,
    sam_l=sam_l,
    pep_l=pep_l,
    gammas_learned=gammas_learned,
)

print(
    "\nordering Sample<=MIP  ISTA:",
    bool(np.all(sam_v <= mip_v * (1 + 1e-4))),
    " Learned:",
    bool(np.all(sam_l <= mip_l * (1 + 1e-4))),
)
print(
    "MIP improvement (vanilla / learned) per k:",
    np.array2string(mip_v / np.maximum(mip_l, 1e-12), precision=2),
)

# ──────────────────────────────────────────────────────────────────────────────
# Figure: two subplots side by side, talk-quality (LaTeX, sans-serif, size 28)
# ──────────────────────────────────────────────────────────────────────────────
# Text (titles, labels) in Helvetica Neue; MATH ($K$, $\gamma_k$, norms) in LaTeX's own
# math font (Computer Modern) -- helvet for text, NO sansmath, so math stays "LaTeX".
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

# Bartolomeo macOS palette (~/Library/Colors/Bartolomeo.clr): "Red" / "Red transparent"
BART_RED = (0.5020, 0.0000, 0.2510)


def bars(ax, gammas):  # bottom row: the actual step sizes used, per iteration
    ax.bar(
        ks,
        gammas * L,
        width=0.72,
        facecolor=(*BART_RED, 0.20),  # "Red transparent" fill
        edgecolor=BART_RED,  # "Red" border
        linewidth=2.0,
    )
    ax.axhline(2.0, color="0.35", ls="--", lw=2.5)  # 2/L stability limit
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
    ax.set_xlabel(r"$k$")
    ax.set_xticks(ks[1::2])
    ax.grid(True, axis="y", alpha=0.25)


def make_figure(show_pep, out_path):
    fig, axes = plt.subplots(
        2, 2, figsize=(16, 11), sharex="col", sharey="row", gridspec_kw={"height_ratios": [2, 1]}
    )

    def draw(ax, mip, sample, pep, title):  # top row: worst-case residual vs k (paper VP/SM)
        if show_pep and np.isfinite(pep).any():
            ax.plot(ks, pep, "-", color="C2", lw=3, label=r"PEP")  # SDP-PEP -> green, solid
        ax.plot(ks, mip, "-", color="C0", lw=3, label=r"Verification Problem")  # VP, blue
        ax.plot(ks, sample, "-", color="C3", lw=3, label=r"Sample Maximum")  # SM, red
        ax.set_yscale("log")
        ax.set_title(title)
        ax.grid(True, which="major", alpha=0.25)

    draw(axes[0, 0], mip_v, sam_v, pep_v, r"ISTA")
    draw(axes[0, 1], mip_l, sam_l, pep_l, r"Learned ISTA")
    bars(axes[1, 0], gammas_vanilla)
    bars(axes[1, 1], gammas_learned)
    axes[0, 0].set_ylabel(r"Worst-case scaled fixed-point residual", fontsize=23)
    axes[1, 0].set_ylabel(r"Step size")
    axes[0, 0].legend(fontsize=22, framealpha=0.9, loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure to {out_path}")


_here = os.path.dirname(__file__)
make_figure(True, os.path.join(_here, "learned_ista_comparison_pep.pdf"))
make_figure(False, os.path.join(_here, "learned_ista_comparison_nopep.pdf"))
