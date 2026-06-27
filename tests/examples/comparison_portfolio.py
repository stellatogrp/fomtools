"""
ADMM worst-case bound: MIP vs SDP vs PEPit.

Problem: mean-variance portfolio optimization
    minimize  (1/2) x^T Sigma x - mu^T x
    subject to  0 <= x_i <= u_max,  1^T x = 1

Algorithm: OSQP-style ADMM with stacked constraint matrix A = [I; 1^T]:
    w^{k+1} = Pi_{[l, u]}(v^k)                                       [satlin]
    (Sigma + sigma*I + rho*A^T A) x^{k+1}
        = sigma*x^k + mu + rho*A^T(2w^{k+1} - v^k)       [implicit linear]
    v^{k+1} = v^k + A x^{k+1} - w^{k+1}

Metric (all methods): ||(x^k - x^{k-1}, v^k - v^{k-1})||_2^2


MATCH_MIP_AT_K1 = True  -> R calibrated so PEPit[k=1] = MIP[k=1]
MATCH_MIP_AT_K1 = False -> R = 1.0
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import csr_matrix

from docs.tutorials.helper_functions.pepit_runner import PEPitRunner
from fomtools.verifier import Verifier

# ── Problem setup ──────────────────────────────────────────────────────────────
np.random.seed(0)
n = 4  # number of assets

_F = np.random.normal(size=(n, n))
Sigma = (_F @ _F.T) / n + 0.1 * np.eye(n)  # random PSD covariance

mu = np.random.uniform(0.05, 0.15, size=n)  # expected returns
u_max = 0.6 * np.ones(n)                    # position limits

# A = [I; 1^T],  l = [0_n; 1],  u = [u_max; 1]
A_mat = np.vstack([np.eye(n), np.ones((1, n))])   # shape (n+1, n)
l_val = np.hstack([np.zeros(n), [1.0]])
u_val = np.hstack([u_max, [1.0]])

sigma = 1e-4
rho = 1.0
K = 3

solver_params = {"OutputFlag": 0, "LogToConsole": 0}
MATCH_MIP_AT_K1 = True

lhs_mat = csr_matrix(Sigma + sigma * np.eye(n) + rho * A_mat.T @ A_mat)
x0_val = np.zeros(n)  # fixed initial primal

# ── MIP — squared 2-norm of combined fixed-point residual ─────────────────────
vp_mip = Verifier(method="mip", solver="gurobi", obbt_handler="highs",
                  params=solver_params)

x_mip = [x0_val]
v_mip = [vp_mip.add_iterate(n + 1, lb=np.zeros(n + 1), ub=np.ones(n + 1))]
w_mip = [None]

mip_residuals = []
for k in range(K):
    w_mip.append(vp_mip.step.satlin(v_mip[k], l=l_val, u=u_val))
    rhs_k = sigma * x_mip[k] + mu + rho * A_mat.T @ (2 * w_mip[k + 1] - v_mip[k])
    x_mip.append(vp_mip.step.implicit_linear(rhs=rhs_k, lhs_mat=lhs_mat))
    v_mip.append(v_mip[k] + (A_mat @ x_mip[k + 1] - w_mip[k + 1]))
    vp_mip.set_objective_norm(
        [x_mip[k + 1] - x_mip[k], v_mip[k + 1] - v_mip[k]], ord=2)
    mip_residuals.append(vp_mip.solve())
    print(f"MIP  k={k + 1}: {mip_residuals[-1]:.6f}")

# ── SDP — squared 2-norm of combined fixed-point residual ─────────────────────
vp_sdp = Verifier(method="sdp", solver="clarabel", obbt_handler="highs")

x_sdp = [x0_val]
v_sdp = [vp_sdp.add_iterate(n + 1, lb=np.zeros(n + 1), ub=np.ones(n + 1))]
w_sdp = [None]

sdp_residuals = []
for k in range(K):
    w_sdp.append(vp_sdp.step.satlin(v_sdp[k], l=l_val, u=u_val))
    rhs_k = sigma * x_sdp[k] + mu + rho * A_mat.T @ (2 * w_sdp[k + 1] - v_sdp[k])
    x_sdp.append(vp_sdp.step.implicit_linear(rhs=rhs_k, lhs_mat=lhs_mat))
    v_sdp.append(v_sdp[k] + (A_mat @ x_sdp[k + 1] - w_sdp[k + 1]))
    vp_sdp.set_objective_norm(
        [x_sdp[k + 1] - x_sdp[k], v_sdp[k + 1] - v_sdp[k]], ord=2)
    sdp_residuals.append(vp_sdp.solve())
    print(f"SDP  k={k + 1}: {sdp_residuals[-1]:.6f}")

# ── PEPit — squared 2-norm of combined fixed-point residual ───────────────────
# DRS/ADMM bound on f + g with f = smooth strongly convex (quadratic Sigma),
# g = convex indicator of constraint set.
Sigma_eigs = np.linalg.eigvalsh(Sigma)
mu_f = float(Sigma_eigs.min())
L_f = float(Sigma_eigs.max())

pepit_runner = PEPitRunner()
pepit_residuals, R = pepit_runner.admm_residuals(
    K=K,
    mu=mu_f,
    L=L_f,
    R=None if MATCH_MIP_AT_K1 else 1.0,
    calibrate_to=mip_residuals[0] if MATCH_MIP_AT_K1 else None,
)
for k in range(1, K + 1):
    print(f"PEPit k={k}: {pepit_residuals[k - 1]:.6f}  (R={R:.4f})")

# ── Simulator — squared 2-norm, N random initial dual conditions ──────────────
N = 200
rng = np.random.default_rng(0)
v0_samples = rng.uniform(0, 1, size=(N, n + 1)).astype(np.float32)

vp_sim = Verifier(method="simulator")
v0_sim = vp_sim.add_param(n + 1, lb=np.zeros(n + 1), ub=np.ones(n + 1), samples=v0_samples)

x_sim = [x0_val]
v_sim = [v0_sim]
w_sim = [None]

for k in range(K):
    w_sim.append(vp_sim.step.satlin(v_sim[k], l=l_val, u=u_val))
    rhs_k = sigma * x_sim[k] + mu + rho * A_mat.T @ (2 * w_sim[k + 1] - v_sim[k])
    x_sim.append(vp_sim.step.implicit_linear(rhs=rhs_k, lhs_mat=lhs_mat))
    v_sim.append(v_sim[k] + (A_mat @ x_sim[k + 1] - w_sim[k + 1]))
    vp_sim.set_objective_norm(
        [x_sim[k + 1] - x_sim[k], v_sim[k + 1] - v_sim[k]], ord=2)

sim_results = vp_sim.solve()
sim_residuals = sim_results.max(axis=0).tolist()
for k in range(1, K + 1):
    print(f"Sim  k={k}: {sim_residuals[k - 1]:.6f}  (max over N={N})")

# ── Plot ───────────────────────────────────────────────────────────────────────
iters = range(1, K + 1)

plt.figure(figsize=(7, 4))
plt.plot(iters, mip_residuals,   "o-",  label="MIP")
plt.plot(iters, sdp_residuals,   "s--", label="SDP")
plt.plot(iters, pepit_residuals, "^:",  label="PEPit")
plt.plot(iters, sim_residuals,   "d-.", label=f"Simulation (max over N={N})")
plt.yscale("log")
plt.xlabel(r"Iteration $k$")
plt.ylabel(r"Worst-case $\|(x^k-x^{k-1},\,v^k-v^{k-1})\|_2^2$")
plt.title(f"Portfolio ADMM: MIP vs SDP vs PEPit ({K} iterations)")
plt.legend()
plt.tight_layout()
plt.show()
print("Done.")
