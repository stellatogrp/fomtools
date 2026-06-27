"""
ISTA worst-case bound: MIP vs SDP vs PEPit.
Reproduces the three verification approaches for the Lasso example (Section 5.3).

Problem:
    minimize  (1/2) ||Ds - x||^2 + lambda * ||s||_1
    D in R^{p x n},  x in [x_lb, x_ub]^p,  s0 = 0 (fixed)

Algorithm: ISTA (iterative soft-thresholding)
    s^{k+1} = T_{lambda*eta}( (I - eta*D^T*D)*s^k + eta*D^T*x )

Metric (all methods): ||s^k - s^{k-1}||^2_2  (squared 2-norm of fixed-point residual)

PEPit is solved with ||s0 - s*|| <= R
    MATCH_MIP_AT_K1 = True  -> R calibrated so PEPit[k=1] = MIP[k=1]
    MATCH_MIP_AT_K1 = False -> R = sqrt(n) * max(|x_ub|), geometric worst-case
"""

import matplotlib.pyplot as plt
import numpy as np

from docs.tutorials.helper_functions.pepit_runner import PEPitRunner
from fomtools.verifier import Verifier

# ── Problem setup (same as sparse_coding.py tutorial) ─────────────────────────
np.random.seed(0)
p, n, lambd = 3, 4, 1e-1

D_value = np.random.normal(size=(p, n))
D_mask = np.random.binomial(n=1, p=0.2, size=(p, n))
D = np.multiply(D_value, D_mask)

num_star = 200
z_star_value = np.random.normal(size=(n, num_star))
z_star_mask = np.random.binomial(n=1, p=0.1, size=(n, num_star))
z_star = np.multiply(z_star_value, z_star_mask)
err = np.random.normal(scale=1e-2, size=(p, num_star))
x_samples = D @ z_star + err
x_lb, x_ub = np.min(x_samples, axis=1), np.max(x_samples, axis=1)

L = float(np.real(np.max(np.linalg.eigvals(D.T @ D))))
eta = 1.0 / L
K = 5
s0 = np.zeros(n)

solver_params = {"OutputFlag": 0, "LogToConsole": 0}
MATCH_MIP_AT_K1 = True

# ── MIP — squared 2-norm of fixed-point residual (MIQP, Gurobi only) ──────────
vp_mip = Verifier(method="mip", solver="gurobi", obbt_solver="gurobi", params=solver_params)

x_mip = vp_mip.add_param(p, lb=x_lb, ub=x_ub)
s_mip = [s0]

mip_residuals = []
x_worst_k5 = None
for k in range(1, K + 1):
    s_mip.append(
        vp_mip.step.soft_threshold(
            s_mip[k - 1] - eta * D.T @ (D @ s_mip[k - 1] - x_mip), lambd=lambd * eta
        )
    )
    vp_mip.set_objective_norm([s_mip[k] - s_mip[k - 1]], ord=2)
    mip_residuals.append(vp_mip.solve())
    print(f"MIP  k={k}: {mip_residuals[-1]:.6f}")
    if k == 5:
        x_worst_k5 = vp_mip.get_solution_value(x_mip)

# ── SDP — squared 2-norm of fixed-point residual ──────────────────────────────
vp_sdp = Verifier(method="sdp", solver="clarabel", obbt_solver="highs")

x_sdp = vp_sdp.add_param(p, lb=x_lb, ub=x_ub)
s_sdp = [s0]

sdp_residuals = []
for k in range(1, K + 1):
    s_sdp.append(
        vp_sdp.step.soft_threshold(
            s_sdp[k - 1] - eta * D.T @ (D @ s_sdp[k - 1] - x_sdp), lambd=lambd * eta
        )
    )
    vp_sdp.set_objective_norm([s_sdp[k] - s_sdp[k - 1]], ord=2)
    sdp_residuals.append(vp_sdp.solve())
    print(f"SDP  k={k}: {sdp_residuals[-1]:.6f}")

# ── PEPit — squared 2-norm of fixed-point residual ────────────────────────────
pepit_runner = PEPitRunner()
pepit_residuals, R = pepit_runner.proximal_gradient_residuals(
    K=K,
    L=L,
    eta=eta,
    R=None if MATCH_MIP_AT_K1 else np.sqrt(n) * float(np.max(np.abs(x_ub))),
    calibrate_to=mip_residuals[0] if MATCH_MIP_AT_K1 else None,
    nonsmooth="convex",
)
for k in range(1, K + 1):
    print(f"PEPit k={k}: {pepit_residuals[k - 1]:.6f}  (R={R:.4f})")

# ── Simulator — squared 2-norm of fixed-point residual, num_star samples ──────
vp_sim = Verifier(method="simulator")

x_sim = vp_sim.add_param(ndim=p, lb=x_lb, ub=x_ub, samples=x_samples.T)
s_sim = [s0]  # fixed initial point

for k in range(1, K + 1):
    s_sim.append(
        vp_sim.step.soft_threshold(
            s_sim[k - 1] - eta * D.T @ (D @ s_sim[k - 1] - x_sim), lambd=lambd * eta
        )
    )
    vp_sim.set_objective_norm([s_sim[k] - s_sim[k - 1]], ord=2)

sim_results = vp_sim.solve()
sim_residuals = sim_results.max(axis=0).tolist()
for k in range(1, K + 1):
    print(f"Sim  k={k}: {sim_residuals[k - 1]:.6f}  (max over N={num_star})")


# ── Simulate worst-case instance from MIP k=4 ────────────────────────────────
def _soft_threshold(v, threshold):
    return np.sign(v) * np.maximum(np.abs(v) - threshold, 0)


s_wc = [s0.copy()]
for k in range(1, K + 1):
    v = s_wc[-1] - eta * D.T @ (D @ s_wc[-1] - x_worst_k5)
    s_wc.append(_soft_threshold(v, lambd * eta))

wc_residuals = [float(np.sum((s_wc[k] - s_wc[k - 1]) ** 2)) for k in range(1, K + 1)]
print(f"\nWorst-case x (MIP k=4): {x_worst_k5}")
for k in range(1, K + 1):
    print(f"WC sim k={k}: {wc_residuals[k - 1]:.6f}")

# ── Plot ───────────────────────────────────────────────────────────────────────
iters = range(1, K + 1)

plt.figure(figsize=(7, 4))
plt.plot(iters, mip_residuals, "o-", label="MIP")
plt.plot(iters, sdp_residuals, "s--", label="SDP")
plt.plot(iters, pepit_residuals, "^:", label="PEPit")
plt.plot(iters, wc_residuals, "x:", label="WC instance (MIP k=5)")
plt.plot(iters, sim_residuals, "d-.", label=f"Simulation (max over N={num_star})")
plt.yscale("log")
plt.xlabel(r"Iteration $k$")
plt.ylabel(r"Worst-case $\|s^k - s^{k-1}\|_2^2$")
plt.title(f"Lasso ISTA: MIP vs SDP vs PEPit ({K} iterations)")
plt.legend()
plt.tight_layout()
plt.show()
print("Done.")
