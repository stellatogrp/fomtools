"""
NNLS worst-case bound: MIP vs SDP vs PEPit.
Reproduces the three verification approaches from docs/tutorials/NNLS.py
on a single plot.

Problem:
    minimize  (1/2) ||Az - x||^2   s.t. z >= 0
    A in R^{m x n},  x in [x_lb, x_ub]^m,  z0 in [z0_lb, z0_ub]^n

Algorithm: projected gradient descent
    z^{k+1} = ReLU(z^k - eta * A^T (A z^k - x))

Metric (all methods): ||z^k - z^{k-1}||^2_2  (squared 2-norm of fixed-point residual)

PEPit is solved with ||z0 - z*|| <= R 
    MATCH_MIP_AT_K1 = True  → R calibrated so PEPit[k=1] = MIP[k=1] (curves share the same k=1 value)
    MATCH_MIP_AT_K1 = False → R = z0_ub * sqrt(n), the geometric worst-case distance implied by the box bounds
"""

import matplotlib.pyplot as plt
import numpy as np

from docs.tutorials.pepit_runner import PEPitRunner
from fomtools.verifier import Verifier

# ── Problem setup (identical to NNLS.py tutorial) ────────────────────────────
np.random.seed(0)
m, n = 5, 3
A = np.random.normal(size=(m, n))

x_lb, x_ub = 0.0, 0.25
z0_lb, z0_ub = 5.0, 5.1

L = float(np.max(np.linalg.eigvals(A.T @ A)))
eta = 1.0 / L
K = 5
N = 10  # number of simulation samples

solver_params = {"OutputFlag": 0, "LogToConsole": 0}

MATCH_MIP_AT_K1 = False
RLT_CUTS = True
TRIANGLE_CUTS = True

# ── MIP — squared 2-norm of fixed-point residual (MIQP, Gurobi only) ─────────
vp_mip = Verifier(method="mip", solver="gurobi", obbt_handler="highs",
                  params=solver_params)

x_mip = vp_mip.add_param(m, lb=x_lb, ub=x_ub)
z_mip = [vp_mip.add_iterate(n, lb=z0_lb, ub=z0_ub)]

mip_residuals = []
for k in range(1, K + 1):
    z_mip.append(vp_mip.step.relu(z_mip[k - 1] - eta * A.T @ (A @ z_mip[k - 1] - x_mip)))
    vp_mip.set_objective_norm([z_mip[k] - z_mip[k - 1]], ord=2)
    mip_residuals.append(vp_mip.solve())
    print(f"MIP  k={k}: {mip_residuals[-1]:.6f}")

# ── SDP — 2-norm of fixed-point residual ─────────────────────────────────────
vp_sdp = Verifier(method="sdp",
                  solver="clarabel",
                  obbt_handler="highs",
                  rlt_cuts=RLT_CUTS,
                  triangle_cuts=TRIANGLE_CUTS)

x_sdp = vp_sdp.add_param(m, lb=x_lb, ub=x_ub)
z_sdp = [vp_sdp.add_iterate(n, lb=z0_lb, ub=z0_ub)]

sdp_residuals = []
for k in range(1, K + 1):
    z_sdp.append(vp_sdp.step.relu(z_sdp[k - 1] - eta * A.T @ (A @ z_sdp[k - 1] - x_sdp)))
    vp_sdp.set_objective_norm([z_sdp[k] - z_sdp[k - 1]], ord=2)
    sdp_residuals.append(vp_sdp.solve())
    print(f"SDP  k={k}: {sdp_residuals[-1]:.6f}")

# ── PEPit — squared 2-norm of fixed-point residual ───────────────────────────
pepit_runner = PEPitRunner()
pepit_residuals, R = pepit_runner.proximal_gradient_residuals(
    K=K,
    L=L,
    eta=eta,
    R=None if MATCH_MIP_AT_K1 else z0_ub * np.sqrt(n),
    calibrate_to=mip_residuals[0] if MATCH_MIP_AT_K1 else None,
    nonsmooth="indicator",
)
for k in range(1, K + 1):
    print(f"PEPit k={k}: {pepit_residuals[k - 1]:.6f}  (R={R:.4f})")

# ── Simulation — squared 2-norm of fixed-point residual, N random samples ───
vp_sim = Verifier(method="simulator")

rng = np.random.default_rng(0)
x_samples = rng.uniform(low=x_lb, high=x_ub, size=(N, m))
x_sim = vp_sim.add_param(ndim=m, lb=x_lb, ub=x_ub, samples=x_samples)
z_sim = [vp_sim.add_iterate(ndim=n, lb=z0_lb, ub=z0_ub,
                            init=np.full(n, (z0_lb + z0_ub) / 2))]

for k in range(1, K + 1):
    z_sim.append(vp_sim.step.relu(z_sim[k - 1] - eta * A.T @ (A @ z_sim[k - 1] - x_sim)))
    vp_sim.set_objective_norm([z_sim[k] - z_sim[k - 1]], ord=2)

sim_results = vp_sim.solve()
sim_residuals = sim_results.max(axis=0).tolist()
for k in range(1, K + 1):
    print(f"Sim  k={k}: {sim_residuals[k - 1]:.6f}  (max over N={N})")

# ── Plot ──────────────────────────────────────────────────────────────────────
iters = range(1, K + 1)

plt.figure(figsize=(7, 4))
plt.plot(iters, mip_residuals,   "o-",  label="MIP")
plt.plot(iters, sdp_residuals,   "s--", label="SDP")
plt.plot(iters, pepit_residuals, "^:",  label="PEPit")
plt.plot(iters, sim_residuals,   "d-.", label=f"Simulation (max over N={N})")
plt.yscale("log")
plt.xlabel(r"Iteration $k$")
plt.ylabel(r"Worst-case $\|z^k - z^{k-1}\|_2^2$")
plt.title(f"NNLS: MIP vs SDP vs PEPit ({K} iterations)")
plt.legend()
plt.tight_layout()
# plt.savefig("nnls_sdp_vs_mip.png", dpi=150)
plt.show()
# print("Plot saved to nnls_sdp_vs_mip.png")
