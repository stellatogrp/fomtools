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

import numpy as np
import matplotlib.pyplot as plt
from PEPit import PEP
from PEPit.functions import SmoothConvexFunction, ConvexIndicatorFunction
from PEPit.primitive_steps import proximal_step

from fomtools.verifier import Verifier

# ── Problem setup (identical to NNLS.py tutorial) ────────────────────────────
np.random.seed(0)
m, n = 5, 3
A = np.random.normal(size=(m, n))

x_lb, x_ub = 0.0, 0.5
z0_lb, z0_ub = 5.0, 5.1

L = float(np.max(np.linalg.eigvals(A.T @ A)))
eta = 1.0 / L
K = 5

solver_params = {"OutputFlag": 0, "LogToConsole": 0}

MATCH_MIP_AT_K1 = False

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
vp_sdp = Verifier(method="sdp", solver="clarabel", obbt_handler="highs")

x_sdp = vp_sdp.add_param(m, lb=x_lb, ub=x_ub)
z_sdp = [vp_sdp.add_iterate(n, lb=z0_lb, ub=z0_ub)]

sdp_residuals = []
for k in range(1, K + 1):
    z_sdp.append(vp_sdp.step.relu(z_sdp[k - 1] - eta * A.T @ (A @ z_sdp[k - 1] - x_sdp)))
    vp_sdp.set_objective_norm([z_sdp[k] - z_sdp[k - 1]], ord=2)
    sdp_residuals.append(vp_sdp.solve())
    print(f"SDP  k={k}: {sdp_residuals[-1]:.6f}")

# ── PEPit — squared 2-norm of fixed-point residual ───────────────────────────
if MATCH_MIP_AT_K1:
    # Solve k=1 with R=1 to calibrate R, then re-solve all k with the true R.
    problem = PEP()
    f = problem.declare_function(SmoothConvexFunction, L=L)
    g = problem.declare_function(ConvexIndicatorFunction)
    z_star = (f + g).stationary_point()
    z0_pep = problem.set_initial_point()
    problem.set_initial_condition((z0_pep - z_star) ** 2 <= 1.0)
    z_next, _, _ = proximal_step(z0_pep - eta * f.gradient(z0_pep), g, eta)
    problem.set_performance_metric((z_next - z0_pep) ** 2)
    pepit_unit_k1 = problem.solve(solver="SCS", verbose=False)
    R = np.sqrt(mip_residuals[0] / pepit_unit_k1)
else:
    R = z0_ub * np.sqrt(n)

pepit_residuals = []
for k in range(1, K + 1):
    problem = PEP()
    f = problem.declare_function(SmoothConvexFunction, L=L)
    g = problem.declare_function(ConvexIndicatorFunction)

    z_star = (f + g).stationary_point()
    z0_pep = problem.set_initial_point()
    problem.set_initial_condition((z0_pep - z_star) ** 2 <= R ** 2)

    z = [z0_pep]
    for j in range(1, k + 1):
        z_next, _, _ = proximal_step(z[j - 1] - eta * f.gradient(z[j - 1]), g, eta)
        z.append(z_next)

    problem.set_performance_metric((z[k] - z[k - 1]) ** 2)
    pepit_residuals.append(problem.solve(solver="SCS", verbose=False))
    print(f"PEPit k={k}: {pepit_residuals[-1]:.6f}  (R={R:.4f})")

# ── Plot ──────────────────────────────────────────────────────────────────────
iters = range(1, K + 1)

plt.figure(figsize=(7, 4))
plt.plot(iters, mip_residuals,   "o-",  label="MIP")
plt.plot(iters, sdp_residuals,   "s--", label="SDP")
plt.plot(iters, pepit_residuals, "^:",  label="PEPit")
plt.yscale("log")
plt.xlabel(r"Iteration $k$")
plt.ylabel(r"Worst-case $\|z^k - z^{k-1}\|_2^2$")
plt.title(f"NNLS: MIP vs SDP vs PEPit ({K} iterations)")
plt.legend()
plt.tight_layout()
plt.savefig("nnls_sdp_vs_mip.png", dpi=150)
plt.show()
print("Plot saved to nnls_sdp_vs_mip.png")
