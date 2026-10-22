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
from PEPit import PEP
from PEPit.functions import ConvexFunction, SmoothConvexFunction
from PEPit.primitive_steps import proximal_step

import fomtools as ft

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

MATCH_MIP_AT_K1 = True

# ── MIP — squared 2-norm of fixed-point residual (MIQP, Gurobi only) ──────────
vp_mip = ft.Verifier(method="mip", solver="gurobi", obbt_solver="gurobi")

x_mip = vp_mip.parameter(p, lb=x_lb, ub=x_ub)
s_mip = [s0]

mip_residuals = []
x_worst_k5 = None
for k in range(1, K + 1):
    s_mip.append(
        vp_mip.soft_threshold(
            s_mip[k - 1] - eta * D.T @ (D @ s_mip[k - 1] - x_mip), lambd=lambd * eta
        )
    )
    vp_mip.maximize(ft.sum_squares(s_mip[k] - s_mip[k - 1]))
    mip_residuals.append(vp_mip.solve())
    print(f"MIP  k={k}: {mip_residuals[-1]:.6f}")
    if k == 5:
        x_worst_k5 = x_mip.value

# ── SDP — squared 2-norm of fixed-point residual ──────────────────────────────
vp_sdp = ft.Verifier(method="sdp", solver="clarabel", obbt_solver="highs")

x_sdp = vp_sdp.parameter(p, lb=x_lb, ub=x_ub)
s_sdp = [s0]

sdp_residuals = []
for k in range(1, K + 1):
    s_sdp.append(
        vp_sdp.soft_threshold(
            s_sdp[k - 1] - eta * D.T @ (D @ s_sdp[k - 1] - x_sdp), lambd=lambd * eta
        )
    )
    vp_sdp.maximize(ft.sum_squares(s_sdp[k] - s_sdp[k - 1]))
    sdp_residuals.append(vp_sdp.solve())
    print(f"SDP  k={k}: {sdp_residuals[-1]:.6f}")

# ── PEPit — squared 2-norm of fixed-point residual ────────────────────────────
if MATCH_MIP_AT_K1:
    problem = PEP()
    f = problem.declare_function(SmoothConvexFunction, L=L)
    g = problem.declare_function(ConvexFunction)
    F = f + g
    xs = F.stationary_point()
    s0_pep = problem.set_initial_point()
    problem.set_initial_condition((s0_pep - xs) ** 2 <= 1.0)
    s_next, _, _ = proximal_step(s0_pep - eta * f.gradient(s0_pep), g, eta)
    problem.set_performance_metric((s_next - s0_pep) ** 2)
    pepit_unit_k1 = problem.solve(solver="SCS", verbose=False)
    R = np.sqrt(mip_residuals[0] / pepit_unit_k1)
else:
    R = np.sqrt(n) * float(np.max(np.abs(x_ub)))

pepit_residuals = []
for k in range(1, K + 1):
    problem = PEP()
    f = problem.declare_function(SmoothConvexFunction, L=L)
    g = problem.declare_function(ConvexFunction)
    F = f + g
    xs = F.stationary_point()
    s0_pep = problem.set_initial_point()
    problem.set_initial_condition((s0_pep - xs) ** 2 <= R**2)

    s = [s0_pep]
    for j in range(k):
        s_next, _, _ = proximal_step(s[j] - eta * f.gradient(s[j]), g, eta)
        s.append(s_next)

    problem.set_performance_metric((s[k] - s[k - 1]) ** 2)
    pepit_residuals.append(problem.solve(solver="SCS", verbose=False))
    print(f"PEPit k={k}: {pepit_residuals[-1]:.6f}  (R={R:.4f})")

# ── Simulator — squared 2-norm of fixed-point residual, num_star samples ──────
vp_sim = ft.Verifier(method="simulator")

x_sim = vp_sim.parameter(ndim=p, samples=x_samples.T)
s_sim = [s0]  # fixed initial point

for k in range(1, K + 1):
    s_sim.append(
        vp_sim.soft_threshold(
            s_sim[k - 1] - eta * D.T @ (D @ s_sim[k - 1] - x_sim), lambd=lambd * eta
        )
    )
    vp_sim.maximize(ft.sum_squares(s_sim[k] - s_sim[k - 1]))

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
