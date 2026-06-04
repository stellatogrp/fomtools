"""
ISTA worst-case bound: MIP vs PEPit (no SDP).

Problem:
    minimize  (1/2) ||Ds - x||^2 + lambda * ||s||_1
    D in R^{p x n},  x in [x_lb, x_ub]^p,  s0 = 0 (fixed)

Algorithm: ISTA (iterative soft-thresholding)
    s^{k+1} = T_{lambda*eta}( (I - eta*D^T*D)*s^k + eta*D^T*x )

Metric: ||s^k - s^{k-1}||_inf  (inf-norm of fixed-point residual)
    - MIP and simulation solve the inf-norm directly.
    - PEPit bounds ||s^k - s^{k-1}||_2^2; we record sqrt of that bound,
      valid since ||v||_inf <= ||v||_2.

Results are written to lasso_l2o_results.csv; plot with plot_lasso_l2o.py.

PEPit is solved with ||s0 - s*|| <= R
    MATCH_MIP_AT_K1 = True  -> R calibrated so PEPit[k=1] = MIP[k=1]
    MATCH_MIP_AT_K1 = False -> R = sqrt(n) * max(|x_ub|), geometric worst-case
"""

import csv

import numpy as np
from PEPit import PEP
from PEPit.functions import ConvexFunction, SmoothConvexFunction
from PEPit.primitive_steps import proximal_step

from fomtools.verifier import Verifier

# ── Problem setup (same as sparse_coding.py tutorial) ─────────────────────────
np.random.seed(0)
p, n, lambd = 10, 15, 1e-1

def generate_A(seed, m, n, scaling=1.0):
    """Generate sparse A matrix using NumPy.

    Args:
        seed: Random seed for reproducibility
        m: Number of rows
        n: Number of columns
        scaling: Scale parameter for the normal distribution (default 1.0).
                 Note: columns are subsequently normalized to unit norm, so
                 `scaling` only affects the distribution of the draw prior to
                 normalization; the final A always has columns of norm 1.

    Returns:
        A: (m, n) numpy array with columns normalized to unit norm
    """
    np.random.seed(seed)
    A = np.random.normal(scale=scaling / m, size=(m, n))
    col_norms = np.linalg.norm(A, axis=0, keepdims=True)
    col_norms = np.maximum(col_norms, 1e-10)
    A = A / col_norms
    return A

D_value = np.random.normal(size=(p, n))
D_mask = np.random.binomial(n=1, p=0.2, size=(p, n))
D = np.multiply(D_value, D_mask)
D = generate_A(1000, p, n)

num_star = 200
z_star_value = np.random.normal(size=(n, num_star))
z_star_mask = np.random.binomial(n=1, p=0.1, size=(n, num_star))
z_star = np.multiply(z_star_value, z_star_mask)
err = np.random.normal(scale=1e-2, size=(p, num_star))
x_samples = D @ z_star + err
x_lb, x_ub = np.min(x_samples, axis=1), np.max(x_samples, axis=1)

L = float(np.real(np.max(np.linalg.eigvals(D.T @ D))))
eta = 1.0 / L
K = 10
s0 = np.zeros(n)

print('L:', L)

solver_params = {"OutputFlag": 0, "LogToConsole": 0}
MATCH_MIP_AT_K1 = True

# ── MIP — inf-norm of fixed-point residual (MILP, Gurobi only) ────────────────
vp_mip = Verifier(method="mip", solver="gurobi", obbt_solver="gurobi",
                  params=solver_params)

x_mip = vp_mip.add_param(p, lb=x_lb, ub=x_ub)
s_mip = [s0]

mip_residuals = []
for k in range(1, K + 1):
    s_mip.append(vp_mip.step.soft_threshold(
        s_mip[k - 1] - eta * D.T @ (D @ s_mip[k - 1] - x_mip), lambd=lambd * eta))
    vp_mip.set_objective_norm([s_mip[k] - s_mip[k - 1]], ord="inf")
    mip_residuals.append(vp_mip.solve())
    print(f"MIP  k={k}: {mip_residuals[-1]:.6f}")

# ── PEPit — bounds ||s^k - s^{k-1}||_2^2; plot sqrt as inf-norm bound ─────────
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
    pepit_unit_k1 = problem.solve(solver="CLARABEL", verbose=False)
    R = mip_residuals[0] / np.sqrt(pepit_unit_k1)
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
    problem.set_initial_condition((s0_pep - xs) ** 2 <= R ** 2)

    s = [s0_pep]
    for j in range(k):
        s_next, _, _ = proximal_step(s[j] - eta * f.gradient(s[j]), g, eta)
        s.append(s_next)

    problem.set_performance_metric((s[k] - s[k - 1]) ** 2)
    pepit_residuals.append(np.sqrt(problem.solve(solver="CLARABEL", verbose=False)))
    print(f"PEPit k={k}: {pepit_residuals[-1]:.6f}  (R={R:.4f})")

# ── Simulator — inf-norm of fixed-point residual, num_star samples ────────────
vp_sim = Verifier(method="simulator")

x_sim = vp_sim.add_param(ndim=p, lb=x_lb, ub=x_ub, samples=x_samples.T)
s_sim = [s0]  # fixed initial point

for k in range(1, K + 1):
    s_sim.append(vp_sim.step.soft_threshold(
        s_sim[k - 1] - eta * D.T @ (D @ s_sim[k - 1] - x_sim), lambd=lambd * eta))
    vp_sim.set_objective_norm([s_sim[k] - s_sim[k - 1]], ord="inf")

sim_results = vp_sim.solve()
sim_residuals = sim_results.max(axis=0).tolist()
for k in range(1, K + 1):
    print(f"Sim  k={k}: {sim_residuals[k - 1]:.6f}  (max over N={num_star})")

# ── Save results to CSV (plot with plot_lasso_l2o.py) ─────────────────────────
csv_path = "lasso_l2o_results.csv"
with open(csv_path, "w", newline="") as fh:
    writer = csv.writer(fh)
    writer.writerow(["k", "mip", "pepit", "sim", "R", "num_star"])
    for k in range(1, K + 1):
        writer.writerow([k, mip_residuals[k - 1], pepit_residuals[k - 1],
                         sim_residuals[k - 1], R, num_star])

print(f"Results saved to {csv_path}")
print("Done.")