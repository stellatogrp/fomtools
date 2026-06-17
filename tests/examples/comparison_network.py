"""
PDHG worst-case bound: MIP vs SDP vs PEPit.

Problem: minimum-cost network flow on a bipartite graph with n_s supply nodes
and n_d demand nodes, parameterized by demand x in [x_lb, x_ub].

Algorithm: PDHG (Chambolle-Pock / Primal-Dual Hybrid Gradient)
    z^{k+1} = prox_{I_[0,c]}( z^k - eta*(mu + A_s^T v^k - A_d^T w^k) )
    v^{k+1} = prox_{I_R+}( v^k + eta*(-b_s + A_s*(2z^{k+1} - z^k)) )
    w^{k+1} = w^k + eta*(x - A_d*(2z^{k+1} - z^k))

Metric (all methods): ||(z^k-z^{k-1}, v^k-v^{k-1}, w^k-w^{k-1})||_2^2

PEPit is solved with ||(z0,v0,w0) - (z*,v*,w*)|| <= R
    MATCH_MIP_AT_K1 = True  -> R calibrated so PEPit[k=1] = MIP[k=1]
    MATCH_MIP_AT_K1 = False -> R = 1.0 (unit ball)
"""

import cvxpy as cp
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import scipy.sparse as spa
from PEPit import PEP, Point
from PEPit.functions import ConvexFunction
from PEPit.operators import LinearOperator
from PEPit.primitive_steps import proximal_step

from fomtools.proxable_functions import IndBox, IndNonNegative
from fomtools.verifier import Verifier

# ── Problem setup (same as network_optimization.py tutorial) ──────────────────
np.random.seed(2)
n_s, n_d = 3, 2

G = nx.bipartite.random_graph(n_s, n_d, 1.0, seed=1, directed=False)
A_mat = nx.linalg.graphmatrix.incidence_matrix(G, oriented=False).todense()
A_mat[n_s:, :] *= -1
A_s = np.array(A_mat[:n_s, :])
A_d = np.array(A_mat[n_s:, :])
n_e = A_mat.shape[1]

b_s = 10. * np.ones(n_s)
c   = 5. * np.ones(n_e)
mu  = np.random.uniform(5, 10, size=n_e)

x_lb = -7. * np.ones(n_d)
x_ub = -6. * np.ones(n_d)

# Compute z0_value from LP (primal solution for max demand instance)
_A_block = spa.bmat([
    [spa.csr_matrix(A_s), spa.eye(n_s), None],
    [spa.csr_matrix(A_d), None, None],
    [spa.eye(n_e), None, spa.eye(n_e)]
]).todense()
_b_block = np.hstack([b_s, x_lb, c])
_c_block = np.zeros(_A_block.shape[1])
_c_block[:n_e] = mu

_x_block = cp.Variable(_A_block.shape[1])
_problem_lp = cp.Problem(cp.Minimize(_c_block.T @ _x_block),
                         [_A_block @ _x_block == _b_block, _x_block >= 0])
_problem_lp.solve(solver=cp.SCS)
z0_value = _x_block.value[:n_e]

K = 3
L = float(np.linalg.norm(np.hstack([A_s.T, A_d.T]), 2))
eta = 0.5 / L

solver_params = {"OutputFlag": 0, "LogToConsole": 0}
MATCH_MIP_AT_K1 = True
flow_box = IndBox(l=0.0, u=c)
dual_nonnegative = IndNonNegative()

# ── MIP — squared 2-norm of combined fixed-point residual ─────────────────────
vp_mip = Verifier(method="mip", solver="gurobi", obbt_handler="highs",
                  params=solver_params)

x_mip   = vp_mip.add_param(n_d, lb=x_lb, ub=x_ub)
mu_mip  = vp_mip.add_param(n_e, lb=mu, ub=mu)
b_s_mip = vp_mip.add_param(n_s, lb=b_s, ub=b_s)

z_mip = [vp_mip.add_iterate(n_e, lb=z0_value, ub=z0_value)]
v_mip = [vp_mip.add_iterate(n_s, lb=np.zeros(n_s), ub=np.zeros(n_s))]
w_mip = [vp_mip.add_iterate(n_d, lb=np.zeros(n_d), ub=np.zeros(n_d))]

mip_residuals = []
for k in range(1, K + 1):
    z_new = vp_mip.prox(
        flow_box,
        z_mip[k - 1] - eta * (mu_mip + A_s.T @ v_mip[k - 1] - A_d.T @ w_mip[k - 1]))
    v_new = vp_mip.prox(
        dual_nonnegative,
        v_mip[k - 1] + eta * (-b_s_mip + A_s @ (2 * z_new - z_mip[k - 1])))
    w_new = w_mip[k - 1] + eta * (x_mip - A_d @ (2 * z_new - z_mip[k - 1]))
    z_mip.append(z_new)
    v_mip.append(v_new)
    w_mip.append(w_new)
    vp_mip.set_objective_norm(
        [z_mip[k] - z_mip[k - 1], v_mip[k] - v_mip[k - 1], w_mip[k] - w_mip[k - 1]],
        ord=2)
    mip_residuals.append(vp_mip.solve())
    print(f"MIP  k={k}: {mip_residuals[-1]:.6f}")

# ── SDP — squared 2-norm of combined fixed-point residual ─────────────────────
vp_sdp = Verifier(method="sdp", solver="clarabel", obbt_handler="highs")

x_sdp   = vp_sdp.add_param(n_d, lb=x_lb, ub=x_ub)
mu_sdp  = vp_sdp.add_param(n_e, lb=mu, ub=mu)
b_s_sdp = vp_sdp.add_param(n_s, lb=b_s, ub=b_s)

z_sdp = [vp_sdp.add_iterate(n_e, lb=z0_value, ub=z0_value)]
v_sdp = [vp_sdp.add_iterate(n_s, lb=np.zeros(n_s), ub=np.zeros(n_s))]
w_sdp = [vp_sdp.add_iterate(n_d, lb=np.zeros(n_d), ub=np.zeros(n_d))]

sdp_residuals = []
for k in range(1, K + 1):
    z_new = vp_sdp.prox(
        flow_box,
        z_sdp[k - 1] - eta * (mu_sdp + A_s.T @ v_sdp[k - 1] - A_d.T @ w_sdp[k - 1]))
    v_new = vp_sdp.prox(
        dual_nonnegative,
        v_sdp[k - 1] + eta * (-b_s_sdp + A_s @ (2 * z_new - z_sdp[k - 1])))
    w_new = w_sdp[k - 1] + eta * (x_sdp - A_d @ (2 * z_new - z_sdp[k - 1]))
    z_sdp.append(z_new)
    v_sdp.append(v_new)
    w_sdp.append(w_new)
    vp_sdp.set_objective_norm(
        [z_sdp[k] - z_sdp[k - 1], v_sdp[k] - v_sdp[k - 1], w_sdp[k] - w_sdp[k - 1]],
        ord=2)
    sdp_residuals.append(vp_sdp.solve())
    print(f"SDP  k={k}: {sdp_residuals[-1]:.6f}")

# ── PEPit — squared 2-norm of combined fixed-point residual ───────────────────
def _pepit_pdhg_k1(R_val):
    problem = PEP()
    f = problem.declare_function(ConvexFunction)
    g = problem.declare_function(ConvexFunction)
    linear = problem.declare_function(LinearOperator, L=L)
    xs = Point()
    us = Point()
    f.add_point((xs, -linear.T.gradient(us), f(xs)))
    g.add_point((us, linear.gradient(xs), g(us)))
    x0 = problem.set_initial_point()
    u0 = problem.set_initial_point()
    problem.set_initial_condition((x0 - xs) ** 2 + (u0 - us) ** 2 <= R_val ** 2)
    x1, _, _ = proximal_step(x0 - eta * linear.T.gradient(u0), f, eta)
    u1, _, _ = proximal_step(u0 + eta * linear.gradient(2 * x1 - x0), g, eta)
    problem.set_performance_metric((x1 - x0) ** 2 + (u1 - u0) ** 2)
    return problem.solve(solver="SCS", verbose=False)


if MATCH_MIP_AT_K1:
    pepit_unit_k1 = _pepit_pdhg_k1(1.0)
    R = np.sqrt(mip_residuals[0] / pepit_unit_k1)
else:
    R = 1.0

pepit_residuals = []
for k in range(1, K + 1):
    problem = PEP()
    f = problem.declare_function(ConvexFunction)
    g = problem.declare_function(ConvexFunction)
    linear = problem.declare_function(LinearOperator, L=L)
    xs = Point()
    us = Point()
    f.add_point((xs, -linear.T.gradient(us), f(xs)))
    g.add_point((us, linear.gradient(xs), g(us)))
    x0 = problem.set_initial_point()
    u0 = problem.set_initial_point()
    problem.set_initial_condition((x0 - xs) ** 2 + (u0 - us) ** 2 <= R ** 2)

    x = [x0]
    u = [u0]
    for j in range(k):
        x_next, _, _ = proximal_step(x[-1] - eta * linear.T.gradient(u[-1]), f, eta)
        u_next, _, _ = proximal_step(u[-1] + eta * linear.gradient(2 * x_next - x[-1]), g, eta)
        x.append(x_next)
        u.append(u_next)

    problem.set_performance_metric((x[k] - x[k - 1]) ** 2 + (u[k] - u[k - 1]) ** 2)
    pepit_residuals.append(problem.solve(solver="SCS", verbose=False))
    print(f"PEPit k={k}: {pepit_residuals[-1]:.6f}  (R={R:.4f})")

# ── Simulator — squared 2-norm, N random demand samples from [x_lb, x_ub] ────
N = 200
rng = np.random.default_rng(0)
x_samples = rng.uniform(x_lb, x_ub, size=(N, n_d))

vp_sim = Verifier(method="simulator")
x_sim = vp_sim.add_param(n_d, lb=x_lb, ub=x_ub, samples=x_samples)

z_sim = [vp_sim.add_iterate(n_e, lb=z0_value, ub=z0_value)]
v_sim = [vp_sim.add_iterate(n_s, lb=np.zeros(n_s), ub=np.zeros(n_s))]
w_sim = [vp_sim.add_iterate(n_d, lb=np.zeros(n_d), ub=np.zeros(n_d))]

for k in range(1, K + 1):
    z_new = vp_sim.prox(
        flow_box,
        z_sim[k - 1] - eta * (mu + A_s.T @ v_sim[k - 1] - A_d.T @ w_sim[k - 1]))
    v_new = vp_sim.prox(
        dual_nonnegative,
        v_sim[k - 1] + eta * (-b_s + A_s @ (2 * z_new - z_sim[k - 1])))
    w_new = w_sim[k - 1] + eta * (x_sim - A_d @ (2 * z_new - z_sim[k - 1]))
    z_sim.append(z_new)
    v_sim.append(v_new)
    w_sim.append(w_new)
    vp_sim.set_objective_norm(
        [z_sim[k] - z_sim[k - 1], v_sim[k] - v_sim[k - 1], w_sim[k] - w_sim[k - 1]],
        ord=2)

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
plt.ylabel(r"Worst-case $\|(z^k-z^{k-1},\,v^k-v^{k-1},\,w^k-w^{k-1})\|_2^2$")
plt.title(f"Network PDHG: MIP vs SDP vs PEPit ({K} iterations)")
plt.legend()
plt.tight_layout()
plt.show()
print("Done.")
