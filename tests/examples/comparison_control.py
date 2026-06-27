"""
ADMM worst-case bound: MIP vs SDP vs PEPit.

Problem: input-constrained linear quadratic optimal control solved via ADMM
(scalar rho, no over-relaxation, as in OSQP).

Algorithm iterates z = (x, v):
    w^{k+1} = Pi_{[l(theta), u(theta)]}(v^k)            [satlin step]
    (P + sigma*I + rho*M^T*M) x^{k+1} = sigma*x^k + rho*M^T*(2w^{k+1} - v^k)
    v^{k+1} = v^k + M*x^{k+1} - w^{k+1}

Metric (all methods): ||(x^k-x^{k-1}, v^k-v^{k-1})||_2^2

PEPit is solved with ||(z0-z*)|| <= R
    MATCH_MIP_AT_K1 = True  -> R calibrated so PEPit[k=1] = MIP[k=1]
    MATCH_MIP_AT_K1 = False -> R = 1.0
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as spa
from scipy.sparse import csr_matrix

from docs.tutorials.helper_functions.pepit_runner import PEPitRunner
from fomtools.verifier import Verifier

# ── Problem setup (same as optimal_control.py tutorial) ───────────────────────
A_dym = np.array([
    [1.,      0.,     0., 0., 0., 0., 0.1,     0.,     0.,  0.,     0.,     0.    ],
    [0.,      1.,     0., 0., 0., 0., 0.,      0.1,    0.,  0.,     0.,     0.    ],
    [0.,      0.,     1., 0., 0., 0., 0.,      0.,     0.1, 0.,     0.,     0.    ],
    [0.0488,  0.,     0., 1., 0., 0., 0.0016,  0.,     0.,  0.0992, 0.,     0.    ],
    [0.,     -0.0488, 0., 0., 1., 0., 0.,     -0.0016, 0.,  0.,     0.0992, 0.    ],
    [0.,      0.,     0., 0., 0., 1., 0.,      0.,     0.,  0.,     0.,     0.0992],
    [0.,      0.,     0., 0., 0., 0., 1.,      0.,     0.,  0.,     0.,     0.    ],
    [0.,      0.,     0., 0., 0., 0., 0.,      1.,     0.,  0.,     0.,     0.    ],
    [0.,      0.,     0., 0., 0., 0., 0.,      0.,     1.,  0.,     0.,     0.    ],
    [0.9734,  0.,     0., 0., 0., 0., 0.0488,  0.,     0.,  0.9846, 0.,     0.    ],
    [0.,     -0.9734, 0., 0., 0., 0., 0.,     -0.0488, 0.,  0.,     0.9846, 0.    ],
    [0.,      0.,     0., 0., 0., 0., 0.,      0.,     0.,  0.,     0.,     0.9846]
])
B_dym = np.array([
    [0.,      -0.0726,  0.,     0.0726],
    [-0.0726,  0.,      0.0726, 0.    ],
    [-0.0152,  0.0152, -0.0152, 0.0152],
    [-0.,     -0.0006, -0.,     0.0006],
    [0.0006,   0.,     -0.0006, 0.0000],
    [0.0106,   0.0106,  0.0106, 0.0106],
    [0,       -1.4512,  0.,     1.4512],
    [-1.4512,  0.,      1.4512, 0.    ],
    [-0.3049,  0.3049, -0.3049, 0.3049],
    [-0.,     -0.0236,  0.,     0.0236],
    [0.0236,   0.,     -0.0236, 0.    ],
    [0.2107,   0.2107,  0.2107, 0.2107]
])
n_s, n_u = B_dym.shape
T = 2

Q = spa.diags([0., 0., 10., 10., 10., 10., 0., 0., 0., 5., 5., 5.])
R_mat = 0.1 * spa.eye(n_u)

x_init = 2.0 * np.array([1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
u_min = -0.99 * np.ones(n_u)
u_max =  2.41 * np.ones(n_u)

# Build QP matrices: min (1/2) xz^T P xz  s.t. u_stack_min <= u_stack <= u_stack_max
n_z = (T - 1) * n_u
n_xz = n_s + n_z

SX = spa.vstack([spa.linalg.matrix_power(A_dym, i) for i in range(T)])
SZ = spa.csc_matrix(((T - 1) * n_s, (T - 1) * n_u))
for i in range(T - 1):
    AiB = spa.linalg.matrix_power(A_dym, i) @ B_dym
    SZ += spa.kron(spa.eye(T - 1, k=-i), AiB)
SZ = spa.vstack([spa.csc_matrix((n_s, n_z)), SZ])

P_block = spa.kron(spa.eye(T), Q)
R_block = spa.kron(spa.eye(T - 1), R_mat)

P = spa.bmat([
    [SX.T @ P_block @ SX, SX.T @ P_block @ SZ],
    [SZ.T @ P_block @ SX, SZ.T @ P_block @ SZ + R_block]
])
q = np.zeros(n_xz)
M = spa.eye(n_xz)

u_stack_min = np.kron(np.ones(T - 1), u_min)
u_stack_max = np.kron(np.ones(T - 1), u_max)

l_x_val = np.hstack([x_init, u_stack_min])
u_x_val = np.hstack([x_init, u_stack_max])

sigma = 1e-4
rho = 1.0
K = 2

solver_params = {"OutputFlag": 0, "LogToConsole": 0}
MATCH_MIP_AT_K1 = True

lhs_mat = csr_matrix(P + sigma * spa.eye(n_xz) + rho * (M.T @ M))

# ── MIP — squared 2-norm of combined fixed-point residual ─────────────────────
vp_mip = Verifier(method="mip", solver="gurobi", obbt_handler="highs",
                  params=solver_params)

z_mip = [np.zeros(n_xz)]
v_mip = [vp_mip.add_iterate(n_xz, lb=np.zeros(n_xz), ub=np.ones(n_xz))]
w_mip = [None]

mip_residuals = []
for k in range(K):
    w_mip.append(vp_mip.step.satlin(v_mip[k], l=l_x_val, u=u_x_val))
    rhs_k = sigma * z_mip[k] - q + rho * M.T @ (2 * w_mip[k + 1] - v_mip[k])
    z_mip.append(vp_mip.step.implicit_linear(rhs=rhs_k, lhs_mat=lhs_mat))
    v_mip.append(v_mip[k] + (M @ z_mip[k + 1] - w_mip[k + 1]))
    vp_mip.set_objective_norm(
        [z_mip[k + 1] - z_mip[k], v_mip[k + 1] - v_mip[k]], ord=2)
    mip_residuals.append(vp_mip.solve())
    print(f"MIP  k={k+1}: {mip_residuals[-1]:.6f}")

# ── SDP — squared 2-norm of combined fixed-point residual ─────────────────────
vp_sdp = Verifier(method="sdp", solver="clarabel", obbt_handler="highs")

z_sdp = [np.zeros(n_xz)]
v_sdp = [vp_sdp.add_iterate(n_xz, lb=np.zeros(n_xz), ub=np.ones(n_xz))]
w_sdp = [None]

sdp_residuals = []
for k in range(K):
    w_sdp.append(vp_sdp.step.satlin(v_sdp[k], l=l_x_val, u=u_x_val))
    rhs_k = sigma * z_sdp[k] - q + rho * M.T @ (2 * w_sdp[k + 1] - v_sdp[k])
    z_sdp.append(vp_sdp.step.implicit_linear(rhs=rhs_k, lhs_mat=lhs_mat))
    v_sdp.append(v_sdp[k] + (M @ z_sdp[k + 1] - w_sdp[k + 1]))
    vp_sdp.set_objective_norm(
        [z_sdp[k + 1] - z_sdp[k], v_sdp[k + 1] - v_sdp[k]], ord=2)
    sdp_residuals.append(vp_sdp.solve())
    print(f"SDP  k={k+1}: {sdp_residuals[-1]:.6f}")

# ── PEPit — squared 2-norm of combined fixed-point residual ───────────────────
P_eigs = np.real(spa.linalg.eigs(P, k=min(n_xz - 1, 10))[0])
mu_f = float(np.min(P_eigs[P_eigs > 1e-10]))
L_f  = float(np.max(P_eigs))

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

# ── Simulator — squared 2-norm, N random initial conditions from [0,1]^n_xz ──
N = 200
rng = np.random.default_rng(0)
v0_samples = rng.uniform(0, 1, size=(N, n_xz)).astype(np.float32)

vp_sim = Verifier(method="simulator")
v0_sim = vp_sim.add_param(n_xz, lb=np.zeros(n_xz), ub=np.ones(n_xz), samples=v0_samples)

z_sim = [np.zeros(n_xz)]
v_sim = [v0_sim]
w_sim = [None]

for k in range(K):
    w_sim.append(vp_sim.step.satlin(v_sim[k], l=l_x_val, u=u_x_val))
    rhs_k = sigma * z_sim[k] - q + rho * M.T @ (2 * w_sim[k + 1] - v_sim[k])
    z_sim.append(vp_sim.step.implicit_linear(rhs=rhs_k, lhs_mat=lhs_mat))
    v_sim.append(v_sim[k] + (M @ z_sim[k + 1] - w_sim[k + 1]))
    vp_sim.set_objective_norm(
        [z_sim[k + 1] - z_sim[k], v_sim[k + 1] - v_sim[k]], ord=2)

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
plt.title(f"Optimal Control ADMM: MIP vs SDP vs PEPit ({K} iterations)")
plt.legend()
plt.tight_layout()
plt.show()
print("Done.")
