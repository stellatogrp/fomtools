"""Do PEP cross / subgradient-monotonicity SDP cuts tighten the relaxation?

This is a measurement harness (not a pytest test). It compares the SDP relaxation
bound against the *exact* MIP value on two verification problems:

  - NNLS  : projected-gradient with a ReLU (orthant projection) step,
  - ISTA   : proximal-gradient with a soft-threshold (ell-1 prox) step,

under four cut configurations layered on a fixed light base:

  base                = rlt_cuts + relu_envelope_cuts
  base + cross        = + cross_cut          (z_a r_b <= 0, PDF eq 3)
  base + subgrad      = + subgrad_mono_cut   (<s^i/tau_i - s^j/tau_j, z^i> >= 0, eq 4)
  base + both

Both backends use the squared two-norm objective (``set_objective_norm(expr, 2)``),
so they are directly comparable: the MIP value is exact and lower-bounds the SDP
relaxation, and ``gap = sdp - mip`` measures looseness. A cut "helps" when it
reduces that gap. We deliberately do NOT use ``full_psd_cone`` (slow, dense) — the
new cuts self-register the minimal small PSD cones they need to bind.

Notes
-----
* MIP two-norm requires Gurobi (HiGHS raises). If Gurobi is unavailable the script
  still prints SDP bounds but cannot compute the gap.
* ``obbt_solver=None`` everywhere so the only thing changing between configs is the
  cut set (cleaner signal, fewer external deps). The exact MIP value is independent
  of OBBT anyway.

Run:  python tests/examples/cut_experiment.py
"""

import time

import numpy as np

from fomtools.verifier import Verifier

SDP_SOLVER = "clarabel"
MIP_SOLVER = "gurobi"
GUROBI_SILENT = {"OutputFlag": 0, "LogToConsole": 0}
EPS = 1e-5

# Base config + the three cut layers. Each entry: (label, extra SDP kwargs).
BASE = {"rlt_cuts": True, "relu_envelope_cuts": True}
CONFIGS = [
    ("base", {}),
    ("base+cross", {"cross_cut": True}),
    ("base+subgrad", {"subgrad_mono_cut": True}),
    ("base+both", {"cross_cut": True, "subgrad_mono_cut": True}),
]


# --------------------------------------------------------------------------- #
# Problem builders. Each takes a Verifier and returns the objective expression #
# (the last-step increment z_K - z_{K-1}), so SDP and MIP build it identically.#
# --------------------------------------------------------------------------- #
def build_nnls(vp, K, A, t):
    """Projected-gradient NNLS: z_{k+1} = relu(z_k - t A^T (A z_k - x))."""
    m, n = A.shape
    x = vp.add_param(m, lb=0, ub=0.1)
    z = [vp.add_iterate(n, lb=5, ub=5.1)]
    for k in range(K):
        z.append(vp.step.relu(z[k] - t * A.T @ (A @ z[k] - x)))
    return z[K] - z[K - 1]


def build_ista(vp, K, At, Bt, lambda_t, b):
    """ISTA: z_{k+1} = soft_threshold(At z_k + Bt b, lambda_t)."""
    n = At.shape[0]
    m = Bt.shape[1]
    b_p = vp.add_param(m, lb=b - 0.1, ub=b + 0.1)
    z = [vp.add_iterate(n, lb=1, ub=1.1)]
    for k in range(K):
        z.append(vp.step.soft_threshold(At @ z[k] + Bt @ b_p, lambd=lambda_t))
    return z[K] - z[K - 1]


def solve_sdp(build, K, build_args, cut_kwargs):
    """Build + solve one SDP config; return (value, wall_seconds)."""
    vp = Verifier(
        method="sdp",
        solver=SDP_SOLVER,
        obbt_solver=None,
        **BASE,
        **cut_kwargs,
    )
    expr = build(vp, K, *build_args)
    vp.set_objective_norm(expr, 2)
    t0 = time.perf_counter()
    val = vp.solve()
    return float(val), time.perf_counter() - t0


def solve_mip_exact(build, K, build_args):
    """Build + solve the exact MIP (squared two-norm). Return value or None."""
    try:
        vp = Verifier(
            method="mip",
            solver=MIP_SOLVER,
            obbt_solver=None,
            params=dict(GUROBI_SILENT),
        )
        expr = build(vp, K, *build_args)
        vp.set_objective_norm(expr, 2)
        return float(vp.solve())
    except Exception as e:  # noqa: BLE001 - harness: report and continue
        print(f"  [MIP unavailable: {type(e).__name__}: {e}]")
        return None


def run_problem(name, build, build_args, K_list):
    """Run all configs for one problem family across K values and print a table."""
    print(f"\n{'=' * 78}\n{name}\n{'=' * 78}")
    header = f"{'K':>2}  {'config':<14}  {'sdp (||.||^2)':>14}  {'mip exact':>12}  {'gap':>10}  {'gap red%':>8}  {'time(s)':>8}"
    print(header)
    print("-" * len(header))

    overall_ok = True
    for K in K_list:
        mip_val = solve_mip_exact(build, K, build_args)
        base_gap = None
        base_sdp = None
        for label, cut_kwargs in CONFIGS:
            sdp_val, secs = solve_sdp(build, K, build_args, cut_kwargs)

            if label == "base":
                base_sdp = sdp_val
                base_gap = (sdp_val - mip_val) if mip_val is not None else None

            # Gap to exact MIP.
            if mip_val is not None:
                gap = sdp_val - mip_val
                gap_str = f"{gap:>10.3e}"
                mip_str = f"{mip_val:>12.6f}"
                # %gap reduction vs base.
                if base_gap is not None and base_gap > EPS:
                    red = 100.0 * (base_gap - gap) / base_gap
                    red_str = f"{red:>7.1f}%"
                else:
                    red_str = "  (tight)"
                # Validity: relaxation must not drop below the exact value.
                if gap < -EPS:
                    print(f"  !! INVALID: sdp {sdp_val:.6f} < mip {mip_val:.6f} (cut bug?)")
                    overall_ok = False
            else:
                gap_str = f"{'-':>10}"
                mip_str = f"{'-':>12}"
                red_str = f"{'-':>8}"

            # Monotonicity: adding valid cuts can only lower/keep the SDP value.
            if base_sdp is not None and sdp_val > base_sdp + 1e-6:
                print(
                    f"  !! NON-MONOTONE: {label} sdp {sdp_val:.6f} > base {base_sdp:.6f} (cut bug?)"
                )
                overall_ok = False

            print(
                f"{K:>2}  {label:<14}  {sdp_val:>14.6f}  {mip_str}  {gap_str}  {red_str}  {secs:>8.2f}"
            )
        print("-" * len(header))
    return overall_ok


def main():
    np.random.seed(0)
    ok = True

    # ---- NNLS -------------------------------------------------------------- #
    m, n = 3, 2
    A_nnls = np.random.normal(size=(m, n))
    t = 0.01
    ok &= run_problem("NNLS (ReLU projection)", build_nnls, (A_nnls, t), K_list=[1, 2])

    # ---- ISTA -------------------------------------------------------------- #
    np.random.seed(1)
    m, n = 3, 2
    A = np.random.normal(size=(m, n))
    b = np.random.normal(size=(m,))
    lambd_max = np.max(np.linalg.eigvalsh(A.T @ A))
    t_ista = 1.0 / lambd_max
    At = np.eye(n) - t_ista * A.T @ A
    Bt = t_ista * A.T
    lambda_t = 0.5 * t_ista
    ok &= run_problem("ISTA (soft-threshold)", build_ista, (At, Bt, lambda_t, b), K_list=[1, 2])

    print(f"\n{'=' * 78}")
    print("Verdict: read the 'gap red%' column.")
    print("  > 0%  : the cut tightened the SDP relaxation toward the exact value.")
    print("  ~ 0%  : no measurable effect for this problem/config.")
    print("  (tight): base relaxation already matches MIP; nothing to tighten.")
    if not ok:
        print("\n!! One or more configs were INVALID or NON-MONOTONE — investigate.")
    print("=" * 78)


if __name__ == "__main__":
    main()
