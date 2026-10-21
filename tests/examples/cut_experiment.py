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

The instances are deliberately *hard*: initial-iterate / parameter boxes straddle
zero so the ReLU / soft-threshold steps sit in their nonconvex "mixed" regime
(where the relaxation is loosest), and we sweep dimension n, measurements m, and
unrolled depth K upward to grow the SDP-vs-MIP gap.

Notes
-----
* MIP two-norm requires Gurobi (HiGHS raises). If Gurobi is unavailable the script
  still prints SDP bounds but cannot compute the gap. A time limit guards the MIP;
  if it is hit, ``mip`` is an incumbent (lower bound on the true max), so the gap is
  an over-estimate but validity (sdp >= true >= incumbent) still holds.
* ``obbt_solver=None`` everywhere so the only thing changing between configs is the
  cut set (cleaner signal, fewer external deps). The exact MIP value is independent
  of OBBT anyway.
* The ``#cone`` / ``#cut`` columns report the PSD-cone count and added linear-cut
  count on the solved copy — i.e. the cost the cut family imposes.

Run:  python tests/examples/cut_experiment.py
"""

import time

import numpy as np

from fomtools.verifier import Verifier

SDP_SOLVER = "clarabel"
MIP_SOLVER = "gurobi"
MIP_TIME_LIMIT = 60.0
GUROBI_PARAMS = {
    "OutputFlag": 0,
    "LogToConsole": 0,
    "TimeLimit": MIP_TIME_LIMIT,
    "MIPGap": 1e-6,
}
EPS = 1e-5

# Base config + the three cut layers. Each entry: (label, extra SDP kwargs).
# The last row is a labeled REFERENCE only (dense full moment matrix, slow, NOT a
# recommended config) used to bound how much of the base gap is theoretically
# removable, so the "% gap reduction" of the small-cone cuts is interpretable.
BASE = {"rlt_cuts": True, "relu_envelope_cuts": True}
CONFIGS = [
    ("base", {}),
    ("base+cross", {"cross_cut": True}),
    ("base+subgrad", {"subgrad_mono_cut": True}),
    ("base+both", {"cross_cut": True, "subgrad_mono_cut": True}),
    ("full_psd[ref]", {"full_psd_cone": True}),
]


# --------------------------------------------------------------------------- #
# Problem builders. Each takes a Verifier and returns the objective expression #
# (the last-step increment z_K - z_{K-1}), so SDP and MIP build it identically.#
# --------------------------------------------------------------------------- #
def build_nnls(vp, K, A, t, z0_lo, z0_hi, x_lo, x_hi):
    """Projected-gradient NNLS: z_{k+1} = relu(z_k - t A^T (A z_k - x))."""
    m, n = A.shape
    x = vp.parameter(m, lb=x_lo, ub=x_hi)
    z = [vp.iterate(n, lb=z0_lo, ub=z0_hi)]
    for k in range(K):
        z.append(vp.relu(z[k] - t * A.T @ (A @ z[k] - x)))
    return z[K] - z[K - 1]


def build_ista(vp, K, At, Bt, lambda_t, b, b_hw, z0_lo, z0_hi):
    """ISTA: z_{k+1} = soft_threshold(At z_k + Bt b, lambda_t)."""
    n = At.shape[0]
    m = Bt.shape[1]
    b_p = vp.parameter(m, lb=b - b_hw, ub=b + b_hw)
    z = [vp.iterate(n, lb=z0_lo, ub=z0_hi)]
    for k in range(K):
        z.append(vp.soft_threshold(At @ z[k] + Bt @ b_p, lambd=lambda_t))
    return z[K] - z[K - 1]


def build_boxpgd(vp, K, P, q, lo, hi, t, q_hw, z0_lo, z0_hi):
    """Box-PGD: z_{k+1} = satlin(z_k - t (P z_k + q), l, u) (box projection)."""
    n = P.shape[0]
    q_p = vp.parameter(n, lb=q - q_hw, ub=q + q_hw)
    z = [vp.iterate(n, lb=z0_lo, ub=z0_hi)]
    for k in range(K):
        z.append(vp.satlin(z[k] - t * (P @ z[k] + q_p), l=lo, u=hi))
    return z[K] - z[K - 1]


# --------------------------------------------------------------------------- #
# Instance generators. Each returns (build_fn, build_args, label) for a given  #
# size, with gap-inducing boxes (z0 straddles zero -> mixed-regime steps).     #
# --------------------------------------------------------------------------- #
def make_nnls_instance(n, m, K, seed, box=2.0):
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(m, n))
    L = float(np.max(np.linalg.eigvalsh(A.T @ A)))
    t = 1.0 / L  # full gradient step
    # z0 straddles zero so the gradient-step argument crosses the ReLU kink;
    # x kept positive (NNLS data term) but wide.
    args = (A, t, -box, box, 0.0, box)
    return build_nnls, args, f"NNLS n={n} m={m} K={K}"


def make_ista_instance(n, m, K, seed, box=1.5, lam_scale=2.0):
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(m, n))
    b = rng.normal(size=(m,))
    L = float(np.max(np.linalg.eigvalsh(A.T @ A)))
    t = 1.0 / L
    At = np.eye(n) - t * A.T @ A
    Bt = t * A.T
    # Larger lambda -> more active soft-threshold kinks -> looser relaxation.
    lambda_t = lam_scale * t
    args = (At, Bt, lambda_t, b, 0.5, -box, box)
    return build_ista, args, f"ISTA n={n} m={m} K={K}"


def make_boxpgd_instance(n, K, seed, box=1.5):
    rng = np.random.default_rng(seed)
    P = np.eye(n) + 0.1 * rng.normal(size=(n, n))
    P = (P + P.T) / 2
    q = rng.normal(size=n)
    # Asymmetric box so lower/upper faces differ; iterate box straddles both faces.
    lo = -0.5 - 0.5 * np.abs(rng.normal(size=n))
    hi = 0.5 + 0.5 * np.abs(rng.normal(size=n))
    t = 0.5
    args = (P, q, lo, hi, t, 0.5, -box, box)
    return build_boxpgd, args, f"BoxPGD n={n} K={K}"


def _cut_cone_counts(vp):
    """Return (#psd_cones, #added_linear_cuts) from the solved copy, if present."""
    copy = getattr(vp.canonicalizer, "_solved_copy", None)
    if copy is None:
        return None, None
    return len(copy.psd_cone_indices), len(copy._cut_ineq_matrices)


def solve_sdp(build, K, build_args, cut_kwargs):
    """Build + solve one SDP config; return (value, wall_seconds, n_cones, n_cuts)."""
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
    secs = time.perf_counter() - t0
    n_cones, n_cuts = _cut_cone_counts(vp)
    return float(val), secs, n_cones, n_cuts


def solve_mip_exact(build, K, build_args):
    """Build + solve the exact MIP (squared two-norm). Return (value|None, secs)."""
    try:
        vp = Verifier(
            method="mip",
            solver=MIP_SOLVER,
            obbt_solver=None,
            params=dict(GUROBI_PARAMS),
        )
        expr = build(vp, K, *build_args)
        vp.set_objective_norm(expr, 2)
        t0 = time.perf_counter()
        val = float(vp.solve())
        return val, time.perf_counter() - t0
    except Exception as e:  # noqa: BLE001 - harness: report and continue
        print(f"  [MIP unavailable: {type(e).__name__}: {e}]")
        return None, 0.0


def run_instance(build, build_args, K, label):
    """Run all configs for one concrete instance and print a block."""
    mip_val, mip_secs = solve_mip_exact(build, K, build_args)
    mip_note = f"   [mip {mip_secs:.1f}s]" if mip_val is not None else ""
    print(f"\n{label}{mip_note}")
    header = (
        f"  {'config':<14}  {'sdp (||.||^2)':>14}  {'mip exact':>12}  "
        f"{'gap':>10}  {'gap red%':>8}  {'#cone':>6}  {'#cut':>6}  {'t(s)':>7}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))

    ok = True
    base_gap = base_sdp = None
    for cfg_label, cut_kwargs in CONFIGS:
        sdp_val, secs, n_cones, n_cuts = solve_sdp(build, K, build_args, cut_kwargs)

        if cfg_label == "base":
            base_sdp = sdp_val
            base_gap = (sdp_val - mip_val) if mip_val is not None else None

        if mip_val is not None:
            gap = sdp_val - mip_val
            gap_str = f"{gap:>10.3e}"
            mip_str = f"{mip_val:>12.6f}"
            if base_gap is not None and base_gap > EPS:
                red = 100.0 * (base_gap - gap) / base_gap
                red_str = f"{red:>7.1f}%"
            else:
                red_str = " (tight)"
            if gap < -EPS:
                print(f"  !! INVALID: sdp {sdp_val:.6f} < mip {mip_val:.6f} (cut bug?)")
                ok = False
        else:
            gap_str = f"{'-':>10}"
            mip_str = f"{'-':>12}"
            red_str = f"{'-':>8}"

        if base_sdp is not None and sdp_val > base_sdp + 1e-6:
            print(f"  !! NON-MONOTONE: {cfg_label} sdp {sdp_val:.6f} > base {base_sdp:.6f}")
            ok = False

        print(
            f"  {cfg_label:<14}  {sdp_val:>14.6f}  {mip_str}  {gap_str}  "
            f"{red_str}  {n_cones:>6}  {n_cuts:>6}  {secs:>7.2f}"
        )
    return ok


def main():
    ok = True

    print("\n" + "=" * 92)
    print("NNLS (ReLU projection) — z0 straddles 0 so the gradient step crosses the kink")
    print("=" * 92)
    nnls_specs = [
        (2, 3, 2, 0),
        (3, 4, 2, 1),
        (3, 4, 3, 2),
        (4, 6, 2, 5),
        (4, 6, 3, 6),
    ]
    for n, m, K, seed in nnls_specs:
        build, args, label = make_nnls_instance(n, m, K, seed)
        ok &= run_instance(build, args, K, label)

    print("\n" + "=" * 92)
    print("ISTA (soft-threshold) — wide b/z0 boxes and larger lambda -> more active kinks")
    print("=" * 92)
    ista_specs = [
        (2, 3, 2, 10),
        (3, 4, 2, 11),
        (3, 4, 3, 12),
        (4, 6, 2, 15),
        (4, 6, 3, 16),
    ]
    for n, m, K, seed in ista_specs:
        build, args, label = make_ista_instance(n, m, K, seed)
        ok &= run_instance(build, args, K, label)

    print("\n" + "=" * 92)
    print("BoxPGD (saturated-linear) — z0 straddles the box faces -> mixed-regime satlin")
    print("=" * 92)
    boxpgd_specs = [
        (2, 2, 22),
        (2, 3, 13),
        (3, 2, 30),
        (3, 3, 31),
        (4, 2, 40),
    ]
    for n, K, seed in boxpgd_specs:
        build, args, label = make_boxpgd_instance(n, K, seed)
        ok &= run_instance(build, args, K, label)

    print("\n" + "=" * 92)
    print("Verdict: read 'gap red%'. > 0% = cut tightened the relaxation toward exact.")
    print("  Compare 'base+cross' vs 'base+subgrad' to see where the two families diverge.")
    print("  '#cone'/'#cut' show the cost each family adds (small cones, not a dense block).")
    if not ok:
        print("\n!! One or more configs were INVALID or NON-MONOTONE — investigate.")
    print("=" * 92)


if __name__ == "__main__":
    main()
