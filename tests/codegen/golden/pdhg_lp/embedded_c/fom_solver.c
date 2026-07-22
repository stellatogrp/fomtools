/*
 * Golden embedded-C solver for plain PDHG (Chambolle-Pock) on a linear program
 * (no restarts).
 *
 * LP: min c'x  s.t.  A x >= b,  l <= x <= u, solved through the saddle problem
 *   min_{l<=x<=u} max_{y>=0}  c'x + y'(b - A x).
 *
 * State is (x, y). One iteration (see test/SPEC.md), two body steps + a state
 * update, all pure first-order (matrix-vector products + projections, no linear
 * solve):
 *   x+   = clip( t0_state @ state + t0_p_p0 @ c , l, u )      (box projection)
 *        = clip( x - tau (c - A' y), l, u )
 *   y+   = relu( t1_state @ state + t1_t_t0 @ x+ + t1_c )     (y >= 0 projection)
 *        = relu( y + sigma (b - A (2 x+ - x)) )
 *   new_state = su_t_t0 @ x+ + su_t_t1 @ y+
 *
 * Every matrix-vector product goes through the sparse-aware fom_apply_*
 * routines; the clip bounds and the dual constant live in fom_data.c.
 */
#include "fom_solver.h"
#include "fom_data.h"
#include "fom_ops.h"
#include "fom_profile.h"

void fom_set_default_settings(FomSettings *settings) {
    settings->max_iter = FOM_MAX_ITER;
    settings->tol = FOM_TOL;
}

static void reset_state(FomWorkspace *work) {
    int i;
    for (i = 0; i < FOM_STATE_DIM; ++i) work->state[i] = FOM_STATE_INIT[i];
}

void fom_setup(FomWorkspace *work, const FomSettings *settings) {
    work->settings = *settings;
    reset_state(work);
    work->solution.iter = 0;
    work->solution.residual = 0.0;
    work->solution.status = FOM_MAX_ITER_REACHED;
}

void fom_update_params(FomWorkspace *work, const FomParams *params) {
    work->params = *params;
    reset_state(work);
}

fom_int fom_solve(FomWorkspace *work) {
    int it, i;

    for (it = 0; it < work->settings.max_iter; ++it) {
        /* primal step: x+ = clip(rhs, l, u), box projection (saturated-linear) */
        FOM_PROF_BEGIN(primal);
        for (i = 0; i < FOM_T0_DIM; ++i) work->acc[i] = 0.0;
        fom_apply_t0_state(work->state, work->acc);
        fom_apply_t0_p_p0(work->params.p0, work->acc);
        for (i = 0; i < FOM_T0_DIM; ++i) {
            fom_float v = work->acc[i];
            if (v < FOM_MAT_t0_lo[i]) v = FOM_MAT_t0_lo[i];
            if (v > FOM_MAT_t0_hi[i]) v = FOM_MAT_t0_hi[i];
            work->t0[i] = v;
        }
        FOM_PROF_END(primal);

        /* dual step: y+ = relu(rhs), projection onto the nonnegative orthant.
           The extrapolation (2 x+ - x) is folded into t1_state / t1_t_t0. */
        FOM_PROF_BEGIN(dual);
        for (i = 0; i < FOM_T1_DIM; ++i) work->t1[i] = FOM_MAT_t1_c[i];
        fom_apply_t1_state(work->state, work->t1);
        fom_apply_t1_t_t0(work->t0, work->t1);
        for (i = 0; i < FOM_T1_DIM; ++i) {
            if (work->t1[i] < 0.0) work->t1[i] = 0.0;
        }
        FOM_PROF_END(dual);

        /* state update: new_state = (x+, y+) */
        FOM_PROF_BEGIN(state_update);
        for (i = 0; i < FOM_STATE_DIM; ++i) work->new_state[i] = 0.0;
        fom_apply_su_t_t0(work->t0, work->new_state);
        fom_apply_su_t_t1(work->t1, work->new_state);
        FOM_PROF_END(state_update);

        fom_float res = 0.0;
        for (i = 0; i < FOM_STATE_DIM; ++i) {
            fom_float d = work->new_state[i] - work->state[i];
            if (d < 0.0) d = -d;
            if (d > res) res = d;
            work->state[i] = work->new_state[i];
        }
        work->solution.iter = it + 1;
        work->solution.residual = res;
        if (FOM_USE_TOL && res <= work->settings.tol) {
            for (i = 0; i < FOM_STATE_DIM; ++i) work->solution.x[i] = work->state[i];
            work->solution.status = FOM_SOLVED;
            return FOM_SOLVED;
        }
    }
    for (i = 0; i < FOM_STATE_DIM; ++i) work->solution.x[i] = work->state[i];
    work->solution.status = FOM_MAX_ITER_REACHED;
    return work->solution.status;
}

void fom_cleanup(FomWorkspace *work) {
    (void)work;
}
