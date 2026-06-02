/*
 * Golden embedded-C solver for Douglas-Rachford-style first-order methods.
 *
 * Structure (see test/SPEC.md), two body steps + a state update:
 *   t0 = FOM_MAT_t0_state @ state + FOM_MAT_t0_p_p0 @ p0          (identity; the
 *        affine system's inverse is folded into the coefficients)
 *   t1 = relu( FOM_MAT_t1_state @ state + FOM_MAT_t1_t_t0 @ t0 )  (projection)
 *   new_state = FOM_MAT_su_state @ state
 *             + FOM_MAT_su_t_t0 @ t0 + FOM_MAT_su_t_t1 @ t1
 *
 * Data-agnostic and division-free: every coefficient comes from fom_data.c.
 * Hand-written reference used to test the deterministic codegen pipeline without
 * the LLM coding agent.
 */
#include "fom_solver.h"
#include "fom_data.h"

static void matvec_add(fom_float *out, int rows, const fom_float *M, int cols, const fom_float *v) {
    int i, j;
    for (i = 0; i < rows; ++i) {
        fom_float acc = 0.0;
        for (j = 0; j < cols; ++j) {
            acc += M[i * cols + j] * v[j];
        }
        out[i] += acc;
    }
}

void fom_set_default_settings(FomSettings *settings) {
    settings->max_iter = FOM_MAX_ITER;
    settings->tol = FOM_TOL;
}

static void reset_state(FomWorkspace *work) {
    int i;
    for (i = 0; i < FOM_STATE_DIM; ++i) {
        work->state[i] = FOM_STATE_INIT[i];
    }
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
        /* t0 = M_state @ state + M_p0 @ p0 */
        for (i = 0; i < FOM_T0_DIM; ++i) work->t0[i] = 0.0;
        matvec_add(work->t0, FOM_T0_DIM, FOM_MAT_t0_state, FOM_MAT_t0_state_COLS, work->state);
        matvec_add(work->t0, FOM_T0_DIM, FOM_MAT_t0_p_p0, FOM_MAT_t0_p_p0_COLS, work->params.p0);

        /* t1 = relu( M_state @ state + M_t0 @ t0 ) */
        for (i = 0; i < FOM_T1_DIM; ++i) work->t1[i] = 0.0;
        matvec_add(work->t1, FOM_T1_DIM, FOM_MAT_t1_state, FOM_MAT_t1_state_COLS, work->state);
        matvec_add(work->t1, FOM_T1_DIM, FOM_MAT_t1_t_t0, FOM_MAT_t1_t_t0_COLS, work->t0);
        for (i = 0; i < FOM_T1_DIM; ++i) {
            if (work->t1[i] < 0.0) work->t1[i] = 0.0;
        }

        /* new_state = su_state @ state + su_t_t0 @ t0 + su_t_t1 @ t1 */
        for (i = 0; i < FOM_STATE_DIM; ++i) work->new_state[i] = 0.0;
        matvec_add(work->new_state, FOM_STATE_DIM, FOM_MAT_su_state, FOM_MAT_su_state_COLS,
                   work->state);
        matvec_add(work->new_state, FOM_STATE_DIM, FOM_MAT_su_t_t0, FOM_MAT_su_t_t0_COLS, work->t0);
        matvec_add(work->new_state, FOM_STATE_DIM, FOM_MAT_su_t_t1, FOM_MAT_su_t_t1_COLS, work->t1);

        /* residual = max_abs(new_state - state) */
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
