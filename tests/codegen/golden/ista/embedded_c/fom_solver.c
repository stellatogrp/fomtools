/*
 * Golden embedded-C solver for ISTA-style first-order methods.
 *
 * Structure (see test/SPEC.md): one soft-threshold body step
 *     t0 = soft_threshold(FOM_MAT_t0_state @ state + FOM_MAT_t0_p_p0 @ p0, lambda)
 * and a state update that is the identity on t0
 *     new_state = FOM_MAT_su_t_t0 @ t0
 *
 * Data-agnostic: all coefficients and the threshold come from fom_data.c /
 * fom_data.h, so this same source works for any ISTA instance. Allocation-free
 * and division-free, as required by the embedded target.
 *
 * This is a hand-written reference used to exercise the deterministic codegen
 * pipeline (scaffold -> build -> verify -> Python wrapper) without invoking the
 * LLM coding agent.
 */
#include "fom_solver.h"
#include "fom_data.h"

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
    /* Each solve starts from the initial state (matches the reference oracle). */
    reset_state(work);
}

fom_int fom_solve(FomWorkspace *work) {
    int it, i, j;
    fom_float lam = FOM_T0_LAMBDA;

    for (it = 0; it < work->settings.max_iter; ++it) {
        /* t0 = soft_threshold(At @ state + Bt @ p0) */
        for (i = 0; i < FOM_T0_DIM; ++i) {
            fom_float acc = 0.0;
            for (j = 0; j < FOM_MAT_t0_state_COLS; ++j) {
                acc += FOM_MAT_t0_state[i * FOM_MAT_t0_state_COLS + j] * work->state[j];
            }
            for (j = 0; j < FOM_MAT_t0_p_p0_COLS; ++j) {
                acc += FOM_MAT_t0_p_p0[i * FOM_MAT_t0_p_p0_COLS + j] * work->params.p0[j];
            }
            if (acc > lam) {
                work->t0[i] = acc - lam;
            } else if (acc < -lam) {
                work->t0[i] = acc + lam;
            } else {
                work->t0[i] = 0.0;
            }
        }

        /* new_state = su_t_t0 @ t0 */
        for (i = 0; i < FOM_STATE_DIM; ++i) {
            fom_float acc = 0.0;
            for (j = 0; j < FOM_MAT_su_t_t0_COLS; ++j) {
                acc += FOM_MAT_su_t_t0[i * FOM_MAT_su_t_t0_COLS + j] * work->t0[j];
            }
            work->new_state[i] = acc;
        }

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
