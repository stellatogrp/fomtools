/*
 * Golden embedded-C solver for a method with a *declared* varying coefficient:
 * one repeated iteration whose coefficient is read from a table.
 *
 * The fixture happens to be an accelerated lasso, but nothing below depends on
 * which method it is -- only on the shape a schedule lowers to.
 *
 * Structure (see test/SPEC.md): one soft-threshold body step whose state
 * coefficient is split into a fixed base and a scheduled slope,
 *     t0 = soft_threshold(  FOM_MAT_t0_state         @ state
 *                         + gamma_k * (FOM_MAT_t0_state__d_gamma @ state)
 *                         + FOM_MAT_t0_p_p0          @ p0, lambda)
 * and a state update that shifts the two-slot state (y_prev, y_curr):
 *     new_state = FOM_MAT_su_state @ state + FOM_MAT_su_t_t0 @ t0
 *
 * The coefficient is tabulated at generation time and read through
 * fom_sched_gamma(it), which clamps its index. Computing it here instead would
 * need a square root and a division, and the embedded target allows neither.
 * Contrast golden/fista/, where the same method is recorded as an ordinary loop
 * and every iteration carries its own coefficients instead.
 *
 * Data-agnostic: every coefficient, the threshold, and the schedule come from
 * fom_data.c / fom_data.h. Allocation-free and division-free.
 *
 * This is a hand-written reference used to exercise the deterministic codegen
 * pipeline (scaffold -> build -> verify -> Python wrapper) without invoking the
 * LLM coding agent.
 */
#include "fom_solver.h"
#include "fom_data.h"
#include "fom_ops.h"
#include "fom_profile.h"

void fom_set_default_settings(FomSettings *settings) {
    settings->max_iter = FOM_MAX_ITER;
    settings->tol = FOM_TOL;
    settings->check_termination = FOM_CHECK_TERMINATION;
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
        /* The momentum coefficient is fixed within one iteration. */
        fom_float gamma = fom_sched_gamma(it);

        /* t0 = soft_threshold((base + gamma * slope) @ state + Bt @ p0) */
        FOM_PROF_BEGIN(soft_threshold);
        for (i = 0; i < FOM_T0_DIM; ++i) {
            fom_float acc = 0.0;
            fom_float mom = 0.0;
            for (j = 0; j < FOM_MAT_t0_state_COLS; ++j) {
                acc += FOM_MAT_t0_state[i * FOM_MAT_t0_state_COLS + j] * work->state[j];
            }
            for (j = 0; j < FOM_MAT_t0_state__d_gamma_COLS; ++j) {
                mom += FOM_MAT_t0_state__d_gamma[i * FOM_MAT_t0_state__d_gamma_COLS + j]
                       * work->state[j];
            }
            acc += gamma * mom;
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
        FOM_PROF_END(soft_threshold);

        /* new_state = su_state @ state + su_t_t0 @ t0 */
        FOM_PROF_BEGIN(state_update);
        for (i = 0; i < FOM_STATE_DIM; ++i) {
            work->new_state[i] = 0.0;
        }
        fom_apply_su_state(work->state, work->new_state);
        for (i = 0; i < FOM_STATE_DIM; ++i) {
            fom_float acc = 0.0;
            for (j = 0; j < FOM_MAT_su_t_t0_COLS; ++j) {
                acc += FOM_MAT_su_t_t0[i * FOM_MAT_su_t_t0_COLS + j] * work->t0[j];
            }
            work->new_state[i] += acc;
        }
        FOM_PROF_END(state_update);

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
