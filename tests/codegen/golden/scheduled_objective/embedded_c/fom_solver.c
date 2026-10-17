/*
 * Golden embedded-C solver for the scheduled accelerated lasso (see
 * golden/scheduled/) stopping on the verification objective.
 *
 * Body, over the two-slot state (y_prev, y_curr):
 *     t0 = soft_threshold(t0_state @ state + gamma_k (t0_state__d_gamma @ state)
 *                         + t0_p_p0 @ p0, lambda)
 *     new_state = su_state @ state + su_t_t0 @ t0
 * Termination: on check iterations (and the last one) r = fom_objective(it,
 * state, new_state). The objective here is the prox-gradient residual at the
 * extrapolated point, y_{k+1} - (y_k + gamma_k (y_k - y_{k-1})), so it reads the
 * schedule too; fom_objective does that itself from `it`.
 *
 * Allocation- and division-free.
 */
#include "fom_data.h"
#include "fom_ops.h"
#include "fom_profile.h"
#include "fom_solver.h"

void fom_set_default_settings(FomSettings *settings) {
    settings->max_iter = FOM_MAX_ITER;
    settings->tol = FOM_TOL;
    settings->check_termination = FOM_CHECK_TERMINATION;
}

static void reset_state(FomWorkspace *work) {
    fom_int i;
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
    fom_int it, i;
    const fom_int max_iter = work->settings.max_iter;
    const fom_int check = work->settings.check_termination;
    const fom_float lam = FOM_T0_LAMBDA;
    fom_float mom[FOM_T0_DIM];

    work->solution.residual = 0.0;
    for (it = 0; it < max_iter; ++it) {
        const fom_float gamma = fom_sched_gamma(it);

        FOM_PROF_BEGIN(soft_threshold);
        for (i = 0; i < FOM_T0_DIM; ++i) {
            work->acc[i] = 0.0;
            mom[i] = 0.0;
        }
        fom_apply_t0_state(work->state, work->acc);
        fom_apply_t0_state__d_gamma(work->state, mom);
        for (i = 0; i < FOM_T0_DIM; ++i) work->acc[i] += gamma * mom[i];
        fom_apply_t0_p_p0(work->params.p0, work->acc);
        for (i = 0; i < FOM_T0_DIM; ++i) {
            fom_float a = work->acc[i];
            work->t0[i] = a > lam ? a - lam : (a < -lam ? a + lam : 0.0);
        }
        FOM_PROF_END(soft_threshold);

        FOM_PROF_BEGIN(state_update);
        for (i = 0; i < FOM_STATE_DIM; ++i) work->new_state[i] = 0.0;
        fom_apply_su_state(work->state, work->new_state);
        fom_apply_su_t_t0(work->t0, work->new_state);
        FOM_PROF_END(state_update);

        work->solution.iter = it + 1;
        if ((check > 0 && (it + 1) % check == 0) || it + 1 == max_iter) {
            fom_float r = fom_objective(it, work->state, work->new_state);
            work->solution.residual = r;
            for (i = 0; i < FOM_STATE_DIM; ++i) work->state[i] = work->new_state[i];
            if (r <= work->settings.tol) {
                for (i = 0; i < FOM_STATE_DIM; ++i) work->solution.x[i] = work->state[i];
                work->solution.status = FOM_SOLVED;
                return FOM_SOLVED;
            }
        } else {
            for (i = 0; i < FOM_STATE_DIM; ++i) work->state[i] = work->new_state[i];
        }
    }
    for (i = 0; i < FOM_STATE_DIM; ++i) work->solution.x[i] = work->state[i];
    work->solution.status = FOM_MAX_ITER_REACHED;
    return FOM_MAX_ITER_REACHED;
}

void fom_cleanup(FomWorkspace *work) {
    (void)work;
}
