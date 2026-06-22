/*
 * Golden embedded-C solver for the cvxpygen integration test QP
 * (OSQP-style ADMM built by fomtools.cvxpygen.builders.build_admm_qp).
 *
 * Structure (see test/SPEC.md):
 *   rhs0 = t0_state @ state + t0_p_p0 @ p0
 *   t0   = solve(KKT, rhs0)  via fom_lin_solve_t0 (sparse LDL^T; no inverse)
 *   t1   = relu(t1_c + t1_state @ state + t1_t_t0 @ t0)   (ineq projection)
 *   new_state = su_c + su_state @ state + su_t_t0 @ t0 + su_t_t1 @ t1
 *
 * All matrix-vector products go through the sparse-aware fom_apply_* routines
 * and the KKT solve through fom_linsolve.c. No matrix inverse, no division.
 */
#include "fom_solver.h"
#include "fom_data.h"
#include "fom_linsolve.h"
#include "fom_ops.h"
#include "fom_profile.h"

void fom_set_default_settings(FomSettings *settings) {
    settings->max_iter = FOM_MAX_ITER;
    settings->tol = FOM_TOL;
    settings->check_termination = FOM_CHECK_TERMINATION;
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
    /* Runtime-tunable check interval (defaults to FOM_CHECK_TERMINATION). */
    fom_int check = work->settings.check_termination;

    for (it = 0; it < work->settings.max_iter; ++it) {
        /* affine KKT solve */
        FOM_PROF_BEGIN(kkt_solve);
        for (i = 0; i < FOM_T0_DIM; ++i) work->acc[i] = 0.0;
        fom_apply_t0_state(work->state, work->acc);
        fom_apply_t0_p_p0(work->params.p0, work->acc);
        fom_lin_solve_t0(work->acc, work->t0);
        FOM_PROF_END(kkt_solve);

        /* inequality projection: relu(t1_c + t1_state @ state + t1_t_t0 @ t0) */
        FOM_PROF_BEGIN(project);
        for (i = 0; i < FOM_T1_DIM; ++i) work->t1[i] = FOM_MAT_t1_c[i];
        fom_apply_t1_state(work->state, work->t1);
        fom_apply_t1_t_t0(work->t0, work->t1);
        for (i = 0; i < FOM_T1_DIM; ++i) {
            if (work->t1[i] < 0.0) work->t1[i] = 0.0;
        }
        FOM_PROF_END(project);

        /* state update */
        FOM_PROF_BEGIN(state_update);
        for (i = 0; i < FOM_STATE_DIM; ++i) work->new_state[i] = FOM_MAT_su_c[i];
        fom_apply_su_state(work->state, work->new_state);
        fom_apply_su_t_t0(work->t0, work->new_state);
        fom_apply_su_t_t1(work->t1, work->new_state);
        FOM_PROF_END(state_update);

        /* residual + termination check only every `check` iterations (the
           residual is skipped entirely in between); `check` is read from the
           runtime settings, so callers can tune it without regenerating */
        if (FOM_USE_TOL && check > 0 && ((it + 1) % check == 0)) {
            fom_float res = 0.0;
            for (i = 0; i < FOM_STATE_DIM; ++i) {
                fom_float d = work->new_state[i] - work->state[i];
                if (d < 0.0) d = -d;
                if (d > res) res = d;
                work->state[i] = work->new_state[i];
            }
            work->solution.iter = it + 1;
            work->solution.residual = res;
            if (res <= work->settings.tol) {
                for (i = 0; i < FOM_STATE_DIM; ++i) work->solution.x[i] = work->state[i];
                work->solution.status = FOM_SOLVED;
                return FOM_SOLVED;
            }
        } else {
            for (i = 0; i < FOM_STATE_DIM; ++i) work->state[i] = work->new_state[i];
            work->solution.iter = it + 1;
        }
    }
    for (i = 0; i < FOM_STATE_DIM; ++i) work->solution.x[i] = work->state[i];
    work->solution.status = FOM_MAX_ITER_REACHED;
    return work->solution.status;
}

void fom_cleanup(FomWorkspace *work) {
    (void)work;
}
