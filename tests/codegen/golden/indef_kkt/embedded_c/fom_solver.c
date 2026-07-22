/*
 * Golden embedded-C solver for an affine fixed-point on a symmetric-INDEFINITE,
 * non-quasidefinite KKT system  K = [[0, A^T], [A, 0]].
 *
 * Because K has an all-zero diagonal, no-pivot LDL^T is not viable; code
 * generation falls back to a pivoted LU factorization, so fom_lin_solve_t0 is an
 * LU solve here (still never an inverse). The solver body is agnostic to which
 * direct method was chosen — it just calls fom_lin_solve_t0.
 *
 * Structure (see test/SPEC.md):
 *   rhs0      = t0_state @ state + t0_p_p0 @ p0      (= x + q)
 *   t0        = solve(K, rhs0)  via fom_lin_solve_t0 (pivoted LU; no inverse)
 *   new_state = su_t_t0 @ t0                          (= t0)
 */
#include "fom_solver.h"
#include "fom_data.h"
#include "fom_linsolve.h"
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
        /* affine: assemble rhs = x + q, then solve K t0 = rhs (pivoted LU) */
        FOM_PROF_BEGIN(lu_solve);
        for (i = 0; i < FOM_T0_DIM; ++i) work->acc[i] = 0.0;
        fom_apply_t0_state(work->state, work->acc);
        fom_apply_t0_p_p0(work->params.p0, work->acc);
        fom_lin_solve_t0(work->acc, work->t0);
        FOM_PROF_END(lu_solve);

        /* state update: new_state = t0 */
        FOM_PROF_BEGIN(state_update);
        for (i = 0; i < FOM_STATE_DIM; ++i) work->new_state[i] = 0.0;
        fom_apply_su_t_t0(work->t0, work->new_state);
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
