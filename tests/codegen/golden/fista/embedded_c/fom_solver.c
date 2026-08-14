/*
 * Golden embedded-C solver for FISTA recorded as an ordinary loop.
 *
 * The momentum coefficient genuinely changes every iteration, so the recording
 * does not roll back up: this SPEC is UNROLLED, and its body is the whole run.
 * There is no `for` over iterations and no schedule table. Each block carries
 * its own coefficients, and momentum shows up structurally as a block reading
 * the two *previous* temporaries:
 *
 *     t0 = soft_threshold(t0_state @ state + t0_p_p0 @ p0)
 *     t1 = soft_threshold(t1_state @ state + t0_p_p0 @ p0 + t1_t_t0 @ t0)
 *     tk = soft_threshold(t0_p_p0 @ p0 + tk_t_t{k-2} @ t{k-2} + tk_t_t{k-1} @ t{k-1})
 *     new_state = su_t_t5 @ t5
 *
 * Note `t0_p_p0` appears in every block: the parameter coefficient is the same
 * every iteration, so deduplication emitted it once and the blocks share it.
 *
 * `fom_solve` must execute this exactly once and ignore `settings.max_iter`;
 * the verify harness raises that budget and requires the answer not to move.
 */
#include "fom_data.h"
#include "fom_ops.h"
#include "fom_solver.h"

static void soft_threshold(const fom_float *in, fom_float lam, fom_float *out, fom_int n) {
    fom_int i;
    for (i = 0; i < n; ++i) {
        fom_float v = in[i];
        if (v > lam) {
            out[i] = v - lam;
        } else if (v < -lam) {
            out[i] = v + lam;
        } else {
            out[i] = (fom_float)0.0;
        }
    }
}

static void zero(fom_float *v, fom_int n) {
    fom_int i;
    for (i = 0; i < n; ++i) {
        v[i] = (fom_float)0.0;
    }
}

void fom_set_default_settings(FomSettings *settings) {
    settings->max_iter = FOM_MAX_ITER;
    settings->tol = FOM_TOL;
    settings->check_termination = FOM_CHECK_TERMINATION;
}

void fom_setup(FomWorkspace *work, const FomSettings *settings) {
    fom_int i;
    work->settings = *settings;
    for (i = 0; i < FOM_STATE_DIM; ++i) {
        work->state[i] = FOM_STATE_INIT[i];
    }
}

void fom_update_params(FomWorkspace *work, const FomParams *params) {
    fom_int i;
    work->params = *params;
    for (i = 0; i < FOM_STATE_DIM; ++i) {
        work->state[i] = FOM_STATE_INIT[i];
    }
}

fom_int fom_solve(FomWorkspace *work) {
    fom_int i;
    fom_float res = (fom_float)0.0;

    /* The recorded run, once, in order. */
    zero(work->acc, FOM_T0_DIM);
    fom_apply_t0_state(work->state, work->acc);
    fom_apply_t0_p_p0(work->params.p0, work->acc);
    soft_threshold(work->acc, FOM_T0_LAMBDA, work->t0, FOM_T0_DIM);

    zero(work->acc, FOM_T1_DIM);
    fom_apply_t1_state(work->state, work->acc);
    fom_apply_t0_p_p0(work->params.p0, work->acc);
    fom_apply_t1_t_t0(work->t0, work->acc);
    soft_threshold(work->acc, FOM_T1_LAMBDA, work->t1, FOM_T1_DIM);

    zero(work->acc, FOM_T2_DIM);
    fom_apply_t0_p_p0(work->params.p0, work->acc);
    fom_apply_t2_t_t0(work->t0, work->acc);
    fom_apply_t2_t_t1(work->t1, work->acc);
    soft_threshold(work->acc, FOM_T2_LAMBDA, work->t2, FOM_T2_DIM);

    zero(work->acc, FOM_T3_DIM);
    fom_apply_t0_p_p0(work->params.p0, work->acc);
    fom_apply_t3_t_t1(work->t1, work->acc);
    fom_apply_t3_t_t2(work->t2, work->acc);
    soft_threshold(work->acc, FOM_T3_LAMBDA, work->t3, FOM_T3_DIM);

    zero(work->acc, FOM_T4_DIM);
    fom_apply_t0_p_p0(work->params.p0, work->acc);
    fom_apply_t4_t_t2(work->t2, work->acc);
    fom_apply_t4_t_t3(work->t3, work->acc);
    soft_threshold(work->acc, FOM_T4_LAMBDA, work->t4, FOM_T4_DIM);

    zero(work->acc, FOM_T5_DIM);
    fom_apply_t0_p_p0(work->params.p0, work->acc);
    fom_apply_t5_t_t3(work->t3, work->acc);
    fom_apply_t5_t_t4(work->t4, work->acc);
    soft_threshold(work->acc, FOM_T5_LAMBDA, work->t5, FOM_T5_DIM);

    zero(work->new_state, FOM_STATE_DIM);
    fom_apply_su_t_t5(work->t5, work->new_state);

    for (i = 0; i < FOM_STATE_DIM; ++i) {
        fom_float d = work->new_state[i] - work->state[i];
        if (d < (fom_float)0.0) {
            d = -d;
        }
        if (d > res) {
            res = d;
        }
        work->state[i] = work->new_state[i];
    }

    for (i = 0; i < FOM_STATE_DIM; ++i) {
        work->solution.x[i] = work->state[i];
    }
    work->solution.iter = 1;
    work->solution.residual = res;
    work->solution.status = FOM_MAX_ITER_REACHED;
    return work->solution.status;
}

void fom_cleanup(FomWorkspace *work) {
    (void)work;
}
