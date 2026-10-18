/*
 * Golden large-scale CUDA solver for the scheduled accelerated lasso stopping
 * on the verification objective (see golden/scheduled_objective/embedded_c/).
 *
 * Device-resident loop over the two-slot state:
 *     t0 = soft_threshold(t0_state @ state + gamma_k (t0_state__d_gamma @ state)
 *                         + t0_p_p0 @ p0, lambda)
 *     new_state = su_state @ state + su_t_t0 @ t0
 * The momentum is read on the host through fom_sched_gamma(it) and passed to
 * the kernels; fom_objective(it, state, new_state) reads the same schedule
 * itself.
 */
#include <cuda_runtime.h>

#include "fom_data.h"
#include "fom_ops.h"
#include "fom_solver.h"

#define THREADS 256

static fom_float *d_state = 0, *d_new_state = 0, *d_acc = 0, *d_mom = 0, *d_t0 = 0;
static fom_float *d_p0 = 0;

/* t0 = soft_threshold(acc + gamma * mom) */
__global__ void step_kern(const fom_float *acc, const fom_float *mom, fom_float gamma,
                          fom_float *t0, fom_float lam, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    fom_float v = acc[i] + gamma * mom[i];
    t0[i] = v > lam ? v - lam : (v < -lam ? v + lam : 0.0);
}

static void reset_state(FomWorkspace *work) {
    int i;
    for (i = 0; i < FOM_STATE_DIM; ++i) work->state[i] = FOM_STATE_INIT[i];
}

extern "C" void fom_set_default_settings(FomSettings *settings) {
    settings->max_iter = FOM_MAX_ITER;
    settings->tol = FOM_TOL;
    settings->check_termination = FOM_CHECK_TERMINATION;
}

extern "C" void fom_setup(FomWorkspace *work, const FomSettings *settings) {
    work->settings = *settings;
    if (!d_state) {
        cudaMalloc((void **)&d_state, sizeof(fom_float) * FOM_STATE_DIM);
        cudaMalloc((void **)&d_new_state, sizeof(fom_float) * FOM_STATE_DIM);
        cudaMalloc((void **)&d_acc, sizeof(fom_float) * FOM_T0_DIM);
        cudaMalloc((void **)&d_mom, sizeof(fom_float) * FOM_T0_DIM);
        cudaMalloc((void **)&d_t0, sizeof(fom_float) * FOM_T0_DIM);
        cudaMalloc((void **)&d_p0, sizeof(fom_float) * FOM_PARAM_P0_DIM);
    }
    reset_state(work);
    work->solution.iter = 0;
    work->solution.residual = 0.0;
    work->solution.status = FOM_MAX_ITER_REACHED;
}

extern "C" void fom_update_params(FomWorkspace *work, const FomParams *params) {
    work->params = *params;
    reset_state(work);
}

extern "C" fom_int fom_solve(FomWorkspace *work) {
    const int n = FOM_STATE_DIM;
    const int blocks_t0 = (FOM_T0_DIM + THREADS - 1) / THREADS;
    const fom_int max_iter = work->settings.max_iter;
    const fom_int check = work->settings.check_termination;
    int it, i, solved = 0;
    fom_float *tmp;

    cudaMemcpy(d_state, work->state, sizeof(fom_float) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_p0, work->params.p0, sizeof(fom_float) * FOM_PARAM_P0_DIM,
               cudaMemcpyHostToDevice);

    work->solution.residual = 0.0;
    for (it = 0; it < max_iter; ++it) {
        const fom_float gamma = fom_sched_gamma(it);

        cudaMemset(d_acc, 0, sizeof(fom_float) * FOM_T0_DIM);
        cudaMemset(d_mom, 0, sizeof(fom_float) * FOM_T0_DIM);
        fom_apply_t0_state(d_state, d_acc);
        fom_apply_t0_state__d_gamma(d_state, d_mom);
        fom_apply_t0_p_p0(d_p0, d_acc);
        step_kern<<<blocks_t0, THREADS>>>(d_acc, d_mom, gamma, d_t0, FOM_T0_LAMBDA, FOM_T0_DIM);

        cudaMemset(d_new_state, 0, sizeof(fom_float) * n);
        fom_apply_su_state(d_state, d_new_state);
        fom_apply_su_t_t0(d_t0, d_new_state);

        work->solution.iter = it + 1;
        fom_float r = 0.0;
        int tested = (check > 0 && (it + 1) % check == 0) || it + 1 == max_iter;
        if (tested) {
            r = fom_objective(it, d_state, d_new_state);
            work->solution.residual = r;
        }
        tmp = d_state;
        d_state = d_new_state;
        d_new_state = tmp;
        if (tested && r <= work->settings.tol) {
            solved = 1;
            break;
        }
    }

    cudaMemcpy(work->state, d_state, sizeof(fom_float) * n, cudaMemcpyDeviceToHost);
    for (i = 0; i < n; ++i) work->solution.x[i] = work->state[i];
    work->solution.status = solved ? FOM_SOLVED : FOM_MAX_ITER_REACHED;
    return work->solution.status;
}

extern "C" void fom_cleanup(FomWorkspace *work) {
    (void)work;
    cudaFree(d_state); cudaFree(d_new_state); cudaFree(d_acc);
    cudaFree(d_mom); cudaFree(d_t0); cudaFree(d_p0);
    d_state = d_new_state = d_acc = d_mom = d_t0 = d_p0 = 0;
}
