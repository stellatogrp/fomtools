/*
 * Golden large-scale CUDA solver for ISTA-style first-order methods.
 *
 * One kernel evaluates the soft-thresholded update row by row; the host loop
 * drives iterations. Coefficients are copied to the device once in fom_setup.
 * Compile-checked in CI (no GPU); run on a GPU machine to verify numerically.
 */
#include <cuda_runtime.h>

#include "fom_solver.h"
#include "fom_data.h"

/* Device-side copies of the (small) coefficient blocks and working vectors. */
static fom_float *d_At = 0;     /* FOM_MAT_t0_state */
static fom_float *d_Bt = 0;     /* FOM_MAT_t0_p_p0  */
static fom_float *d_su = 0;     /* FOM_MAT_su_t_t0  */
static fom_float *d_state = 0;
static fom_float *d_newstate = 0;
static fom_float *d_t0 = 0;
static fom_float *d_p0 = 0;

__global__ void soft_threshold_step(const fom_float *At, const fom_float *Bt,
                                    const fom_float *state, const fom_float *p0,
                                    fom_float *t0, fom_float lam,
                                    int n, int m) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    fom_float acc = 0;
    for (int j = 0; j < n; ++j) acc += At[i * n + j] * state[j];
    for (int j = 0; j < m; ++j) acc += Bt[i * m + j] * p0[j];
    if (acc > lam) t0[i] = acc - lam;
    else if (acc < -lam) t0[i] = acc + lam;
    else t0[i] = 0;
}

__global__ void apply_su(const fom_float *su, const fom_float *t0,
                         fom_float *new_state, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    fom_float acc = 0;
    for (int j = 0; j < n; ++j) acc += su[i * n + j] * t0[j];
    new_state[i] = acc;
}

extern "C" void fom_set_default_settings(FomSettings *settings) {
    settings->max_iter = FOM_MAX_ITER;
    settings->tol = FOM_TOL;
}

extern "C" void fom_setup(FomWorkspace *work, const FomSettings *settings) {
    work->settings = *settings;
    int n = FOM_STATE_DIM, m = FOM_PARAM_P0_DIM;
    cudaMalloc(&d_At, sizeof(fom_float) * n * n);
    cudaMalloc(&d_Bt, sizeof(fom_float) * n * m);
    cudaMalloc(&d_su, sizeof(fom_float) * n * n);
    cudaMalloc(&d_state, sizeof(fom_float) * n);
    cudaMalloc(&d_newstate, sizeof(fom_float) * n);
    cudaMalloc(&d_t0, sizeof(fom_float) * n);
    cudaMalloc(&d_p0, sizeof(fom_float) * m);
    cudaMemcpy(d_At, FOM_MAT_t0_state, sizeof(fom_float) * n * n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_Bt, FOM_MAT_t0_p_p0, sizeof(fom_float) * n * m, cudaMemcpyHostToDevice);
    cudaMemcpy(d_su, FOM_MAT_su_t_t0, sizeof(fom_float) * n * n, cudaMemcpyHostToDevice);
    for (int i = 0; i < n; ++i) work->state[i] = FOM_STATE_INIT[i];
}

extern "C" void fom_update_params(FomWorkspace *work, const FomParams *params) {
    work->params = *params;
    int n = FOM_STATE_DIM;
    for (int i = 0; i < n; ++i) work->state[i] = FOM_STATE_INIT[i];
}

extern "C" fom_int fom_solve(FomWorkspace *work) {
    int n = FOM_STATE_DIM, m = FOM_PARAM_P0_DIM;
    fom_float lam = FOM_T0_LAMBDA;
    int threads = 64, blocks = (n + threads - 1) / threads;
    cudaMemcpy(d_state, work->state, sizeof(fom_float) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_p0, work->params.p0, sizeof(fom_float) * m, cudaMemcpyHostToDevice);

    for (int it = 0; it < work->settings.max_iter; ++it) {
        soft_threshold_step<<<blocks, threads>>>(d_At, d_Bt, d_state, d_p0, d_t0, lam, n, m);
        apply_su<<<blocks, threads>>>(d_su, d_t0, d_newstate, n);
        cudaMemcpy(work->new_state, d_newstate, sizeof(fom_float) * n, cudaMemcpyDeviceToHost);

        fom_float res = 0;
        for (int i = 0; i < n; ++i) {
            fom_float d = work->new_state[i] - work->state[i];
            if (d < 0) d = -d;
            if (d > res) res = d;
            work->state[i] = work->new_state[i];
        }
        cudaMemcpy(d_state, work->state, sizeof(fom_float) * n, cudaMemcpyHostToDevice);
        work->solution.iter = it + 1;
        work->solution.residual = res;
        if (FOM_USE_TOL && res <= work->settings.tol) {
            for (int i = 0; i < n; ++i) work->solution.x[i] = work->state[i];
            work->solution.status = FOM_SOLVED;
            return FOM_SOLVED;
        }
    }
    for (int i = 0; i < n; ++i) work->solution.x[i] = work->state[i];
    work->solution.status = FOM_MAX_ITER_REACHED;
    return work->solution.status;
}

extern "C" void fom_cleanup(FomWorkspace *work) {
    (void)work;
    cudaFree(d_At); cudaFree(d_Bt); cudaFree(d_su);
    cudaFree(d_state); cudaFree(d_newstate); cudaFree(d_t0); cudaFree(d_p0);
}
