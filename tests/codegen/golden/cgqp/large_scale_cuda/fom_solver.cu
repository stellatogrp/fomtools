/*
 * Golden large-scale CUDA solver for an SPD linear-solve FOM (CG backend).
 *
 * Structure (see test/SPEC.md), one body step + a state update:
 *   rhs0      = t0_state @ state + t0_p_p0 @ p0      (= x + q)
 *   t0        = solve(M, rhs0)  via fom_lin_solve_t0 (device-resident
 *               preconditioned CG; the SPD system is never inverted)
 *   new_state = su_t_t0 @ t0                          (= t0)
 *
 * All vectors stay on the device for the whole loop (pointer swap, no
 * copies); every matvec goes through the deterministic fom_apply_* kernels
 * and the affine solve through the device CG in fom_linsolve.cu. The only
 * per-iteration device->host traffic is the residual scalar and the CG
 * stop flag.
 */
#include <cuda_runtime.h>

#include "fom_solver.h"
#include "fom_data.h"
#include "fom_linsolve.h"
#include "fom_ops.h"

#define THREADS 256
#define RED_BLOCKS 64

static fom_float *d_state = 0, *d_new_state = 0, *d_acc = 0, *d_t0 = 0;
static fom_float *d_p0 = 0, *d_red = 0;

/* per-block partial max of |a - b| */
__global__ void maxdiff_kern(const fom_float *a, const fom_float *b, int n,
                             fom_float *partials) {
    __shared__ fom_float sh[THREADS];
    fom_float m = 0.0;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += gridDim.x * blockDim.x) {
        fom_float d = a[i] - b[i];
        if (d < 0.0) d = -d;
        if (d > m) m = d;
    }
    sh[threadIdx.x] = m;
    __syncthreads();
    for (int w = blockDim.x >> 1; w > 0; w >>= 1) {
        if (threadIdx.x < w && sh[threadIdx.x + w] > sh[threadIdx.x])
            sh[threadIdx.x] = sh[threadIdx.x + w];
        __syncthreads();
    }
    if (threadIdx.x == 0) partials[blockIdx.x] = sh[0];
}

/* one block folds the per-block partials into partials[0] */
__global__ void maxfold_kern(fom_float *partials) {
    __shared__ fom_float sh[RED_BLOCKS];
    sh[threadIdx.x] = partials[threadIdx.x];
    __syncthreads();
    for (int w = RED_BLOCKS >> 1; w > 0; w >>= 1) {
        if (threadIdx.x < w && sh[threadIdx.x + w] > sh[threadIdx.x])
            sh[threadIdx.x] = sh[threadIdx.x + w];
        __syncthreads();
    }
    if (threadIdx.x == 0) partials[0] = sh[0];
}

static fom_float device_max_diff(const fom_float *a, const fom_float *b, int n) {
    fom_float m;
    maxdiff_kern<<<RED_BLOCKS, THREADS>>>(a, b, n, d_red);
    maxfold_kern<<<1, RED_BLOCKS>>>(d_red);
    cudaMemcpy(&m, d_red, sizeof(fom_float), cudaMemcpyDeviceToHost);
    return m;
}

static void reset_state(FomWorkspace *work) {
    int i;
    for (i = 0; i < FOM_STATE_DIM; ++i) work->state[i] = FOM_STATE_INIT[i];
}

extern "C" void fom_set_default_settings(FomSettings *settings) {
    settings->max_iter = FOM_MAX_ITER;
    settings->tol = FOM_TOL;
}

extern "C" void fom_setup(FomWorkspace *work, const FomSettings *settings) {
    work->settings = *settings;
    cudaMalloc((void **)&d_state, sizeof(fom_float) * FOM_STATE_DIM);
    cudaMalloc((void **)&d_new_state, sizeof(fom_float) * FOM_STATE_DIM);
    cudaMalloc((void **)&d_acc, sizeof(fom_float) * FOM_MAX_VEC_DIM);
    cudaMalloc((void **)&d_t0, sizeof(fom_float) * FOM_T0_DIM);
    cudaMalloc((void **)&d_p0, sizeof(fom_float) * FOM_PARAM_P0_DIM);
    cudaMalloc((void **)&d_red, sizeof(fom_float) * RED_BLOCKS);
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
    int it, i, solved = 0;
    fom_float *tmp;

    /* upload the start state and parameters once; the loop is device-resident */
    cudaMemcpy(d_state, work->state, sizeof(fom_float) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_p0, work->params.p0, sizeof(fom_float) * FOM_PARAM_P0_DIM,
               cudaMemcpyHostToDevice);

    for (it = 0; it < work->settings.max_iter; ++it) {
        /* affine: rhs = state + q, then solve M t0 = rhs via device CG */
        cudaMemset(d_acc, 0, sizeof(fom_float) * FOM_T0_DIM);
        fom_apply_t0_state(d_state, d_acc);
        fom_apply_t0_p_p0(d_p0, d_acc);
        fom_lin_solve_t0(d_acc, d_t0);

        /* state update: new_state = t0 */
        cudaMemset(d_new_state, 0, sizeof(fom_float) * n);
        fom_apply_su_t_t0(d_t0, d_new_state);

        fom_float res = device_max_diff(d_new_state, d_state, n);

        /* state <- new_state by pointer swap (no copy) */
        tmp = d_state;
        d_state = d_new_state;
        d_new_state = tmp;

        work->solution.iter = it + 1;
        work->solution.residual = res;
        if (FOM_USE_TOL && res <= work->settings.tol) {
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
    cudaFree(d_t0); cudaFree(d_p0); cudaFree(d_red);
    d_state = d_new_state = d_acc = d_t0 = d_p0 = d_red = 0;
}
