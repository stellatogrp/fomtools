/*
 * Golden large-scale CUDA solver for plain PDHG (Chambolle-Pock) on an LP.
 *
 * Pure first-order: matrix-vector products + projections, no linear solve -
 * the all-matvec GPU showcase. State is (x, y); one iteration (see test/SPEC.md),
 * two body steps + a state update:
 *   x+   = clip( t0_state @ state + t0_p_p0 @ c , l, u )      (box projection)
 *   y+   = relu( t1_c + t1_state @ state + t1_t_t0 @ x+ )     (y >= 0 projection)
 *   new_state = su_t_t0 @ x+ + su_t_t1 @ y+
 *
 * Every matvec goes through the deterministic device fom_apply_* kernels
 * (fom_ops.cu). The clip bounds and the dual constant are host arrays in
 * fom_data.c, so we upload them to the device once in fom_setup. All vectors
 * stay on the device for the whole loop (pointer swap, no per-iteration full-
 * vector copies); only the residual scalar crosses back to the host.
 */
#include <cuda_runtime.h>

#include "fom_solver.h"
#include "fom_data.h"
#include "fom_ops.h"

#define THREADS 256
#define RED_BLOCKS 64
#define NBLOCKS(n) (((n) + THREADS - 1) / THREADS)

static fom_float *d_state = 0, *d_new_state = 0, *d_t0 = 0, *d_t1 = 0;
static fom_float *d_p0 = 0, *d_lo = 0, *d_hi = 0, *d_t1c = 0, *d_red = 0;

/* box projection: v = clip(v, lo, hi) elementwise */
__global__ void clip_kern(fom_float *v, const fom_float *lo, const fom_float *hi, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    fom_float x = v[i];
    if (x < lo[i]) x = lo[i];
    if (x > hi[i]) x = hi[i];
    v[i] = x;
}

/* projection onto the nonnegative orthant: v = max(v, 0) */
__global__ void relu_kern(fom_float *v, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    if (v[i] < 0.0) v[i] = 0.0;
}

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
    cudaMalloc((void **)&d_t0, sizeof(fom_float) * FOM_T0_DIM);
    cudaMalloc((void **)&d_t1, sizeof(fom_float) * FOM_T1_DIM);
    cudaMalloc((void **)&d_p0, sizeof(fom_float) * FOM_PARAM_P0_DIM);
    cudaMalloc((void **)&d_lo, sizeof(fom_float) * FOM_T0_DIM);
    cudaMalloc((void **)&d_hi, sizeof(fom_float) * FOM_T0_DIM);
    cudaMalloc((void **)&d_t1c, sizeof(fom_float) * FOM_T1_DIM);
    cudaMalloc((void **)&d_red, sizeof(fom_float) * RED_BLOCKS);
    /* upload the constant clip bounds and dual constant once (host -> device) */
    cudaMemcpy(d_lo, FOM_MAT_t0_lo, sizeof(fom_float) * FOM_T0_DIM, cudaMemcpyHostToDevice);
    cudaMemcpy(d_hi, FOM_MAT_t0_hi, sizeof(fom_float) * FOM_T0_DIM, cudaMemcpyHostToDevice);
    cudaMemcpy(d_t1c, FOM_MAT_t1_c, sizeof(fom_float) * FOM_T1_DIM, cudaMemcpyHostToDevice);
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

    cudaMemcpy(d_state, work->state, sizeof(fom_float) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_p0, work->params.p0, sizeof(fom_float) * FOM_PARAM_P0_DIM,
               cudaMemcpyHostToDevice);

    for (it = 0; it < work->settings.max_iter; ++it) {
        /* primal: x+ = clip(t0_state @ state + t0_p_p0 @ c, l, u) */
        cudaMemset(d_t0, 0, sizeof(fom_float) * FOM_T0_DIM);
        fom_apply_t0_state(d_state, d_t0);
        fom_apply_t0_p_p0(d_p0, d_t0);
        clip_kern<<<NBLOCKS(FOM_T0_DIM), THREADS>>>(d_t0, d_lo, d_hi, FOM_T0_DIM);

        /* dual: y+ = relu(t1_c + t1_state @ state + t1_t_t0 @ x+) */
        cudaMemcpy(d_t1, d_t1c, sizeof(fom_float) * FOM_T1_DIM, cudaMemcpyDeviceToDevice);
        fom_apply_t1_state(d_state, d_t1);
        fom_apply_t1_t_t0(d_t0, d_t1);
        relu_kern<<<NBLOCKS(FOM_T1_DIM), THREADS>>>(d_t1, FOM_T1_DIM);

        /* state update: new_state = su_t_t0 @ x+ + su_t_t1 @ y+ */
        cudaMemset(d_new_state, 0, sizeof(fom_float) * n);
        fom_apply_su_t_t0(d_t0, d_new_state);
        fom_apply_su_t_t1(d_t1, d_new_state);

        fom_float res = device_max_diff(d_new_state, d_state, n);

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
    cudaFree(d_state); cudaFree(d_new_state); cudaFree(d_t0); cudaFree(d_t1);
    cudaFree(d_p0); cudaFree(d_lo); cudaFree(d_hi); cudaFree(d_t1c); cudaFree(d_red);
    d_state = d_new_state = d_t0 = d_t1 = 0;
    d_p0 = d_lo = d_hi = d_t1c = d_red = 0;
}
