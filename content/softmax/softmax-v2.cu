#include <cuda_runtime.h>
#include <cmath>

// each block handles one row, use ncol threads
__global__ void kernel_v2(float* out, const float* inp, int nrow, int ncol) {
    const auto num_thread_in_block = blockDim.x;
    const auto num_block_in_grid = gridDim.x;

    const auto idx_thread_in_block = threadIdx.x;
    const auto idx_block_in_grid = blockIdx.x;
    const auto idx_thread_in_grid = threadIdx.x + blockIdx.x * blockDim.x;

    const auto i = idx_thread_in_grid;
    if (i >= nrow) {return;}

    const float* ai_ptr = inp + i * ncol;
    float* ci_ptr = out + i * ncol;

    extern __shared__ float smem[];

    float ai_t_max = -INFINITY;
    for (auto j = t; j < ncol; j += stride) {
        float aij = ai_ptr[j];
        ai_t_max = fmaxf(ai_t_max, aij);
    }
    smem[t] = ai_t_max;
    __syncthreads();

    // Tree reduction for max
    for (auto offset = stride / 2; offset > 0; offset /= 2) {
        if (t < offset) {
            smem[t] = fmaxf(smem[t], smem[t + offset]);
        }
        __syncthreads();
    }
    auto ai_max = smem[0];
    __syncthreads();

    float ai_t_sum = 0.0f;
    for (auto j = t; j < ncol; j += stride) {
        float aij = ai_ptr[j];
        ai_t_sum += expf(aij - ai_max);
    }
    smem[t] = ai_t_sum;
    __syncthreads();

    // Tree reduction for sum
    for (auto offset = stride / 2; offset > 0; offset /= 2) {
        if (t < offset) {
            smem[t] += smem[t + offset];
        }
        __syncthreads();
    }
    auto ai_sum = smem[0];
    __syncthreads();

    for (auto j = t; j < ncol; j += stride) {
        float aij = ai_ptr[j];
        float cij = expf(aij - ai_max) / ai_sum;
        ci_ptr[j] = cij;
    }
}
