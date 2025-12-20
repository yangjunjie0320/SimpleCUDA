#include "utils.cu"

// each block handles one row, 1 thread, naive serial implementation
__global__ void kernel_v1(float* out, const float* inp, int nrow, int ncol) {
    const auto idx_thread_in_block = threadIdx.x;
    const auto num_thread_in_block = blockDim.x;
    const auto idx_block_in_grid = blockIdx.x;
    const auto num_block_in_grid = gridDim.x;
    const auto idx_thread_in_grid = idx_block_in_grid * num_thread_in_block + idx_thread_in_block;

    // sanity check
    assert(num_thread_in_block == 1);
    assert(num_block_in_grid == nrow);

    const auto i = idx_thread_in_grid;
    const auto ai_ptr = inp + i * ncol;
    const auto ci_ptr = out + i * ncol;

    float ai_max = -INFINITY;
    float ai_sum = 0.0;

    for (auto j = 0; j < ncol; j++) {
        const float aij = ai_ptr[j];
        ai_max = fmaxf(ai_max, aij);
    }

    for (auto j = 0; j < ncol; j++) {
        const float aij = ai_ptr[j];
        const float exp_aij = expf(aij - ai_max);
        ai_sum += exp_aij;
        ci_ptr[j] = exp_aij;
    }

    for (auto j = 0; j < ncol; j++) {
        const float ai_sum_inv = 1.0 / ai_sum;
        ci_ptr[j] *= ai_sum_inv;
    }
}
