#include "utils.cu"

// each warp handles one row NUM_THREAD_IN_WARP threads; each block contains NUM_WARP_IN_BLOCK warps
__global__ void kernel_v5(float* out, const float* inp, int nrow, int ncol) {
    const auto num_thread_in_warp = blockDim.x;
    const auto num_warp_in_block = blockDim.y;
    const auto num_block_in_grid = gridDim.x;

    const auto idx_lane = threadIdx.x;
    const auto idx_block_in_grid = blockIdx.x;
    const auto idx_warp_in_block = threadIdx.y;
    constexpr auto offset0 = NUM_THREAD_IN_WARP / 2;

    // sanity check
    assert(num_thread_in_warp == NUM_THREAD_IN_WARP);
    assert(num_warp_in_block == NUM_WARP_IN_BLOCK);
    assert(num_block_in_grid * NUM_WARP_IN_BLOCK == nrow);
    assert(offset0 * 2 == num_thread_in_warp);

    const auto i = idx_block_in_grid * num_warp_in_block + idx_warp_in_block;
    const float* ai_ptr = inp + i * ncol;

    const auto col_base = idx_lane;
    const auto col_step = num_thread_in_warp;

    float ai_max_in_thread = -INFINITY;
    for (auto j = col_base; j < ncol; j += col_step) {
        const float aij = ai_ptr[j];
        ai_max_in_thread = fmaxf(ai_max_in_thread, aij);
    }

    float ai_max_in_warp = ai_max_in_thread;

#pragma unroll
    for (int offset = offset0; offset > 0; offset >>= 1) {
        auto ai_max_curr_lane = ai_max_in_warp;
        auto ai_max_next_lane = __shfl_down_sync(FULL, ai_max_in_warp, offset);
        ai_max_in_warp = fmaxf(ai_max_curr_lane, ai_max_next_lane);
    }
    const float ai_max = __shfl_sync(FULL, ai_max_in_warp, 0);

    float ai_sum_in_thread = 0.0;
    for (auto j = col_base; j < ncol; j += col_step) {
        const float aij = ai_ptr[j];
        ai_sum_in_thread += expf(aij - ai_max);
    }
    float ai_sum_in_warp = ai_sum_in_thread;

#pragma unroll
    for (int offset = offset0; offset > 0; offset >>= 1) {
        auto ai_sum_curr_lane = ai_sum_in_warp;
        auto ai_sum_next_lane = __shfl_down_sync(FULL, ai_sum_in_warp, offset);
        ai_sum_in_warp += ai_sum_next_lane;
    }
    const float ai_sum = __shfl_sync(FULL, ai_sum_in_warp, 0);
    const float ai_sum_inv = 1.0 / ai_sum;

    float* ci_ptr = out + i * ncol;
    for (auto j = col_base; j < ncol; j += col_step) {
        const float aij = ai_ptr[j];
        const float cij = expf(aij - ai_max) * ai_sum_inv;
        ci_ptr[j] = cij;
    }
}