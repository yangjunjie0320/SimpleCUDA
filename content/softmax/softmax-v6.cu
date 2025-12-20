#include "utils.cu"

// Each block handles one row with NUM_WARP_IN_BLOCK warps.
// Each thread handles one element (ncol must equal block size).
__global__ void kernel_v6(float* out, const float* inp, int nrow, int ncol) {
    const auto num_thread_in_warp = blockDim.x;
    const auto num_warp_in_block = blockDim.y;
    const auto num_thread_in_block = num_thread_in_warp * num_warp_in_block;
    const auto num_block_in_grid = gridDim.x;

    const auto idx_lane = threadIdx.x;
    const auto idx_warp_in_block = threadIdx.y;
    const auto idx_thread_in_block = idx_lane + idx_warp_in_block * num_thread_in_warp;
    const auto idx_block_in_grid = blockIdx.x;
    constexpr auto offset0 = NUM_THREAD_IN_WARP / 2;

    __shared__ float ai_max_in_warp[NUM_WARP_IN_BLOCK];
    __shared__ float ai_sum_in_warp[NUM_WARP_IN_BLOCK];

    // sanity check
    assert(num_thread_in_warp == NUM_THREAD_IN_WARP);
    assert(num_warp_in_block == NUM_WARP_IN_BLOCK);
    assert(num_thread_in_block == ncol);
    assert(num_block_in_grid == nrow);
    assert(offset0 * 2 == num_thread_in_warp);

    const auto i = idx_block_in_grid;
    const auto j = idx_lane + idx_warp_in_block * num_thread_in_warp;

    const float aij = *(inp + i * ncol + j);
    float ai_max = aij;

#pragma unroll
    for (int offset = offset0; offset > 0; offset >>= 1) {
        auto ai_max_curr_lane = ai_max;
        auto ai_max_next_lane = __shfl_down_sync(FULL, ai_max, offset);
        ai_max = fmaxf(ai_max_curr_lane, ai_max_next_lane);
    }

    if (idx_lane == 0) {
        ai_max_in_warp[idx_warp_in_block] = ai_max;
    }
    __syncthreads();

    if (idx_thread_in_block == 0) {
        ai_max = -INFINITY;
        for (int ii = 0; ii < num_warp_in_block; ii++) {
            ai_max = fmaxf(ai_max, ai_max_in_warp[ii]);
        }
        ai_max_in_warp[0] = ai_max;
    }
    __syncthreads();
    ai_max = ai_max_in_warp[0];

    const float exp_aij = expf(aij - ai_max);
    float ai_sum = exp_aij;
#pragma unroll
    for (int offset = offset0; offset > 0; offset >>= 1) {
        auto ai_sum_curr_lane = ai_sum;
        auto ai_sum_next_lane = __shfl_down_sync(FULL, ai_sum, offset);
        ai_sum = ai_sum_curr_lane + ai_sum_next_lane;
    }

    if (idx_lane == 0) {
        ai_sum_in_warp[idx_warp_in_block] = ai_sum;
    }
    __syncthreads();

    if (idx_thread_in_block == 0) {
        ai_sum = 0.0;
        for (int ii = 0; ii < num_warp_in_block; ii++) {
            ai_sum += ai_sum_in_warp[ii];
        }
        ai_sum_in_warp[0] = ai_sum;
    }
    __syncthreads();
    ai_sum = ai_sum_in_warp[0];

    const float ai_sum_inv = 1.0 / ai_sum;
    const float cij = exp_aij * ai_sum_inv;
    *(out + i * ncol + j) = cij;
}