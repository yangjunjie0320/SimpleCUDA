#include "utils.cu"

__global__ void kernel_v3(float* out, const float* inp, int nrow, int ncol) {
    const auto num_thread_in_warp = blockDim.x;
    const auto num_warp_in_block = blockDim.y;
    const auto num_thread_in_block = num_thread_in_warp * num_warp_in_block;
    const auto num_block_in_grid = gridDim.x;

    const auto idx_lane = threadIdx.x;
    const auto idx_warp_in_block = threadIdx.y;
    const auto idx_thread_in_block = threadIdx.x + threadIdx.y * blockDim.x;
    const auto idx_block_in_grid = blockIdx.x;

    // sanity check
    assert(num_thread_in_warp == NUM_THREAD_IN_WARP);
    assert(num_thread_in_block == NUM_THREAD_IN_WARP * num_warp_in_block);
    assert(num_block_in_grid == nrow);

    const auto i = idx_block_in_grid;
    const float* ai_ptr = inp + i * ncol;

    const auto col_base = idx_thread_in_block;
    const auto col_step = num_thread_in_block;

    float ai_sum_in_thread = 0.0;
    for (auto j = col_base; j < ncol; j += col_step) {
        const float aij = ai_ptr[j];
        ai_sum_in_thread += aij;
    }

    extern __shared__ float buff[];
    buff[col_base] = ai_sum_in_thread;
    __syncthreads();

    const auto j = col_base;
    // Block-level reduction: reduce to one value per warp
    const auto offset0 = num_thread_in_block / 2;
    for (int offset = offset0; offset >= NUM_THREAD_IN_WARP; offset >>= 1) {
        if (j < offset) {
            const float ai_curr_thread = buff[j];
            const float ai_next_thread = buff[j + offset];
            buff[j] = ai_curr_thread + ai_next_thread;
        }
        __syncthreads();
    }

    // Warp-level reduction using shuffle instructions
    // Only the first thread in each warp participates
    float ai_sum = (j < NUM_THREAD_IN_WARP) ? buff[j] : 0.0f;
    if (idx_warp_in_block == 0) {
#pragma unroll
        for (int offset = NUM_THREAD_IN_WARP / 2; offset > 0; offset >>= 1) {
            const float ai_next_lane = __shfl_down_sync(FULL, ai_sum, offset);
            ai_sum += ai_next_lane;
        }
    }

    if (j == 0) {
        out[i] = ai_sum;
    }
}