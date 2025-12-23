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
    assert(num_block_in_grid == nrow && num_thread_in_block == ncol);

    const auto i = idx_block_in_grid;
    const auto j = idx_thread_in_block;
    const float* ai_ptr = inp + i * ncol;
    const float aij = ai_ptr[j];

    extern __shared__ float buff[];
    buff[j] = aij;
    __syncthreads();

    const auto offset0 = num_thread_in_block / 2;
    for (int offset = offset0; offset > NUM_THREAD_IN_WARP; offset >>= 1) {
        if (j < offset) {
            float ai_curr_thread = buff[j];
            float ai_next_thread = buff[j + offset];
            buff[j] = ai_curr_thread + ai_next_thread;
        }
        __syncthreads();
    }
    __syncthreads();

    float ai_sum = buff[j];
    if (j < NUM_THREAD_IN_WARP) {
        ai_sum += buff[j + NUM_THREAD_IN_WARP];
#pragma unroll
        for (int offset = NUM_THREAD_IN_WARP / 2; offset > 0; offset >>= 1) {
            if (j < offset) {
                float ai_curr_lane = __shfl_down_sync(FULL, ai_sum, offset);
                float ai_next_lane = __shfl_down_sync(FULL, ai_sum, offset);
                ai_sum = ai_curr_lane + ai_next_lane;
            }
        }
    }

    if (j == 0) {
        out[i] = ai_sum;
    }
}