#include "utils.cu"

__global__ void kernel_v2(float* out, const float* inp, int nrow, int ncol) {
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
    assert(num_block_in_grid == nrow && num_thread_in_block * 2 == ncol);

    const auto i = idx_block_in_grid;
    const auto j = idx_thread_in_block;
    const float* ai_ptr = inp + i * ncol;
    const float aij = ai_ptr[j];

    extern __shared__ float buff[];
    buff[j] = aij + ai_ptr[j + num_thread_in_block];
    __syncthreads();

    const auto offset0 = 1;
    for (int offset = offset0; offset < num_thread_in_block; offset *= 2) {
        if (j % (offset * 2) == 0) {
            float ai_curr_thread = buff[j];
            float ai_next_thread = buff[j + offset];
            buff[j] = ai_curr_thread + ai_next_thread;
        }
        __syncthreads();
    }
    __syncthreads();

    if (j == 0) {
        out[i] = buff[0];
    }
}