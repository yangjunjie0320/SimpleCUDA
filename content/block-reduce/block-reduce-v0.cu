#include "utils.cu"

// use cub reduce
__global__ void kernel_v0(float* out, const float* inp, int nrow, int ncol) {
    const auto idx_block_in_grid = blockIdx.x;
    const auto idx_thread_in_block = threadIdx.x + threadIdx.y * blockDim.x;
    const auto num_thread_in_block = blockDim.x * blockDim.y;

    const auto i = idx_block_in_grid;
    const float* ai_ptr = inp + i * ncol;

    float local_sum = 0.0f;
    for (int j = idx_thread_in_block; j < ncol; j += num_thread_in_block) {
        local_sum += ai_ptr[j];
    }

    using BlockReduce =
        cub::BlockReduce<float, NUM_THREAD_IN_WARP, cub::BLOCK_REDUCE_WARP_REDUCTIONS,
                         NUM_WARP_IN_BLOCK_MAX, 1>;
    __shared__ BlockReduce::TempStorage buff;
    const float ai_sum = BlockReduce(buff).Reduce(local_sum, cub::Sum());

    if (idx_thread_in_block == 0) {
        out[i] = ai_sum;
    }
}