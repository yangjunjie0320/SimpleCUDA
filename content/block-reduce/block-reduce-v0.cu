#include "utils.cu"

// use cub reduce
#define NUM_THREAD_IN_BLOCK_MAX 2048
__global__ void kernel_v0(float* out, const float* inp, int nrow, int ncol) {
    const auto num_thread_in_warp = blockDim.x;
    const auto num_warp_in_block = blockDim.y;
    const auto num_thread_in_block = num_thread_in_warp * num_warp_in_block;
    const auto num_block_in_grid = gridDim.x;

    const auto idx_thread_in_block = threadIdx.x + threadIdx.y * blockDim.x;
    const auto idx_block_in_grid = blockIdx.x;

    // sanity check
    assert(num_thread_in_warp == NUM_THREAD_IN_WARP);
    assert(num_thread_in_block == NUM_THREAD_IN_WARP * num_warp_in_block);
    assert(num_thread_in_block == ncol && num_block_in_grid == nrow);

    const auto i = idx_block_in_grid;
    const auto j = idx_thread_in_block;
    const float* ai_ptr = inp + i * ncol;
    const float aij = ai_ptr[j];

    using BlockReduce =
        cub::BlockReduce<float, NUM_THREAD_IN_WARP, cub::BLOCK_REDUCE_WARP_REDUCTIONS,
                         NUM_THREAD_IN_BLOCK_MAX, 1>;

    extern __shared__ BlockReduce::TempStorage buff[];
    auto& tmp = buff[0];
    const float ai_sum = BlockReduce(tmp).Reduce(aij, cub::Sum());
    if (j == 0) {
        out[i] = ai_sum;
    }
}