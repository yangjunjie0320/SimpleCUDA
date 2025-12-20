#include "utils.cu"

// each block handles one row with ncol threads, CUB BlockReduce
__global__ void kernel_v3(float* out, const float* inp, int nrow, int ncol) {
    const auto num_thread_in_warp = blockDim.x;
    const auto num_warp_in_block = blockDim.y;
    const auto num_thread_in_block = num_thread_in_warp * num_warp_in_block;
    const auto num_block_in_grid = gridDim.x;

    const auto idx_thread_in_block = threadIdx.x + threadIdx.y * blockDim.x;
    const auto idx_block_in_grid = blockIdx.x;

    // sanity check
    assert(num_thread_in_warp == NUM_THREAD_IN_WARP);
    assert(num_warp_in_block == NUM_WARP_IN_BLOCK);
    assert(num_thread_in_block == NUM_THREAD_IN_WARP * NUM_WARP_IN_BLOCK);
    assert(num_thread_in_block == ncol && num_block_in_grid == nrow);

    const auto i = idx_block_in_grid;
    const auto j = idx_thread_in_block;
    const float* ai_ptr = inp + i * ncol;
    const float aij = ai_ptr[j];

    using BlockReduce = cub::BlockReduce<float, NUM_THREAD_IN_WARP,
                                         cub::BLOCK_REDUCE_WARP_REDUCTIONS, NUM_WARP_IN_BLOCK, 1>;
    __shared__ typename BlockReduce::TempStorage tmp;
    __shared__ float ai_max, ai_sum;

    const float ai_max_in_block = BlockReduce(tmp).Reduce(aij, cub::Max());
    if (idx_thread_in_block == 0) {
        ai_max = ai_max_in_block;
    }
    __syncthreads();

    const float exp_aij = expf(aij - ai_max);
    __syncthreads();

    const float ai_sum_in_block = BlockReduce(tmp).Reduce(exp_aij, cub::Sum());
    if (idx_thread_in_block == 0) {
        ai_sum = ai_sum_in_block;
    }
    __syncthreads();

    const float ai_sum_inv = 1.0 / ai_sum;
    const float cij = exp_aij * ai_sum_inv;
    float* ci_ptr = out + i * ncol;
    ci_ptr[j] = cij;
}
