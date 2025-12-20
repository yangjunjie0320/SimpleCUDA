#include "utils.cu"

// each block handles one row with ncol threads, smem tree reduction, ncol must equal block size
__global__ void kernel_v2(float* out, const float* inp, int nrow, int ncol) {
    const auto num_thread_in_warp = blockDim.x;
    const auto num_warp_in_block = blockDim.y;
    const auto num_thread_in_block = num_thread_in_warp * num_warp_in_block;
    const auto num_block_in_grid = gridDim.x;

    const auto idx_thread_in_block = threadIdx.x + threadIdx.y * NUM_THREAD_IN_WARP;
    const auto idx_block_in_grid = blockIdx.x;
    constexpr auto offset0 = NUM_THREAD_IN_WARP * NUM_WARP_IN_BLOCK / 2;

    // sanity check
    assert(num_warp_in_block == NUM_WARP_IN_BLOCK);
    assert(num_thread_in_block == NUM_THREAD_IN_WARP * NUM_WARP_IN_BLOCK);
    assert(num_thread_in_block == ncol && num_block_in_grid == nrow);
    assert(offset0 * 2 == num_thread_in_block);

    const auto i = idx_block_in_grid;
    const auto j = idx_thread_in_block;
    const float* ai_ptr = inp + i * ncol;
    const float aij = ai_ptr[j];

    __shared__ float smem[NUM_THREAD_IN_WARP * NUM_WARP_IN_BLOCK];

    smem[idx_thread_in_block] = aij;
    __syncthreads();

#pragma unroll
    for (auto offset = offset0; offset > 0; offset >>= 1) {
        if (idx_thread_in_block < offset) {
            auto ai_max_curr_thread = smem[idx_thread_in_block];
            auto ai_max_next_thread = smem[idx_thread_in_block + offset];
            smem[idx_thread_in_block] = fmaxf(ai_max_curr_thread, ai_max_next_thread);
        }
        __syncthreads();
    }

    const float ai_max_in_block = smem[0];
    const float ai_max = ai_max_in_block;
    __syncthreads();

    const float exp_aij = expf(aij - ai_max);
    smem[idx_thread_in_block] = exp_aij;
    __syncthreads();

#pragma unroll
    for (auto offset = offset0; offset > 0; offset >>= 1) {
        if (idx_thread_in_block < offset) {
            auto ai_sum_curr_thread = smem[idx_thread_in_block];
            auto ai_sum_next_thread = smem[idx_thread_in_block + offset];
            smem[idx_thread_in_block] += ai_sum_next_thread;
        }
        __syncthreads();
    }

    const float ai_sum_in_block = smem[0];
    const float ai_sum = ai_sum_in_block;
    __syncthreads();

    const float ai_sum_inv = 1.0 / ai_sum;
    const float cij = exp_aij * ai_sum_inv;
    float* ci_ptr = out + i * ncol;
    ci_ptr[j] = cij;
}
