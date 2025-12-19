#include "utils.cu"

// each block handles one row, use ncol threads
__global__ void kernel_v2(float* out, const float* inp, int nrow, int ncol) {
    const auto idx_lane = threadIdx.x;
    
    const auto num_thread_in_block = blockDim.x;
    const auto num_warp_in_block = blockDim.y;
    const auto num_block_in_grid = gridDim.x;

    const auto idx_thread_in_block = threadIdx.x;
    const auto idx_block_in_grid = blockIdx.x;
    const auto idx_thread_in_grid = threadIdx.x + blockIdx.x * blockDim.x;

    // sanity check
    assert (num_thread_in_block == NUM_THREAD_IN_WARP);
    assert (num_warp_in_block == 1);
    assert (num_block_in_grid == nrow);

    const auto i = idx_block_in_grid;
    const float* ai_ptr = inp + i * ncol;
    float* ci_ptr = out + i * ncol;

    __shared__ float ai_max, ai_sum;

    auto col_base = idx_thread_in_block;
    auto col_step = num_thread_in_block;

    float ai_max_in_thread = -INFINITY;
    for (auto j = col_base; j < ncol; j += col_step) {
        float aij = ai_ptr[j];
        ai_max_in_thread = fmaxf(ai_max_in_thread, aij);
    }
    ai_max = ai_max_in_thread;

    #pragma unroll
    for (auto offset = NUM_THREAD_IN_WARP / 2; offset > 0; offset >>= 1) {
        if (idx_lane < offset) {
            auto ai_max_curr_lane = ai_max_in_thread;
            auto ai_max_next_lane = __shfl_down_sync(FULL, ai_max_curr_lane, offset);
            ai_max_in_thread = fmaxf(ai_max_curr_lane, ai_max_next_lane);
        }
    }

    float ai_max_in_warp = smem[0];
    ai_max = ai_max_in_warp;
    __syncthreads();

    float ai_sum_in_thread = 0.0;
    for (auto j = col_base; j < ncol; j += col_step) {
        float aij = ai_ptr[j];
        ai_sum_in_thread += expf(aij - ai_max);
    }
    smem[idx_thread_in_block] = ai_sum_in_thread;
    __syncthreads();

    // Tree reduction for sum
    for (auto offset = NUM_THREAD_IN_WARP / 2; offset > 0; offset >>= 1) {
        if (idx_lane < offset) {
            auto ai_sum_curr_lane = smem[idx_lane];
            auto ai_sum_next_lane = smem[idx_lane + offset];
            smem[idx_lane] += ai_sum_next_lane;
        }
        __syncthreads();
    }
    float ai_sum_in_warp = smem[0];
    ai_sum = ai_sum_in_warp;
    __syncthreads();

    for (auto j = col_base; j < ncol; j += col_step) {
        float aij = ai_ptr[j];
        float cij = expf(aij - ai_max) / ai_sum;
        ci_ptr[j] = cij;
    }   
}
