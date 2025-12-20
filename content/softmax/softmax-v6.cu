#include "utils.cu"

template <int num_rows_per_access, int num_cols_per_thread>
__global__ void kernel_v6(float* out, const float* inp, int nrow, int ncol) {
    const auto num_thread_in_warp = blockDim.x;
    const auto num_warp_in_block = blockDim.y;
    const auto num_block_in_grid = gridDim.x;

    const auto idx_lane = threadIdx.x;
    const auto idx_block_in_grid = blockIdx.x;
    const auto idx_warp_in_block = threadIdx.y;
    const auto idx_warp_in_grid = idx_block_in_grid * num_warp_in_block + idx_warp_in_block;
    const auto num_warp_in_grid = num_block_in_grid * num_warp_in_block;
    constexpr auto offset0 = NUM_THREAD_IN_WARP / 2;

    // sanity check
    assert(num_thread_in_warp == NUM_THREAD_IN_WARP);
    assert(num_warp_in_block == NUM_WARP_IN_BLOCK);
    assert(offset0 * 2 == num_thread_in_warp);

    float buff[num_rows_per_access][num_cols_per_thread];
    const auto row_base = idx_warp_in_grid * num_rows_per_access;
    const auto row_step = num_warp_in_grid * num_rows_per_access;
    const auto col_step = NUM_THREAD_IN_WARP;

    for (auto i0 = row_base; i0 < nrow; i0 += row_step) {
        auto i1 = i0 + num_rows_per_access;
        // j0 = idx_warp_in_block * NUM_THREAD_IN_WARP
        // j1 = j0 + NUM_THREAD_IN_WARP

        // store the max/sum for a[i, ::num_threads_in_warp] in each thread
        float ai_max_in_thread[num_rows_per_access];
        float ai_sum_in_thread[num_rows_per_access];

        // store the max/sum for a[i, j0:j1] in each warp
        float ai_max_in_warp[num_rows_per_access];
        float ai_sum_in_warp[num_rows_per_access];

#pragma unroll
        for (auto ii = 0; ii < num_rows_per_access; ii++) {
            ai_max_in_thread[ii] = -INFINITY;
            ai_sum_in_thread[ii] = 0.0f;
        }

#pragma unroll
        for (auto ii = 0; ii < num_rows_per_access; ii++) {
            auto i = i0 + ii;
            const auto ai_ptr = inp + i * ncol;

#pragma unroll
            for (auto jj = 0; jj < num_cols_per_thread; jj++) {
                auto j = idx_lane + jj * NUM_THREAD_IN_WARP;
                const auto aij = ai_ptr[j];
                ai_max_in_thread[ii] = fmaxf(ai_max_in_thread[ii], aij);
                buff[ii][jj] = aij;
            }
        }

#pragma unroll
        for (auto ii = 0; ii < num_rows_per_access; ii++) {
            float ai_t_max = ai_max_in_thread[ii];

#pragma unroll
            for (auto offset = NUM_THREAD_IN_WARP / 2; offset > 0; offset >>= 1) {
                ai_t_max = fmaxf(ai_t_max, __shfl_down_sync(FULL, ai_t_max, offset));
            }

            ai_max_in_warp[ii] = __shfl_sync(FULL, ai_t_max, 0);
        }

#pragma unroll
        for (auto ii = 0; ii < num_rows_per_access; ii++) {
#pragma unroll
            for (auto jj = 0; jj < num_cols_per_thread; jj++) {
                float aij = buff[ii][jj];
                float exp_aij = expf(aij - ai_max_in_warp[ii]);
                buff[ii][jj] = exp_aij;

                ai_sum_in_thread[ii] += exp_aij;
            }
        }

#pragma unroll
        for (auto ii = 0; ii < num_rows_per_access; ii++) {
            float ai_t_sum = ai_sum_in_thread[ii];

#pragma unroll
            for (auto offset = NUM_THREAD_IN_WARP / 2; offset > 0; offset >>= 1) {
                ai_t_sum += __shfl_down_sync(FULL, ai_t_sum, offset);
            }
            ai_sum_in_warp[ii] = __shfl_sync(FULL, ai_t_sum, 0);
        }

#pragma unroll
        for (auto ii = 0; ii < num_rows_per_access; ii++) {
            auto i = i0 + ii;
            auto ai_sum_inv = 1.0f / ai_sum_in_warp[ii];

            if (i < nrow) {
                float* ci_ptr = out + i * ncol;

#pragma unroll
                for (auto jj = 0; jj < num_cols_per_thread; jj++) {
                    auto j = idx_lane + jj * NUM_THREAD_IN_WARP;
                    float aij = buff[ii][jj];
                    float cij = aij * ai_sum_inv;
                    ci_ptr[j] = cij;
                }
            }
        }
    }
}