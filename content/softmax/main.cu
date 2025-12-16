#include <cuda_runtime.h>
#include <armadillo>

#include <cassert>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <string>
#include <tuple>

#define NUM_THREAD_IN_WARP 32
#define NUM_WARP_IN_BLOCK 4
namespace softmax {
#include "softmax_v1.h"
#include "softmax_v2.h"
#include "softmax_v3.h"
#include "softmax_v4.h"
#include "softmax_v5.h"

void kernel_cpu(float* out, const float* inp, const size_t nrow, const size_t ncol) {
    for (auto i = 0; i < nrow; i++) {
        const float* ai_ptr = inp + i * ncol;
        float* ci_ptr = out + i * ncol;

        float ai_max = -INFINITY;
        for (auto j = 0; j < ncol; j++) {
            float aij = *(ai_ptr + j);
            ai_max = fmaxf(ai_max, aij);
        }

        float ai_sum = 0.0;
        for (auto j = 0; j < ncol; j++) {
            float aij = *(ai_ptr + j);
            float exp_aij = expf(aij - ai_max);
            ai_sum += exp_aij;
            *(ci_ptr + j) = exp_aij;
        }

        for (auto j = 0; j < ncol; j++) {
            *(ci_ptr + j) /= ai_sum;
        }
    }
}
}  // namespace softmax

#include "utils.h"

int main() {
    arma::arma_rng::set_seed_random();

    for (int nrow = 256; nrow <= 4096; nrow *= 2) {
        const int ncol = NUM_THREAD_IN_WARP * NUM_WARP_IN_BLOCK;
        arma::fmat inp = arma::randn<arma::fmat>(nrow, ncol);

        int num_block_in_grid = nrow;
        auto block_dim = dim3(1);
        auto grid_dim = dim3(num_block_in_grid);
        auto config = KernelLaunchConfig(softmax::kernel_v1, "softmax_f32_v1", 10, 100, block_dim, grid_dim, 0);
        auto result = config.run(inp);
        result.print(true, true);

        block_dim = dim3(1);
        config = KernelLaunchConfig(softmax::kernel_v2, "softmax_f32_v2", 10, 100, block_dim, grid_dim, 0);
        result = config.run(inp);
        result.print(false, true);

        block_dim = dim3(NUM_THREAD_IN_WARP);
        config = KernelLaunchConfig(softmax::kernel_v3, "softmax_f32_v3", 10, 100, block_dim, grid_dim, 0);
        result = config.run(inp);
        result.print(false, true);

        block_dim = dim3(NUM_THREAD_IN_WARP, NUM_WARP_IN_BLOCK);
        config = KernelLaunchConfig(softmax::kernel_v4, "softmax_f32_v4", 10, 100, block_dim, grid_dim, 0);
        result = config.run(inp);
        result.print(false, true);

        constexpr int num_rows_per_access = 4;
        constexpr int num_cols_per_thread = NUM_WARP_IN_BLOCK;

        auto kernel_v5 = softmax::kernel_v5<num_rows_per_access, num_cols_per_thread>;
        auto num_warp_in_grid = (nrow + num_rows_per_access - 1) / num_rows_per_access;
        num_block_in_grid = (num_warp_in_grid + NUM_WARP_IN_BLOCK - 1) / NUM_WARP_IN_BLOCK;
        block_dim = dim3(NUM_THREAD_IN_WARP, NUM_WARP_IN_BLOCK);
        grid_dim = dim3(num_block_in_grid);
        config = KernelLaunchConfig(kernel_v5, "softmax_f32_v5", 10, 100, block_dim, grid_dim, 0);
        result = config.run(inp);
        result.print(false, true);
    }

    return 0;
}