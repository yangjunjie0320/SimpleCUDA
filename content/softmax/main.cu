#include <cuda_runtime.h>
#include <xtensor/containers/xarray.hpp>
#include <xtensor/generators/xrandom.hpp>

#include <cassert>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <string>
#include <tuple>
#include <random>

#define NUM_THREAD_IN_WARP 32
#define NUM_WARP_IN_BLOCK 4

namespace softmax {
    #include "softmax_v1.h"
    #include "softmax_v2.h"
    #include "softmax_v3.h"
    #include "softmax_v4.h"
    #include "softmax_v5.h"

    void kernel_cpu(float* out, const float* inp, const size_t nrow, const size_t ncol) {
        for (size_t i = 0; i < nrow; i++) {
            const float* ai_ptr = inp + i * ncol;
            float* ci_ptr = out + i * ncol;

            float ai_max = -INFINITY;
            for (size_t j = 0; j < ncol; j++) {
                float aij = *(ai_ptr + j);
                ai_max = fmaxf(ai_max, aij);
            }

            float ai_sum = 0.0;
            for (size_t j = 0; j < ncol; j++) {
                float aij = *(ai_ptr + j);
                float exp_aij = expf(aij - ai_max);
                ai_sum += exp_aij;
                *(ci_ptr + j) = exp_aij;
            }

            for (size_t j = 0; j < ncol; j++) {
                *(ci_ptr + j) /= ai_sum;
            }
        }
    }

    void kernel_ref(xt::xarray<float>& out, const xt::xarray<float>& inp) {
        xt::xarray<float> a_max = xt::amax(inp, {1}, xt::keep_dims);
        xt::xarray<float> a_exp = xt::exp(inp - a_max);
        xt::xarray<float> a_sum = xt::sum(a_exp, {1}, xt::keep_dims);
        out = a_exp / a_sum;
    }
}  // namespace softmax

#include "utils.h"

int main() {
    for (int nrow = 256; nrow <= 4096; nrow *= 2) {
        const int ncol = NUM_THREAD_IN_WARP * NUM_WARP_IN_BLOCK;
        xt::xarray<float> inp = xt::random::randn<float>({nrow, ncol});

        int num_block_in_grid = nrow;
        auto block_dim = dim3(1);
        auto grid_dim = dim3(num_block_in_grid);
        auto config = KernelLaunchConfig(softmax::kernel_v1, "softmax_f32_v1",
                                         block_dim, grid_dim, 0);
        auto result = config.run(inp);
        result.print(true, true);

        block_dim = dim3(1);
        config = KernelLaunchConfig(softmax::kernel_v2, "softmax_f32_v2", block_dim, grid_dim, ncol * nrow * sizeof(float));
        result = config.run(inp);
        result.print(false, true);

        block_dim = dim3(NUM_THREAD_IN_WARP);
        config = KernelLaunchConfig(softmax::kernel_v3, "softmax_f32_v3", block_dim, grid_dim, 0);
        result = config.run(inp);
        result.print(false, true);

        block_dim = dim3(NUM_THREAD_IN_WARP, NUM_WARP_IN_BLOCK);
        config = KernelLaunchConfig(softmax::kernel_v4, "softmax_f32_v4", block_dim, grid_dim, 0);
        result = config.run(inp);
        result.print(false, true);

        // constexpr int num_rows_per_access = 4;
        // constexpr int num_cols_per_thread = NUM_WARP_IN_BLOCK;
        // auto kernel_v5 = softmax::kernel_v5<num_rows_per_access, num_cols_per_thread>;
        // auto num_warp_in_grid = (nrow + num_rows_per_access - 1) / num_rows_per_access;
        // num_block_in_grid = (num_warp_in_grid + NUM_WARP_IN_BLOCK - 1) / NUM_WARP_IN_BLOCK;
        // block_dim = dim3(NUM_THREAD_IN_WARP, NUM_WARP_IN_BLOCK);
        // grid_dim = dim3(num_block_in_grid);
        // config = KernelLaunchConfig(kernel_v5, "softmax_f32_v5", block_dim, grid_dim, 0);
        // result = config.run(inp);
        // result.print(false, true);
    }

    return 0;
}