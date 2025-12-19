#include <cuda_runtime.h>

#include <xtensor/containers/xarray.hpp>
#include <xtensor/generators/xrandom.hpp>

#include "utils.cu"

namespace softmax {
#include "softmax-v1.cu"
#include "softmax-v2.cu"
#include "softmax-v3.cu"
// #include "softmax-v4.cu"
// #include "softmax-v5.cu"
// #include "softmax-v6.cu"
}  // namespace softmax

int main() {
    for (int nrow = 256; nrow <= 8192; nrow *= 2) {
        const int ncol = NUM_THREAD_IN_WARP * NUM_WARP_IN_BLOCK;
        xt::xarray<float> inp = xt::random::randn<float>({nrow, ncol});

        int num_block_in_grid = nrow;
        auto block_dim = dim3(1);
        auto grid_dim = dim3(num_block_in_grid);
        auto config =
            KernelLaunchConfig(softmax::kernel_v1, "softmax_f32_v1", block_dim, grid_dim, 0);
        auto result = config.run(inp);
        result.print(true, true);

        block_dim = dim3(NUM_THREAD_IN_WARP, NUM_WARP_IN_BLOCK);
        config = KernelLaunchConfig(softmax::kernel_v2, "softmax_f32_v2", block_dim, grid_dim, 0);
        result = config.run(inp);
        result.print(false, true);

        block_dim = dim3(NUM_THREAD_IN_WARP, NUM_WARP_IN_BLOCK);
        config = KernelLaunchConfig(softmax::kernel_v3, "softmax_f32_v3", block_dim, grid_dim, 0);
        result = config.run(inp);
        result.print(false, true);

        // block_dim = dim3(NUM_THREAD_IN_WARP);
        // config = KernelLaunchConfig(softmax::kernel_v3, "softmax_f32_v3", block_dim, grid_dim, 0);
        // result = config.run(inp);
        // result.print(false, true);

        // block_dim = dim3(NUM_THREAD_IN_WARP, NUM_WARP_IN_BLOCK);
        // config = KernelLaunchConfig(softmax::kernel_v4, "softmax_f32_v4", block_dim, grid_dim, 0);
        // result = config.run(inp);
        // result.print(false, true);

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