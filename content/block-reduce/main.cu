#include <cuda_runtime.h>

#include <xtensor/containers/xarray.hpp>
#include <xtensor/generators/xrandom.hpp>

#include "utils.cu"

namespace block_reduce {
#include "block-reduce-v1.cu"
#include "block-reduce-v2.cu"
#include "block-reduce-v3.cu"
}  // namespace block_reduce

int main() {
    for (int n = 64; n <= 2048; n *= 2) {
        const int nrow = n;
        const int ncol = n;
        const xt::xarray<float> inp = xt::random::randn<float>({nrow, ncol});

        int num_block_in_grid = nrow;
        int num_warp_in_block = ncol / NUM_THREAD_IN_WARP;
        int num_thread_in_block = num_warp_in_block * NUM_THREAD_IN_WARP;
        assert(num_thread_in_block == ncol);
        auto block_dim = dim3(NUM_THREAD_IN_WARP, num_warp_in_block);
        auto grid_dim = dim3(num_block_in_grid);
        auto config = KernelLaunchConfig(block_reduce::kernel_v1, "block_reduce_f32_v1", block_dim,
                                         grid_dim, ncol * sizeof(float));
        auto result = config.run(inp);
        result.print(true, true);

        num_warp_in_block /= 2;
        num_thread_in_block = num_warp_in_block * NUM_THREAD_IN_WARP;
        assert(num_thread_in_block * 2 == ncol);
        block_dim = dim3(NUM_THREAD_IN_WARP, num_warp_in_block);
        config = KernelLaunchConfig(block_reduce::kernel_v2, "block_reduce_f32_v2", block_dim,
                                    grid_dim, ncol * sizeof(float));
        result = config.run(inp);
        result.print(false, true);

        num_warp_in_block = ncol / NUM_THREAD_IN_WARP;
        num_thread_in_block = num_warp_in_block * NUM_THREAD_IN_WARP;
        assert(num_thread_in_block == ncol);
        block_dim = dim3(NUM_THREAD_IN_WARP, num_warp_in_block);
        config = KernelLaunchConfig(block_reduce::kernel_v3, "block_reduce_f32_v3", block_dim,
                                    grid_dim, ncol * sizeof(float));
        result = config.run(inp);
        result.print(false, true);
    }

    return 0;
}