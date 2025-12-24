#include <cuda_runtime.h>

#include <format>
#include <xtensor/containers/xarray.hpp>
#include <xtensor/generators/xrandom.hpp>

#include "utils.cu"

namespace block_reduce {
#include "block-reduce-v0.cu"
#include "block-reduce-v1.cu"
#include "block-reduce-v2.cu"
#include "block-reduce-v3.cu"
}  // namespace block_reduce

int main() {
    const int nrow = 1024;
    for (int ncol = 128; ncol <= 4096; ncol *= 2) {
        const xt::xarray<float> inp = xt::random::randn<float>({nrow, ncol});

        const int num_block_in_grid = nrow;

        int num_thread_in_block = std::min(ncol, NUM_THREAD_IN_BLOCK_MAX);
        int num_warp_in_block = num_thread_in_block / NUM_THREAD_IN_WARP;
        auto block_dim = dim3(NUM_THREAD_IN_WARP, num_warp_in_block);
        auto grid_dim = dim3(num_block_in_grid);

        std::string title =
            std::format("block_reduce_f32_v0 (block_size = {:4})", num_thread_in_block);
        auto config =
            KernelLaunchConfig(block_reduce::kernel_v0, title.c_str(), block_dim, grid_dim);
        auto result = config.run(inp);
        result.print(true, true);

        int shared_mem_size = num_thread_in_block * sizeof(float);
        title = std::format("block_reduce_f32_v1 (block_size = {:4})", num_thread_in_block);
        config = KernelLaunchConfig(block_reduce::kernel_v1, title.c_str(), block_dim, grid_dim,
                                    shared_mem_size);
        result = config.run(inp);
        result.print(false, true);

        title = std::format("block_reduce_f32_v2 (block_size = {:4})", num_thread_in_block);
        config = KernelLaunchConfig(block_reduce::kernel_v2, title.c_str(), block_dim, grid_dim,
                                    shared_mem_size);
        result = config.run(inp);
        result.print(false, true);

        title = std::format("block_reduce_f32_v3 (block_size = {:4})", num_thread_in_block);
        config = KernelLaunchConfig(block_reduce::kernel_v3, title.c_str(), block_dim, grid_dim,
                                    shared_mem_size);
        result = config.run(inp);
        result.print(false, true);

        num_thread_in_block /= 2;
        num_warp_in_block = num_thread_in_block / NUM_THREAD_IN_WARP;
        block_dim = dim3(NUM_THREAD_IN_WARP, num_warp_in_block);
        grid_dim = dim3(num_block_in_grid);

        title = std::format("block_reduce_f32_v0 (block_size = {:4})", num_thread_in_block);
        config = KernelLaunchConfig(block_reduce::kernel_v0, title.c_str(), block_dim, grid_dim);
        result = config.run(inp);
        result.print(false, true);

        shared_mem_size = num_thread_in_block * sizeof(float);
        title = std::format("block_reduce_f32_v1 (block_size = {:4})", num_thread_in_block);
        config = KernelLaunchConfig(block_reduce::kernel_v1, title.c_str(), block_dim, grid_dim,
                                    shared_mem_size);
        result = config.run(inp);
        result.print(false, true);

        title = std::format("block_reduce_f32_v2 (block_size = {:4})", num_thread_in_block);
        config = KernelLaunchConfig(block_reduce::kernel_v2, title.c_str(), block_dim, grid_dim,
                                    shared_mem_size);
        result = config.run(inp);
        result.print(false, true);

        title = std::format("block_reduce_f32_v3 (block_size = {:4})", num_thread_in_block);
        config = KernelLaunchConfig(block_reduce::kernel_v3, title.c_str(), block_dim, grid_dim,
                                    shared_mem_size);
        result = config.run(inp);
        result.print(false, true);
    }

    return 0;
}