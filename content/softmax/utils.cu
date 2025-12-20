#pragma once
// basic utilities
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cub/cub.cuh>

// xtensor related
#include <xtensor/containers/xarray.hpp>
#include <xtensor/generators/xrandom.hpp>

// timing utilities
#include <chrono>
using hrc = std::chrono::high_resolution_clock;
using dt = std::chrono::duration<float, std::milli>;

// assertion utilities, only used in debug mode
#include <cassert>

// constants
#define FULL 0xffffffff
#define NUM_THREAD_IN_WARP 32
#define NUM_WARP_IN_BLOCK 4

namespace softmax {
void kernel_cpu(float* out, const float* inp, const int nrow, const int ncol) {
    for (int i = 0; i < nrow; i++) {
        const float* ai_ptr = inp + i * ncol;
        float* ci_ptr = out + i * ncol;

        float ai_max = -INFINITY;
        for (int j = 0; j < ncol; j++) {
            float aij = *(ai_ptr + j);
            ai_max = fmaxf(ai_max, aij);
        }

        float ai_sum = 0.0;
        for (int j = 0; j < ncol; j++) {
            float aij = *(ai_ptr + j);
            float exp_aij = expf(aij - ai_max);
            ai_sum += exp_aij;
            *(ci_ptr + j) = exp_aij;
        }

        for (int j = 0; j < ncol; j++) {
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

class BenchmarkResult {
  public:
    float time_cpu_ms;
    float time_gpu_ms;
    xt::xarray<float> error_cpu;
    xt::xarray<float> error_gpu;
    const char* title;

    void print(bool cpu = true, bool gpu = true) const {
        auto time_cpu_ms = this->time_cpu_ms;
        auto time_gpu_ms = this->time_gpu_ms;
        auto error_cpu = xt::amax(this->error_cpu);
        auto error_gpu = xt::amax(this->error_gpu);
        auto nrow = this->error_cpu.shape(0);

        if (cpu) {
            auto error = xt::amax(this->error_cpu)();
            printf("\n%-16s, nrow: %6zu, time: %-6.2e ms,  error: %-6.2e\n", "softmax_f32_cpu",
                   nrow, time_cpu_ms, error);
        }
        if (gpu) {
            auto error = xt::amax(this->error_gpu)();
            printf("%-16s, nrow: %6zu, time: %-6.2e ms,  error: %-6.2e\n", this->title, nrow,
                   time_gpu_ms, error);
        }
    }
};

template <typename kernel_t>
class KernelLaunchConfig {
  public:
    kernel_t kernel;
    const char* title;
    int warmup;
    int repeat;
    dim3 block_dim;
    dim3 grid_dim;
    int shared_mem_size;

    KernelLaunchConfig(kernel_t kernel, const char* title, int warmup, int repeat, dim3 block_dim,
                       dim3 grid_dim, int shared_mem_size)
        : kernel(kernel), title(title), warmup(warmup), repeat(repeat), block_dim(block_dim),
          grid_dim(grid_dim), shared_mem_size(shared_mem_size) {}

    KernelLaunchConfig(kernel_t kernel, const char* title, dim3 block_dim, dim3 grid_dim,
                       int shared_mem_size)
        : kernel(kernel), title(title), warmup(10), repeat(100), block_dim(block_dim),
          grid_dim(grid_dim), shared_mem_size(shared_mem_size) {}

    BenchmarkResult run(const xt::xarray<float>& inp) {
        const int nrow = inp.shape(0);
        const int ncol = inp.shape(1);
        const int size = nrow * ncol;
        const int mem_size = size * sizeof(float);

        xt::xarray<float> out_ref = xt::zeros<float>({nrow, ncol});
        softmax::kernel_ref(out_ref, inp);

        xt::xarray<float> out_cpu = xt::zeros<float>({nrow, ncol});
        xt::xarray<float> out_gpu = xt::zeros<float>({nrow, ncol});
        const float* inp_cpu_ptr = inp.data();
        float* out_cpu_ptr = out_cpu.data();

        // CPU softmax
        auto t0_cpu = hrc::now();
        for (int count = 0; count < repeat; count++) {
            softmax::kernel_cpu(out_cpu_ptr, inp_cpu_ptr, nrow, ncol);
        }
        auto t1_cpu = hrc::now();
        float time_cpu_ms = dt(t1_cpu - t0_cpu).count();
        time_cpu_ms /= this->repeat;

        // GPU buffers
        float *inp_gpu_ptr, *out_gpu_ptr;
        cudaMalloc(&inp_gpu_ptr, mem_size);
        cudaMalloc(&out_gpu_ptr, mem_size);
        cudaMemcpy(inp_gpu_ptr, inp_cpu_ptr, mem_size, cudaMemcpyHostToDevice);

        // lambda to launch the kernel
        auto launch = [&](float* y, const float* x) {
            void* args[] = {(void*)&y, (void*)&x, (void*)&nrow, (void*)&ncol};

            cudaLaunchKernel((void*)this->kernel, this->grid_dim, this->block_dim, args,
                             this->shared_mem_size,
                             0  // stream
            );
        };

        // Warmup
        for (int x = 0; x < this->warmup; x++) {
            launch(out_gpu_ptr, inp_gpu_ptr);
        }
        cudaDeviceSynchronize();

        // Timing the CUDA execution
        cudaEvent_t t0, t1;
        cudaEventCreate(&t0);
        cudaEventCreate(&t1);

        cudaEventRecord(t0);
        for (int count = 0; count < this->repeat; count++) {
            launch(out_gpu_ptr, inp_gpu_ptr);
        }
        cudaEventRecord(t1);
        cudaEventSynchronize(t1);

        float time_gpu_ms = 0.0;
        cudaEventElapsedTime(&time_gpu_ms, t0, t1);
        time_gpu_ms /= this->repeat;

        // copy GPU result to CPU (out_gpu to sol.data())
        cudaMemcpy(out_gpu.data(), out_gpu_ptr, mem_size, cudaMemcpyDeviceToHost);
        // copy CPU result to CPU (ref.data() to out_cpu)
        // memcpy(out_cpu.data(), out_cpu, mem_size);

        // Cleanup
        cudaEventDestroy(t0);
        cudaEventDestroy(t1);
        cudaFree(inp_gpu_ptr);
        cudaFree(out_gpu_ptr);

        //
        auto result = BenchmarkResult();
        result.time_cpu_ms = time_cpu_ms;
        result.time_gpu_ms = time_gpu_ms;
        result.error_cpu = xt::abs(out_ref - out_cpu);
        result.error_gpu = xt::abs(out_ref - out_gpu);
        result.title = this->title;
        return result;
    }
};
