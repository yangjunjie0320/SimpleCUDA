#include <cuda_runtime.h>
#include <torch/torch.h>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <tuple>

#define WARP_SIZE 32
#include "softmax_v1.cuh"
#include "softmax_v2.cuh"

void softmax_f32_cpu(float* out, const float* inp, const size_t ncol, const size_t nrow) {
    for (auto i = 0; i < nrow; i++) {
        const float* ai_ptr = inp + i * ncol;
        float* ci_ptr = out + i * ncol;

        float ai_max = -INFINITY;
        for (auto j = 0; j < ncol; j++) {
            auto aij = ai_ptr[j];
            ai_max = std::max(ai_max, aij);
        }

        float norm = 0.0;
        for (auto j = 0; j < ncol; j++) {
            auto aij = ai_ptr[j];
            norm += std::exp(aij - ai_max);
        }

        for (auto j = 0; j < ncol; j++) {
            auto aij = ai_ptr[j];
            auto cij = std::exp(aij - ai_max) / norm;
            ci_ptr[j] = cij;
        }
    }
}

using kernel_t = void (*)(float*, const float*, size_t, size_t);

std::tuple<float, float, float> benchmark(kernel_t kernel, const torch::Tensor& inp,
                                          const size_t warmup = 10, const size_t repeat = 100) {
    auto ref = torch::zeros_like(inp);
    auto sol = torch::zeros_like(inp);

    auto ncol = inp.size(1);
    auto nrow = inp.size(0);
    auto mem_size = ncol * nrow * sizeof(float);

    float *inp_cpu, *out_cpu;
    inp_cpu = (float*)malloc(mem_size);
    out_cpu = (float*)malloc(mem_size);
    memcpy(inp_cpu, inp.data_ptr<float>(), mem_size);

    auto t0_cpu = std::chrono::high_resolution_clock::now();
    for (size_t x = 0; x < repeat; x++) {
        softmax_f32_cpu(out_cpu, inp_cpu, ncol, nrow);
    }
    auto t1_cpu = std::chrono::high_resolution_clock::now();
    float time_cpu_ms = std::chrono::duration<float, std::milli>(t1_cpu - t0_cpu).count();
    time_cpu_ms /= repeat;

    float *inp_gpu, *out_gpu;
    cudaMalloc(&inp_gpu, mem_size);
    cudaMalloc(&out_gpu, mem_size);
    cudaMemcpy(inp_gpu, inp_cpu, mem_size, cudaMemcpyHostToDevice);

    // Warmup
    for (size_t x = 0; x < warmup; x++) {
        kernel(out_gpu, inp_gpu, ncol, nrow);
    }
    cudaDeviceSynchronize();

    // Timing the CUDA execution
    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);

    cudaEventRecord(t0);
    for (size_t x = 0; x < repeat; x++) {
        kernel(out_gpu, inp_gpu, ncol, nrow);
    }
    cudaEventRecord(t1);
    cudaEventSynchronize(t1);

    float time_gpu_ms;
    cudaEventElapsedTime(&time_gpu_ms, t0, t1);
    time_gpu_ms /= repeat;

    memcpy(ref.data_ptr<float>(), out_cpu, mem_size);
    cudaMemcpy(sol.data_ptr<float>(), out_gpu, mem_size, cudaMemcpyDeviceToHost);
    auto err = torch::abs(ref - sol).max().item<float>();

    // Cleanup
    free(inp_cpu);
    free(out_cpu);
    cudaFree(inp_gpu);
    cudaFree(out_gpu);

    return {time_cpu_ms, time_gpu_ms, err};
}

int main() {
    // printf("--------------------------------------------------\n");
    // printf("%-10s %-12s %-12s %-12s\n", "N", "CPU (ms)", "V1 (ms)", "Error");
    // printf("--------------------------------------------------\n");

    for (size_t nrow = 256; nrow <= 4096; nrow *= 2) {
        const int ncol = WARP_SIZE * 4;
        torch::Tensor inp = torch::randn({n, ncol});

        auto [time_cpu_ms, time_gpu_ms, err] = benchmark(softmax_f32_v1, inp);
        printf("%-16s, nrow: %6d, time: %-6.2e ms,  error: %-6.2e\n", "softmax_f32_cpu", nrow,
               time_cpu_ms, 0.0);
        printf("%-16s, nrow: %6d, time: %-6.2e ms,  error: %-6.2e\n", "softmax_f32_v1", nrow,
               time_gpu_ms, err);

        auto [time_cpu_ms, time_gpu_ms, err] = benchmark(softmax_f32_v2, inp);
        printf("%-16s, nrow: %6d, time: %-6.2e ms,  error: %-6.2e\n", "softmax_f32_v2", nrow,
               time_gpu_ms, err);
    }

    return 0;
}