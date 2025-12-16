#include <armadillo>
#include <string>
#include <chrono>

#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <cmath>

class BenchmarkResult {
  public:
    float time_cpu_ms;
    float time_gpu_ms;
    float max_error;
    std::size_t nrow;
    std::string kernel_name;

    void print(bool cpu = true, bool gpu = true) const {
        auto name = this->kernel_name.c_str();
        auto time_cpu_ms = this->time_cpu_ms;
        auto time_gpu_ms = this->time_gpu_ms;
        auto nrow = this->nrow;

        if (cpu) {
            printf("\n%-16s, nrow: %6zu, time: %-6.2e ms,  error: %-6.2e\n",
                   "softmax_f32_cpu",
                   nrow,
                   time_cpu_ms,
                   0.0);
        }
        if (gpu) {
            printf("%-16s, nrow: %6zu, time: %-6.2e ms,  error: %-6.2e\n",
                   name,
                   nrow,
                   time_gpu_ms,
                   max_error);
        }
    }
};

template <typename kernel_t>
class KernelLaunchConfig {
  public:
    kernel_t   kernel;
    const char* kernel_name;
    std::size_t warmup;
    std::size_t repeat;
    dim3       block_dim;
    dim3       grid_dim;
    std::size_t shared_mem_size;

    KernelLaunchConfig(kernel_t kernel,
                       const char* kernel_name,
                       std::size_t warmup,
                       std::size_t repeat,
                       dim3 block_dim,
                       dim3 grid_dim,
                       std::size_t shared_mem_size)
        : kernel(kernel)
        , kernel_name(kernel_name)
        , warmup(warmup)
        , repeat(repeat)
        , block_dim(block_dim)
        , grid_dim(grid_dim)
        , shared_mem_size(shared_mem_size) {}

    KernelLaunchConfig(kernel_t kernel, const char* kernel_name, dim3 block_dim, dim3 grid_dim, std::size_t shared_mem_size)
        : kernel(kernel)
        , kernel_name(kernel_name)
        , warmup(10)
        , repeat(100)
        , block_dim(block_dim)
        , grid_dim(grid_dim)
        , shared_mem_size(shared_mem_size) {}

    BenchmarkResult run(const arma::fmat& inp_mat) {
        auto kernel_name = this->kernel_name;
        auto warmup = this->warmup;
        auto repeat = this->repeat;

        const std::size_t nrow = inp_mat.n_rows;
        const std::size_t ncol = inp_mat.n_cols;
        const std::size_t size = nrow * ncol;
        const std::size_t mem_size = size * sizeof(float);

        auto out_cpu = arma::fmat(nrow, ncol);
        auto out_gpu = arma::fmat(nrow, ncol);

        float* inp_cpu_ptr = inp_mat.memptr();
        float* out_cpu_ptr = malloc(mem_size);

        // CPU softmax
        auto t0_cpu = std::chrono::high_resolution_clock::now();
        for (std::size_t x = 0; x < repeat; x++) {
            softmax::kernel_cpu(out_cpu_ptr, inp_cpu_ptr, nrow, ncol);
        }
        auto t1_cpu = std::chrono::high_resolution_clock::now();
        float time_cpu_ms = std::chrono::duration<float, std::milli>(t1_cpu - t0_cpu).count();
        time_cpu_ms /= static_cast<float>(repeat);

        // GPU buffers
        float *inp_gpu = nullptr, *out_gpu = nullptr;
        cudaMalloc(&inp_gpu, mem_size);
        cudaMalloc(&out_gpu, mem_size);
        cudaMemcpy(inp_gpu, inp_cpu_ptr, mem_size, cudaMemcpyHostToDevice);

        // lambda to launch the kernel
        auto launch = [&](float* out_gpu_local, const float* inp_gpu_local) {
            void* args[] = {
                (void*)&out_gpu_local,
                (void*)&inp_gpu_local,
                (void*)&nrow,
                (void*)&ncol
            };
            cudaLaunchKernel(
                (void*)this->kernel,
                this->grid_dim,
                this->block_dim,
                args,
                this->shared_mem_size,
                0  // stream
            );
        };

        // Warmup
        for (std::size_t x = 0; x < warmup; x++) {
            launch(out_gpu, inp_gpu);
        }
        cudaDeviceSynchronize();

        // Timing the CUDA execution
        cudaEvent_t t0, t1;
        cudaEventCreate(&t0);
        cudaEventCreate(&t1);

        cudaEventRecord(t0);
        for (std::size_t x = 0; x < repeat; x++) {
            launch(out_gpu, inp_gpu);
        }
        cudaEventRecord(t1);
        cudaEventSynchronize(t1);

        float time_gpu_ms = 0.0f;
        cudaEventElapsedTime(&time_gpu_ms, t0, t1);
        time_gpu_ms /= static_cast<float>(repeat);

        // copy GPU result back
        cudaMemcpy(out_gpu_host.data(), out_gpu, mem_size, cudaMemcpyDeviceToHost);

        // compute max absolute error between CPU and GPU outputs
        auto err = arma::abs(out_cpu - out_gpu).max();

        // Cleanup
        cudaFree(inp_gpu);
        cudaFree(out_gpu);

        BenchmarkResult result{time_cpu_ms, time_gpu_ms, max_err, nrow, kernel_name};
        return result;
    }
};
