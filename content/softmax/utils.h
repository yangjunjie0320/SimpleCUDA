
class BenchmarkResult {
    public:
        float time_cpu_ms;
        float time_gpu_ms;
        torch::Tensor error;
        std::string kernel_name;
    
        void print(bool cpu=true, bool gpu=true) const {
            auto name = this->kernel_name.c_str();
            auto time_cpu_ms = this->time_cpu_ms;
            auto time_gpu_ms = this->time_gpu_ms;
            
            auto nrow = this->error.size(0);
            auto error = this->error.max().item<float>();
            
            if (cpu) {
                printf("\n%-16s, nrow: %6zu, time: %-6.2e ms,  error: %-6.2e\n", 
                    "softmax_f32_cpu", nrow, time_cpu_ms, 0.0
                );
            }
            if (gpu) {
                printf("%-16s, nrow: %6zu, time: %-6.2e ms,  error: %-6.2e\n", 
                    name, nrow, time_gpu_ms, error
                );
            }
        }
    };
    
    template<typename kernel_t>
    class KernelLaunchConfig {
    public:
        kernel_t kernel;
        const char* kernel_name;
        size_t warmup;
        size_t repeat;
        dim3 block_dim;
        dim3 grid_dim;
        size_t shared_mem_size;
    
        KernelLaunchConfig(kernel_t kernel, const char* kernel_name, size_t warmup, size_t repeat, dim3 block_dim, dim3 grid_dim, size_t shared_mem_size)
            : kernel(kernel), kernel_name(kernel_name), warmup(warmup), repeat(repeat), block_dim(block_dim), grid_dim(grid_dim), shared_mem_size(shared_mem_size) {}
    
        BenchmarkResult run(torch::Tensor& inp) {
            auto kernel_name = this->kernel_name;
            auto warmup = this->warmup;
            auto repeat = this->repeat;
        
            auto ref = torch::zeros_like(inp);
            auto sol = torch::zeros_like(inp);
            
            auto nrow = inp.size(0);
            auto ncol = inp.size(1);
            auto mem_size = ncol * nrow * sizeof(float);
        
            float *inp_cpu, *out_cpu;
            inp_cpu = (float*) malloc(mem_size);
            out_cpu = (float*) malloc(mem_size);
            memcpy(inp_cpu, inp.data_ptr<float>(), mem_size);
    
            auto t0_cpu = std::chrono::high_resolution_clock::now();
            for (size_t x = 0; x < repeat; x++) {
                softmax::kernel_cpu(out_cpu, inp_cpu, nrow, ncol);
            }
            auto t1_cpu = std::chrono::high_resolution_clock::now();
            float time_cpu_ms = std::chrono::duration<float, std::milli>(t1_cpu - t0_cpu).count();
            time_cpu_ms /= repeat;
        
            float *inp_gpu, *out_gpu;
            cudaMalloc(&inp_gpu, mem_size);
            cudaMalloc(&out_gpu, mem_size);
            cudaMemcpy(inp_gpu, inp_cpu, mem_size, cudaMemcpyHostToDevice);
        
            // lambda function to launch the kernel
            auto launch = [&](float* out_gpu, const float* inp_gpu) {
                void* args[] = {
                    (void*) &out_gpu,
                    (void*) &inp_gpu,
                    (void*) &nrow,
                    (void*) &ncol
                };
                cudaLaunchKernel(
                    (void*) this->kernel,
                    this->grid_dim,
                    this->block_dim,
                    args,
                    this->shared_mem_size,
                    0  // stream
                );
            };
    
            // Warmup
            for (size_t x = 0; x < warmup; x++) {
                launch(out_gpu, inp_gpu);
            }
            cudaDeviceSynchronize();
        
            // Timing the CUDA execution
            cudaEvent_t t0, t1;
            cudaEventCreate(&t0);
            cudaEventCreate(&t1);
        
            cudaEventRecord(t0);
            for (size_t x = 0; x < repeat; x++) {
                launch(out_gpu, inp_gpu);
            }
            cudaEventRecord(t1);
            cudaEventSynchronize(t1);
        
            float time_gpu_ms;
            cudaEventElapsedTime(&time_gpu_ms, t0, t1);
            time_gpu_ms /= repeat;
        
            memcpy(ref.data_ptr<float>(), out_cpu, mem_size);
            cudaMemcpy(sol.data_ptr<float>(), out_gpu, mem_size, cudaMemcpyDeviceToHost);
            auto err = torch::abs(ref - sol);
        
            // Cleanup
            free(inp_cpu);
            free(out_cpu);
            cudaFree(inp_gpu);
            cudaFree(out_gpu);
        
            auto result = BenchmarkResult{
                time_cpu_ms,
                time_gpu_ms,
                err,
                kernel_name
            };
            return result;
        }
};
