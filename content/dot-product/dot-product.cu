#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <torch/script.h>
#include <torch/types.h>
#include <cassert>

#define WARP_SIZE 32
#define BLOCK_SIZE 256
using TorchDtype = torch::Dtype;

// FP32
// Dot Product grid(N/256),
// block(256) a: Nx1, b: Nx1, c: Nx1, c = dot_product(a, b)
__global__ void dot_product_f32_1_kernel_v1(const float* a, const float* b, float* c, int n) {
    // int idx = blockIdx.x * blockDim.x + threadIdx.x;
    auto thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    auto thread_num = gridDim.x * blockDim.x;
    auto step_size = thread_num;
    unsigned int len = (unsigned int) n;

    const float* a_ptr = a;
    const float* b_ptr = b;
    float c_ix = 0.0;

    for (int ix = thread_idx; ix < len; ix += step_size) {
        float a_i = a_ptr[ix];
        float b_i = b_ptr[ix];
        c_ix += a_i * b_i;
    }

    atomicAdd(c, c_ix);
}

template <typename element_t, TorchDtype torch_t, int batch_size,
          void (*kernel)(const element_t*, const element_t*, element_t*, int)>
void generate_dot_product(const torch::Tensor& a, const torch::Tensor& b, torch::Tensor& c) {
    const auto dim = a.dim();
    assert(dim == 1);

    const auto size0 = a.size(0);
    const auto size = (int) (size0);

    assert(a.options().dtype() == torch_t);
    assert(a.size(0) == size0);
    assert(b.options().dtype() == torch_t);
    assert(b.size(0) == size0);
    assert(c.options().dtype() == torch_t);
    assert(c.size(0) == 1);

    const auto block_size = BLOCK_SIZE / batch_size;
    const auto grid_size = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    const auto block = dim3(block_size);
    const auto grid = dim3(grid_size);

    const element_t* a_ptr = (const element_t*) a.data_ptr();
    const element_t* b_ptr = (const element_t*) b.data_ptr();

    c *= 0.0;
    element_t* c_ptr = (element_t*) c.data_ptr();

    kernel<<<grid, block>>>(a_ptr, b_ptr, c_ptr, size);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("dot_product_f32_1_v1",
          &generate_dot_product<float, torch::kFloat32, 1, dot_product_f32_1_kernel_v1>,
          "CUDA kernel for dot_product_f32_1");
}