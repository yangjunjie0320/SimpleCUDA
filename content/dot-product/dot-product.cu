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
__global__ void dot_product_f32_1_kernel(const float* a, const float* b, float& c, int n) {
    // int idx = blockIdx.x * blockDim.x + threadIdx.x;
    auto thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    auto ix = thread_idx;
    unsigned int len = (unsigned int) n;

    const float* a_ptr = a;
    const float* b_ptr = b;

    if (ix < len) {
        float a_i = a_ptr[ix];
        float b_i = b_ptr[ix];
        c += a_i * b_i;
    }
}

// Dot Product + Vec4
// grid(N/256), block(256/4)
// a: Nx1, b: Nx1, c: Nx1, c = dot_product(a, b)
__global__ void dot_product_f32_4_kernel(const float* a, const float* b, float& c, int n) {
    auto thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    auto ix = thread_idx;
    unsigned int len = (unsigned int) n / 4;

    const float4* a_ptr = (float4*) a;
    const float4* b_ptr = (float4*) b;

    if (ix < len) {
        float4 a4_i = a_ptr[ix];
        float4 b4_i = b_ptr[ix];
        float cx = a4_i.x * b4_i.x;
        float cy = a4_i.y * b4_i.y;
        float cz = a4_i.z * b4_i.z;
        float cw = a4_i.w * b4_i.w;
        c += cx + cy + cz + cw;
    }
}

// FP16
// Dot Product grid(N/256),
// block(256) a: Nx1, b: Nx1, c: Nx1, c = dot_product(a, b)
__global__ void dot_product_f16_1_kernel(const half* a, const half* b, half& c, int n) {
    auto thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    auto ix = thread_idx;
    unsigned int len = (unsigned int) n;

    const half* a_ptr = a;
    const half* b_ptr = b;

    if (ix < len) {
        half a_i = a_ptr[ix];
        half b_i = b_ptr[ix];
        c += __hmul(a_i, b_i);
    }
}

// a: Nx1, b: Nx1, c: Nx1, c = dot_product(a, b)
__global__ void dot_product_f16_2_kernel(const half* a, const half* b, half& c, int n) {
    auto thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    auto ix = thread_idx;
    unsigned int len = (unsigned int) n / 2;

    const half2* a_ptr = (half2*)a;
    const half2* b_ptr = (half2*)b;

    if (ix < len) {
        half2 a2_i = a_ptr[ix];
        half2 b2_i = b_ptr[ix];
        half cx = __hmul(a2_i.x, b2_i.x);
        half cy = __hmul(a2_i.y, b2_i.y);
        c += cx + cy;
    }
}

template <typename element_t, TorchDtype torch_t, int batch_size,
          void (*kernel)(const element_t*, const element_t*, element_t&, int)>
void generate_dot_product(const torch::Tensor& a, const torch::Tensor& b, element_t& c) {
    const auto dim = a.dim();
    assert(dim == 1);

    const auto size0 = a.size(0);
    const auto size = (int) (size0);

    assert(a.options().dtype() == torch_t);
    assert(a.size(0) == size0);
    assert(b.options().dtype() == torch_t);
    assert(b.size(0) == size0);
    // assert(c.options().dtype() == torch_t);

    const auto block_size = BLOCK_SIZE / batch_size;
    const auto grid_size = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    const auto block = dim3(block_size);
    const auto grid = dim3(grid_size);

    const element_t* a_ptr = (const element_t*) a.data_ptr();
    const element_t* b_ptr = (const element_t*) b.data_ptr();

    kernel<<<grid, block>>>(a_ptr, b_ptr, c, size);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("dot_product_f32_1",
          &generate_dot_product<float, torch::kFloat32, 1, dot_product_f32_1_kernel>,
          "CUDA kernel for dot_product_f32_1");
    m.def("dot_product_f32_4",
          &generate_dot_product<float, torch::kFloat32, 4, dot_product_f32_4_kernel>,
          "CUDA kernel for dot_product_f32_4");
    m.def("dot_product_f16_1",
          &generate_dot_product<half, torch::kFloat16, 1, dot_product_f16_1_kernel>,
          "CUDA kernel for dot_product_f16_1");
    m.def("dot_product_f16_2",
          &generate_dot_product<half, torch::kFloat16, 2, dot_product_f16_2_kernel>,
          "CUDA kernel for dot_product_f16_2");
}