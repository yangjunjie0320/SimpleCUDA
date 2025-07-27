#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <torch/script.h>
#include <torch/types.h>
#include <cassert>

#define WARP_SIZE 32
#define INT4(x) (*(int4*)&(x))
#define FLOAT4(x) (*(float4*)&(x))
#define HALF2(x) (*(half2*)&(x))
#define LDST128BITS(x) (*(float4*)&(x))

#define BLOCK_SIZE 256
using TorchDtype = torch::Dtype;

// FP32
// ElementWise Add grid(N/256),
// block(256) a: Nx1, b: Nx1, c: Nx1, c = elementwise_add(a, b)
__global__ void elementwise_add_f32_1_kernel(const float* a, const float* b, float* c, int n) {
    // int idx = blockIdx.x * blockDim.x + threadIdx.x;
    auto thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    auto ix = thread_idx;
    unsigned int len = (unsigned int) n;

    const float* a_ptr = a;
    const float* b_ptr = b;
    float* c_ptr = c;

    if (ix < len) {
        float a_i = a_ptr[ix];
        float b_i = b_ptr[ix];
        float c_i = a_i + b_i;
        c_ptr[ix] = c_i;
    }
}

// ElementWise Add + Vec4
// grid(N/256), block(256/4)
// a: Nx1, b: Nx1, c: Nx1, c = elementwise_add(a, b)
__global__ void elementwise_add_f32_4_kernel(const float* a, const float* b, float* c, int n) {
    auto thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    auto ix = thread_idx;
    unsigned int len = (unsigned int) n / 4;

    const float4* a_ptr = (float4*) a;
    const float4* b_ptr = (float4*) b;
    float4* c_ptr = (float4*)c;

    if (ix < len) {
        float4 a4_i = a_ptr[ix];
        float4 b4_i = b_ptr[ix];
        float4 c4_i;
        c4_i.x = a4_i.x + b4_i.x;
        c4_i.y = a4_i.y + b4_i.y;
        c4_i.z = a4_i.z + b4_i.z;
        c4_i.w = a4_i.w + b4_i.w;
        c_ptr[ix] = c4_i;
    }
}

// FP16
// ElementWise Add grid(N/256),
// block(256) a: Nx1, b: Nx1, c: Nx1, c = elementwise_add(a, b)
__global__ void elementwise_add_f16_1_kernel(const half* a, const half* b, half* c, int n) {
    auto thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    auto ix = thread_idx;
    unsigned int len = (unsigned int) n;

    const half* a_ptr = a;
    const half* b_ptr = b;
    half* c_ptr = c;

    if (ix < len) {
        half a_i = a_ptr[ix];
        half b_i = b_ptr[ix];
        half c_i = __hadd(a_i, b_i);
        c_ptr[ix] = c_i;
    }
}

// a: Nx1, b: Nx1, c: Nx1, c = elementwise_add(a, b)
__global__ void elementwise_add_f16_2_kernel(const half* a, const half* b, half* c, int n) {
    auto thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    auto ix = thread_idx;
    unsigned int len = (unsigned int) n / 2;

    const half2* a_ptr = (half2*)a;
    const half2* b_ptr = (half2*)b;
    half2* c_ptr = (half2*)c;

    if (ix < len) {
        half2 a2_i = a_ptr[ix];
        half2 b2_i = b_ptr[ix];
        half2 c2_i;
        c2_i.x = __hadd(a2_i.x, b2_i.x);
        c2_i.y = __hadd(a2_i.y, b2_i.y);
        c_ptr[ix] = c2_i;
    }
}

__global__ void elementwise_add_f16_8_kernel(const half* a, const half* b, half* c, int n) {
    auto thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    auto ix = thread_idx * 4;
    unsigned int len = (unsigned int) (n / 2);

    auto i0 = ix + 0, i1 = ix + 1;
    auto i2 = ix + 2, i3 = ix + 3;

    const half2* a_ptr = (half2*)a;
    const half2* b_ptr = (half2*)b;
    half2* c_ptr = (half2*)c;

    half2 a2_i0 = a_ptr[i0];
    half2 a2_i1 = a_ptr[i1];
    half2 a2_i2 = a_ptr[i2];
    half2 a2_i3 = a_ptr[i3];

    half2 b2_i0 = b_ptr[i0];
    half2 b2_i1 = b_ptr[i1];
    half2 b2_i2 = b_ptr[i2];
    half2 b2_i3 = b_ptr[i3];

    half2 c2_i0, c2_i1, c2_i2, c2_i3;
    c2_i0.x = __hadd(a2_i0.x, b2_i0.x);
    c2_i0.y = __hadd(a2_i0.y, b2_i0.y);
    c2_i1.x = __hadd(a2_i1.x, b2_i1.x);
    c2_i1.y = __hadd(a2_i1.y, b2_i1.y);
    c2_i2.x = __hadd(a2_i2.x, b2_i2.x);
    c2_i2.y = __hadd(a2_i2.y, b2_i2.y);
    c2_i3.x = __hadd(a2_i3.x, b2_i3.x);
    c2_i3.y = __hadd(a2_i3.y, b2_i3.y);

    if (i0 < len) {
        c_ptr[i0] = c2_i0;
    }
    if (i1 < len) {
        c_ptr[i1] = c2_i1;
    }
    if (i2 < len) {
        c_ptr[i2] = c2_i2;
    }
    if (i3 < len) {
        c_ptr[i3] = c2_i3;
    }
}

template <typename element_t, TorchDtype torch_t, int batch_size,
          void (*kernel)(const element_t*, const element_t*, element_t*, int)>
void generate_elementwise_add(const torch::Tensor& a, const torch::Tensor& b,
                              const torch::Tensor& c) {
    const auto dim = a.dim();
    assert(dim == 2);

    const auto size0 = a.size(0);
    const auto size1 = a.size(1);
    const auto size = (int) (size0 * size1);

    assert(a.options().dtype() == torch_t);
    assert(a.size(0) == size0 && a.size(1) == size1);
    assert(b.options().dtype() == torch_t);
    assert(b.size(0) == size0 && b.size(1) == size1);
    assert(c.options().dtype() == torch_t);

    const auto block_size = BLOCK_SIZE / batch_size;
    const auto grid_size = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    const auto block = dim3(block_size);
    const auto grid = dim3(grid_size);

    const element_t* a_ptr = (const element_t*) a.data_ptr();
    const element_t* b_ptr = (const element_t*) b.data_ptr();
    element_t* c_ptr = (element_t*) c.data_ptr();

    kernel<<<grid, block>>>(a_ptr, b_ptr, c_ptr, size);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("elementwise_add_f32_1",
          &generate_elementwise_add<float, torch::kFloat32, 1, elementwise_add_f32_1_kernel>,
          "CUDA kernel for elementwise_add_f32_1");
    m.def("elementwise_add_f32_4",
          &generate_elementwise_add<float, torch::kFloat32, 4, elementwise_add_f32_4_kernel>,
          "CUDA kernel for elementwise_add_f32_4");
    m.def("elementwise_add_f16_1",
          &generate_elementwise_add<half, torch::kFloat16, 1, elementwise_add_f16_1_kernel>,
          "CUDA kernel for elementwise_add_f16_1");
    m.def("elementwise_add_f16_2",
          &generate_elementwise_add<half, torch::kFloat16, 2, elementwise_add_f16_2_kernel>,
          "CUDA kernel for elementwise_add_f16_2");
    m.def("elementwise_add_f16_8",
          &generate_elementwise_add<half, torch::kFloat16, 8, elementwise_add_f16_8_kernel>,
          "CUDA kernel for elementwise_add_f16_8");
}