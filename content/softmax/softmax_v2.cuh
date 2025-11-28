#include <cuda_runtime.h>
#include <cmath>

__global__ void softmax_v2_kernel(float* out, const float* inp, size_t ncol, size_t nrow) {
    int row_idx = blockIdx.x;
    int tid = threadIdx.x;

    if (row_idx >= nrow) return;

    const float* row_input = inp + row_idx * ncol;
    float* row_output = out + row_idx * ncol;

    extern __shared__ float shared_mem[];

    // Phase 1: Parallel max reduction
    float thread_max = -INFINITY;
    for (int i = tid; i < ncol; i += blockDim.x) {
        thread_max = fmaxf(thread_max, row_input[i]);
    }
    shared_mem[tid] = thread_max;
    __syncthreads();

    // Tree reduction for max
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            shared_mem[tid] = fmaxf(shared_mem[tid], shared_mem[tid + stride]);
        }
        __syncthreads();
    }
    float max_val = shared_mem[0];
    __syncthreads();

    // Phase 2: Parallel sum reduction
    float thread_sum = 0.0f;
    for (int i = tid; i < ncol; i += blockDim.x) {
        thread_sum += expf(row_input[i] - max_val);
    }
    shared_mem[tid] = thread_sum;
    __syncthreads();

    // Tree reduction for sum
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            shared_mem[tid] += shared_mem[tid + stride];
        }
        __syncthreads();
    }
    float sum_val = shared_mem[0];
    __syncthreads();

    // Phase 3: Parallel normalization
    for (int i = tid; i < ncol; i += blockDim.x) {
        row_output[i] = expf(row_input[i] - max_val) / sum_val;
    }
}

#define WARP_SIZE 32

void softmax_f32_v2(float* out, const float* inp, size_t ncol, size_t nrow) {
    int block_size = WARP_SIZE;
    int shared_mem_size = block_size * sizeof(float);
    softmax_v2_kernel<<<nrow, block_size, shared_mem_size>>>(out, inp, ncol, nrow);
}