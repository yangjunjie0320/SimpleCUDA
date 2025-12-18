#include <cuda_runtime.h>
#include <cmath>

#include "utils.cu"

__global__ void kernel_v4(float* out, const float* inp, int nrow, int ncol) {
    // 每个 block 处理 blockDim.y 行
    // threadIdx.y 表示 block 内的行索引
    auto i = blockIdx.x * blockDim.y + threadIdx.y;
    if (i >= nrow)
        return;

    // threadIdx.x 是 warp 内的线程索引（0-31）
    auto t = threadIdx.x;
    auto stride = blockDim.x;  // 应该是 WARP_SIZE (32)

    const float* ai_ptr = inp + i * ncol;
    float* ci_ptr = out + i * ncol;

    // 每个线程处理部分列
    float ai_t_max = -INFINITY;
    for (auto j = t; j < ncol; j += stride) {
        ai_t_max = fmaxf(ai_t_max, ai_ptr[j]);
    }

    // Warp shuffle reduction - 使用 threadIdx.x，不是 t
    for (int offset = stride / 2; offset > 0; offset /= 2) {
        ai_t_max = fmaxf(ai_t_max, __shfl_down_sync(FULL, ai_t_max, offset));
    }
    float ai_max = __shfl_sync(FULL, ai_t_max, 0);

    float ai_t_sum = 0.0f;
    for (auto j = t; j < ncol; j += stride) {
        ai_t_sum += expf(ai_ptr[j] - ai_max);
    }

    for (int offset = stride / 2; offset > 0; offset /= 2) {
        ai_t_sum += __shfl_down_sync(FULL, ai_t_sum, offset);
    }
    float ai_sum = __shfl_sync(FULL, ai_t_sum, 0);

    for (auto j = t; j < ncol; j += stride) {
        float aij = ai_ptr[j];
        float cij = expf(aij - ai_max) / ai_sum;
        ci_ptr[j] = cij;
    }
}