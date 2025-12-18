#include <cuda_runtime.h>
#include <cmath>

#include "utils.cu"

__global__ void kernel_v3(float* out, const float* inp, int nrow, int ncol) {
    auto i = blockIdx.x;
    if (i >= nrow)
        return;

    auto t = threadIdx.x;
    auto stride = blockDim.x;

    const float* ai_ptr = inp + i * ncol;
    float* ci_ptr = out + i * ncol;

    float ai_t_max = -INFINITY;
    for (auto j = t; j < ncol; j += stride) {
        ai_t_max = fmaxf(ai_t_max, ai_ptr[j]);
    }

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
