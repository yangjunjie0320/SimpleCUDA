#include <cuda_runtime.h>
#include <cmath>

__global__ void softmax(float* out, const float* inp, size_t ncol, size_t nrow) {
    auto i = blockIdx.x;
    if (i >= nrow) return;

    auto ai_ptr = inp + i * ncol;
    auto ci_ptr = out + i * ncol;

    float ai_max = -INFINITY;
    for (auto j = 0; j < ncol; j++) {
        float aij = *(ai_ptr + j);
        ai_max = max(ai_max, aij);
    }

    float norm = 0.0;
    for (auto j = 0; j < ncol; j++) {
        float aij = *(ai_ptr + j);
        norm += exp(aij - ai_max);
    }

    for (auto j = 0; j < ncol; j++) {
        float aij = *(ai_ptr + j);
        float cij = exp(aij - ai_max) / norm;
        *(ci_ptr + j) = cij;
    }
}

void softmax_f32_v1(float* out, const float* inp, size_t ncol, size_t nrow) {
    softmax<<<nrow, 1>>>(out, inp, ncol, nrow);
}