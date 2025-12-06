#include <cuda_runtime.h>
#include <torch/torch.h>

#include <cassert>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <string>
#include <tuple>

#define WARP_SIZE 32
namespace softmax {
#include "softmax_v1.h"
#include "softmax_v2.h"
#include "softmax_v3.h"
#include "softmax_v4.h"

void kernel_cpu(float* out, const float* inp, const size_t nrow, const size_t ncol) {
    for (auto i = 0; i < nrow; i++) {
        const float* ai_ptr = inp + i * ncol;
        float* ci_ptr = out + i * ncol;

        float ai_max = -INFINITY;
        for (auto j = 0; j < ncol; j++) {
            float aij = *(ai_ptr + j);
            ai_max = fmaxf(ai_max, aij);
        }

        float ai_sum = 0.0;
        for (auto j = 0; j < ncol; j++) {
            float aij = *(ai_ptr + j);
            float exp_aij = expf(aij - ai_max);
            ai_sum += exp_aij;
            *(ci_ptr + j) = exp_aij;
        }

        for (auto j = 0; j < ncol; j++) {
            *(ci_ptr + j) /= ai_sum;
        }
    }
}
}  // namespace softmax

#include "utils.h"

int main() {
    for (int nrow = 256; nrow <= 4096; nrow *= 2) {
        const int ncol = WARP_SIZE * 4;
        auto inp = torch::randn({nrow, ncol});

        auto config = KernelLaunchConfig(softmax::kernel_v1, "softmax_f32_v1", 10, 100, dim3(1),
                                         dim3(nrow), 0);
        auto result = config.run(inp);
        result.print(true, true);

        config = KernelLaunchConfig(softmax::kernel_v2, "softmax_f32_v2", 10, 100, dim3(1),
                                    dim3(nrow), 0);
        result = config.run(inp);
        result.print(false, true);

        config = KernelLaunchConfig(softmax::kernel_v3, "softmax_f32_v3", 10, 100, dim3(WARP_SIZE),
                                    dim3(nrow), 0);
        result = config.run(inp);
        result.print(false, true);
    }

    return 0;
}