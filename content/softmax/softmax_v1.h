__global__ void kernel_v1(float* out, const float* inp, size_t nrow, size_t ncol) {
    auto i = blockIdx.x;
    if (i >= nrow) return;

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
