import os, sys, time
pwd = os.path.dirname(os.path.abspath(__file__))

from functools import partial
from typing import Optional
from typing import Callable

import torch
torch.set_grad_enabled(False)
os.environ["TORCH_CUDA_ARCH_LIST"] = "%s.%s" % (torch.cuda.get_device_capability())

from torch.utils.cpp_extension import load
lib = load(
    name="lib",
    sources=[os.path.join(pwd, "elementwise-add.cu")],
    extra_cuda_cflags=[
        "-O3", "--use_fast_math",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_HALF2_OPERATORS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
    ],
    extra_cflags=["-std=c++17"],
)


def benchmark(
    f: Callable,
    a: torch.Tensor,
    b: torch.Tensor,
    num_warmup: int = 10,
    num_repeat: int = 1000,
):
    y = torch.zeros_like(a)
    for _ in range(num_warmup):
        f(a, b, y)

    torch.cuda.synchronize()
    t0 = time.time()

    for _ in range(num_repeat):
        f(a, b, y)
    
    torch.cuda.synchronize()
    t1 = time.time()
    dt = (t1 - t0) / num_repeat
    return y, dt

if __name__ == "__main__":
    dimension_list = [(1024, 1024), (2048, 2048), (4096, 4096)]

    for d1, d2 in dimension_list:
        print("-" * 64)
        print(f"d1 = {d1}, d2 = {d2}")
        
        a = torch.randn((d1, d2)).cuda().float().contiguous()
        b = torch.randn((d1, d2)).cuda().float().contiguous()

        res = []
        
        f = lambda a, b, y: torch.add(a, b, out=y)
        ref, t = benchmark(f, a, b)
        res.append(["torch.add (f32)", t, 0.0])

        f = lib.elementwise_add_f32_1
        sol, t = benchmark(f, a, b)
        res.append(["elementwise_add_f32_1", t, abs(ref - sol).max()])

        f = lib.elementwise_add_f32_4
        sol, t = benchmark(f, a, b)
        res.append(["elementwise_add_f32_4", t, abs(ref - sol).max()])

        a = torch.randn((d1, d2)).cuda().half().contiguous()
        b = torch.randn((d1, d2)).cuda().half().contiguous()

        f = lambda a, b, y: torch.add(a, b, out=y)
        ref, t = benchmark(f, a, b)
        res.append(["torch.add (f16)", t, 0.0])

        f = lib.elementwise_add_f16_1
        sol, t = benchmark(f, a, b)
        res.append(["elementwise_add_f16_1", t, abs(ref - sol).max()])
        
        f = lib.elementwise_add_f16_2
        sol, t = benchmark(f, a, b)
        res.append(["elementwise_add_f16_2", t, abs(ref - sol).max()])

        f = lib.elementwise_add_f16_8
        sol, t = benchmark(f, a, b)
        res.append(["elementwise_add_f16_8", t, abs(ref - sol).max()])

        for n, t, e in res:
            print(f"name: {n:>32}, time: {t: 10.2e} s, error: {e: 10.2e}")
