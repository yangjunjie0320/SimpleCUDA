# SimpleCUDA

[![Format Check](https://github.com/yangjunjie0320/SimpleCUDA/actions/workflows/format-check.yml/badge.svg)](https://github.com/yangjunjie0320/SimpleCUDA/actions/workflows/format-check.yml)
[![Build](https://github.com/yangjunjie0320/SimpleCUDA/actions/workflows/build.yml/badge.svg)](https://github.com/yangjunjie0320/SimpleCUDA/actions/workflows/build.yml)

A CUDA learning project demonstrating efficient GPU kernel implementations with PyTorch integration.


## Overview

This project provides hands-on examples of CUDA programming.

## Installation

1. Clone the repository:
```bash
git clone git@github.com:yangjunjie0320/SimpleCUDA.git
cd SimpleCUDA
```

2. Install dependencies (CUDA, xtensor) with conda:
```bash
conda env create -f environment.yml -n simple-cuda
conda activate simple-cuda
```

3. Build the project:
```bash
conda activate simple-cuda; rm -rf build
cmake -B build -DCMAKE_CUDA_ARCHITECTURES=89 -DENABLE_DEBUG=ON
cmake --build build
```

## Format Check
```bash
clang-format --style=file ./content/softmax/softmax_v4.h
```

## Run
Run the `softmax` example:
```bash
./build/softmax.x
```

Use `ncu` to profile the CUDA kernel:
```bash
ncu \
  --kernel-name kernel_v4 \
  --launch-skip 500 \
  --launch-count 1 \
  --target-processes all \
  ./build/softmax.x > profile.log
```
