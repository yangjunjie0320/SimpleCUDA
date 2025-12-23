# SimpleCUDA

[![Format Check](https://github.com/yangjunjie0320/SimpleCUDA/actions/workflows/format-check.yml/badge.svg)](https://github.com/yangjunjie0320/SimpleCUDA/actions/workflows/format-check.yml)
[![Build](https://github.com/yangjunjie0320/SimpleCUDA/actions/workflows/build.yml/badge.svg)](https://github.com/yangjunjie0320/SimpleCUDA/actions/workflows/build.yml)

A CUDA learning project demonstrating efficient GPU kernel implementations.


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

If you are using WSL2, you can fetch the `nvidia-smi` command from:
```bash
alias "nvidia-smi"="/usr/lib/wsl/lib/nvidia-smi"
```

Check the CUDA architecture:
```bash
nvidia-smi --query-gpu=name,compute_cap --format=csv
```

Write the CUDA architecture to the environment variable `CUDAARCHS`:
```bash
conda activate simple-cuda; export ENABLE_DEBUG=OFF; export INDEX=0
export CUDAARCHS=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader --id=$INDEX | tr -d '.')
```

3. Build the project:
```bash
rm -rf build
cmake -B build -DENABLE_DEBUG=$ENABLE_DEBUG
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
ncu --set detailed \
  --launch-skip 600 --launch-count 1 \
  --kernel-name kernel_v5 ./build/softmax.x
```
