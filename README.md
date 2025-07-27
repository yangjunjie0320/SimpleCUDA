# SimpleCUDA

A CUDA learning project demonstrating efficient GPU kernel implementations with PyTorch integration.

## Overview

This project provides hands-on examples of CUDA programming, comparing different implementation approaches:

1. PyTorch Internal Functions (when possible).
2. Custom PyTorch Functions.
3. Custom CUDA Kernels.

## Installation

1. Clone the repository:
```bash
git clone git@github.com:yangjunjie0320/SimpleCUDA.git
cd SimpleCUDA
```

2. Create and activate the conda environment:
```bash
conda env create -f environment.yml -n simple-cuda
conda activate simple-cuda
```

In order to help the compiler to find correct `libtorch` headers,
```bash
export CUDA_INCLUDE_DIRS=$CONDA_PREFIX/include/cuda/
export CMAKE_PREFIX_PATH=$CONDA_PREFIX/include/

export TORCH_HEADER_PATH=$(python -c "import torch; print(torch.__path__[0])")
export CMAKE_PREFIX_PATH=$CMAKE_PREFIX_PATH:$TORCH_HEADER_PATH
```

3. Verify `pytorch` installation:
```bash
which python
python -c "import torch; print(torch.__version__)"
python -c "import torch; print(torch.cuda.is_available())"
python -c "import torch; print(torch.cuda.get_device_name(0))"
```

## Usage

Run the `elementwise-add` example:
```bash
python content/elementwise-add/elementwise-add.py
```