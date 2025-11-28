# SimpleCUDA

A CUDA learning project demonstrating efficient GPU kernel implementations with PyTorch integration.

## Overview

This project provides hands-on examples of CUDA programming.

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

Verify `pytorch` installation:
```bash
which python
python -c "import torch; print(torch.__version__)"
python -c "import torch; print(torch.cuda.is_available())"
python -c "import torch; print(torch.cuda.get_device_name(0))"
```

3. In order to help the compiler to find correct `libtorch` headers,
```bash
export CUDA_HOME=$CONDA_PREFIX
export CUDA_INCLUDE_DIRS=$CUDA_HOME/include/
export CMAKE_PREFIX_PATH=$CONDA_PREFIX/include/

export TORCH_HEADER_PATH=$(python -c "import torch; print(torch.__path__[0])")
export CMAKE_PREFIX_PATH=$CMAKE_PREFIX_PATH:$TORCH_HEADER_PATH

export TORCH_CUDA_ARCH_LIST=$(python -c "import torch; p=torch.cuda.get_device_properties(0); print(f'{p.major}.{p.minor}')")
```

Build the project:
```bash
rm -rf build
cmake -B build \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
    -DCUDA_TOOLKIT_ROOT_DIR=$CONDA_PREFIX/targets/x86_64-linux \
    -DTORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST
cmake --build build
```

## Usage

Run the `softmax` example:
```bash
python content/softmax/run.py
```