# SimpleCUDA

A CUDA learning project demonstrating efficient GPU kernel implementations with PyTorch integration.

## Overview

This project provides hands-on examples of CUDA programming, comparing different implementation approaches:

1. **PyTorch Implementation** - Pure PyTorch solutions using built-in operations
2. **PyTorch Internal Functions** - Understanding how PyTorch implements operations internally  
3. **Custom CUDA Kernels** - Hand-written CUDA kernels with PyTorch Python bindings

Each implementation includes comprehensive tests to ensure correctness and performance comparisons.

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd SimpleCUDA
```

2. Create and activate the conda environment:
```bash
conda env create -f environment.yml -n simple-cuda
conda activate simple-cuda
```

In order to help the compiler to find correct `libtorch` headers, you need to set the `CMAKE_PREFIX_PATH` environment variable to the path of the `libtorch` installation.
```bash
export CUDA_INCLUDE_DIRS=$CONDA_PREFIX/include/cuda/
export CMAKE_PREFIX_PATH=$CONDA_PREFIX/include/
export CMAKE_PREFIX_PATH=$CMAKE_PREFIX_PATH:$(python -c "import torch; print(torch.__path__[0])")
```

Run `cmake` to generate the `compile_commands.json` file:
```bash
cmake -B build -S .
```

3. Verify `pytorch` installation:
```bash
which python
python -c "import torch; print(torch.__version__)"
python -c "import torch; print(torch.cuda.is_available())"
python -c "import torch; print(torch.cuda.get_device_name(0))"
```

## Development Setup

This project uses Google C++ style guidelines. Code formatting is handled by `.clang-format`:

```bash
# Format all C++/CUDA files
find . -name "*.cu" -o -name "*.cpp" -o -name "*.h" | xargs clang-format -i
```
