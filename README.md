<!--
SPDX-FileCopyrightText: 2022 Intel Corporation

SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

# MLIR-based numba backend

The goal of this project is to provide efficient code generation for CPUs and GPUs
using Multi-Level Intermediate Representation (MLIR) infrastructure.
It uses Numba infrastructure as a frontend but have completely separate codepaths
through MLIR infrastructure for low level code generation.

Package provides set of decorators similar to Numba decorators to compile python code.

Example:
```Python
from numba_dpcomp import njit
import numpy as np

@njit
def foo(a, b):
    return a + b

result = foo(np.array([1,2,3]), np.array([4,5,6]))
print(result)
```

## Building and testing

You will need LLVM built from specific commit, found in `llvm-sha.txt`.

### Linux

Building llvm
```Bash
git clone https://github.com/llvm/llvm-project.git
cd llvm-project
git checkout $SHA
cd ..
mkdir llvm-build
cd llvm-build
cmake ../llvm-project/llvm -GNinja -DCMAKE_BUILD_TYPE=Release -DLLVM_ENABLE_PROJECTS=mlir -DLLVM_ENABLE_ASSERTIONS=ON -DLLVM_ENABLE_RTTI=ON -DLLVM_USE_LINKER=gold -DLLVM_INSTALL_UTILS=ON -DCMAKE_INSTALL_PREFIX=../llvm-install
ninja install
```

Getting TBB
```Bash
wget -O tbb.tgz "https://github.com/oneapi-src/oneTBB/releases/download/v2021.6.0/oneapi-tbb-2021.6.0-lin.tgz"
mkdir tbb
tar -xf "tbb.tgz" -C tbb --strip-components=1
```

Getting Level Zero loader (Optional, needed for Intel GPU support)
```Bash
git clone https://github.com/oneapi-src/level-zero.git
cd level-zero
git checkout v1.6.2
cd ..
mkdir level-zero-build
cmake ../level-zero -GNinja -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=../level-zero-install
ninja install
```

Building and testing Python package
```Bash
cd numba_dpcomp
conda create -n test-env python=3.9 numba=0.56 numpy=1.22 "setuptools<65.6" scikit-learn pytest-xdist ninja scipy pybind11 pytest lit tbb=2021.6.0 cmake "mkl-devel-dpcpp>=2022.2" -c conda-forge -c intel
conda activate test-env
export TBB_PATH=<...>/tbb
export LLVM_PATH=<...>/llvm-install
export LEVEL_ZERO_DIR=<...>/level-zero-install # Optional
export LEVEL_ZERO_VERSION_CHECK_OFF=1 # Optional
python setup.py develop
pytest -n8 --capture=tee-sys -rXF
```

### Windows

TBD


## Contributing

We are using github issues to report issues and github pull requests for development.

[Code of Conduct](https://github.com/numba/numba-governance/blob/accepted/code-of-conduct.md)
