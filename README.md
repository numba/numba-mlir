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
from numba_mlir import njit
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

Building and testing Python package
```Bash
cd numba_mlir
conda create -n test-env python=3.11 --file ../scripts/numba-mlir.env -c conda-forge
conda activate test-env
conda install dpcpp_linux-64=2024.2 --file ../scripts/mkl.env -c https://software.repos.intel.com/python/conda/
export LLVM_PATH=<...>/llvm-install
export NUMBA_MLIR_USE_SYCL=ON # Optional
python setup.py develop
pytest -n8 --capture=tee-sys -rXF
```

### Windows

TBD

## Using GPU offload

* Install Intel GPU drivers: https://dgpu-docs.intel.com/installation-guides/index.html
* Install dpctl `conda install dpctl -c dppy/label/dev -c https://software.repos.intel.com/python/conda/`

Kernel offload example:
```Python
from numba_mlir.kernel import kernel, get_global_id, DEFAULT_LOCAL_SIZE
import numpy as np
import dpctl.tensor as dpt

@kernel
def foo(a, b, c):
    i = get_global_id(0)
    j = get_global_id(1)
    c[i, j] = a[i, j] + b[i, j]

a = np.array([[1,2,3],[4,5,6]])
b = np.array([[7,8,9],[-1,-2,-3]])

print(a + b)

device = "gpu"
a = dpt.asarray(a, device=device)
b = dpt.asarray(b, device=device)
c = dpt.empty(a.shape, dtype=a.dtype, device=device)

foo[a.shape, DEFAULT_LOCAL_SIZE] (a, b, c)

result = dpt.asnumpy(c)
print(result)
```

Numpy offload example:
```Python
from numba_mlir import njit
import numpy as np
import dpctl.tensor as dpt

@njit(parallel=True)
def foo(a, b):
    return a + b

a = np.array([[1,2,3],[4,5,6]])
b = np.array([[1,2,3]])

print(a + b)

a = dpt.asarray(a, device="gpu")
b = dpt.asarray(b, device="gpu")

result = foo(a, b)
print(result)
```


## Contributing

We are using github issues to report issues and github pull requests for development.

[Code of Conduct](https://github.com/numba/numba-governance/blob/accepted/code-of-conduct.md)
