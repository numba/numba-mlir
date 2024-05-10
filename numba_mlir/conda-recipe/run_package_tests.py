# SPDX-FileCopyrightText: 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import multiprocessing
import os
import pytest
import sys


def run_tests(params):
    np = multiprocessing.cpu_count()
    args = [
        "--capture=tee-sys",
        "-rXF",
        "-p",
        "no:faulthandler",
        "--pyargs",
        "numba_mlir.mlir.tests",
    ]

    args += [f"-n{np}"]

    if "smoke" in params:
        args += ["-m", "smoke"]

    if "gpu" in params:
        os.environ["NUMBA_MLIR_ENABLE_GPU_TESTS"] = "1"
        args += ["-m", "test_gpu"]

    if "verbose" in params:
        args += ["-vv"]

    if "replace_parfors" not in params:
        args += ["-k", "not replace_parfors"]
    else:
        args += ["-k", "replace_parfors"]

    print(f"INFO: nproc {np}")

    os.environ["NUMBA_DISABLE_PERFORMANCE_WARNINGS"] = "1"
    return pytest.main(args)


if __name__ == "__main__":
    import numba_mlir

    args = set(sys.argv[1:])
    retcode = run_tests(args)
    sys.exit(retcode)
