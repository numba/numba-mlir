# SPDX-FileCopyrightText: 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


import os
import sys
import subprocess


def run_asv(args):
    os.environ["NUMBA_MLIR_BENCH_PRESETS"] = "S"
    os.environ["NUMBA_MLIR_BENCH_VALIDATE"] = "1"
    subprocess.check_call(["python", "-m", "asv", "run"] + args)


def run_test():
    run_asv(["--python=same", "--quick", "--show-stderr", "--dry-run"])


def run_bench(params):
    if "test" in params:
        return run_test()


if __name__ == "__main__":
    import numba_mlir

    args = set(sys.argv[1:])
    run_bench(args)
