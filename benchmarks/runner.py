# SPDX-FileCopyrightText: 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


import os
import sys
import subprocess


def run_asv(args):
    subprocess.check_call(["python", "-m", "asv", "run"] + args)


def run_test():
    os.environ["NUMBA_MLIR_BENCH_PRESETS"] = "S"
    os.environ["NUMBA_MLIR_BENCH_VALIDATE"] = "1"
    run_asv(["--python=same", "--quick", "--show-stderr", "--dry-run"])


def run_bench():
    os.environ["NUMBA_MLIR_BENCH_PRESETS"] = "S,M,paper"
    os.environ["NUMBA_MLIR_BENCH_VALIDATE"] = "0"
    run_asv(["--python=same", "--show-stderr", "--dry-run"])


def run_cmd(params):
    cmds = [
        ("test", run_test),
        ("bench", run_bench),
    ]

    for name, cmd in cmds:
        if name in params:
            return cmd()

    assert False, f"Invalid params: {params}"


if __name__ == "__main__":
    import numba_mlir

    args = set(sys.argv[1:])
    run_cmd(args)
