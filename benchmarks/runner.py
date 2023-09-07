# SPDX-FileCopyrightText: 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


import os
import sys
import subprocess


def asv_run(args):
    subprocess.check_call(["python", "-m", "asv", "run"] + args)


def asv_show(args):
    subprocess.check_call(["python", "-m", "asv", "show"] + args)


def get_head_hash():
    return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()


def run_test(params):
    os.environ["NUMBA_MLIR_BENCH_PRESETS"] = "S"
    os.environ["NUMBA_MLIR_BENCH_VALIDATE"] = "1"
    asv_run(["--python=same", "--quick", "--show-stderr", "--dry-run"])


def run_bench(params):
    os.environ["NUMBA_MLIR_BENCH_PRESETS"] = "S,M,paper"
    os.environ["NUMBA_MLIR_BENCH_VALIDATE"] = "0"
    commit = get_head_hash()
    asv_run(
        [
            "--environment=existing:python",
            "--show-stderr",
            f"--set-commit-hash={commit}",
        ]
    )
    asv_show([commit])


def run_cmd(cmd, params):
    cmds = [
        ("test", run_test),
        ("bench", run_bench),
    ]

    for n, c in cmds:
        if n == cmd:
            return c(params)

    assert False, f"Invalid cmd: {cmd}"


if __name__ == "__main__":
    args = sys.argv[1:]
    cmd = args[0]
    run_cmd(cmd, args[1:])
