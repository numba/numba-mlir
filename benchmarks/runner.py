# SPDX-FileCopyrightText: 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


import os
import sys
import subprocess
import glob
import json
from math import isnan
import itertools


def asv_run(args):
    subprocess.check_call(["python", "-m", "asv", "run"] + args)


def asv_show(args):
    subprocess.check_call(["python", "-m", "asv", "show"] + args)


def get_head_hash():
    return (
        subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
        .decode("utf-8")
        .strip()
    )


def load_results(commit):
    res_path = os.path.join(os.getcwd(), ".asv", "results")
    machine_dirs = list(
        filter(lambda a: os.path.isdir(os.path.join(res_path, a)), os.listdir(res_path))
    )
    assert len(machine_dirs) == 1

    pattern = os.path.join(res_path, machine_dirs[0], f"{commit}-existing*.json")
    files = glob.glob(pattern)
    assert len(files) == 1

    with open(files[0]) as file:
        file_contents = file.read()

    return json.loads(file_contents)


def convert_results(raw_results):
    from asv_runner.statistics import get_err

    result_columns = raw_results["result_columns"]

    ret = []
    for name, val in raw_results["results"].items():
        res = {k: v for k, v in zip(result_columns, val)}

        parts = name.split(".")
        framework = parts[-3]
        bench = ".".join(parts[:-3])

        params = list(itertools.product(*res["params"]))
        result = res["result"]

        empty = [None] * len(params)
        q25stats = res.get("stats_q_25", empty)
        q75stats = res.get("stats_q_75", empty)

        if result is None:
            result = empty

        for r, q25, q75, p in zip(result, q25stats, q75stats, params):
            full_bench = bench + str(list(p)).replace("'", "").replace(",", ";")
            if r is None or (isinstance(r, float) and isnan(r)):
                err = None
            else:
                err = get_err(r, {"q_25": q25, "q_75": q75})
            ret.append((full_bench, framework, r, err))

    return ret


def results_to_csv(results):
    from asv.util import human_value

    frameworks = {}
    for bench, framework, value, err in results:
        if framework not in frameworks:
            c = len(frameworks)
            frameworks[framework] = c

    count = len(frameworks)
    res = {}
    for bench, framework, value, err in results:
        if bench not in res:
            res[bench] = [""] * count

        value = human_value(value, "seconds", err)

        res[bench][frameworks[framework]] = value

    csv_str = f"bench\\framework," + ",".join(frameworks.keys()) + "\n"
    for name, val in res.items():
        csv_str += name + "," + ",".join(val) + "\n"

    return csv_str


def run_test(params):
    os.environ["NUMBA_MLIR_BENCH_PRESETS"] = "S"
    os.environ["NUMBA_MLIR_BENCH_VALIDATE"] = "1"
    asv_run(["--python=same", "--quick", "--show-stderr", "--dry-run"])


def run_bench(params):
    os.environ["NUMBA_MLIR_BENCH_PRESETS"] = "S,M,paper"
    os.environ["NUMBA_MLIR_BENCH_VALIDATE"] = "0"
    commit = get_head_hash()
    try:
        asv_run(
            [
                "--environment=existing:python",
                "--show-stderr",
                f"--set-commit-hash={commit}",
            ]
        )
        asv_show([commit])
    except:
        pass

    results = load_results(commit)
    results = convert_results(results)
    results = results_to_csv(results)
    print()
    print(results)


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
