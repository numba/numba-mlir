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
from datetime import datetime, timedelta
from timeit import default_timer as timer

BASE_PATH = os.path.join(os.getcwd(), ".asv")


def get_results_dir():
    return os.environ.get("NUMBA_MLIR_BENCH_RUNNER_RESULTS_DIR", BASE_PATH)


def get_config_file():
    new_results_dir = get_results_dir()
    if BASE_PATH == new_results_dir:
        return None

    orig_file = os.path.join(os.getcwd(), "asv.conf.json")
    with open(orig_file) as file:
        file_contents = file.read()

    import json5

    config = json5.loads(file_contents)

    config["results_dir"] = os.path.join(new_results_dir, "results")
    config["repo"] = os.path.join("..", "..")
    config["benchmark_dir"] = os.path.join("..", "benchmarks")

    new_filename = os.path.join(BASE_PATH, "temp.asv.conf.json")
    try:
        os.remove(new_filename)
    except OSError:
        pass

    with open(new_filename, "w") as outfile:
        outfile.write(str(config))

    return new_filename


def get_machine_name():
    from asv.machine import Machine

    return os.environ.get(
        "NUMBA_MLIR_BENCH_RUNNER_MACHINE", Machine.get_unique_machine_name()
    )


def wrap_args(args):
    machine = get_machine_name()
    config = get_config_file()

    args = args + ["--machine", str(machine)]
    if config:
        args = args + ["--config", str(config)]

    return args


def asv_run(args):
    subprocess.check_call(wrap_args(["python", "-m", "asv", "run"]) + args)


def asv_show(args):
    subprocess.check_call(wrap_args(["python", "-m", "asv", "show"]) + args)


def asv_machine(args):
    subprocess.check_call(["python", "-m", "asv", "machine"] + args)


def add_bench_arg(args, bench):
    if bench is not None:
        return args + [f"--bench={bench}"]

    return args


def get_head_hash():
    return (
        subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
        .decode("utf-8")
        .strip()
    )


def load_results(commit, machine):
    res_path = os.path.join(get_results_dir(), "results")

    pattern = os.path.join(res_path, machine, f"{commit}-existing*.json")
    files = glob.glob(pattern)
    assert len(files) == 1, files

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

        value = human_value(value, "seconds", err).replace("n/a", "skipped")

        res[bench][frameworks[framework]] = value

    csv_str = f"bench\\framework," + ",".join(frameworks.keys()) + "\n"
    for name, val in res.items():
        csv_str += name + "," + ",".join(val) + "\n"

    return csv_str


def ensure_dir(dir_path):
    try:
        os.makedirs(dir_path)
    except FileExistsError:
        pass


def sanitize_filename(name):
    chars = ":#. "
    return name.translate(str.maketrans(chars, "_" * len(chars)))


def save_report(data, commit, machine, reports_dir):
    ensure_dir(reports_dir)
    file_name = sanitize_filename(f"{commit}_{machine}_{str(datetime.now())}") + ".csv"
    file_path = os.path.join(reports_dir, file_name)

    with open(file_path, "wb") as file:
        file.write(data.encode("utf-8"))


def get_bench_arg(params):
    if len(params) > 0:
        return params[0]

    return None


def run_test(params):
    bench = get_bench_arg(params)
    os.environ["NUMBA_MLIR_BENCH_PRESETS"] = "S"
    os.environ["NUMBA_MLIR_BENCH_VALIDATE"] = "1"
    asv_run(
        add_bench_arg(["--python=same", "--quick", "--show-stderr", "--dry-run"], bench)
    )


def run_bench(params):
    bench = get_bench_arg(params)
    os.environ["NUMBA_MLIR_BENCH_PRESETS"] = "S,M,paper"
    os.environ["NUMBA_MLIR_BENCH_VALIDATE"] = "0"
    commit = get_head_hash()
    try:
        asv_run(
            add_bench_arg(
                [
                    "--environment=existing:python",
                    "--show-stderr",
                    f"--set-commit-hash={commit}",
                ],
                bench,
            )
        )
        asv_show([commit])
    except:
        pass

    machine = get_machine_name()
    results = load_results(commit, machine)
    results = convert_results(results)
    results = results_to_csv(results)
    print("csv report:")
    print(results)

    reports_dir = os.path.join(get_results_dir(), "csv_reports")
    save_report(results, commit, machine, reports_dir)


def setup_machine(params):
    import cpuinfo

    machine = get_machine_name()
    info = cpuinfo.get_cpu_info()
    arch = info.get("arch")
    cpu = info.get("brand_raw")
    num_cpu = info.get("count")
    args = [
        "--machine",
        str(machine),
        "--arch",
        str(arch),
        "--cpu",
        str(cpu),
        "--num_cpu",
        str(num_cpu),
    ]
    asv_machine(["--yes"])
    asv_machine(args)


def run_cmd(cmd, params):
    cmds = [
        ("test", run_test),
        ("bench", run_bench),
        ("machine", setup_machine),
    ]

    for n, c in cmds:
        if n == cmd:
            return c(params)

    assert False, f"Invalid cmd: {cmd}"


if __name__ == "__main__":
    start = timer()
    try:
        args = sys.argv[1:]
        cmd = args[0]
        run_cmd(cmd, args[1:])
    finally:
        end = timer()
        print(f"Execution took: {timedelta(seconds=end-start)}")
