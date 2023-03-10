# SPDX-FileCopyrightText: 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os
import subprocess
import re
import argparse
from itertools import takewhile, dropwhile


def run_command(args):
    res = subprocess.run(args, check=True, capture_output=True, text=True)
    return str(res.stdout) + str(res.stderr)


def extract_versions(text):
    lines = text.splitlines()

    # skip until "Versions:"
    lines = list(dropwhile(lambda a: a != "Versions:", lines))[1:]

    ver_pred = "   + "
    lines = list(
        map(
            lambda a: a[len(ver_pred) :],
            filter(lambda a: a.startswith(ver_pred), lines),
        )
    )
    return lines


def take_dev_versions(versions):
    def func(txt):
        # X.Y.ZdevW
        return bool(re.search("\\d\\.\\d\\.\\ddev\\d", txt))

    return list(filter(func, versions))


def extract_filenames(text):
    lines = text.splitlines()

    # skip until "version"
    lines = list(dropwhile(lambda a: not a.startswith("version"), lines))[1:]

    ver_pred = "   + "
    lines = list(
        map(
            lambda a: a[len(ver_pred) :],
            filter(lambda a: a.startswith(ver_pred), lines),
        )
    )
    return lines


def extract_timestamp(text):
    lines = text.splitlines()

    timestamp_pred = "    + timestamp"
    elem = next(x for x in lines if x.startswith(timestamp_pred))

    res = elem[elem.find(":") + 1 :]
    return int(res.strip())


def cleanup_packages(package_path, keep_count, token, dry_run, verbose):
    def print_verbose(*args):
        if verbose:
            print(*args)

    client = "anaconda"
    res = run_command([client, "show", package_path])
    versions = extract_versions(res)
    print_verbose("versions", versions)

    versions = take_dev_versions(versions)
    print_verbose("dev versions", versions)

    to_remove = []

    for version in versions:
        version_str = package_path + "/" + version
        res = run_command([client, "show", version_str])

        filenames = extract_filenames(res)
        print_verbose("filenames", version, " ", filenames)

        files = []
        for file in filenames:
            full_path = version_str + "/" + file
            res = run_command([client, "show", full_path])
            time = extract_timestamp(res)

            print_verbose("timestamp", file, time)
            files.append((time, full_path))

        print_verbose("files", files)
        files.sort(key=lambda x: x[0])
        print_verbose("sorted files", files)

        files = files[: len(files) - keep_count]
        print_verbose("to remove files", files)

        to_remove += map(lambda a: a[1], files)

    print_verbose("to remove", to_remove)

    if dry_run:
        print("command will be")
        args = [client, "remove", "--token", "<token>", "-f"] + to_remove
        print(" ".join(args))
    else:
        args = [client, "remove", "--token", token, "-f"] + to_remove
        subprocess.run(args, check=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", dest="package_path", type=str, required=True)
    parser.add_argument("--keep_count", dest="keep_count", type=int, required=True)
    parser.add_argument("--token", dest="token", type=str, default="")
    parser.add_argument("--dry_run", dest="dry_run", action="store_true", default=False)
    parser.add_argument("--verbose", dest="verbose", action="store_true", default=False)

    args = parser.parse_args()

    cleanup_packages(
        package_path=args.package_path,
        keep_count=args.keep_count,
        token=args.token,
        dry_run=args.dry_run,
        verbose=args.verbose,
    )
