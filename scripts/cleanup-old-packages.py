# SPDX-FileCopyrightText: 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import re
import argparse
from collections import defaultdict
from operator import itemgetter

try:
    from binstar_client.utils import get_server_api, parse_specs, bool_input
except ImportError:
    raise Exception(
        "Script requires anaconda-clinet. Please install it in "
        + "the same environment you are running this script."
    )


def symantic_version(version: str) -> str:
    match = re.search(r"^(\d+\.\d+\.\d+)\.?(dev\d+)?", version)
    ver = match.group(1)
    if match.group(2):
        ver += "." + match.group(2)

    return ver


def is_dev_version(version: str) -> bool:
    # X.Y.ZdevW
    return bool(re.search(r"^\d+\.\d+\.\d+\.?(dev|rc)\d+", version))


def is_dev_file(file) -> bool:
    return is_dev_version(file.get("version"))


def take_dev_versions(versions):
    return list(filter(is_dev_version, versions))


def build_number(file) -> int:
    return max(
        file.get("attrs", {}).get("build_number", 0),
        # wheels attribute is a string
        int(file.get("attrs", {}).get("build_no", 0)),
    )


def max_build(files: list) -> int:
    return max(map(build_number, files), default=0)


def cleanup_packages(
    package_path,
    label,
    token,
    keep_count,
    max_size,
    max_priority,
    dry_run,
    force,
    verbose,
):
    def print_verbose(*args):
        if verbose:
            print(*args)

    aserver_api = get_server_api(token, None)
    spec = parse_specs(package_path)
    package = aserver_api.package(spec.user, spec.package)

    versions = package["versions"]
    print_verbose("versions", versions)

    files = package["files"]
    if label is not None:
        files = list(filter(lambda a: label in a["labels"], files))
    files_by_version = defaultdict(lambda: [])
    for f in files:
        files_by_version[f["version"]].append(f)

    # NIT: max does not work on semantic vesrions.
    # last_dev = max(filter(is_dev_version, versions), default=None)
    last_dev = None
    for v in versions:
        if is_dev_version(v):
            last_dev = v

    print_verbose("last_dev:", last_dev)

    total_size = sum(map(lambda f: f["size"], files))
    print_verbose("total size:", total_size)

    for version in versions:
        is_dev = is_dev_version(version)
        mbd = max_build(files_by_version[version])

        for file in files_by_version[version]:
            prioity = 4

            if build_number(file) == mbd:
                if is_dev and file["version"] != last_dev:
                    prioity = 2
            elif is_dev and file["version"] == last_dev:
                prioity = 3
            elif is_dev:
                prioity = 0
            else:
                prioity = 1

            file["cleanup_priority"] = prioity

    for file in files:
        print_verbose(
            "priority:",
            file["cleanup_priority"],
            "build:",
            build_number(file),
            file["full_name"],
        )
    print_verbose("")

    files.sort(key=itemgetter("cleanup_priority", "upload_time"))

    last_version, last_build = None, None

    cleanup_size = 0
    while len(files) > 0:
        file = files[0]
        version, build = file["version"], build_number(file)

        # check if we have to remove this file
        need_clean = (
            (max_size is not None and total_size > max_size)
            or (max_priority is not None and file["cleanup_priority"] <= max_priority)
            or (keep_count is not None and len(files) > keep_count)
        )

        # clean up all releases of last removed version
        if not need_clean and (
            last_version is None or (last_version != version or last_build != build)
        ):
            last_version, last_build = None, None
            break
        last_version, last_build = file["version"], build_number(file)

        if dry_run or force:
            print_verbose(f"Removing {file['full_name']}", need_clean)

        if not dry_run:
            remove_spec = parse_specs(file["full_name"])
            msg = "Are you sure you want to remove file %s ?" % (remove_spec,)
            if force or bool_input(msg, False):
                aserver_api.remove_dist(
                    remove_spec.user,
                    remove_spec.package,
                    remove_spec.version,
                    remove_spec.basename,
                )

        # iterantion
        files = files[1:]
        total_size -= file["size"]
        cleanup_size += file["size"]

    print_verbose("Cleaned size:", cleanup_size)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--package", dest="package_path", type=str, required=True)
    parser.add_argument("--label", dest="label", type=str, default=None, required=False)
    parser.add_argument("--token", dest="token", type=str, default="")
    parser.add_argument(
        "--keep-count", dest="keep_count", type=int, default=None, required=False
    )
    parser.add_argument(
        "--max-size", dest="max_size", type=int, default=None, required=False
    )
    parser.add_argument(
        "--max-priority",
        dest="max_priority",
        type=int,
        default=None,
        required=False,
        help="0 - removes builds for dev versions other than the most recent build,\n"
        + "    except for builds of the latest dev version;\n"
        + "1 - same as above but also applies to released versions;\n"
        + "2 - removes dev versions except for the last dev version;\n"
        + "3 - in last dev version keep only the latest build;\n"
        + "4 - removes regular builds and latest dev version (e.g. remove everything)\n"
        + "Example:\n"
        + "- pkg_v1.0_bld-0      priority: 1\n"
        + "- pkg_v1.0_bld-1      priority: 4\n"
        + "- pkg_v1.2dev1_bld-0  priority: 2\n"
        + "- pkg_v2.0_bld-0      priority: 4\n"
        + "- pkg_v2.1dev1_bld-0  priority: 0\n"
        + "- pkg_v2.1dev1_bld-1  priotiry: 2\n"
        + "- pkg_v2.1dev2_bld-0  priority: 3\n"
        + "- pkg_v2.1dev2_bld-2  priority: 4\n"
        + "The set of files considered by a given priority includes all the sets considered by lower priorities.\n",
    )
    parser.add_argument("--dry-run", dest="dry_run", action="store_true", default=False)
    parser.add_argument("--force", dest="force", action="store_true", default=False)
    parser.add_argument("--verbose", dest="verbose", action="store_true", default=False)

    args = parser.parse_args()

    if args.max_size is None and args.max_priority is None and args.keep_count is None:
        raise Exception(
            "at least one of --keep-count, --max-size, --max-priority must be set"
        )

    cleanup_packages(
        package_path=args.package_path,
        label=args.label,
        token=args.token,
        keep_count=args.keep_count,
        max_priority=args.max_priority,
        max_size=args.max_size,
        dry_run=args.dry_run,
        force=args.force,
        verbose=args.verbose,
    )
