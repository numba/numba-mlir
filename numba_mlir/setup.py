# SPDX-FileCopyrightText: 2021 - 2022 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os
from setuptools import find_packages, setup
import versioneer
import platform


root_dir = os.path.dirname(os.path.abspath(__file__))


if int(os.environ.get("NUMBA_MLIR_SETUP_RUN_CMAKE", 1)):
    import setup_helper

    install_dir = os.path.join(root_dir, "numba_mlir")
    setup_helper.build_runtime(install_dir)

packages = find_packages(where=root_dir, include=["numba_mlir", "numba_mlir.*"])

metadata = dict(
    name="numba-mlir",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    packages=packages,
    install_requires=["numba>=0.59.1,<0.60"],
    include_package_data=True,
)

setup(**metadata)
