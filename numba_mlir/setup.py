# SPDX-FileCopyrightText: 2021 - 2022 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os
import shutil
import tempfile
import sys
import subprocess
from setuptools import find_packages, setup
import versioneer
import numpy
import platform


root_dir = os.path.dirname(os.path.abspath(__file__))

# CMAKE =======================================================================


def buildSyclMathRuntime(
    root_path, build_prefix, install_prefix, use_mkl, use_sycl, tbb_dir
):
    NUMBA_MLIR_USE_MKL = use_mkl
    NUMBA_MLIR_USE_SYCL = use_sycl
    MATH_SYCL_RUNTIME_INSTALL_PATH = install_prefix

    cmake_build_dir_parent = os.path.join(build_prefix)
    cmake_build_dir = os.path.join(build_prefix, "sycl/math_runtime")
    sycl_math_runtime_path = os.path.join(root_path, "numba_mlir/math_runtime/sycl")
    cmake_cmd = [
        "cmake",
        sycl_math_runtime_path,
    ]

    cmake_cmd += ["-GNinja"]

    cmake_cmd += [
        "-DCMAKE_BUILD_TYPE=Release",
        "-DMATH_SYCL_RUNTIME_INSTALL_PATH=" + MATH_SYCL_RUNTIME_INSTALL_PATH,
        "-DTBB_DIR=" + TBB_DIR,
    ]

    if NUMBA_MLIR_USE_MKL is not None:
        cmake_cmd += ["-DNUMBA_MLIR_USE_MKL=" + NUMBA_MLIR_USE_MKL]

    if NUMBA_MLIR_USE_SYCL is not None:
        cmake_cmd += ["-DNUMBA_MLIR_USE_SYCL=" + NUMBA_MLIR_USE_SYCL]

    try:
        os.makedirs(cmake_build_dir)
    except FileExistsError:
        pass

    env = os.environ.copy()

    sys_name = platform.system()
    if sys_name.casefold() == "Windows".casefold():
        c_compiler = "icx"
        cpp_compiler = c_compiler
    else:
        c_compiler = "icx"
        cpp_compiler = "icpx"

    env["CC"] = c_compiler
    env["CXX"] = cpp_compiler

    subprocess.check_call(
        cmake_cmd, stderr=subprocess.STDOUT, shell=False, cwd=cmake_build_dir, env=env
    )
    subprocess.check_call(
        ["cmake", "--build", ".", "--config", "Release"], cwd=cmake_build_dir
    )
    subprocess.check_call(
        ["cmake", "--install", ".", "--config", "Release"], cwd=cmake_build_dir
    )

    return ["numba_mlir-math-sycl-runtime"], [MATH_SYCL_RUNTIME_INSTALL_PATH]


if int(os.environ.get("NUMBA_MLIR_SETUP_RUN_CMAKE", 1)):
    LLVM_PATH = os.environ["LLVM_PATH"]
    LLVM_DIR = os.path.join(LLVM_PATH, "lib", "cmake", "llvm")
    MLIR_DIR = os.path.join(LLVM_PATH, "lib", "cmake", "mlir")
    TBB_DIR = os.path.join(os.environ["TBB_PATH"], "lib", "cmake", "tbb")
    NUMBA_MLIR_USE_MKL = os.environ.get("NUMBA_MLIR_USE_MKL")
    NUMBA_MLIR_USE_SYCL = os.environ.get("NUMBA_MLIR_USE_SYCL")
    CMAKE_INSTALL_PREFIX = os.path.join(root_dir, "..")

    cmake_build_dir = os.path.join(CMAKE_INSTALL_PREFIX, "numba_mlir_cmake_build")

    install_dir = os.path.join(CMAKE_INSTALL_PREFIX, "numba_mlir/numba_mlir")
    buildSyclMathRuntime(
        root_dir,
        cmake_build_dir,
        install_dir,
        NUMBA_MLIR_USE_MKL,
        NUMBA_MLIR_USE_SYCL,
        TBB_DIR,
    )

    cmake_cmd = [
        "cmake",
        CMAKE_INSTALL_PREFIX,
    ]

    cmake_cmd += ["-GNinja"]

    NUMPY_INCLUDE_DIR = numpy.get_include()

    cmake_cmd += [
        "-DCMAKE_BUILD_TYPE=Release",
        "-DLLVM_DIR=" + LLVM_DIR,
        "-DMLIR_DIR=" + MLIR_DIR,
        "-DTBB_DIR=" + TBB_DIR,
        "-DCMAKE_INSTALL_PREFIX=" + CMAKE_INSTALL_PREFIX,
        "-DPython3_NumPy_INCLUDE_DIRS=" + NUMPY_INCLUDE_DIR,
        "-DPython3_FIND_STRATEGY=LOCATION",
        "-DNUMBA_MLIR_ENABLE_NUMBA_FE=ON",
        "-DNUMBA_MLIR_ENABLE_TBB_SUPPORT=ON",
        "-DLLVM_ENABLE_ZSTD=OFF",
    ]

    if NUMBA_MLIR_USE_SYCL is not None:
        cmake_cmd += ["-DNUMBA_MLIR_USE_SYCL=" + NUMBA_MLIR_USE_SYCL]

    if NUMBA_MLIR_USE_MKL is not None:
        cmake_cmd += ["-DNUMBA_MLIR_USE_MKL=" + NUMBA_MLIR_USE_MKL]

    # DPNP
    try:
        from dpnp import get_include as dpnp_get_include

        DPNP_LIBRARY_DIR = os.path.join(dpnp_get_include(), "..", "..")
        DPNP_INCLUDE_DIR = dpnp_get_include()
        cmake_cmd += [
            "-DDPNP_LIBRARY_DIR=" + DPNP_LIBRARY_DIR,
            "-DDPNP_INCLUDE_DIR=" + DPNP_INCLUDE_DIR,
            "-DNUMBA_MLIR_USE_DPNP=ON",
        ]
        print("Found DPNP at", DPNP_LIBRARY_DIR)
    except ImportError:
        print("DPNP not found")

    # GPU/L0
    LEVEL_ZERO_DIR = os.getenv("LEVEL_ZERO_DIR", None)
    if LEVEL_ZERO_DIR is None:
        print("LEVEL_ZERO_DIR is not set")
    else:
        print("LEVEL_ZERO_DIR is", LEVEL_ZERO_DIR)
        cmake_cmd += [
            "-DNUMBA_MLIR_ENABLE_IGPU_DIALECT=ON",
        ]

    try:
        os.mkdir(cmake_build_dir)
    except FileExistsError:
        pass

    # dpcpp conda package installs it's own includes to conda/include folders
    # breaking every other compiler. So, if dpcpp is installed we need to temporaly move
    # complex file and then restore it
    conda_prefix = os.getenv("CONDA_PREFIX", None)
    files_to_move = ["complex", "float.h"]
    orig_path_to_files = [None, None]
    tmp_path = [None, None]
    tmp_path_to_files = [None, None]

    def move_file(filename):
        orig_path, tmp_path_to_file, tmp_path = None, None, None
        try:
            orig_path = os.path.join(conda_prefix, "include/" + filename)
            if os.path.isfile(orig_path):
                tf = tempfile.NamedTemporaryFile(delete=False)
                tf.close()
                tmp_path = tf.name
                shutil.move(orig_path, tmp_path)
                tmp_path_to_file, tmp_path = tmp_path, None
        except:
            pass

        return orig_path, tmp_path_to_file, tmp_path

    try:
        if conda_prefix is not None:
            for i, filename in enumerate(files_to_move):
                orig_path_to_files[i], tmp_path_to_files[i], tmp_path[i] = move_file(
                    filename
                )

        subprocess.check_call(
            cmake_cmd, stderr=subprocess.STDOUT, shell=False, cwd=cmake_build_dir
        )
        subprocess.check_call(
            ["cmake", "--build", ".", "--config", "Release"], cwd=cmake_build_dir
        )
        subprocess.check_call(
            ["cmake", "--install", ".", "--config", "Release"], cwd=cmake_build_dir
        )
    finally:
        for i, path in enumerate(tmp_path_to_files):
            if path is not None:
                shutil.move(path, orig_path_to_files[i])

        for path in tmp_path:
            if path is not None:
                os.remove(path)

# =============================================================================

packages = find_packages(where=root_dir, include=["numba_mlir", "numba_mlir.*"])

metadata = dict(
    name="numba-mlir",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    packages=packages,
    install_requires=["numba>=0.56,<0.57"],
    include_package_data=True,
)

setup(**metadata)
