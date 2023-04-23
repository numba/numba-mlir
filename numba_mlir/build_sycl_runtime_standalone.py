# SPDX-FileCopyrightText: 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import sys

import setup_helper

if __name__ == "__main__":
    install_prefix = sys.argv[1]
    setup_helper.build_sycl_runtime(install_prefix)
    setup_helper.build_sycl_math_runtime(install_prefix)
