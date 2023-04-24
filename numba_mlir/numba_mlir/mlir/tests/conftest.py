# SPDX-FileCopyrightText: 2022 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


def pytest_configure(config):
    config.addinivalue_line("markers", "smoke")
    config.addinivalue_line("markers", "numba_parfor")
    config.addinivalue_line("markers", "test_gpu")
