# SPDX-FileCopyrightText: 2022 - 2023 Intel Corporation
# SPDX-FileCopyrightText: 2023 Numba project
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

def pytest_configure(config):
    config.addinivalue_line("markers", "smoke")
    config.addinivalue_line("markers", "numba_parfor")
