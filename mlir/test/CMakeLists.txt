# SPDX-FileCopyrightText: 2021 - 2022 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Needed for llvm lit to work.
find_package (Python3 COMPONENTS Interpreter REQUIRED)

configure_lit_site_cfg(
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
        ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
        MAIN_CONFIG
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

set(NUMBA_MLIR_TEST_DEPENDS
        FileCheck
        numba-mlir-opt
        level_zero_runner
        dpcomp-runtime
        dpcomp-gpu-runtime
        )

add_lit_testsuite(check-numba-mlir "Running the numba mlir regression tests"
        ${CMAKE_CURRENT_BINARY_DIR}
        DEPENDS ${NUMBA_MLIR_TEST_DEPENDS}
        )
set_target_properties(check-numba-mlir PROPERTIES FOLDER "Tests")

add_lit_testsuites(numba-mlir-tests ${CMAKE_CURRENT_SOURCE_DIR} DEPENDS ${NUMBA_MLIR_TEST_DEPENDS})
