# SPDX-FileCopyrightText: 2021 - 2022 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

cmake_minimum_required(VERSION 3.20)

project(numba-mlir-gpu-runtime-sycl LANGUAGES CXX C)

find_package(IntelDPCPP REQUIRED)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/../cmake/modules/")

find_package(LevelZero REQUIRED)

include(GenerateExportHeader)

set(SOURCES_LIST
    lib/GpuRuntime.cpp
    lib/GpuModule.cpp
)
set(HEADERS_LIST
    lib/GpuModule.hpp
    lib/Utils.hpp
)

add_library(${PROJECT_NAME} SHARED ${SOURCES_LIST} ${HEADERS_LIST})
generate_export_header(${PROJECT_NAME})

if (CMAKE_SYSTEM_NAME STREQUAL Linux)
    target_link_options(${PROJECT_NAME} PRIVATE "LINKER:--version-script=${CMAKE_CURRENT_SOURCE_DIR}/../numba_mlir_gpu_common/export.txt")
endif()

target_include_directories(${PROJECT_NAME} PRIVATE
    ${PROJECT_BINARY_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../numba_mlir_gpu_common
    ${LevelZero_INCLUDE_DIR}
)
target_compile_options(${PROJECT_NAME} PRIVATE -fno-sycl-id-queries-fit-in-int)

install(TARGETS ${PROJECT_NAME}
        DESTINATION "${SYCL_RUNTIME_INSTALL_PATH}"
        )
