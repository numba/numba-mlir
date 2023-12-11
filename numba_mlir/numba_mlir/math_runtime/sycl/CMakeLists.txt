# SPDX-FileCopyrightText: 2021 - 2022 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

cmake_minimum_required(VERSION 3.20)

project(numba-mlir-math-sycl-runtime LANGUAGES CXX C)

include(GenerateExportHeader)

set(SOURCES_LIST
    lib/NumpyLinalg.cpp
    )
set(HEADERS_LIST
    ../include/Common.hpp
    )

add_library(${PROJECT_NAME} SHARED ${SOURCES_LIST} ${HEADERS_LIST})
generate_export_header(${PROJECT_NAME})

target_include_directories(${PROJECT_NAME} PRIVATE
    ${PROJECT_BINARY_DIR}
    "../include"
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../../numba_mlir_gpu_common
)

target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_17)

option(NUMBA_MLIR_USE_MKL
    "Enable mkl support"
    ON
)

option(NUMBA_MLIR_USE_SYCL
    "Enable sycl support"
    OFF
)

message(STATUS "NUMBA_MLIR_USE_SYCL ${NUMBA_MLIR_USE_SYCL}")
message(STATUS "NUMBA_MLIR_USE_MKL ${NUMBA_MLIR_USE_MKL}")

if(NUMBA_MLIR_USE_SYCL AND NUMBA_MLIR_USE_MKL)
    find_package(IntelSYCL REQUIRED)
    add_sycl_to_target(TARGET ${PROJECT_NAME} SOURCES ${SOURCES_LIST})
    if (NOT DEFINED(MKL_THREADING))
        set(MKL_THREADING tbb_thread)
    endif()

    find_package(MKL CONFIG REQUIRED)

    target_compile_options(${PROJECT_NAME} PUBLIC
        $<TARGET_PROPERTY:MKL::MKL_DPCPP,INTERFACE_COMPILE_OPTIONS>)

    target_include_directories(${PROJECT_NAME} PRIVATE
        $<TARGET_PROPERTY:MKL::MKL_DPCPP,INTERFACE_INCLUDE_DIRECTORIES>)

    target_link_libraries(${PROJECT_NAME} PUBLIC $<LINK_ONLY:MKL::MKL_DPCPP>)

    target_compile_definitions(${PROJECT_NAME} PRIVATE NUMBA_MLIR_USE_SYCL_MKL=1)
    target_compile_options(${PROJECT_NAME} PRIVATE -fno-sycl-id-queries-fit-in-int)
endif()

install(TARGETS ${PROJECT_NAME}
        DESTINATION "${MATH_SYCL_RUNTIME_INSTALL_PATH}"
        )
