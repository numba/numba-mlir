# SPDX-FileCopyrightText: 2021 - 2022 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

project(numba-mlir-math-runtime LANGUAGES CXX C)

include(GenerateExportHeader)

set(SOURCES_LIST
    lib/Common.cpp
    lib/NumpyLinalg.cpp
    )
set(HEADERS_LIST
    include/Common.hpp
    )

add_library(${PROJECT_NAME} SHARED ${SOURCES_LIST} ${HEADERS_LIST})
generate_export_header(${PROJECT_NAME})

target_include_directories(${PROJECT_NAME} PRIVATE
    ${PROJECT_BINARY_DIR} "include")

target_link_libraries(${PROJECT_NAME})

if(NUMBA_MLIR_USE_MKL)
    if (NOT DEFINED(MKL_THREADING))
        set(MKL_THREADING tbb_thread)
    endif()

    find_package(MKL CONFIG REQUIRED)

    target_compile_options(${PROJECT_NAME} PUBLIC
        $<TARGET_PROPERTY:MKL::MKL,INTERFACE_COMPILE_OPTIONS>)

    target_include_directories(${PROJECT_NAME} PRIVATE
        $<TARGET_PROPERTY:MKL::MKL,INTERFACE_INCLUDE_DIRECTORIES>)

    target_link_libraries(${PROJECT_NAME} PUBLIC $<LINK_ONLY:MKL::MKL>)
    target_compile_definitions(${PROJECT_NAME} PRIVATE NUMBA_MLIR_USE_MKL=1)
endif()
