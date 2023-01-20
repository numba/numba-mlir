# SPDX-FileCopyrightText: 2021 - 2023 Intel Corporation
# SPDX-FileCopyrightText: 2023 Numba project
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

project(dpcomp-gpu-runtime LANGUAGES CXX C)

include(GenerateExportHeader)

set(SOURCES_LIST
    lib/FilterStringParser.cpp
    lib/GpuRuntime.cpp
)
set(HEADERS_LIST
    lib/FilterStringParser.hpp
    lib/LevelZeroPrinting.hpp
    lib/LevelZeroWrapper.hpp
)

add_library(${PROJECT_NAME} SHARED ${SOURCES_LIST} ${HEADERS_LIST})
generate_export_header(${PROJECT_NAME})

target_include_directories(${PROJECT_NAME} PRIVATE
    ${PROJECT_BINARY_DIR}
    ${LevelZero_INCLUDE_DIR}
)

target_link_directories(${PROJECT_NAME} PRIVATE
    ${LevelZero_LIBRARIES_DIR}
)

target_link_libraries(${PROJECT_NAME} PRIVATE ${LevelZero_LIBRARY})
