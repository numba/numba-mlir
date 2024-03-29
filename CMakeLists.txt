# SPDX-FileCopyrightText: 2021 - 2022 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

cmake_minimum_required(VERSION 3.20)

project(numba-mlir LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN 1)

if(POLICY CMP0077)
    cmake_policy(SET CMP0077 NEW)
endif()

if(POLICY CMP0116)
    cmake_policy(SET CMP0116 OLD)
endif()

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules/")

# SDL
# set(CMAKE_VERBOSE_MAKEFILE on) # enable for debug
if (CMAKE_SYSTEM_NAME STREQUAL Windows)
    add_compile_options(-GS -D_CRT_SECURE_NO_WARNINGS)
    add_link_options(-DYNAMICBASE -NXCOMPAT -GUARD:CF)
    # add_link_options(-INTEGRITYCHECK) # require signatures of libs, only recommended
endif()
if (CMAKE_SYSTEM_NAME STREQUAL Linux)
    string(CONCAT WARN_FLAGS
        "-Wall "
        "-Wextra "
        "-Winit-self "
        "-Wunused-function "
        "-Wuninitialized "
        "-fdiagnostics-color=auto "
        "-Wno-deprecated-declarations "
    )
    string(CONCAT SDL_FLAGS
        "-D_FORTIFY_SOURCE=2 "
        "-Wformat "
        "-Wformat-security "
        "-Werror=format-security "
        "-fno-delete-null-pointer-checks "
        "-fstack-protector-strong  "
        "-fno-strict-overflow "
        "-fstack-clash-protection "
        "-fcf-protection=full "
    )
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${WARN_FLAGS} ${SDL_FLAGS}")
    # add_compile_options(-mcet) # v8.0 and newer # unrecognized command line option '-mcet', only recommended
    add_link_options(-Wl,-z,noexecstack,-z,relro,-z,now)
endif()
if (CMAKE_SYSTEM_NAME STREQUAL Darwin)
    add_compile_options(-D_FORTIFY_SOURCE=2 -Wformat -Wformat-security -Werror=format-security -fno-delete-null-pointer-checks -fstack-protector-strong -fno-strict-overflow -Wall)
    add_compile_options(-fcf-protection=full) # v8.0 and newer
    # add_compile_options(-mcet) # v8.0 and newer # unrecognized command line option '-mcet', only recommended
endif()

option(NUMBA_MLIR_USE_MKL
    "Enable mkl support"
    ON
)

option(NUMBA_MLIR_USE_SYCL
    "Enable sycl support"
    OFF
)

option(NUMBA_MLIR_ENABLE_IGPU_DIALECT
    "Enable GPU codegen"
    OFF
)
option(NUMBA_MLIR_ENABLE_NUMBA_FE
    "Enable numba-based python frontend"
    OFF
)
option(NUMBA_MLIR_ENABLE_TBB_SUPPORT
    "Enable TBB"
    OFF
)
option(NUMBA_MLIR_ENABLE_TESTS
    "Enable CTests"
    OFF
)

message(STATUS "NUMBA_MLIR_USE_MKL ${NUMBA_MLIR_USE_MKL}")
message(STATUS "NUMBA_MLIR_USE_SYCL ${NUMBA_MLIR_USE_SYCL}")
message(STATUS "NUMBA_MLIR_ENABLE_IGPU_DIALECT ${NUMBA_MLIR_ENABLE_IGPU_DIALECT}")
message(STATUS "NUMBA_MLIR_ENABLE_TESTS ${NUMBA_MLIR_ENABLE_TESTS}")
message(STATUS "NUMBA_MLIR_ENABLE_NUMBA_FE ${NUMBA_MLIR_ENABLE_NUMBA_FE}")
message(STATUS "NUMBA_MLIR_ENABLE_TBB_SUPPORT ${NUMBA_MLIR_ENABLE_TBB_SUPPORT}")

macro(apply_llvm_compile_flags target)
    if (MSVC)
        target_compile_options(${target} PRIVATE /EHsc)
    endif ()
    target_compile_definitions(${target} PRIVATE ${LLVM_DEFINITIONS})
endmacro()

add_subdirectory(mlir)
add_subdirectory(numba_mlir_runtime)

if(NUMBA_MLIR_ENABLE_NUMBA_FE)
    add_subdirectory(numba_mlir)
endif()
