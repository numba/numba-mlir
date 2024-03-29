# SPDX-FileCopyrightText: 2021 - 2022 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

project(mlir_compiler LANGUAGES CXX C)

find_package(pybind11 REQUIRED)

find_package(LLVM REQUIRED CONFIG)
find_package(MLIR REQUIRED CONFIG)

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

set(SOURCES_LIST
    lib/CheckGpuCaps.cpp
    lib/Lowering.cpp
    lib/Mangle.cpp
    lib/NumpyResolver.cpp
    lib/PyFuncResolver.cpp
    lib/PyLinalgResolver.cpp
    lib/PyMapTypes.cpp
    lib/PyModule.cpp
    lib/PyTypeConverter.cpp
    lib/pipelines/BasePipeline.cpp
    lib/pipelines/LowerToGpu.cpp
    lib/pipelines/LowerToGpuTypeConversion.cpp
    lib/pipelines/LowerToLlvm.cpp
    lib/pipelines/ParallelToTbb.cpp
    lib/pipelines/PlierToLinalg.cpp
    lib/pipelines/PlierToLinalgTypeConversion.cpp
    lib/pipelines/PlierToScf.cpp
    lib/pipelines/PlierToStd.cpp
    lib/pipelines/PlierToStdTypeConversion.cpp
    lib/pipelines/PreLowSimplifications.cpp
    )
set(HEADERS_LIST
    lib/CheckGpuCaps.hpp
    lib/Lowering.hpp
    lib/Mangle.hpp
    lib/NumpyResolver.hpp
    lib/PyFuncResolver.hpp
    lib/PyLinalgResolver.hpp
    lib/PyMapTypes.hpp
    lib/PyModule.hpp
    lib/PyTypeConverter.hpp
    lib/pipelines/BasePipeline.hpp
    lib/pipelines/LowerToGpu.hpp
    lib/pipelines/LowerToGpuTypeConversion.hpp
    lib/pipelines/LowerToLlvm.hpp
    lib/pipelines/ParallelToTbb.hpp
    lib/pipelines/PlierToLinalg.hpp
    lib/pipelines/PlierToLinalgTypeConversion.hpp
    lib/pipelines/PlierToScf.hpp
    lib/pipelines/PlierToStd.hpp
    lib/pipelines/PlierToStdTypeConversion.hpp
    lib/pipelines/PreLowSimplifications.hpp
    )

pybind11_add_module(${PROJECT_NAME} ${SOURCES_LIST} ${HEADERS_LIST})

if (CMAKE_SYSTEM_NAME STREQUAL Linux)
    target_link_options(${PROJECT_NAME} PRIVATE "LINKER:--version-script=${CMAKE_CURRENT_SOURCE_DIR}/export.txt")
endif()

if (CMAKE_SYSTEM_NAME STREQUAL Darwin)
    target_link_libraries(${PROJECT_NAME} PRIVATE "-Wl,-exported_symbols_list,${CMAKE_CURRENT_SOURCE_DIR}/export_darwin.txt")
endif()

apply_llvm_compile_flags(${PROJECT_NAME})

target_link_libraries(${PROJECT_NAME} PRIVATE
    numba-mlir

    LLVM${LLVM_NATIVE_ARCH}AsmParser
    LLVM${LLVM_NATIVE_ARCH}CodeGen
    LLVM${LLVM_NATIVE_ARCH}Desc
    LLVMOrcJIT
    LLVMTarget
    MLIRBufferizationPipelines
    MLIRComplexToLLVM
    MLIRComplexToStandard
    MLIRControlFlowTransforms
    MLIRFuncInlinerExtension # TODO: Remove
    MLIRFuncTransforms
    MLIRGPUToGPURuntimeTransforms
    MLIRGPUToSPIRV
    MLIRIR
    MLIRLLVMDialect
    MLIRLLVMToLLVMIRTranslation
    MLIRLinalgTransforms
    MLIRMathToLLVM
    MLIRMathToLibm
    MLIRMathTransforms
    MLIRReconcileUnrealizedCasts
    MLIRSCFToControlFlow
    MLIRSCFToGPU
    MLIRSPIRVSerialization
    MLIRSPIRVTransforms
    MLIRTensorTransforms
    MLIRTransforms
    MLIRUBToLLVM
    )

target_include_directories(${PROJECT_NAME} SYSTEM PRIVATE
    ${MLIR_INCLUDE_DIRS}
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../numba_mlir_gpu_common
    ./lib)

if(NUMBA_MLIR_USE_MKL)
    target_compile_definitions(${PROJECT_NAME} PRIVATE NUMBA_MLIR_USE_MKL=1)
endif()

if(NUMBA_MLIR_USE_SYCL AND NUMBA_MLIR_USE_MKL)
    target_compile_definitions(${PROJECT_NAME} PRIVATE NUMBA_MLIR_USE_SYCL_MKL=1)
endif()


if(NUMBA_MLIR_ENABLE_IGPU_DIALECT)
    target_compile_definitions(${PROJECT_NAME} PRIVATE NUMBA_MLIR_ENABLE_IGPU_DIALECT=1)
endif()

install(TARGETS numba-mlir-runtime numba-mlir-math-runtime numba-mlir-python-runtime mlir_compiler
        DESTINATION "${CMAKE_INSTALL_PREFIX}/numba_mlir/numba_mlir"
        )

get_filename_component(_tmp_path1 "${CMAKE_SOURCE_DIR}" REALPATH)
get_filename_component(_tmp_path2 "${CMAKE_INSTALL_PREFIX}" REALPATH)
cmake_path(IS_PREFIX _tmp_path1 ${_tmp_path2} NORMALIZE _result)

if (NOT ${_result})
    message(STATUS "Copy python files")
    install(
        DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../../
        DESTINATION ./numba_mlir
        FILES_MATCHING
        PATTERN "__pycache__" EXCLUDE
        PATTERN "*.py"
        PATTERN "*.cfg"
        PATTERN "*.in"
        PATTERN "*.ini"
        )
endif()
