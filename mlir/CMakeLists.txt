# SPDX-FileCopyrightText: 2021 - 2022 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

find_package(MLIR REQUIRED CONFIG)

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

file(STRINGS ${CMAKE_CURRENT_SOURCE_DIR}/../llvm-sha.txt EXPECTED_LLVM_SHA)
message(STATUS "Expected llvm sha: \"${EXPECTED_LLVM_SHA}\"")

file(STRINGS ${LLVM_INCLUDE_DIR}/llvm/Support/VCSRevision.h REVISION_FILE_DATA)
message(DEBUG "VCSRevision: ${REVISION_FILE_DATA}")
string(REGEX MATCH "\"([^\"]*)\"" LLVM_SHA ${REVISION_FILE_DATA})
string(REPLACE "\"" "" LLVM_SHA ${LLVM_SHA})
message(STATUS "llvm sha: \"${LLVM_SHA}\"")

if (NOT EXPECTED_LLVM_SHA STREQUAL LLVM_SHA)
    message(FATAL_ERROR "Invalid llvm version")
endif()

add_subdirectory(include/numba/Dialect/plier)
add_subdirectory(include/numba/Dialect/numba_util)
add_subdirectory(include/numba/Dialect/gpu_runtime/IR)
add_subdirectory(include/numba/Dialect/ntensor/IR)
add_subdirectory(include/numba/Dialect/math_ext/IR)
add_subdirectory(include/legacy/Dialect/Arith/Transforms)
add_subdirectory(include/legacy/Dialect/Bufferization/Transforms)
add_subdirectory(include/legacy/Dialect/Linalg/Transforms)
add_subdirectory(include/legacy/Dialect/Tensor/Transforms)

set(SOURCES_LIST
    lib/Analysis/AliasAnalysis.cpp
    lib/Analysis/MemorySsa.cpp
    lib/Analysis/MemorySsaAnalysis.cpp
    lib/Compiler/Compiler.cpp
    lib/Compiler/PipelineRegistry.cpp
    lib/Conversion/CfgToScf.cpp
    lib/Conversion/DecomposeMemrefs.cpp
    lib/Conversion/GpuRuntimeToLlvm.cpp
    lib/Conversion/GpuToGpuRuntime.cpp
    lib/Conversion/MathExtToLibm.cpp
    lib/Conversion/NtensorToLinalg.cpp
    lib/Conversion/NtensorToMemref.cpp
    lib/Conversion/SCFToAffine/SCFToAffine.cpp
    lib/Conversion/UtilConversion.cpp
    lib/Conversion/UtilToLlvm.cpp
    lib/Dialect/gpu_runtime/IR/GpuRuntimeOps.cpp
    lib/Dialect/gpu_runtime/Transforms/MakeBarriersUniform.cpp
    lib/Dialect/math_ext/IR/MathExtDialect.cpp
    lib/Dialect/math_ext/IR/MathExtOps.cpp
    lib/Dialect/ntensor/IR/NTensorOps.cpp
    lib/Dialect/ntensor/Transforms/PropagateEnvironment.cpp
    lib/Dialect/ntensor/Transforms/ResolveArrayOps.cpp
    lib/Dialect/numba_util/Dialect.cpp
    lib/Dialect/plier/Dialect.cpp
    lib/ExecutionEngine/ExecutionEngine.cpp
    lib/legacy/Dialect/Arith/Transforms/Bufferize.cpp
    lib/legacy/Dialect/Bufferization/Transforms/Bufferize.cpp
    lib/legacy/Dialect/Linalg/Transforms/Bufferize.cpp
    lib/legacy/Dialect/Tensor/Transforms/Bufferize.cpp
    lib/Transforms/CallLowering.cpp
    lib/Transforms/CanonicalizeReductions.cpp
    lib/Transforms/CastLowering.cpp
    lib/Transforms/CastUtils.cpp
    lib/Transforms/CommonOpts.cpp
    lib/Transforms/CompositePass.cpp
    lib/Transforms/ConstUtils.cpp
    lib/Transforms/CopyRemoval.cpp
    lib/Transforms/ExpandTuple.cpp
    lib/Transforms/FuncTransforms.cpp
    lib/Transforms/FuncUtils.cpp
    lib/Transforms/IfRewrites.cpp
    lib/Transforms/IndexTypePropagation.cpp
    lib/Transforms/InlineUtils.cpp
    lib/Transforms/LoopRewrites.cpp
    lib/Transforms/LoopUtils.cpp
    lib/Transforms/MakeSignless.cpp
    lib/Transforms/MemoryRewrites.cpp
    lib/Transforms/PipelineUtils.cpp
    lib/Transforms/PromoteBoolMemref.cpp
    lib/Transforms/PromoteToParallel.cpp
    lib/Transforms/SCFVectorize.cpp
    lib/Transforms/ScalarOpsConversion.cpp
    lib/Transforms/ShapeIntegerRangePropagation.cpp
    lib/Transforms/TypeConversion.cpp
    lib/Transforms/UpliftMath.cpp
    lib/Utils.cpp
    )
set(HEADERS_LIST
    include/numba/Analysis/AliasAnalysis.hpp
    include/numba/Analysis/MemorySsa.hpp
    include/numba/Analysis/MemorySsaAnalysis.hpp
    include/numba/Compiler/Compiler.hpp
    include/numba/Compiler/PipelineRegistry.hpp
    include/numba/Conversion/CfgToScf.hpp
    include/numba/Conversion/GpuRuntimeToLlvm.hpp
    include/numba/Conversion/GpuToGpuRuntime.hpp
    include/numba/Conversion/MathExtToLibm.hpp
    include/numba/Conversion/NtensorToLinalg.hpp
    include/numba/Conversion/NtensorToMemref.hpp
    include/numba/Conversion/SCFToAffine/SCFToAffine.h
    include/numba/Conversion/UtilConversion.hpp
    include/numba/Conversion/UtilToLlvm.hpp
    include/numba/Dialect/gpu_runtime/IR/GpuRuntimeOps.hpp
    include/numba/Dialect/gpu_runtime/Transforms/MakeBarriersUniform.hpp
    include/numba/Dialect/math_ext/IR/MathExt.hpp
    include/numba/Dialect/ntensor/IR/NTensorOps.hpp
    include/numba/Dialect/ntensor/Transforms/PropagateEnvironment.hpp
    include/numba/Dialect/ntensor/Transforms/ResolveArrayOps.hpp
    include/numba/Dialect/numba_util/Dialect.hpp
    include/numba/Dialect/numba_util/Utils.hpp
    include/numba/Dialect/plier/Dialect.hpp
    include/numba/ExecutionEngine/ExecutionEngine.hpp
    include/numba/Transforms/CallLowering.hpp
    include/numba/Transforms/CanonicalizeReductions.hpp
    include/numba/Transforms/CastLowering.hpp
    include/numba/Transforms/CastUtils.hpp
    include/numba/Transforms/CommonOpts.hpp
    include/numba/Transforms/CompositePass.hpp
    include/numba/Transforms/ConstUtils.hpp
    include/numba/Transforms/CopyRemoval.hpp
    include/numba/Transforms/ExpandTuple.hpp
    include/numba/Transforms/FuncTransforms.hpp
    include/numba/Transforms/FuncUtils.hpp
    include/numba/Transforms/IfRewrites.hpp
    include/numba/Transforms/IndexTypePropagation.hpp
    include/numba/Transforms/InlineUtils.hpp
    include/numba/Transforms/LoopRewrites.hpp
    include/numba/Transforms/LoopUtils.hpp
    include/numba/Transforms/MakeSignless.hpp
    include/numba/Transforms/MemoryRewrites.hpp
    include/numba/Transforms/PipelineUtils.hpp
    include/numba/Transforms/PromoteBoolMemref.hpp
    include/numba/Transforms/PromoteToParallel.hpp
    include/numba/Transforms/RewriteWrapper.hpp
    include/numba/Transforms/SCFVectorize.hpp
    include/numba/Transforms/ScalarOpsConversion.hpp
    include/numba/Transforms/ShapeIntegerRangePropagation.hpp
    include/numba/Transforms/TypeConversion.hpp
    include/numba/Transforms/UpliftMath.hpp
    include/numba/Utils.hpp
    )

set(NUMBA_MLIR_LIB "numba-mlir")

add_library(${NUMBA_MLIR_LIB} STATIC ${SOURCES_LIST} ${HEADERS_LIST})

apply_llvm_compile_flags(${NUMBA_MLIR_LIB})

include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG(-Winvalid-offsetof __offsetof_flag)
if (__offsetof_flag)
    set_source_files_properties(
        lib/Analysis/MemorySsa.cpp
        PROPERTIES
        COMPILE_FLAGS "-Wno-invalid-offsetof"
    )
endif()
unset(__offsetof_flag)

target_link_libraries(${NUMBA_MLIR_LIB} PRIVATE
    MLIRComplexToSPIRV
    MLIRControlFlowToSCF
    MLIRControlFlowToSPIRV
    MLIRFuncTransforms
    MLIRIR
    MLIRLLVMDialect
    MLIRLinalgTransforms
    MLIRMathToSPIRV
    MLIRTensorTransforms
    MLIRTransforms
    MLIRUBDialect
    MLIRUBToSPIRV
    )

target_include_directories(${NUMBA_MLIR_LIB} SYSTEM PRIVATE
    ${MLIR_INCLUDE_DIRS}
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../numba_mlir_gpu_common
    ./lib
    )

target_include_directories(${NUMBA_MLIR_LIB} PUBLIC
    ./include
    ${PROJECT_BINARY_DIR}/mlir/include
    ${PROJECT_BINARY_DIR}/numba/include
    )

add_dependencies(
    ${NUMBA_MLIR_LIB}
    MLIRPlierOpsIncGen
    MLIRNumbaUtilOpsIncGen
    MLIRGpuRuntimeOpsIncGen
    MLIRNTensorOpsIncGen
    MLIRMathExtOpsIncGen
    MLIRArithLegacyIncGen
    MLIRBufferizationLegacyIncGen
    MLIRLinalgLegacyIncGen
    MLIRTensorLegacyIncGen)

add_subdirectory(tools)
if(NUMBA_MLIR_ENABLE_TESTS)
    add_subdirectory(test)
endif()
