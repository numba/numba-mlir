// SPDX-FileCopyrightText: 2021 - 2022 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

namespace numba {
class PipelineRegistry;
}

namespace llvm {
class StringRef;
}

namespace mlir {
class MLIRContext;
class TypeConverter;
} // namespace mlir

void registerPreLowSimpleficationsPipeline(numba::PipelineRegistry &registry);

llvm::StringRef untuplePipelineName();
llvm::StringRef removeSignPipelineName();
