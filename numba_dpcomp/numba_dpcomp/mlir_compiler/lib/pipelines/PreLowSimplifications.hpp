// SPDX-FileCopyrightText: 2021 - 2023 Intel Corporation
// SPDX-FileCopyrightText: 2023 Numba project
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

namespace imex {
class PipelineRegistry;
}

namespace llvm {
class StringRef;
}

namespace mlir {
class MLIRContext;
class TypeConverter;
} // namespace mlir

void registerPreLowSimpleficationsPipeline(imex::PipelineRegistry &registry);

llvm::StringRef untuplePipelineName();
llvm::StringRef removeSignPipelineName();
