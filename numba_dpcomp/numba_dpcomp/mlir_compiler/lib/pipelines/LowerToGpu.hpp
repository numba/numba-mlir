// SPDX-FileCopyrightText: 2021 - 2022 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

namespace imex {
class PipelineRegistry;
}

namespace llvm {
class StringRef;
}

void registerLowerToGPUPipeline(imex::PipelineRegistry &registry);

llvm::StringRef lowerToGPUPipelineNameHigh();
llvm::StringRef lowerToGPUPipelineNameLow();
