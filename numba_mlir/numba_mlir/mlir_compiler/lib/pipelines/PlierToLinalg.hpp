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

void registerPlierToLinalgPipeline(numba::PipelineRegistry &registry);

llvm::StringRef plierToLinalgRegionPipelineName();
llvm::StringRef plierToLinalgGenPipelineName();
llvm::StringRef plierToLinalgOptPipelineName();
