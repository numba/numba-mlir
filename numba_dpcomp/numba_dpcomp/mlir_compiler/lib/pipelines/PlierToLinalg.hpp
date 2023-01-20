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

void registerPlierToLinalgPipeline(imex::PipelineRegistry &registry);

llvm::StringRef plierToLinalgGenPipelineName();
llvm::StringRef plierToLinalgOptPipelineName();
