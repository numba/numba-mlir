#pragma once

#include "mlir/Pass/Pass.h"

namespace mlir {
namespace linalg {
namespace legacy {
/// Creates an instance of the `linalg` dialect bufferization pass.
std::unique_ptr<Pass> createLinalgBufferizePass();

} // namespace legacy
} // namespace linalg
} // namespace mlir
