#pragma once

#include "mlir/Pass/Pass.h"

namespace mlir {
namespace bufferization {
namespace legacy {
/// Create a pass that bufferizes ops from the bufferization dialect.
std::unique_ptr<Pass> createBufferizationBufferizePass();

} // namespace legacy
} // namespace bufferization
} // namespace mlir
