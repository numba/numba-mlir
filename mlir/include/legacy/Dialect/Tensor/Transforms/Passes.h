#pragma once

#include "mlir/Pass/Pass.h"

namespace mlir {
namespace tensor {
namespace legacy {
/// Creates an instance of the `tensor` dialect bufferization pass.
std::unique_ptr<Pass> createTensorBufferizePass();

} // namespace legacy
} // namespace tensor
} // namespace mlir
