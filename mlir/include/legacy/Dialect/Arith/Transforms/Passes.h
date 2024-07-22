#pragma once

#include "mlir/Pass/Pass.h"

namespace mlir {
namespace arith {
namespace legacy {
/// Create a pass to bufferize arith.constant ops.
std::unique_ptr<Pass> createConstantBufferizePass(uint64_t alignment = 0);

} // namespace legacy
} // namespace arith
} // namespace mlir
