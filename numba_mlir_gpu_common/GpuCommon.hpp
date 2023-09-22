// SPDX-FileCopyrightText: 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <cstdint>
#include <string_view>

namespace sycl {
inline namespace _V1 {
class queue;
}
} // namespace sycl

namespace numba {

class GPUStreamInterface {
public:
  virtual ~GPUStreamInterface() = default;

  /// Returns device name.
  virtual std::string_view getDeviceName() = 0;

  /// Returns sycl queue associated with this stream or null if queue is not
  /// available.
  virtual sycl::queue *getQueue() = 0;
};

enum class GpuAllocType { Device = 0, Shared = 1, Local = 2 };

enum class GpuParamType : int32_t {
  null = 0,
  bool_,
  int8,
  int16,
  int32,
  int64,
  float32,
  float64,
  ptr,
};

// Must be kept in sync with the compiler.
struct GPUParamDesc {
  const void *data;
  int32_t size;
  GpuParamType type;

  bool operator==(const GPUParamDesc &rhs) const {
    return data == rhs.data && size == rhs.size && type == rhs.type;
  }

  bool operator!=(const GPUParamDesc &rhs) const { return !(*this == rhs); }
};

typedef void (*MemInfoDtorFunction)(void *ptr, size_t size, void *info);
using MemInfoAllocFuncT = void *(*)(void *, size_t, MemInfoDtorFunction,
                                    void *);

struct GPUAllocResult {
  void *ptr;
  void *event;
};

} // namespace numba
