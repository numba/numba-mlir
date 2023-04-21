// SPDX-FileCopyrightText: 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <string_view>

namespace numba {

class GPUStreamInterface {
public:
  virtual ~GPUStreamInterface() = default;
  virtual std::string_view getDeviceName() = 0;
};

struct OffloadDeviceCapabilities {
  uint16_t spirvMajorVersion;
  uint16_t spirvMinorVersion;
  bool hasFP16;
  bool hasFP64;
};

enum class GpuAllocType { Device = 0, Shared = 1, Local = 2 };

} // namespace numba
