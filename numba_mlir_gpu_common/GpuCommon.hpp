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

// Must be kept in sync with the compiler.
struct OffloadDeviceCapabilities {
  uint16_t spirvMajorVersion;
  uint16_t spirvMinorVersion;
  bool hasFP16;
  bool hasFP64;
};

} // namespace numba
