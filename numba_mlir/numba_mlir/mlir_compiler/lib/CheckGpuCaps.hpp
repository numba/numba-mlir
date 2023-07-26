// SPDX-FileCopyrightText: 2022 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <cstdint>
#include <optional>
#include <string>

namespace numba {
struct OffloadDeviceCapabilities {
  uint16_t spirvMajorVersion;
  uint16_t spirvMinorVersion;
  bool hasFP16;
  bool hasFP64;
};

std::optional<std::pair<std::string, numba::OffloadDeviceCapabilities>>
getDefaultDevice();
} // namespace numba
