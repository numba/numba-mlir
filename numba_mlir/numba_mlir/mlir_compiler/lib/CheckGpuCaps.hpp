// SPDX-FileCopyrightText: 2022 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <cstdint>
#include <optional>
#include <string>

// Must be kept in sync with gpu_runtime version.
struct OffloadDeviceCapabilities {
  uint16_t spirvMajorVersion;
  uint16_t spirvMinorVersion;
  bool hasFP16;
  bool hasFP64;
};

// TODO: device name
std::optional<OffloadDeviceCapabilities>
getOffloadDeviceCapabilities(const std::string &name);

std::optional<std::string> getDefaultDevice();
