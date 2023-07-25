// SPDX-FileCopyrightText: 2022 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <cstdint>
#include <optional>
#include <string>

#include "GpuCommon.hpp"

// TODO: device name
std::optional<numba::OffloadDeviceCapabilities>
getOffloadDeviceCapabilities(const std::string &name);

std::optional<std::pair<std::string, numba::OffloadDeviceCapabilities>>
getDefaultDevice();
