// SPDX-FileCopyrightText: 2022 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "CheckGpuCaps.hpp"

#include <pybind11/pybind11.h>

namespace py = pybind11;

std::optional<std::pair<std::string, numba::OffloadDeviceCapabilities>>
numba::getDefaultDevice() {
  py::object mod = py::module::import("numba_mlir.mlir.dpctl_interop");
  py::object res = mod.attr("get_default_device")();
  if (res.is_none())
    return std::nullopt;

  numba::OffloadDeviceCapabilities caps{};
  auto name = res.attr("filter_string").cast<std::string>();
  caps.spirvMajorVersion = res.attr("spirv_major_version").cast<int16_t>();
  caps.spirvMinorVersion = res.attr("spirv_minor_version").cast<int16_t>();
  caps.hasFP16 = res.attr("has_fp16").cast<bool>();
  caps.hasFP64 = res.attr("has_fp64").cast<bool>();
  return std::pair{name, caps};
}
