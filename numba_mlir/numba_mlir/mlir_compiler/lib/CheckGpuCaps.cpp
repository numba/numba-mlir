// SPDX-FileCopyrightText: 2022 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "CheckGpuCaps.hpp"

#include <pybind11/pybind11.h>

namespace py = pybind11;

using ResolveFptr = bool (*)(numba::OffloadDeviceCapabilities *, const char *);

static ResolveFptr getResolver() {
  static ResolveFptr resolver = []() {
    py::object mod = py::module::import("numba_mlir.mlir.gpu_runtime");
    py::object attr = mod.attr("get_device_caps_addr");
    return reinterpret_cast<ResolveFptr>(attr.cast<uintptr_t>());
  }();
  return resolver;
}

std::optional<numba::OffloadDeviceCapabilities>
getOffloadDeviceCapabilities(const std::string &name) {
  auto resolver = getResolver();
  if (!resolver)
    return std::nullopt;

  numba::OffloadDeviceCapabilities ret;
  if (!resolver(&ret, name.c_str()))
    return std::nullopt;

  if (ret.spirvMajorVersion == 0 && ret.spirvMinorVersion == 0)
    return std::nullopt;

  return ret;
}

std::optional<std::pair<std::string, numba::OffloadDeviceCapabilities>>
getDefaultDevice() {
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
