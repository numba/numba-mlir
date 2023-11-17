// SPDX-FileCopyrightText: 2022 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "LowerToGpuTypeConversion.hpp"

#include "PyTypeConverter.hpp"

#include "numba/Dialect/gpu_runtime/IR/GpuRuntimeOps.hpp"
#include "numba/Dialect/ntensor/IR/NTensorOps.hpp"

#include <pybind11/pybind11.h>

#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Types.h>

namespace py = pybind11;

namespace {
struct Conversion {
  Conversion(PyTypeConverter &conv) : converter(conv) {
    py::object mod = py::module::import("numba_mlir.mlir.dpctl_interop");
    usmArrayTypes[0] = mod.attr("USMNdArrayType");
    usmArrayTypes[1] = mod.attr("OtherUSMNdArray");
  }

  std::optional<mlir::Type> operator()(mlir::MLIRContext &context,
                                       py::handle obj) {
    if (!checkArrayType(obj))
      return std::nullopt;

    auto elemType = converter.convertType(context, obj.attr("dtype"));
    if (!elemType)
      return std::nullopt;

    auto layout = obj.attr("layout").cast<std::string>();

    auto ndim = obj.attr("ndim").cast<size_t>();

    llvm::SmallVector<int64_t> shape(ndim, mlir::ShapedType::kDynamic);

    if (py::hasattr(obj, "fixed_dims")) {
      auto fixedDims = obj.attr("fixed_dims").cast<py::tuple>();
      if (fixedDims.size() != ndim)
        return std::nullopt;

      for (auto &&[i, dim] : llvm::enumerate(fixedDims))
        if (!dim.is_none())
          shape[i] = dim.cast<int64_t>();
    }

    auto caps = obj.attr("get_device_caps")();
    if (caps.is_none())
      return std::nullopt;

    auto usmType = obj.attr("usm_type").cast<std::string>();

    auto device = caps.attr("filter_string").cast<std::string>();
    auto spirvMajor = caps.attr("spirv_major_version").cast<int16_t>();
    auto spirvMinor = caps.attr("spirv_minor_version").cast<int16_t>();
    auto hasFP16 = caps.attr("has_fp16").cast<bool>();
    auto hasFP64 = caps.attr("has_fp64").cast<bool>();
    auto env = gpu_runtime::GPURegionDescAttr::get(
        &context, device, usmType, spirvMajor, spirvMinor, hasFP16, hasFP64);

    return numba::ntensor::NTensorType::get(shape, elemType, env,
                                            llvm::StringRef(layout));
  }

private:
  PyTypeConverter &converter;

  py::object usmArrayTypes[2];

  bool checkArrayType(py::handle obj) const {
    for (auto &&type : usmArrayTypes)
      if (!type.is_none() && py::isinstance(obj, type))
        return true;

    return false;
  }
};
} // namespace

void populateGpuTypeConverter(PyTypeConverter &converter) {
  converter.addConversion(Conversion(converter));
}
