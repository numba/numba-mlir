// SPDX-FileCopyrightText: 2022 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "PlierToLinalgTypeConversion.hpp"

#include "PyTypeConverter.hpp"

#include "numba/Dialect/ntensor/IR/NTensorOps.hpp"
#include "numba/Dialect/numba_util/Dialect.hpp"

#include <pybind11/pybind11.h>

#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/TypeRange.h>
#include <mlir/IR/Types.h>

namespace py = pybind11;

namespace {
struct Conversion {
  Conversion(PyTypeConverter &conv) : converter(conv) {
    py::object mod = py::module::import("numba.core.types");
    dtype = mod.attr("DType");
    array = mod.attr("Array");
    iterator = mod.attr("ArrayIterator");

    py::object arrayMod = py::module::import("numba_mlir.mlir.array_type");
    fixedArray = arrayMod.attr("FixedArray");
  }

  std::optional<mlir::Type> operator()(mlir::MLIRContext &context,
                                       py::handle obj) {
    if (py::isinstance(obj, dtype)) {
      auto type = converter.convertType(context, obj.attr("dtype"));
      if (!type)
        return std::nullopt;

      return numba::util::TypeVarType::get(type);
    }

    if (py::isinstance(obj, fixedArray)) {
      auto elemType = converter.convertType(context, obj.attr("dtype"));
      if (!elemType)
        return std::nullopt;

      auto layout = obj.attr("layout").cast<std::string>();

      auto ndim = obj.attr("ndim").cast<size_t>();

      auto fixedDims = obj.attr("fixed_dims").cast<py::tuple>();
      if (fixedDims.size() != ndim)
        return std::nullopt;

      llvm::SmallVector<int64_t> shape(ndim);
      for (auto &&[i, dim] : llvm::enumerate(fixedDims)) {
        if (dim.is_none()) {
          shape[i] = mlir::ShapedType::kDynamic;
        } else {
          shape[i] = dim.cast<int64_t>();
        }
      }

      return numba::ntensor::NTensorType::get(shape, elemType, /*env*/ {},
                                              llvm::StringRef(layout));
    }

    if (py::isinstance(obj, array)) {
      auto elemType = converter.convertType(context, obj.attr("dtype"));
      if (!elemType)
        return std::nullopt;

      auto layout = obj.attr("layout").cast<std::string>();

      auto ndim = obj.attr("ndim").cast<size_t>();
      llvm::SmallVector<int64_t> shape(ndim, mlir::ShapedType::kDynamic);

      return numba::ntensor::NTensorType::get(shape, elemType, /*env*/ {},
                                              llvm::StringRef(layout));
    }

    if (py::isinstance(obj, iterator)) {
      auto arrayType = (*this)(context, obj.attr("array_type"));
      if (!arrayType || !mlir::isa<numba::ntensor::NTensorType>(*arrayType))
        return std::nullopt;

      return numba::ntensor::IteratorType::get(
          mlir::cast<numba::ntensor::NTensorType>(*arrayType));
    }

    return std::nullopt;
  }

private:
  PyTypeConverter &converter;

  py::object dtype;
  py::object array;
  py::object iterator;
  py::object fixedArray;
};
} // namespace

void populateArrayTypeConverter(PyTypeConverter &converter) {
  converter.addConversion(Conversion(converter));
}
