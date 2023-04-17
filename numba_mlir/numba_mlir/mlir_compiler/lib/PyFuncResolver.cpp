// SPDX-FileCopyrightText: 2021 - 2022 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "PyFuncResolver.hpp"

#include "Mangle.hpp"
#include "PyMapTypes.hpp"

#include <pybind11/pybind11.h>

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/BuiltinOps.h>

namespace py = pybind11;

struct PyFuncResolver::Context {
  py::handle resolver;
  py::handle compiler;
  py::handle types;
};

PyFuncResolver::PyFuncResolver() : context(std::make_unique<Context>()) {
  auto registryMod = py::module::import("numba_mlir.mlir.func_registry");
  auto compilerMod = py::module::import("numba_mlir.mlir.inner_compiler");
  context->resolver = registryMod.attr("find_active_func");
  context->compiler = compilerMod.attr("compile_func");
  context->types = py::module::import("numba.core.types");
}

PyFuncResolver::~PyFuncResolver() {}

std::optional<PyFuncResolver::Result> PyFuncResolver::getFunc(
    mlir::ModuleOp module, llvm::StringRef name, mlir::ValueRange args,
    llvm::ArrayRef<llvm::StringRef> kwnames, mlir::ValueRange kwargs) const {
  assert(!name.empty());
  auto funcDesc = context->resolver(py::str(name.data(), name.size()));
  if (funcDesc.is_none())
    return std::nullopt;

  auto funcDescTuple = funcDesc.cast<py::tuple>();

  auto pyFunc = funcDescTuple[0];
  auto flags = funcDescTuple[1];
  auto argNames = funcDescTuple[2].cast<py::list>();

  Result res;

  res.mappedArgs.reserve(args.size() + kwargs.size());
  for (auto arg : argNames) {
    auto kwarg = [&]() -> mlir::Value {
      auto argName = arg.cast<std::string>();
      for (auto &&[name, kw] : llvm::zip(kwnames, kwargs)) {
        if (argName == name)
          return kw;
      }
      return {};
    }();

    if (kwarg) {
      res.mappedArgs.emplace_back(kwarg);
      continue;
    }

    if (args.empty())
      return std::nullopt;

    res.mappedArgs.emplace_back(args.front());
    args = args.drop_front();
  }

  mlir::ValueRange argsRande(res.mappedArgs);
  auto types = argsRande.getTypes();
  auto pyTypes = mapTypesToNumba(context->types, types);
  if (pyTypes.is_none())
    return std::nullopt;

  auto mangledName = mangle(name, types);
  auto externalFunc = module.lookupSymbol<mlir::func::FuncOp>(mangledName);
  if (externalFunc) {
    res.func = externalFunc;
  } else {
    auto resOp = static_cast<mlir::Operation *>(
        context->compiler(pyFunc, pyTypes, flags).cast<py::capsule>());
    if (!resOp)
      return std::nullopt;

    res.func = mlir::cast<mlir::func::FuncOp>(resOp);
    res.func.setPrivate();
    res.func.setName(mangledName);
  }

  return res;
}
