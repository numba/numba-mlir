// SPDX-FileCopyrightText: 2021 - 2022 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "PyFuncResolver.hpp"

#include "Mangle.hpp"
#include "PyMapTypes.hpp"

#include <pybind11/pybind11.h>

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/UB/IR/UBOps.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/PatternMatch.h>

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
    mlir::PatternRewriter &rewriter, mlir::Location loc, mlir::ModuleOp module,
    llvm::StringRef name, mlir::ValueRange args,
    llvm::ArrayRef<llvm::StringRef> kwnames, mlir::ValueRange kwargs) const {
  assert(!name.empty());
  auto funcDesc = context->resolver(py::str(name.data(), name.size()));
  if (funcDesc.is_none())
    return std::nullopt;

  auto funcDescTuple = funcDesc.cast<py::tuple>();

  auto pyFunc = funcDescTuple[0];
  auto flags = funcDescTuple[1];
  auto argNames = funcDescTuple[2].cast<py::list>();
  auto argHasDefault = funcDescTuple[3].cast<py::list>();
  auto argDefaults = funcDescTuple[4].cast<py::list>();

  Result res;

  auto reserveSize = args.size() + kwargs.size();
  res.mappedArgs.reserve(reserveSize);

  bool failedToConvert = false;
  py::list pyTypes;
  auto convertTypeToNumba = [&](mlir::Type type) {
    auto pyType = mapTypeToNumba(context->types, type);
    if (pyType.is_none()) {
      failedToConvert = true;
    } else {
      pyTypes.append(pyType);
    }
  };

  auto omitted = context->types.attr("Omitted");

  auto addArg = [&](mlir::Value val) {
    convertTypeToNumba(val.getType());
    res.mappedArgs.emplace_back(val);
  };

  for (auto it : llvm::zip(argNames, argHasDefault, argDefaults)) {
    auto arg = std::get<0>(it);
    auto kwarg = [&]() -> mlir::Value {
      auto argName = arg.cast<std::string>();
      for (auto &&[name, kw] : llvm::zip(kwnames, kwargs)) {
        if (argName == name)
          return kw;
      }
      return {};
    }();

    if (kwarg) {
      addArg(kwarg);
      continue;
    }

    if (args.empty()) {
      auto hasDefValue = std::get<1>(it);
      if (hasDefValue.cast<bool>()) {
        mlir::Value newVal = rewriter.create<mlir::ub::PoisonOp>(
            loc, rewriter.getNoneType(), nullptr);
        res.mappedArgs.emplace_back(newVal);
        auto def = std::get<2>(it);
        pyTypes.append(omitted(def));
        continue;
      }

      return std::nullopt;
    }

    addArg(args.front());
    args = args.drop_front();
  }

  if (failedToConvert)
    return std::nullopt;

  mlir::ValueRange argsRande(res.mappedArgs);
  auto types = argsRande.getTypes();
  auto mangledName = mangle(name, types);
  auto externalFunc = module.lookupSymbol<mlir::func::FuncOp>(mangledName);
  if (externalFunc) {
    res.func = externalFunc;
  } else {
    rewriter.startOpModification(module);
    auto resOp = static_cast<mlir::Operation *>(
        context->compiler(pyFunc, pyTypes, flags).cast<py::capsule>());
    if (!resOp) {
      rewriter.cancelOpModification(module);
      return std::nullopt;
    }

    res.func = mlir::cast<mlir::func::FuncOp>(resOp);
    res.func.setPrivate();
    res.func.setName(mangledName);
    rewriter.finalizeOpModification(module);
  }

  return res;
}
