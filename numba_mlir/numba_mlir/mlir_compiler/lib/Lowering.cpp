// SPDX-FileCopyrightText: 2021 - 2022 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "Lowering.hpp"

#include <algorithm>
#include <array>
#include <unordered_map>
#include <vector>

#include <pybind11/complex.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Arith/Transforms/BufferDeallocationOpInterfaceImpl.h>
#include <mlir/Dialect/Complex/IR/Complex.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlowOps.h>
#include <mlir/Dialect/ControlFlow/Transforms/BufferDeallocationOpInterfaceImpl.h>
#include <mlir/Dialect/Func/Extensions/InlinerExtension.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/GPU/Transforms/BufferDeallocationOpInterfaceImpl.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/SCF/Transforms/BufferDeallocationOpInterfaceImpl.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>

#include <mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h>
#include <mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h>
#include <mlir/Target/LLVMIR/Export.h>

#include <llvm/Bitcode/BitcodeWriter.h>
#include <llvm/ExecutionEngine/Orc/Mangling.h>
#include <llvm/Support/Debug.h>
#include <llvm/Support/ManagedStatic.h>
#include <llvm/Support/TargetSelect.h>

#include "numba/Dialect/gpu_runtime/IR/GpuRuntimeOps.hpp"
#include "numba/Dialect/ntensor/IR/NTensorOps.hpp"
#include "numba/Dialect/numba_util/Dialect.hpp"
#include "numba/Dialect/plier/Dialect.hpp"

#include "numba/Compiler/Compiler.hpp"
#include "numba/Compiler/PipelineRegistry.hpp"
#include "numba/ExecutionEngine/ExecutionEngine.hpp"
#include "numba/Utils.hpp"

#include "PyTypeConverter.hpp"
#include "pipelines/BasePipeline.hpp"
#include "pipelines/LowerToGpu.hpp"
#include "pipelines/LowerToGpuTypeConversion.hpp"
#include "pipelines/LowerToLlvm.hpp"
#include "pipelines/ParallelToTbb.hpp"
#include "pipelines/PlierToLinalg.hpp"
#include "pipelines/PlierToLinalgTypeConversion.hpp"
#include "pipelines/PlierToScf.hpp"
#include "pipelines/PlierToStd.hpp"
#include "pipelines/PlierToStdTypeConversion.hpp"
#include "pipelines/PreLowSimplifications.hpp"

namespace py = pybind11;
namespace {
#if 0 // Enable func timings
struct Timer {
  using clock = std::chrono::high_resolution_clock;

  Timer(const char *name_) : name(name_), begin(clock::now()) { ++depth; }

  ~Timer() {
    auto d = --depth;
    auto end = clock::now();
    auto dur =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - begin);
    for (auto i : llvm::seq(0, d)) {
      (void)i;
      llvm::errs() << "  ";
    }

    llvm::errs() << name << " took " << dur.count() << "ms\n";
  }

private:
  const char *name;
  clock::time_point begin;
  static thread_local int depth;
};
thread_local int Timer::depth = 0;

#define TIME_FUNC() Timer _scope_timer(__func__)
#else
#define TIME_FUNC() (void)0
#endif

class dummy_complex : public py::object {
public:
  static bool check_(py::handle h) {
    return h.ptr() != nullptr && PyComplex_Check(h.ptr());
  }
};

class CallbackOstream : public llvm::raw_ostream {
public:
  using Func = std::function<void(llvm::StringRef)>;

  CallbackOstream(Func func = nullptr)
      : raw_ostream(/*unbuffered=*/false), callback(std::move(func)), pos(0u) {}

  ~CallbackOstream() override { flush(); }

  void write_impl(const char *ptr, size_t size) override {
    if (callback)
      callback(llvm::StringRef(ptr, size));
    pos += size;
  }

  uint64_t current_pos() const override { return pos; }

  void setCallback(Func func) { callback = std::move(func); }

private:
  Func callback;
  uint64_t pos;
};

static llvm::SmallVector<std::pair<int, py::object>>
getBlocks(py::handle func) {
  llvm::SmallVector<std::pair<int, py::object>> ret;
  auto blocks = func.cast<py::dict>();
  ret.reserve(blocks.size());
  for (auto &&[id, block] : blocks)
    ret.emplace_back(id.cast<int>(), block.cast<py::object>());

  auto pred = [](auto lhs, auto rhs) { return lhs.first < rhs.first; };

  llvm::sort(ret, pred);
  return ret;
}

static py::list getBody(py::handle block) {
  return block.attr("body").cast<py::list>();
}

template <typename T>
static mlir::DenseElementsAttr getArrayData(mlir::ShapedType type,
                                            const py::array &arr) {
  auto sz = static_cast<size_t>(arr.size());
  auto tmp = arr.attr("flatten")().cast<py::array>();
  auto data = static_cast<const T *>(tmp.data());
  return mlir::DenseElementsAttr::get(type, mlir::ArrayRef(data, sz));
}

static std::optional<mlir::Attribute> makeElementsAttr(mlir::ShapedType type,
                                                       const py::array &arr) {
  using fptr_t =
      mlir::DenseElementsAttr (*)(mlir::ShapedType, const py::array &);
  auto dtype = type.getElementType();
  if (auto intType = mlir::dyn_cast<mlir::IntegerType>(dtype)) {
    const std::pair<unsigned, fptr_t> handlers[] = {
        // clang-format off
      {1, &getArrayData<bool>},
      {8, &getArrayData<int8_t>},
      {16, &getArrayData<int16_t>},
      {32, &getArrayData<int32_t>},
      {64, &getArrayData<int64_t>},
        // clang-format on
    };
    auto w = intType.getWidth();
    for (auto &&[ww, handler] : handlers) {
      if (ww == w)
        return handler(type, arr);
    }
  } else if (auto floatType = mlir::dyn_cast<mlir::FloatType>(dtype)) {
    const std::pair<unsigned, fptr_t> handlers[] = {
        // clang-format off
      {32, &getArrayData<float>},
      {64, &getArrayData<double>},
        // clang-format on
    };
    auto w = floatType.getWidth();
    for (auto &&[ww, handler] : handlers) {
      if (ww == w)
        return handler(type, arr);
    }
  } else if (auto complexType = mlir::dyn_cast<mlir::ComplexType>(dtype)) {
    auto elemType =
        mlir::dyn_cast<mlir::FloatType>(complexType.getElementType());
    if (!elemType)
      return std::nullopt;

    const std::pair<unsigned, fptr_t> handlers[] = {
        // clang-format off
      {32, &getArrayData<std::complex<float>>},
      {64, &getArrayData<std::complex<double>>},
        // clang-format on
    };
    auto w = elemType.getWidth();
    for (auto &&[ww, handler] : handlers) {
      if (ww == w)
        return handler(type, arr);
    }
  }

  return std::nullopt;
}

static int64_t getPyInt(py::handle obj) {
  auto max = py::int_(std::numeric_limits<int64_t>::max());
  if (obj <= max)
    return obj.cast<int64_t>();

  return static_cast<int64_t>(obj.cast<uint64_t>());
}

struct InstHandles {
  InstHandles() {
    auto mod = py::module::import("numba.core.ir");
    Assign = mod.attr("Assign");
    Del = mod.attr("Del");
    Return = mod.attr("Return");
    Branch = mod.attr("Branch");
    Jump = mod.attr("Jump");
    SetItem = mod.attr("SetItem");
    StaticSetItem = mod.attr("StaticSetItem");

    auto parforMod = py::module::import("numba.parfors.parfor");
    Parfor = parforMod.attr("Parfor");

    Arg = mod.attr("Arg");
    Expr = mod.attr("Expr");
    Var = mod.attr("Var");
    Const = mod.attr("Const");
    Global = mod.attr("Global");
    FreeVar = mod.attr("FreeVar");

    auto ops = py::module::import("operator");

    for (auto elem : llvm::zip(plier::getOperators(), opsHandles)) {
      auto name = std::get<0>(elem).name;
      if (py::hasattr(ops, name.data())) {
        std::get<1>(elem) = ops.attr(name.data());
      } else {
        llvm::SmallVector<char> storage;
        auto str = (name + "_").toNullTerminatedStringRef(storage);
        std::get<1>(elem) = ops.attr(str.data());
      }
    }

    auto numpy = py::module::import("numpy");
    npInt = numpy.attr("integer");
    npFloat = numpy.attr("floating");
  }

  // Use py::handle to avoid extending python obects lifetime beyound
  // interpreter lifetime.
  py::handle Assign;
  py::handle Del;
  py::handle Return;
  py::handle Branch;
  py::handle Jump;
  py::handle SetItem;
  py::handle StaticSetItem;
  py::handle Parfor;

  py::handle Arg;
  py::handle Expr;
  py::handle Var;
  py::handle Const;
  py::handle Global;
  py::handle FreeVar;

  std::array<py::handle, plier::OperatorsCount> opsHandles;

  py::handle npInt;
  py::handle npFloat;
};

static InstHandles &getInstHandles() {
  static InstHandles ret;
  return ret;
}

static mlir::Attribute parseAttr(mlir::OpBuilder &builder, py::handle obj) {
  if (obj.is_none())
    return builder.getUnitAttr();

  if (py::isinstance<py::str>(obj))
    return builder.getStringAttr(obj.cast<std::string>());

  if (py::isinstance<py::bool_>(obj))
    return builder.getBoolAttr(obj.cast<bool>());

  if (py::isinstance<py::int_>(obj))
    return builder.getI64IntegerAttr(getPyInt(obj));

  numba::reportError(llvm::Twine("Invalid attribute: ") +
                     py::str(obj).cast<std::string>());
}

static void parseAttributes(mlir::Operation *op, py::handle dict) {
  assert(op && "Invalid op");
  mlir::OpBuilder builder(op->getContext());
  for (auto &&[key, value] : dict.cast<py::dict>()) {
    auto attr = parseAttr(builder, value);
    auto keyName = key.cast<std::string>();
    op->setAttr(keyName, attr);
  }
}

struct PlierLowerer final {
  PlierLowerer(mlir::MLIRContext &context, PyTypeConverter &conv)
      : ctx(context), builder(&ctx), insts(getInstHandles()),
        typeConverter(conv) {
    ctx.loadDialect<gpu_runtime::GpuRuntimeDialect>();
    ctx.loadDialect<mlir::cf::ControlFlowDialect>();
    ctx.loadDialect<mlir::func::FuncDialect>();
    ctx.loadDialect<mlir::scf::SCFDialect>();
    ctx.loadDialect<numba::ntensor::NTensorDialect>();
    ctx.loadDialect<numba::util::NumbaUtilDialect>();
    ctx.loadDialect<plier::PlierDialect>();

    numbaUndefined = getNumbaUndefined();
  }

  py::object getNumbaUndefined() {
    py::module_ numbaIr = py::module_::import("numba.core.ir");
    return numbaIr.attr("UNDEFINED");
  }

  mlir::func::FuncOp lower(const py::object &compilationContext,
                           mlir::ModuleOp mod, const py::object &funcIr) {
    TIME_FUNC();
    auto newFunc = createFunc(compilationContext, mod);
    lowerFuncBody(funcIr);
    return newFunc;
  }

  mlir::func::FuncOp lowerParfor(const py::object &compilationContext,
                                 mlir::ModuleOp mod,
                                 const py::object &parforInst) {
    TIME_FUNC();
    auto newFunc = createFunc(compilationContext, mod);
    auto block = func.addEntryBlock();
    mlir::ValueRange blockArgs = block->getArguments();
    auto getNextBlockArg = [&]() -> mlir::Value {
      assert(!blockArgs.empty());
      auto res = blockArgs.front();
      blockArgs = blockArgs.drop_front();
      return res;
    };

    for (auto param : compilationContext["parfor_params"].cast<py::list>()) {
      auto name = param.cast<std::string>();
      varsMap[name] = getNextBlockArg();
    }

    auto getIndexVal = [&](py::handle obj) {
      if (!py::isinstance(obj, insts.Var))
        return;

      auto var = getNextBlockArg();
      varsMap[obj.attr("name").cast<std::string>()] = var;
    };

    for (auto loop : parforInst.attr("loop_nests")) {
      getIndexVal(loop.attr("start"));
      getIndexVal(loop.attr("stop"));
      getIndexVal(loop.attr("step"));
    }

    auto caps = compilationContext["device_caps"];
    mlir::Attribute env;
    if (!caps.is_none()) {
      auto device = caps.attr("filter_string").cast<std::string>();
      auto usmType = "device";
      auto spirvMajor = caps.attr("spirv_major_version").cast<int16_t>();
      auto spirvMinor = caps.attr("spirv_minor_version").cast<int16_t>();
      auto hasFP16 = caps.attr("has_fp16").cast<bool>();
      auto hasFP64 = caps.attr("has_fp64").cast<bool>();
      env = gpu_runtime::GPURegionDescAttr::get(builder.getContext(), device,
                                                usmType, spirvMajor, spirvMinor,
                                                hasFP16, hasFP64);
    }

    builder.setInsertionPointToStart(block);
    numba::util::EnvironmentRegionOp regionOp;
    auto redVars = parforInst.attr("redvars");

    auto numRets =
        py::len(redVars) + py::len(compilationContext["parfor_output_arrays"]);
    llvm::SmallVector<mlir::Type> reductionTypes(numRets,
                                                 builder.getNoneType());

    if (env) {
      auto loc = getCurrentLoc();
      regionOp = builder.create<numba::util::EnvironmentRegionOp>(
          loc, env, /*args*/ std::nullopt, reductionTypes);
      mlir::Block &regionBlock = regionOp.getRegion().front();
      assert(llvm::hasSingleElement(regionBlock));
      regionBlock.getTerminator()->erase();
      builder.setInsertionPointToStart(&regionBlock);
    }

    auto loopResults = lowerParforBody(parforInst);

    llvm::SmallVector<mlir::Value> results;
    for (auto var :
         compilationContext["parfor_output_arrays"].cast<py::list>()) {
      auto varName = var.cast<std::string>();
      results.emplace_back(loadvar(varName));
    }

    results.append(loopResults.begin(), loopResults.end());

    auto loc = getCurrentLoc();
    if (env) {
      builder.create<numba::util::EnvironmentRegionYieldOp>(loc, results);
      for (auto &&[i, res] : llvm::enumerate(reductionTypes))
        regionOp->getResult(i).setType(results[i].getType());

      results = regionOp.getResults();
      builder.setInsertionPointToEnd(block);
    }

    mlir::Value result;
    if (results.empty()) {
      result =
          builder.create<plier::ConstOp>(loc, builder.getNoneType(), nullptr);
    } else if (results.size() == 1) {
      result = results.front();
    } else {
      mlir::ValueRange resultsRange(results);
      auto resType = builder.getTupleType(resultsRange.getTypes());
      result = builder.create<numba::util::BuildTupleOp>(loc, resType, results);
    }

    auto origFuncType = func.getFunctionType();
    assert(origFuncType.getNumResults() == 1);
    auto funcsResultType = origFuncType.getResults().front();
    if (result.getType() != funcsResultType)
      result = builder.create<plier::CastOp>(loc, funcsResultType, result);

    builder.create<mlir::func::ReturnOp>(loc, result);

    fixupPhis();
    return newFunc;
  }

private:
  mlir::MLIRContext &ctx;
  mlir::OpBuilder builder;
  std::vector<mlir::Block *> blocks;
  std::unordered_map<int, mlir::Block *> blocksMap;
  InstHandles &insts;
  mlir::func::FuncOp func;
  std::unordered_map<std::string, mlir::Value> varsMap;
  struct BlockInfo {
    struct PhiDesc {
      mlir::Block *destBlock = nullptr;
      std::string varName;
      unsigned argIndex = 0;
    };
    llvm::SmallVector<PhiDesc, 2> outgoingPhiNodes;
  };
  py::handle currentInstr;
  py::object typemap;
  py::object funcNameResolver;
  py::object globals;
  py::object cellvars;
  py::object numbaUndefined;

  std::unordered_map<mlir::Block *, BlockInfo> blockInfos;

  PyTypeConverter &typeConverter;

  void insertBlock(int id, mlir::Block *block) {
    assert(block && "Invalid block");
    if (blocksMap.count(id))
      numba::reportError(llvm::Twine("Duplicated block id: ") +
                         llvm::Twine(id));

    blocks.emplace_back(block);
    blocksMap.insert(std::pair{id, block});
  }

  mlir::Block *getBlock(py::handle id) const {
    auto i = id.cast<int>();
    auto it = blocksMap.find(i);
    if (it == blocksMap.end())
      numba::reportError(llvm::Twine("Invalid block id: ") + llvm::Twine(i));

    return it->second;
  }

  mlir::func::FuncOp createFunc(const py::object &compilationContext,
                                mlir::ModuleOp mod) {
    TIME_FUNC();
    assert(!func);
    typemap = compilationContext["typemap"];
    funcNameResolver = compilationContext["resolve_func"];
    auto name = compilationContext["fnname"]().cast<std::string>();
    auto typ = getFuncType(compilationContext["fnargs"],
                           compilationContext["restype"]);
    func = mlir::func::FuncOp::create(builder.getUnknownLoc(), name, typ);

    parseAttributes(func, compilationContext["func_attrs"]);

    globals = compilationContext["globals"]();
    cellvars = compilationContext["cellvars"]();

    mod.push_back(func);
    return func;
  }

  mlir::Type getObjType(py::handle obj) const {
    if (auto type = typeConverter.convertType(ctx, obj))
      return type;

    numba::reportError(llvm::Twine("Unhandled type: ") +
                       py::str(obj).cast<std::string>());
  }

  mlir::Type getType(py::handle inst) const {
    auto type = typemap(inst);
    return getObjType(type);
  }

  void lowerFuncBody(py::handle funcIr) {
    TIME_FUNC();
    auto irBlocks = getBlocks(funcIr.attr("blocks"));
    assert(!irBlocks.empty());
    blocks.reserve(irBlocks.size());
    for (auto &&[i, irBlock] : llvm::enumerate(irBlocks)) {
      auto block = (0 == i ? func.addEntryBlock() : func.addBlock());
      insertBlock(irBlock.first, block);
    }

    for (auto &&[i, irBlock] : llvm::enumerate(irBlocks))
      lowerBlock(blocks[i], irBlock.second);

    fixupPhis();
  }

  mlir::ValueRange lowerParforBody(py::handle parforInst) {
    TIME_FUNC();
    auto indexType = builder.getIndexType();
    auto getIndexVal = [&](py::handle obj) -> mlir::Value {
      auto loc = getCurrentLoc();
      if (py::isinstance(obj, insts.Var)) {
        auto var = loadvar(obj);
        return builder.create<plier::CastOp>(loc, indexType, var);
      }
      auto val = getPyInt(obj);
      return builder.create<mlir::arith::ConstantIndexOp>(loc, val);
    };

    llvm::SmallVector<mlir::Value, 4> begins;
    llvm::SmallVector<mlir::Value, 4> ends;
    llvm::SmallVector<mlir::Value, 4> steps;
    llvm::SmallVector<py::object, 4> indexVars;

    for (auto loop : parforInst.attr("loop_nests")) {
      begins.emplace_back(getIndexVal(loop.attr("start")));
      ends.emplace_back(getIndexVal(loop.attr("stop")));
      steps.emplace_back(getIndexVal(loop.attr("step")));
      indexVars.emplace_back(loop.attr("index_variable"));
    }

    auto redVars = parforInst.attr("redvars");
    llvm::SmallVector<mlir::Type> reductionTypes(py::len(redVars),
                                                 builder.getNoneType());

    lowerBlock(parforInst.attr("init_block"));

    llvm::SmallVector<mlir::Value> reductionInits;
    std::unordered_map<std::string, unsigned> reductionIndices;
    for (auto &&[i, redvar] : llvm::enumerate(redVars)) {
      auto name = redvar.cast<std::string>();
      reductionInits.emplace_back(loadvar(name));
      reductionTypes[i] = reductionInits.back().getType();
      reductionIndices[name] = static_cast<unsigned>(i);
    }

    auto bodyBuilder = [&](mlir::OpBuilder &b, mlir::Location l,
                           mlir::ValueRange indices,
                           mlir::ValueRange iterVars) -> mlir::ValueRange {
      assert(!indices.empty());

      auto setIndexVar = [&](py::handle indexVar, mlir::Value index) {
        auto indexType = getObjType(typemap(indexVar));
        index = b.create<plier::CastOp>(l, indexType, index);
        auto indexVarName = indexVar.attr("name").cast<std::string>();
        varsMap[indexVarName] = index;
      };

      {
        mlir::Value index;
        if (indices.size() == 1) {
          index = indices.front();
        } else {
          auto resType = b.getTupleType(indices.getTypes());
          index = b.create<numba::util::BuildTupleOp>(l, resType, indices);
        }

        setIndexVar(parforInst.attr("index_var"), index);
      }

      for (auto &&[i, redvar] : llvm::enumerate(redVars)) {
        assert(i < iterVars.size());
        auto name = redvar.cast<std::string>();
        varsMap[name] = iterVars[i];
      }

      for (auto &&[idx, indexVar] : llvm::zip(indices, indexVars))
        setIndexVar(indexVar, idx);

      auto regionOp = b.create<mlir::scf::ExecuteRegionOp>(l, reductionTypes);
      auto &region = regionOp.getRegion();

      auto irBlocks = getBlocks(parforInst.attr("loop_body"));

      mlir::OpBuilder::InsertionGuard g(b);

      // Make a separate copy as `blocks` vector can be reallocated during
      // lowerInst.
      llvm::SmallVector<mlir::Block *> addedBlocks;
      addedBlocks.reserve(irBlocks.size());
      for (auto &&irBlock : irBlocks) {
        auto block = b.createBlock(&region, region.end());
        blocks.emplace_back(block);
        addedBlocks.emplace_back(block);
        blocksMap[irBlock.first] = block;
      }

      llvm::SmallVector<mlir::Value> reductionsRet(reductionInits.size());
      for (auto &&[bb, irBlock] : llvm::zip(addedBlocks, irBlocks)) {
        b.setInsertionPointToEnd(bb);
        for (auto inst : getBody(irBlock.second)) {
          if (py::isinstance(inst, insts.Assign)) {
            auto target = inst.attr("target");
            auto name = target.attr("name").cast<std::string>();
            auto it = reductionIndices.find(name);
            if (reductionIndices.end() != it) {
              assert(it->second < reductionsRet.size());
              auto val = lowerAssign(inst, target);
              reductionsRet[it->second] = val;
              storevar(val, target);
              continue;
            }
          }

          lowerInst(inst);
        }

        bool hasTerminator =
            !bb->empty() && bb->back().hasTrait<mlir::OpTrait::IsTerminator>();
        if (!hasTerminator)
          b.create<mlir::scf::YieldOp>(l, reductionsRet);
      }

      return regionOp.getResults();
    };

    auto results = buildNestedParallelLoop(begins, ends, steps, reductionInits,
                                           bodyBuilder);
    for (auto &&[i, redvar] : llvm::enumerate(redVars)) {
      assert(i < results.size());
      auto name = redvar.cast<std::string>();
      varsMap[name] = results[i];
    }
    return results;
  }

  mlir::ValueRange buildNestedParallelLoop(
      mlir::ValueRange begins, mlir::ValueRange ends, mlir::ValueRange steps,
      mlir::ValueRange iterArgs,
      llvm::function_ref<mlir::ValueRange(mlir::OpBuilder &, mlir::Location l,
                                          mlir::ValueRange, mlir::ValueRange)>
          bodyBuilder) {
    TIME_FUNC();
    assert(!begins.empty());
    assert(begins.size() == ends.size());
    assert(begins.size() == steps.size());

    auto resultTypes = iterArgs.getTypes();
    auto env = numba::util::ParallelAttr::get(builder.getContext());
    auto loc = getCurrentLoc();
    auto regionOp = builder.create<numba::util::EnvironmentRegionOp>(
        loc, env, /*args*/ mlir::ValueRange{}, resultTypes);
    mlir::Block &body = regionOp.getRegion().front();
    body.getTerminator()->erase();
    mlir::OpBuilder::InsertionGuard g(builder);
    builder.setInsertionPointToStart(&body);

    llvm::SmallVector<mlir::Value> indices;
    indices.reserve(begins.size());
    auto results = buildNestedParallelLoopImpl(
        builder, loc, begins, ends, steps, iterArgs, bodyBuilder, indices);
    builder.create<numba::util::EnvironmentRegionYieldOp>(loc, results);
    return regionOp.getResults();
  }

  mlir::ValueRange buildNestedParallelLoopImpl(
      mlir::OpBuilder &loopBuilder, mlir::Location loc, mlir::ValueRange begins,
      mlir::ValueRange ends, mlir::ValueRange steps, mlir::ValueRange iterArgs,
      llvm::function_ref<mlir::ValueRange(mlir::OpBuilder &, mlir::Location l,
                                          mlir::ValueRange, mlir::ValueRange)>
          bodyBuilder,
      llvm::SmallVectorImpl<mlir::Value> &indices) {
    auto forBodyBuilder = [&](mlir::OpBuilder &b, mlir::Location l,
                              mlir::Value index, mlir::ValueRange vars) {
      indices.emplace_back(index);
      mlir::ValueRange res;
      if (llvm::hasSingleElement(begins)) {
        res = bodyBuilder(b, l, indices, vars);
      } else {
        res = buildNestedParallelLoopImpl(b, l, begins.drop_front(),
                                          ends.drop_front(), steps.drop_front(),
                                          vars, bodyBuilder, indices);
      }
      b.create<mlir::scf::YieldOp>(l, res);
    };
    auto loop = loopBuilder.create<mlir::scf::ForOp>(
        loc, begins.front(), ends.front(), steps.front(), iterArgs,
        forBodyBuilder);
    return loop.getResults();
  }

  void lowerBlock(py::handle irBlock) {
    for (auto it : getBody(irBlock))
      lowerInst(it);
  }

  void lowerBlock(mlir::Block *bb, py::handle irBlock) {
    assert(nullptr != bb);
    mlir::OpBuilder::InsertionGuard g(builder);
    builder.setInsertionPointToEnd(bb);
    lowerBlock(irBlock);
  }

  void lowerInst(py::handle inst) {
    currentInstr = inst;
    if (py::isinstance(inst, insts.Assign)) {
      auto target = inst.attr("target");
      auto val = lowerAssign(inst, target);
      storevar(val, target);
    } else if (py::isinstance(inst, insts.SetItem)) {
      setitem(inst.attr("target"), inst.attr("index"), inst.attr("value"));
    } else if (py::isinstance(inst, insts.StaticSetItem)) {
      staticSetitem(inst.attr("target"), inst.attr("index"),
                    inst.attr("value"));
    } else if (py::isinstance(inst, insts.Del)) {
      delvar(inst.attr("value"));
    } else if (py::isinstance(inst, insts.Return)) {
      retvar(inst.attr("value"));
    } else if (py::isinstance(inst, insts.Branch)) {
      branch(inst.attr("cond"), inst.attr("truebr"), inst.attr("falsebr"));
    } else if (py::isinstance(inst, insts.Jump)) {
      jump(inst.attr("target"));
    } else if (py::isinstance(inst, insts.Parfor)) {
      lowerParforBody(inst);
    } else {
      numba::reportError(llvm::Twine("lower_inst not handled: \"") +
                         py::str(inst.get_type()).cast<std::string>() + "\"");
    }
    currentInstr = nullptr;
  }

  mlir::Value lowerAssign(py::handle inst, py::handle target) {
    auto value = inst.attr("value");
    if (py::isinstance(value, insts.Arg)) {
      auto index = value.attr("index").cast<std::size_t>();
      auto args = func.getFunctionBody().front().getArguments();
      if (index >= args.size())
        numba::reportError(llvm::Twine("Invalid arg index: \"") +
                           llvm::Twine(index) + "\"");

      return args[index];
    }

    if (py::isinstance(value, insts.Expr))
      return lowerExpr(value);

    if (py::isinstance(value, insts.Var))
      return loadvar(value);

    if (py::isinstance(value, insts.Const))
      return getConst(value.attr("value"));

    if (py::isinstance(value, insts.Global) ||
        py::isinstance(value, insts.FreeVar)) {
      auto constVal = getConstOrNull(value.attr("value"));
      if (constVal)
        return *constVal;
      auto name = value.attr("name").cast<std::string>();
      return builder.create<plier::GlobalOp>(getCurrentLoc(), name);
    }

    numba::reportError(llvm::Twine("lower_assign not handled: \"") +
                       py::str(value.get_type()).cast<std::string>() + "\"");
  }

  mlir::Value lowerExpr(py::handle expr) {
    auto op = expr.attr("op").cast<std::string>();
    using func_t = mlir::Value (PlierLowerer::*)(py::handle);
    const std::pair<mlir::StringRef, func_t> handlers[] = {
        {"binop", &PlierLowerer::lowerBinop},
        {"inplace_binop", &PlierLowerer::lowerInplaceBinop},
        {"unary", &PlierLowerer::lowerUnary},
        {"cast", &PlierLowerer::lowerCast},
        {"call", &PlierLowerer::lowerCall},
        {"phi", &PlierLowerer::lowerPhi},
        {"build_tuple", &PlierLowerer::lowerBuildTuple},
        {"getitem", &PlierLowerer::lowerGetitem},
        {"static_getitem", &PlierLowerer::lowerStaticGetitem},
        {"getiter", &PlierLowerer::lowerSimple<plier::GetiterOp>},
        {"iternext", &PlierLowerer::lowerSimple<plier::IternextOp>},
        {"pair_first", &PlierLowerer::lowerSimple<plier::PairfirstOp>},
        {"pair_second", &PlierLowerer::lowerSimple<plier::PairsecondOp>},
        {"getattr", &PlierLowerer::lowerGetattr},
        {"exhaust_iter", &PlierLowerer::lowerExhaustIter},
    };
    for (auto &h : handlers)
      if (h.first == op)
        return (this->*h.second)(expr);

    numba::reportError(llvm::Twine("lower_expr not handled: \"") + op + "\"");
  }

  template <typename T> mlir::Value lowerSimple(py::handle inst) {
    auto value = loadvar(inst.attr("value"));
    return builder.create<T>(getCurrentLoc(), value);
  }

  mlir::Value lowerCast(py::handle inst) {
    auto value = loadvar(inst.attr("value"));
    auto resType = getType(currentInstr.attr("target"));
    return builder.create<plier::CastOp>(getCurrentLoc(), resType, value);
  }

  mlir::Value lowerGetitem(py::handle inst) {
    auto value = loadvar(inst.attr("value"));
    auto index = loadvar(inst.attr("index"));
    return builder.create<plier::GetItemOp>(getCurrentLoc(), value, index);
  }

  mlir::Value lowerStaticIndex(mlir::Location loc, py::handle obj) {
    if (obj.is_none()) {
      return builder.create<plier::ConstOp>(loc, builder.getNoneType(),
                                            nullptr);
    }
    if (py::isinstance<py::int_>(obj)) {
      auto index = getPyInt(obj);
      return builder.create<mlir::arith::ConstantIndexOp>(loc, index);
    }
    if (py::isinstance<py::slice>(obj)) {
      auto start = lowerStaticIndex(loc, obj.attr("start"));
      auto stop = lowerStaticIndex(loc, obj.attr("stop"));
      auto step = lowerStaticIndex(loc, obj.attr("step"));
      return builder.create<plier::BuildSliceOp>(loc, start, stop, step);
    }
    if (py::isinstance<py::tuple>(obj)) {
      auto len = py::len(obj);
      llvm::SmallVector<mlir::Value> args(len);
      llvm::SmallVector<mlir::Type> types(len);
      for (auto &&[i, val] : llvm::enumerate(obj)) {
        auto arg = lowerStaticIndex(loc, val);
        args[i] = arg;
        types[i] = arg.getType();
      }

      auto tupleType = builder.getTupleType(types);
      return builder.create<plier::BuildTupleOp>(loc, tupleType, args);
    }
    if (py::isinstance(obj, insts.npInt)) {
      auto index = getPyInt(obj);
      return builder.create<mlir::arith::ConstantIndexOp>(loc, index);
    }

    numba::reportError(llvm::Twine("Unhandled index type: ") +
                       py::str(obj.get_type()).cast<std::string>());
  }

  mlir::Value lowerStaticGetitem(py::handle inst) {
    auto value = loadvar(inst.attr("value"));
    auto loc = getCurrentLoc();
    auto indexVar = lowerStaticIndex(loc, inst.attr("index"));
    return builder.create<plier::GetItemOp>(loc, value, indexVar);
  }

  mlir::Value lowerBuildTuple(py::handle inst) {
    auto items = inst.attr("items").cast<py::list>();
    mlir::SmallVector<mlir::Value> args;
    for (auto item : items)
      args.push_back(loadvar(item));

    return builder.create<plier::BuildTupleOp>(getCurrentLoc(), args);
  }

  bool isNumbaUndefined(py::handle val) { return val.is(numbaUndefined); }

  std::string makeUndefName() {
    size_t id = 2 * varsMap.size();

    std::string undef = "Undef??";
    while (varsMap.find(undef + std::to_string(id)) != varsMap.end()) {
      ++id;
    }

    return undef + std::to_string(id);
  }

  std::string getPlierUndefined(mlir::Block *block, mlir::Type type) {
    mlir::OpBuilder::InsertionGuard guard(builder);

    builder.setInsertionPointToStart(block);
    auto loc = builder.getUnknownLoc();
    auto undef = builder.create<plier::UndefOp>(loc, type);
    auto undefName = makeUndefName();
    varsMap[undefName] = undef;

    return undefName;
  }

  std::string getNameOrUndefined(py::handle val, mlir::Block *block,
                                 mlir::Type type) {
    if (isNumbaUndefined(val)) {
      return getPlierUndefined(block, type);
    }

    return val.attr("name").cast<std::string>();
  }

  mlir::Value lowerPhi(py::handle expr) {
    auto incomingVals = expr.attr("incoming_values").cast<py::list>();
    auto incomingBlocks = expr.attr("incoming_blocks").cast<py::list>();
    assert(incomingVals.size() == incomingBlocks.size());

    auto currentBlock = builder.getBlock();
    assert(nullptr != currentBlock);

    auto argIndex = currentBlock->getNumArguments();
    auto loc = builder.getUnknownLoc();
    auto phiRetType = getType(currentInstr.attr("target"));
    auto arg = currentBlock->addArgument(phiRetType, loc);

    for (auto i : llvm::seq<size_t>(0, incomingVals.size())) {
      auto block = getBlock(incomingBlocks[i]);
      auto var = getNameOrUndefined(incomingVals[i], block, phiRetType);
      blockInfos[block].outgoingPhiNodes.push_back(
          {currentBlock, std::move(var), argIndex});
    }

    return arg;
  }

  mlir::Value lowerCall(py::handle expr) {
    auto pyPunc = expr.attr("func");
    auto func = loadvar(pyPunc);
    auto args = expr.attr("args").cast<py::list>();
    auto kws = expr.attr("kws").cast<py::list>();
    auto vararg = expr.attr("vararg");

    auto varargVar = (vararg.is_none() ? mlir::Value() : loadvar(vararg));

    mlir::SmallVector<mlir::Value> argsList;
    argsList.reserve(args.size());
    for (auto a : args)
      argsList.push_back(loadvar(a));

    mlir::SmallVector<std::pair<std::string, mlir::Value>> kwargsList;
    for (auto a : kws) {
      auto item = a.cast<py::tuple>();
      auto name = item[0];
      auto valName = item[1];
      kwargsList.push_back({name.cast<std::string>(), loadvar(valName)});
    }

    auto pyFuncName = funcNameResolver(typemap(pyPunc));
    if (pyFuncName.is_none())
      numba::reportError(llvm::Twine("Can't resolve function: ") +
                         py::str(typemap(pyPunc)).cast<std::string>());

    auto funcName = pyFuncName.cast<std::string>();

    return builder.create<plier::PyCallOp>(getCurrentLoc(), func, funcName,
                                           argsList, varargVar, kwargsList);
  }

  mlir::Value lowerBinop(py::handle expr) {
    auto op = expr.attr("fn");
    auto lhsName = expr.attr("lhs");
    auto rhsName = expr.attr("rhs");
    auto lhs = loadvar(lhsName);
    auto rhs = loadvar(rhsName);
    auto opName = resolveOp(op);
    return builder.create<plier::BinOp>(getCurrentLoc(), lhs, rhs, opName);
  }

  mlir::Value lowerInplaceBinop(py::handle expr) {
    auto op = expr.attr("immutable_fn");
    auto lhsName = expr.attr("lhs");
    auto rhsName = expr.attr("rhs");
    auto lhs = loadvar(lhsName);
    auto rhs = loadvar(rhsName);
    auto opName = resolveOp(op);
    return builder.create<plier::InplaceBinOp>(getCurrentLoc(), lhs, rhs,
                                               opName);
  }

  mlir::Value lowerUnary(py::handle expr) {
    auto op = expr.attr("fn");
    auto valName = expr.attr("value");
    auto val = loadvar(valName);
    auto opName = resolveOp(op);
    return builder.create<plier::UnaryOp>(getCurrentLoc(), val, opName);
  }

  llvm::StringRef resolveOp(py::handle op) {
    for (auto elem : llvm::zip(plier::getOperators(), insts.opsHandles))
      if (op.is(std::get<1>(elem)))
        return std::get<0>(elem).op;

    numba::reportError(llvm::Twine("resolve_op not handled: \"") +
                       py::str(op).cast<std::string>() + "\"");
  }

  std::optional<mlir::Attribute> resolveConstant(py::handle val) {
    if (py::isinstance<py::int_>(val)) {
      auto type = builder.getIntegerType(64, /*isSigned*/ true);
      return builder.getIntegerAttr(type, getPyInt(val));
    }

    if (py::isinstance<py::float_>(val))
      return builder.getF64FloatAttr(val.cast<double>());

    if (py::isinstance<dummy_complex>(val)) {
      auto c = val.cast<std::complex<double>>();
      auto type = mlir::ComplexType::get(builder.getF64Type());
      return mlir::complex::NumberAttr::get(type, c.real(), c.imag());
    }

    if (py::isinstance<py::tuple>(val)) {
      auto tup = val.cast<py::tuple>();
      llvm::SmallVector<mlir::Attribute> values(tup.size());
      for (auto &&[i, elem] : llvm::enumerate(tup)) {
        auto val = resolveConstant(elem);
        if (!val)
          return std::nullopt;

        values[i] = *val;
      }
      return builder.getArrayAttr(values);
    }

    if (py::isinstance<py::str>(val))
      return builder.getStringAttr(val.cast<std::string>());

    if (py::isinstance<py::none>(val))
      return builder.getUnitAttr();

    if (py::isinstance<py::array>(val)) {
      auto a = val.cast<py::array>();
      auto dtype = typeConverter.convertType(*builder.getContext(), a.dtype());
      if (!dtype)
        return nullptr;

      auto ndim = static_cast<unsigned>(a.ndim());
      auto *s = a.shape();
      llvm::SmallVector<int64_t> shape(s, s + ndim);
      auto resType = mlir::RankedTensorType::get(shape, dtype);
      return makeElementsAttr(resType, a);
    }

    if (py::isinstance(val, insts.npInt)) {
      auto type = builder.getIntegerType(64, /*isSigned*/ true);
      return builder.getIntegerAttr(type, getPyInt(val));
    }

    if (py::isinstance(val, insts.npFloat))
      return builder.getF64FloatAttr(val.cast<double>());

    return std::nullopt;
  }

  std::optional<mlir::Attribute>
  resolveGlobalAttrImpl(mlir::Value val,
                        llvm::SmallVectorImpl<llvm::StringRef> &names) {
    if (auto global = val.getDefiningOp<plier::GlobalOp>()) {
      std::string name = global.getName().str();
      py::object attr;
      if (cellvars.contains(name.c_str())) {
        attr = cellvars[name.c_str()];
      } else if (globals.contains(name.c_str())) {
        attr = globals[name.c_str()];
      } else {
        return std::nullopt;
      }

      while (!names.empty()) {
        name = names.pop_back_val().str();
        if (!py::hasattr(attr, name.c_str()))
          return std::nullopt;

        attr = attr.attr(name.c_str());
      }

      return resolveConstant(attr);
    }

    if (auto getattr = val.getDefiningOp<plier::GetattrOp>()) {
      auto name = getattr.getName();
      names.emplace_back(name);
      return resolveGlobalAttrImpl(getattr.getValue(), names);
    }

    return std::nullopt;
  }

  std::optional<mlir::Attribute> resolveGlobalAttr(mlir::Value val,
                                                   llvm::StringRef name) {
    llvm::SmallVector<llvm::StringRef> names;
    names.emplace_back(name);
    return resolveGlobalAttrImpl(val, names);
  }

  mlir::Value lowerGetattr(py::handle inst) {
    auto value = loadvar(inst.attr("value"));
    auto name = inst.attr("attr").cast<std::string>();
    auto loc = getCurrentLoc();
    if (auto attr = resolveGlobalAttr(value, name)) {
      // Resolve constant attributes early.
      // TODO: Should be done as part of actual lowering pipeline.
      return builder.create<plier::ConstOp>(loc, *attr);
    }

    return builder.create<plier::GetattrOp>(loc, value, name);
  }

  mlir::Value lowerExhaustIter(py::handle inst) {
    auto value = loadvar(inst.attr("value"));
    auto count = getPyInt(inst.attr("count"));
    return builder.create<plier::ExhaustIterOp>(getCurrentLoc(), value, count);
  }

  void setitem(py::handle target, py::handle index, py::handle value) {
    builder.create<plier::SetItemOp>(getCurrentLoc(), loadvar(target),
                                     loadvar(index), loadvar(value));
  }

  void staticSetitem(py::handle target, py::handle index, py::handle value) {
    auto loc = getCurrentLoc();
    builder.create<plier::SetItemOp>(
        loc, loadvar(target), lowerStaticIndex(loc, index), loadvar(value));
  }

  void storevar(mlir::Value val, py::handle inst) {
    auto type = getType(inst);
    if (val.getDefiningOp()) {
      val.setType(type);
    } else {
      // TODO: unify
      val = builder.create<plier::CastOp>(getCurrentLoc(), type, val);
    }
    varsMap[inst.attr("name").cast<std::string>()] = val;
  }

  mlir::Value loadvar(const std::string &name) const {
    auto it = varsMap.find(name);
    if (varsMap.end() == it)
      numba::reportError(llvm::Twine("Invalid var: ") + name);
    return it->second;
  }

  mlir::Value loadvar(py::handle inst) const {
    auto name = inst.attr("name").cast<std::string>();
    return loadvar(name);
  }

  void delvar(py::handle inst) {
    auto var = loadvar(inst);
    builder.create<plier::DelOp>(getCurrentLoc(), var);
  }

  void retvar(py::handle inst) {
    auto var = loadvar(inst);
    auto funcType = func.getFunctionType();
    auto retType = funcType.getResult(0);
    auto varType = var.getType();
    if (retType != varType)
      var = builder.create<plier::CastOp>(getCurrentLoc(), retType, var);

    builder.create<mlir::func::ReturnOp>(getCurrentLoc(), var);
  }

  void branch(py::handle cond, py::handle tr, py::handle fl) {
    auto c = loadvar(cond);
    auto trBlock = getBlock(tr);
    auto flBlock = getBlock(fl);
    auto condVal = builder.create<plier::CastOp>(
        getCurrentLoc(), mlir::IntegerType::get(&ctx, 1), c);
    builder.create<mlir::cf::CondBranchOp>(getCurrentLoc(), condVal, trBlock,
                                           flBlock);
  }

  void jump(py::handle target) {
    auto block = getBlock(target);
    builder.create<mlir::cf::BranchOp>(getCurrentLoc(), std::nullopt, block);
  }

  std::optional<mlir::Value> getConstOrNull(py::handle val) {
    auto attr = resolveConstant(val);
    if (!attr)
      return std::nullopt;

    return builder.create<plier::ConstOp>(getCurrentLoc(), *attr);
  }

  mlir::Value getConst(py::handle val) {
    auto ret = getConstOrNull(val);
    if (!ret)
      numba::reportError(llvm::Twine("getConst unhandled type \"") +
                         py::str(val.get_type()).cast<std::string>() + "\"");
    return *ret;
  }

  mlir::FunctionType getFuncType(py::handle fnargs, py::handle restype) {
    auto ret = getObjType(restype());
    llvm::SmallVector<mlir::Type> args;
    for (auto arg : fnargs())
      args.push_back(getObjType(arg));

    return mlir::FunctionType::get(&ctx, args, {ret});
  }

  mlir::Location getCurrentLoc() {
    return builder.getUnknownLoc(); // TODO
  }

  void fixupPhis() {
    auto buildArgList = [&](mlir::Block *block, auto &outgoingPhiNodes) {
      mlir::SmallVector<mlir::Value> list;
      for (auto &o : outgoingPhiNodes) {
        if (o.destBlock == block) {
          auto argIndex = o.argIndex;
          if (list.size() <= argIndex)
            list.resize(argIndex + 1);

          auto it = varsMap.find(o.varName);
          assert(varsMap.end() != it);
          auto argType = block->getArgument(argIndex).getType();
          auto val = builder.create<plier::CastOp>(builder.getUnknownLoc(),
                                                   argType, it->second);
          list[argIndex] = val;
        }
      }
      return list;
    };
    for (auto bb : blocks) {
      auto it = blockInfos.find(bb);
      if (blockInfos.end() == it)
        continue;

      auto &info = it->second;
      auto term = bb->getTerminator();
      if (nullptr == term)
        numba::reportError("broken ir: block without terminator");

      builder.setInsertionPointToEnd(bb);

      auto loc = builder.getUnknownLoc();
      if (auto op = mlir::dyn_cast<mlir::cf::BranchOp>(term)) {
        auto dest = op.getDest();
        auto args = buildArgList(dest, info.outgoingPhiNodes);
        op.erase();
        builder.create<mlir::cf::BranchOp>(loc, dest, args);
      } else if (auto op = mlir::dyn_cast<mlir::cf::CondBranchOp>(term)) {
        auto trueDest = op.getTrueDest();
        auto falseDest = op.getFalseDest();
        auto cond = op.getCondition();
        auto trueArgs = buildArgList(trueDest, info.outgoingPhiNodes);
        auto falseArgs = buildArgList(falseDest, info.outgoingPhiNodes);
        op.erase();
        builder.create<mlir::cf::CondBranchOp>(loc, cond, trueDest, trueArgs,
                                               falseDest, falseArgs);
      } else {
        numba::reportError(llvm::Twine("Unhandled terminator: ") +
                           term->getName().getStringRef());
      }
    }
  }
};

numba::CompilerContext::Settings getSettings(py::handle settings,
                                             CallbackOstream &os) {
  numba::CompilerContext::Settings ret;
  ret.verify = settings["verify"].cast<bool>();
  ret.passStatistics = settings["pass_statistics"].cast<bool>();
  ret.passTimings = settings["pass_timings"].cast<bool>();
  ret.irDumpStderr = settings["ir_printing"].cast<bool>();
  ret.diagDumpStderr = settings["diag_printing"].cast<bool>();

  auto printBefore = settings["print_before"].cast<py::list>();
  auto printAfter = settings["print_after"].cast<py::list>();
  if (!printBefore.empty() || !printAfter.empty()) {
    auto callback = settings["print_callback"].cast<py::function>();
    auto getList = [](py::list src) {
      llvm::SmallVector<std::string, 1> res(src.size());
      for (auto &&[i, val] : llvm::enumerate(src))
        res[i] = py::str(val).cast<std::string>();

      return res;
    };
    os.setCallback([callback](llvm::StringRef text) {
      callback(py::str(text.data(), text.size()));
    });
    using S = numba::CompilerContext::Settings::IRPrintingSettings;
    ret.irPrinting = S{getList(printBefore), getList(printAfter), &os};
  }
  return ret;
}

struct ModuleSettings {
  bool enableGpuPipeline = false;
};

static void createPipeline(numba::PipelineRegistry &registry,
                           PyTypeConverter &converter,
                           const ModuleSettings &settings) {
  converter.addConversion(
      [](mlir::MLIRContext &ctx, py::handle obj) -> std::optional<mlir::Type> {
        return plier::PyType::get(&ctx, py::str(obj).cast<std::string>());
      });

  registerBasePipeline(registry);

  registerLowerToLLVMPipeline(registry);

  registerPlierToScfPipeline(registry);

  populateStdTypeConverter(converter);
  registerPlierToStdPipeline(registry);

  populateArrayTypeConverter(converter);
  registerPlierToLinalgPipeline(registry);

  registerPreLowSimpleficationsPipeline(registry);

  registerParallelToTBBPipeline(registry);

  if (settings.enableGpuPipeline) {
#ifdef NUMBA_MLIR_ENABLE_IGPU_DIALECT
    populateGpuTypeConverter(converter);
    registerLowerToGPUPipeline(registry);
    // TODO(nbpatel): Add Gpu->GpuRuntime & GpuRuntimetoLlvm Transformation
#else
    numba::reportError("Numba-MLIR was compiled without GPU support");
#endif
  }
}

struct DialectReg {
  mlir::DialectRegistry registry;
  DialectReg() {
    // TODO: remove this.
    mlir::func::registerInlinerExtension(registry);
    mlir::arith::registerBufferDeallocationOpInterfaceExternalModels(registry);
    mlir::cf::registerBufferDeallocationOpInterfaceExternalModels(registry);
    mlir::gpu::registerBufferDeallocationOpInterfaceExternalModels(registry);
    mlir::scf::registerBufferDeallocationOpInterfaceExternalModels(registry);
  }
};

struct Module {
  DialectReg dialectReg;
  mlir::MLIRContext context;
  numba::PipelineRegistry registry;
  mlir::ModuleOp module;
  PyTypeConverter typeConverter;

  Module(const ModuleSettings &settings) : context(dialectReg.registry) {
    createPipeline(registry, typeConverter, settings);
  }
};

static void runCompiler(Module &mod, const py::object &compilationContext) {
  auto &context = mod.context;
  auto &module = mod.module;
  auto &registry = mod.registry;

  CallbackOstream printStream;
  auto settings =
      getSettings(compilationContext["compiler_settings"], printStream);
  numba::CompilerContext compiler(context, settings, registry);
  compiler.run(module);
}

static auto getLLModulePrinter(py::handle printer) {
  return [func = printer.cast<py::function>()](llvm::Module &m) -> llvm::Error {
    std::string str;
    llvm::raw_string_ostream os(str);
    m.print(os, nullptr);
    os.flush();

    func(py::str(str));
    return llvm::Error::success();
  };
}

static auto getPrinter(py::handle printer) {
  return [func = printer.cast<py::function>()](llvm::StringRef str) {
    func(py::str(str.data(), str.size()));
  };
}

struct GlobalCompilerContext {
  GlobalCompilerContext(const py::dict &settings)
      : executionEngine(getOpts(settings)) {}

  llvm::llvm_shutdown_obj s;
  llvm::SmallVector<std::pair<std::string, void *>, 0> symbolList;
  numba::ExecutionEngine executionEngine;

private:
  numba::ExecutionEngineOptions getOpts(const py::dict &settings) const {
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();
    llvm::InitializeNativeTargetAsmParser();

    numba::ExecutionEngineOptions opts;
    opts.symbolMap =
        [this](llvm::orc::MangleAndInterner m) -> llvm::orc::SymbolMap {
      llvm::orc::SymbolMap ret;
      for (auto &&[name, ptr] : symbolList) {
        llvm::orc::ExecutorSymbolDef jitPtr{
            llvm::orc::ExecutorAddr::fromPtr(ptr),
            llvm::JITSymbolFlags::Exported};
        ret.insert({m(name), jitPtr});
      }
      return ret;
    };
    opts.jitCodeGenOptLevel = llvm::CodeGenOptLevel::Aggressive;

    auto llvmPrinter = settings["llvm_printer"];
    if (!llvmPrinter.is_none())
      opts.transformer = getLLModulePrinter(llvmPrinter);

    auto optimizedPrinter = settings["optimized_printer"];
    if (!optimizedPrinter.is_none())
      opts.lateTransformer = getLLModulePrinter(optimizedPrinter);

    auto asmPrinter = settings["asm_printer"];
    if (!asmPrinter.is_none())
      opts.asmPrinter = getPrinter(asmPrinter);

    return opts;
  }
};
} // namespace

py::capsule initCompiler(py::dict settings) {
  TIME_FUNC();
  auto debugType = settings["debug_type"].cast<py::list>();
  auto debugTypeSize = debugType.size();
  if (debugTypeSize != 0) {
    llvm::DebugFlag = true;
    llvm::BumpPtrAllocator alloc;
    auto types = alloc.Allocate<const char *>(debugTypeSize);
    llvm::StringSaver strSaver(alloc);
    for (size_t i = 0; i < debugTypeSize; ++i)
      types[i] = strSaver.save(debugType[i].cast<std::string>()).data();

    llvm::setCurrentDebugTypes(types, static_cast<unsigned>(debugTypeSize));
  }

  auto context = std::make_unique<GlobalCompilerContext>(settings);
  return py::capsule(context.release(), [](void *ptr) {
    delete static_cast<GlobalCompilerContext *>(ptr);
  });
}

template <typename T>
static bool getDictVal(py::dict &dict, const char *str, T &&def) {
  auto key = py::str(str);
  if (dict.contains(key))
    return dict[key].cast<T>();

  return def;
}

py::capsule createModule(py::dict settings) {
  TIME_FUNC();
  ModuleSettings modSettings;
  modSettings.enableGpuPipeline =
      getDictVal(settings, "enable_gpu_pipeline", false);

  auto mod = std::make_unique<Module>(modSettings);
  {
    mlir::OpBuilder builder(&mod->context);
    mod->module = mlir::ModuleOp::create(builder.getUnknownLoc());
  }
  py::capsule capsule(mod.get(),
                      [](void *ptr) { delete static_cast<Module *>(ptr); });
  mod.release();
  return capsule;
}

py::capsule lowerFunction(const py::object &compilationContext,
                          const py::capsule &pyMod, const py::object &funcIr) {
  TIME_FUNC();
  auto mod = static_cast<Module *>(pyMod);
  auto &context = mod->context;
  auto &module = mod->module;
  auto func = PlierLowerer(context, mod->typeConverter)
                  .lower(compilationContext, module, funcIr);
  return py::capsule(func.getOperation()); // no dtor, func owned by the module.
}

py::capsule lowerParfor(const pybind11::object &compilationContext,
                        const pybind11::capsule &pyMod,
                        const pybind11::object &parforInst) {
  TIME_FUNC();
  auto mod = static_cast<Module *>(pyMod);
  auto &context = mod->context;
  auto &module = mod->module;
  auto func = PlierLowerer(context, mod->typeConverter)
                  .lowerParfor(compilationContext, module, parforInst);
  return py::capsule(func.getOperation()); // no dtor, func owned by the module.
}

py::capsule compileModule(const py::capsule &compiler,
                          const py::object &compilationContext,
                          const py::capsule &pyMod) {
  TIME_FUNC();
  auto context = static_cast<GlobalCompilerContext *>(compiler);
  assert(context);
  auto mod = static_cast<Module *>(pyMod);
  assert(mod);

  runCompiler(*mod, compilationContext);

  auto &mlirCtx = *mod->module->getContext();
  mlir::registerLLVMDialectTranslation(mlirCtx);
  mlir::registerBuiltinDialectTranslation(mlirCtx);
  auto res = context->executionEngine.loadModule(mod->module);
  if (!res)
    numba::reportError(llvm::Twine("Failed to load MLIR module:\n") +
                       llvm::toString(res.takeError()));

  return py::capsule(static_cast<void *>(res.get()));
}

void registerSymbol(const py::capsule &compiler, const py::str &name,
                    const py::int_ &ptr) {
  auto context = static_cast<GlobalCompilerContext *>(compiler);
  assert(context);

  auto ptrValue = reinterpret_cast<void *>(ptr.cast<intptr_t>());
  context->symbolList.emplace_back(name.cast<std::string>(), ptrValue);
}

py::int_ getFunctionPointer(const py::capsule &compiler,
                            const py::capsule &module, py::str funcName) {
  TIME_FUNC();
  auto context = static_cast<GlobalCompilerContext *>(compiler);
  assert(context);
  auto handle = static_cast<numba::ExecutionEngine::ModuleHandle *>(module);
  assert(handle);

  auto name = funcName.cast<std::string>();
  auto res = context->executionEngine.lookup(handle, name);
  if (!res)
    numba::reportError(llvm::Twine("Failed to get function pointer:\n") +
                       llvm::toString(res.takeError()));

  return py::int_(reinterpret_cast<intptr_t>(res.get()));
}

void releaseModule(const py::capsule &compiler, const py::capsule &module) {
  TIME_FUNC();
  auto context = static_cast<GlobalCompilerContext *>(compiler);
  assert(context);
  auto handle = static_cast<numba::ExecutionEngine::ModuleHandle *>(module);
  assert(handle);

  context->executionEngine.releaseModule(handle);
}

py::str moduleStr(const py::capsule &pyMod) {
  auto mod = static_cast<Module *>(pyMod);
  std::string ret;
  llvm::raw_string_ostream ss(ret);
  mod->module.print(ss);
  ss.flush();
  return py::str(ss.str());
}
