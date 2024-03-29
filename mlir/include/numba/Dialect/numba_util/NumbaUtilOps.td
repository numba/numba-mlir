// SPDX-FileCopyrightText: 2021 - 2022 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef PLIER_UTIL_OPS
#define PLIER_UTIL_OPS

include "mlir/Interfaces/CastInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/LoopLikeInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/ViewLikeInterface.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/SymbolInterfaces.td"

def NumbaUtil_Dialect : Dialect {
  let name = "numba_util";
  let cppNamespace = "::numba::util";

  let dependentDialects = [
    "::mlir::tensor::TensorDialect",
    "::mlir::memref::MemRefDialect"
  ];

  let hasCanonicalizer = 1;
  let hasConstantMaterializer = 1;
  let useDefaultAttributePrinterParser = 1;
  let useDefaultTypePrinterParser = 1;
}

class NumbaUtil_Type<string name, string typeMnemonic, list<Trait> traits = [],
                   string baseCppClass = "::mlir::Type">
    : TypeDef<NumbaUtil_Dialect, name, traits, baseCppClass> {
  let mnemonic = typeMnemonic;
}

class NumbaUtil_Attr<string attrName, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<NumbaUtil_Dialect, attrName, traits> {
  let mnemonic = attrMnemonic;
}

class NumbaUtil_Op<string mnemonic, list<Trait> traits = []>
    : Op<NumbaUtil_Dialect, mnemonic, traits>;

def NumbaUtil_OpaqueType : NumbaUtil_Type<"Opaque", "opaque", [], "::mlir::Type">;

def NumbaUtil_StringType : NumbaUtil_Type<"String", "string", [], "::mlir::Type">;

def NumbaUtil_TypeVar : NumbaUtil_Type<"TypeVar", "typevar", [], "::mlir::Type"> {
  let parameters = (ins
    "::mlir::Type":$type
  );

  let builders = [
    TypeBuilderWithInferredContext<(ins "::mlir::Type":$type), [{
      return $_get(type.getContext(), type);
    }]>
  ];

  let assemblyFormat = "`<` $type `>`";
}

def IndexRangeAttr
    : NumbaUtil_Attr<"IndexRange", "index_range"> {
  let parameters = (ins
    "int64_t":$min,
    "int64_t":$max
    );
  let assemblyFormat = "`<` `[` $min `,` $max `]` `>`";
}

def ParallelAttr : NumbaUtil_Attr<"Parallel", "parallel">;

def AtomicAttr : NumbaUtil_Attr<"Atomic", "atomic">;

def EnforceShapeOp : NumbaUtil_Op<"enforce_shape", [Pure, ViewLikeOpInterface]> {

  let arguments = (ins AnyShaped:$value, Variadic<Index>:$sizes);
  let results = (outs AnyShaped:$result);

  let builders = [OpBuilder<(ins
    "::mlir::Value":$value,
    "::mlir::ValueRange":$shape)
  >];

  let hasCanonicalizer = 1;

  let assemblyFormat = "attr-dict $value `:` type($value) `(` $sizes `)` `->` type($result)";

  let extraClassDeclaration = [{
      ::mlir::Value getViewSource() { return getValue(); }
  }];
}

def ParallelOp : NumbaUtil_Op<"parallel", [
  AttrSizedOperandSegments, DeclareOpInterfaceMethods<LoopLikeOpInterface>,
  SingleBlockImplicitTerminator<"::numba::util::YieldOp">, RecursiveMemoryEffects
]> {

  let arguments = (ins Variadic<Index>:$lowerBounds,
                         Variadic<Index>:$upperBounds,
                         Variadic<Index>:$steps);
  let regions = (region SizedRegion<1> : $region);

  let skipDefaultBuilders = 1;
  let builders = [OpBuilder<(
      ins "::mlir::ValueRange"
      : $lowerBounds, "::mlir::ValueRange"
      : $upperBounds, "::mlir::ValueRange"
      : $steps,
        CArg<"::mlir::function_ref<void(::mlir::OpBuilder &, ::mlir::Location, "
             "::mlir::ValueRange, ::mlir::ValueRange, ::mlir::Value)>",
             "nullptr">)>];

    let extraClassDeclaration = [{
        unsigned getNumLoops() { return getSteps().size(); }

        ::mlir::Block *getBodyBlock() { return &getRegion().front(); }

        ::mlir::ValueRange getBodyLowerBounds() {
          return getBodyBlock()->getArguments().take_front(getNumLoops());
        }

        ::mlir::ValueRange getBodyUpperBounds() {
          auto count = getNumLoops();
          return getBodyBlock()->getArguments().drop_front(count).take_front(count);
        }

        ::mlir::Value getBodyThreadIndex() { return getBodyBlock()->getArguments().back(); }
    }];
}

def YieldOp : NumbaUtil_Op<"yield", [
  Pure, ReturnLike, Terminator, ParentOneOf<["::numba::util::ParallelOp"]>
]> {
  let arguments = (ins Variadic<AnyType> : $results);
  let builders = [OpBuilder<(ins), [{/* nothing to do */}]>];
  // Override default verifier (defined in SCF_Op), no custom verification
  // needed.
  let hasVerifier = 0;
}

def RetainOp : NumbaUtil_Op<"retain", [ViewLikeOpInterface]> {
  let arguments = (ins AnyMemRef : $source);

  let results = (outs AnyMemRef : $output);

  let builders = [OpBuilder<(ins "::mlir::Value" : $value)>];

  let extraClassDeclaration = [{
      ::mlir::Value getViewSource() { return getSource(); }
  }];

  let hasCanonicalizer = 1;

  let assemblyFormat = "$source attr-dict `:` type($source) `to` type($output)";
}

def ChangeLayoutOp : NumbaUtil_Op<"change_layout", [
  DeclareOpInterfaceMethods<CastOpInterface>,
  ViewLikeOpInterface,
  Pure]> {
  let arguments = (ins AnyMemRef : $source);

  let results = (outs AnyMemRef : $dest);
  let assemblyFormat = "$source attr-dict `:` type($source) `to` type($dest)";

  let hasFolder = 1;
  let hasCanonicalizer = 1;

  let extraClassDeclaration = [{
      ::mlir::Value getViewSource() { return getSource(); }
  }];
}

def SignCastOp : NumbaUtil_Op<"sign_cast", [
  DeclareOpInterfaceMethods<CastOpInterface>,
  ViewLikeOpInterface,
  Pure]> {
  let arguments = (ins AnyType:$source);

  let results = (outs AnyType:$dest);
  let hasFolder = 1;
  let hasCanonicalizer = 1;

  let assemblyFormat = "$source attr-dict `:` type($source) `to` type($dest)";

  let extraClassDeclaration = [{
      ::mlir::Value getViewSource() { return getSource(); }
  }];
}

def TakeContextOp : NumbaUtil_Op<"take_context"> {
  let arguments = (ins OptionalAttr<SymbolRefAttr>:$initFunc,
                       OptionalAttr<SymbolRefAttr>:$releaseFunc);

  let results = (outs NumbaUtil_OpaqueType:$context,
                      Variadic<AnyType>:$results);

    let builders = [
      OpBuilder<(ins "::mlir::SymbolRefAttr":$initFunc,
                     "::mlir::SymbolRefAttr":$releaseFunc,
                     "::mlir::TypeRange":$resultTypes)>
   ];
}

def ReleaseContextOp : NumbaUtil_Op<"release_context"> {
  let arguments = (ins NumbaUtil_OpaqueType:$context);
}

def BuildTupleOp : NumbaUtil_Op<"build_tuple", [Pure]> {
  let summary = "Constructs tuple from provided values";
  let description = [{
    "build_tuple" takes multiple SSA values and constructs tuple from them.

    Zero arguments are allowed and will result in an empty tuple.
  }];

  let arguments = (ins Variadic<AnyType>:$args);
  let results = (outs AnyTuple:$result);

  let assemblyFormat = "attr-dict ($args^ `:` type($args) `->`)? type($result)";

  let builders = [
    OpBuilder<(ins "::mlir::ValueRange":$args)>
  ];
}

def TupleExtractOp : NumbaUtil_Op<"tuple_extract", [Pure]> {
  let summary = "Extracts value from tuple";
  let description = [{
    "tuple_extract" extracts element with specific index from tuple.

    If index is out of bounds behaviour is unspecified and is left to the
    lowering passes.
  }];

  let arguments = (ins AnyTuple:$source,
                       Index:$index);
  let results = (outs AnyType:$result);

  let assemblyFormat = "attr-dict $source `:` type($source) `,` $index `->` type($result)";

  let extraClassDeclaration = [{
    /// Helper function to get the index as a simple integer if it is constant.
    ::std::optional<int64_t> getConstantIndex();
  }];

  let hasFolder = 1;

  let builders = [
    OpBuilder<(ins "::mlir::Value":$arg, "size_t":$index)>
  ];
}

def EnvironmentRegionOp : NumbaUtil_Op<"env_region", [
    DeclareOpInterfaceMethods<RegionBranchOpInterface>,
    SingleBlockImplicitTerminator<"::numba::util::EnvironmentRegionYieldOp">,
    RecursiveMemoryEffects]> {
  let summary = "Operation that executes its region with a specific environment";
  let description = [{
    "env_region" executes operations inside its region within a specific
    enviroment. Operations are executed exactly once. All SSA values that
    dominate the op can be accessed inside the op.

    "env_region" takes "environment" attribute and zero or more SSA arguments.

    Actual intepretation of the "environment" attribute and arguments is not
    specified here and is left to the lowering passes.

    Values can be yielded from "env_region" region using "env_region_yield" op.
    Values yielded from this op's region define the op's results.
  }];


  let arguments = (ins AnyAttr:$environment, Variadic<AnyType>:$args);
  let results = (outs Variadic<AnyType>:$results);

  let regions = (region SizedRegion<1>:$region);

  let assemblyFormat =
      "attr-dict $environment ($args^ `:` type($args))? (`->` type($results)^)? $region";

  let extraClassDeclaration = [{
    /// Inline op body into parent region and erase the op.
    static void inlineIntoParent(::mlir::PatternRewriter &builder, EnvironmentRegionOp op);
  }];

  let builders = [
    OpBuilder<(ins "::mlir::Attribute":$environment,
      CArg<"::mlir::ValueRange", "std::nullopt">:$args,
      CArg<"::mlir::TypeRange", "std::nullopt">:$results,
      CArg<"::llvm::function_ref<void(::mlir::OpBuilder &, ::mlir::Location)>", "nullptr">)>
  ];

  let hasCanonicalizer = 1;
}

def EnvironmentRegionYieldOp : NumbaUtil_Op<"env_region_yield", [
  Pure,
  ReturnLike,
  Terminator,
  HasParent<"::numba::util::EnvironmentRegionOp">
  ]> {

  let summary = "Environment region yield and termination operation";
  let description = [{
    "env_region_yield" yields an SSA value from the "env_region" op region and
    terminates it.

    If "env_region_yield" has any operands, the operands must match the parent
    operation's results.
  }];

  let arguments = (ins Variadic<AnyType>:$results);
  let builders = [OpBuilder<(ins), [{ /* nothing to do */ }]>];

  let assemblyFormat = "attr-dict ($results^ `:` type($results))?";
}

def MemrefApplyOffsetOp : NumbaUtil_Op<"memref_apply_offset", [
    Pure, ViewLikeOpInterface
  ]> {

  let summary = "Applies memref offset to the underlying data pointer";
  let description = [{
    "memref_apply_offset" applies memref offset to the underlying data pointer.
    Returned memref will always have dynamic offset of 0.
  }];

  let arguments = (ins AnyMemRef:$source);
  let results = (outs AnyMemRef:$result);

  let extraClassDeclaration = [{
      ::mlir::Value getViewSource() { return getSource(); }
  }];

  let assemblyFormat = "$source attr-dict `:` type($source) `to` type($result)";
}

// TODO: Upstream arith.bitcast doesnt suuport casting between vectors and
// non-vectors
def BitcastOp : NumbaUtil_Op<"bitcast", [Pure]> {

  let arguments = (ins AnyType:$source);
  let results = (outs AnyType:$result);

  let hasFolder = 1;
  let hasVerifier = 1;

  let assemblyFormat = "$source attr-dict `:` type($source) `to` type($result)";
}

// TODO: Upstream
def MemrefBitcastOp : NumbaUtil_Op<"memref_bitcast", [
    Pure, ViewLikeOpInterface, SameOperandsElementType, SameOperandsShape
  ]> {

  let arguments = (ins AnyMemRef:$source);
  let results = (outs AnyMemRef:$result);

  let extraClassDeclaration = [{
      ::mlir::Value getViewSource() { return getSource(); }
  }];

  let hasFolder = 1;
  let hasVerifier = 1;

  let assemblyFormat = "$source attr-dict `:` type($source) `to` type($result)";
}

def StringConstOp : NumbaUtil_Op<"str_constant",
    [ConstantLike, Pure]> {
  let summary = "integer or floating point constant";

  let arguments = (ins StrAttr:$value);
  let results = (outs NumbaUtil_StringType:$result);

  let assemblyFormat = "attr-dict $value";
  let hasFolder = 1;
}

def WrapAllocatedPointer : NumbaUtil_Op<"wrap_allocated_pointer",
    [Pure, DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
  let arguments = (ins AnyType:$ptr, FlatSymbolRefAttr:$dtor, AnyType:$dtor_data);
  let results = (outs AnyType:$result);
}

def GetAllocTokenOp : NumbaUtil_Op<"get_alloc_token", [
    Pure, SameVariadicResultSize]> {

  let arguments = (ins
    AnyStridedMemRef:$source
  );
  let results = (outs Index:$token);

  let hasCanonicalizer = 1;

  let assemblyFormat = [{
    $source `:` type($source) `->` type(results) attr-dict
  }];
}

def ReshapeOp : NumbaUtil_Op<"reshape", [
    Pure, ViewLikeOpInterface]> {

  let summary = "Shaped type reshape operation";
  let description = [{
    Reshapes source into provided shape.

    Total elements count in source array must match destination shape.

    Reshape may be lowered either into copy or memory view, depending on
    lowering strategy, arg types and inferred layout.
  }];

  let arguments = (ins
    AnyShaped:$source,
    Variadic<Index>:$shape
  );

  let results = (outs AnyShaped:$result);

  let extraClassDeclaration = [{
      ::mlir::Value getViewSource() { return getSource(); }
  }];

  let hasCanonicalizer = 1;

  let assemblyFormat = [{
    $source `(` $shape `)` attr-dict `:` functional-type(operands, results)
  }];

  let builders = [
    OpBuilder<(ins
      "::mlir::Value":$source,
      "::mlir::ValueRange":$shape
    )>
  ];
}


#endif // PLIER_UTIL_OPS
