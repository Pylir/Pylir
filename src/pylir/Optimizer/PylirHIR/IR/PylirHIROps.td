// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef PYLIR_HIR_OPS_TABLEGEN
#define PYLIR_HIR_OPS_TABLEGEN

include "mlir/IR/OpBase.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"

include "pylir/Optimizer/PylirHIR/IR/PylirHIRDialect.td"
include "pylir/Optimizer/PylirHIR/IR/PylirHIRFunctionInterface.td"

include "pylir/Optimizer/PylirPy/IR/PylirPyTypes.td"

class PylirHIR_Op<string mneomic, list<Trait> traits = []>
  : Op<PylirHIR_Dialect, mneomic, traits>;

//===----------------------------------------------------------------------===//
// Basics
//===----------------------------------------------------------------------===//

def PylirHIR_BuiltinsRefOp : PylirHIR_Op<"builtinsRef", [Pure]> {
  let arguments = (ins FlatSymbolRefAttr:$reference);

  let results = (outs DynamicType:$result);

  let description = [{
    Op used to refer to objects in Pythons `builtins` module.
    Returns a reference to that object.
    Unlike most ops using symbol references, this op does not verify, nor care
    whether the symbol exists.
    Symbol names should be fully qualified.

    Example:
    ```
    %0 = pyHIR.builtinsRef @builtins.None
    ```
  }];

  let assemblyFormat = [{
    $reference attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// Functions
//===----------------------------------------------------------------------===//

def PylirHIR_FunctionType : Type<
  And<[
    FunctionType.predicate,
    CPred<"$_self.cast<::mlir::FunctionType>().getNumResults() == 1">,
    SubstLeaves<"$_self", "$_self.cast<::mlir::FunctionType>().getResult(0)",
      DynamicType.predicate>,
    Concat<"llvm::all_of($_self.cast<::mlir::FunctionType>().getInputs(), "
      "[](mlir::Type t) { return t && (",
        SubstLeaves<"$_self", "t", DynamicType.predicate>, ");"
    "})">,
  ]>>
{
  let summary = "python function";
  let cppClassName = FunctionType.cppClassName;
}

def PylirHIR_GlobalFuncOp : PylirHIR_Op<"globalFunc",
  [Symbol, FunctionOpInterface, IsolatedFromAbove, OpAsmOpInterface,
    PylirHIR_FunctionInterface]> {

  let arguments = (ins StrAttr:$sym_name,
             TypeAttrOf<PylirHIR_FunctionType>:$function_type,
             OptionalAttr<DictArrayAttr>:$arg_attrs,
             OptionalAttr<DictArrayAttr>:$res_attrs,
             StrArrayAttr:$parameter_names,
             DenseI32ArrayAttr:$parameter_name_mapping,
             DenseI32ArrayAttr:$keyword_only_mapping,
             OptionalAttr<I32Attr>:$pos_rest,
             OptionalAttr<I32Attr>:$keyword_rest
  );

  let regions = (region MinSizedRegion<1>:$body);

  let builders = [
    OpBuilder<(ins "llvm::Twine":$symbolName,
                   "llvm::ArrayRef<FunctionParameterSpec>":$parameters)>
  ];

  let description = [{

    Op representing a function in global scope with no use of non-local
    variables or similar.
    This is meant as a more restrictive and more optimized form of `pyHIR.func`.

    Syntax:
    ```
    arg ::= (`*` | `**`) value-def [[`only`] string-attr] [ dictionary-attr ]
    global_func ::= `pyHIR.global_func` sym_name `(` [ <arg> { `,` <arg> } ] `)` [ `->` dictionary-attr ] attr-with-keyword
            `{` body `}`
    ```
  }];

  let extraClassDeclaration = [{

    //===------------------------------------------------------------------===//
    // FunctionOpInterface implementation
    //===------------------------------------------------------------------===//

    /// Returns the argument types of this function.
    llvm::ArrayRef<mlir::Type> getArgumentTypes() {
      return getFunctionType().getInputs();
    }

    /// Returns the result types of this function.
    llvm::ArrayRef<mlir::Type> getResultTypes() {
      return getFunctionType().getResults();
    }

    //===------------------------------------------------------------------===//
    // OpAsmOpInterface implementation
    //===------------------------------------------------------------------===//

    static llvm::StringRef getDefaultDialect() {
      return pylir::HIR::PylirHIRDialect::getDialectNamespace();
    }

    //===------------------------------------------------------------------===//
    // HIR::FunctionInterface implementation
    //===------------------------------------------------------------------===//

    mlir::ValueRange getDefaultValues() {
      return {};
    }

    llvm::ArrayRef<std::int32_t> getDefaultValuesMapping() {
      return {};
    }

    //===------------------------------------------------------------------===//
    // CallableOpInterface implementations
    //===------------------------------------------------------------------===//

    mlir::Region* getCallableRegion() {
      return &getBody();
    }

    llvm::ArrayRef<mlir::Type> getCallableResults() {
      return getResultTypes();
    }

    mlir::ArrayAttr getCallableArgAttrs() {
      return getArgAttrs().value_or(nullptr);
    }

    mlir::ArrayAttr getCallableResAttrs() {
      return getResAttrs().value_or(nullptr);
    }
  }];

  let hasVerifier = 1;
  let hasCustomAssemblyFormat = 1;
}

def PylirHIR_FuncOp : PylirHIR_Op<"func", [OpAsmOpInterface,
  PylirHIR_FunctionInterface]> {

  let arguments = (ins StrAttr:$name, Variadic<AnyType>:$default_values,
             DenseI32ArrayAttr:$default_values_mapping,
             TypeAttrOf<PylirHIR_FunctionType>:$function_type,
             OptionalAttr<DictArrayAttr>:$arg_attrs,
             OptionalAttr<DictArrayAttr>:$res_attrs,
             StrArrayAttr:$parameter_names,
             DenseI32ArrayAttr:$parameter_name_mapping,
             DenseI32ArrayAttr:$keyword_only_mapping,
             OptionalAttr<I32Attr>:$pos_rest,
             OptionalAttr<I32Attr>:$keyword_rest
             );

  let regions = (region MinSizedRegion<1>:$body);

  let results = (outs DynamicType:$result);

  let builders = [
    OpBuilder<(ins "llvm::Twine":$name,
                   "llvm::ArrayRef<FunctionParameterSpec>":$parameters)>
  ];

  let description = [{

    Op representing a python function object.
    This defines a function with a given name and parameters containing all
    semantically relevant python constructs such as parameter names, default
    values, positional and keyword rest arguments etc.

    It returns a new python function object as value representing the function.
    It is not marked `IsolatedFromAbove` allowing the use and import of
    `nonlocal` variables in the function body.

    Syntax:
    ```
    arg ::= (`*` | `**`) value-def [[`only`] string-attr] [ `=` value-use ] [ dictionary-attr ]
    global_func ::= `pyHIR.func` name `(` [ <arg> { `,` <arg> } ] `)` [ `->` dictionary-attr ] attr-with-keyword
            `{` body `}`
    ```
  }];

  let extraClassDeclaration = [{

    //===------------------------------------------------------------------===//
    // HIR::FunctionInterface implementation
    //===------------------------------------------------------------------===//

    /// Returns the argument types of this function.
    llvm::ArrayRef<mlir::Type> getArgumentTypes() {
      return getFunctionType().getInputs();
    }

    /// Returns the result types of this function.
    llvm::ArrayRef<mlir::Type> getResultTypes() {
      return getFunctionType().getResults();
    }

    //===------------------------------------------------------------------===//
    // OpAsmOpInterface implementation
    //===------------------------------------------------------------------===//

    static llvm::StringRef getDefaultDialect() {
      return pylir::HIR::PylirHIRDialect::getDialectNamespace();
    }

    //===------------------------------------------------------------------===//
    // CallableOpInterface implementations
    //===------------------------------------------------------------------===//

    mlir::Region* getCallableRegion() {
      return &getBody();
    }

    llvm::ArrayRef<mlir::Type> getCallableResults() {
      return getResultTypes();
    }

    mlir::ArrayAttr getCallableArgAttrs() {
      return getArgAttrs().value_or(nullptr);
    }

    mlir::ArrayAttr getCallableResAttrs() {
      return getResAttrs().value_or(nullptr);
    }
  }];

  let hasVerifier = 1;
  let hasCustomAssemblyFormat = 1;
}

def PylirHIR_ReturnOp : PylirHIR_Op<"return", [Pure, ReturnLike, Terminator,
  ParentOneOf<["FuncOp", "GlobalFuncOp"]>]> {
  let arguments = (ins DynamicType:$value);

  let description = [{
    Operation used to return from within a function.
    Control is returned back to the caller and the result of the function call
    is populated with `$value`.
  }];

  let assemblyFormat = "$value attr-dict";
}

//===----------------------------------------------------------------------===//
// Modules
//===----------------------------------------------------------------------===//

def PylirHIR_InitOp : PylirHIR_Op<"init", [NoRegionArguments, IsolatedFromAbove,
  OpAsmOpInterface]> {

  let arguments = (ins StrAttr:$name);

  let regions = (region MinSizedRegion<1>:$body);

  let assemblyFormat = "$name attr-dict-with-keyword $body";

  let description = [{
    This op represents the initializer body of a module `$name`, or in other
    words, the global scope of a python source file.
    The region may only be exited by an instruction throwing an exception or via
    `pyHIR.init_return`.
    In the latter case, a dictionary must be returned which in python are the
    global variables of the module and the namespace of the module.
  }];

  let extraClassDeclaration = [{

    bool isMainModule() {
      return getName() == "__main__";
    }

    //===------------------------------------------------------------------===//
    // OpAsmOpInterface implementation
    //===------------------------------------------------------------------===//

    static llvm::StringRef getDefaultDialect() {
      return pylir::HIR::PylirHIRDialect::getDialectNamespace();
    }
  }];
}

def PylirHIR_InitReturnOp : PylirHIR_Op<"init_return", [HasParent<"InitOp">,
  Terminator, ReturnLike, Pure]> {
  let arguments = (ins DynamicType:$global_dict);

  let description = [{
    Op used to terminate a `pyHIR.init` body.
    `$global_dict` are the global variables of the module and the namespace of
    the module and MUST be a dictionary.
  }];

  let assemblyFormat = "$global_dict attr-dict";
}

#endif
