// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef PYLIR_HIR_OPS_TABLEGEN
#define PYLIR_HIR_OPS_TABLEGEN

include "mlir/IR/EnumAttr.td"
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
include "pylir/Optimizer/PylirPy/Interfaces/ExceptionHandlingInterface.td"

class PylirHIR_Op<string mneomic, list<Trait> traits = []>
  : Op<PylirHIR_Dialect, mneomic, traits>;

//===----------------------------------------------------------------------===//
// Basic Operations
//===----------------------------------------------------------------------===//

def PylirHIR_BinaryOperationAttr : I32EnumAttr<"BinaryOperation",
  "Binary operations in python", [
  I32EnumAttrCase<"Eq", 0, "__eq__">,
  I32EnumAttrCase<"Ne", 1, "__ne__">,
  I32EnumAttrCase<"Lt", 2, "__lt__">,
  I32EnumAttrCase<"Le", 3, "__le__">,
  I32EnumAttrCase<"Gt", 4, "__gt__">,
  I32EnumAttrCase<"Ge", 5, "__ge__">,
  I32EnumAttrCase<"Add", 6, "__add__">,
  I32EnumAttrCase<"Sub", 7, "__sub__">,
  I32EnumAttrCase<"Or", 8, "__or__">,
  I32EnumAttrCase<"Xor", 9, "__xor__">,
  I32EnumAttrCase<"And", 10, "__and__">,
  I32EnumAttrCase<"LShift", 11, "__lshift__">,
  I32EnumAttrCase<"RShift", 12, "__rshift__">,
  I32EnumAttrCase<"Mul", 13, "__mul__">,
  I32EnumAttrCase<"Div", 14, "__div__">,
  I32EnumAttrCase<"FloorDiv", 15, "__floordiv__">,
  I32EnumAttrCase<"Mod", 16, "__mod__">,
  I32EnumAttrCase<"MatMul", 17, "__matmul__">,
]> {
  let cppNamespace = "pylir::HIR";
}

def PylirHIR_BinOp : PylirHIR_Op<"binOp",
  [AddableExceptionHandling<"BinExOp">]> {
  let arguments = (ins PylirHIR_BinaryOperationAttr:$binaryOperation,
                       DynamicType:$lhs, DynamicType:$rhs);
  let results = (outs DynamicType:$result);

  let description = [{
    Operation representing a reversible binary operator in python.
  }];

  let assemblyFormat = [{
    $lhs $binaryOperation $rhs attr-dict
  }];
}

def PylirHIR_BinExOp : CreateExceptionHandlingVariant<PylirHIR_BinOp>;

// Enums values kept in sync with 'PylirHIR_BinaryOperationAttr'.
def PylirHIR_BinaryAssignmentAttr : I32EnumAttr<"BinaryAssignment",
  "Binary assignment operations in python", [
  I32EnumAttrCase<"Add", 6, "__iadd__">,
  I32EnumAttrCase<"Sub", 7, "__isub__">,
  I32EnumAttrCase<"Or", 8, "__ior__">,
  I32EnumAttrCase<"Xor", 9, "__ixor__">,
  I32EnumAttrCase<"And", 10, "__iand__">,
  I32EnumAttrCase<"LShift", 11, "__ilshift__">,
  I32EnumAttrCase<"RShift", 12, "__irshift__">,
  I32EnumAttrCase<"Mul", 13, "__imul__">,
  I32EnumAttrCase<"Div", 14, "__idiv__">,
  I32EnumAttrCase<"FloorDiv", 15, "__ifloordiv__">,
  I32EnumAttrCase<"Mod", 16, "__imod__">,
  I32EnumAttrCase<"MatMul", 17, "__imatmul__">,
]> {
  let cppNamespace = "pylir::HIR";
}

def PylirHIR_BinAssignOp : PylirHIR_Op<"binAssignOp",
  [AddableExceptionHandling<"BinAssignExOp">]> {
  let arguments = (ins PylirHIR_BinaryAssignmentAttr:$binaryAssignment,
                       DynamicType:$lhs, DynamicType:$rhs);
  let results = (outs DynamicType:$result);

  let description = [{
    Operation representing compound assignment operators in python.
  }];

  let assemblyFormat = [{
    $lhs $binaryAssignment $rhs attr-dict
  }];
}

def PylirHIR_BinAssignExOp
  : CreateExceptionHandlingVariant<PylirHIR_BinAssignOp>;

//===----------------------------------------------------------------------===//
// Call Operations
//===----------------------------------------------------------------------===//

def PylirHIR_CallOp : PylirHIR_Op<"call",
  [AddableExceptionHandling<"CallExOp", ["Single", "Variadic"]>]> {
  let arguments = (ins DynamicType:$callable,
    Variadic<DynamicType>:$arguments,
    DefaultValuedAttr<StrArrayAttr, "{}">:$keywords,
    DefaultValuedAttr<DenseI32ArrayAttr, "{}">:$kind_internal
  );

  let results = (outs DynamicType:$result);

  let description = [{
    Operation representing a call operation in Python.

    TODO: Explain call resolution.

    Syntax:
    ```
    arg ::= [`*` | `**` | string-attr `=`] value-use
    global_func ::= `pyHIR.call` value-use `(` [ <arg> { `,` <arg> } ] `)`
    ```
  }];

  let assemblyFormat = [{
    $callable `(` (`)`) : ( ``
      custom<CallArguments>($keywords,
                            $arguments,
                            $kind_internal)^ `)` )? attr-dict
  }];

  let builders = [
    OpBuilder<(ins "mlir::Value":$callable,
                   "llvm::ArrayRef<CallArgument>":$arguments)>,
    OpBuilder<(ins "mlir::Value":$callable,
                   CArg<"mlir::ValueRange", "std::nullopt">:$posArguments)>
  ];

  let skipDefaultBuilders = 1;

  let hasVerifier = 1;

  let extraClassDeclaration = [{
  private:
    friend mlir::ParseResult parseCallArguments(mlir::OpAsmParser& parser,
      mlir::ArrayAttr& keywords,
      llvm::SmallVectorImpl<mlir::OpAsmParser::UnresolvedOperand>& arguments,
      mlir::DenseI32ArrayAttr& kindInternal);

    enum Kind {
      Positional = 1,
      PosExpansion = 2,
      MapExpansion = 3,
      Keyword = 0, // zero and any kind of negative value in reality.
    };

  public:

    /// Returns true if the argument at the given index is a positional
    /// expansion.
    bool isPosExpansion(std::size_t index) {
      return getKindInternal()[index] == Kind::PosExpansion;
    }

    /// Returns true if the argument at the given index is a map expansion.
    bool isMapExpansion(std::size_t index) {
      return getKindInternal()[index] == Kind::MapExpansion;
    }

    /// Returns the keyword of the argument at the given index.
    /// Returns a null attribute if the argument is not a keyword argument.
    mlir::StringAttr getKeyword(std::size_t index) {
      if (getKindInternal()[index] > 0)
        return nullptr;
      return mlir::cast<mlir::StringAttr>(getKeywords()[-getKindInternal()[index]]);
    }
  }];
}

def PylirHIR_CallExOp : CreateExceptionHandlingVariant<PylirHIR_CallOp>;

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
    arg ::= (`*` | `**`) value-def [[`only`] string-attr] [`has_default`] [ dictionary-attr ]
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

  let arguments = (ins StrAttr:$name, Variadic<DynamicType>:$default_values,
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
  OpAsmOpInterface, Symbol]> {

  let arguments = (ins StrAttr:$sym_name);

  let regions = (region MinSizedRegion<1>:$body);

  let assemblyFormat = "$sym_name attr-dict-with-keyword $body";

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

def PylirHIR_InitModuleOp : PylirHIR_Op<"initModule",
  [AddableExceptionHandling<"InitModuleExOp">,
   DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
  let arguments = (ins FlatSymbolRefAttr:$module);

  let description = [{
    Op used to call an `pyHIR.init` operation by executing its body.
    `$module` must be a reference to a `pyHIR.init` operation with the given name.
    As a special case, it is not possible to initialize the `__main__` module.
  }];

  let assemblyFormat = "$module attr-dict";

  let hasVerifier = 1;
}

def PylirHIR_InitModuleExOp
  : CreateExceptionHandlingVariant<PylirHIR_InitModuleOp> {
  let hasVerifier = 1;
}

#endif
