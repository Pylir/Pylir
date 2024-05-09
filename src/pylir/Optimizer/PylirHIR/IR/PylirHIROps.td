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
include "pylir/Optimizer/PylirHIR/IR/PylirHIREnums.td"
include "pylir/Optimizer/PylirHIR/IR/PylirHIRFunctionInterface.td"

include "pylir/Optimizer/PylirPy/IR/PylirPyAttributes.td"
include "pylir/Optimizer/PylirPy/IR/PylirPyTypes.td"
include "pylir/Optimizer/PylirPy/Interfaces/ExceptionHandlingInterface.td"

class PylirHIR_Op<string mneomic, list<Trait> traits = []>
  : Op<PylirHIR_Dialect, mneomic, traits>;

//===----------------------------------------------------------------------===//
// Basic Operations
//===----------------------------------------------------------------------===//

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

def PylirHIR_ContainsOp : PylirHIR_Op<"contains",
  [AddableExceptionHandling<"ContainsExOp">]> {
  let arguments = (ins DynamicType:$container, DynamicType:$item);
  let results = (outs DynamicType:$result);

  let description = [{
    Performs a membership check whether `item` is within `container`.

    This is performed in the following steps:
    * If the type of `container` implements `__contains__`, calls the method
      with `item`.
      Returns true if it returns a truthy value.
    * Otherwise, if the type of `container` implements `__iter__`, it iterates
      over `container` and checks whether for any element `z`,
      `z is item or z == item` is true.
      Any exceptions raised in the process are propagated as is.

    :::{admonition} TODO
    :class: note

    * Describe the behaviour of types implementing `__getitem__` in detail.
    * Implement correctly in lowering: Currently only performs `__contains__`
      check.
    :::

    Reference: https://docs.python.org/3.9/reference/expressions.html#membership-test-operations
  }];

  let assemblyFormat = [{
    $item `in` $container attr-dict
  }];
}

def PylirHIR_ContainsExOp
  : CreateExceptionHandlingVariant<PylirHIR_ContainsOp>;

//===----------------------------------------------------------------------===//
// Subscription Operations
//===----------------------------------------------------------------------===//

def PylirHIR_GetItemOp : PylirHIR_Op<"getItem",
  [AddableExceptionHandling<"GetItemExOp">]> {
  let arguments = (ins DynamicType:$object, DynamicType:$index);
  let results = (outs DynamicType:$result);

  let assemblyFormat = [{
    $object `[` $index `]` attr-dict
  }];
}

def PylirHIR_GetItemExOp
  : CreateExceptionHandlingVariant<PylirHIR_GetItemOp>;

def PylirHIR_SetItemOp : PylirHIR_Op<"setItem",
  [AddableExceptionHandling<"SetItemExOp">]> {
  let arguments = (ins DynamicType:$object,
                       DynamicType:$index,
                       DynamicType:$value);
  let results = (outs DynamicType:$result);

  let assemblyFormat = [{
    $object `[` $index `]` `to` $value attr-dict
  }];
}

def PylirHIR_SetItemExOp
  : CreateExceptionHandlingVariant<PylirHIR_SetItemOp>;

def PylirHIR_DelItemOp : PylirHIR_Op<"delItem",
  [AddableExceptionHandling<"DelItemExOp">]> {
  let arguments = (ins DynamicType:$object, DynamicType:$index);
  let results = (outs DynamicType:$result);

  let assemblyFormat = [{
    $object `[` $index `]` attr-dict
  }];
}

def PylirHIR_DelItemExOp
  : CreateExceptionHandlingVariant<PylirHIR_DelItemOp>;

//===----------------------------------------------------------------------===//
// Attribute Operations
//===----------------------------------------------------------------------===//

def PylirHIR_GetAttributeOp : PylirHIR_Op<"getAttribute",
  [AddableExceptionHandling<"GetAttributeExOp">]> {
  let arguments = (ins DynamicType:$object, StrAttr:$attribute);
  let results = (outs DynamicType:$result);

  let assemblyFormat = [{
    $attribute `of` $object attr-dict
  }];
}

def PylirHIR_GetAttributeExOp
  : CreateExceptionHandlingVariant<PylirHIR_GetAttributeOp>;

def PylirHIR_SetAttrOp : PylirHIR_Op<"setAttr",
  [AddableExceptionHandling<"SetAttrExOp">]> {
  let arguments = (ins DynamicType:$object,
                       StrAttr:$attribute,
                       DynamicType:$value);

  let results = (outs DynamicType:$result);

  let assemblyFormat = [{
     $attribute `of` $object `to` $value attr-dict
  }];
}

def PylirHIR_SetAttrExOp
  : CreateExceptionHandlingVariant<PylirHIR_SetAttrOp>;

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
    CPred<"mlir::cast<::mlir::FunctionType>($_self).getNumResults() == 1">,
    SubstLeaves<"$_self", "mlir::cast<::mlir::FunctionType>($_self).getResult(0)",
      DynamicType.predicate>,
    Concat<"llvm::all_of(mlir::cast<::mlir::FunctionType>($_self).getInputs(), "
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
    PylirHIR_FunctionInterface, EntryArgsBound]> {

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
                   "llvm::ArrayRef<FunctionParameterSpec>":$parameters,
                   CArg<"mlir::ArrayAttr", "nullptr">:$resAttrs)>
  ];

  let description = [{

    Op representing a function in global scope with no use of non-local
    variables or similar.
    This is meant as a more restrictive and more optimized form of `pyHIR.func`.
    Unlike `pyHIR.func`, `pyHIR.globalFunc` must have a first parameter which
    acts as the closure parameter used to implement non-local variables after
    outlining.
    It may not have a default parameter nor a keyword with which it can be
    called.

    Syntax:
    ```
    arg ::= (`*` | `**`) value-def [[`only`] string-attr] [`has_default`] [ dictionary-attr ]
    global_func ::= `pyHIR.global_func` sym_name `(` [ <arg> { `,` <arg> } ] `)` [ `->` dictionary-attr ] attr-with-keyword
            `{` body `}`
    ```
  }];

  let extraClassDeclaration = [{

    mlir::Value getClosureParameter() {
      return getBody().getArguments().front();
    }

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

    void getAsmBlockArgumentNames(mlir::Region& region,
                                  mlir::OpAsmSetValueNameFn setNameFn) {
      setNameFn(getClosureParameter(), "closure");
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
  PylirHIR_FunctionInterface, EntryArgsBound]> {

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
  let arguments = (ins);

  let description = [{
    Op used to terminate a `pyHIR.init` body.
  }];

  let assemblyFormat = "attr-dict";
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

def PylirHIR_ModuleGetAttrOp : PylirHIR_Op<"module_getAttr"> {
  let arguments = (ins
    PylirPy_GlobalValueAttr:$module,
    StrAttr:$attr
  );

  let description = [{
    Op used to access an attribute of a module.
    If the attribute was not previously bound an unbound value is returned.
  }];

  let results = (outs DynamicType:$result);

  let assemblyFormat = [{
    $module `[` $attr `]` attr-dict
  }];
}

def PylirHIR_ModuleSetAttrOp : PylirHIR_Op<"module_setAttr"> {
  let arguments = (ins
    PylirPy_GlobalValueAttr:$module,
    StrAttr:$attr,
    DynamicType:$value
  );

  let description = [{
    Op used to bind a value to an attribute of a module.
  }];

  let assemblyFormat = [{
    $module `[` $attr `]` `to` $value attr-dict
  }];
}

#endif
