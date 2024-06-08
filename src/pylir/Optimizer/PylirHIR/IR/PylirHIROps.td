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
include "pylir/Optimizer/PylirHIR/IR/PylirHIRInterfaces.td"
include "pylir/Optimizer/PylirHIR/IR/PylirHIRDerivedInterfaces.td"

include "pylir/Optimizer/PylirPy/IR/PylirPyTypes.td"

class PylirHIR_Op<string mneomic, list<Trait> traits = []>
  : Op<PylirHIR_Dialect, mneomic, traits>;

/// Trait that should be added to operations that have an exception handling
/// version. 'name' must be the name of the op's C++ class.
///
/// Note: This trait must come last in the trait list due to interface
/// inheritance restrictions.
class PylirHIR_WithExceptionHandling<string name, list<string> shape = []>
  : TraitList<[
  OpInterfaceTrait<"pylir::HIR::" # name # "Interface">,
  AddableExceptionHandling<!substr(name, 0, !sub(!size(name), !size("Op")))
    # "ExOp", shape>,
]> {
  assert !eq(!substr(name, !sub(!size(name), !size("Op"))), "Op"),
    "code assumes all operations have a 'Op' suffix in their name";
}

/// Multiclass used to define the derived interface and exception handling
/// variant of 'op'.
multiclass PylirHIR_ExceptionHandlingVariant<Op op,
  // If 1, the interface will contain a 'verify' method that must be implemented
  // by the user. This should be used rather than the Op's verifier to share a
  // single verifier for both the normal and exception handling op.
  bit hasVerifier = 0,
  bit newline = 1> {

  def NAME : CreateExceptionHandlingVariant<op, "", newline>;

  def : DeriveInterface<op> {
    let extraClassDeclaration = !if(hasVerifier, [{
      mlir::LogicalResult verify();
    }], "");

    let verify = !if(hasVerifier, [{
      return mlir::cast<}] # cppNamespace # "::" # cppInterfaceName # [{>
        ($_op).verify();
    }], "");
  }
}

//===----------------------------------------------------------------------===//
// Basic Operations
//===----------------------------------------------------------------------===//

def PylirHIR_BinOp : PylirHIR_Op<"binOp",
  [PylirHIR_WithExceptionHandling<"BinOp">]> {
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

defm PylirHIR_BinExOp : PylirHIR_ExceptionHandlingVariant<PylirHIR_BinOp>;

def PylirHIR_BinAssignOp : PylirHIR_Op<"binAssignOp",
  [PylirHIR_WithExceptionHandling<"BinAssignOp">]> {
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

defm PylirHIR_BinAssignExOp
  : PylirHIR_ExceptionHandlingVariant<PylirHIR_BinAssignOp>;

def PylirHIR_ContainsOp : PylirHIR_Op<"contains",
  [PylirHIR_WithExceptionHandling<"ContainsOp">]> {
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

defm PylirHIR_ContainsExOp
  : PylirHIR_ExceptionHandlingVariant<PylirHIR_ContainsOp>;

//===----------------------------------------------------------------------===//
// Subscription Operations
//===----------------------------------------------------------------------===//

def PylirHIR_GetItemOp : PylirHIR_Op<"getItem",
  [PylirHIR_WithExceptionHandling<"GetItemOp">]> {
  let arguments = (ins DynamicType:$object, DynamicType:$index);
  let results = (outs DynamicType:$result);

  let assemblyFormat = [{
    $object `[` $index `]` attr-dict
  }];
}

defm PylirHIR_GetItemExOp
  : PylirHIR_ExceptionHandlingVariant<PylirHIR_GetItemOp>;

def PylirHIR_SetItemOp : PylirHIR_Op<"setItem",
  [PylirHIR_WithExceptionHandling<"SetItemOp">]> {
  let arguments = (ins DynamicType:$object,
                       DynamicType:$index,
                       DynamicType:$value);
  let results = (outs DynamicType:$result);

  let assemblyFormat = [{
    $object `[` $index `]` `to` $value attr-dict
  }];
}

defm PylirHIR_SetItemExOp
  : PylirHIR_ExceptionHandlingVariant<PylirHIR_SetItemOp>;

def PylirHIR_DelItemOp : PylirHIR_Op<"delItem",
  [PylirHIR_WithExceptionHandling<"DelItemOp">]> {
  let arguments = (ins DynamicType:$object, DynamicType:$index);
  let results = (outs DynamicType:$result);

  let assemblyFormat = [{
    $object `[` $index `]` attr-dict
  }];
}

defm PylirHIR_DelItemExOp
  : PylirHIR_ExceptionHandlingVariant<PylirHIR_DelItemOp>;

//===----------------------------------------------------------------------===//
// Attribute Operations
//===----------------------------------------------------------------------===//

def PylirHIR_GetAttributeOp : PylirHIR_Op<"getAttribute",
  [PylirHIR_WithExceptionHandling<"GetAttributeOp">]> {
  let arguments = (ins DynamicType:$object, StrAttr:$attribute);
  let results = (outs DynamicType:$result);

  let assemblyFormat = [{
    $attribute `of` $object attr-dict
  }];
}

defm PylirHIR_GetAttributeExOp
  : PylirHIR_ExceptionHandlingVariant<PylirHIR_GetAttributeOp>;

def PylirHIR_SetAttrOp : PylirHIR_Op<"setAttr",
  [PylirHIR_WithExceptionHandling<"SetAttrOp">]> {
  let arguments = (ins DynamicType:$object,
                       StrAttr:$attribute,
                       DynamicType:$value);

  let results = (outs DynamicType:$result);

  let assemblyFormat = [{
     $attribute `of` $object `to` $value attr-dict
  }];
}

defm PylirHIR_SetAttrExOp
  : PylirHIR_ExceptionHandlingVariant<PylirHIR_SetAttrOp>;

//===----------------------------------------------------------------------===//
// Call Operations
//===----------------------------------------------------------------------===//

def PylirHIR_CallOp : PylirHIR_Op<"call",
  [PylirHIR_CallInterface,
   PylirHIR_WithExceptionHandling<"CallOp", ["Single", "Variadic"]>]> {
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
}

defm PylirHIR_CallExOp : PylirHIR_ExceptionHandlingVariant<PylirHIR_CallOp,
  /*hasVerifier=*/1>;

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
// Classes
//===----------------------------------------------------------------------===//

def PylirHIR_ClassOp : PylirHIR_Op<"class", [
    OpAsmOpInterface, PylirHIR_CallInterface,
    PylirHIR_WithExceptionHandling<"ClassOp">,
]> {
  let arguments = (ins
    StrAttr:$name,
    Variadic<DynamicType>:$arguments,
    DefaultValuedAttr<StrArrayAttr, "{}">:$keywords,
    DefaultValuedAttr<DenseI32ArrayAttr, "{}">:$kind_internal
  );

  let results = (outs DynamicType:$result);

  let regions = (region MinSizedRegion<1>:$body);

  let assemblyFormat = [{
    $name ( `(`
      custom<CallArguments>($keywords,
                            $arguments,
                            $kind_internal)^ `)` )? attr-dict-with-keyword
    $body
  }];

  let builders = [
    OpBuilder<(ins "llvm::StringRef":$className,
                   "llvm::ArrayRef<CallArgument>":$arguments,
                   "llvm::function_ref<void(mlir::Block*)>":$bodyBuilder
    )>
  ];

  let extraClassDeclaration = [{

    mlir::BlockArgument getDictParameter() {
      return getBody().front().getArgument(0);
    }

    //===------------------------------------------------------------------===//
    // OpAsmOpInterface implementation
    //===------------------------------------------------------------------===//

    static llvm::StringRef getDefaultDialect() {
      return pylir::HIR::PylirHIRDialect::getDialectNamespace();
    }

    void getAsmBlockArgumentNames(mlir::Region& region,
                                  mlir::OpAsmSetValueNameFn setNameFn) {
      setNameFn(getDictParameter(), "dict");
    }
  }];
}

defm PylirHIR_ClassExOp : PylirHIR_ExceptionHandlingVariant<PylirHIR_ClassOp,
  /*hasVerifier=*/1, /*newline=*/0>;

def PylirHIR_ClassReturnOp : PylirHIR_Op<"class_return",
  [ParentOneOf<["ClassOp", "ClassExOp"]>, Terminator, ReturnLike, Pure]> {
  let arguments = (ins);

  let description = [{
    Op used to terminate a `pyHIR.class` body.
  }];

  let assemblyFormat = "attr-dict";
}

def PylirHIR_BuildClassOp : PylirHIR_Op<"buildClass",
  [PylirHIR_CallInterface, PylirHIR_WithExceptionHandling<"BuildClassOp",
    ["Single", "Variadic"]>]> {

  let description = [{
    Op corresponding to CPython's internal `__build_class__` function.
    Creates a new type instance with the given name and initializes its
    dictionary by calling `function` using a dictionary that must be modified
    in-place.

    The base-class list is derived from all positional arguments.
    The keyword arguments, with the exception of the key `"metaclass"`, are
    passed to various methods during class creation.

    :::{admonition} TODO
    :class: note

    * Describe the the class creation procedure in detail.
    :::
  }];

  let arguments = (ins
    DynamicType:$function,
    StrAttr:$name,
    Variadic<DynamicType>:$arguments,
    DefaultValuedAttr<StrArrayAttr, "{}">:$keywords,
    DefaultValuedAttr<DenseI32ArrayAttr, "{}">:$kind_internal
  );

  let results = (outs DynamicType:$result);

  let assemblyFormat = [{
     `(` $function `,` $name (`,`
      custom<CallArguments>($keywords,
                            $arguments,
                            $kind_internal)^ )? `)` attr-dict-with-keyword
  }];
}

defm PylirHIR_BuildClassExOp
  : PylirHIR_ExceptionHandlingVariant<PylirHIR_BuildClassOp>;

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
  [DeclareOpInterfaceMethods<SymbolUserOpInterface>,
   PylirHIR_WithExceptionHandling<"InitModuleOp">]> {
  let arguments = (ins FlatSymbolRefAttr:$module);

  let description = [{
    Op used to call an `pyHIR.init` operation by executing its body.
    `$module` must be a reference to a `pyHIR.init` operation with the given name.
    As a special case, it is not possible to initialize the `__main__` module.
  }];

  let assemblyFormat = "$module attr-dict";
}

defm PylirHIR_InitModuleExOp
  : PylirHIR_ExceptionHandlingVariant<PylirHIR_InitModuleOp,
    /*hasVerifier=*/1>;

#endif
