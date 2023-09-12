// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef PYLIR_HIR_INTERFACES_FUNCTION_INTERFACE
#define PYLIR_HIR_INTERFACES_FUNCTION_INTERFACE

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/CallInterfaces.td"

def PylirHIR_FunctionInterface : OpInterface<"FunctionInterface",
  [CallableOpInterface]> {
  let cppNamespace = "::pylir::HIR";

  let methods = [
    InterfaceMethod<[{
      Returns the type of the function.
    }],
    "::mlir::Type", "getFunctionType">,
    InterfaceMethod<[{
      Returns the values used as default values for the parameters.
      See `getDefaultValuesMapping` to get a mapping of which default value
      corresponds to which parameter.
    }],
    "::mlir::ValueRange", "getDefaultValues">,
    InterfaceMethod<[{
      Returns the mapping for `getDefaultValues`.
      Every nth slot contains the index of the parameter that the nth default
      value applies to.
    }],
    "::llvm::ArrayRef<std::int32_t>", "getDefaultValuesMapping">,
    InterfaceMethod<[{
      Returns the parameter names used for the named argument syntax.
      See `getParameterNames` to get a mapping of which parameter name
      corresponds to which parameter.
    }],
    "::mlir::ArrayAttr", "getParameterNames">,
    InterfaceMethod<[{
      Returns the mapping for `getParameterNames`.
      Every nth slot contains the index of the parameter that the nth parameter
      name applies to.
    }],
    "::llvm::ArrayRef<std::int32_t>", "getParameterNameMapping">,
    InterfaceMethod<[{
      Returns a sorted range of indices of all parameters which may only be
      called by keyword.
    }],
    "::llvm::ArrayRef<std::int32_t>", "getKeywordOnlyMapping">,
    InterfaceMethod<[{
      Returns the index of the parameter which receives any leftover positional
      arguments or an empty optional if no such parameter exists.
    }],
    "std::optional<std::uint32_t>", "getPosRest">,
    InterfaceMethod<[{
      Returns the index of the parameter which receives any leftover keyword
      arguments or an empty optional if no such parameter exists.
    }],
    "std::optional<std::uint32_t>", "getKeywordRest">
  ];

  let extraSharedClassDeclaration = [{
    mlir::Value getDefaultValue(unsigned index) {
      auto mapping = $_op.getDefaultValuesMapping();
      auto iter = std::lower_bound(mapping.begin(), mapping.end(), index);
      if (iter == mapping.end() || *iter != index)
        return nullptr;
      return $_op.getDefaultValues()[iter - mapping.begin()];
    }

    mlir::StringAttr getParameterName(unsigned index) {
      auto mapping = $_op.getParameterNameMapping();
      auto iter = std::lower_bound(mapping.begin(), mapping.end(), index);
      if (iter == mapping.end() || *iter != index)
        return nullptr;
      return mlir::cast<mlir::StringAttr>($_op.getParameterNames()[iter
        - mapping.begin()]);
    }

    bool isKeywordOnly(unsigned index) {
      auto mapping = $_op.getKeywordOnlyMapping();
      return std::binary_search(mapping.begin(), mapping.end(), index);
    }
  }];
}

#endif
