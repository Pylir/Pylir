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
    /// Returns the default value of the parameter with the given index or
    /// nullptr if the parameter doesn't have a default or the default is
    /// unknown.
    mlir::Value getDefaultValue(unsigned index) {
      std::optional<std::size_t> position = getDefaultValuePosition(index);
      if (!position)
        return nullptr;
      mlir::ValueRange defaultValues = $_op.getDefaultValues();
      if (defaultValues.empty())
        return nullptr;
      return defaultValues[*position];
    }

    /// Returns an index into 'getDefaultValuesMapping' corresponding to the
    /// default value of the parameter with the given index.
    /// If the parameter does not have a default value an empty optional is
    /// returned.
    std::optional<std::size_t> getDefaultValuePosition(unsigned index) {
      auto mapping = $_op.getDefaultValuesMapping();
      auto iter = std::lower_bound(mapping.begin(), mapping.end(), index);
      if (iter == mapping.end() || *iter != index)
        return std::nullopt;
      return iter - mapping.begin();
    }

    /// Returns true if the parameter with the given index has a default value.
    bool hasDefault(unsigned index) {
      return getDefaultValuePosition(index).has_value();
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

def PylirHIR_CallInterface : OpInterface<"CallInterface"> {
  let cppNamespace = "::pylir::HIR";

  let methods = [
    InterfaceMethod<[{

    }],
    "::llvm::ArrayRef<int32_t>", "getKindInternal">,
    InterfaceMethod<[{

    }],
    "::mlir::ArrayAttr", "getKeywords">,
  ];

  let extraSharedClassDeclaration = [{

    enum Kind {
      Positional = 1,
      PosExpansion = 2,
      MapExpansion = 3,
      Keyword = 0, // zero and any kind of negative value in reality.
    };

    /// Returns true if the argument at the given index is a positional
    /// expansion.
    bool isPosExpansion(std::size_t index) {
      return $_op.getKindInternal()[index] == Kind::PosExpansion;
    }

    /// Returns true if the argument at the given index is a map expansion.
    bool isMapExpansion(std::size_t index) {
      return $_op.getKindInternal()[index] == Kind::MapExpansion;
    }

    /// Returns the keyword of the argument at the given index.
    /// Returns a null attribute if the argument is not a keyword argument.
    mlir::StringAttr getKeyword(std::size_t index) {
      if ($_op.getKindInternal()[index] > 0)
        return nullptr;
      return mlir::cast<mlir::StringAttr>(
        $_op.getKeywords()[-$_op.getKindInternal()[index]]);
    }
  }];
}


#endif
