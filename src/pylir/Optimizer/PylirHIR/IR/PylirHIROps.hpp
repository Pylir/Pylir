// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <mlir/Bytecode/BytecodeOpInterface.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/OpImplementation.h>
#include <mlir/IR/SymbolTable.h>
#include <mlir/Interfaces/CallInterfaces.h>
#include <mlir/Interfaces/ControlFlowInterfaces.h>
#include <mlir/Interfaces/FunctionInterfaces.h>
#include <mlir/Interfaces/InferTypeOpInterface.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>

#include <llvm/ADT/iterator.h>

#include <pylir/Optimizer/PylirPy/IR/PylirPyTraits.hpp>
#include <pylir/Optimizer/PylirPy/IR/PylirPyTypes.hpp>
#include <pylir/Optimizer/PylirPy/Interfaces/ExceptionHandlingInterface.hpp>

#include "PylirHIRAttributes.hpp"
#include "PylirHIRDialect.hpp"

// clang-format off
// Derived interfaces may depend on the other interfaces.
#include "pylir/Optimizer/PylirHIR/IR/PylirHIRInterfaces.h.inc"
#include "pylir/Optimizer/PylirHIR/IR/PylirHIRDerivedInterfaces.h.inc"
// clang-format on

namespace pylir::HIR {

/// Function parameter class used to compactly represent a single python
/// parameter and its properties within a function in the pyHIR dialect.
class FunctionParameter {
  mlir::Value m_parameter;
  mlir::StringAttr m_name;
  mlir::DictionaryAttr m_attrs;
  mlir::Value m_defaultValue;
  bool m_isPosRest;
  bool m_isKeywordRest;
  bool m_isKeywordOnly;
  bool m_hasDefault;

public:
  /// Constructor creating a function parameter. 'parameter' is the block
  /// argument representing the parameter within the functions body.
  /// 'optionalName' is the name of the parameter for parameters which may be
  /// specified as a keyword. 'attrs' is the dictionary attribute for any
  /// argument attributes. 'optionalDefaultValue' is the default value for that
  /// parameter, used if no value has been specified for it in a call.
  /// 'isPosRest' is true if this parameter receives any leftover positional
  /// parameters. 'isKeywordRest' is true if this parameter receives any
  /// leftover mapping parameters.
  FunctionParameter(mlir::Value parameter, mlir::StringAttr optionalName,
                    mlir::DictionaryAttr attrs,
                    mlir::Value optionalDefaultValue, bool isPosRest,
                    bool isKeywordRest, bool isKeywordOnly, bool hasDefault);

  /// Returns the parameter value, used within the function body.
  mlir::Value getParameter() const {
    return m_parameter;
  }

  /// Returns the argument attributes of this attribute.
  mlir::DictionaryAttr getAttrs() const {
    return m_attrs;
  }

  /// Returns the default value or null if this parameter has either no default
  /// value or the default value is unknown. The latter is the case for any
  /// 'globalFunc'.
  mlir::Value getDefaultValue() const {
    return m_defaultValue;
  }

  /// Returns true if the parameter has a default value.
  bool hasDefault() const {
    return m_hasDefault;
  }

  /// Returns the name of this parameter used for parameters callable as a
  /// keyword, or null if this parameter does not have a name.
  mlir::StringAttr getName() const {
    return m_name;
  }

  /// Returns true if the parameter is a positional-only parameter.
  bool isPositionalOnly() const {
    return !getName();
  }

  /// Returns true if this parameter receives any leftover positional
  /// parameters.
  bool isPosRest() const {
    return m_isPosRest;
  }

  /// Returns true if this parameter receives any leftover keyword parameters.
  bool isKeywordRest() const {
    return m_isKeywordRest;
  }

  /// Returns true if this parameter can only be assigned to via a keyword
  /// argument.
  bool isKeywordOnly() const {
    return m_isKeywordOnly;
  }
};

/// Range object used to easily iterate over all parameters of a 'pyHIR'
/// function.
class FunctionParameterRange
    : public llvm::indexed_accessor_range<
          FunctionParameterRange, FunctionInterface, FunctionParameter,
          FunctionParameter, FunctionParameter> {
  using Base =
      llvm::indexed_accessor_range<FunctionParameterRange, FunctionInterface,
                                   FunctionParameter, FunctionParameter,
                                   FunctionParameter>;

  friend Base;

  // dereference function required by indexed_accessor_range.
  static FunctionParameter dereference(FunctionInterface function,
                                       std::ptrdiff_t index);

public:
  /// Constructor for any function implementing HIR::FunctionInterface.
  explicit FunctionParameterRange(FunctionInterface function);

  using Base::Base;
};

/// Class used to specify the parameters of a function in the pyHIR dialect.
class FunctionParameterSpec {
  mlir::StringAttr m_name;
  mlir::Value m_defaultValue;
  bool m_isPosRest = false;
  bool m_isKeywordRest = false;
  bool m_isKeywordOnly = false;
  mlir::DictionaryAttr m_parameterAttributes;

public:
  struct PosRest {};
  struct KeywordRest {};

  /// Creates a parameter spec for a positional-only parameter with potentially
  /// a default argument.
  explicit FunctionParameterSpec(mlir::Value maybeDefaultValue = nullptr)
      : m_defaultValue(maybeDefaultValue) {}

  /// Creates a parameter spec for a named parameter with potentially a default
  /// argument. The parameter is a keyword-only parameter if 'keywordOnly' is
  /// true.
  explicit FunctionParameterSpec(mlir::StringAttr name,
                                 mlir::Value maybeDefaultValue,
                                 bool keywordOnly = false)
      : m_name(name), m_defaultValue(maybeDefaultValue),
        m_isKeywordOnly(keywordOnly) {}

  /// Creates a parameter receiving all leftover positional arguments.
  explicit FunctionParameterSpec(PosRest) : m_isPosRest(true) {}

  /// Creates a parameter receiving all leftover keyword arguments.
  explicit FunctionParameterSpec(KeywordRest) : m_isKeywordRest(true) {}

  /// Creates a parameter spec from an existing function parameter.
  explicit FunctionParameterSpec(const FunctionParameter& functionParameter);

  /// Returns the name of the parameter or null if it is a positional-only
  /// parameter.
  mlir::StringAttr getName() const {
    return m_name;
  }

  /// Returns the default value or null if it has no default value.
  mlir::Value getDefaultValue() const {
    return m_defaultValue;
  }

  /// Returns true if this parameter is positional-only.
  bool isPosOnly() const {
    return m_name != nullptr;
  }

  /// Returns true if this parameter receives all leftover positional arguments.
  bool isPosRest() const {
    return m_isPosRest;
  }

  /// Returns true if this parameter receives all leftover keyword arguments.
  bool isKeywordRest() const {
    return m_isKeywordRest;
  }

  /// Returns true if this parameter can only be called as keyword argument.
  bool isKeywordOnly() const {
    return m_isKeywordOnly;
  }

  /// Returns the attributes of this parameter.
  mlir::DictionaryAttr getParameterAttributes() const {
    return m_parameterAttributes;
  }

  /// Sets the attributes of this parameter.
  void setParameterAttributes(mlir::DictionaryAttr parameterAttributes) {
    m_parameterAttributes = parameterAttributes;
  }
};

/// Struct representing an argument of a call operation.
struct CallArgument {
  struct PositionalTag {};
  struct PosExpansionTag {};
  struct MapExpansionTag {};

  mlir::Value value;
  std::variant<PositionalTag, PosExpansionTag, MapExpansionTag,
               mlir::StringAttr>
      kind;
};

} // namespace pylir::HIR

#define GET_OP_CLASSES
#include "pylir/Optimizer/PylirHIR/IR/PylirHIROps.h.inc"

namespace pylir::HIR {

/// Range adaptor allowing easy iteration over the arguments of a call op.
template <class OpT>
class CallArgumentRange
    : public llvm::indexed_accessor_range<CallArgumentRange<OpT>, OpT,
                                          CallArgument, CallArgument,
                                          CallArgument> {
  using Base =
      llvm::indexed_accessor_range<CallArgumentRange, OpT, CallArgument,
                                   CallArgument, CallArgument>;

  friend Base;

  // dereference function required by indexed_accessor_range.
  static CallArgument dereference(OpT call, std::ptrdiff_t index) {
    mlir::Value value = call.getArguments()[index];
    if (call.isPosExpansion(index))
      return CallArgument{value, CallArgument::PosExpansionTag{}};
    if (call.isMapExpansion(index))
      return CallArgument{value, CallArgument::MapExpansionTag{}};
    if (mlir::StringAttr keyword = call.getKeyword(index))
      return CallArgument{value, keyword};

    return CallArgument{value, CallArgument::PositionalTag{}};
  }

public:
  explicit CallArgumentRange(OpT call)
      : Base(call, 0, call.getArguments().size()) {}
};

} // namespace pylir::HIR
