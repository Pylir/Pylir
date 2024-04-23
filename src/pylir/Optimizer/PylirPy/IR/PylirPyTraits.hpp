//  Licensed under the Apache License v2.0 with LLVM Exceptions.
//  See https://llvm.org/LICENSE.txt for license information.
//  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <mlir/IR/Matchers.h>
#include <mlir/IR/OpDefinition.h>

#include <pylir/Interfaces/Builtins.hpp>
#include <pylir/Optimizer/Interfaces/CaptureInterface.hpp>
#include <pylir/Optimizer/Interfaces/MemoryFoldInterface.hpp>
#include <pylir/Optimizer/PylirPy/Interfaces/ExceptionHandlingInterface.hpp>
#include <pylir/Optimizer/PylirPy/Interfaces/KnownTypeObjectInterface.hpp>

#include <algorithm>

#include "PylirPyTypes.hpp"

//===----------------------------------------------------------------------===//
// Implementation classes of ODS traits. See PylirPyTraits.td for descriptions.
//===----------------------------------------------------------------------===//

namespace pylir::Py {

template <class ConcreteType>
class AlwaysBound : public mlir::OpTrait::TraitBase<ConcreteType, AlwaysBound> {
  static mlir::LogicalResult verifyTrait(mlir::Operation*) {
    static_assert(
        !ConcreteType::template hasTrait<mlir::OpTrait::ZeroOperands>(),
        "'Always Bound' trait is ony applicable to ops with results");
    return mlir::success();
  }
};

/// Trait used to mark the entry blockarguments of all regions of an op as
/// always being bound.
template <class ConcreteType>
class EntryArgsBound
    : public mlir::OpTrait::TraitBase<ConcreteType, EntryArgsBound> {
  static mlir::LogicalResult verifyTrait(mlir::Operation*) {
    static_assert(
        !ConcreteType::template hasTrait<mlir::OpTrait::ZeroRegions>(),
        "'EntryArgsBound' trait is ony applicable to ops with regions");
    return mlir::success();
  }
};

template <class ConcreteType>
class ReturnsImmutable
    : public mlir::OpTrait::TraitBase<ConcreteType, ReturnsImmutable> {
  static mlir::LogicalResult verifyTrait(mlir::Operation*) {
    static_assert(
        !ConcreteType::template hasTrait<mlir::OpTrait::ZeroOperands>(),
        "'ReturnsImmutable' trait is ony applicable to ops with results");
    return mlir::success();
  }
};

enum class OperandShape {
  Single,
  Variadic,
};

namespace details {
mlir::Operation* cloneWithExceptionHandlingImpl(
    mlir::OpBuilder& builder, mlir::Operation* operation,
    const mlir::OperationName& invokeVersion, ::mlir::Block* happyPath,
    mlir::Block* exceptionPath, mlir::ValueRange unwindOperands,
    llvm::StringRef attrSizedSegmentName, llvm::ArrayRef<OperandShape> shape);
} // namespace details

template <class InvokeVersion, OperandShape... shape>
struct AddableExceptionHandling {
  template <class ConcreteType>
  class Impl : public AddableExceptionHandlingInterface::Trait<ConcreteType> {
    template <unsigned n>
    constexpr static std::optional<unsigned> checkNOperands() {
      if constexpr (n == 1)
        return std::nullopt;
      else if constexpr (ConcreteType::template hasTrait<
                             mlir::OpTrait::NOperands<n>::template Impl>())
        return n;
      else
        return checkNOperands<n - 1>();
    }

    constexpr static std::optional<unsigned> tryDeduceShape() {
      if constexpr (ConcreteType::template hasTrait<
                        mlir::OpTrait::VariadicOperands>())
        return {};
      else if constexpr (ConcreteType::template hasTrait<
                             mlir::OpTrait::ZeroOperands>())
        return 0;
      else if constexpr (ConcreteType::template hasTrait<
                             mlir::OpTrait::OneOperand>())
        return 1;
      else
        return checkNOperands<5>();
    }

  public:
    mlir::Operation* cloneWithExceptionHandling(
        mlir::OpBuilder& builder, ::mlir::Block* happyPath,
        mlir::Block* exceptionPath, mlir::ValueRange unwindOperands) {
      constexpr auto deduced = tryDeduceShape();
      static_assert(ConcreteType::template hasTrait<
                        mlir::OpTrait::AttrSizedOperandSegments>() ||
                        ConcreteType::template hasTrait<
                            mlir::OpTrait::VariadicOperands>() ||
                        deduced.has_value() || sizeof...(shape) > 0,
                    "Could not deduce shape of the operations operands, nor "
                    "was it explicitly specified");
      constexpr auto shapeSize = deduced.has_value()
                                     ? *deduced
                                     : (ConcreteType::template hasTrait<
                                            mlir::OpTrait::VariadicOperands>()
                                            ? 1
                                            : sizeof...(shape));
      std::array<OperandShape, shapeSize> result{};
      if constexpr (ConcreteType::template hasTrait<
                        mlir::OpTrait::VariadicOperands>()) {
        result[0] = OperandShape::Variadic;
      } else if constexpr (deduced.has_value()) {
        std::fill(result.begin(), result.end(), OperandShape::Single);
      } else {
        auto initList = {shape...};
        std::copy(initList.begin(), initList.end(), result.begin());
      }
      return details::cloneWithExceptionHandlingImpl(
          builder, this->getOperation(),
          mlir::OperationName(InvokeVersion::getOperationName(),
                              this->getOperation()->getContext()),
          happyPath, exceptionPath, unwindOperands,
          mlir::OpTrait::AttrSizedOperandSegments<
              InvokeVersion>::getOperandSegmentSizeAttr(),
          result);
    }

    /// Invoke version of this operation.
    using InvokeOpT = InvokeVersion;
  };
};

} // namespace pylir::Py
