//  Licensed under the Apache License v2.0 with LLVM Exceptions.
//  See https://llvm.org/LICENSE.txt for license information.
//  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <mlir/IR/DialectInterface.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Operation.h>

#include <pylir/Optimizer/Interfaces/DialectInlineCostInterface.hpp>

#include <cstddef>

namespace pylir {

/// Class for fetching the inline cost of an operation. This is a class and not
/// just a function as it has state, namely dialect interface implementations of
/// 'pylir::DialectInlineCostInterface', which would otherwise have to be
/// recomputed each time. See that interface to adjust cost of an operation.
class InlineCost {
  mlir::DialectInterfaceCollection<pylir::DialectInlineCostInterface>
      m_collection;

public:
  /// Constructs a 'InlineCost' instance and fetches all dialect interface
  /// implementations.
  explicit InlineCost(mlir::MLIRContext* context);

  /// Returns the inline cost of an operation in abstract units. If 'recurse' is
  /// true, recurses into any regions of the operation and adds up their cost in
  /// the result as well.
  std::size_t getCostOf(mlir::Operation* operation, bool recurse = false);
};

/// Convenience class for use by MLIRs AnalysisManager to cache inline cost
/// calculations of an schedulable op.
class InlineCostAnalysis {
  std::size_t m_cost{0};

public:
  /// Constructor with the operation whose size should be calculated.
  explicit InlineCostAnalysis(mlir::Operation* operation);

  /// Size of the operation that was used to construct this analysis.
  [[nodiscard]] std::size_t getCost() const {
    return m_cost;
  }
};
} // namespace pylir
