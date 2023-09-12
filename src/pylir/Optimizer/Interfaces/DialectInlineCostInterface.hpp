//  Licensed under the Apache License v2.0 with LLVM Exceptions.
//  See https://llvm.org/LICENSE.txt for license information.
//  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <mlir/IR/DialectInterface.h>

namespace pylir {
class DialectInlineCostInterface
    : public mlir::DialectInterface::Base<DialectInlineCostInterface> {
public:
  explicit DialectInlineCostInterface(mlir::Dialect* dialect) : Base(dialect) {}

  /// Return the inline cost of this dialects op in abstract units. Should only
  /// return the ops cost and not add up any contained regions, blocks or
  /// operations.
  virtual std::size_t getCost(mlir::Operation* op) const = 0;
};
} // namespace pylir
