// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <mlir/IR/Builders.h>
#include <mlir/IR/DialectInterface.h>
#include <mlir/IR/Value.h>

namespace pylir {
/// Dialect interface called by optimization passes to materialize an undefined
/// value of a specific type.
///
/// Note: This is more or less a stop gap solution, that is too generic and also
/// requires too much effort to implement
///       for each type/dialect. Once a UB dialect as is proposed in
///       https://discourse.llvm.org/t/rfc-poison-semantics-for-mlir/66245 is in
///       MLIR transformation should make use of it instead.
class DialectUndefInterface
    : public mlir::DialectInterface::Base<DialectUndefInterface> {
public:
  explicit DialectUndefInterface(mlir::Dialect* dialect) : Base(dialect) {}

  /// Materializes an undefined value of 'type' with the location 'loc'.
  /// 'builder' should be used to create any operations and is already set with
  /// a suitable insertion point.
  virtual mlir::Value materializeUndefined(mlir::OpBuilder& builder,
                                           mlir::Type type,
                                           mlir::Location loc) const = 0;
};
} // namespace pylir
