// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <mlir/IR/Operation.h>
#include <mlir/IR/Value.h>

#include <llvm/ADT/DenseMap.h>

namespace pylir {
/// Analysis class for computing whether a value escapes.
/// The implementation makes use of memoization to avoid recomputing results and
/// to memoize partial results for the escape analysis computations of other
/// values.
///
/// We define escaping as the exact same SSA value created within a
/// 'IsolatedFromAbove' region being accessible through the result of another
/// operation within a 'IsolatedFromAbove' region. This matters very little for
/// SSA values representing things like integers, but is useful for SSA values
/// representing references to objects. If the SSA value of an object reference
/// is allocated in a 'IsolatedFromAbove' region and does not escape, we say it
/// is local to that region.
///
/// The definition of escaping is confusingly similar but not identical to
/// capturing a value. We define capturing as the exact same SSA value
/// potentially being accessible through the result of another operation,
/// including within the same 'IsolatedFromAbove' region, as opposed to
/// escaping, which only cares about the possible accessibility from other
/// 'IsolatedFromAbove' regions.
class EscapeAnalysis {
  llvm::DenseMap<mlir::Value, bool> m_results;

public:
  EscapeAnalysis() = default;

  /// Dummy constructor with a signature required by MLIRs AnalysisManager.
  EscapeAnalysis(mlir::Operation*) : EscapeAnalysis() {}

  /// Returns false if the value does not escape its contained
  /// 'IsolatedFromAbove' region. If it definitely escapes or the implementation
  /// is not capable of further analysis, true is returned as a conservative
  /// value.
  bool escapes(mlir::Value value);
};
} // namespace pylir
