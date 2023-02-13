//  Licensed under the Apache License v2.0 with LLVM Exceptions.
//  See https://llvm.org/LICENSE.txt for license information.
//  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <mlir/Interfaces/CallInterfaces.h>

namespace pylir::Py
{

/// Simple struct returning all ops that were inlined from a use of 'inlineCall'.
struct InlinedOps
{
    /// The first operation of the callable that was inlined.
    mlir::Operation* firstOperationInFirstBlock{};
    /// End iterator of the range of blocks that were inlined.
    /// In other words, one past the last block that was inlined.
    mlir::Region::iterator endBlock{};
};

/// Inlines 'callable' into 'call' and replaces it. 'call' is assumed to resolve to 'callable' and have matching
/// argument count and type as the callable region of 'callable'.
/// Returns a struct allowing iteration of the inlined ops with 'call's parent region.
///
/// Note: This is required over MLIRs version due to non-trivial deficiencies in MLIRs inliner interface
/// (that we should probably fix). In particular, this implementation properly handles inlining of exception handling
/// ops.
InlinedOps inlineCall(mlir::CallOpInterface call, mlir::CallableOpInterface callable);
} // namespace pylir::Py
