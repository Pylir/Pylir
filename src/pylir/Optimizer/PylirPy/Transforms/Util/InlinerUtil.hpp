//  Licensed under the Apache License v2.0 with LLVM Exceptions.
//  See https://llvm.org/LICENSE.txt for license information.
//  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <mlir/IR/IRMapping.h>
#include <mlir/Interfaces/CallInterfaces.h>

namespace pylir::Py {

/// Inlines 'callable' into 'call' and replaces it. 'call' is assumed to resolve
/// to 'callable' and have matching argument count and type as the callable
/// region of 'callable'.
///
/// Returns a mapping of the source IR entities, to the corresponding created
/// clones. Note that during inlining, more transformations than simply cloning
/// may occur, making the map not perfect. In particular, new blocks created
/// through transformations will not be present, nor is the entry block from the
/// source region. A operation may also not map to an operation of the exact
/// same kind.
///
/// Note: This is required over MLIRs version due to non-trivial deficiencies in
/// MLIRs inliner interface (that we should probably fix). In particular, this
/// implementation properly handles inlining of exception handling ops.
mlir::IRMapping inlineCall(mlir::CallOpInterface call,
                           mlir::CallableOpInterface callable);
} // namespace pylir::Py
