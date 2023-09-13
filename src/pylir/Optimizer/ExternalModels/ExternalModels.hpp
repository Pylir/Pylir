//  Licensed under the Apache License v2.0 with LLVM Exceptions.
//  See https://llvm.org/LICENSE.txt for license information.
//  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <mlir/IR/DialectRegistry.h>

namespace pylir {
/// Function used to register implementations of "Generic interfaces"
/// (interfaces that do not code or semantics wise belong to a dialect), for
/// external dialects (i.e. upstream MLIR dialects). The mechanism to do so is
/// to add an extension within the dialect registry to add a callback when a
/// dialect used by the compiler is loaded and then add external models for
/// interfaces to dialect objects there.
void registerExternalModels(mlir::DialectRegistry& registry);
} // namespace pylir
