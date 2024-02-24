// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/OwningOpRef.h>

#include <pylir/CodeGen/CodeGenOptions.hpp>
#include <pylir/Parser/Syntax.hpp>

namespace pylir {

mlir::OwningOpRef<mlir::ModuleOp>
codegenNew(mlir::MLIRContext* context, const Syntax::FileInput& input,
           Diag::DiagnosticsDocManager<>& docManager, CodeGenOptions options);
} // namespace pylir
