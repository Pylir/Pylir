//  Licensed under the Apache License v2.0 with LLVM Exceptions.
//  See https://llvm.org/LICENSE.txt for license information.
//  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/OwningOpRef.h>

namespace pylir {
/// Link together a list of modules into a single module.
/// This function is currently only implements the minimum amount of work needed
/// for modules produced by the compiler. It makes no effort of any error
/// handling, instead assuming none can occur.
///
/// Current fix-ups include:
/// * Removing duplicate declarations of top level symbols
/// * Replacing declarations of top level symbols with their definitions
///
/// No logic exists to merge declarations (if they had differing properties). No
/// logic exists to handle symbol name collision of definitions from other
/// modules, even ones with private symbol visibility! It currently assumes all
/// symbols input have a unique name.
///
/// Input modules are left in an unspecified state.
mlir::OwningOpRef<mlir::ModuleOp>
linkModules(llvm::MutableArrayRef<mlir::OwningOpRef<mlir::ModuleOp>> modules);
} // namespace pylir
