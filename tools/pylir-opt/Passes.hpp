// Copyright 2022 Markus Böck
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/FunctionInterfaces.h>
#include <mlir/Pass/Pass.h>

namespace pylir::MemSSA
{
class MemorySSADialect;
}

namespace pylir::TypeFlow
{
class TypeFlowDialect;
}

namespace pylir::Py
{
class PylirPyDialect;
}

namespace
{
#define GEN_PASS_CLASSES
#include "Passes.h.inc"
} // namespace
