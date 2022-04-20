// Copyright 2022 Markus Böck
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <mlir/Pass/Pass.h>

#include "Passes.hpp"

namespace pylir::Py
{
class PylirPyDialect;
}

namespace pylir::Mem
{
class PylirMemDialect;
}

namespace mlir::LLVM
{
class LLVMDialect;
}

namespace mlir::arith
{
class ArithmeticDialect;
}

namespace pylir
{
#define GEN_PASS_CLASSES
#include <pylir/Optimizer/Conversion/Passes.h.inc>
} // namespace pylir
