// Copyright 2022 Markus Böck
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <mlir/Pass/Pass.h>

#include <llvm/ADT/Triple.h>
#include <llvm/IR/DataLayout.h>

#include <memory>

namespace pylir
{
#define GEN_PASS_REGISTRATION
#define GEN_PASS_DECL_CONVERTPYLIRPYTOPYLIRMEMPASS
#define GEN_PASS_DECL_CONVERTPYLIRTOLLVMPASS
#include "pylir/Optimizer/Conversion/Passes.h.inc"

} // namespace pylir
