
// Copyright 2022 Markus Böck
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <mlir/Pass/Pass.h>

#include <memory>

namespace pylir
{
std::unique_ptr<mlir::Pass> createLoadForwardingPass();

#define GEN_PASS_REGISTRATION
#include "pylir/Optimizer/Transforms/Passes.h.inc"

} // namespace pylir
