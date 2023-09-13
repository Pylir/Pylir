//  Licensed under the Apache License v2.0 with LLVM Exceptions.
//  See https://llvm.org/LICENSE.txt for license information.
//  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <mlir/Pass/Pass.h>

#include <memory>

namespace pylir::test {
#define GEN_PASS_REGISTRATION
#define GEN_PASS_DECL
#include "Passes.h.inc"
} // namespace pylir::test
