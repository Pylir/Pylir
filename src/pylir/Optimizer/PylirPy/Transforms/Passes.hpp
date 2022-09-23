//  Licensed under the Apache License v2.0 with LLVM Exceptions.
//  See https://llvm.org/LICENSE.txt for license information.
//  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <mlir/Pass/Pass.h>

#include <memory>

namespace pylir::Py
{
#define GEN_PASS_REGISTRATION
#define GEN_PASS_DECL_FINALIZEREFATTRSPASS
#define GEN_PASS_DECL_EXPANDPYDIALECTPASS
#define GEN_PASS_DECL_FOLDGLOBALSPASS
#define GEN_PASS_DECL_GLOBALLOADSTOREELIMINATIONPASS
#define GEN_PASS_DECL_MONOMORPHPASS
#define GEN_PASS_DECL_TRIALINLINERPASS
#define GEN_PASS_DECL_GLOBALSROAPASS
#include "pylir/Optimizer/PylirPy/Transforms/Passes.h.inc"
} // namespace pylir::Py
