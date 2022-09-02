// Copyright 2022 Markus Böck
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <mlir/Pass/Pass.h>

#include <memory>

namespace pylir::Py
{
#define GEN_PASS_REGISTRATION
#define GEN_PASS_DECL_EXPANDPYDIALECTPASS
#define GEN_PASS_DECL_FOLDHANDLESPASS
#define GEN_PASS_DECL_HANDLELOADSTOREELIMINATIONPASS
#define GEN_PASS_DECL_INLINERPASS
#define GEN_PASS_DECL_MONOMORPHPASS
#define GEN_PASS_DECL_SROAPASS
#define GEN_PASS_DECL_TRIALINLINERPASS
#include "pylir/Optimizer/PylirPy/Transforms/Passes.h.inc"
} // namespace pylir::Py
