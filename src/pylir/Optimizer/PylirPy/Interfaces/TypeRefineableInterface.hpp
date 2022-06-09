// Copyright 2022 Markus Böck
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/SymbolTable.h>

#include <pylir/Optimizer/PylirPy/IR/PylirPyTypes.hpp>

namespace pylir::Py
{
/// Possible return values of `TypeRefineableInterface::refineTypes`
enum class TypeRefineResult
{
    Failure, /// Failed to compute all the resulting types.
    Approximate, /// Only managed to compute an approximate result. The runtime type may be more precise.
    Success, /// Successfully computed the resulting types.
};
}

#include "pylir/Optimizer/PylirPy/Interfaces/TypeRefineableInterface.h.inc"
