//  Licensed under the Apache License v2.0 with LLVM Exceptions.
//  See https://llvm.org/LICENSE.txt for license information.
//  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <mlir/IR/OpDefinition.h>

#include <pylir/Optimizer/Transforms/Util/SSABuilder.hpp>

namespace pylir
{
using AggregateDefs = llvm::DenseMap<std::pair<mlir::Value, mlir::Attribute>, pylir::SSABuilder::DefinitionsMap>;
}

#include "pylir/Optimizer/Interfaces/SROAInterfaces.h.inc"
