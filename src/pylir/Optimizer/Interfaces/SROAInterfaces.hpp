//  Licensed under the Apache License v2.0 with LLVM Exceptions.
//  See https://llvm.org/LICENSE.txt for license information.
//  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <mlir/IR/OpDefinition.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>

#include <pylir/Optimizer/Transforms/Util/SSABuilder.hpp>

#include "pylir/Optimizer/Interfaces/SROAAttrInterfaces.h.inc"
#include "pylir/Optimizer/Interfaces/SROAOpInterfaces.h.inc"

namespace pylir {
mlir::LogicalResult
aggregateUseCanParticipateInSROA(const mlir::OpOperand& aggregateUse);
} // namespace pylir
