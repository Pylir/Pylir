//  Licensed under the Apache License v2.0 with LLVM Exceptions.
//  See https://llvm.org/LICENSE.txt for license information.
//  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <mlir/Interfaces/CallInterfaces.h>

namespace pylir::Py
{

struct InlinedOps
{
    mlir::Operation* firstOperationInFirstBlock;
    mlir::Region::iterator endBlock;
};

InlinedOps inlineCall(mlir::CallOpInterface call, mlir::CallableOpInterface callable);
} // namespace pylir::Py
