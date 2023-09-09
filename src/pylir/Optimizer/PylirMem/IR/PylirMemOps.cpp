//  Licensed under the Apache License v2.0 with LLVM Exceptions.
//  See https://llvm.org/LICENSE.txt for license information.
//  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "PylirMemOps.hpp"

#include <mlir/IR/Builders.h>
#include <mlir/IR/OpImplementation.h>

#include <llvm/ADT/TypeSwitch.h>

#include <pylir/Support/Macros.hpp>

#include "PylirMemAttributes.hpp"

#define GET_OP_CLASSES
#include <pylir/Optimizer/PylirMem/IR/PylirMemOps.cpp.inc>
#include <pylir/Optimizer/PylirMem/IR/PylirMemOpsExtra.cpp.inc>
