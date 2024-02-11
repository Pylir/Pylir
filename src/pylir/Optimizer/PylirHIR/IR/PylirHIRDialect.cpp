// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "PylirHIRDialect.hpp"

#include <mlir/IR/Builders.h>
#include <mlir/IR/DialectImplementation.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/OpImplementation.h>

#include <llvm/ADT/TypeSwitch.h>

#include <pylir/Optimizer/PylirPy/IR/PylirPyDialect.hpp>

#include "PylirHIRAttributes.hpp"
#include "PylirHIROps.hpp"

#define GET_TYPEDEF_CLASSES
#define GET_ATTRDEF_CLASSES
#include "pylir/Optimizer/PylirHIR/IR/PylirHIRAttributes.cpp.inc"
#include "pylir/Optimizer/PylirHIR/IR/PylirHIRDialect.cpp.inc"
#include "pylir/Optimizer/PylirHIR/IR/PylirHIREnums.cpp.inc"
#include "pylir/Optimizer/PylirHIR/IR/PylirHIRTypes.cpp.inc"

void pylir::HIR::PylirHIRDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "pylir/Optimizer/PylirHIR/IR/PylirHIROps.cpp.inc"
      >();
  addTypes<
#define GET_TYPEDEF_LIST
#include "pylir/Optimizer/PylirHIR/IR/PylirHIRTypes.cpp.inc"
      >();
  addAttributes<
#define GET_ATTRDEF_LIST
#include "pylir/Optimizer/PylirHIR/IR/PylirHIRAttributes.cpp.inc"
      >();
}
