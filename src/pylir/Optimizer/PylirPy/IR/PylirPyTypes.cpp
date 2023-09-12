//  Licensed under the Apache License v2.0 with LLVM Exceptions.
//  See https://llvm.org/LICENSE.txt for license information.
//  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "PylirPyTypes.hpp"

#include <mlir/IR/DialectImplementation.h>

#include <llvm/ADT/SetVector.h>
#include <llvm/ADT/TypeSwitch.h>

#include <pylir/Interfaces/Builtins.hpp>

#include "PylirPyAttributes.hpp"
#include "PylirPyDialect.hpp"
#include "PylirPyOps.hpp"

void pylir::Py::PylirPyDialect::initializeTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "pylir/Optimizer/PylirPy/IR/PylirPyTypes.cpp.inc"
      >();
}

#define GET_TYPEDEF_CLASSES
#include "pylir/Optimizer/PylirPy/IR/PylirPyTypes.cpp.inc"
