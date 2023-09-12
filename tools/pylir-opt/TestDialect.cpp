//  Licensed under the Apache License v2.0 with LLVM Exceptions.
//  See https://llvm.org/LICENSE.txt for license information.
//  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "TestDialect.hpp"

#include <mlir/IR/DialectImplementation.h>
#include <mlir/IR/OpImplementation.h>

#include <llvm/ADT/TypeSwitch.h>

void pylir::test::TestDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "TestOps.cpp.inc"
      >();
  addTypes<
#define GET_TYPEDEF_LIST
#include "TestTypes.cpp.inc"
      >();
  addAttributes<
#define GET_ATTRDEF_LIST
#include "TestAttributes.cpp.inc"
      >();
}

#include "TestDialect.cpp.inc"

#define GET_OP_CLASSES
#include "TestOps.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "TestTypes.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "TestAttributes.cpp.inc"
