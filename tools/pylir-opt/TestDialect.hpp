
#pragma once

#include <mlir/IR/OpImplementation.h>
#include <mlir/Interfaces/InferTypeOpInterface.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>

#include "TestDialect.h.inc"

#define GET_ATTRDEF_CLASSES
#include "TestDialectAttributes.h.inc"

#define GET_TYPEDEF_CLASSES
#include "TestDialectTypes.h.inc"

#define GET_OP_CLASSES
#include "TestDialectOps.h.inc"
