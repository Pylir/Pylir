#pragma once

#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Dialect.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/SymbolTable.h>
#include <mlir/Interfaces/CastInterfaces.h>
#include <mlir/Interfaces/InferTypeOpInterface.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>

#include <pylir/Dialect/PylirOpsEnums.h.inc>

#include "PylirTypes.hpp"

#define GET_OP_CLASSES
#include <pylir/Dialect/PylirOps.h.inc>
