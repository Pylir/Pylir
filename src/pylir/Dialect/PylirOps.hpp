#pragma once

#include <mlir/IR/Dialect.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/Interfaces/CastInterfaces.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>

#include "PylirTypes.hpp"

#define GET_OP_CLASSES
#include "pylir/Dialect/PylirOps.h.inc"
