#include "PylirOps.hpp"

#include <mlir/IR/OpImplementation.h>

#include "PylirDialect.hpp"

#define GET_OP_CLASSES
#include "pylir/Dialect/PylirOps.cpp.inc"
