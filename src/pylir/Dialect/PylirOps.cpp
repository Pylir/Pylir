#include "PylirOps.hpp"

#include <mlir/IR/OpImplementation.h>

#include "PylirDialect.hpp"

//TODO: Remove in MLIR 13
using namespace mlir;
#define GET_OP_CLASSES
#include "pylir/Dialect/PylirOps.cpp.inc"
