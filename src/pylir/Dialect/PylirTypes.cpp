
#include "PylirTypes.hpp"

#include <llvm/ADT/TypeSwitch.h>
#include <mlir/IR/DialectImplementation.h>

#define GET_TYPEDEF_CLASSES
#include "pylir/Dialect/PylirOpsTypes.cpp.inc"
