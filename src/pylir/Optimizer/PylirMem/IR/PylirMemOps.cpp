#include "PylirMemOps.hpp"

#include <mlir/IR/Builders.h>
#include <mlir/IR/OpImplementation.h>

#include <llvm/ADT/TypeSwitch.h>

#include <pylir/Support/Macros.hpp>

#include "PylirMemAttributes.hpp"

#include <pylir/Optimizer/PylirMem/IR/PylirMemOpsEnums.cpp.inc>

// TODO: Remove in MLIR 14
using namespace mlir;

#define GET_OP_CLASSES
#include <pylir/Optimizer/PylirMem/IR/PylirMemOps.cpp.inc>
