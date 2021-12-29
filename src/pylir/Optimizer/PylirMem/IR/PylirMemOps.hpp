#pragma once

#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Dialect.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/OpImplementation.h>
#include <mlir/IR/SymbolTable.h>
#include <mlir/Interfaces/InferTypeOpInterface.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>

#include <pylir/Optimizer/PylirMem/IR/PylirMemOpsEnums.h.inc>
#include <pylir/Optimizer/PylirPy/IR/PylirPyAttributes.hpp>
#include <pylir/Optimizer/PylirPy/IR/PylirPyTraits.hpp>
#include <pylir/Optimizer/PylirPy/IR/PylirPyTypes.hpp>

#include "PylirMemAttributes.hpp"
#include "PylirMemTypes.hpp"

#define GET_OP_CLASSES
#include <pylir/Optimizer/PylirMem/IR/PylirMemOps.h.inc>
