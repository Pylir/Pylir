
#pragma once

#include <mlir/Analysis/AliasAnalysis.h>
#include <mlir/IR/OpImplementation.h>
#include <mlir/IR/RegionKindInterface.h>
#include <mlir/Interfaces/ControlFlowInterfaces.h>
#include <mlir/Interfaces/InferTypeOpInterface.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>

#include "pylir/Optimizer/Analysis/MemorySSAIRDialect.h.inc"

#define GET_ATTRDEF_CLASSES
#include "pylir/Optimizer/Analysis/MemorySSAIRAttributes.h.inc"

#define GET_TYPEDEF_CLASSES
#include "pylir/Optimizer/Analysis/MemorySSAIRTypes.h.inc"

#define GET_OP_CLASSES
#include "pylir/Optimizer/Analysis/MemorySSAIROps.h.inc"
