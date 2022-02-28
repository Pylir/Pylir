
#pragma once

#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/OpImplementation.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/SymbolTable.h>
#include <mlir/Interfaces/CallInterfaces.h>
#include <mlir/Interfaces/ControlFlowInterfaces.h>
#include <mlir/Interfaces/InferTypeOpInterface.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>

#include <pylir/Optimizer/Interfaces/CaptureInterface.hpp>
#include <pylir/Optimizer/Interfaces/MemoryFoldInterface.hpp>
#include <pylir/Optimizer/PylirPy/Interfaces/RuntimeTypeInterface.hpp>

#include <variant>

#include "PylirPyAttributes.hpp"
#include "PylirPyTraits.hpp"
#include "PylirPyTypes.hpp"

namespace pylir::Py
{
struct IterExpansion
{
    mlir::Value value;
};

struct MappingExpansion
{
    mlir::Value value;
};

using DictArg = std::variant<std::pair<mlir::Value, mlir::Value>, MappingExpansion>;
using IterArg = std::variant<mlir::Value, IterExpansion>;

} // namespace pylir::Py

#include <pylir/Optimizer/PylirPy/IR/PylirPyOpsEnums.h.inc>

#define GET_OP_CLASSES
#include <pylir/Optimizer/PylirPy/IR/PylirPyOps.h.inc>

namespace pylir::Py
{
template <class Op>
LandingPadOp getLandingPad(Op op)
{
    static_assert(Op::template hasTrait<ExceptionHandling>());
    return mlir::cast<LandingPadOp>(&op->getSuccessor(1)->front());
}
} // namespace pylir::Py
