
#pragma once

#include <mlir/IR/OpDefinition.h>
#include <mlir/Interfaces/CallInterfaces.h>
#include <mlir/Interfaces/InferTypeOpInterface.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>

#include <variant>

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

#define GET_OP_CLASSES
#include <pylir/Optimizer/PylirPy/IR/PylirPyOps.h.inc>
