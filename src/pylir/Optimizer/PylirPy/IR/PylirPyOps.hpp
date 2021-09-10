
#pragma once

#include <mlir/IR/OpDefinition.h>
#include <mlir/Interfaces/CallInterfaces.h>
#include <mlir/Interfaces/InferTypeOpInterface.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>

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

struct KeywordArg
{
    llvm::StringRef name;
    mlir::Value value;
};

using CallArgs = std::variant<mlir::Value, IterExpansion, MappingExpansion, KeywordArg>;

} // namespace pylir::Py

#define GET_OP_CLASSES
#include <pylir/Optimizer/PylirPy/IR/PylirPyOps.h.inc>
