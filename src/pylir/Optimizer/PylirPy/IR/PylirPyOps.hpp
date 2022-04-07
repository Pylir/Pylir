
#pragma once

#include <mlir/IR/FunctionInterfaces.h>
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
#include <pylir/Optimizer/PylirPy/Interfaces/ExceptionHandlingInterface.hpp>
#include <pylir/Optimizer/PylirPy/Interfaces/TypeRefineableInterface.hpp>

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

namespace details
{
mlir::Operation* cloneWithExceptionHandlingImpl(mlir::OpBuilder& builder, mlir::Operation* operation,
                                                const mlir::OperationName& invokeVersion, ::mlir::Block* happyPath,
                                                mlir::Block* exceptionPath, mlir::ValueRange unwindOperands,
                                                llvm::StringRef attrSizedSegmentName, llvm::ArrayRef<int> shape);
} // namespace details

template <class InvokeVersion, int... shape>
struct AddableExceptionHandling
{
    template <class ConcreteType>
    class Impl : public AddableExceptionHandlingInterface::Trait<ConcreteType>
    {
    public:
        mlir::Operation* cloneWithExceptionHandling(mlir::OpBuilder& builder, ::mlir::Block* happyPath,
                                                    mlir::Block* exceptionPath, mlir::ValueRange unwindOperands)
        {
            return details::cloneWithExceptionHandlingImpl(
                builder, this->getOperation(),
                mlir::OperationName(InvokeVersion::getOperationName(), builder.getContext()), happyPath, exceptionPath,
                unwindOperands, mlir::OpTrait::AttrSizedOperandSegments<InvokeVersion>::getOperandSegmentSizeAttr(),
                {shape...});
        }
    };
};

} // namespace pylir::Py

#include <pylir/Optimizer/PylirPy/IR/PylirPyOpsEnums.h.inc>

#define GET_OP_CLASSES
#include <pylir/Optimizer/PylirPy/IR/PylirPyOps.h.inc>
