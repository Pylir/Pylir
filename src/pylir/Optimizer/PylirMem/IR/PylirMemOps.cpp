#include "PylirMemOps.hpp"

#include <mlir/IR/Builders.h>
#include <mlir/IR/OpImplementation.h>

#include <llvm/ADT/TypeSwitch.h>

#include <pylir/Support/Macros.hpp>

#include "PylirMemAttributes.hpp"

#include <pylir/Optimizer/PylirMem/IR/PylirMemOpsEnums.cpp.inc>

#define GET_OP_CLASSES
#include <pylir/Optimizer/PylirMem/IR/PylirMemOps.cpp.inc>

mlir::LogicalResult
    pylir::Mem::InitTupleOp::inferReturnTypes(::mlir::MLIRContext* context, ::llvm::Optional<::mlir::Location>,
                                              ::mlir::ValueRange, ::mlir::DictionaryAttr, ::mlir::RegionRange,
                                              ::llvm::SmallVectorImpl<::mlir::Type>& inferredReturnTypes)
{
    inferredReturnTypes.emplace_back(Py::DynamicType::get(context));
    return mlir::success();
}

mlir::LogicalResult pylir::Mem::InitListOp::inferReturnTypes(::mlir::MLIRContext* context,
                                                             ::llvm::Optional<::mlir::Location>, ::mlir::ValueRange,
                                                             ::mlir::DictionaryAttr, ::mlir::RegionRange,
                                                             ::llvm::SmallVectorImpl<::mlir::Type>& inferredReturnTypes)
{
    inferredReturnTypes.emplace_back(Py::DynamicType::get(context));
    return mlir::success();
}

mlir::LogicalResult pylir::Mem::InitSetOp::inferReturnTypes(::mlir::MLIRContext* context,
                                                            ::llvm::Optional<::mlir::Location>, ::mlir::ValueRange,
                                                            ::mlir::DictionaryAttr, ::mlir::RegionRange,
                                                            ::llvm::SmallVectorImpl<::mlir::Type>& inferredReturnTypes)
{
    inferredReturnTypes.emplace_back(Py::DynamicType::get(context));
    return mlir::success();
}
