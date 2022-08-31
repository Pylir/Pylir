//  Copyright 2022 Markus Böck
//
//  Licensed under the Apache License v2.0 with LLVM Exceptions.
//  See https://llvm.org/LICENSE.txt for license information.
//  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "Value.hpp"

#include <mlir/IR/FunctionInterfaces.h>

#include <pylir/Optimizer/PylirPy/IR/PylirPyAttributes.hpp>
#include <pylir/Optimizer/PylirPy/IR/PylirPyTraits.hpp>
#include <pylir/Optimizer/PylirPy/Interfaces/ObjectFromTypeObjectInterface.hpp>
#include <pylir/Support/Macros.hpp>

#include "TypeRefineableInterface.hpp"

mlir::OpFoldResult pylir::Py::getTypeOf(mlir::Value value)
{
    if (auto op = value.getDefiningOp<pylir::Py::ObjectFromTypeObjectInterface>())
    {
        return op.getTypeObject();
    }
    if (auto refineable = value.getDefiningOp<Py::TypeRefineableInterface>())
    {
        llvm::SmallVector<Py::TypeAttrUnion> operandTypes(refineable->getNumOperands(), nullptr);
        mlir::SymbolTableCollection collection;
        llvm::SmallVector<Py::ObjectTypeInterface> res;
        if (refineable.refineTypes(operandTypes, res, collection) == TypeRefineResult::Failure)
        {
            return nullptr;
        }
        return res[value.cast<mlir::OpResult>().getResultNumber()].getTypeObject();
    }
    return nullptr;
}

llvm::Optional<bool> pylir::Py::isUnbound(mlir::Value value)
{
    mlir::Attribute constant;
    if (mlir::matchPattern(value, mlir::m_Constant(&constant)))
    {
        return constant.isa<Py::UnboundAttr>();
    }
    if (auto blockArg = value.dyn_cast<mlir::BlockArgument>())
    {
        if (mlir::isa_and_nonnull<mlir::FunctionOpInterface>(blockArg.getOwner()->getParentOp())
            && blockArg.getOwner()->isEntryBlock())
        {
            return false;
        }
        return llvm::None;
    }
    // If the defining op has the AlwaysBound trait then it is false.
    auto* op = value.getDefiningOp();
    PYLIR_ASSERT(op);
    if (op->hasTrait<Py::AlwaysBound>())
    {
        return false;
    }
    return llvm::None;
}
