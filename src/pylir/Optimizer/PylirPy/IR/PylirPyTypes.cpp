// Copyright 2022 Markus Böck
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "PylirPyTypes.hpp"

#include <mlir/IR/DialectImplementation.h>

#include <llvm/ADT/SetVector.h>
#include <llvm/ADT/TypeSwitch.h>

#include <pylir/Optimizer/PylirPy/Util/Builtins.hpp>

#include "PylirPyAttributes.hpp"
#include "PylirPyDialect.hpp"
#include "PylirPyOps.hpp"

void pylir::Py::PylirPyDialect::initializeTypes()
{
    addTypes<
#define GET_TYPEDEF_LIST
#include "pylir/Optimizer/PylirPy/IR/PylirPyOpsTypes.cpp.inc"
        >();
}

pylir::Py::ObjectTypeInterface pylir::Py::joinTypes(pylir::Py::ObjectTypeInterface lhs,
                                                    pylir::Py::ObjectTypeInterface rhs)
{
    if (lhs == rhs)
    {
        return lhs;
    }
    if (lhs.isa<pylir::Py::UnboundType>())
    {
        return rhs;
    }
    if (rhs.isa<pylir::Py::UnboundType>())
    {
        return lhs;
    }
    if (!lhs || !rhs)
    {
        return {};
    }
    llvm::SmallSetVector<mlir::Type, 4> elementTypes;
    if (auto variant = lhs.dyn_cast<Py::VariantType>())
    {
        elementTypes.insert(variant.getElements().begin(), variant.getElements().end());
    }
    else
    {
        elementTypes.insert(lhs);
    }
    if (auto variant = rhs.dyn_cast<Py::VariantType>())
    {
        elementTypes.insert(variant.getElements().begin(), variant.getElements().end());
    }
    else
    {
        elementTypes.insert(rhs);
    }
    llvm::SmallVector<pylir::Py::ObjectTypeInterface> temp(elementTypes.begin(), elementTypes.end());
    return pylir::Py::VariantType::get(lhs.getContext(), temp);
}

pylir::Py::ObjectTypeInterface pylir::Py::typeOfConstant(mlir::Attribute constant,
                                                         mlir::SymbolTableCollection& collection,
                                                         mlir::Operation* context)
{
    if (auto ref = constant.dyn_cast<mlir::SymbolRefAttr>())
    {
        auto globalVal = collection.lookupNearestSymbolFrom<pylir::Py::GlobalValueOp>(context, ref);
        if (globalVal.isDeclaration())
        {
            return nullptr;
        }
        return typeOfConstant(globalVal.getInitializerAttr(), collection, context);
    }
    if (constant.isa<pylir::Py::UnboundAttr>())
    {
        return pylir::Py::UnboundType::get(constant.getContext());
    }
    if (auto tuple = constant.dyn_cast<pylir::Py::TupleAttr>())
    {
        llvm::SmallVector<pylir::Py::ObjectTypeInterface> elementTypes;
        for (const auto& iter : tuple.getValue())
        {
            elementTypes.push_back(typeOfConstant(iter, collection, context));
        }
        return pylir::Py::TupleType::get(tuple.getTypeObject(), elementTypes);
    }
    auto object = constant.cast<pylir::Py::ObjectAttrInterface>();
    return pylir::Py::ClassType::get(object.getTypeObject());
}

#define GET_TYPEDEF_CLASSES
#include "pylir/Optimizer/PylirPy/IR/PylirPyOpsTypes.cpp.inc"
