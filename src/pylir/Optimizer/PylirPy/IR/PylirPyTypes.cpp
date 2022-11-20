//  Licensed under the Apache License v2.0 with LLVM Exceptions.
//  See https://llvm.org/LICENSE.txt for license information.
//  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "PylirPyTypes.hpp"

#include <mlir/IR/DialectImplementation.h>

#include <llvm/ADT/SetVector.h>
#include <llvm/ADT/TypeSwitch.h>

#include <pylir/Interfaces/Builtins.hpp>

#include "PylirPyAttributes.hpp"
#include "PylirPyDialect.hpp"
#include "PylirPyOps.hpp"

void pylir::Py::PylirPyDialect::initializeTypes()
{
    addTypes<
#define GET_TYPEDEF_LIST
#include "pylir/Optimizer/PylirPy/IR/PylirPyTypes.cpp.inc"
        >();
}

pylir::Py::ObjectTypeInterface pylir::Py::joinTypes(pylir::Py::ObjectTypeInterface lhs,
                                                    pylir::Py::ObjectTypeInterface rhs)
{
    if (lhs == rhs)
    {
        return lhs;
    }
    if (!lhs || !rhs)
    {
        return {};
    }
    if (lhs.isa<pylir::Py::UnboundType>())
    {
        return rhs;
    }
    if (rhs.isa<pylir::Py::UnboundType>())
    {
        return lhs;
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

bool pylir::Py::isMoreSpecific(pylir::Py::ObjectTypeInterface lhs, pylir::Py::ObjectTypeInterface rhs)
{
    if (lhs == rhs)
    {
        return false;
    }
    if (rhs.isa<Py::UnboundType>())
    {
        return true;
    }
    if (lhs.isa<Py::UnboundType>())
    {
        return false;
    }
    if (auto lhsVariant = lhs.dyn_cast<Py::VariantType>())
    {
        auto rhsVariant = rhs.dyn_cast<Py::VariantType>();
        if (!rhsVariant)
        {
            return false;
        }
        llvm::SmallDenseSet<mlir::Type> lhsSet(lhsVariant.getElements().begin(), lhsVariant.getElements().end());
        return !llvm::any_of(rhsVariant.getElements(), [&](mlir::Type type) { return !lhsSet.contains(type); });
    }
    if (auto lhsTuple = lhs.dyn_cast<Py::TupleType>())
    {
        return rhs.isa<Py::TupleType>();
    }
    return rhs.isa<Py::TupleType>();
}

pylir::Py::ObjectTypeInterface pylir::Py::typeOfConstant(mlir::Attribute constant)
{
    if (auto ref = constant.dyn_cast<RefAttr>())
    {
        if (!ref.getSymbol().getInitializerAttr())
        {
            return nullptr;
        }
        return typeOfConstant(ref.getSymbol().getInitializerAttr());
    }
    if (constant.isa<UnboundAttr>())
    {
        return UnboundType::get(constant.getContext());
    }
    if (auto tuple = constant.dyn_cast<TupleAttr>())
    {
        llvm::SmallVector<ObjectTypeInterface> elementTypes;
        for (const auto& iter : tuple)
        {
            elementTypes.push_back(typeOfConstant(iter));
        }
        return TupleType::get(tuple.getTypeObject(), elementTypes);
    }
    auto object = constant.cast<ObjectAttrInterface>();
    return ClassType::get(object.getTypeObject());
}

#define GET_TYPEDEF_CLASSES
#include "pylir/Optimizer/PylirPy/IR/PylirPyTypes.cpp.inc"
