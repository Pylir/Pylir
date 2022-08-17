// Copyright 2022 Markus Böck
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "TypeRefineableInterface.hpp"

#include "pylir/Optimizer/PylirPy/IR/TypeRefineableInterface.cpp.inc"

#include "PylirPyAttributes.hpp"

pylir::Py::TypeAttrUnion pylir::Py::TypeAttrUnion::join(pylir::Py::TypeAttrUnion rhs,
                                                        mlir::SymbolTableCollection& collection,
                                                        mlir::Operation* context)
{
    if (!rhs || !*this)
    {
        return {};
    }
    if (*this == rhs)
    {
        return *this;
    }
    if (isa<Py::UnboundAttr>())
    {
        return rhs;
    }
    if (rhs.isa<Py::UnboundAttr>())
    {
        return *this;
    }
    if (auto thisType = dyn_cast<ObjectTypeInterface>())
    {
        if (auto rhsType = rhs.dyn_cast<ObjectTypeInterface>())
        {
            return joinTypes(thisType, rhsType);
        }
        if (rhs.isa<Py::UnboundAttr, mlir::SymbolRefAttr, Py::ObjectAttrInterface>())
        {
            return joinTypes(thisType, Py::typeOfConstant(rhs.cast<mlir::Attribute>(), collection, context));
        }
    }
    else if (auto rhsType = rhs.dyn_cast<ObjectTypeInterface>())
    {
        return joinTypes(Py::typeOfConstant(cast<mlir::Attribute>(), collection, context), rhsType);
    }
    else if (rhs.isa<mlir::SymbolRefAttr, Py::ObjectAttrInterface>()
             && isa<mlir::SymbolRefAttr, Py::ObjectAttrInterface>())
    {
        return joinTypes(Py::typeOfConstant(cast<mlir::Attribute>(), collection, context),
                         Py::typeOfConstant(rhs.cast<mlir::Attribute>(), collection, context));
    }
    return {};
}

void pylir::Py::TypeAttrUnion::dump()
{
    if (!*this)
    {
        llvm::errs() << "null";
        return;
    }
    if (auto attr = dyn_cast<mlir::Attribute>())
    {
        return attr.dump();
    }
    else if (auto type = dyn_cast<Py::ObjectTypeInterface>())
    {
        return type.dump();
    }
}
