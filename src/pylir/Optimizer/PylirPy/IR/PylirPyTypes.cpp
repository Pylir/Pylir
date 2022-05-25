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
    if (lhs.isa<pylir::Py::UnknownType>() || rhs.isa<pylir::Py::UnknownType>())
    {
        return Py::UnknownType::get(lhs.getContext());
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
    if (rhs.isa<Py::UnknownType>())
    {
        return true;
    }
    if (lhs.isa<Py::UnknownType>())
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
        return true;
    }
    return rhs.isa<Py::VariantType>();
}

pylir::Py::ObjectTypeInterface pylir::Py::typeOfConstant(mlir::Attribute constant, mlir::SymbolTable* table)
{
    if (table)
    {
        if (auto ref = constant.dyn_cast<mlir::FlatSymbolRefAttr>())
        {
            auto globalVal = table->lookup<pylir::Py::GlobalValueOp>(ref.getAttr());
            if (globalVal.isDeclaration())
            {
                return pylir::Py::UnknownType::get(constant.getContext());
            }
            return typeOfConstant(globalVal.getInitializerAttr(), table);
        }
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
            elementTypes.push_back(typeOfConstant(iter, table));
        }
        return pylir::Py::TupleType::get(constant.getContext(), tuple.getTypeObject(), elementTypes);
    }
    // TODO: Handle slots?
    if (auto object = constant.dyn_cast<pylir::Py::ObjectAttrInterface>())
    {
        if (auto typeObject = object.getTypeObject())
        {
            return pylir::Py::ClassType::get(constant.getContext(), typeObject, llvm::None);
        }
    }
    return pylir::Py::UnknownType::get(constant.getContext());
}

namespace
{
mlir::LogicalResult parseSlotSuffix(
    mlir::AsmParser& parser,
    mlir::FailureOr<llvm::SmallVector<std::pair<mlir::StringAttr, pylir::Py::ObjectTypeInterface>>>& result)
{
    result = llvm::SmallVector<std::pair<mlir::StringAttr, pylir::Py::ObjectTypeInterface>>{};
    if (parser.parseOptionalComma())
    {
        return mlir::success();
    }
    if (parser.parseCommaSeparatedList(::mlir::AsmParser::Delimiter::Braces,
                                       [&]() -> mlir::ParseResult
                                       {
                                           auto temp = std::pair<mlir::StringAttr, pylir::Py::ObjectTypeInterface>{};
                                           if (parser.parseAttribute(temp.first) || parser.parseEqual()
                                               || parser.parseType(temp.second))
                                           {
                                               return mlir::failure();
                                           }
                                           result->push_back(std::move(temp));
                                           return mlir::success();
                                       }))
    {
        return ::mlir::failure();
    }
    return mlir::success();
}

void printSlotSuffix(mlir::AsmPrinter& parser,
                     llvm::ArrayRef<std::pair<mlir::StringAttr, pylir::Py::ObjectTypeInterface>> result)
{
    if (result.empty())
    {
        return;
    }
    parser << ", {";
    llvm::interleaveComma(result, parser.getStream(),
                          [&](const auto& pair) { parser << pair.first << " = " << pair.second; });
    parser << "}";
}
} // namespace

#define GET_TYPEDEF_CLASSES
#include "pylir/Optimizer/PylirPy/IR/PylirPyOpsTypes.cpp.inc"
