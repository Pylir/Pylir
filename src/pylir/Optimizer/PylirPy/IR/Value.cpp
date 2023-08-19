//  Licensed under the Apache License v2.0 with LLVM Exceptions.
//  See https://llvm.org/LICENSE.txt for license information.
//  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "Value.hpp"

#include <mlir/IR/FunctionInterfaces.h>

#include <pylir/Optimizer/PylirPy/Interfaces/KnownTypeObjectInterface.hpp>
#include <pylir/Support/Macros.hpp>

#include "PylirPyAttributes.hpp"
#include "PylirPyTraits.hpp"

mlir::OpFoldResult pylir::Py::getTypeOf(mlir::Value value)
{
    if (auto op = value.getDefiningOp<pylir::Py::KnownTypeObjectInterface>())
    {
        return op.getKnownTypeObject();
    }
    return nullptr;
}

std::optional<bool> pylir::Py::isUnbound(mlir::Value value)
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
        return std::nullopt;
    }
    // If the defining op has the AlwaysBound trait then it is false.
    auto* op = value.getDefiningOp();
    PYLIR_ASSERT(op);
    if (op->hasTrait<Py::AlwaysBound>())
    {
        return false;
    }
    return std::nullopt;
}

namespace
{
pylir::Py::BuiltinMethodKind getBuiltinMethod(mlir::Attribute attribute, llvm::StringRef method)
{
    auto typeObject = pylir::Py::ref_cast<pylir::Py::ObjectAttrInterface>(attribute, false).getTypeObject();
    if (!typeObject)
    {
        return pylir::Py::BuiltinMethodKind::Unknown;
    }

    auto getBuiltinMethod = [](pylir::Py::RefAttr ref)
    {
        return llvm::StringSwitch<std::optional<pylir::Py::BuiltinMethodKind>>(ref.getRef().getValue())
            .Case(pylir::Builtins::Int.name, pylir::Py::BuiltinMethodKind::Int)
            .Case(pylir::Builtins::Float.name, pylir::Py::BuiltinMethodKind::Float)
            .Case(pylir::Builtins::Str.name, pylir::Py::BuiltinMethodKind::Str)
            .Case(pylir::Builtins::Object.name, pylir::Py::BuiltinMethodKind::Object)
            .Default(std::nullopt);
    };
    if (auto opt = getBuiltinMethod(typeObject))
    {
        return *opt;
    }

    auto mro = pylir::Py::ref_cast_or_null<pylir::Py::TupleAttr>(
        pylir::Py::ref_cast<pylir::Py::TypeAttr>(typeObject).getMroTuple());
    if (!mro)
    {
        return pylir::Py::BuiltinMethodKind::Unknown;
    }
    for (const auto& iter : mro)
    {
        if (!iter)
        {
            // This can probably only be a result of undefined behaviour.
            continue;
        }
        if (auto ref = iter.dyn_cast<pylir::Py::RefAttr>())
        {
            if (auto opt = getBuiltinMethod(ref))
            {
                return *opt;
            }
        }
        auto baseType = pylir::Py::ref_cast_or_null<pylir::Py::TypeAttr>(iter, true);
        if (!baseType)
        {
            return pylir::Py::BuiltinMethodKind::Unknown;
        }
        auto func = baseType.getSlots().get(method);
        if (!func)
        {
            continue;
        }
        return pylir::Py::BuiltinMethodKind::Unknown;
    }

    return pylir::Py::BuiltinMethodKind::Object;
}
} // namespace

pylir::Py::BuiltinMethodKind pylir::Py::getHashFunction(mlir::Attribute attribute)
{
    return getBuiltinMethod(attribute, "__hash__");
}

pylir::Py::BuiltinMethodKind pylir::Py::getEqualsFunction(mlir::Attribute attribute)
{
    return getBuiltinMethod(attribute, "__eq__");
}

mlir::Attribute pylir::Py::getCanonicalEqualsForm(mlir::Attribute attribute)
{
    if (attribute.isa<UnboundAttr>())
    {
        return attribute;
    }
    BuiltinMethodKind implementation = getEqualsFunction(attribute);
    switch (implementation)
    {
        case BuiltinMethodKind::Unknown: return nullptr;
        case BuiltinMethodKind::Object:
            // There is only really RefAttr that can tell us the object identity.
            return attribute.dyn_cast<RefAttr>();
        case BuiltinMethodKind::Str:
            return mlir::StringAttr::get(attribute.getContext(), ref_cast<StrAttr>(attribute).getValue());
        case BuiltinMethodKind::Int:
            return FractionalAttr::get(attribute.getContext(), ref_cast<IntAttr>(attribute).getValue(), BigInt(1));
        case BuiltinMethodKind::Float:
            auto [nom, denom] = toRatio(ref_cast<FloatAttr>(attribute).getDoubleValue());
            return FractionalAttr::get(attribute.getContext(), std::move(nom), std::move(denom));
    }
    PYLIR_UNREACHABLE;
}

std::optional<bool> pylir::Py::isEqual(mlir::Attribute lhs, mlir::Attribute rhs)
{
    lhs = getCanonicalEqualsForm(lhs);
    if (!lhs)
    {
        return std::nullopt;
    }
    rhs = getCanonicalEqualsForm(rhs);
    if (!rhs)
    {
        return std::nullopt;
    }
    return lhs == rhs;
}
