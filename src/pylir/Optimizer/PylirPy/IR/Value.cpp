//  Licensed under the Apache License v2.0 with LLVM Exceptions.
//  See https://llvm.org/LICENSE.txt for license information.
//  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "Value.hpp"

#include <mlir/IR/FunctionInterfaces.h>

#include <pylir/Optimizer/PylirPy/Interfaces/ObjectFromTypeObjectInterface.hpp>
#include <pylir/Support/Macros.hpp>

#include "PylirPyAttributes.hpp"
#include "PylirPyOps.hpp"
#include "PylirPyTraits.hpp"

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

pylir::Py::BuiltinMethodKind pylir::Py::getHashFunction(pylir::Py::ObjectAttrInterface attribute,
                                                        mlir::Operation* context)
{
    if (!attribute)
    {
        return BuiltinMethodKind::Unknown;
    }

    auto typeAttr = ref_cast_or_null<TypeAttr>(attribute.getTypeObject(), false);
    if (!typeAttr)
    {
        return BuiltinMethodKind::Unknown;
    }
    auto mro = ref_cast_or_null<TupleAttr>(typeAttr.getMroTuple(), false);
    if (!mro)
    {
        return BuiltinMethodKind::Unknown;
    }
    for (const auto& iter : mro.getValue())
    {
        if (!iter)
        {
            // This can probably only be a result of undefined behaviour.
            continue;
        }
        if (auto ref = iter.dyn_cast<RefAttr>())
        {
            auto opt = llvm::StringSwitch<std::optional<BuiltinMethodKind>>(ref.getRef().getValue())
                           .Case(Builtins::Int.name, BuiltinMethodKind::Int)
                           .Case(Builtins::Str.name, BuiltinMethodKind::Str)
                           .Case(Builtins::Object.name, BuiltinMethodKind::Object)
                           .Default(std::nullopt);
            if (opt)
            {
                return *opt;
            }
        }
        auto baseType = ref_cast_or_null<TypeAttr>(iter);
        if (!baseType)
        {
            return BuiltinMethodKind::Unknown;
        }
        auto hashFunc = baseType.getSlots().get("__hash__");
        if (!hashFunc)
        {
            continue;
        }
        return BuiltinMethodKind::Unknown;
    }

    return pylir::Py::BuiltinMethodKind::Object;
}
