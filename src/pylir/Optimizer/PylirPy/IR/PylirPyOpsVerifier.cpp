//  Licensed under the Apache License v2.0 with LLVM Exceptions.
//  See https://llvm.org/LICENSE.txt for license information.
//  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "PylirPyOps.hpp"

#include <llvm/ADT/TypeSwitch.h>

#include "PylirPyAttributes.hpp"
#include "Value.hpp"

namespace
{

template <class SymbolOp>
mlir::FailureOr<SymbolOp> verifySymbolUse(mlir::Operation* op, mlir::SymbolRefAttr name,
                                          mlir::SymbolTableCollection& symbolTable,
                                          llvm::StringRef kindName = SymbolOp::getOperationName())
{
    if (auto* symbol = symbolTable.lookupNearestSymbolFrom(op, name))
    {
        auto casted = mlir::dyn_cast<SymbolOp>(symbol);
        if (!casted)
        {
            return op->emitOpError("Expected '")
                   << name << "' to be of kind '" << kindName << "', not '" << symbol->getName() << "'";
        }
        return casted;
    }
    return op->emitOpError("Failed to find symbol named '") << name << "'";
}

mlir::LogicalResult verify(mlir::Operation* op, mlir::Attribute attribute, mlir::SymbolTableCollection& collection)
{
    auto object = attribute.dyn_cast<pylir::Py::ObjectAttrInterface>();
    if (!object)
    {
        if (auto ref = attribute.dyn_cast<pylir::Py::RefAttr>())
        {
            if (!ref.getSymbol())
            {
                return op->emitOpError("RefAttr does not refer to a 'py.globalValue'");
            }
            return mlir::success();
        }
        if (!attribute.isa<pylir::Py::UnboundAttr>())
        {
            return op->emitOpError("Not allowed attribute '") << attribute << "' found\n";
        }
        return mlir::success();
    }
    if (mlir::failed(::verify(op, object.getTypeObject(), collection)))
    {
        return mlir::failure();
    }
    for (auto iter : object.getSlots())
    {
        if (mlir::failed(verify(op, iter.getValue(), collection)))
        {
            return mlir::failure();
        }
    }
    return llvm::TypeSwitch<mlir::Attribute, mlir::LogicalResult>(object)
        .Case<pylir::Py::TupleAttr, pylir::Py::SetAttr, pylir::Py::ListAttr>(
            [&](auto sequence)
            {
                for (auto iter : sequence.getValue())
                {
                    if (mlir::failed(verify(op, iter, collection)))
                    {
                        return mlir::failure();
                    }
                }
                return mlir::success();
            })
        .Case(
            [&](pylir::Py::DictAttr dict) -> mlir::LogicalResult
            {
                for (auto [key, value] : dict.getValue())
                {
                    if (mlir::failed(verify(op, key, collection)))
                    {
                        return mlir::failure();
                    }
                    if (mlir::failed(verify(op, value, collection)))
                    {
                        return mlir::failure();
                    }
                    if (pylir::Py::getHashFunction(key) == pylir::Py::BuiltinMethodKind::Unknown)
                    {
                        return op->emitOpError(
                            "Constant dictionary not allowed to have key whose type's '__hash__' method is not off of a builtin.");
                    }
                }
                return mlir::success();
            })
        .Case(
            [&](pylir::Py::FunctionAttr functionAttr) -> mlir::LogicalResult
            {
                if (!functionAttr.getValue())
                {
                    return op->emitOpError("Expected function attribute to contain a symbol reference\n");
                }
                if (mlir::failed(verifySymbolUse<mlir::FunctionOpInterface>(op, functionAttr.getValue(), collection,
                                                                            "FunctionOpInterface")))
                {
                    return mlir::failure();
                }

                // These shouldn't return failure as they are just fancy slot accessors (for now), which have been
                // verified above.
                if (auto ref = functionAttr.getKwDefaults().dyn_cast_or_null<pylir::Py::RefAttr>();
                    !ref || ref.getRef().getValue() != pylir::Builtins::None.name)
                {
                    if (!pylir::Py::ref_cast<pylir::Py::DictAttr>(functionAttr.getKwDefaults()))
                    {
                        return op->emitOpError("Expected __kwdefaults__ to refer to a dictionary\n");
                    }
                }
                if (auto ref = functionAttr.getDefaults().dyn_cast_or_null<pylir::Py::RefAttr>();
                    !ref || ref.getRef().getValue() != pylir::Builtins::None.name)
                {
                    if (!pylir::Py::ref_cast<pylir::Py::TupleAttr>(functionAttr.getDefaults()))
                    {
                        return op->emitOpError("Expected __defaults__ to refer to a tuple\n");
                    }
                }
                if (functionAttr.getDict())
                {
                    if (!pylir::Py::ref_cast<pylir::Py::DictAttr>(functionAttr.getDict()))
                    {
                        return op->emitOpError("Expected __dict__ to refer to a dictionary\n");
                    }
                }
                return mlir::success();
            })
        .Case(
            [&](pylir::Py::TypeAttr typeAttr) -> mlir::LogicalResult
            {
                if (mlir::failed(verify(op, typeAttr.getMroTuple(), collection)))
                {
                    return mlir::failure();
                }
                auto mro = pylir::Py::ref_cast<pylir::Py::TupleAttr>(typeAttr.getMroTuple());
                if (!mro)
                {
                    return op->emitOpError("Expected MRO to refer to a tuple\n");
                }
                if (auto result = typeAttr.getSlots().get("__slots__"); result)
                {
                    if (!pylir::Py::ref_cast<pylir::Py::TupleAttr>(result))
                    {
                        return op->emitOpError("Expected __slots__ to refer to a tuple\n");
                    }
                }
                return mlir::success();
            })
        .Default(mlir::success());
}

mlir::LogicalResult verifyCall(::mlir::SymbolTableCollection& symbolTable, mlir::Operation* call,
                               mlir::ValueRange callOperands, mlir::FlatSymbolRefAttr callee)
{
    auto funcOp = symbolTable.lookupNearestSymbolFrom<mlir::FunctionOpInterface>(call, callee);
    if (!funcOp)
    {
        return call->emitOpError("failed to find function named '") << callee << "'";
    }
    auto argumentTypes = funcOp.getArgumentTypes();
    llvm::SmallVector<mlir::Type> operandTypes;
    for (auto iter : callOperands)
    {
        operandTypes.push_back(iter.getType());
    }
    if (!std::equal(argumentTypes.begin(), argumentTypes.end(), operandTypes.begin(), operandTypes.end()))
    {
        return call->emitOpError("call operand types are not compatible with argument types of '") << callee << "'";
    }
    return mlir::success();
}

} // namespace

mlir::LogicalResult pylir::Py::CallOp::verifySymbolUses(::mlir::SymbolTableCollection& symbolTable)
{
    return verifyCall(symbolTable, *this, getCallOperands(), getCalleeAttr());
}

mlir::LogicalResult pylir::Py::InvokeOp::verifySymbolUses(::mlir::SymbolTableCollection& symbolTable)
{
    return verifyCall(symbolTable, *this, getCallOperands(), getCalleeAttr());
}

mlir::LogicalResult pylir::Py::LoadOp::verifySymbolUses(::mlir::SymbolTableCollection& symbolTable)
{
    return verifySymbolUse<GlobalOp>(*this, getGlobalAttr(), symbolTable, GlobalOp::getOperationName());
}

mlir::LogicalResult pylir::Py::StoreOp::verifySymbolUses(::mlir::SymbolTableCollection& symbolTable)
{
    auto global = verifySymbolUse<GlobalOp>(*this, getGlobalAttr(), symbolTable, GlobalOp::getOperationName());
    if (mlir::failed(global))
    {
        return mlir::failure();
    }
    if (global->getType() != getValue().getType())
    {
        return emitOpError("Type of value to store '")
               << getValue().getType() << "' does not match type of global '" << global->getSymName() << " : "
               << global->getType() << "' to store into";
    }
    return mlir::success();
}

mlir::LogicalResult pylir::Py::MakeFuncOp::verifySymbolUses(::mlir::SymbolTableCollection& symbolTable)
{
    return verifySymbolUse<mlir::FunctionOpInterface>(*this, getFunctionAttr(), symbolTable, "FunctionOpInterface");
}

mlir::LogicalResult pylir::Py::GlobalOp::verifySymbolUses(::mlir::SymbolTableCollection& symbolTable)
{
    if (!getInitializerAttr())
    {
        return mlir::success();
    }
    return llvm::TypeSwitch<mlir::Type, mlir::LogicalResult>(getType())
        .Case(
            [&](DynamicType) -> mlir::LogicalResult
            {
                if (!getInitializerAttr().isa<ObjectAttrInterface, RefAttr, UnboundAttr>())
                {
                    return emitOpError(
                        "Expected initializer of type 'ObjectAttrInterface' or 'RefAttr' to global value");
                }
                return ::verify(*this, getInitializerAttr(), symbolTable);
            })
        .Case(
            [&](mlir::IndexType) -> mlir::LogicalResult
            {
                if (!getInitializerAttr().isa<mlir::IntegerAttr>())
                {
                    return emitOpError("Expected integer attribute initializer");
                }
                return mlir::success();
            })
        .Case(
            [&](mlir::FloatType) -> mlir::LogicalResult
            {
                if (!getInitializerAttr().isa<mlir::FloatAttr>())
                {
                    return emitOpError("Expected float attribute initializer");
                }
                return mlir::success();
            });
}

mlir::LogicalResult pylir::Py::ConstantOp::verifySymbolUses(::mlir::SymbolTableCollection& symbolTable)
{
    return ::verify(*this, getConstantAttr(), symbolTable);
}

mlir::LogicalResult pylir::Py::GlobalValueOp::verifySymbolUses(::mlir::SymbolTableCollection& symbolTable)
{
    if (!isDeclaration())
    {
        return ::verify(*this, getInitializerAttr(), symbolTable);
    }
    return mlir::success();
}

mlir::LogicalResult pylir::Py::UnpackOp::verify()
{
    if (!getAfter().empty() && !getRest())
    {
        return emitOpError("'after_rest' results specified, without a rest argument");
    }
    return mlir::success();
}

mlir::LogicalResult pylir::Py::UnpackExOp::verify()
{
    if (!getAfter().empty() && !getRest())
    {
        return emitOpError("'after_rest' results specified, without a rest argument");
    }
    return mlir::success();
}
