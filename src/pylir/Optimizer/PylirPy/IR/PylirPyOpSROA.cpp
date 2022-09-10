//  Copyright 2022 Markus Böck
//
//  Licensed under the Apache License v2.0 with LLVM Exceptions.
//  See https://llvm.org/LICENSE.txt for license information.
//  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "PylirPyOps.hpp"

#include <mlir/Dialect/Arithmetic/IR/Arithmetic.h>

namespace
{
mlir::Value materializeUndefined(mlir::OpBuilder& builder, mlir::Type type, mlir::Location loc)
{
    PYLIR_ASSERT(type.isa<pylir::Py::DynamicType>());
    return builder.create<pylir::Py::ConstantOp>(loc, builder.getAttr<pylir::Py::UnboundAttr>());
}

mlir::LogicalResult validateDictKey(::mlir::Attribute key)
{
    // TODO: We currently don't allow symbol ref attrs as keys as we have to do a lookup to figure out their actual
    //       value and I haven't yet thought about how to best do that.
    return mlir::failure(key.isa<mlir::SymbolRefAttr>());
}
} // namespace

mlir::LogicalResult pylir::Py::DictSetItemOp::validateKey(::mlir::Attribute key)
{
    return validateDictKey(key);
}

mlir::LogicalResult pylir::Py::DictTryGetItemOp::validateKey(::mlir::Attribute key)
{
    return validateDictKey(key);
}

namespace
{
template <class T>
mlir::LogicalResult dictOpCanParticipateInSROA(T op)
{
    return mlir::success(op.getMappingExpansion().empty()
                         && llvm::all_of(op.getKeys(),
                                         [](mlir::Value val)
                                         {
                                             mlir::Attribute attr;
                                             return mlir::matchPattern(val, mlir::m_Constant(&attr))
                                                    && mlir::succeeded(validateDictKey(attr));
                                         }));
}
} // namespace

mlir::LogicalResult pylir::Py::MakeDictOp::canParticipateInSROA()
{
    return dictOpCanParticipateInSROA(*this);
}

mlir::Value pylir::Py::MakeDictOp::materializeUndefined(::mlir::OpBuilder& builder, ::mlir::Type type,
                                                        ::mlir::Location loc)
{
    return ::materializeUndefined(builder, type, loc);
}

mlir::LogicalResult pylir::Py::MakeDictExOp::canParticipateInSROA()
{
    return dictOpCanParticipateInSROA(*this);
}

mlir::Value pylir::Py::MakeDictExOp::materializeUndefined(::mlir::OpBuilder& builder, ::mlir::Type type,
                                                          ::mlir::Location loc)
{
    return ::materializeUndefined(builder, type, loc);
}

mlir::LogicalResult pylir::Py::MakeListOp::canParticipateInSROA()
{
    return mlir::success(getIterExpansion().empty());
}

mlir::Value pylir::Py::MakeListOp::materializeUndefined(::mlir::OpBuilder& builder, ::mlir::Type type,
                                                        ::mlir::Location loc)
{
    return ::materializeUndefined(builder, type, loc);
}

mlir::LogicalResult pylir::Py::MakeListExOp::canParticipateInSROA()
{
    return mlir::success(getIterExpansion().empty());
}

mlir::Value pylir::Py::MakeListExOp::materializeUndefined(::mlir::OpBuilder& builder, ::mlir::Type type,
                                                          ::mlir::Location loc)
{
    return ::materializeUndefined(builder, type, loc);
}

mlir::Value pylir::Py::MakeObjectOp::materializeUndefined(::mlir::OpBuilder& builder, ::mlir::Type type,
                                                          ::mlir::Location loc)
{
    return ::materializeUndefined(builder, type, loc);
}

mlir::Value pylir::Py::TupleCopyOp::materializeUndefined(::mlir::OpBuilder& builder, ::mlir::Type type,
                                                         ::mlir::Location loc)
{
    return ::materializeUndefined(builder, type, loc);
}

mlir::Value pylir::Py::StrCopyOp::materializeUndefined(::mlir::OpBuilder& builder, ::mlir::Type type,
                                                       ::mlir::Location loc)
{
    return ::materializeUndefined(builder, type, loc);
}

mlir::Value pylir::Py::MakeFuncOp::materializeUndefined(::mlir::OpBuilder& builder, ::mlir::Type type,
                                                        ::mlir::Location loc)
{
    return ::materializeUndefined(builder, type, loc);
}

namespace
{
template <class T>
void replaceListAggregate(T op, pylir::AggregateDefs& defs, mlir::OpBuilder& builder)
{
    for (const auto& iter : llvm::enumerate(op.getArguments()))
    {
        defs[{op, builder.getIndexAttr(iter.index())}][op->getBlock()] = iter.value();
    }
    auto len = builder.create<mlir::arith::ConstantIndexOp>(op.getLoc(), op.getArguments().size());
    defs[{op, nullptr}][op->getBlock()] = len;
}
} // namespace

void pylir::Py::MakeListOp::replaceAggregate(AggregateDefs& defs, SSABuilder&, mlir::OpBuilder& builder)
{
    replaceListAggregate(*this, defs, builder);
}

void pylir::Py::MakeListExOp::replaceAggregate(AggregateDefs& defs, SSABuilder&, mlir::OpBuilder& builder)
{
    replaceListAggregate(*this, defs, builder);
}

void pylir::Py::ListSetItemOp::replaceAggregate(AggregateDefs& defs, SSABuilder&, mlir::OpBuilder&, mlir::Attribute key)
{
    defs[{getList(), key}][(*this)->getBlock()] = getElement();
}

void pylir::Py::ListGetItemOp::replaceAggregate(AggregateDefs& defs, SSABuilder& ssaBuilder, mlir::OpBuilder&,
                                                mlir::Attribute key)
{
    replaceAllUsesWith(ssaBuilder.readVariable(getLoc(), getType(), defs[{getList(), key}], (*this)->getBlock()));
}

void pylir::Py::ListResizeOp::replaceAggregate(AggregateDefs& defs, SSABuilder&, mlir::OpBuilder&, mlir::Attribute)
{
    defs[{getList(), nullptr}][(*this)->getBlock()] = getLength();
}

void pylir::Py::ListLenOp::replaceAggregate(AggregateDefs& defs, SSABuilder& ssaBuilder, mlir::OpBuilder&,
                                            mlir::Attribute)
{
    replaceAllUsesWith(ssaBuilder.readVariable(getLoc(), getType(), defs[{getList(), nullptr}], (*this)->getBlock()));
}

namespace
{
template <class T>
void replaceDictAggregate(T op, pylir::AggregateDefs& defs)
{
    for (auto [key, value] : llvm::zip(op.getKeys(), op.getValues()))
    {
        mlir::Attribute attr;
        bool result = mlir::matchPattern(key, mlir::m_Constant(&attr));
        PYLIR_ASSERT(result);
        defs[{op, attr}][op->getBlock()] = value;
    }
}
} // namespace

void pylir::Py::MakeDictOp::replaceAggregate(AggregateDefs& defs, SSABuilder&, mlir::OpBuilder&)
{
    replaceDictAggregate(*this, defs);
}

void pylir::Py::MakeDictExOp::replaceAggregate(AggregateDefs& defs, SSABuilder&, mlir::OpBuilder&)
{
    replaceDictAggregate(*this, defs);
}

void pylir::Py::DictTryGetItemOp::replaceAggregate(AggregateDefs& defs, SSABuilder& ssaBuilder, mlir::OpBuilder&,
                                                   mlir::Attribute key)
{
    replaceAllUsesWith(ssaBuilder.readVariable(getLoc(), getType(), defs[{getDict(), key}], (*this)->getBlock()));
}

void pylir::Py::DictSetItemOp::replaceAggregate(AggregateDefs& defs, SSABuilder&, mlir::OpBuilder&, mlir::Attribute key)
{
    defs[{getDict(), key}][(*this)->getBlock()] = getValue();
}

void pylir::Py::SetSlotOp::replaceAggregate(AggregateDefs& defs, SSABuilder&, mlir::OpBuilder&, mlir::Attribute)
{
    defs[{getObject(), getSlotAttr()}][(*this)->getBlock()] = getValue();
}

void pylir::Py::GetSlotOp::replaceAggregate(AggregateDefs& defs, SSABuilder& ssaBuilder, mlir::OpBuilder&,
                                            mlir::Attribute)
{
    replaceAllUsesWith(
        ssaBuilder.readVariable(getLoc(), getType(), defs[{getObject(), getSlotAttr()}], (*this)->getBlock()));
}
