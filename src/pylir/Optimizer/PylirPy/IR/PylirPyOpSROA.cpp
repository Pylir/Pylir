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

mlir::LogicalResult pylir::Py::DictDelItemOp::validateKey(::mlir::Attribute key)
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
void replaceListAggregate(T op, llvm::function_ref<void(mlir::Attribute, mlir::Value)> write, mlir::OpBuilder& builder)
{
    for (const auto& iter : llvm::enumerate(op.getArguments()))
    {
        write(builder.getIndexAttr(iter.index()), iter.value());
    }
    auto len = builder.create<mlir::arith::ConstantIndexOp>(op.getLoc(), op.getArguments().size());
    write(nullptr, len);
}
} // namespace

void pylir::Py::MakeListOp::replaceAggregate(mlir::OpBuilder& builder,
                                             llvm::function_ref<void(mlir::Attribute, mlir::Value)> write)
{
    replaceListAggregate(*this, write, builder);
}

void pylir::Py::MakeListExOp::replaceAggregate(mlir::OpBuilder& builder,
                                               llvm::function_ref<void(mlir::Attribute, mlir::Value)> write)
{
    replaceListAggregate(*this, write, builder);
}

void pylir::Py::ListSetItemOp::replaceAggregate(mlir::OpBuilder&, mlir::Attribute key,
                                                llvm::function_ref<mlir::Value(mlir::Attribute, mlir::Type)>,
                                                llvm::function_ref<void(mlir::Attribute, mlir::Value)> write)
{
    write(key, getElement());
}

void pylir::Py::ListGetItemOp::replaceAggregate(mlir::OpBuilder&, mlir::Attribute key,
                                                llvm::function_ref<mlir::Value(mlir::Attribute, mlir::Type)> read,
                                                llvm::function_ref<void(mlir::Attribute, mlir::Value)>)
{
    replaceAllUsesWith(read(key, getType()));
}

void pylir::Py::ListResizeOp::replaceAggregate(mlir::OpBuilder&, mlir::Attribute,
                                               llvm::function_ref<mlir::Value(mlir::Attribute, mlir::Type)>,
                                               llvm::function_ref<void(mlir::Attribute, mlir::Value)> write)
{
    write(nullptr, getLength());
}

void pylir::Py::ListLenOp::replaceAggregate(mlir::OpBuilder&, mlir::Attribute,
                                            llvm::function_ref<mlir::Value(mlir::Attribute, mlir::Type)> read,
                                            llvm::function_ref<void(mlir::Attribute, mlir::Value)>)
{
    replaceAllUsesWith(read(nullptr, getType()));
}

namespace
{
template <class T>
void replaceDictAggregate(T op, llvm::function_ref<void(mlir::Attribute, mlir::Value)> write)
{
    for (auto [key, value] : llvm::zip(op.getKeys(), op.getValues()))
    {
        mlir::Attribute attr;
        bool result = mlir::matchPattern(key, mlir::m_Constant(&attr));
        PYLIR_ASSERT(result);
        write(attr, value);
    }
}
} // namespace

void pylir::Py::MakeDictOp::replaceAggregate(mlir::OpBuilder&,
                                             llvm::function_ref<void(mlir::Attribute, mlir::Value)> write)
{
    replaceDictAggregate(*this, write);
}

void pylir::Py::MakeDictExOp::replaceAggregate(mlir::OpBuilder&,
                                               llvm::function_ref<void(mlir::Attribute, mlir::Value)> write)
{
    replaceDictAggregate(*this, write);
}

void pylir::Py::DictTryGetItemOp::replaceAggregate(mlir::OpBuilder&, mlir::Attribute key,
                                                   llvm::function_ref<mlir::Value(mlir::Attribute, mlir::Type)> read,
                                                   llvm::function_ref<void(mlir::Attribute, mlir::Value)>)
{
    replaceAllUsesWith(read(key, getType()));
}

void pylir::Py::DictSetItemOp::replaceAggregate(mlir::OpBuilder&, mlir::Attribute key,
                                                llvm::function_ref<mlir::Value(mlir::Attribute, mlir::Type)>,
                                                llvm::function_ref<void(mlir::Attribute, mlir::Value)> write)
{
    write(key, getValue());
}

void pylir::Py::DictDelItemOp::replaceAggregate(mlir::OpBuilder& builder, mlir::Attribute key,
                                                llvm::function_ref<mlir::Value(mlir::Attribute, mlir::Type)> read,
                                                llvm::function_ref<void(mlir::Attribute, mlir::Value)> write)
{
    auto value = read(key, builder.getType<pylir::Py::DynamicType>());
    auto unbound = builder.create<ConstantOp>(getLoc(), builder.getAttr<UnboundAttr>());
    write(key, unbound);
    auto didNotExist = builder.create<IsUnboundValueOp>(getLoc(), value);
    auto one = builder.create<mlir::arith::ConstantIntOp>(getLoc(), true, 1);
    mlir::Value existed = builder.create<mlir::arith::XOrIOp>(getLoc(), didNotExist, one);
    replaceAllUsesWith(existed);
}

void pylir::Py::SetSlotOp::replaceAggregate(mlir::OpBuilder&, mlir::Attribute,
                                            llvm::function_ref<mlir::Value(mlir::Attribute, mlir::Type)>,
                                            llvm::function_ref<void(mlir::Attribute, mlir::Value)> write)
{
    write(getSlotAttr(), getValue());
}

void pylir::Py::GetSlotOp::replaceAggregate(mlir::OpBuilder&, mlir::Attribute,
                                            llvm::function_ref<mlir::Value(mlir::Attribute, mlir::Type)> read,
                                            llvm::function_ref<void(mlir::Attribute, mlir::Value)>)
{
    replaceAllUsesWith(read(getSlotAttr(), getType()));
}

namespace
{
void destructureSlots(pylir::Py::ObjectAttrInterface attr,
                      llvm::function_ref<void(mlir::Attribute, mlir::Type, mlir::Attribute)> write)
{
    for (const auto& iter : attr.getSlots())
    {
        write(iter.getName(), pylir::Py::DynamicType::get(attr.getContext()), iter.getValue());
    }
}
} // namespace

void pylir::Py::ObjectAttr::destructureAggregate(
    llvm::function_ref<void(mlir::Attribute, mlir::Type, mlir::Attribute)> write) const
{
    destructureSlots(*this, write);
}

void pylir::Py::IntAttr::destructureAggregate(
    llvm::function_ref<void(mlir::Attribute, mlir::Type, mlir::Attribute)> write) const
{
    destructureSlots(*this, write);
}

void pylir::Py::BoolAttr::destructureAggregate(
    llvm::function_ref<void(mlir::Attribute, mlir::Type, mlir::Attribute)> write) const
{
    destructureSlots(*this, write);
}

void pylir::Py::FloatAttr::destructureAggregate(
    llvm::function_ref<void(mlir::Attribute, mlir::Type, mlir::Attribute)> write) const
{
    destructureSlots(*this, write);
}

void pylir::Py::StrAttr::destructureAggregate(
    llvm::function_ref<void(mlir::Attribute, mlir::Type, mlir::Attribute)> write) const
{
    destructureSlots(*this, write);
}

void pylir::Py::TupleAttr::destructureAggregate(
    llvm::function_ref<void(mlir::Attribute, mlir::Type, mlir::Attribute)> write) const
{
    destructureSlots(*this, write);
}

void pylir::Py::ListAttr::destructureAggregate(
    llvm::function_ref<void(mlir::Attribute, mlir::Type, mlir::Attribute)> write) const
{
    destructureSlots(*this, write);
    auto indexType = mlir::IndexType::get(getContext());
    write(nullptr, indexType, mlir::IntegerAttr::get(indexType, getValue().size()));
    for (auto [index, iter] : llvm::enumerate(getValue()))
    {
        write(mlir::IntegerAttr::get(indexType, index), DynamicType::get(getContext()), iter);
    }
}

void pylir::Py::SetAttr::destructureAggregate(
    llvm::function_ref<void(mlir::Attribute, mlir::Type, mlir::Attribute)> write) const
{
    destructureSlots(*this, write);
    // TODO: Support SetAttr
}

void pylir::Py::DictAttr::destructureAggregate(
    llvm::function_ref<void(mlir::Attribute, mlir::Type, mlir::Attribute)> write) const
{
    destructureSlots(*this, write);
    // TODO: This is problematic with keys that are FlatSymbolRefAttr. But that is somewhat a problem of DictAttr in
    //  general. I need to flesh it out with more proper APIs and just generally nail down its semantics.
    for (auto [key, value] : getValue())
    {
        write(key, DynamicType::get(getContext()), value);
    }
}

void pylir::Py::FunctionAttr::destructureAggregate(
    llvm::function_ref<void(mlir::Attribute, mlir::Type, mlir::Attribute)> write) const
{
    destructureSlots(*this, write);
}

void pylir::Py::TypeAttr::destructureAggregate(
    llvm::function_ref<void(mlir::Attribute, mlir::Type, mlir::Attribute)> write) const
{
    destructureSlots(*this, write);
}
