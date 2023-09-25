//  Licensed under the Apache License v2.0 with LLVM Exceptions.
//  See https://llvm.org/LICENSE.txt for license information.
//  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "PylirPyOps.hpp"

#include <mlir/Dialect/Arith/IR/Arith.h>

#include "Value.hpp"

using namespace pylir::Py;

namespace {

mlir::FailureOr<mlir::Attribute> getDictKey(mlir::Value keyOperand) {
  EqualsAttrInterface attr;
  if (!mlir::matchPattern(keyOperand, mlir::m_Constant(&attr)))
    return mlir::failure();

  auto canon = attr.getCanonicalAttribute();
  if (!canon)
    return mlir::failure();

  return canon;
}

template <class T>
mlir::LogicalResult dictOpCanParticipateInSROA(T op) {
  return mlir::success(op.getMappingExpansion().empty() &&
                       llvm::all_of(op.getKeys(), [](mlir::Value val) {
                         EqualsAttrInterface attr;
                         return mlir::matchPattern(val,
                                                   mlir::m_Constant(&attr));
                       }));
}

template <class T>
void replaceListAggregate(
    T op,
    llvm::function_ref<void(mlir::Attribute, mlir::SideEffects::Resource*,
                            mlir::Value)>
        write,
    mlir::OpBuilder& builder) {
  for (const auto& iter : llvm::enumerate(op.getArguments()))
    write(builder.getIndexAttr(iter.index()), pylir::Py::ListResource::get(),
          iter.value());

  auto len = builder.create<mlir::arith::ConstantIndexOp>(
      op.getLoc(), op.getArguments().size());
  write(nullptr, pylir::Py::ListResource::get(), len);
}

template <class T>
void replaceDictAggregate(
    T op, mlir::OpBuilder& builder,
    llvm::function_ref<void(mlir::Attribute, mlir::SideEffects::Resource*,
                            mlir::Value)>
        write) {
  auto size = builder.create<mlir::arith::ConstantIndexOp>(op.getLoc(),
                                                           op.getKeys().size());
  write(nullptr, pylir::Py::DictResource::get(), size);
  for (auto [key, value] : llvm::zip(op.getKeys(), op.getValues())) {
    EqualsAttrInterface attr;
    bool result = mlir::matchPattern(key, mlir::m_Constant(&attr));
    PYLIR_ASSERT(result);
    write(attr.getCanonicalAttribute(), pylir::Py::DictResource::get(), value);
  }
}

void destructureSlots(
    pylir::Py::ConstObjectAttrInterface attr,
    llvm::function_ref<void(mlir::Attribute, mlir::SideEffects::Resource*,
                            mlir::Type, mlir::Attribute)>
        write) {
  for (const auto& iter : attr.getSlots())
    write(iter.getName(), pylir::Py::ObjectResource::get(),
          pylir::Py::DynamicType::get(attr.getContext()), iter.getValue());
}

} // namespace

//===----------------------------------------------------------------------===//
// GetSlotOp implementation
//===----------------------------------------------------------------------===//

void pylir::Py::GetSlotOp::replaceAggregate(
    mlir::OpBuilder&, mlir::Attribute key,
    llvm::function_ref<mlir::Value(mlir::Attribute,
                                   mlir::SideEffects::Resource*, mlir::Type)>
        read,
    llvm::function_ref<void(mlir::Attribute, mlir::SideEffects::Resource*,
                            mlir::Value)>) {
  replaceAllUsesWith(read(key, ObjectResource::get(), getType()));
}

//===----------------------------------------------------------------------===//
// SetSlotOp implementation
//===----------------------------------------------------------------------===//

void pylir::Py::SetSlotOp::replaceAggregate(
    mlir::OpBuilder&, mlir::Attribute key,
    llvm::function_ref<mlir::Value(mlir::Attribute,
                                   mlir::SideEffects::Resource*, mlir::Type)>,
    llvm::function_ref<void(mlir::Attribute, mlir::SideEffects::Resource*,
                            mlir::Value)>
        write) {
  write(key, ObjectResource::get(), getValue());
}

//===----------------------------------------------------------------------===//
// DictTryGetItemOp implementation
//===----------------------------------------------------------------------===//

mlir::FailureOr<mlir::Attribute> pylir::Py::DictTryGetItemOp::getSROAKey() {
  return getDictKey(getKey());
}

void pylir::Py::DictTryGetItemOp::replaceAggregate(
    mlir::OpBuilder&, mlir::Attribute key,
    llvm::function_ref<mlir::Value(mlir::Attribute,
                                   mlir::SideEffects::Resource*, mlir::Type)>
        read,
    llvm::function_ref<void(mlir::Attribute, mlir::SideEffects::Resource*,
                            mlir::Value)>) {
  replaceAllUsesWith(read(key, DictResource::get(), getType()));
}

//===----------------------------------------------------------------------===//
// DictSetItemOp implementation
//===----------------------------------------------------------------------===//

mlir::FailureOr<mlir::Attribute> pylir::Py::DictSetItemOp::getSROAKey() {
  return getDictKey(getKey());
}

void pylir::Py::DictSetItemOp::replaceAggregate(
    mlir::OpBuilder& builder, mlir::Attribute key,
    llvm::function_ref<mlir::Value(mlir::Attribute,
                                   mlir::SideEffects::Resource*, mlir::Type)>
        read,
    llvm::function_ref<void(mlir::Attribute, mlir::SideEffects::Resource*,
                            mlir::Value)>
        write) {
  auto size = read(nullptr, DictResource::get(), builder.getIndexType());
  auto value =
      read(key, DictResource::get(), builder.getType<pylir::Py::DynamicType>());
  auto didNotExist = builder.create<IsUnboundValueOp>(getLoc(), value);
  auto oneValue = builder.create<mlir::arith::ConstantIndexOp>(getLoc(), 1);
  auto incremented =
      builder.create<mlir::arith::AddIOp>(getLoc(), size, oneValue);
  write(nullptr, DictResource::get(),
        builder.create<mlir::arith::SelectOp>(getLoc(), didNotExist,
                                              incremented, size));
  write(key, DictResource::get(), getValue());
}

//===----------------------------------------------------------------------===//
// DictDelItemOp implementation
//===----------------------------------------------------------------------===//

mlir::FailureOr<mlir::Attribute> pylir::Py::DictDelItemOp::getSROAKey() {
  return getDictKey(getKey());
}

void pylir::Py::DictDelItemOp::replaceAggregate(
    mlir::OpBuilder& builder, mlir::Attribute key,
    llvm::function_ref<mlir::Value(mlir::Attribute,
                                   mlir::SideEffects::Resource*, mlir::Type)>
        read,
    llvm::function_ref<void(mlir::Attribute, mlir::SideEffects::Resource*,
                            mlir::Value)>
        write) {
  auto size = read(nullptr, DictResource::get(), builder.getIndexType());
  auto value =
      read(key, DictResource::get(), builder.getType<pylir::Py::DynamicType>());
  auto unbound =
      builder.create<ConstantOp>(getLoc(), builder.getAttr<UnboundAttr>());
  write(key, DictResource::get(), unbound);
  auto didNotExist = builder.create<IsUnboundValueOp>(getLoc(), value);
  auto oneValue = builder.create<mlir::arith::ConstantIndexOp>(getLoc(), 1);
  auto trueValue =
      builder.create<mlir::arith::ConstantIntOp>(getLoc(), true, 1);
  mlir::Value existed =
      builder.create<mlir::arith::XOrIOp>(getLoc(), didNotExist, trueValue);
  replaceAllUsesWith(existed);
  auto decremented =
      builder.create<mlir::arith::SubIOp>(getLoc(), size, oneValue);
  write(nullptr, DictResource::get(),
        builder.create<mlir::arith::SelectOp>(getLoc(), didNotExist, size,
                                              decremented));
}

//===----------------------------------------------------------------------===//
// DictLenOp implementation
//===----------------------------------------------------------------------===//

void pylir::Py::DictLenOp::replaceAggregate(
    ::mlir::OpBuilder&, ::mlir::Attribute,
    ::llvm::function_ref<mlir::Value(mlir::Attribute,
                                     mlir::SideEffects::Resource*, mlir::Type)>
        read,
    ::llvm::function_ref<void(mlir::Attribute, mlir::SideEffects::Resource*,
                              mlir::Value)>) {
  replaceAllUsesWith(
      read(nullptr, DictResource::get(), mlir::IndexType::get(getContext())));
}

//===----------------------------------------------------------------------===//
// MakeDictOp implementation
//===----------------------------------------------------------------------===//

mlir::LogicalResult pylir::Py::MakeDictOp::canParticipateInSROA() {
  return dictOpCanParticipateInSROA(*this);
}

void pylir::Py::MakeDictOp::replaceAggregate(
    mlir::OpBuilder& builder,
    llvm::function_ref<void(mlir::Attribute, mlir::SideEffects::Resource*,
                            mlir::Value)>
        write) {
  replaceDictAggregate(*this, builder, write);
}

//===----------------------------------------------------------------------===//
// MakeDictExOp implementation
//===----------------------------------------------------------------------===//

mlir::LogicalResult pylir::Py::MakeDictExOp::canParticipateInSROA() {
  return dictOpCanParticipateInSROA(*this);
}

void pylir::Py::MakeDictExOp::replaceAggregate(
    mlir::OpBuilder& builder,
    llvm::function_ref<void(mlir::Attribute, mlir::SideEffects::Resource*,
                            mlir::Value)>
        write) {
  replaceDictAggregate(*this, builder, write);
}

//===----------------------------------------------------------------------===//
// ListGetItemOp implementation
//===----------------------------------------------------------------------===//

void pylir::Py::ListGetItemOp::replaceAggregate(
    mlir::OpBuilder&, mlir::Attribute key,
    llvm::function_ref<mlir::Value(mlir::Attribute,
                                   mlir::SideEffects::Resource*, mlir::Type)>
        read,
    llvm::function_ref<void(mlir::Attribute, mlir::SideEffects::Resource*,
                            mlir::Value)>) {
  replaceAllUsesWith(read(key, ListResource::get(), getType()));
}

//===----------------------------------------------------------------------===//
// ListSetItemOp implementation
//===----------------------------------------------------------------------===//

void pylir::Py::ListSetItemOp::replaceAggregate(
    mlir::OpBuilder&, mlir::Attribute key,
    llvm::function_ref<mlir::Value(mlir::Attribute,
                                   mlir::SideEffects::Resource*, mlir::Type)>,
    llvm::function_ref<void(mlir::Attribute, mlir::SideEffects::Resource*,
                            mlir::Value)>
        write) {
  write(key, ListResource::get(), getElement());
}

//===----------------------------------------------------------------------===//
// ListLenOp implementation
//===----------------------------------------------------------------------===//

void pylir::Py::ListLenOp::replaceAggregate(
    mlir::OpBuilder&, mlir::Attribute,
    llvm::function_ref<mlir::Value(mlir::Attribute,
                                   mlir::SideEffects::Resource*, mlir::Type)>
        read,
    llvm::function_ref<void(mlir::Attribute, mlir::SideEffects::Resource*,
                            mlir::Value)>) {
  replaceAllUsesWith(read(nullptr, ListResource::get(), getType()));
}

//===----------------------------------------------------------------------===//
// ListResizeOp implementation
//===----------------------------------------------------------------------===//

void pylir::Py::ListResizeOp::replaceAggregate(
    mlir::OpBuilder&, mlir::Attribute,
    llvm::function_ref<mlir::Value(mlir::Attribute,
                                   mlir::SideEffects::Resource*, mlir::Type)>,
    llvm::function_ref<void(mlir::Attribute, mlir::SideEffects::Resource*,
                            mlir::Value)>
        write) {
  write(nullptr, ListResource::get(), getLength());
}

//===----------------------------------------------------------------------===//
// MakeListOp implementation
//===----------------------------------------------------------------------===//

mlir::LogicalResult pylir::Py::MakeListOp::canParticipateInSROA() {
  return mlir::success(getIterExpansion().empty());
}

void pylir::Py::MakeListOp::replaceAggregate(
    mlir::OpBuilder& builder,
    llvm::function_ref<void(mlir::Attribute, mlir::SideEffects::Resource*,
                            mlir::Value)>
        write) {
  replaceListAggregate(*this, write, builder);
}

//===----------------------------------------------------------------------===//
// MakeListExOp implementation
//===----------------------------------------------------------------------===//

mlir::LogicalResult pylir::Py::MakeListExOp::canParticipateInSROA() {
  return mlir::success(getIterExpansion().empty());
}

void pylir::Py::MakeListExOp::replaceAggregate(
    mlir::OpBuilder& builder,
    llvm::function_ref<void(mlir::Attribute, mlir::SideEffects::Resource*,
                            mlir::Value)>
        write) {
  replaceListAggregate(*this, write, builder);
}

//===----------------------------------------------------------------------===//
// SROAAttrInterface implementations
//===----------------------------------------------------------------------===//

void pylir::Py::ObjectAttr::destructureAggregate(
    llvm::function_ref<void(mlir::Attribute, mlir::SideEffects::Resource*,
                            mlir::Type, mlir::Attribute)>
        write) const {
  destructureSlots(*this, write);
}

void pylir::Py::IntAttr::destructureAggregate(
    llvm::function_ref<void(mlir::Attribute, mlir::SideEffects::Resource*,
                            mlir::Type, mlir::Attribute)>
        write) const {
  destructureSlots(*this, write);
}

void pylir::Py::BoolAttr::destructureAggregate(
    llvm::function_ref<void(mlir::Attribute, mlir::SideEffects::Resource*,
                            mlir::Type, mlir::Attribute)>
        write) const {
  destructureSlots(*this, write);
}

void pylir::Py::FloatAttr::destructureAggregate(
    llvm::function_ref<void(mlir::Attribute, mlir::SideEffects::Resource*,
                            mlir::Type, mlir::Attribute)>
        write) const {
  destructureSlots(*this, write);
}

void pylir::Py::StrAttr::destructureAggregate(
    llvm::function_ref<void(mlir::Attribute, mlir::SideEffects::Resource*,
                            mlir::Type, mlir::Attribute)>
        write) const {
  destructureSlots(*this, write);
}

void pylir::Py::TupleAttr::destructureAggregate(
    llvm::function_ref<void(mlir::Attribute, mlir::SideEffects::Resource*,
                            mlir::Type, mlir::Attribute)>
        write) const {
  destructureSlots(*this, write);
}

void pylir::Py::ListAttr::destructureAggregate(
    llvm::function_ref<void(mlir::Attribute, mlir::SideEffects::Resource*,
                            mlir::Type, mlir::Attribute)>
        write) const {
  destructureSlots(*this, write);
  auto indexType = mlir::IndexType::get(getContext());
  write(nullptr, ListResource::get(), indexType,
        mlir::IntegerAttr::get(indexType, getElements().size()));
  for (auto [index, iter] : llvm::enumerate(getElements()))
    write(mlir::IntegerAttr::get(indexType, index), ListResource::get(),
          DynamicType::get(getContext()), iter);
}

void pylir::Py::DictAttr::destructureAggregate(
    llvm::function_ref<void(mlir::Attribute, mlir::SideEffects::Resource*,
                            mlir::Type, mlir::Attribute)>
        write) const {
  destructureSlots(*this, write);
  auto indexType = mlir::IndexType::get(getContext());
  write(nullptr, DictResource::get(), indexType,
        mlir::IntegerAttr::get(indexType, getKeyValuePairs().size()));
  for (auto [key, value] : getKeyValuePairs())
    write(mlir::cast<EqualsAttrInterface>(key).getCanonicalAttribute(),
          DictResource::get(), DynamicType::get(getContext()), value);
}

void pylir::Py::FunctionAttr::destructureAggregate(
    llvm::function_ref<void(mlir::Attribute, mlir::SideEffects::Resource*,
                            mlir::Type, mlir::Attribute)>
        write) const {
  destructureSlots(*this, write);
}

void pylir::Py::TypeAttr::destructureAggregate(
    llvm::function_ref<void(mlir::Attribute, mlir::SideEffects::Resource*,
                            mlir::Type, mlir::Attribute)>
        write) const {
  destructureSlots(*this, write);
}
