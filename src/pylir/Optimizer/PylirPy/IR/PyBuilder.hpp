//  Licensed under the Apache License v2.0 with LLVM Exceptions.
//  See https://llvm.org/LICENSE.txt for license information.
//  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/IR/Builders.h>

#include <pylir/Support/BigInt.hpp>

#include "PylirPyOps.hpp"

namespace pylir {
class PyBuilder : public mlir::OpBuilder {
  mlir::Location m_loc;

  void implementBlock(mlir::Block* block) {
    if (auto* next = getBlock()->getNextNode())
      block->insertBefore(next);
    else
      getBlock()->getParent()->push_back(block);

    setInsertionPointToStart(block);
  }

public:
  explicit PyBuilder(mlir::Operation* operationBefore)
      : mlir::OpBuilder(operationBefore), m_loc(operationBefore->getLoc()) {}

  explicit PyBuilder(mlir::MLIRContext* context,
                     std::optional<mlir::Location> loc = {})
      : mlir::OpBuilder(context), m_loc(loc.value_or(getUnknownLoc())) {}

  [[nodiscard]] mlir::Location getCurrentLoc() const {
    return m_loc;
  }

  void setCurrentLoc(mlir::Location loc) {
    m_loc = loc;
  }

  Py::UnboundAttr getUnboundAttr() {
    return Py::UnboundAttr::get(getContext());
  }

  Py::ObjectAttr getObjectAttr(Py::ObjectBaseAttribute type,
                               mlir::DictionaryAttr slots = {}) {
    return Py::ObjectAttr::get(context, type, slots);
  }

  Py::TypeAttr getTypeAttr(mlir::Attribute mroTuple = {},
                           pylir::Py::TupleAttr instanceSlots = {},
                           mlir::DictionaryAttr slots = {}) {
    return Py::TypeAttr::get(context, mroTuple, instanceSlots, slots);
  }

  Py::IntAttr getIntAttr(const BigInt& bigInt) {
    return Py::IntAttr::get(getContext(), bigInt);
  }

  Py::BoolAttr getPyBoolAttr(bool value) {
    return Py::BoolAttr::get(getContext(), value);
  }

  Py::FloatAttr getFloatAttr(double value) {
    return Py::FloatAttr::get(getContext(), llvm::APFloat(value));
  }

  Py::StrAttr getStrAttr(llvm::StringRef value) {
    return Py::StrAttr::get(getContext(), value);
  }

  Py::TupleAttr getTupleAttr(llvm::ArrayRef<mlir::Attribute> value = {}) {
    return Py::TupleAttr::get(getContext(), value);
  }

  Py::ListAttr getListAttr(llvm::ArrayRef<mlir::Attribute> value = {}) {
    return Py::ListAttr::get(getContext(), value);
  }

  Py::DictAttr getDictAttr(llvm::ArrayRef<Py::DictAttr::Entry> value = {}) {
    return Py::DictAttr::get(getContext(), value);
  }

  Py::FunctionAttr getFunctionAttr(mlir::FlatSymbolRefAttr value,
                                   mlir::Attribute qualName = {},
                                   mlir::Attribute defaults = {},
                                   mlir::Attribute kwDefaults = {},
                                   mlir::Attribute dict = {}) {
    return Py::FunctionAttr::get(context, value, qualName, defaults, kwDefaults,
                                 dict);
  }

  Py::DynamicType getDynamicType() {
    return getType<Py::DynamicType>();
  }

  using mlir::OpBuilder::create;

  template <class Op>
  Op create() {
    return mlir::OpBuilder::create<Op>(getCurrentLoc());
  }

  template <class Op, class First, class... Args>
  std::enable_if_t<!std::is_same_v<std::decay_t<First>, mlir::Location>, Op>
  create(First&& first, Args&&... args) {
    return mlir::OpBuilder::create<Op>(getCurrentLoc(),
                                       std::forward<First>(first),
                                       std::forward<Args>(args)...);
  }

#define BUILTIN(name, str, ...)                           \
  Py::GlobalValueAttr get##name##Builtin() {              \
    return Py::GlobalValueAttr::get(getContext(), (str)); \
  }
#include <pylir/Interfaces/BuiltinsModule.def>

#define BUILTIN(name, ...)                               \
  Py::ConstantOp create##name##Ref() {                   \
    return create<Py::ConstantOp>(get##name##Builtin()); \
  }
#include <pylir/Interfaces/BuiltinsModule.def>

#define COMPILER_BUILTIN_TERNARY_OP(name, slotName)                          \
  mlir::Value createPylir##name##Intrinsic(                                  \
      mlir::Value first, mlir::Value second, mlir::Value third,              \
      mlir::Block* exceptBlock = nullptr) {                                  \
    if (!exceptBlock)                                                        \
      return create<Py::CallOp>(getDynamicType(),                            \
                                COMPILER_BUILTIN_SLOT_TO_API_NAME(slotName), \
                                mlir::ValueRange{first, second, third})      \
          .getResult(0);                                                     \
                                                                             \
    auto* happyPath = new mlir::Block;                                       \
    auto op = create<Py::InvokeOp>(                                          \
        getDynamicType(), COMPILER_BUILTIN_SLOT_TO_API_NAME(slotName),       \
        mlir::ValueRange{first, second, third}, mlir::ValueRange{},          \
        mlir::ValueRange{}, happyPath, exceptBlock);                         \
    implementBlock(happyPath);                                               \
    return op.getResult(0);                                                  \
  }

#define COMPILER_BUILTIN_BIN_OP(name, slotName)                               \
  mlir::Value createPylir##name##Intrinsic(                                   \
      mlir::Value lhs, mlir::Value rhs, mlir::Block* exceptBlock = nullptr) { \
    if (!exceptBlock)                                                         \
      return create<Py::CallOp>(getDynamicType(),                             \
                                COMPILER_BUILTIN_SLOT_TO_API_NAME(slotName),  \
                                mlir::ValueRange{lhs, rhs})                   \
          .getResult(0);                                                      \
                                                                              \
    auto* happyPath = new mlir::Block;                                        \
    auto op = create<Py::InvokeOp>(                                           \
        getDynamicType(), COMPILER_BUILTIN_SLOT_TO_API_NAME(slotName),        \
        mlir::ValueRange{lhs, rhs}, mlir::ValueRange{}, mlir::ValueRange{},   \
        happyPath, exceptBlock);                                              \
    implementBlock(happyPath);                                                \
    return op.getResult(0);                                                   \
  }

#define COMPILER_BUILTIN_UNARY_OP(name, slotName)                            \
  mlir::Value createPylir##name##Intrinsic(                                  \
      mlir::Value val, mlir::Block* exceptBlock = nullptr) {                 \
    if (!exceptBlock)                                                        \
      return create<Py::CallOp>(getDynamicType(),                            \
                                COMPILER_BUILTIN_SLOT_TO_API_NAME(slotName), \
                                val)                                         \
          .getResult(0);                                                     \
                                                                             \
    auto* happyPath = new mlir::Block;                                       \
    auto op = create<Py::InvokeOp>(                                          \
        getDynamicType(), COMPILER_BUILTIN_SLOT_TO_API_NAME(slotName), val,  \
        mlir::ValueRange{}, mlir::ValueRange{}, happyPath, exceptBlock);     \
    implementBlock(happyPath);                                               \
    return op.getResult(0);                                                  \
  }

#include <pylir/Interfaces/CompilerBuiltins.def>

  Py::ConstantOp createConstant(mlir::Attribute constant) {
    if (auto ref = mlir::dyn_cast<Py::GlobalValueAttr>(constant))
      return create<Py::ConstantOp>(ref);

    if (auto unbound = mlir::dyn_cast<Py::UnboundAttr>(constant))
      return create<Py::ConstantOp>(unbound);

    return create<Py::ConstantOp>(
        mlir::cast<Py::ObjectAttrInterface>(constant));
  }

  template <std::size_t n>
  Py::ConstantOp createConstant(const char (&c)[n]) {
    return create<Py::ConstantOp>(getStrAttr(c));
  }

  Py::ConstantOp createConstant(llvm::StringRef string) {
    return create<Py::ConstantOp>(getStrAttr(string));
  }

  template <class Integer>
  std::enable_if_t<std::is_integral_v<std::decay_t<Integer>> &&
                   !std::is_same_v<bool, std::decay_t<Integer>>>
      createConstant(Integer) = delete;

  Py::ConstantOp createConstant(bool value) {
    return create<Py::ConstantOp>(getPyBoolAttr(value));
  }

  Py::ConstantOp createConstant(const BigInt& bigInt) {
    return create<Py::ConstantOp>(getIntAttr(bigInt));
  }

  Py::ConstantOp createConstant(double value) {
    return create<Py::ConstantOp>(getFloatAttr(value));
  }

  Py::TypeOfOp createTypeOf(mlir::Value value) {
    return create<Py::TypeOfOp>(value);
  }

  template <class T, std::enable_if_t<std::disjunction_v<
                         std::is_integral<T>, std::is_enum<T>>>* = nullptr>
  Py::GetSlotOp createGetSlot(mlir::Value object, T slot) {
    return create<Py::GetSlotOp>(object, create<mlir::arith::ConstantIndexOp>(
                                             static_cast<std::size_t>(slot)));
  }

  template <class T, std::enable_if_t<std::disjunction_v<
                         std::is_integral<T>, std::is_enum<T>>>* = nullptr>
  Py::SetSlotOp createSetSlot(mlir::Value object, T slot, mlir::Value value) {
    return create<Py::SetSlotOp>(
        object,
        create<mlir::arith::ConstantIndexOp>(static_cast<std::size_t>(slot)),
        value);
  }

  Py::DictTryGetItemOp createDictTryGetItem(mlir::Value dict, mlir::Value index,
                                            mlir::Value hash) {
    return create<Py::DictTryGetItemOp>(dict, index, hash);
  }

  Py::DictSetItemOp createDictSetItem(mlir::Value dict, mlir::Value key,
                                      mlir::Value hash, mlir::Value value) {
    return create<Py::DictSetItemOp>(dict, key, hash, value);
  }

  Py::DictDelItemOp createDictDelItem(mlir::Value dict, mlir::Value key,
                                      mlir::Value hash) {
    return create<Py::DictDelItemOp>(dict, key, hash);
  }

  Py::DictLenOp createDictLen(mlir::Value dict) {
    return create<Py::DictLenOp>(dict);
  }

  Py::TupleGetItemOp createTupleGetItem(mlir::Value tuple, mlir::Value index) {
    return create<Py::TupleGetItemOp>(tuple, index);
  }

  Py::TupleLenOp createTupleLen(mlir::Value tuple) {
    return create<Py::TupleLenOp>(tuple);
  }

  Py::TuplePrependOp createTuplePrepend(mlir::Value element,
                                        mlir::Value tuple) {
    return create<Py::TuplePrependOp>(element, tuple);
  }

  Py::TupleDropFrontOp createTupleDropFront(mlir::Value count,
                                            mlir::Value tuple) {
    return create<Py::TupleDropFrontOp>(count, tuple);
  }

  Py::ListResizeOp createListResize(mlir::Value list, mlir::Value length) {
    return create<Py::ListResizeOp>(list, length);
  }

  Py::ListSetItemOp createListSetItem(mlir::Value list, mlir::Value index,
                                      mlir::Value element) {
    return create<Py::ListSetItemOp>(list, index, element);
  }

  Py::ListLenOp createListLen(mlir::Value list) {
    return create<Py::ListLenOp>(list);
  }

  Py::ListToTupleOp createListToTuple(mlir::Value list) {
    return create<Py::ListToTupleOp>(list);
  }

  Py::FunctionCallOp createFunctionCall(mlir::Value function,
                                        llvm::ArrayRef<mlir::Value> arguments) {
    return create<Py::FunctionCallOp>(function, arguments);
  }

  Py::ObjectHashOp createObjectHash(mlir::Value object) {
    return create<Py::ObjectHashOp>(object);
  }

  Py::ObjectIdOp createObjectId(mlir::Value object) {
    return create<Py::ObjectIdOp>(object);
  }

  Py::TypeMROOp createTypeMRO(mlir::Value typeObject) {
    return create<Py::TypeMROOp>(typeObject);
  }

  Py::StrCopyOp createStrCopy(mlir::Value string, mlir::Value typeObject) {
    return create<Py::StrCopyOp>(string, typeObject);
  }

  Py::StrHashOp createStrHash(mlir::Value string) {
    return create<Py::StrHashOp>(string);
  }

  Py::StrEqualOp createStrEqual(mlir::Value lhs, mlir::Value rhs) {
    return create<Py::StrEqualOp>(lhs, rhs);
  }

  Py::StrConcatOp createStrConcat(llvm::ArrayRef<mlir::Value> strings) {
    return create<Py::StrConcatOp>(strings);
  }

  template <class T, std::enable_if_t<std::disjunction_v<
                         std::is_integral<T>, std::is_enum<T>>>* = nullptr>
  Py::MROLookupOp createMROLookup(mlir::Value mroTuple, T slot) {
    return create<Py::MROLookupOp>(
        mroTuple,
        create<mlir::arith::ConstantIndexOp>(static_cast<std::size_t>(slot)));
  }

  Py::TupleContainsOp createTupleContains(mlir::Value tuple,
                                          mlir::Value element) {
    return create<Py::TupleContainsOp>(tuple, element);
  }

  Py::MakeTupleOp createMakeTuple() {
    return create<Py::MakeTupleOp>(llvm::ArrayRef<Py::IterArg>{});
  }

  mlir::Value createMakeTuple(llvm::ArrayRef<Py::IterArg> args,
                              mlir::Block* unwindPath) {
    if (!unwindPath || llvm::all_of(args, [](const Py::IterArg& arg) {
          return std::holds_alternative<mlir::Value>(arg);
        }))
      return create<Py::MakeTupleOp>(args);

    auto* happyPath = new mlir::Block;
    auto op = create<Py::MakeTupleExOp>(args, happyPath, mlir::ValueRange{},
                                        unwindPath, mlir::ValueRange{});
    implementBlock(happyPath);
    return op;
  }

  Py::MakeListOp createMakeList() {
    return create<Py::MakeListOp>(llvm::ArrayRef<Py::IterArg>{});
  }

  mlir::Value createMakeList(llvm::ArrayRef<Py::IterArg> args,
                             mlir::Block* unwindPath) {
    if (!unwindPath || llvm::all_of(args, [](const Py::IterArg& arg) {
          return std::holds_alternative<mlir::Value>(arg);
        }))
      return create<Py::MakeListOp>(args);

    auto* happyPath = new mlir::Block;
    auto op = create<Py::MakeListExOp>(args, happyPath, mlir::ValueRange{},
                                       unwindPath, mlir::ValueRange{});
    implementBlock(happyPath);
    return op;
  }

  Py::MakeSetOp createMakeSet() {
    return create<Py::MakeSetOp>(llvm::ArrayRef<Py::IterArg>{});
  }

  mlir::Value createMakeSet(llvm::ArrayRef<Py::IterArg> args,
                            mlir::Block* unwindPath) {
    if (!unwindPath || llvm::all_of(args, [](const Py::IterArg& arg) {
          return std::holds_alternative<mlir::Value>(arg);
        }))
      return create<Py::MakeSetOp>(args);

    auto* happyPath = new mlir::Block;
    auto op = create<Py::MakeSetExOp>(args, happyPath, mlir::ValueRange{},
                                      unwindPath, mlir::ValueRange{});
    implementBlock(happyPath);
    return op;
  }

  Py::MakeDictOp createMakeDict() {
    return create<Py::MakeDictOp>(llvm::ArrayRef<Py::DictArg>{});
  }

  mlir::Value createMakeDict(llvm::ArrayRef<Py::DictArg> args,
                             mlir::Block* unwindPath) {
    if (!unwindPath || llvm::all_of(args, [](const Py::DictArg& arg) {
          return std::holds_alternative<Py::DictEntry>(arg);
        }))
      return create<Py::MakeDictOp>(args);

    auto* happyPath = new mlir::Block;
    auto op = create<Py::MakeDictExOp>(args, happyPath, mlir::ValueRange{},
                                       unwindPath, mlir::ValueRange{});
    implementBlock(happyPath);
    return op;
  }

  Py::MakeFuncOp createMakeFunc(mlir::FlatSymbolRefAttr function) {
    return create<Py::MakeFuncOp>(function, mlir::ValueRange());
  }

  Py::MakeObjectOp createMakeObject(mlir::Value typeObject) {
    return create<Py::MakeObjectOp>(typeObject);
  }

  mlir::Operation* createUnpack(std::size_t count,
                                std::optional<std::size_t> restIndex,
                                mlir::Value iterable, mlir::Block* unwindPath) {
    if (!unwindPath)
      return create<Py::UnpackOp>(count, restIndex, iterable);

    auto* happyPath = new mlir::Block;
    auto op = create<Py::UnpackExOp>(count, restIndex, iterable, happyPath,
                                     mlir::ValueRange{}, unwindPath,
                                     mlir::ValueRange{});
    implementBlock(happyPath);
    return op;
  }

  Py::IsOp createIs(mlir::Value lhs, mlir::Value rhs) {
    return create<Py::IsOp>(lhs, rhs);
  }

  Py::BoolToI1Op createBoolToI1(mlir::Value input) {
    return create<Py::BoolToI1Op>(input);
  }

  Py::BoolFromI1Op createBoolFromI1(mlir::Value input) {
    return create<Py::BoolFromI1Op>(input);
  }

  Py::IntFromSignedOp createIntFromSigned(mlir::Value integer) {
    return create<Py::IntFromSignedOp>(integer);
  }

  Py::IntFromUnsignedOp createIntFromUnsigned(mlir::Value integer) {
    return create<Py::IntFromUnsignedOp>(integer);
  }

  Py::IntToIndexOp createIntToIndex(mlir::Value object) {
    return create<Py::IntToIndexOp>(object);
  }

  Py::IntCmpOp createIntCmp(Py::IntCmpKind kind, mlir::Value lhs,
                            mlir::Value rhs) {
    return create<Py::IntCmpOp>(kind, lhs, rhs);
  }

  Py::IntAddOp createIntAdd(mlir::Value lhs, mlir::Value rhs) {
    return create<Py::IntAddOp>(lhs, rhs);
  }

  Py::IntToStrOp createIntToStr(mlir::Value object) {
    return create<Py::IntToStrOp>(object);
  }

  Py::GlobalValueAttr
  createGlobalValue(llvm::StringRef symbolName, bool constant = false,
                    Py::ConcreteObjectAttribute initializer = {},
                    bool external = false) {
    auto result = getAttr<Py::GlobalValueAttr>(symbolName);
    result.setInitializer(initializer);
    result.setConstant(constant);
    if (external)
      create<Py::ExternalOp>(symbolName, result);

    return result;
  }

  Py::GlobalOp createGlobal(llvm::StringRef symbolName) {
    return create<Py::GlobalOp>(symbolName, getStringAttr("private"),
                                getDynamicType(), nullptr);
  }

  Py::StoreOp createStore(mlir::Value value, mlir::FlatSymbolRefAttr handle) {
    return create<Py::StoreOp>(value, handle);
  }

  Py::LoadOp createLoad(mlir::FlatSymbolRefAttr handle) {
    return create<Py::LoadOp>(getDynamicType(), handle);
  }

  Py::IsUnboundValueOp createIsUnboundValue(mlir::Value value) {
    return create<Py::IsUnboundValueOp>(value);
  }

  Py::RaiseOp createRaise(mlir::Value exception) {
    return create<Py::RaiseOp>(exception);
  }

  Py::InvokeOp createInvoke(llvm::ArrayRef<mlir::Type> resultTypes,
                            mlir::FlatSymbolRefAttr callee,
                            llvm::ArrayRef<mlir::Value> operands,
                            mlir::Block* happyPath,
                            llvm::ArrayRef<mlir::Value> normalOperands,
                            mlir::Block* unwindPath,
                            llvm::ArrayRef<mlir::Value> unwindOperands) {
    return create<Py::InvokeOp>(resultTypes, callee, operands, normalOperands,
                                unwindOperands, happyPath, unwindPath);
  }

  Py::FunctionInvokeOp createFunctionInvoke(
      mlir::Value callee, llvm::ArrayRef<mlir::Value> operands,
      mlir::Block* happyPath, llvm::ArrayRef<mlir::Value> normalOperands,
      mlir::Block* unwindPath, llvm::ArrayRef<mlir::Value> unwindOperands) {
    return create<Py::FunctionInvokeOp>(callee, operands, normalOperands,
                                        unwindOperands, happyPath, unwindPath);
  }
};
} // namespace pylir
