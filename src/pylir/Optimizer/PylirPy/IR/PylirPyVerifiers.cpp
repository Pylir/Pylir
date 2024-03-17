//  Licensed under the Apache License v2.0 with LLVM Exceptions.
//  See https://llvm.org/LICENSE.txt for license information.
//  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <llvm/ADT/TypeSwitch.h>

#include "PylirPyAttributes.hpp"
#include "PylirPyOps.hpp"
#include "Value.hpp"

using namespace mlir;
using namespace pylir;
using namespace pylir::Py;

namespace {

template <class SymbolOp>
mlir::FailureOr<SymbolOp>
verifySymbolUse(mlir::Operation* op, mlir::SymbolRefAttr name,
                mlir::SymbolTableCollection& symbolTable,
                llvm::StringRef kindName = SymbolOp::getOperationName()) {
  if (auto* symbol = symbolTable.lookupNearestSymbolFrom(op, name)) {
    auto casted = mlir::dyn_cast<SymbolOp>(symbol);
    if (!casted)
      return op->emitError("Expected '")
             << name << "' to be of kind '" << kindName << "', not '"
             << symbol->getName() << "'";

    return casted;
  }
  return op->emitOpError("Failed to find symbol named '") << name << "'";
}

LogicalResult verifyCall(SymbolTableCollection& symbolTable, Operation* call,
                         ValueRange callOperands, FlatSymbolRefAttr callee) {
  auto funcOp =
      symbolTable.lookupNearestSymbolFrom<FunctionOpInterface>(call, callee);
  if (!funcOp)
    return call->emitOpError("failed to find function named '")
           << callee << "'";

  ArrayRef<Type> argumentTypes = funcOp.getArgumentTypes();
  SmallVector<Type> operandTypes;
  for (auto iter : callOperands)
    operandTypes.push_back(iter.getType());

  if (!std::equal(argumentTypes.begin(), argumentTypes.end(),
                  operandTypes.begin(), operandTypes.end()))
    return call->emitOpError(
               "call operand types are not compatible with argument types of '")
           << callee << "'";

  ArrayRef<Type> resultTypes = funcOp.getResultTypes();
  auto callResults = call->getResultTypes();
  if (!std::equal(resultTypes.begin(), resultTypes.end(), callResults.begin(),
                  callResults.end()))
    return call->emitOpError("call result types '")
           << callResults << "' are not compatible with output types "
           << resultTypes << " of '" << callee << "'";

  return success();
}

} // namespace

//===----------------------------------------------------------------------===//
// DictAttr verification
//===----------------------------------------------------------------------===//

LogicalResult DictAttr::verifyStructure(Operation* op,
                                        SymbolTableCollection&) const {
  for (auto [canonicalKey, index] : llvm::make_filter_range(
           getNormalizedKeysInternal(), [](auto pair) { return pair.first; })) {
    auto [key, value] = getKeyValuePairs()[index];
    auto equalsAttrInterface = llvm::dyn_cast<EqualsAttrInterface>(key);
    if (!equalsAttrInterface)
      return op->emitError() << "Expected key in '" << DictAttr::getMnemonic()
                             << "' to implement 'EqualsAttrInterface'";

    if (canonicalKey != equalsAttrInterface.getCanonicalAttribute())
      return op->emitError()
             << "Incorrect normalized key entry '" << canonicalKey
             << "' for key-value pair '(" << key << ", " << value << ")'";
  }

  return success();
}

//===----------------------------------------------------------------------===//
// FunctionAttr verification
//===----------------------------------------------------------------------===//

LogicalResult
FunctionAttr::verifyStructure(Operation* op,
                              SymbolTableCollection& collection) const {
  if (failed(verifySymbolUse<FunctionOpInterface>(op, getValue(), collection,
                                                  "FunctionOpInterface")))
    return failure();

  if (auto ref = llvm::dyn_cast_or_null<GlobalValueAttr>(getKwDefaults());
      !ref || ref.getName() != Builtins::None.name)
    if (!llvm::isa<DictAttrInterface>(getKwDefaults()))
      return op->emitError(
          "Expected __kwdefaults__ to refer to a dictionary\n");

  if (auto ref = llvm::dyn_cast_or_null<GlobalValueAttr>(getDefaults());
      !ref || ref.getName() != Builtins::None.name)
    if (!llvm::dyn_cast<TupleAttrInterface>(getDefaults()))
      return op->emitError("Expected __defaults__ to refer to a tuple\n");

  if (getDict())
    if (!llvm::isa<DictAttrInterface>(getDict()))
      return op->emitError("Expected __dict__ to refer to a dictionary\n");

  return success();
}

//===----------------------------------------------------------------------===//
// TypeAttr verification
//===----------------------------------------------------------------------===//

LogicalResult Py::TypeAttr::verifyStructure(Operation* op,
                                            SymbolTableCollection&) const {
  auto mro = llvm::dyn_cast<TupleAttrInterface>(getMroTuple());
  if (!mro)
    return op->emitOpError("Expected MRO to refer to a tuple\n");

  if (!llvm::all_of(getInstanceSlots(),
                    [](Attribute attr) { return llvm::isa<StrAttr>(attr); }))
    return op->emitError(
        "Expected 'instance_slots' to refer to a tuple of strings\n");

  return success();
}

//===----------------------------------------------------------------------===//
// MakeFuncOp verification
//===----------------------------------------------------------------------===//

mlir::LogicalResult pylir::Py::MakeFuncOp::verifySymbolUses(
    ::mlir::SymbolTableCollection& symbolTable) {
  if (!symbolTable.lookupNearestSymbolFrom(*this, getFunctionAttr()))
    return emitOpError("Failed to find symbol named '")
           << getFunctionAttr() << "'";
  return success();
}

//===----------------------------------------------------------------------===//
// FunctionGetClosureArgOp verification
//===----------------------------------------------------------------------===//

LogicalResult FunctionGetClosureArgOp::verify() {
  if (getIndex() >= getClosureTypes().size())
    return emitOpError("index '") << getIndex() << "' out of bounds";
  return success();
}

//===----------------------------------------------------------------------===//
// UnpackOp verification
//===----------------------------------------------------------------------===//

mlir::LogicalResult pylir::Py::UnpackOp::verify() {
  if (!getAfter().empty() && !getRest())
    return emitOpError(
        "'after_rest' results specified, without a rest argument");

  return mlir::success();
}

//===----------------------------------------------------------------------===//
// UnpackExOp verification
//===----------------------------------------------------------------------===//

mlir::LogicalResult pylir::Py::UnpackExOp::verify() {
  if (!getAfter().empty() && !getRest())
    return emitOpError(
        "'after_rest' results specified, without a rest argument");

  return mlir::success();
}

//===----------------------------------------------------------------------===//
// GlobalOp verification
//===----------------------------------------------------------------------===//

LogicalResult GlobalOp::verify() {
  if (!getInitializerAttr())
    return success();

  return llvm::TypeSwitch<Type, LogicalResult>(getType())
      .Case([&](DynamicType) -> LogicalResult {
        if (!isa<ObjectAttrInterface, GlobalValueAttr, UnboundAttr>(
                getInitializerAttr()))
          return emitOpError("Expected initializer of type "
                             "'ObjectAttrInterface' or 'GlobalValueAttr' "
                             "to global value");
        return success();
      })
      .Case([&](IndexType) -> LogicalResult {
        if (!isa<IntegerAttr>(getInitializerAttr()))
          return emitOpError("Expected integer attribute initializer");

        return success();
      })
      .Case([&](FloatType) -> LogicalResult {
        if (!isa<FloatAttr>(getInitializerAttr()))
          return emitOpError("Expected float attribute initializer");

        return success();
      });
}

//===----------------------------------------------------------------------===//
// LoadOp verification
//===----------------------------------------------------------------------===//

mlir::LogicalResult pylir::Py::LoadOp::verifySymbolUses(
    ::mlir::SymbolTableCollection& symbolTable) {
  return verifySymbolUse<GlobalOp>(*this, getGlobalAttr(), symbolTable,
                                   GlobalOp::getOperationName());
}

//===----------------------------------------------------------------------===//
// StoreOp verification
//===----------------------------------------------------------------------===//

mlir::LogicalResult pylir::Py::StoreOp::verifySymbolUses(
    ::mlir::SymbolTableCollection& symbolTable) {
  auto global = verifySymbolUse<GlobalOp>(*this, getGlobalAttr(), symbolTable,
                                          GlobalOp::getOperationName());
  if (mlir::failed(global))
    return mlir::failure();

  if (global->getType() != getValue().getType())
    return emitOpError("Type of value to store '")
           << getValue().getType() << "' does not match type of global '"
           << global->getSymName() << " : " << global->getType()
           << "' to store into";

  return mlir::success();
}

//===----------------------------------------------------------------------===//
// CallOp verification
//===----------------------------------------------------------------------===//

mlir::LogicalResult pylir::Py::CallOp::verifySymbolUses(
    ::mlir::SymbolTableCollection& symbolTable) {
  return verifyCall(symbolTable, *this, getCallOperands(), getCalleeAttr());
}

//===----------------------------------------------------------------------===//
// InvokeOp verification
//===----------------------------------------------------------------------===//

mlir::LogicalResult pylir::Py::InvokeOp::verifySymbolUses(
    ::mlir::SymbolTableCollection& symbolTable) {
  return verifyCall(symbolTable, *this, getCallOperands(), getCalleeAttr());
}

//===----------------------------------------------------------------------===//
// ReturnOp verification
//===----------------------------------------------------------------------===//

mlir::LogicalResult pylir::Py::ReturnOp::verify() {
  auto funcOp = (*this)->getParentOfType<FuncOp>();
  llvm::ArrayRef<mlir::Type> resultTypes = funcOp.getResultTypes();
  auto argumentTypes = llvm::to_vector(llvm::map_range(
      getArguments(), [](mlir::Value value) { return value.getType(); }));
  if (!std::equal(resultTypes.begin(), resultTypes.end(), argumentTypes.begin(),
                  argumentTypes.end()))
    return emitOpError("return value types '")
           << argumentTypes << "' incompatible with result types of function '"
           << resultTypes << "'";

  return mlir::success();
}
