//  Licensed under the Apache License v2.0 with LLVM Exceptions.
//  See https://llvm.org/LICENSE.txt for license information.
//  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "PylirPyOps.hpp"

#include <llvm/ADT/TypeSwitch.h>

#include "PylirPyAttributes.hpp"
#include "Value.hpp"

using namespace mlir;
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
      return op->emitOpError("Expected '")
             << name << "' to be of kind '" << kindName << "', not '"
             << symbol->getName() << "'";

    return casted;
  }
  return op->emitOpError("Failed to find symbol named '") << name << "'";
}

mlir::LogicalResult verify(mlir::Operation* op, mlir::Attribute attribute,
                           mlir::SymbolTableCollection& collection) {
  if (isa<GlobalValueAttr>(attribute))
    return success();
  auto object = attribute.dyn_cast<pylir::Py::ObjectAttrInterface>();
  if (!object) {
    if (!attribute.isa<pylir::Py::UnboundAttr, pylir::Py::GlobalValueAttr>())
      return op->emitOpError("Not allowed attribute '")
             << attribute << "' found\n";

    return mlir::success();
  }
  if (mlir::failed(::verify(op, object.getTypeObject(), collection)))
    return mlir::failure();

  if (auto constantObjectAttr =
          mlir::dyn_cast<pylir::Py::ConstObjectAttrInterface>(attribute))
    for (auto iter : constantObjectAttr.getSlots())
      if (mlir::failed(verify(op, iter.getValue(), collection)))
        return mlir::failure();

  return llvm::TypeSwitch<mlir::Attribute, mlir::LogicalResult>(object)
      .Case<pylir::Py::TupleAttr, pylir::Py::ListAttr>([&](auto sequence) {
        for (auto iter : sequence.getElements())
          if (mlir::failed(verify(op, iter, collection)))
            return mlir::failure();

        return mlir::success();
      })
      .Case([&](pylir::Py::DictAttr dict) -> mlir::LogicalResult {
        for (auto [key, value] : dict.getKeyValuePairs()) {
          if (mlir::failed(verify(op, key, collection)))
            return mlir::failure();

          if (mlir::failed(verify(op, value, collection)))
            return mlir::failure();
        }
        for (auto [canonicalKey, index] :
             llvm::make_filter_range(dict.getNormalizedKeysInternal(),
                                     [](auto pair) { return pair.first; })) {
          auto [key, value] = dict.getKeyValuePairs()[index];
          auto equalsAttrInterface = dyn_cast<EqualsAttrInterface>(key);
          if (!equalsAttrInterface)
            return op->emitOpError("Expected key in '")
                   << DictAttr::getMnemonic()
                   << "' to implement 'EqualsAttrInterface'";

          if (canonicalKey != equalsAttrInterface.getCanonicalAttribute())
            return op->emitOpError("Incorrect normalized key entry '")
                   << canonicalKey << "' for key-value pair '(" << key << ", "
                   << value << ")'";
        }
        return mlir::success();
      })
      .Case([&](pylir::Py::FunctionAttr functionAttr) -> mlir::LogicalResult {
        if (!functionAttr.getValue())
          return op->emitOpError(
              "Expected function attribute to contain a symbol reference\n");

        if (mlir::failed(verifySymbolUse<mlir::FunctionOpInterface>(
                op, functionAttr.getValue(), collection,
                "FunctionOpInterface")))
          return mlir::failure();

        // These shouldn't return failure as they are just fancy slot accessors
        // (for now), which have been verified above.
        if (auto ref = functionAttr.getKwDefaults()
                           .dyn_cast_or_null<pylir::Py::GlobalValueAttr>();
            !ref || ref.getName() != pylir::Builtins::None.name)
          if (!isa<DictAttrInterface>(functionAttr.getKwDefaults()))
            return op->emitOpError(
                "Expected __kwdefaults__ to refer to a dictionary\n");

        if (auto ref = functionAttr.getDefaults()
                           .dyn_cast_or_null<pylir::Py::GlobalValueAttr>();
            !ref || ref.getName() != pylir::Builtins::None.name)
          if (!dyn_cast<TupleAttrInterface>(functionAttr.getDefaults()))
            return op->emitOpError(
                "Expected __defaults__ to refer to a tuple\n");

        if (functionAttr.getDict())
          if (!isa<DictAttrInterface>(functionAttr.getDict()))
            return op->emitOpError(
                "Expected __dict__ to refer to a dictionary\n");

        return mlir::success();
      })
      .Case([&](pylir::Py::TypeAttr typeAttr) -> mlir::LogicalResult {
        if (mlir::failed(verify(op, typeAttr.getMroTuple(), collection)))
          return mlir::failure();

        auto mro = dyn_cast<TupleAttrInterface>(typeAttr.getMroTuple());
        if (!mro)
          return op->emitOpError("Expected MRO to refer to a tuple\n");

        if (!llvm::all_of(typeAttr.getInstanceSlots(),
                          [](mlir::Attribute attr) {
                            return attr.isa<pylir::Py::StrAttr>();
                          }))
          return op->emitOpError(
              "Expected 'instance_slots' to refer to a tuple of strings\n");

        return mlir::success();
      })
      .Default(mlir::success());
}

mlir::LogicalResult verifyCall(::mlir::SymbolTableCollection& symbolTable,
                               mlir::Operation* call,
                               mlir::ValueRange callOperands,
                               mlir::FlatSymbolRefAttr callee) {
  auto funcOp = symbolTable.lookupNearestSymbolFrom<mlir::FunctionOpInterface>(
      call, callee);
  if (!funcOp)
    return call->emitOpError("failed to find function named '")
           << callee << "'";

  auto argumentTypes = funcOp.getArgumentTypes();
  llvm::SmallVector<mlir::Type> operandTypes;
  for (auto iter : callOperands)
    operandTypes.push_back(iter.getType());

  if (!std::equal(argumentTypes.begin(), argumentTypes.end(),
                  operandTypes.begin(), operandTypes.end()))
    return call->emitOpError(
               "call operand types are not compatible with argument types of '")
           << callee << "'";

  return mlir::success();
}

} // namespace

//===----------------------------------------------------------------------===//
// ConstantOp verification
//===----------------------------------------------------------------------===//

mlir::LogicalResult pylir::Py::ConstantOp::verifySymbolUses(
    ::mlir::SymbolTableCollection& symbolTable) {
  return ::verify(*this, getConstantAttr(), symbolTable);
}

//===----------------------------------------------------------------------===//
// MakeFuncOp verification
//===----------------------------------------------------------------------===//

mlir::LogicalResult pylir::Py::MakeFuncOp::verifySymbolUses(
    ::mlir::SymbolTableCollection& symbolTable) {
  return verifySymbolUse<mlir::FunctionOpInterface>(
      *this, getFunctionAttr(), symbolTable, "FunctionOpInterface");
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

mlir::LogicalResult pylir::Py::GlobalOp::verifySymbolUses(
    ::mlir::SymbolTableCollection& symbolTable) {
  if (!getInitializerAttr())
    return mlir::success();

  return llvm::TypeSwitch<mlir::Type, mlir::LogicalResult>(getType())
      .Case([&](DynamicType) -> mlir::LogicalResult {
        if (!getInitializerAttr()
                 .isa<ObjectAttrInterface, GlobalValueAttr, UnboundAttr>())
          return emitOpError("Expected initializer of type "
                             "'ObjectAttrInterface' or 'GlobalValueAttr' "
                             "to global value");

        return ::verify(*this, getInitializerAttr(), symbolTable);
      })
      .Case([&](mlir::IndexType) -> mlir::LogicalResult {
        if (!getInitializerAttr().isa<mlir::IntegerAttr>())
          return emitOpError("Expected integer attribute initializer");

        return mlir::success();
      })
      .Case([&](mlir::FloatType) -> mlir::LogicalResult {
        if (!getInitializerAttr().isa<mlir::FloatAttr>())
          return emitOpError("Expected float attribute initializer");

        return mlir::success();
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
