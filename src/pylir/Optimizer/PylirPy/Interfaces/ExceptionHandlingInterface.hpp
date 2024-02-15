//  Licensed under the Apache License v2.0 with LLVM Exceptions.
//  See https://llvm.org/LICENSE.txt for license information.
//  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <mlir/IR/Builders.h>
#include <mlir/IR/OpDefinition.h>

#include "pylir/Optimizer/PylirPy/Interfaces/ExceptionHandlingInterface.h.inc"

namespace pylir::Py {
/// Helper function called by 'CreateExceptionHandlingVariant' in TableGen to
/// create the non-exception handling version from 'exceptionOp'.
template <class T>
mlir::Operation*
cloneWithoutExceptionHandlingImpl(mlir::OpBuilder& builder, T exceptionOp,
                                  llvm::StringRef normalOpName) {
  using namespace mlir;
  StringAttr operandSegmentSizeAttrName =
      exceptionOp.getOperandSegmentSizesAttrName();

  auto operationName = OperationName(normalOpName, builder.getContext());

  // Operands and result types can be copied except that the normal and unwind
  // operands must be removed. 'CreateExceptionHandlingVariant' guarantees their
  // position at the back.
  OperationState state(exceptionOp->getLoc(), operationName);
  state.addTypes(exceptionOp->getResultTypes());
  state.addOperands(exceptionOp->getOperands().drop_back(
      exceptionOp.getNormalDestOperandsMutable().size() +
      exceptionOp.getUnwindDestOperandsMutable().size()));

  // Go through all the inherent attributes and copy them except
  // 'AttrSizedOperandSegments'. It must be removed or adjusted depending on
  // whether 'operationName' requires is.
  ArrayRef<NamedAttribute> attrs = exceptionOp->getAttrs();
  SmallVector<NamedAttribute> attributes;
  attributes.reserve(attrs.size());
  if (!operationName.hasTrait<OpTrait::AttrSizedOperandSegments>()) {
    // If the normal version does not have 'AttrSizedOperandSegments' remove it
    // from the attributes.
    llvm::copy_if(attrs, std::back_inserter(attributes),
                  [&](NamedAttribute attribute) {
                    return attribute.getName() != operandSegmentSizeAttrName;
                  });
  } else {
    llvm::transform(attrs, std::back_inserter(attributes),
                    [&](NamedAttribute attribute) {
                      if (attribute.getName() == operandSegmentSizeAttrName) {
                        // Pop the last two entries. These corresponded to
                        // '$normal_dest_operands' and '$unwind_dest_operands'
                        // respectively.
                        attribute.setValue(builder.getDenseI32ArrayAttr(
                            cast<DenseI32ArrayAttr>(attribute.getValue())
                                .asArrayRef()
                                .drop_back(2)));
                      }
                      return attribute;
                    });
  }

  state.addAttributes(attributes);
  return builder.create(state);
}
} // namespace pylir::Py

template <>
struct llvm::PointerLikeTypeTraits<pylir::Py::ExceptionHandlingInterface> {
  static inline void*
  getAsVoidPointer(pylir::Py::ExceptionHandlingInterface p) {
    return const_cast<void*>(p.getAsOpaquePointer());
  }

  static inline pylir::Py::ExceptionHandlingInterface
  getFromVoidPointer(void* p) {
    return pylir::Py::ExceptionHandlingInterface::getFromOpaquePointer(p);
  }

  static constexpr int NumLowBitsAvailable =
      llvm::PointerLikeTypeTraits<mlir::Operation*>::NumLowBitsAvailable;
};
