// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "PylirPyTraits.hpp"

#include <mlir/IR/Builders.h>
#include <mlir/IR/IRMapping.h>

#include <pylir/Support/Macros.hpp>

mlir::Operation* pylir::Py::details::cloneWithExceptionHandlingImpl(
    mlir::OpBuilder& builder, mlir::Operation* operation,
    const mlir::OperationName& invokeVersion, ::mlir::Block* happyPath,
    mlir::Block* exceptionPath, mlir::ValueRange unwindOperands,
    llvm::StringRef attrSizedSegmentName, llvm::ArrayRef<OperandShape> shape) {
  mlir::OperationState state(operation->getLoc(), invokeVersion);
  state.addTypes(operation->getResultTypes());
  state.addSuccessors(happyPath);
  state.addSuccessors(exceptionPath);
  auto vector = llvm::to_vector(operation->getOperands());
  vector.insert(vector.end(), unwindOperands.begin(), unwindOperands.end());
  state.addOperands(vector);
  llvm::SmallVector<mlir::NamedAttribute> attributes;
  for (const auto& iter : operation->getAttrs()) {
    attributes.push_back(iter);
    if (iter.getName() == attrSizedSegmentName) {
      llvm::SmallVector<std::int32_t> sizes;
      for (auto integer :
           mlir::cast<mlir::DenseI32ArrayAttr>(iter.getValue()).asArrayRef())
        sizes.push_back(integer);

      sizes.push_back(0);
      sizes.push_back(unwindOperands.size());
      attributes.back().setValue(builder.getDenseI32ArrayAttr(sizes));
    }
  }
  if (!operation->hasTrait<mlir::OpTrait::AttrSizedOperandSegments>()) {
    auto numOperands = operation->getNumOperands();
    llvm::SmallVector<std::int32_t> values;
    while (!shape.empty() && shape.front() != OperandShape::Variadic) {
      numOperands--;
      values.push_back(1);
      shape = shape.drop_front();
    }
    auto index = values.size();
    while (!shape.empty() && shape.back() != OperandShape::Variadic) {
      numOperands--;
      values.insert(values.begin() + index, 1);
      shape = shape.drop_back();
    }
    PYLIR_ASSERT(shape.size() <= 1);
    if (shape.size() == 1)
      values.insert(values.begin() + index, numOperands);

    values.push_back(0);
    values.push_back(unwindOperands.size());
    attributes.emplace_back(builder.getStringAttr(attrSizedSegmentName),
                            builder.getDenseI32ArrayAttr(values));
  }
  state.addAttributes(attributes);
  // Reuse the capacity of the IR maps across different regions. Avoids memory
  // reallocations.
  mlir::IRMapping mapping;
  for (mlir::Region& region : operation->getRegions()) {
    mapping.clear();
    region.cloneInto(state.addRegion(), mapping);
  }
  return builder.create(state);
}
