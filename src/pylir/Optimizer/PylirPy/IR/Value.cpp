//  Licensed under the Apache License v2.0 with LLVM Exceptions.
//  See https://llvm.org/LICENSE.txt for license information.
//  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "Value.hpp"

#include <mlir/Interfaces/FunctionInterfaces.h>

#include <pylir/Optimizer/PylirPy/Interfaces/KnownTypeObjectInterface.hpp>
#include <pylir/Support/Macros.hpp>

#include "PylirPyAttributes.hpp"
#include "PylirPyTraits.hpp"

using namespace mlir;
using namespace pylir;
using namespace pylir::Py;

mlir::OpFoldResult pylir::Py::getTypeOf(mlir::Value value) {
  if (auto op = value.getDefiningOp<pylir::Py::KnownTypeObjectInterface>())
    return op.getKnownTypeObject();

  return nullptr;
}

std::optional<bool> pylir::Py::isUnbound(mlir::Value value) {
  mlir::Attribute constant;
  if (mlir::matchPattern(value, mlir::m_Constant(&constant)))
    return isa<Py::UnboundAttr>(constant);

  // The region's entry args of an Op marked 'EntryArgsBound' are known to be
  // bound.
  if (auto blockArg = dyn_cast<BlockArgument>(value)) {
    Block* owner = blockArg.getOwner();
    // TODO: This should be removed but is currently needed due to the dialect
    //       conversion infrastructure calling 'fold' for legalization despite
    //       not replacing the 'BlockArgument' after signature conversion.
    //       We are therefore seeing the block argument of the unliked block
    //       before conversion.
    if (!owner->getParent())
      return std::nullopt;

    if (owner->isEntryBlock() &&
        owner->getParentOp()->hasTrait<EntryArgsBound>())
      return false;

    return std::nullopt;
  }
  // If the defining op has the AlwaysBound trait then it is false.
  auto* op = value.getDefiningOp();
  PYLIR_ASSERT(op);
  if (op->hasTrait<Py::AlwaysBound>())
    return false;

  return std::nullopt;
}

std::optional<bool> pylir::Py::isEqual(Attribute lhs, Attribute rhs) {
  auto lhsC = dyn_cast<EqualsAttrInterface>(lhs);
  if (!lhsC)
    return std::nullopt;

  auto rhsC = dyn_cast<EqualsAttrInterface>(rhs);
  if (!rhsC)
    return std::nullopt;

  return lhsC.getCanonicalAttribute() == rhsC.getCanonicalAttribute();
}
