//  Licensed under the Apache License v2.0 with LLVM Exceptions.
//  See https://llvm.org/LICENSE.txt for license information.
//  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "PylirPyAttrInterfaces.hpp"

#include "PylirPyAttributes.hpp"

#include "pylir/Optimizer/PylirPy/IR/PylirPyAttrInterfaces.cpp.inc"

using namespace pylir::Py;

bool ObjectBaseAttribute::classof(mlir::Attribute attribute) {
  return llvm::isa<GlobalValueAttr>(attribute) ||
         ConcreteObjectAttribute::classof(attribute);
}
