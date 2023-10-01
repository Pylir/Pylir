//  Licensed under the Apache License v2.0 with LLVM Exceptions.
//  See https://llvm.org/LICENSE.txt for license information.
//  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <mlir/IR/Attributes.h>
#include <mlir/IR/BuiltinAttributes.h>

#include <pylir/Support/BigInt.hpp>

namespace pylir::Py {
class TypeAttrInterface;

/// Base class of all attributes that represent python objects.
class ObjectBaseAttribute : public mlir::Attribute {
public:
  using mlir::Attribute::Attribute;

  static bool classof(mlir::Attribute attribute);
};

} // namespace pylir::Py

#include "pylir/Optimizer/PylirPy/IR/PylirPyAttrInterfaces.h.inc"
