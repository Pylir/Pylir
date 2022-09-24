//  Licensed under the Apache License v2.0 with LLVM Exceptions.
//  See https://llvm.org/LICENSE.txt for license information.
//  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <mlir/IR/SymbolTable.h>
#include <mlir/IR/TypeRange.h>
#include <mlir/IR/Types.h>

#include "ObjectTypeInterface.hpp"
#include "PylirPyRefAttr.hpp"

#define GET_TYPEDEF_CLASSES
#include "pylir/Optimizer/PylirPy/IR/PylirPyTypes.h.inc"

namespace pylir::Py
{

pylir::Py::ObjectTypeInterface joinTypes(pylir::Py::ObjectTypeInterface lhs, pylir::Py::ObjectTypeInterface rhs);

bool isMoreSpecific(pylir::Py::ObjectTypeInterface lhs, pylir::Py::ObjectTypeInterface rhs);

/// Given a constant attribute, returns the type of the constant, or nullptr if the type is unknown.
/// Constant has to be a `py` dialect attribute. Any other kind of attribute is undefined behaviour.
pylir::Py::ObjectTypeInterface typeOfConstant(mlir::Attribute constant);
} // namespace pylir::Py
