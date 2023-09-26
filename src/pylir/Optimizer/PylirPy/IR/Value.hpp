//  Licensed under the Apache License v2.0 with LLVM Exceptions.
//  See https://llvm.org/LICENSE.txt for license information.
//  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/SymbolTable.h>

#include "PylirPyAttrInterfaces.hpp"
#include "PylirPyOps.hpp"

namespace pylir::Py {
/// Returns the type of the value. This may either be a value referring to the
/// type object or an attribute that is the type object. This operation may also
/// fail in which case it is a null value.
mlir::OpFoldResult getTypeOf(mlir::Value value);

/// Returns whether the value is definitely bound, unbound or unknown. If the
/// optional does not have a value, it is unknown whether it's bound or not,
/// otherwise the optional contains whether the value is unbound.
std::optional<bool> isUnbound(mlir::Value value);

/// Returns whether 'lhs' and 'rhs' are equal according to
/// 'EqualsAttrInterface'. Returns unknown if either of the two operands
/// do not implement the interface.
std::optional<bool> isEqual(mlir::Attribute lhs, mlir::Attribute rhs);

} // namespace pylir::Py
