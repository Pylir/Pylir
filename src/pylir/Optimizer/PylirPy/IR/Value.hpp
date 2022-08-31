//  Copyright 2022 Markus Böck
//
//  Licensed under the Apache License v2.0 with LLVM Exceptions.
//  See https://llvm.org/LICENSE.txt for license information.
//  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <mlir/IR/OpDefinition.h>

#include "ObjectAttrInterface.hpp"

namespace pylir::Py
{
/// Returns the type of the value. This may either be a value referring to the type object or an attribute that is the
/// type object. This operation may also fail in which case it is a null value.
mlir::OpFoldResult getTypeOf(mlir::Value value);

/// Returns whether the value is definitely bound, unbound or unknown. If the optional does not have a value, it is
/// unknown whether it's bound or not, otherwise the optional contains whether the value is unbound.
llvm::Optional<bool> isUnbound(mlir::Value value);
} // namespace pylir::Py
