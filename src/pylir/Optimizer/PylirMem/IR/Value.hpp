//  Licensed under the Apache License v2.0 with LLVM Exceptions.
//  See https://llvm.org/LICENSE.txt for license information.
//  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <mlir/IR/Attributes.h>
#include <mlir/IR/Value.h>

#include <llvm/ADT/DenseMap.h>

#include <pylir/Optimizer/PylirPy/IR/PylirPyAttributes.hpp>

#include <optional>

#include "PylirMemAttributes.hpp"

namespace pylir::Mem {

/// Returns the layout type of the type object 'value' if possible to deduce
/// statically. The layout type is the super type of 'value' which defines the
/// size and layout of the object minus the slots. For most user defined types,
/// this simply Object, but most of the builtin types are layout types which
/// cause any subtypes to have the same object layout. Optionally one may also
/// pass 'cache' to memoize the computation.
std::optional<LayoutType>
getLayoutType(mlir::Value value,
              llvm::DenseMap<mlir::Attribute, LayoutType>* cache = nullptr);

/// Same as the function above but with 'attr' being the type object.
std::optional<LayoutType>
getLayoutType(mlir::Attribute attr,
              llvm::DenseMap<mlir::Attribute, LayoutType>* cache = nullptr);

/// Returns the 'GlobalValueAttr' for the builtin type object corresponding to
/// the given layout type.
Py::GlobalValueAttr layoutTypeToTypeObject(mlir::MLIRContext* context,
                                           LayoutType layoutType);

} // namespace pylir::Mem
