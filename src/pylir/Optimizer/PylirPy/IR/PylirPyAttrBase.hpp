//  Licensed under the Apache License v2.0 with LLVM Exceptions.
//  See https://llvm.org/LICENSE.txt for license information.
//  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <mlir/IR/Attributes.h>
#include <mlir/IR/BuiltinAttributes.h>

namespace pylir::Py {
class ConcreteObjectAttrInterface;
class GlobalValueOp;
namespace detail {
struct GlobalValueAttrStorage;
struct RefAttrStorage;
} // namespace detail
} // namespace pylir::Py

#define GET_ATTRDEF_CLASSES
#include "pylir/Optimizer/PylirPy/IR/PylirPyAttrBase.h.inc"

template <>
struct llvm::PointerLikeTypeTraits<pylir::Py::RefAttr> {
  static inline void* getAsVoidPointer(pylir::Py::RefAttr p) {
    return const_cast<void*>(p.getAsOpaquePointer());
  }

  static inline pylir::Py::RefAttr getFromVoidPointer(void* p) {
    return pylir::Py::RefAttr::getFromOpaquePointer(p);
  }

  static constexpr int NumLowBitsAvailable =
      llvm::PointerLikeTypeTraits<mlir::Attribute>::NumLowBitsAvailable;
};
