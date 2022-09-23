//  Licensed under the Apache License v2.0 with LLVM Exceptions.
//  See https://llvm.org/LICENSE.txt for license information.
//  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/Types.h>

#include "PylirPyRefAttr.hpp"

#include "pylir/Optimizer/PylirPy/IR/ObjectTypeInterface.h.inc"

template <>
struct llvm::PointerLikeTypeTraits<pylir::Py::ObjectTypeInterface>
{
    static inline void* getAsVoidPointer(pylir::Py::ObjectTypeInterface p)
    {
        return const_cast<void*>(p.getAsOpaquePointer());
    }

    static inline pylir::Py::ObjectTypeInterface getFromVoidPointer(void* p)
    {
        return pylir::Py::ObjectTypeInterface::getFromOpaquePointer(p);
    }

    static constexpr int NumLowBitsAvailable = llvm::PointerLikeTypeTraits<mlir::Type>::NumLowBitsAvailable;
};
