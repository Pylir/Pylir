// Copyright 2022 Markus Böck
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <mlir/IR/OpDefinition.h>

#include "pylir/Optimizer/PylirPy/Interfaces/ExceptionHandlingInterface.h.inc"

template <>
struct llvm::PointerLikeTypeTraits<pylir::Py::ExceptionHandlingInterface>
{
    static inline void* getAsVoidPointer(pylir::Py::ExceptionHandlingInterface p)
    {
        return const_cast<void*>(p.getAsOpaquePointer());
    }

    static inline pylir::Py::ExceptionHandlingInterface getFromVoidPointer(void* p)
    {
        return pylir::Py::ExceptionHandlingInterface::getFromOpaquePointer(p);
    }

    static constexpr int NumLowBitsAvailable = llvm::PointerLikeTypeTraits<mlir::Operation*>::NumLowBitsAvailable;
};
