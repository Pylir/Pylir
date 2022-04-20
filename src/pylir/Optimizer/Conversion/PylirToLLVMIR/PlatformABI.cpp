// Copyright 2022 Markus Böck
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "PlatformABI.hpp"

#include <llvm/ADT/TypeSwitch.h>

#include <pylir/Support/Macros.hpp>

std::size_t pylir::PlatformABI::getAlignOf(mlir::Type type) const
{
    // For now assumes sizeof == alignof until we have proper DataLayout functionality at the module level
    return llvm::TypeSwitch<mlir::Type, std::size_t>(type)
        .Case<mlir::IntegerType, mlir::FloatType, mlir::LLVM::LLVMPointerType>(
            [this](mlir::Type type) { return m_dataLayout.getTypeSize(type); })
        .Case<mlir::LLVM::LLVMArrayType>([this](mlir::LLVM::LLVMArrayType array)
                                         { return getAlignOf(array.getElementType()); })
        .Case<mlir::LLVM::LLVMStructType>(
            [this](mlir::LLVM::LLVMStructType structType)
            {
                PYLIR_ASSERT((!structType.isIdentified() || structType.isInitialized()) && !structType.isPacked());
                std::size_t max = 0;
                for (const auto& iter : structType.getBody())
                {
                    max = std::max(max, getAlignOf(iter));
                }
                return max;
            })
        .Default([](auto) -> std::size_t { PYLIR_UNREACHABLE; });
}

std::size_t pylir::PlatformABI::getSizeOf(mlir::Type type) const
{
    return llvm::TypeSwitch<mlir::Type, std::size_t>(type)
        .Case<mlir::IntegerType, mlir::FloatType, mlir::LLVM::LLVMPointerType>(
            [this](mlir::Type type) { return m_dataLayout.getTypeSize(type); })
        .Case<mlir::LLVM::LLVMArrayType>([this](mlir::LLVM::LLVMArrayType array)
                                         { return getSizeOf(array.getElementType()) * array.getNumElements(); })
        .Case<mlir::LLVM::LLVMStructType>(
            [this](mlir::LLVM::LLVMStructType structType)
            {
                PYLIR_ASSERT((!structType.isIdentified() || structType.isInitialized()) && !structType.isPacked());
                std::size_t size = 0;
                std::size_t alignment = 0;
                for (auto iter : structType.getBody())
                {
                    auto elementSize = getSizeOf(iter);
                    auto elementAlign = getAlignOf(iter);
                    alignment = std::max(elementAlign, alignment);
                    size = llvm::alignTo(size, elementAlign);
                    size += elementSize;
                }
                size = llvm::alignTo(size, alignment);
                return size;
            })
        .Default([](auto) -> std::size_t { PYLIR_UNREACHABLE; });
}
