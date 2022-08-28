// Copyright 2022 Markus Böck
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <mlir/Dialect/LLVMIR/LLVMDialect.h>

namespace pylir
{
class PlatformABI
{
    mlir::DataLayout m_dataLayout;

protected:
    
    explicit PlatformABI(mlir::DataLayout dataLayout) : m_dataLayout(std::move(dataLayout)) {}

public:
    virtual ~PlatformABI() = default;

    PlatformABI(const PlatformABI&) = delete;
    PlatformABI& operator=(const PlatformABI&) = delete;
    PlatformABI(PlatformABI&&) = delete;
    PlatformABI& operator=(PlatformABI&&) = delete;

    std::size_t getSizeOf(mlir::Type type) const;

    std::size_t getAlignOf(mlir::Type type) const;

    virtual mlir::LLVM::LLVMFuncOp declareFunc(mlir::OpBuilder& builder, mlir::Location loc, mlir::Type returnType,
                                               llvm::StringRef name, mlir::TypeRange inputTypes) = 0;

    virtual mlir::Value callFunc(mlir::OpBuilder& builder, mlir::Location loc, mlir::LLVM::LLVMFuncOp func,
                                 mlir::ValueRange operands) = 0;

    virtual mlir::Type getInt(mlir::MLIRContext* context) const = 0;

    virtual mlir::Type getSizeT(mlir::MLIRContext* context) const = 0;
};
} // namespace pylir::Dialect
