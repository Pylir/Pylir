// Copyright 2022 Markus Böck
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <tuple>
#include <variant>

#include "PlatformABI.hpp"

namespace pylir
{
class ARM64 final : public PlatformABI
{
    struct Unchanged
    {
    };

    struct PointerToTemporary
    {
    };

    struct OnStack
    {
    };

    struct Flattened
    {
    };

    struct MultipleArgs
    {
        std::size_t size;
    };

    struct Adjustments
    {
        std::variant<Unchanged, PointerToTemporary, Flattened> returnType;
        mlir::Type originalRetType;
        using Arg = std::variant<Unchanged, OnStack, MultipleArgs>;
        std::vector<Arg> arguments;
    };

    llvm::DenseMap<mlir::Operation*, Adjustments> m_adjustments;

    std::tuple<Adjustments::Arg, mlir::Type, mlir::Type>
        flattenSingleArg(mlir::Type type, std::uint8_t* takenIntegers = nullptr, std::uint8_t* takenFloats = nullptr);

public:
    explicit ARM64(mlir::DataLayout dataLayout);

    mlir::LLVM::LLVMFuncOp declareFunc(mlir::OpBuilder& builder, mlir::Location loc, mlir::Type returnType,
                                       llvm::StringRef name, mlir::TypeRange inputTypes) override;

    mlir::Value callFunc(mlir::OpBuilder& builder, mlir::Location loc, mlir::LLVM::LLVMFuncOp func,
                         mlir::ValueRange operands) override;

    mlir::Type getInt(mlir::MLIRContext* context) const override
    {
        return mlir::IntegerType::get(context, 32);
    }

    mlir::Type getSizeT(mlir::MLIRContext* context) const override
    {
        return mlir::IntegerType::get(context, 64);
    }
};
} // namespace pylir::Dialect
