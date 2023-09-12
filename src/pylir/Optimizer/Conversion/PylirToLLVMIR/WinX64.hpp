//  Licensed under the Apache License v2.0 with LLVM Exceptions.
//  See https://llvm.org/LICENSE.txt for license information.
//  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <llvm/ADT/DenseMap.h>

#include "PlatformABI.hpp"

namespace pylir {
class WinX64 final : public PlatformABI {
  enum Transformation { Nothing, IntegerRegister, PointerToTemporary };

  struct Adjustments {
    Transformation returnType = Nothing;
    mlir::Type originalRetType;
    llvm::SmallVector<Transformation> arguments;
  };

  llvm::DenseMap<mlir::Operation*, Adjustments> m_adjustments;

public:
  using PlatformABI::PlatformABI;

  mlir::LLVM::LLVMFuncOp declareFunc(mlir::OpBuilder& builder,
                                     mlir::Location loc, mlir::Type returnType,
                                     llvm::StringRef name,
                                     mlir::TypeRange parameterTypes) override;

  mlir::Value callFunc(mlir::OpBuilder& builder, mlir::Location loc,
                       mlir::LLVM::LLVMFuncOp func,
                       mlir::ValueRange arguments) override;

  mlir::Type getInt(mlir::MLIRContext* context) const override {
    return mlir::IntegerType::get(context, 32);
  }

  mlir::Type getSizeT(mlir::MLIRContext* context) const override {
    return mlir::IntegerType::get(context, 64);
  }

  mlir::Type
  getUnwindExceptionHeader(mlir::MLIRContext* context) const override;
};
} // namespace pylir
