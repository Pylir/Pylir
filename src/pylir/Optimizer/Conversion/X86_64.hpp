
#pragma once

#include "CABI.hpp"

namespace pylir::Dialect
{
class X86_64 final : public CABI
{
public:
    explicit X86_64(mlir::DataLayout dataLayout);

    mlir::LLVM::LLVMFuncOp declareFunc(mlir::OpBuilder& builder, mlir::Location loc, mlir::Type returnType,
                                       llvm::StringRef name, mlir::TypeRange inputTypes) override;
    mlir::Value callFunc(mlir::OpBuilder& builder, mlir::Location loc, mlir::LLVM::LLVMFuncOp func,
                         mlir::ValueRange operands) override;
};
} // namespace pylir::Dialect
