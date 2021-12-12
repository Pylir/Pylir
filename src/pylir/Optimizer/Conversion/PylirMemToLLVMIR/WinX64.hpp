
#pragma once

#include <llvm/ADT/DenseMap.h>

#include "CABI.hpp"

namespace pylir
{
class WinX64 final : public CABI
{
    enum Transformation
    {
        Nothing,
        IntegerRegister,
        PointerToTemporary
    };

    struct Adjustments
    {
        Transformation returnType = Nothing;
        mlir::Type originalRetType;
        llvm::SmallVector<Transformation> arguments;
    };

    llvm::DenseMap<mlir::Operation*, Adjustments> m_adjustments;

public:
    explicit WinX64(mlir::DataLayout dataLayout);

    mlir::LLVM::LLVMFuncOp declareFunc(mlir::OpBuilder& builder, mlir::Location loc, mlir::Type returnType,
                                       llvm::StringRef name, mlir::TypeRange inputTypes) override;

    mlir::Value callFunc(mlir::OpBuilder& builder, mlir::Location loc, mlir::LLVM::LLVMFuncOp func,
                         mlir::ValueRange operands) override;
};
} // namespace pylir::Dialect
