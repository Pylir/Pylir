
#pragma once

#include <llvm/ADT/DenseMap.h>

#include "PlatformABI.hpp"

namespace pylir
{
class WinX64 final : public PlatformABI
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
