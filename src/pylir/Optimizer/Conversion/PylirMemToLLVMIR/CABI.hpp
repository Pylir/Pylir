
#pragma once

#include <mlir/Dialect/LLVMIR/LLVMDialect.h>

namespace pylir
{
class CABI
{
    mlir::DataLayout m_dataLayout;

protected:
    // TODO use DataLayout exclusively once LLVM Dialect makes use of it

    std::size_t getSizeOf(mlir::Type type) const;

    explicit CABI(mlir::DataLayout dataLayout) : m_dataLayout(std::move(dataLayout)) {}

public:
    virtual ~CABI() = default;

    CABI(const CABI&) = delete;
    CABI(CABI&&) = delete;

    virtual mlir::LLVM::LLVMFuncOp declareFunc(mlir::OpBuilder& builder, mlir::Location loc, mlir::Type returnType,
                                               llvm::StringRef name, mlir::TypeRange inputTypes) = 0;

    virtual mlir::Value callFunc(mlir::OpBuilder& builder, mlir::Location loc, mlir::LLVM::LLVMFuncOp func,
                                 mlir::ValueRange operands) = 0;

    virtual mlir::Type getInt(mlir::MLIRContext* context) const = 0;
};
} // namespace pylir::Dialect
