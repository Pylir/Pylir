
#include "X86_64.hpp"

mlir::LLVM::LLVMFuncOp pylir::Dialect::X86_64::declareFunc(mlir::OpBuilder& builder, mlir::Location loc,
                                                           mlir::Type returnType, llvm::StringRef name,
                                                           mlir::TypeRange inputTypes)
{
    return mlir::LLVM::LLVMFuncOp();
}

mlir::Value pylir::Dialect::X86_64::callFunc(mlir::OpBuilder& builder, mlir::Location loc, mlir::LLVM::LLVMFuncOp func,
                                             mlir::ValueRange operands)
{
    return mlir::Value();
}

pylir::Dialect::X86_64::X86_64(mlir::DataLayout dataLayout) : CABI(std::move(dataLayout)) {}
