
#pragma once

#include <tuple>
#include <variant>

#include "CABI.hpp"

namespace pylir
{
class X86_64 final : public CABI
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
    explicit X86_64(mlir::DataLayout dataLayout);

    mlir::LLVM::LLVMFuncOp declareFunc(mlir::OpBuilder& builder, mlir::Location loc, mlir::Type returnType,
                                       llvm::StringRef name, mlir::TypeRange inputTypes) override;

    mlir::Value callFunc(mlir::OpBuilder& builder, mlir::Location loc, mlir::LLVM::LLVMFuncOp func,
                         mlir::ValueRange operands) override;

    mlir::Type getInt(mlir::MLIRContext* context) const override
    {
        return mlir::IntegerType::get(context, 32);
    }
};
} // namespace pylir::Dialect
