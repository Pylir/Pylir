#include "PylirOps.hpp"

#include <mlir/IR/OpImplementation.h>

#include <pylir/Support/Macros.hpp>

#include "PylirAttributes.hpp"
#include "PylirDialect.hpp"

mlir::OpFoldResult pylir::Dialect::ConstantOp::fold(::llvm::ArrayRef<::mlir::Attribute>)
{
    return value();
}

mlir::OpFoldResult pylir::Dialect::IAddOp::fold(::llvm::ArrayRef<::mlir::Attribute> operands)
{
    PYLIR_ASSERT(operands.size() == 2);
    auto lhs = operands[0].dyn_cast_or_null<Dialect::IntegerAttr>();
    auto rhs = operands[1].dyn_cast_or_null<Dialect::IntegerAttr>();
    if (lhs && rhs)
    {
        auto newSize = std::max(lhs.getValue().getBitWidth(), rhs.getValue().getBitWidth()) + 1;
        return Dialect::IntegerAttr::get(getContext(),
                                         lhs.getValue().sextOrSelf(newSize) + rhs.getValue().sextOrSelf(newSize));
    }
    if (lhs || rhs)
    {
        auto integer = lhs ? lhs : rhs;
        auto other = lhs ? getOperand(1) : getOperand(0);
        if (integer.getValue() == 0)
        {
            return other;
        }
    }
    return nullptr;
}

mlir::OpFoldResult pylir::Dialect::ISubOp::fold(::llvm::ArrayRef<::mlir::Attribute> operands)
{
    PYLIR_ASSERT(operands.size() == 2);
    auto lhs = operands[0].dyn_cast_or_null<Dialect::IntegerAttr>();
    auto rhs = operands[1].dyn_cast_or_null<Dialect::IntegerAttr>();
    if (lhs && rhs)
    {
        auto newSize = std::max(lhs.getValue().getBitWidth(), rhs.getValue().getBitWidth()) + 1;
        return Dialect::IntegerAttr::get(getContext(),
                                         lhs.getValue().sextOrSelf(newSize) - rhs.getValue().sextOrSelf(newSize));
    }
    if (rhs && rhs.getValue() == 0)
    {
        return getOperand(0);
    }
    if (getOperand(0) == getOperand(1))
    {
        return Dialect::IntegerAttr::get(getContext(), llvm::APInt(2, 0));
    }
    return nullptr;
}

mlir::OpFoldResult pylir::Dialect::IMulOp::fold(::llvm::ArrayRef<::mlir::Attribute> operands)
{
    PYLIR_ASSERT(operands.size() == 2);
    auto lhs = operands[0].dyn_cast_or_null<Dialect::IntegerAttr>();
    auto rhs = operands[1].dyn_cast_or_null<Dialect::IntegerAttr>();
    if (lhs && rhs)
    {
        auto newSize = std::max(lhs.getValue().getBitWidth(), rhs.getValue().getBitWidth()) * 2;
        return Dialect::IntegerAttr::get(getContext(),
                                         lhs.getValue().sextOrSelf(newSize) * rhs.getValue().sextOrSelf(newSize));
    }
    if (lhs || rhs)
    {
        auto integer = lhs ? lhs : rhs;
        auto other = lhs ? getOperand(1) : getOperand(0);
        if (integer.getValue() == 0)
        {
            return integer;
        }
        if (integer.getValue() == 1)
        {
            return other;
        }
    }
    return nullptr;
}

mlir::OpFoldResult pylir::Dialect::IDivOp::fold(::llvm::ArrayRef<::mlir::Attribute> operands)
{
    PYLIR_ASSERT(operands.size() == 2);
    auto lhs = operands[0].dyn_cast_or_null<Dialect::IntegerAttr>();
    auto rhs = operands[1].dyn_cast_or_null<Dialect::IntegerAttr>();
    if (lhs && rhs)
    {
        return Dialect::FloatAttr::get(getContext(), lhs.getValue().roundToDouble() / rhs.getValue().roundToDouble());
    }
    if (getOperand(0) == getOperand(1))
    {
        return Dialect::FloatAttr::get(getContext(), 1);
    }
    return nullptr;
}

mlir::OpFoldResult pylir::Dialect::IFloorDivOp::fold(::llvm::ArrayRef<::mlir::Attribute> operands)
{
    PYLIR_ASSERT(operands.size() == 2);
    auto lhs = operands[0].dyn_cast_or_null<Dialect::IntegerAttr>();
    auto rhs = operands[1].dyn_cast_or_null<Dialect::IntegerAttr>();
    if (lhs && rhs)
    {
        auto newSize = std::max(lhs.getValue().getBitWidth(), rhs.getValue().getBitWidth());
        return Dialect::IntegerAttr::get(getContext(),
                                         lhs.getValue().sextOrSelf(newSize).sdiv(rhs.getValue().sextOrSelf(newSize)));
    }
    if (getOperand(0) == getOperand(1))
    {
        return Dialect::IntegerAttr::get(getContext(), llvm::APInt(2, 1));
    }
    return nullptr;
}

mlir::OpFoldResult pylir::Dialect::IModOp::fold(::llvm::ArrayRef<::mlir::Attribute> operands)
{
    PYLIR_ASSERT(operands.size() == 2);
    auto lhs = operands[0].dyn_cast_or_null<Dialect::IntegerAttr>();
    auto rhs = operands[1].dyn_cast_or_null<Dialect::IntegerAttr>();
    if (lhs && rhs)
    {
        // Python's % operator does modulo, aka the sign is copied from the dividend.
        // LLVM's APInt however only has remainder where the sign is taken from the divisor.
        // Implement modulo by making sure lhs is positive and then copying the rhs sign

        // Plus one to make sure that lhsValue can be cast to positive. If it's the max negative value this would fail
        // otherwise
        auto newSize = std::max(lhs.getValue().getBitWidth(), rhs.getValue().getBitWidth()) + 1;
        auto lhsValue = lhs.getValue().sextOrSelf(newSize).abs();
        auto rhsValue = rhs.getValue().sextOrSelf(newSize);
        auto mod = lhsValue.urem(rhsValue);
        if (rhsValue.isNegative())
        {
            mod.negate();
        }
        return Dialect::IntegerAttr::get(getContext(), std::move(mod));
    }
    if (getOperand(0) == getOperand(1))
    {
        return Dialect::IntegerAttr::get(getContext(), llvm::APInt(2, 0));
    }
    return nullptr;
}

mlir::OpFoldResult pylir::Dialect::ItoF::fold(::llvm::ArrayRef<::mlir::Attribute> operands)
{
    PYLIR_ASSERT(operands.size() == 1);
    if (auto input = operands[0].dyn_cast_or_null<Dialect::IntegerAttr>())
    {
        return Dialect::FloatAttr::get(getContext(), input.getValue().roundToDouble());
    }
    return nullptr;
}

mlir::OpFoldResult pylir::Dialect::FAddOp::fold(::llvm::ArrayRef<::mlir::Attribute> operands)
{
    PYLIR_ASSERT(operands.size() == 2);
    auto lhs = operands[0].dyn_cast_or_null<Dialect::FloatAttr>();
    auto rhs = operands[1].dyn_cast_or_null<Dialect::FloatAttr>();
    if (lhs && rhs)
    {
        return Dialect::FloatAttr::get(getContext(), lhs.getValue() + rhs.getValue());
    }
    if (lhs || rhs)
    {
        auto constant = lhs ? lhs : rhs;
        auto other = lhs ? getOperand(1) : getOperand(0);
        if (constant.getValue() == 0)
        {
            return other;
        }
    }
    return nullptr;
}

mlir::OpFoldResult pylir::Dialect::FSubOp::fold(::llvm::ArrayRef<::mlir::Attribute> operands)
{
    PYLIR_ASSERT(operands.size() == 2);
    auto lhs = operands[0].dyn_cast_or_null<Dialect::FloatAttr>();
    auto rhs = operands[1].dyn_cast_or_null<Dialect::FloatAttr>();
    if (lhs && rhs)
    {
        return Dialect::FloatAttr::get(getContext(), lhs.getValue() - rhs.getValue());
    }
    if (rhs && rhs.getValue() == 0)
    {
        return getOperand(0);
    }
    if (getOperand(0) == getOperand(1))
    {
        return Dialect::FloatAttr::get(getContext(), 0);
    }
    return nullptr;
}

mlir::OpFoldResult pylir::Dialect::FMulOp::fold(::llvm::ArrayRef<::mlir::Attribute> operands)
{
    PYLIR_ASSERT(operands.size() == 2);
    auto lhs = operands[0].dyn_cast_or_null<Dialect::FloatAttr>();
    auto rhs = operands[1].dyn_cast_or_null<Dialect::FloatAttr>();
    if (lhs && rhs)
    {
        return Dialect::FloatAttr::get(getContext(), lhs.getValue() * rhs.getValue());
    }
    if (lhs || rhs)
    {
        auto constant = lhs ? lhs : rhs;
        auto other = lhs ? getOperand(1) : getOperand(0);
        if (constant.getValue() == 0)
        {
            return constant;
        }
        if (constant.getValue() == 1)
        {
            return other;
        }
    }
    return nullptr;
}

mlir::OpFoldResult pylir::Dialect::FDivOp::fold(::llvm::ArrayRef<::mlir::Attribute> operands)
{
    PYLIR_ASSERT(operands.size() == 2);
    auto lhs = operands[0].dyn_cast_or_null<Dialect::FloatAttr>();
    auto rhs = operands[1].dyn_cast_or_null<Dialect::FloatAttr>();
    if (lhs && rhs)
    {
        return Dialect::FloatAttr::get(getContext(), lhs.getValue() / rhs.getValue());
    }
    if (getOperand(0) == getOperand(1))
    {
        return Dialect::FloatAttr::get(getContext(), 1);
    }
    return nullptr;
}

mlir::OpFoldResult pylir::Dialect::FFloorDivOp::fold(::llvm::ArrayRef<::mlir::Attribute> operands)
{
    PYLIR_ASSERT(operands.size() == 2);
    auto lhs = operands[0].dyn_cast_or_null<Dialect::FloatAttr>();
    auto rhs = operands[1].dyn_cast_or_null<Dialect::FloatAttr>();
    if (lhs && rhs)
    {
        return Dialect::FloatAttr::get(getContext(), std::floor(lhs.getValue() / rhs.getValue()));
    }
    if (getOperand(0) == getOperand(1))
    {
        return Dialect::FloatAttr::get(getContext(), 1);
    }
    return nullptr;
}

mlir::OpFoldResult pylir::Dialect::FModOp::fold(::llvm::ArrayRef<::mlir::Attribute> operands)
{
    PYLIR_ASSERT(operands.size() == 2);
    auto lhs = operands[0].dyn_cast_or_null<Dialect::FloatAttr>();
    auto rhs = operands[1].dyn_cast_or_null<Dialect::FloatAttr>();
    if (lhs && rhs)
    {
        return Dialect::FloatAttr::get(getContext(), std::fmod(lhs.getValue(), rhs.getValue()));
    }
    if (getOperand(0) == getOperand(1))
    {
        return Dialect::FloatAttr::get(getContext(), 0);
    }
    return nullptr;
}

// TODO: Remove in MLIR 13
using namespace mlir;
#define GET_OP_CLASSES
#include "pylir/Dialect/PylirOps.cpp.inc"
