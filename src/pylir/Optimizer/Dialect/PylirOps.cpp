#include "PylirOps.hpp"

#include <mlir/IR/Builders.h>
#include <mlir/IR/OpImplementation.h>

#include <pylir/Support/Macros.hpp>

#include "PylirAttributes.hpp"

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
    if (rhs && rhs.getValue() == 0)
    {
        return getOperand(0);
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
    if (rhs)
    {
        if (rhs.getValue() == 0)
        {
            return rhs;
        }
        if (rhs.getValue() == 1)
        {
            return getOperand(0);
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

mlir::OpFoldResult pylir::Dialect::INegOp::fold(llvm::ArrayRef<mlir::Attribute> operands)
{
    PYLIR_ASSERT(operands.size() == 1);
    if (auto input = operands[0].dyn_cast_or_null<Dialect::IntegerAttr>())
    {
        auto value = input.getValue();
        value.flipAllBits();
        return Dialect::IntegerAttr::get(getContext(), std::move(value));
    }
    return nullptr;
}

mlir::OpFoldResult pylir::Dialect::IShlOp::fold(::llvm::ArrayRef<::mlir::Attribute> operands)
{
    PYLIR_ASSERT(operands.size() == 2);
    auto lhs = operands[0].dyn_cast_or_null<Dialect::IntegerAttr>();
    auto rhs = operands[1].dyn_cast_or_null<Dialect::IntegerAttr>();
    if (lhs && rhs && !rhs.getValue().isNegative())
    {
        auto newSize = lhs.getValue().getBitWidth() + rhs.getValue();
        return Dialect::IntegerAttr::get(
            getContext(),
            lhs.getValue().sextOrSelf(newSize.getZExtValue()).shl(rhs.getValue().zextOrSelf(newSize.getZExtValue())));
    }
    if (rhs && rhs.getValue() == 0)
    {
        return getOperand(0);
    }
    if (lhs && lhs.getValue() == 0)
    {
        return lhs;
    }
    return nullptr;
}

mlir::OpFoldResult pylir::Dialect::IShrOp::fold(::llvm::ArrayRef<::mlir::Attribute> operands)
{
    PYLIR_ASSERT(operands.size() == 2);
    auto lhs = operands[0].dyn_cast_or_null<Dialect::IntegerAttr>();
    auto rhs = operands[1].dyn_cast_or_null<Dialect::IntegerAttr>();
    if (lhs && rhs && !rhs.getValue().isNegative())
    {
        auto newSize = std::max(lhs.getValue().getBitWidth(), rhs.getValue().getBitWidth());
        return Dialect::IntegerAttr::get(getContext(),
                                         lhs.getValue().sextOrSelf(newSize).ashr(rhs.getValue().zextOrSelf(newSize)));
    }
    if (rhs && rhs.getValue() == 0)
    {
        return getOperand(0);
    }
    if (lhs && lhs.getValue() == 0)
    {
        return lhs;
    }
    return nullptr;
}

mlir::OpFoldResult pylir::Dialect::IAndOp::fold(::llvm::ArrayRef<::mlir::Attribute> operands)
{
    PYLIR_ASSERT(operands.size() == 2);
    auto lhs = operands[0].dyn_cast_or_null<Dialect::IntegerAttr>();
    auto rhs = operands[1].dyn_cast_or_null<Dialect::IntegerAttr>();
    if (lhs && rhs)
    {
        auto newSize = std::max(lhs.getValue().getBitWidth(), rhs.getValue().getBitWidth());
        return Dialect::IntegerAttr::get(getContext(),
                                         lhs.getValue().sextOrSelf(newSize) & rhs.getValue().zextOrSelf(newSize));
    }
    if (rhs)
    {
        if (rhs.getValue() == 0)
        {
            return rhs;
        }
        if (rhs.getValue().isAllOnesValue())
        {
            return getOperand(0);
        }
    }
    return nullptr;
}

mlir::OpFoldResult pylir::Dialect::IXorOp::fold(::llvm::ArrayRef<::mlir::Attribute> operands)
{
    PYLIR_ASSERT(operands.size() == 2);
    auto lhs = operands[0].dyn_cast_or_null<Dialect::IntegerAttr>();
    auto rhs = operands[1].dyn_cast_or_null<Dialect::IntegerAttr>();
    if (lhs && rhs)
    {
        auto newSize = std::max(lhs.getValue().getBitWidth(), rhs.getValue().getBitWidth());
        return Dialect::IntegerAttr::get(getContext(),
                                         lhs.getValue().sextOrSelf(newSize) ^ rhs.getValue().zextOrSelf(newSize));
    }
    if (rhs && rhs.getValue() == 0)
    {
        return getOperand(0);
    }
    return nullptr;
}

mlir::OpFoldResult pylir::Dialect::IOrOp::fold(::llvm::ArrayRef<::mlir::Attribute> operands)
{
    PYLIR_ASSERT(operands.size() == 2);
    auto lhs = operands[0].dyn_cast_or_null<Dialect::IntegerAttr>();
    auto rhs = operands[1].dyn_cast_or_null<Dialect::IntegerAttr>();
    if (lhs && rhs)
    {
        auto newSize = std::max(lhs.getValue().getBitWidth(), rhs.getValue().getBitWidth());
        return Dialect::IntegerAttr::get(getContext(),
                                         lhs.getValue().sextOrSelf(newSize) | rhs.getValue().zextOrSelf(newSize));
    }

    if (rhs)
    {
        if (rhs.getValue() == 0)
        {
            return getOperand(0);
        }
        else if (rhs.getValue().isAllOnesValue())
        {
            return rhs;
        }
    }
    return nullptr;
}

mlir::OpFoldResult pylir::Dialect::ICmpOp::fold(::llvm::ArrayRef<::mlir::Attribute> operands)
{
    auto lhs = operands[0].dyn_cast_or_null<Dialect::IntegerAttr>();
    auto rhs = operands[1].dyn_cast_or_null<Dialect::IntegerAttr>();
    if (!lhs || !rhs)
    {
        return nullptr;
    }
    auto equalSize = std::max(lhs.getValue().getBitWidth(), rhs.getValue().getBitWidth());
    switch (predicate())
    {
        case CmpPredicate::EQ:
            return Dialect::BoolAttr::get(getContext(),
                                          lhs.getValue().sextOrSelf(equalSize) == rhs.getValue().sextOrSelf(equalSize));
        case CmpPredicate::NE:
            return Dialect::BoolAttr::get(getContext(),
                                          lhs.getValue().sextOrSelf(equalSize) != rhs.getValue().sextOrSelf(equalSize));
        case CmpPredicate::LT:
            return Dialect::BoolAttr::get(
                getContext(), lhs.getValue().sextOrSelf(equalSize).slt(rhs.getValue().sextOrSelf(equalSize)));
        case CmpPredicate::LE:
            return Dialect::BoolAttr::get(
                getContext(), lhs.getValue().sextOrSelf(equalSize).sle(rhs.getValue().sextOrSelf(equalSize)));
        case CmpPredicate::GT:
            return Dialect::BoolAttr::get(
                getContext(), lhs.getValue().sextOrSelf(equalSize).sgt(rhs.getValue().sextOrSelf(equalSize)));
        case CmpPredicate::GE:
            return Dialect::BoolAttr::get(
                getContext(), lhs.getValue().sextOrSelf(equalSize).sge(rhs.getValue().sextOrSelf(equalSize)));
        default: PYLIR_UNREACHABLE;
    }
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
    if (rhs && rhs.getValue() == 0)
    {
        return getOperand(0);
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
    if (rhs)
    {
        if (rhs.getValue() == 0)
        {
            return rhs;
        }
        if (rhs.getValue() == 1)
        {
            return getOperand(0);
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

mlir::OpFoldResult pylir::Dialect::FCmpOp::fold(::llvm::ArrayRef<::mlir::Attribute> operands)
{
    auto lhs = operands[0].dyn_cast_or_null<Dialect::FloatAttr>();
    auto rhs = operands[1].dyn_cast_or_null<Dialect::FloatAttr>();
    if (!lhs || !rhs)
    {
        return nullptr;
    }
    switch (predicate())
    {
        case CmpPredicate::EQ: return Dialect::BoolAttr::get(getContext(), lhs.getValue() == rhs.getValue());
        case CmpPredicate::NE: return Dialect::BoolAttr::get(getContext(), lhs.getValue() != rhs.getValue());
        case CmpPredicate::LT: return Dialect::BoolAttr::get(getContext(), lhs.getValue() < rhs.getValue());
        case CmpPredicate::LE: return Dialect::BoolAttr::get(getContext(), lhs.getValue() <= rhs.getValue());
        case CmpPredicate::GT: return Dialect::BoolAttr::get(getContext(), lhs.getValue() > rhs.getValue());
        case CmpPredicate::GE: return Dialect::BoolAttr::get(getContext(), lhs.getValue() >= rhs.getValue());
        default: PYLIR_UNREACHABLE;
    }
}

mlir::OpFoldResult pylir::Dialect::BAndOp::fold(::llvm::ArrayRef<::mlir::Attribute> operands)
{
    auto lhs = operands[0].dyn_cast_or_null<Dialect::BoolAttr>();
    auto rhs = operands[1].dyn_cast_or_null<Dialect::BoolAttr>();
    if (lhs && rhs)
    {
        return Dialect::BoolAttr::get(getContext(), lhs.getValue() && rhs.getValue());
    }
    if (rhs)
    {
        if (rhs.getValue())
        {
            return getOperand(0);
        }
        return rhs;
    }
    return nullptr;
}

mlir::OpFoldResult pylir::Dialect::BXorOp::fold(::llvm::ArrayRef<::mlir::Attribute> operands)
{
    auto lhs = operands[0].dyn_cast_or_null<Dialect::BoolAttr>();
    auto rhs = operands[1].dyn_cast_or_null<Dialect::BoolAttr>();
    if (lhs && rhs)
    {
        return Dialect::BoolAttr::get(getContext(), lhs.getValue() ^ rhs.getValue());
    }
    if (rhs && !rhs.getValue())
    {
        return getOperand(0);
    }
    return nullptr;
}

mlir::OpFoldResult pylir::Dialect::BOrOp::fold(::llvm::ArrayRef<::mlir::Attribute> operands)
{
    auto lhs = operands[0].dyn_cast_or_null<Dialect::BoolAttr>();
    auto rhs = operands[1].dyn_cast_or_null<Dialect::BoolAttr>();
    if (lhs && rhs)
    {
        return Dialect::BoolAttr::get(getContext(), lhs.getValue() || rhs.getValue());
    }
    if (rhs)
    {
        if (rhs.getValue())
        {
            return rhs;
        }
        return getOperand(0);
    }
    return nullptr;
}

mlir::OpFoldResult pylir::Dialect::BNegOp::fold(llvm::ArrayRef<mlir::Attribute> operands)
{
    PYLIR_ASSERT(operands.size() == 1);
    if (auto input = operands[0].dyn_cast_or_null<Dialect::BoolAttr>())
    {
        return Dialect::BoolAttr::get(getContext(), !input.getValue());
    }
    return nullptr;
}

mlir::OpFoldResult pylir::Dialect::ItoFOp::fold(::llvm::ArrayRef<::mlir::Attribute> operands)
{
    PYLIR_ASSERT(operands.size() == 1);
    if (auto input = operands[0].dyn_cast_or_null<Dialect::IntegerAttr>())
    {
        return Dialect::FloatAttr::get(getContext(), input.getValue().roundToDouble());
    }
    return nullptr;
}

bool pylir::Dialect::ItoFOp::areCastCompatible(mlir::TypeRange inputs, mlir::TypeRange outputs)
{
    if (inputs.size() != 1 && outputs.size() != 1)
    {
        return false;
    }
    return inputs[0].isa<IntegerType>() && outputs[0].isa<FloatType>();
}

mlir::OpFoldResult pylir::Dialect::BtoIOp::fold(::llvm::ArrayRef<::mlir::Attribute> operands)
{
    PYLIR_ASSERT(operands.size() == 1);
    if (auto input = operands[0].dyn_cast_or_null<Dialect::BoolAttr>())
    {
        return Dialect::IntegerAttr::get(getContext(), llvm::APInt(2, input.getValue()));
    }
    return nullptr;
}

bool pylir::Dialect::BtoIOp::areCastCompatible(mlir::TypeRange inputs, mlir::TypeRange outputs)
{
    if (inputs.size() != 1 && outputs.size() != 1)
    {
        return false;
    }
    return inputs[0].isa<BoolType>() && outputs[0].isa<IntegerType>();
}

bool pylir::Dialect::BtoI1Op::areCastCompatible(mlir::TypeRange inputs, mlir::TypeRange outputs)
{
    if (inputs.size() != 1 && outputs.size() != 1)
    {
        return false;
    }
    return inputs[0].isa<BoolType>() && outputs[0].isInteger(1);
}

pylir::Dialect::GlobalOp pylir::Dialect::GlobalOp::create(mlir::Location location, llvm::StringRef name)
{
    mlir::OpBuilder builder(location.getContext());
    return builder.create<GlobalOp>(location, name);
}

pylir::Dialect::ConstantGlobalOp pylir::Dialect::ConstantGlobalOp::create(mlir::Location location, llvm::StringRef name,
                                                                          ObjectType type, mlir::Attribute initializer)
{
    mlir::OpBuilder builder(location.getContext());
    return builder.create<ConstantGlobalOp>(location, name, type, initializer);
}

mlir::OpFoldResult pylir::Dialect::MakeListOp::fold(::llvm::ArrayRef<::mlir::Attribute> operands)
{
    if (std::all_of(operands.begin(), operands.end(), [](mlir::Attribute attr) { return static_cast<bool>(attr); }))
    {
        return ListAttr::get(getContext(), operands);
    }
    return nullptr;
}

mlir::OpFoldResult pylir::Dialect::MakeTupleOp::fold(::llvm::ArrayRef<::mlir::Attribute> operands)
{
    if (std::all_of(operands.begin(), operands.end(), [](mlir::Attribute attr) { return static_cast<bool>(attr); }))
    {
        return TupleAttr::get(getContext(), operands);
    }
    return nullptr;
}

mlir::OpFoldResult pylir::Dialect::TupleToListOp::fold(::llvm::ArrayRef<::mlir::Attribute> operands)
{
    PYLIR_ASSERT(operands.size() == 1);
    if (auto tuple = operands[0].dyn_cast_or_null<TupleAttr>())
    {
        return ListAttr::get(getContext(), tuple.getValue());
    }
    return nullptr;
}

bool pylir::Dialect::TupleToListOp::areCastCompatible(mlir::TypeRange inputs, mlir::TypeRange outputs)
{
    if (inputs.size() != 1 && outputs.size() != 1)
    {
        return false;
    }
    return inputs[0].isa<TupleType>() && outputs[0].isa<ListType>();
}

mlir::OpFoldResult pylir::Dialect::ListToTupleOp::fold(::llvm::ArrayRef<::mlir::Attribute> operands)
{
    if (auto list = operands[0].dyn_cast_or_null<ListAttr>())
    {
        return TupleAttr::get(getContext(), list.getValue());
    }
    return nullptr;
}

bool pylir::Dialect::ListToTupleOp::areCastCompatible(mlir::TypeRange inputs, mlir::TypeRange outputs)
{
    if (inputs.size() != 1 && outputs.size() != 1)
    {
        return false;
    }
    return outputs[0].isa<TupleType>() && inputs[0].isa<ListType>();
}

mlir::LogicalResult pylir::Dialect::HandleOfOp::verifySymbolUses(::mlir::SymbolTableCollection& symbolTable)
{
    auto result = symbolTable.lookupNearestSymbolFrom<Dialect::GlobalOp>(*this, globalNameAttr());
    return mlir::success(result != nullptr);
}

mlir::LogicalResult pylir::Dialect::DataOfOp::verifySymbolUses(::mlir::SymbolTableCollection& symbolTable)
{
    auto result = symbolTable.lookupNearestSymbolFrom<Dialect::ConstantGlobalOp>(*this, globalNameAttr());
    return mlir::success(result != nullptr);
}

mlir::OpFoldResult pylir::Dialect::GetItemOp::fold(llvm::ArrayRef<mlir::Attribute> operands)
{
    auto integer = operands[0].dyn_cast_or_null<pylir::Dialect::IntegerAttr>();
    if (!integer)
    {
        return nullptr;
    }
    if (auto list = operands[1].dyn_cast_or_null<ListAttr>())
    {
        return list.getValue()[integer.getValue().getZExtValue()];
    }
    if (auto tuple = operands[1].dyn_cast_or_null<TupleAttr>())
    {
        return tuple.getValue()[integer.getValue().getZExtValue()];
    }
    if (auto string = operands[1].dyn_cast_or_null<StringAttr>())
    {
        // TODO probably needs to be a codepoint
        auto character = string.getValue()[integer.getValue().getZExtValue()];
        return StringAttr::get(getContext(), llvm::StringRef(&character, 1));
    }
    return nullptr;
}

bool pylir::Dialect::ReinterpretOp::areCastCompatible(mlir::TypeRange inputs, mlir::TypeRange outputs)
{
    if (inputs.size() != 1 && outputs.size() != 1)
    {
        return false;
    }
    return inputs[0].isa<ObjectType>() || outputs[0].isa<ObjectType>();
}

mlir::Type pylir::Dialect::GetTypeSlotOp::returnTypeFromPredicate(mlir::MLIRContext* context,
                                                                  TypeSlotPredicate predicate)
{
    switch (predicate)
    {
        case TypeSlotPredicate::DictPtr: return mlir::IndexType::get(context);
        case TypeSlotPredicate::Call:
        case TypeSlotPredicate::New:
        case TypeSlotPredicate::Init:
            return mlir::FunctionType::get(
                context, {ObjectType::get(context), Dialect::TupleType::get(context), Dialect::DictType::get(context)},
                {ObjectType::get(context)});
        case TypeSlotPredicate::Add:
        case TypeSlotPredicate::Subtract:
        case TypeSlotPredicate::Multiply:
        case TypeSlotPredicate::Remainder:
        case TypeSlotPredicate::Divmod:
        case TypeSlotPredicate::LShift:
        case TypeSlotPredicate::RShift:
        case TypeSlotPredicate::And:
        case TypeSlotPredicate::Xor:
        case TypeSlotPredicate::Or:
        case TypeSlotPredicate::InPlaceAdd:
        case TypeSlotPredicate::InPlaceSubtract:
        case TypeSlotPredicate::InPlaceMultiply:
        case TypeSlotPredicate::InPlaceRemainder:
        case TypeSlotPredicate::InPlaceLShift:
        case TypeSlotPredicate::InPlaceRShift:
        case TypeSlotPredicate::InPlaceAnd:
        case TypeSlotPredicate::InPlaceXor:
        case TypeSlotPredicate::InPlaceOr:
        case TypeSlotPredicate::FloorDivide:
        case TypeSlotPredicate::TrueDivide:
        case TypeSlotPredicate::InPlaceTrueDivide:
        case TypeSlotPredicate::InPlaceFloorDivide:
        case TypeSlotPredicate::MatrixMultiply:
        case TypeSlotPredicate::InPlaceMatrixMultiply:
        case TypeSlotPredicate::GetItem:
        case TypeSlotPredicate::Missing:
        case TypeSlotPredicate::DelItem:
        case TypeSlotPredicate::Contains:
        case TypeSlotPredicate::GetAttr:
        case TypeSlotPredicate::Eq:
        case TypeSlotPredicate::Ne:
        case TypeSlotPredicate::Lt:
        case TypeSlotPredicate::Gt:
        case TypeSlotPredicate::Le:
        case TypeSlotPredicate::Ge:
            return mlir::FunctionType::get(context, {ObjectType::get(context), ObjectType::get(context)},
                                           {ObjectType::get(context)});
        case TypeSlotPredicate::Power:
        case TypeSlotPredicate::InPlacePower:
        case TypeSlotPredicate::SetItem:
        case TypeSlotPredicate::SetAttr:
        case TypeSlotPredicate::DescrGet:
        case TypeSlotPredicate::DescrSet:
            return mlir::FunctionType::get(
                context, {ObjectType::get(context), ObjectType::get(context), ObjectType::get(context)},
                {ObjectType::get(context)});
        case TypeSlotPredicate::Negative:
        case TypeSlotPredicate::Positive:
        case TypeSlotPredicate::Absolute:
        case TypeSlotPredicate::Bool:
        case TypeSlotPredicate::Invert:
        case TypeSlotPredicate::Int:
        case TypeSlotPredicate::Float:
        case TypeSlotPredicate::Index:
        case TypeSlotPredicate::Length:
        case TypeSlotPredicate::Iter:
        case TypeSlotPredicate::Hash:
        case TypeSlotPredicate::Str:
        case TypeSlotPredicate::Repr:
        case TypeSlotPredicate::IterNext:
        case TypeSlotPredicate::Del:
            return mlir::FunctionType::get(context, {ObjectType::get(context)}, {ObjectType::get(context)});
        case TypeSlotPredicate::Dict: return Dialect::DictType::get(context);
        case TypeSlotPredicate::Bases: return Dialect::TupleType::get(context);
    }
    PYLIR_UNREACHABLE;
}

mlir::LogicalResult pylir::Dialect::GetTypeSlotOp::inferReturnTypes(
    ::mlir::MLIRContext* context, ::llvm::Optional<::mlir::Location>, ::mlir::ValueRange operands,
    ::mlir::DictionaryAttr attributes, ::mlir::RegionRange, ::llvm::SmallVectorImpl<::mlir::Type>& inferredReturnTypes)
{
    Adaptor adaptor(operands, attributes);
    auto pred = symbolizeTypeSlotPredicate(adaptor.predicate().getInt());
    if (!pred)
    {
        return mlir::failure();
    }
    inferredReturnTypes.push_back(returnTypeFromPredicate(context, *pred));
    inferredReturnTypes.push_back(mlir::IntegerType::get(context, 1));
    return mlir::success();
}

#include <pylir/Optimizer/Dialect/PylirOpsEnums.cpp.inc>
#include <pylir/Optimizer/Dialect/PylirTypeObjects.hpp>

// TODO: Remove in MLIR 13
using namespace mlir;
#define GET_OP_CLASSES
#include <pylir/Optimizer/Dialect/PylirOps.cpp.inc>
