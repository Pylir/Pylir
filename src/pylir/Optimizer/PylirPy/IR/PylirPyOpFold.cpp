// Copyright 2022 Markus Böck
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <mlir/Dialect/Arithmetic/IR/Arithmetic.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlowOps.h>
#include <mlir/IR/Matchers.h>

#include <llvm/ADT/StringSet.h>
#include <llvm/ADT/TypeSwitch.h>

#include <pylir/Support/Variant.hpp>

#include "PylirPyDialect.hpp"
#include "PylirPyOps.hpp"

namespace
{
template <class T>
struct TupleExpansionRemover : mlir::OpRewritePattern<T>
{
    using mlir::OpRewritePattern<T>::OpRewritePattern;

    mlir::LogicalResult match(T op) const final
    {
        return mlir::success(
            llvm::any_of(op.getIterArgs(),
                         [&](const auto& variant)
                         {
                             auto* expansion = std::get_if<pylir::Py::IterExpansion>(&variant);
                             if (!expansion)
                             {
                                 return false;
                             }
                             auto definingOp = expansion->value.getDefiningOp();
                             if (!definingOp)
                             {
                                 return false;
                             }
                             if (auto constant = mlir::dyn_cast<pylir::Py::ConstantOp>(definingOp))
                             {
                                 // TODO: StringAttr
                                 return constant.getConstant()
                                     .template isa<pylir::Py::ListAttr, pylir::Py::TupleAttr, pylir::Py::SetAttr>();
                             }
                             return mlir::isa<pylir::Py::MakeTupleOp, pylir::Py::MakeTupleExOp>(definingOp);
                         }));
    }

protected:
    llvm::SmallVector<pylir::Py::IterArg> getNewExpansions(T op, mlir::OpBuilder& builder) const
    {
        builder.setInsertionPoint(op);
        llvm::SmallVector<pylir::Py::IterArg> currentArgs = op.getIterArgs();
        for (auto begin = currentArgs.begin(); begin != currentArgs.end();)
        {
            auto* expansion = std::get_if<pylir::Py::IterExpansion>(&*begin);
            if (!expansion)
            {
                begin++;
                continue;
            }
            llvm::TypeSwitch<mlir::Operation*>(expansion->value.getDefiningOp())
                .Case<pylir::Py::MakeTupleOp, pylir::Py::MakeTupleExOp>(
                    [&](auto subOp)
                    {
                        auto subRange = subOp.getIterArgs();
                        begin = currentArgs.erase(begin);
                        begin = currentArgs.insert(begin, subRange.begin(), subRange.end());
                    })
                .Case(
                    [&](pylir::Py::ConstantOp constant)
                    {
                        llvm::TypeSwitch<mlir::Attribute>(constant.getConstant())
                            .Case<pylir::Py::ListAttr, pylir::Py::SetAttr, pylir::Py::TupleAttr>(
                                [&](auto attr)
                                {
                                    auto values = attr.getValue();
                                    begin = currentArgs.erase(begin);
                                    auto range = llvm::map_range(values,
                                                                 [&](mlir::Attribute attribute)
                                                                 {
                                                                     // TODO: More accurate type?
                                                                     return constant->getDialect()
                                                                         ->materializeConstant(
                                                                             builder, attribute,
                                                                             builder.getType<pylir::Py::DynamicType>(),
                                                                             op.getLoc())
                                                                         ->getResult(0);
                                                                 });
                                    begin = currentArgs.insert(begin, range.begin(), range.end());
                                })
                            .Default([&](auto&&) { begin++; });
                    })
                .Default([&](auto&&) { begin++; });
        }
        return currentArgs;
    }
};

template <class T>
struct MakeOpTupleExpansionRemove : TupleExpansionRemover<T>
{
    using TupleExpansionRemover<T>::TupleExpansionRemover;

    void rewrite(T op, mlir::PatternRewriter& rewriter) const override
    {
        rewriter.replaceOpWithNewOp<T>(op, this->getNewExpansions(op, rewriter));
    }
};

template <class T>
struct MakeExOpTupleExpansionRemove : TupleExpansionRemover<T>
{
    using TupleExpansionRemover<T>::TupleExpansionRemover;

    void rewrite(T op, mlir::PatternRewriter& rewriter) const override
    {
        rewriter.replaceOpWithNewOp<T>(op, this->getNewExpansions(op, rewriter), op.getHappyPath(),
                                       op.getNormalDestOperands(), op.getExceptionPath(), op.getUnwindDestOperands());
    }
};

template <class ExOp, llvm::ArrayRef<std::int32_t> (ExOp::*expansionAttr)()>
struct MakeExOpExceptionSimplifier : mlir::OpRewritePattern<ExOp>
{
    using mlir::OpRewritePattern<ExOp>::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(ExOp op, mlir::PatternRewriter& rewriter) const override
    {
        if (!(op.*expansionAttr)().empty())
        {
            return mlir::failure();
        }
        auto happyPath = op.getHappyPath();
        if (!happyPath->getSinglePredecessor())
        {
            auto newOp = op.cloneWithoutExceptionHandling(rewriter);
            rewriter.replaceOp(op, newOp->getResults());
            rewriter.setInsertionPointAfter(newOp);
            rewriter.create<mlir::cf::BranchOp>(newOp->getLoc(), happyPath);
            return mlir::success();
        }
        rewriter.mergeBlocks(happyPath, op->getBlock(), op.getNormalDestOperands());
        auto newOp = op.cloneWithoutExceptionHandling(rewriter);
        rewriter.replaceOp(op, newOp->getResults());
        return mlir::success();
    }
};

struct NoopBlockArgRemove : mlir::OpInterfaceRewritePattern<mlir::BranchOpInterface>
{
    using mlir::OpInterfaceRewritePattern<mlir::BranchOpInterface>::OpInterfaceRewritePattern;

    mlir::LogicalResult matchAndRewrite(mlir::BranchOpInterface op, mlir::PatternRewriter& rewriter) const override
    {
        bool changed = false;
        for (auto& iter : llvm::enumerate(op->getSuccessors()))
        {
            if (!iter.value()->getSinglePredecessor())
            {
                continue;
            }
            auto succOps = op.getSuccessorOperands(iter.index());
            if (iter.value()->getNumArguments() == succOps.getProducedOperandCount())
            {
                continue;
            }
            changed = true;
            auto* newSucc = rewriter.splitBlock(iter.value(), iter.value()->begin());
            auto nonProduced = iter.value()->getArguments().take_front(succOps.getProducedOperandCount());
            newSucc->addArguments(
                llvm::to_vector(llvm::map_range(nonProduced, [](mlir::BlockArgument arg) { return arg.getType(); })),
                llvm::to_vector(llvm::map_range(nonProduced, [](mlir::BlockArgument arg) { return arg.getLoc(); })));
            rewriter.updateRootInPlace(
                op,
                [&]
                {
                    for (auto [blockArg, repl] :
                         llvm::zip(iter.value()->getArguments().drop_front(succOps.getProducedOperandCount()),
                                   succOps.getForwardedOperands()))
                    {
                        blockArg.replaceAllUsesWith(repl);
                    }
                    op->setSuccessor(newSucc, iter.index());
                    succOps.erase(succOps.getProducedOperandCount(), succOps.getForwardedOperands().size());
                });
        }
        return mlir::success(changed);
    }
};

struct PassthroughArgRemove : mlir::OpInterfaceRewritePattern<mlir::BranchOpInterface>
{
    using mlir::OpInterfaceRewritePattern<mlir::BranchOpInterface>::OpInterfaceRewritePattern;

    mlir::LogicalResult matchAndRewrite(mlir::BranchOpInterface op, mlir::PatternRewriter& rewriter) const override
    {
        bool changed = false;
        for (auto& iter : llvm::enumerate(op->getSuccessors()))
        {
            if (iter.value()->getNumArguments() != 0)
            {
                continue;
            }
            auto brOp = mlir::dyn_cast_or_null<mlir::cf::BranchOp>(iter.value()->getTerminator());
            if (!brOp)
            {
                continue;
            }
            if (&iter.value()->front() != brOp)
            {
                continue;
            }
            if (llvm::any_of(brOp.getDestOperands(), [op](mlir::Value value) { return value.getDefiningOp() == op; }))
            {
                continue;
            }
            changed = true;
            rewriter.updateRootInPlace(op,
                                       [&]
                                       {
                                           op->setSuccessor(brOp.getSuccessor(), iter.index());
                                           op.getSuccessorOperands(iter.index()).append(brOp.getDestOperands());
                                       });
        }
        return mlir::success(changed);
    }
};

} // namespace

void pylir::Py::MakeTupleOp::getCanonicalizationPatterns(::mlir::RewritePatternSet& results,
                                                         ::mlir::MLIRContext* context)
{
    results.add<MakeOpTupleExpansionRemove<MakeTupleOp>>(context);
}

void pylir::Py::MakeListOp::getCanonicalizationPatterns(::mlir::RewritePatternSet& results,
                                                        ::mlir::MLIRContext* context)
{
    results.add<MakeOpTupleExpansionRemove<MakeListOp>>(context);
}

void pylir::Py::MakeSetOp::getCanonicalizationPatterns(::mlir::RewritePatternSet& results, ::mlir::MLIRContext* context)
{
    results.add<MakeOpTupleExpansionRemove<pylir::Py::MakeSetOp>>(context);
}

void pylir::Py::MakeTupleExOp::getCanonicalizationPatterns(::mlir::RewritePatternSet& results,
                                                           ::mlir::MLIRContext* context)
{
    results.add<MakeExOpTupleExpansionRemove<MakeTupleExOp>>(context);
    results.add<MakeExOpExceptionSimplifier<MakeTupleExOp, &MakeTupleExOp::getIterExpansion>>(context);
}

void pylir::Py::MakeListExOp::getCanonicalizationPatterns(::mlir::RewritePatternSet& results,
                                                          ::mlir::MLIRContext* context)
{
    results.add<MakeExOpTupleExpansionRemove<MakeListExOp>>(context);
    results.add<MakeExOpExceptionSimplifier<MakeListExOp, &MakeListExOp::getIterExpansion>>(context);
}

void pylir::Py::MakeSetExOp::getCanonicalizationPatterns(::mlir::RewritePatternSet& results,
                                                         ::mlir::MLIRContext* context)
{
    results.add<MakeExOpTupleExpansionRemove<MakeSetExOp>>(context);
    results.add<MakeExOpExceptionSimplifier<MakeSetExOp, &MakeSetExOp::getIterExpansion>>(context);
}

void pylir::Py::MakeDictExOp::getCanonicalizationPatterns(::mlir::RewritePatternSet& results,
                                                          ::mlir::MLIRContext* context)
{
    results.add<MakeExOpExceptionSimplifier<MakeDictExOp, &MakeDictExOp::getMappingExpansion>>(context);
}

mlir::OpFoldResult pylir::Py::ConstantOp::fold(::llvm::ArrayRef<::mlir::Attribute>)
{
    return getConstantAttr();
}

namespace
{
pylir::Py::ObjectAttrInterface resolveValue(mlir::Operation* op, mlir::Attribute attr, bool onlyConstGlobal = true)
{
    auto ref = attr.dyn_cast_or_null<mlir::SymbolRefAttr>();
    if (!ref)
    {
        return attr.dyn_cast_or_null<pylir::Py::ObjectAttrInterface>();
    }
    auto value = mlir::SymbolTable::lookupNearestSymbolFrom<pylir::Py::GlobalValueOp>(op, ref);
    if (!value || (!value.getConstant() && onlyConstGlobal))
    {
        return attr.dyn_cast_or_null<pylir::Py::ObjectAttrInterface>();
    }
    return value.getInitializerAttr();
}

llvm::SmallVector<mlir::OpFoldResult> resolveTupleOperands(mlir::Operation* context, mlir::Value operand)
{
    llvm::SmallVector<mlir::OpFoldResult> result;
    mlir::Attribute attr;
    if (mlir::matchPattern(operand, mlir::m_Constant(&attr)))
    {
        auto tuple = resolveValue(context, attr).dyn_cast_or_null<pylir::Py::TupleAttr>();
        if (!tuple)
        {
            result.emplace_back(nullptr);
            return result;
        }
        result.insert(result.end(), tuple.getValue().begin(), tuple.getValue().end());
        return result;
    }
    if (!operand.getDefiningOp())
    {
        result.emplace_back(nullptr);
        return result;
    }
    llvm::TypeSwitch<mlir::Operation*>(operand.getDefiningOp())
        .Case<pylir::Py::MakeTupleOp, pylir::Py::MakeTupleExOp>(
            [&](auto makeTuple)
            {
                auto args = makeTuple.getIterArgs();
                for (auto& arg : args)
                {
                    pylir::match(
                        arg,
                        [&](mlir::Value value)
                        {
                            mlir::Attribute attr;
                            if (mlir::matchPattern(value, mlir::m_Constant(&attr)))
                            {
                                result.emplace_back(attr);
                            }
                            else
                            {
                                result.emplace_back(value);
                            }
                        },
                        [&](auto) { result.emplace_back(nullptr); });
                }
            })
        .Case(
            [&](pylir::Py::TuplePrependOp op)
            {
                mlir::Attribute attr;
                if (mlir::matchPattern(op.getInput(), mlir::m_Constant(&attr)))
                {
                    result.emplace_back(attr);
                }
                else
                {
                    result.emplace_back(op.getInput());
                }
                auto rest = resolveTupleOperands(context, op.getTuple());
                result.insert(result.end(), rest.begin(), rest.end());
            })
        .Case(
            [&](pylir::Py::TupleDropFrontOp op)
            {
                auto tuple = resolveTupleOperands(context, op.getTuple());
                mlir::IntegerAttr attr;
                if (!mlir::matchPattern(op.getCount(), mlir::m_Constant(&attr)))
                {
                    result.emplace_back(nullptr);
                    return;
                }
                auto begin = tuple.begin();
                for (std::size_t i = 0; attr.getValue().ugt(i) && begin != tuple.end() && *begin; i++, begin++)
                    ;
                result.insert(result.end(), begin, tuple.end());
            })
        .Default([&](auto) { result.emplace_back(nullptr); });
    return result;
}

} // namespace

mlir::OpFoldResult pylir::Py::TypeOfOp::fold(llvm::ArrayRef<mlir::Attribute> operands)
{
    if (auto input = resolveValue(*this, operands[0], false))
    {
        return input.getTypeObject();
    }
    if (auto op = getObject().getDefiningOp<pylir::Py::ObjectFromTypeObjectInterface>())
    {
        return op.getTypeObject();
    }
    if (auto refineable = getObject().getDefiningOp<Py::TypeRefineableInterface>())
    {
        llvm::SmallVector<Py::TypeAttrUnion> operandTypes(refineable->getNumOperands(), nullptr);
        mlir::SymbolTableCollection collection;
        llvm::SmallVector<Py::ObjectTypeInterface> res;
        if (refineable.refineTypes(operandTypes, res, collection) == TypeRefineResult::Failure)
        {
            return nullptr;
        }
        return res[getObject().cast<mlir::OpResult>().getResultNumber()].getTypeObject();
    }
    return nullptr;
}

mlir::OpFoldResult pylir::Py::GetSlotOp::fold(::llvm::ArrayRef<::mlir::Attribute> operands)
{
    auto object = resolveValue(*this, operands[0]);
    if (!object)
    {
        return nullptr;
    }
    const auto& map = object.getSlots();
    auto result = map.get(getSlotAttr());
    if (!result)
    {
        return Py::UnboundAttr::get(getContext());
    }
    return result;
}

mlir::OpFoldResult pylir::Py::TupleGetItemOp::fold(::llvm::ArrayRef<::mlir::Attribute> operands)
{
    if (auto tupleCopy = getTuple().getDefiningOp<pylir::Py::TupleCopyOp>())
    {
        getTupleMutable().assign(tupleCopy.getTuple());
        return mlir::Value{*this};
    }
    auto indexAttr = operands[1].dyn_cast_or_null<mlir::IntegerAttr>();
    if (!indexAttr)
    {
        return nullptr;
    }
    auto index = indexAttr.getValue().getZExtValue();
    auto tupleOperands = resolveTupleOperands(*this, getTuple());
    auto ref = llvm::makeArrayRef(tupleOperands).take_front(index + 1);
    if (ref.size() != index + 1 || llvm::any_of(ref, [](auto result) -> bool { return !result; }))
    {
        return nullptr;
    }
    return ref[index];
}

mlir::OpFoldResult pylir::Py::TupleLenOp::fold(llvm::ArrayRef<mlir::Attribute> operands)
{
    if (auto tupleCopy = getInput().getDefiningOp<pylir::Py::TupleCopyOp>())
    {
        getInputMutable().assign(tupleCopy.getTuple());
        return mlir::Value{*this};
    }
    if (auto makeTuple = getInput().getDefiningOp<Py::MakeTupleOp>();
        makeTuple && makeTuple.getIterExpansionAttr().empty())
    {
        return mlir::IntegerAttr::get(getType(), makeTuple.getArguments().size());
    }
    if (auto tuple = resolveValue(*this, operands[0]).dyn_cast_or_null<Py::TupleAttr>())
    {
        return mlir::IntegerAttr::get(getType(), tuple.getValue().size());
    }
    return nullptr;
}

mlir::OpFoldResult pylir::Py::TuplePrependOp::fold(::llvm::ArrayRef<::mlir::Attribute> operands)
{
    if (auto tupleCopy = getTuple().getDefiningOp<pylir::Py::TupleCopyOp>())
    {
        getTupleMutable().assign(tupleCopy.getTuple());
        return mlir::Value{*this};
    }
    auto element = operands[0];
    auto tuple = resolveValue(*this, operands[1]).dyn_cast_or_null<Py::TupleAttr>();
    if (tuple && element)
    {
        llvm::SmallVector<mlir::Attribute> values{element};
        values.append(tuple.getValue().begin(), tuple.getValue().end());
        return Py::TupleAttr::get(getContext(), values);
    }
    return nullptr;
}

::mlir::OpFoldResult pylir::Py::TupleDropFrontOp::fold(::llvm::ArrayRef<::mlir::Attribute> operands)
{
    if (auto tupleCopy = getTuple().getDefiningOp<pylir::Py::TupleCopyOp>())
    {
        getTupleMutable().assign(tupleCopy.getTuple());
        return mlir::Value{*this};
    }
    auto constant = resolveValue(*this, operands[1]).dyn_cast_or_null<Py::TupleAttr>();
    if (constant && constant.getValue().empty())
    {
        return Py::TupleAttr::get(getContext());
    }
    auto index = operands[0].dyn_cast_or_null<mlir::IntegerAttr>();
    if (!index || !constant)
    {
        return nullptr;
    }
    if (index.getValue().getZExtValue() > constant.getValue().size())
    {
        return Py::TupleAttr::get(getContext());
    }
    return Py::TupleAttr::get(getContext(), constant.getValue().drop_front(index.getValue().getZExtValue()));
}

::mlir::OpFoldResult pylir::Py::TupleCopyOp::fold(::llvm::ArrayRef<::mlir::Attribute> operands)
{
    if (auto tupleCopy = getTuple().getDefiningOp<pylir::Py::TupleCopyOp>())
    {
        getTupleMutable().assign(tupleCopy.getTuple());
        return mlir::Value{*this};
    }
    auto type = operands[1].dyn_cast_or_null<mlir::FlatSymbolRefAttr>();
    if (type && type.getValue() == Builtins::Tuple.name
        && mlir::isa_and_nonnull<Py::TupleDropFrontOp, Py::TuplePrependOp, Py::MakeTupleOp, Py::MakeTupleExOp>(
            getTuple().getDefiningOp()))
    {
        return getTuple();
    }
    auto constant = resolveValue(*this, operands[0]).dyn_cast_or_null<Py::TupleAttr>();
    if (!constant || !type)
    {
        return nullptr;
    }
    return Py::TupleAttr::get(getContext(), constant.getValue(), type);
}

namespace
{
template <class Attr>
llvm::Optional<Attr> doConstantIterExpansion(::llvm::ArrayRef<::mlir::Attribute> operands,
                                             llvm::ArrayRef<int32_t> iterExpansion, mlir::MLIRContext* context)
{
    if (!std::all_of(operands.begin(), operands.end(),
                     [](mlir::Attribute attr) -> bool { return static_cast<bool>(attr); }))
    {
        return llvm::None;
    }
    llvm::SmallVector<mlir::Attribute> result;
    auto range = iterExpansion;
    const auto* begin = range.begin();
    for (const auto& pair : llvm::enumerate(operands))
    {
        if (begin == range.end() || static_cast<std::int32_t>(pair.index()) != *begin)
        {
            result.push_back(pair.value());
            continue;
        }
        begin++;
        if (!llvm::TypeSwitch<mlir::Attribute, bool>(pair.value())
                 .Case<pylir::Py::TupleAttr, pylir::Py::ListAttr, pylir::Py::SetAttr>(
                     [&](auto attr)
                     {
                         result.insert(result.end(), attr.getValue().begin(), attr.getValue().end());
                         return true;
                     })
                 // TODO: string attr
                 .Default(false))
        {
            return llvm::None;
        }
    }
    return Attr::get(context, result);
}
} // namespace

mlir::OpFoldResult pylir::Py::MakeTupleOp::fold(::llvm::ArrayRef<::mlir::Attribute> operands)
{
    if (auto result = doConstantIterExpansion<pylir::Py::TupleAttr>(operands, getIterExpansion(), getContext()))
    {
        return *result;
    }
    return nullptr;
}

mlir::OpFoldResult pylir::Py::BoolToI1Op::fold(::llvm::ArrayRef<mlir::Attribute> operands)
{
    if (auto op = getInput().getDefiningOp<Py::BoolFromI1Op>())
    {
        return op.getInput();
    }
    auto boolean = operands[0].dyn_cast_or_null<Py::BoolAttr>();
    if (!boolean)
    {
        return nullptr;
    }
    return mlir::BoolAttr::get(getContext(), boolean.getValue());
}

mlir::OpFoldResult pylir::Py::BoolFromI1Op::fold(::llvm::ArrayRef<mlir::Attribute> operands)
{
    if (auto op = getInput().getDefiningOp<Py::BoolToI1Op>())
    {
        return op.getInput();
    }
    auto boolean = operands[0].dyn_cast_or_null<mlir::BoolAttr>();
    if (!boolean)
    {
        return nullptr;
    }
    return Py::BoolAttr::get(getContext(), boolean.getValue());
}

mlir::OpFoldResult pylir::Py::IntFromUnsignedOp::fold(::llvm::ArrayRef<::mlir::Attribute> operands)
{
    auto integer = operands[0].dyn_cast_or_null<mlir::IntegerAttr>();
    if (!integer)
    {
        return nullptr;
    }
    return Py::IntAttr::get(getContext(), BigInt(integer.getValue().getZExtValue()));
}

mlir::OpFoldResult pylir::Py::IntFromSignedOp::fold(::llvm::ArrayRef<::mlir::Attribute> operands)
{
    if (auto op = getInput().getDefiningOp<IntToIndexOp>())
    {
        return op.getInput();
    }
    auto integer = operands[0].dyn_cast_or_null<mlir::IntegerAttr>();
    if (!integer)
    {
        return nullptr;
    }
    return Py::IntAttr::get(getContext(), BigInt(integer.getValue().getSExtValue()));
}

mlir::OpFoldResult pylir::Py::IntToIndexOp::fold(::llvm::ArrayRef<::mlir::Attribute> operands)
{
    if (auto op = getInput().getDefiningOp<IntFromSignedOp>())
    {
        return op.getInput();
    }
    if (auto op = getInput().getDefiningOp<IntFromUnsignedOp>())
    {
        return op.getInput();
    }

    auto integer = operands[0].dyn_cast_or_null<Py::IntAttrInterface>();
    if (!integer)
    {
        return nullptr;
    }
    std::size_t bitWidth = mlir::DataLayout::closest(*this).getTypeSizeInBits(getResult().getType());
    if (integer.getIntegerValue() < BigInt(0))
    {
        auto optional = integer.getIntegerValue().tryGetInteger<std::intmax_t>();
        if (!optional || !llvm::APInt(sizeof(*optional) * 8, *optional, true).isSignedIntN(bitWidth))
        {
            // TODO: I will probably want a poison value here in the future.
            return mlir::IntegerAttr::get(getType(), 0);
        }
        return mlir::IntegerAttr::get(getType(), *optional);
    }
    auto optional = integer.getIntegerValue().tryGetInteger<std::uintmax_t>();
    if (!optional || !llvm::APInt(sizeof(*optional) * 8, *optional, false).isIntN(bitWidth))
    {
        // TODO: I will probably want a poison value here in the future.
        return mlir::IntegerAttr::get(getType(), 0);
    }
    return mlir::IntegerAttr::get(getType(), *optional);
}

mlir::OpFoldResult pylir::Py::IntCmpOp::fold(::llvm::ArrayRef<::mlir::Attribute> operands)
{
    auto lhs = operands[0].dyn_cast_or_null<IntAttrInterface>();
    auto rhs = operands[1].dyn_cast_or_null<IntAttrInterface>();
    if (!lhs || !rhs)
    {
        return nullptr;
    }
    bool result;
    switch (getPred())
    {
        case IntCmpKind::eq: result = lhs.getIntegerValue() == rhs.getIntegerValue(); break;
        case IntCmpKind::ne: result = lhs.getIntegerValue() != rhs.getIntegerValue(); break;
        case IntCmpKind::lt: result = lhs.getIntegerValue() < rhs.getIntegerValue(); break;
        case IntCmpKind::le: result = lhs.getIntegerValue() <= rhs.getIntegerValue(); break;
        case IntCmpKind::gt: result = lhs.getIntegerValue() > rhs.getIntegerValue(); break;
        case IntCmpKind::ge: result = lhs.getIntegerValue() >= rhs.getIntegerValue(); break;
    }
    return mlir::BoolAttr::get(getContext(), result);
}

mlir::OpFoldResult pylir::Py::IntToStrOp::fold(::llvm::ArrayRef<::mlir::Attribute> operands)
{
    auto integer = operands[0].dyn_cast_or_null<IntAttrInterface>();
    if (!integer)
    {
        return nullptr;
    }
    return StrAttr::get(getContext(), integer.getIntegerValue().toString());
}

mlir::OpFoldResult pylir::Py::IsUnboundValueOp::fold(::llvm::ArrayRef<::mlir::Attribute> operands)
{
    if (operands[0])
    {
        return mlir::BoolAttr::get(getContext(), operands[0].isa<Py::UnboundAttr>());
    }
    if (auto blockArg = getValue().dyn_cast<mlir::BlockArgument>(); blockArg)
    {
        if (mlir::isa_and_nonnull<mlir::FunctionOpInterface>(blockArg.getOwner()->getParentOp())
            && blockArg.getOwner()->isEntryBlock())
        {
            return mlir::BoolAttr::get(getContext(), false);
        }
        return nullptr;
    }
    // If the defining op has the AlwaysBound trait then it is false. Also manually sanction some ops from other
    // dialects
    auto* op = getValue().getDefiningOp();
    if (!op)
    {
        return nullptr;
    }
    if (op->hasTrait<Py::AlwaysBound>())
    {
        return mlir::BoolAttr::get(getContext(), false);
    }
    return nullptr;
}

mlir::OpFoldResult pylir::Py::IsOp::fold(::llvm::ArrayRef<::mlir::Attribute> operands)
{
    if (operands[0] && operands[1] && operands[0] == operands[1])
    {
        return mlir::BoolAttr::get(getContext(), true);
    }
    if (getLhs() == getRhs())
    {
        return mlir::BoolAttr::get(getContext(), true);
    }
    {
        auto lhsEffect = mlir::dyn_cast_or_null<mlir::MemoryEffectOpInterface>(getLhs().getDefiningOp());
        auto rhsEffect = mlir::dyn_cast_or_null<mlir::MemoryEffectOpInterface>(getRhs().getDefiningOp());
        bool lhsAlloc = lhsEffect && lhsEffect.hasEffect<mlir::MemoryEffects::Allocate>();
        bool rhsAlloc = rhsEffect && rhsEffect.hasEffect<mlir::MemoryEffects::Allocate>();
        if ((lhsAlloc && rhsAlloc) || (operands[0].dyn_cast_or_null<mlir::SymbolRefAttr>() && rhsAlloc)
            || (lhsAlloc && operands[1].dyn_cast_or_null<mlir::SymbolRefAttr>()))
        {
            return mlir::BoolAttr::get(getContext(), false);
        }
    }
    if (auto* lhsDef = getLhs().getDefiningOp(); lhsDef && lhsDef->hasTrait<Py::ReturnsImmutable>())
    {
        if (auto* rhsDef = getRhs().getDefiningOp(); rhsDef && rhsDef->hasTrait<Py::ReturnsImmutable>())
        {
            return mlir::BoolAttr::get(getContext(), false);
        }
    }
    return nullptr;
}

mlir::OpFoldResult pylir::Py::TypeMROOp::fold(::llvm::ArrayRef<::mlir::Attribute> attributes)
{
    auto object = resolveValue(*this, attributes[0], false).dyn_cast_or_null<pylir::Py::TypeAttr>();
    if (!object)
    {
        return nullptr;
    }
    return object.getMroTuple();
}

mlir::LogicalResult pylir::Py::MROLookupOp::fold(::llvm::ArrayRef<::mlir::Attribute> constantOperands,
                                                 ::llvm::SmallVectorImpl<::mlir::OpFoldResult>& results)
{
    if (auto tuple = resolveValue(*this, constantOperands[0], false).dyn_cast_or_null<pylir::Py::TupleAttr>())
    {
        for (auto iter : tuple.getValue())
        {
            auto object = resolveValue(*this, iter);
            if (!object)
            {
                return mlir::failure();
            }
            const auto& map = object.getSlots();
            if (auto result = map.get(getSlotAttr()))
            {
                results.emplace_back(result);
                results.emplace_back(mlir::BoolAttr::get(getContext(), true));
                return mlir::success();
            }
        }
        results.emplace_back(Py::UnboundAttr::get(getContext()));
        results.emplace_back(mlir::BoolAttr::get(getContext(), false));
        return mlir::success();
    }
    auto operands = resolveTupleOperands(*this, getMroTuple());
    for (auto& iter : operands)
    {
        if (!iter || !iter.is<mlir::Attribute>())
        {
            return mlir::failure();
        }
        auto object = resolveValue(*this, iter.get<mlir::Attribute>());
        if (!object)
        {
            return mlir::failure();
        }
        const auto& map = object.getSlots();
        auto result = map.get(getSlotAttr());
        if (result)
        {
            results.emplace_back(result);
            results.emplace_back(mlir::BoolAttr::get(getContext(), true));
            return mlir::success();
        }
    }
    results.emplace_back(Py::UnboundAttr::get(getContext()));
    results.emplace_back(mlir::BoolAttr::get(getContext(), false));
    return mlir::success();
}

mlir::OpFoldResult pylir::Py::TupleContainsOp::fold(::llvm::ArrayRef<::mlir::Attribute> operands)
{
    if (auto tuple = resolveValue(*this, operands[0], false).dyn_cast_or_null<pylir::Py::TupleAttr>())
    {
        if (auto element = operands[1])
        {
            return mlir::BoolAttr::get(getContext(), llvm::is_contained(tuple.getValue(), element));
        }
    }
    if (auto tupleCopy = getTuple().getDefiningOp<pylir::Py::TupleCopyOp>())
    {
        getTupleMutable().assign(tupleCopy.getTuple());
        return mlir::Value{*this};
    }
    auto tupleOperands = resolveTupleOperands(*this, getTuple());
    bool hadWildcard = false;
    for (auto& op : tupleOperands)
    {
        if (!op)
        {
            hadWildcard = true;
            continue;
        }
        if (op == mlir::OpFoldResult{getElement()} || op == mlir::OpFoldResult{operands[1]})
        {
            return mlir::BoolAttr::get(getContext(), true);
        }
    }
    if (hadWildcard)
    {
        return nullptr;
    }
    return mlir::BoolAttr::get(getContext(), false);
}

mlir::OpFoldResult pylir::Py::StrConcatOp::fold(::llvm::ArrayRef<::mlir::Attribute> operands)
{
    std::string res;
    for (const auto& iter : operands)
    {
        auto str = iter.dyn_cast_or_null<StrAttr>();
        if (!str)
        {
            return nullptr;
        }
        res += str.getValue();
    }
    return StrAttr::get(getContext(), res);
}

mlir::OpFoldResult pylir::Py::DictTryGetItemOp::fold(::llvm::ArrayRef<mlir::Attribute> operands)
{
    auto constantDict = resolveValue(*this, operands[0]).dyn_cast_or_null<Py::DictAttr>();
    if (constantDict && constantDict.getValue().empty())
    {
        return Py::UnboundAttr::get(getContext());
    }
    if (!constantDict)
    {
        return nullptr;
    }
    auto resolvedKey = resolveValue(*this, operands[1]);
    if (!constantDict || !operands[1])
    {
        return nullptr;
    }
    // TODO: Make this work in the general case for builtin types (that have a known __eq__ impl)
    for (const auto& [key, value] : constantDict.getValue())
    {
        if (key == operands[1] || resolveValue(*this, key) == resolvedKey)
        {
            return value;
        }
    }
    return nullptr;
}

mlir::OpFoldResult pylir::Py::DictLenOp::fold(::llvm::ArrayRef<mlir::Attribute> operands)
{
    auto constantDict = resolveValue(*this, operands[0]).dyn_cast_or_null<Py::DictAttr>();
    if (!constantDict)
    {
        return nullptr;
    }
    return mlir::IntegerAttr::get(mlir::IndexType::get(getContext()), constantDict.getValue().size());
}

mlir::LogicalResult pylir::Py::GlobalValueOp::fold(::llvm::ArrayRef<mlir::Attribute>,
                                                   llvm::SmallVectorImpl<mlir::OpFoldResult>&)
{
    static llvm::StringSet<> immutableTypes = {
        Builtins::Float.name, Builtins::Int.name, Builtins::Bool.name, Builtins::Str.name, Builtins::Tuple.name,
    };
    if (!getConstant() && getInitializer() && immutableTypes.contains(getInitializer()->getTypeObject().getValue()))
    {
        setConstantAttr(mlir::UnitAttr::get(getContext()));
        return mlir::success();
    }
    return mlir::failure();
}

mlir::LogicalResult pylir::Py::FunctionCallOp::canonicalize(FunctionCallOp op, ::mlir::PatternRewriter& rewriter)
{
    mlir::FlatSymbolRefAttr callee;
    if (auto makeFuncOp = op.getFunction().getDefiningOp<pylir::Py::MakeFuncOp>())
    {
        callee = makeFuncOp.getFunctionAttr();
    }
    else
    {
        mlir::Attribute attribute;
        if (!mlir::matchPattern(op.getFunction(), mlir::m_Constant(&attribute)))
        {
            return mlir::failure();
        }
        auto functionAttr = resolveValue(op, attribute, false).dyn_cast_or_null<pylir::Py::FunctionAttr>();
        if (!functionAttr)
        {
            return mlir::failure();
        }
        callee = functionAttr.getValue();
    }
    rewriter.replaceOpWithNewOp<Py::CallOp>(op, op.getType(), callee, op.getCallOperands());
    return mlir::success();
}

mlir::LogicalResult pylir::Py::FunctionInvokeOp::canonicalize(FunctionInvokeOp op, ::mlir::PatternRewriter& rewriter)
{
    mlir::FlatSymbolRefAttr callee;
    if (auto makeFuncOp = op.getFunction().getDefiningOp<pylir::Py::MakeFuncOp>())
    {
        callee = makeFuncOp.getFunctionAttr();
    }
    else
    {
        mlir::Attribute attribute;
        if (!mlir::matchPattern(op.getFunction(), mlir::m_Constant(&attribute)))
        {
            return mlir::failure();
        }
        auto functionAttr = resolveValue(op, attribute, false).dyn_cast_or_null<pylir::Py::FunctionAttr>();
        if (!functionAttr)
        {
            return mlir::failure();
        }
        callee = functionAttr.getValue();
    }
    rewriter.replaceOpWithNewOp<Py::InvokeOp>(op, op.getType(), callee, op.getCallOperands(),
                                              op.getNormalDestOperands(), op.getUnwindDestOperands(), op.getHappyPath(),
                                              op.getExceptionPath());
    return mlir::success();
}

mlir::LogicalResult pylir::Py::GetSlotOp::foldUsage(mlir::Operation* lastClobber,
                                                    ::llvm::SmallVectorImpl<::mlir::OpFoldResult>& results)
{
    auto setSlotOp = mlir::dyn_cast<Py::SetSlotOp>(lastClobber);
    if (!setSlotOp)
    {
        if (mlir::isa<Py::MakeObjectOp>(lastClobber))
        {
            results.emplace_back(Py::UnboundAttr::get(getContext()));
            return mlir::success();
        }
        return mlir::failure();
    }
    if (setSlotOp.getSlotAttr() == getSlotAttr())
    {
        results.emplace_back(setSlotOp.getValue());
        return mlir::success();
    }
    return mlir::failure();
}

mlir::LogicalResult pylir::Py::DictLenOp::foldUsage(mlir::Operation* lastClobber,
                                                    ::llvm::SmallVectorImpl<::mlir::OpFoldResult>& results)
{
    auto makeDictOp = mlir::dyn_cast<Py::MakeDictOp>(lastClobber);
    // I can not fold a non empty one as I can't tell whether there are any duplicates in the arguments
    if (!makeDictOp || !makeDictOp.getKeys().empty())
    {
        return mlir::failure();
    }
    results.emplace_back(mlir::IntegerAttr::get(getType(), 0));
    return mlir::success();
}

mlir::LogicalResult pylir::Py::DictTryGetItemOp::foldUsage(mlir::Operation* lastClobber,
                                                           ::llvm::SmallVectorImpl<::mlir::OpFoldResult>& results)
{
    return llvm::TypeSwitch<mlir::Operation*, mlir::LogicalResult>(lastClobber)
        .Case(
            [&](Py::DictSetItemOp op)
            {
                if (op.getKey() == getKey())
                {
                    results.emplace_back(op.getValue());
                    return mlir::success();
                }
                return mlir::failure();
            })
        .Case(
            [&](Py::DictDelItemOp op)
            {
                if (op.getKey() == getKey())
                {
                    results.emplace_back(Py::UnboundAttr::get(getContext()));
                    return mlir::success();
                }
                return mlir::failure();
            })
        .Case<Py::MakeDictExOp, Py::MakeDictOp>(
            [&](auto op)
            {
                // We have to reverse through the map as the last key appearing in the list is the one appearing in the
                // map. Additionally, if there are any unknown values inbetween that could be equal to our key, we have
                // to abort as we can't be sure it would not be equal to our key at runtime.
                for (auto&& variant : llvm::reverse(op.getDictArgs()))
                {
                    if (std::holds_alternative<MappingExpansion>(variant))
                    {
                        return mlir::failure();
                    }
                    auto [key, value] = pylir::get<std::pair<mlir::Value, mlir::Value>>(variant);
                    if (key == getKey())
                    {
                        results.emplace_back(value);
                        return mlir::success();
                    }
                    // TODO:
                    //  some more generic mechanism to not automatically fail if we know they are definitely NOT
                    //  equal (trivial for constants at least, except for references as they need to be resolved).
                    mlir::Attribute attr1, attr2;
                    if (mlir::matchPattern(key, mlir::m_Constant(&attr1))
                        && mlir::matchPattern(getKey(), mlir::m_Constant(&attr2)) && attr1 != attr2
                        && !attr1.isa<mlir::SymbolRefAttr>())
                    {
                        continue;
                    }
                    return mlir::failure();
                }
                results.emplace_back(Py::UnboundAttr::get(getContext()));
                return mlir::success();
            })
        .Default(mlir::failure());
}

mlir::LogicalResult pylir::Py::ListLenOp::foldUsage(mlir::Operation* lastClobber,
                                                    ::llvm::SmallVectorImpl<::mlir::OpFoldResult>& results)
{
    auto makeListOp = mlir::dyn_cast<Py::MakeListOp>(lastClobber);
    if (!makeListOp || !makeListOp.getIterExpansion().empty())
    {
        return mlir::failure();
    }
    results.emplace_back(mlir::IntegerAttr::get(getType(), makeListOp.getArguments().size()));
    return mlir::success();
}

pylir::Py::TypeRefineResult
    pylir::Py::ConstantOp::refineTypes(llvm::ArrayRef<Py::TypeAttrUnion>,
                                       llvm::SmallVectorImpl<pylir::Py::ObjectTypeInterface>& result,
                                       mlir::SymbolTableCollection& table)
{
    result.push_back(typeOfConstant(getConstantAttr(), table, *this));
    return TypeRefineResult::Success;
}

pylir::Py::TypeRefineResult
    pylir::Py::MakeTupleExOp::refineTypes(llvm::ArrayRef<pylir::Py::TypeAttrUnion>,
                                          llvm::SmallVectorImpl<pylir::Py::ObjectTypeInterface>& result,
                                          mlir::SymbolTableCollection&)
{
    result.emplace_back(Py::ClassType::get(mlir::FlatSymbolRefAttr::get(getContext(), Builtins::Tuple.name)));
    return TypeRefineResult::Approximate;
}

pylir::Py::TypeRefineResult
    pylir::Py::MakeTupleOp::refineTypes(llvm::ArrayRef<pylir::Py::TypeAttrUnion> argumentTypes,
                                        llvm::SmallVectorImpl<pylir::Py::ObjectTypeInterface>& result,
                                        mlir::SymbolTableCollection&)
{
    if (!getIterExpansionAttr().empty())
    {
        result.emplace_back(Py::ClassType::get(mlir::FlatSymbolRefAttr::get(getContext(), Builtins::Tuple.name)));
        return TypeRefineResult::Approximate;
    }
    llvm::SmallVector<pylir::Py::ObjectTypeInterface> elementTypes;
    for (auto iter : argumentTypes)
    {
        if (!iter)
        {
            result.emplace_back(Py::ClassType::get(mlir::FlatSymbolRefAttr::get(getContext(), Builtins::Tuple.name)));
            return TypeRefineResult::Approximate;
        }
        elementTypes.push_back(iter.cast<Py::ObjectTypeInterface>());
    }
    result.emplace_back(Py::TupleType::get(getContext(), {}, elementTypes));
    return TypeRefineResult::Success;
}

pylir::Py::TypeRefineResult
    pylir::Py::TupleCopyOp::refineTypes(::llvm::ArrayRef<::pylir::Py::TypeAttrUnion> inputs,
                                        ::llvm::SmallVectorImpl<::pylir::Py::ObjectTypeInterface>& resultTypes,
                                        ::mlir::SymbolTableCollection&)
{
    auto typeObject = inputs[1].dyn_cast_or_null<mlir::FlatSymbolRefAttr>();
    if (!typeObject)
    {
        return TypeRefineResult::Failure;
    }
    auto tuple = inputs[0].dyn_cast_or_null<pylir::Py::TupleType>();
    if (!tuple)
    {
        resultTypes.emplace_back(Py::ClassType::get(getContext(), typeObject));
        return TypeRefineResult::Approximate;
    }
    resultTypes.emplace_back(Py::TupleType::get(getContext(), typeObject, tuple.getElements()));
    return TypeRefineResult::Success;
}

pylir::Py::TypeRefineResult
    pylir::Py::TupleGetItemOp::refineTypes(llvm::ArrayRef<Py::TypeAttrUnion> argumentTypes,
                                           llvm::SmallVectorImpl<pylir::Py::ObjectTypeInterface>& result,
                                           mlir::SymbolTableCollection&)
{
    auto tupleType = argumentTypes[0].dyn_cast_or_null<pylir::Py::TupleType>();
    if (!tupleType)
    {
        return TypeRefineResult::Failure;
    }
    if (tupleType.getElements().empty())
    {
        result.emplace_back(UnboundType::get(getContext()));
        return TypeRefineResult::Success;
    }
    auto index = argumentTypes[1].dyn_cast_or_null<mlir::IntegerAttr>();
    if (!index)
    {
        Py::ObjectTypeInterface sumType = tupleType.getElements().front();
        for (auto iter : tupleType.getElements().drop_front())
        {
            sumType = joinTypes(sumType, iter);
        }
        result.emplace_back(sumType);
        return TypeRefineResult::Success;
    }
    auto zExtValue = index.getValue().getZExtValue();
    if (zExtValue >= tupleType.getElements().size())
    {
        result.emplace_back(UnboundType::get(getContext()));
        return TypeRefineResult::Success;
    }
    result.emplace_back(tupleType.getElements()[zExtValue]);
    return TypeRefineResult::Success;
}

pylir::Py::TypeRefineResult
    pylir::Py::TupleDropFrontOp::refineTypes(llvm::ArrayRef<pylir::Py::TypeAttrUnion> argumentTypes,
                                             llvm::SmallVectorImpl<pylir::Py::ObjectTypeInterface>& result,
                                             mlir::SymbolTableCollection&)
{
    auto tupleType = argumentTypes[1].dyn_cast_or_null<Py::TupleType>();
    if (!tupleType)
    {
        result.emplace_back(Py::ClassType::get(mlir::FlatSymbolRefAttr::get(getContext(), Builtins::Tuple.name)));
        return TypeRefineResult::Approximate;
    }
    if (tupleType.getElements().empty())
    {
        result.emplace_back(tupleType);
        return TypeRefineResult::Success;
    }
    auto index = argumentTypes[0].dyn_cast_or_null<mlir::IntegerAttr>();
    if (!index)
    {
        Py::ObjectTypeInterface sumType = tupleType.getElements().front();
        for (auto iter : tupleType.getElements().drop_front())
        {
            sumType = joinTypes(sumType, iter);
        }
        result.emplace_back(sumType);
        return TypeRefineResult::Success;
    }
    if (tupleType.getElements().size() >= index.getValue().getZExtValue())
    {
        result.emplace_back(Py::TupleType::get(getContext()));
        return TypeRefineResult::Success;
    }
    result.emplace_back(
        Py::TupleType::get(getContext(), {}, tupleType.getElements().drop_front(index.getValue().getZExtValue())));
    return TypeRefineResult::Success;
}

pylir::Py::TypeRefineResult
    pylir::Py::TuplePrependOp::refineTypes(llvm::ArrayRef<pylir::Py::TypeAttrUnion> argumentTypes,
                                           llvm::SmallVectorImpl<pylir::Py::ObjectTypeInterface>& result,
                                           mlir::SymbolTableCollection&)
{
    auto tupleType = argumentTypes[1].dyn_cast_or_null<Py::TupleType>();
    // TODO: Once/if tuple type accepts nullptr elements (for unknown), the below or should not be necessary
    if (!tupleType || !argumentTypes[0].isa_and_nonnull<Py::ObjectTypeInterface>())
    {
        result.emplace_back(Py::ClassType::get(mlir::FlatSymbolRefAttr::get(getContext(), Builtins::Tuple.name)));
        return TypeRefineResult::Approximate;
    }
    llvm::SmallVector<Py::ObjectTypeInterface> elements = llvm::to_vector(tupleType.getElements());
    elements.insert(elements.begin(), argumentTypes[0].cast<Py::ObjectTypeInterface>());
    result.emplace_back(Py::TupleType::get(getContext(), {}, elements));
    return TypeRefineResult::Success;
}

namespace
{

struct ArithSelectTypeRefinable
    : public pylir::Py::TypeRefineableInterface::ExternalModel<ArithSelectTypeRefinable, mlir::arith::SelectOp>
{
    pylir::Py::TypeRefineResult refineTypes(mlir::Operation*, ::llvm::ArrayRef<::pylir::Py::TypeAttrUnion> inputs,
                                            ::llvm::SmallVectorImpl<::pylir::Py::ObjectTypeInterface>& resultTypes,
                                            ::mlir::SymbolTableCollection&) const
    {
        auto lhsType = inputs[1].dyn_cast_or_null<pylir::Py::ObjectTypeInterface>();
        auto rhsType = inputs[2].dyn_cast_or_null<pylir::Py::ObjectTypeInterface>();
        if (lhsType && rhsType && lhsType == rhsType)
        {
            resultTypes.emplace_back(lhsType);
            return pylir::Py::TypeRefineResult::Success;
        }
        auto boolean = inputs[0].dyn_cast_or_null<mlir::BoolAttr>();
        if (!boolean)
        {
            auto joined = pylir::Py::joinTypes(lhsType, rhsType);
            if (!joined)
            {
                return pylir::Py::TypeRefineResult::Failure;
            }
            resultTypes.emplace_back(joined);
            return pylir::Py::TypeRefineResult::Approximate;
        }
        if (boolean.getValue() ? !lhsType : !rhsType)
        {
            return pylir::Py::TypeRefineResult::Failure;
        }
        resultTypes.emplace_back(boolean.getValue() ? lhsType : rhsType);
        return pylir::Py::TypeRefineResult::Success;
    }
};

// select %con, (Op %lhs..., %x, %rhs...), (Op %lhs..., %y, %rhs...) -> Op %lhs..., (select %con, %x, %y), %rhs...
struct ArithSelectTransform : mlir::OpRewritePattern<mlir::arith::SelectOp>
{
    using mlir::OpRewritePattern<mlir::arith::SelectOp>::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(mlir::arith::SelectOp op, mlir::PatternRewriter& rewriter) const override
    {
        auto* lhs = op.getTrueValue().getDefiningOp();
        auto* rhs = op.getFalseValue().getDefiningOp();
        if (!lhs || !rhs || lhs->getAttrDictionary() != rhs->getAttrDictionary() || lhs->getName() != rhs->getName()
            || op.getTrueValue().cast<mlir::OpResult>().getResultNumber()
                   != op.getFalseValue().cast<mlir::OpResult>().getResultNumber()
            || lhs->getResultTypes() != rhs->getResultTypes() || lhs->hasTrait<mlir::OpTrait::IsTerminator>()
            || lhs->getNumRegions() != 0 || rhs->getNumRegions() != 0 || lhs->getNumOperands() != rhs->getNumOperands()
            || !mlir::MemoryEffectOpInterface::hasNoEffect(lhs) || !mlir::MemoryEffectOpInterface::hasNoEffect(rhs))
        {
            return mlir::failure();
        }
        std::optional<std::size_t> differing;
        for (auto [lhsOp, rhsOp] : llvm::zip(lhs->getOpOperands(), rhs->getOpOperands()))
        {
            if (lhsOp.get() == rhsOp.get())
            {
                continue;
            }
            if (differing)
            {
                return mlir::failure();
            }
            differing = lhsOp.getOperandNumber();
        }
        if (!differing)
        {
            rewriter.replaceOp(op, op.getTrueValue());
            return mlir::success();
        }
        if (lhs->getOperand(*differing).getType() != rhs->getOperand(*differing).getType())
        {
            return mlir::failure();
        }

        auto newSelect = rewriter.create<mlir::arith::SelectOp>(
            op.getLoc(), op.getCondition(), lhs->getOperand(*differing), rhs->getOperand(*differing));
        mlir::OperationState state(op.getLoc(), lhs->getName());
        state.addAttributes(lhs->getAttrs());
        state.addTypes(lhs->getResultTypes());
        auto operands = llvm::to_vector(lhs->getOperands());
        operands[*differing] = newSelect;
        state.addOperands(operands);
        auto* newOp = rewriter.create(state);
        rewriter.replaceOp(op, newOp->getResult(op.getTrueValue().cast<mlir::OpResult>().getResultNumber()));
        return mlir::success();
    }
};

pylir::Py::MakeTupleOp prependTupleConst(mlir::OpBuilder& builder, mlir::Location loc, mlir::Value input,
                                         mlir::Attribute attr)
{
    llvm::SmallVector<mlir::Value> arguments{input};
    for (const auto& iter : attr.cast<pylir::Py::TupleAttr>().getValue())
    {
        arguments.emplace_back(builder.create<pylir::Py::ConstantOp>(loc, iter));
    }
    return builder.create<pylir::Py::MakeTupleOp>(loc, input.getType(), arguments, builder.getDenseI32ArrayAttr({}));
}

bool isTypeSlot(llvm::StringRef ref)
{
    static llvm::StringSet<> set = {
#define TYPE_SLOT(x, ...) #x,
#include <pylir/Interfaces/Slots.def>
    };
    return set.contains(ref);
}

pylir::Py::IntCmpKindAttr invertPredicate(pylir::Py::IntCmpKindAttr kind)
{
    switch (kind.getValue())
    {
        case pylir::Py::IntCmpKind::eq:
            return pylir::Py::IntCmpKindAttr::get(kind.getContext(), pylir::Py::IntCmpKind::ne);
        case pylir::Py::IntCmpKind::ne:
            return pylir::Py::IntCmpKindAttr::get(kind.getContext(), pylir::Py::IntCmpKind::eq);
        case pylir::Py::IntCmpKind::lt:
            return pylir::Py::IntCmpKindAttr::get(kind.getContext(), pylir::Py::IntCmpKind::ge);
        case pylir::Py::IntCmpKind::le:
            return pylir::Py::IntCmpKindAttr::get(kind.getContext(), pylir::Py::IntCmpKind::gt);
        case pylir::Py::IntCmpKind::gt:
            return pylir::Py::IntCmpKindAttr::get(kind.getContext(), pylir::Py::IntCmpKind::le);
        case pylir::Py::IntCmpKind::ge:
            return pylir::Py::IntCmpKindAttr::get(kind.getContext(), pylir::Py::IntCmpKind::lt);
    }
    PYLIR_UNREACHABLE;
}

pylir::Py::IntCmpKindAttr reversePredicate(pylir::Py::IntCmpKindAttr kind)
{
    switch (kind.getValue())
    {
        case pylir::Py::IntCmpKind::eq:
            return pylir::Py::IntCmpKindAttr::get(kind.getContext(), pylir::Py::IntCmpKind::eq);
        case pylir::Py::IntCmpKind::ne:
            return pylir::Py::IntCmpKindAttr::get(kind.getContext(), pylir::Py::IntCmpKind::ne);
        case pylir::Py::IntCmpKind::lt:
            return pylir::Py::IntCmpKindAttr::get(kind.getContext(), pylir::Py::IntCmpKind::gt);
        case pylir::Py::IntCmpKind::le:
            return pylir::Py::IntCmpKindAttr::get(kind.getContext(), pylir::Py::IntCmpKind::ge);
        case pylir::Py::IntCmpKind::gt:
            return pylir::Py::IntCmpKindAttr::get(kind.getContext(), pylir::Py::IntCmpKind::lt);
        case pylir::Py::IntCmpKind::ge:
            return pylir::Py::IntCmpKindAttr::get(kind.getContext(), pylir::Py::IntCmpKind::le);
    }
    PYLIR_UNREACHABLE;
}

mlir::LogicalResult resolvesToPattern(mlir::Operation* operation, mlir::Attribute& result, bool constOnly)
{
    if (!mlir::matchPattern(operation->getResult(0), mlir::m_Constant(&result)))
    {
        return mlir::failure();
    }
    result = resolveValue(operation, result, constOnly);
    return mlir::success();
}

#include "pylir/Optimizer/PylirPy/IR/PylirPyPatterns.cpp.inc"
} // namespace

#include "PylirPyDialect.hpp"

void pylir::Py::PylirPyDialect::getCanonicalizationPatterns(::mlir::RewritePatternSet& results) const
{
    populateWithGenerated(results);
    results.insert<NoopBlockArgRemove>(getContext());
    results.insert<PassthroughArgRemove>(getContext());
    results.insert<ArithSelectTransform>(getContext());
}

void pylir::Py::PylirPyDialect::initializeExternalModels()
{
    mlir::arith::SelectOp::attachInterface<ArithSelectTypeRefinable>(*getContext());
}
