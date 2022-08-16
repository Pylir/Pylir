// Copyright 2022 Markus Böck
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <mlir/Dialect/ControlFlow/IR/ControlFlowOps.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Transforms/DialectConversion.h>

#include <llvm/ADT/TypeSwitch.h>

#include <pylir/Interfaces/Builtins.hpp>
#include <pylir/Optimizer/PylirPy/IR/PylirPyOps.hpp>

#include "PassDetail.hpp"
#include "Passes.hpp"

namespace
{

mlir::Value callOrInvoke(mlir::Location loc, mlir::OpBuilder& builder, mlir::Block* dest,
                         const pylir::Builtins::Builtin& callable, llvm::ArrayRef<pylir::Py::IterArg> args,
                         mlir::Block* exceptionHandlerBlock)
{
    mlir::Value result;
    if (exceptionHandlerBlock)
    {
        auto* happyPath = new mlir::Block;
        result = builder
                     .create<pylir::Py::InvokeOp>(
                         loc, builder.getType<pylir::Py::DynamicType>(), pylir::Builtins::PylirCall.name,
                         mlir::ValueRange{
                             builder.create<pylir::Py::ConstantOp>(
                                 loc, mlir::FlatSymbolRefAttr::get(builder.getContext(), callable.name)),
                             builder.create<pylir::Py::MakeTupleOp>(loc, args),
                             builder.create<pylir::Py::ConstantOp>(loc, pylir::Py::DictAttr::get(builder.getContext())),
                         },
                         mlir::ValueRange{}, mlir::ValueRange{}, happyPath, exceptionHandlerBlock)
                     .getResult(0);
        happyPath->insertBefore(dest);
        builder.setInsertionPointToStart(happyPath);
    }
    else
    {
        result = builder
                     .create<pylir::Py::CallOp>(
                         loc, builder.getType<pylir::Py::DynamicType>(), pylir::Builtins::PylirCall.name,
                         mlir::ValueRange{
                             builder.create<pylir::Py::ConstantOp>(
                                 loc, mlir::FlatSymbolRefAttr::get(builder.getContext(), callable.name)),
                             builder.create<pylir::Py::MakeTupleOp>(loc, args),
                             builder.create<pylir::Py::ConstantOp>(loc, pylir::Py::DictAttr::get(builder.getContext())),
                         })
                     .getResult(0);
    }
    return result;
}

struct MROLookupPattern : mlir::OpRewritePattern<pylir::Py::MROLookupOp>
{
    using mlir::OpRewritePattern<pylir::Py::MROLookupOp>::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(pylir::Py::MROLookupOp op, mlir::PatternRewriter& rewriter) const override
    {
        auto loc = op.getLoc();
        auto tuple = op.getMroTuple();
        auto* block = op->getBlock();
        auto* endBlock = block->splitBlock(op);
        endBlock->addArguments(op->getResultTypes(), llvm::SmallVector(op->getNumResults(), loc));

        rewriter.setInsertionPointToEnd(block);
        auto tupleSize = rewriter.create<pylir::Py::TupleLenOp>(loc, rewriter.getIndexType(), tuple);
        auto startConstant = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 0);
        auto* conditionBlock = new mlir::Block;
        conditionBlock->addArgument(rewriter.getIndexType(), loc);
        rewriter.create<mlir::cf::BranchOp>(loc, conditionBlock, mlir::ValueRange{startConstant});

        conditionBlock->insertBefore(endBlock);
        rewriter.setInsertionPointToStart(conditionBlock);
        auto isLess = rewriter.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::ult,
                                                           conditionBlock->getArgument(0), tupleSize);
        auto* body = new mlir::Block;
        auto unbound = rewriter.create<pylir::Py::ConstantOp>(loc, pylir::Py::UnboundAttr::get(getContext()));
        auto falseConstant = rewriter.create<mlir::arith::ConstantOp>(loc, rewriter.getBoolAttr(false));
        rewriter.create<mlir::cf::CondBranchOp>(loc, isLess, body, endBlock, mlir::ValueRange{unbound, falseConstant});

        body->insertBefore(endBlock);
        rewriter.setInsertionPointToStart(body);
        auto entry = rewriter.create<pylir::Py::TupleGetItemOp>(loc, tuple, conditionBlock->getArgument(0));
        auto entryType = rewriter.create<pylir::Py::TypeOfOp>(loc, entry);
        auto fetch = rewriter.create<pylir::Py::GetSlotOp>(loc, entry, entryType, op.getSlotAttr());
        auto failure = rewriter.create<pylir::Py::IsUnboundValueOp>(loc, fetch);
        auto trueConstant = rewriter.create<mlir::arith::ConstantOp>(loc, rewriter.getBoolAttr(true));
        auto* notFound = new mlir::Block;
        rewriter.create<mlir::cf::CondBranchOp>(loc, failure, notFound, endBlock,
                                                mlir::ValueRange{fetch, trueConstant});

        notFound->insertBefore(endBlock);
        rewriter.setInsertionPointToStart(notFound);
        auto one = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 1);
        auto nextIter = rewriter.create<mlir::arith::AddIOp>(loc, conditionBlock->getArgument(0), one);
        rewriter.create<mlir::cf::BranchOp>(loc, conditionBlock, mlir::ValueRange{nextIter});

        rewriter.replaceOp(op, endBlock->getArguments());
        return mlir::success();
    }
};

struct TupleUnrollPattern : mlir::OpRewritePattern<pylir::Py::MakeTupleOp>
{
    using mlir::OpRewritePattern<pylir::Py::MakeTupleOp>::OpRewritePattern;

    void rewrite(pylir::Py::MakeTupleOp op, mlir::PatternRewriter& rewriter) const override
    {
        rewriter.setInsertionPoint(op);
        auto list = rewriter.create<pylir::Py::MakeListOp>(op.getLoc(), op.getArguments(), op.getIterExpansion());
        rewriter.replaceOpWithNewOp<pylir::Py::ListToTupleOp>(op, list);
    }

    mlir::LogicalResult match(pylir::Py::MakeTupleOp op) const override
    {
        return mlir::success(!op.getIterExpansion().empty());
    }
};

struct TupleExUnrollPattern : mlir::OpRewritePattern<pylir::Py::MakeTupleExOp>
{
    using mlir::OpRewritePattern<pylir::Py::MakeTupleExOp>::OpRewritePattern;

    void rewrite(pylir::Py::MakeTupleExOp op, mlir::PatternRewriter& rewriter) const override
    {
        rewriter.setInsertionPoint(op);
        auto list = rewriter.create<pylir::Py::MakeListExOp>(op.getLoc(), op.getArguments(), op.getIterExpansion(),
                                                             op.getNormalDestOperands(), op.getUnwindDestOperands(),
                                                             op.getHappyPath(), op.getExceptionPath());
        rewriter.replaceOpWithNewOp<pylir::Py::ListToTupleOp>(op, list);
    }

    mlir::LogicalResult match(pylir::Py::MakeTupleExOp op) const override
    {
        return mlir::success(!op.getIterExpansion().empty());
    }
};

template <class TargetOp>
struct ListUnrollPattern : mlir::OpRewritePattern<TargetOp>
{
    using mlir::OpRewritePattern<TargetOp>::OpRewritePattern;

    void rewrite(TargetOp op, mlir::PatternRewriter& rewriter) const override
    {
        mlir::Block* exceptionHandlerBlock = nullptr;
        auto exceptionHandler = mlir::dyn_cast<pylir::Py::ExceptionHandlingInterface>(*op);
        if (exceptionHandler)
        {
            exceptionHandlerBlock = exceptionHandler.getExceptionPath();
        }
        auto block = op->getBlock();
        auto dest = block->splitBlock(op);
        rewriter.setInsertionPointToEnd(block);
        auto loc = op.getLoc();
        auto range = op.getIterExpansion().template getAsRange<mlir::IntegerAttr>();
        PYLIR_ASSERT(!range.empty());
        auto begin = range.begin();
        auto prefix = op.getOperands().take_front((*begin).getValue().getZExtValue());
        auto one = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 1);
        auto list = rewriter.create<pylir::Py::MakeListOp>(loc, prefix, rewriter.getI32ArrayAttr({}));
        for (const auto& iter : llvm::drop_begin(llvm::enumerate(op.getOperands()), (*begin).getValue().getZExtValue()))
        {
            if (begin == range.end() || (*begin).getValue() != iter.index())
            {
                auto len = rewriter.create<pylir::Py::ListLenOp>(loc, list);
                auto newLen = rewriter.create<mlir::arith::AddIOp>(loc, len, one);
                rewriter.create<pylir::Py::ListResizeOp>(loc, list, newLen);
                rewriter.create<pylir::Py::ListSetItemOp>(loc, list, len, iter.value());
                continue;
            }
            begin++;
            auto iterObject =
                callOrInvoke(loc, rewriter, dest, pylir::Builtins::Iter, {iter.value()}, exceptionHandlerBlock);
            auto* condition = new mlir::Block;
            rewriter.create<mlir::cf::BranchOp>(loc, condition);

            condition->insertBefore(dest);
            rewriter.setInsertionPointToStart(condition);
            auto* stopIterationHandler = new mlir::Block;
            stopIterationHandler->addArgument(rewriter.getType<pylir::Py::DynamicType>(), loc);

            auto next = callOrInvoke(loc, rewriter, dest, pylir::Builtins::Next, {iterObject}, stopIterationHandler);

            auto len = rewriter.create<pylir::Py::ListLenOp>(loc, list);
            auto newLen = rewriter.create<mlir::arith::AddIOp>(loc, len, one);
            rewriter.create<pylir::Py::ListResizeOp>(loc, list, newLen);
            rewriter.create<pylir::Py::ListSetItemOp>(loc, list, len, next);
            rewriter.create<mlir::cf::BranchOp>(loc, condition);

            stopIterationHandler->insertBefore(dest);
            rewriter.setInsertionPointToStart(stopIterationHandler);
            auto stopIterationType = rewriter.create<pylir::Py::ConstantOp>(
                loc, mlir::FlatSymbolRefAttr::get(this->getContext(), pylir::Builtins::StopIteration.name));
            auto typeOf = rewriter.create<pylir::Py::TypeOfOp>(loc, stopIterationHandler->getArgument(0));
            auto isStopIteration = rewriter.create<pylir::Py::IsOp>(loc, stopIterationType, typeOf);
            auto* continueBlock = new mlir::Block;
            auto* reraiseBlock = new mlir::Block;
            rewriter.create<mlir::cf::CondBranchOp>(loc, isStopIteration, continueBlock, reraiseBlock);

            reraiseBlock->insertBefore(dest);
            rewriter.setInsertionPointToStart(reraiseBlock);
            rewriter.create<pylir::Py::RaiseOp>(loc, stopIterationHandler->getArgument(0));

            continueBlock->insertBefore(dest);
            rewriter.setInsertionPointToStart(continueBlock);
        }
        rewriter.mergeBlocks(dest, rewriter.getBlock());

        if (exceptionHandlerBlock)
        {
            rewriter.setInsertionPointAfter(op);
            mlir::Block* happyPath = exceptionHandler.getHappyPath();
            if (!happyPath->getSinglePredecessor())
            {
                rewriter.template create<mlir::cf::BranchOp>(loc, happyPath);
            }
            else
            {
                rewriter.mergeBlocks(happyPath, op->getBlock(),
                                     static_cast<mlir::OperandRange>(exceptionHandler.getNormalDestOperandsMutable()));
            }
        }
        rewriter.replaceOp(op, {list});
    }

    mlir::LogicalResult match(TargetOp op) const override
    {
        return mlir::success(!op.getIterExpansion().empty());
    }
};

struct ExpandPyDialectPass : public ExpandPyDialectBase<ExpandPyDialectPass>
{
    void runOnOperation() override;
};

void ExpandPyDialectPass::runOnOperation()
{
    mlir::ConversionTarget target(getContext());
    target.addDynamicallyLegalOp<pylir::Py::MakeTupleOp, pylir::Py::MakeListOp, pylir::Py::MakeSetOp,
                                 pylir::Py::MakeDictOp>(
        [](mlir::Operation* op) -> bool
        {
            return llvm::TypeSwitch<mlir::Operation*, bool>(op)
                .Case([](pylir::Py::MakeDictOp op) { return op.getMappingExpansionAttr().empty(); })
                .Case<pylir::Py::MakeTupleOp, pylir::Py::MakeListOp, pylir::Py::MakeSetOp>(
                    [](auto op) { return op.getIterExpansion().empty(); })
                .Default(false);
        });
    target.addIllegalOp<pylir::Py::MROLookupOp, pylir::Py::MakeTupleExOp, pylir::Py::MakeListExOp,
                        pylir::Py::MakeSetExOp, pylir::Py::MakeDictExOp>();
    target.markUnknownOpDynamicallyLegal([](auto...) { return true; });

    mlir::RewritePatternSet patterns(&getContext());
    patterns.add<MROLookupPattern>(&getContext());
    patterns.add<TupleUnrollPattern>(&getContext());
    patterns.add<TupleExUnrollPattern>(&getContext());
    patterns.add<ListUnrollPattern<pylir::Py::MakeListOp>>(&getContext());
    patterns.add<ListUnrollPattern<pylir::Py::MakeListExOp>>(&getContext());
    if (mlir::failed(mlir::applyPartialConversion(getOperation(), target, std::move(patterns))))
    {
        signalPassFailure();
        return;
    }
}
} // namespace

std::unique_ptr<mlir::Pass> pylir::Py::createExpandPyDialectPass()
{
    return std::make_unique<ExpandPyDialectPass>();
}
