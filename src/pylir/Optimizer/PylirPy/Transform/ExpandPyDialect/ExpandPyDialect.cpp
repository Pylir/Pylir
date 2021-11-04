#include "ExpandPyDialect.hpp"

#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/Transforms/DialectConversion.h>

#include <llvm/ADT/TypeSwitch.h>

#include <pylir/Optimizer/PylirPy/IR/PylirPyOps.hpp>
#include <pylir/Optimizer/PylirPy/Transform/PassDetail.hpp>
#include <pylir/Optimizer/PylirPy/Util/Builtins.hpp>
#include <pylir/Optimizer/PylirPy/Util/Util.hpp>

namespace
{

struct GetFunctionPattern : mlir::OpRewritePattern<pylir::Py::GetFunctionOp>
{
    using mlir::OpRewritePattern<pylir::Py::GetFunctionOp>::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(pylir::Py::GetFunctionOp op, mlir::PatternRewriter& rewriter) const override
    {
        auto loc = op.getLoc();
        auto block = op->getBlock();
        auto endBlock = block->splitBlock(op);
        endBlock->addArgument(rewriter.getType<pylir::Py::DynamicType>(), loc);
        endBlock->addArgument(rewriter.getI1Type(), loc);

        rewriter.setInsertionPointToEnd(block);
        auto func = rewriter.create<pylir::Py::GetGlobalValueOp>(loc, pylir::Py::Builtins::Function.name);
        auto condition = new mlir::Block;
        condition->addArgument(rewriter.getType<pylir::Py::DynamicType>(), loc);
        rewriter.create<mlir::BranchOp>(loc, condition, mlir::ValueRange{op.callable()});

        condition->insertBefore(endBlock);
        rewriter.setInsertionPointToStart(condition);
        auto type = rewriter.create<pylir::Py::TypeOfOp>(loc, condition->getArgument(0));
        auto isFunction = rewriter.create<pylir::Py::IsOp>(loc, type, func);
        auto trueConstant = rewriter.create<mlir::ConstantOp>(loc, rewriter.getBoolAttr(true));
        auto body = new mlir::Block;
        rewriter.create<mlir::CondBranchOp>(loc, isFunction, endBlock,
                                            mlir::ValueRange{condition->getArgument(0), trueConstant}, body,
                                            mlir::ValueRange{});

        body->insertBefore(endBlock);
        rewriter.setInsertionPointToStart(body);
        auto mroTuple = rewriter.create<pylir::Py::GetAttrOp>(loc, type, "__mro__").result();
        auto lookup = rewriter.create<pylir::Py::MROLookupOp>(loc, mroTuple, "__call__");
        auto unboundValue = rewriter.create<pylir::Py::ConstantOp>(loc, pylir::Py::UnboundAttr::get(getContext()));
        auto falseConstant = rewriter.create<mlir::ConstantOp>(loc, rewriter.getBoolAttr(false));
        rewriter.create<mlir::CondBranchOp>(loc, lookup.success(), condition, mlir::ValueRange{lookup.result()},
                                            endBlock, mlir::ValueRange{unboundValue, falseConstant});

        rewriter.replaceOp(op, endBlock->getArguments());
        return mlir::success();
    }
};

struct MROLookupPattern : mlir::OpRewritePattern<pylir::Py::MROLookupOp>
{
    using mlir::OpRewritePattern<pylir::Py::MROLookupOp>::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(pylir::Py::MROLookupOp op, mlir::PatternRewriter& rewriter) const override
    {
        auto loc = op.getLoc();
        auto tuple = op.mroTuple();
        auto block = op->getBlock();
        auto endBlock = block->splitBlock(op);
        endBlock->addArgument(rewriter.getType<pylir::Py::DynamicType>(), loc);
        endBlock->addArgument(rewriter.getI1Type(), loc);

        rewriter.setInsertionPointToEnd(block);
        auto tupleSize = rewriter.create<pylir::Py::TupleIntegerLenOp>(loc, rewriter.getIndexType(), tuple);
        auto startConstant = rewriter.create<mlir::ConstantOp>(loc, rewriter.getIndexAttr(0));
        auto conditionBlock = new mlir::Block;
        conditionBlock->addArgument(rewriter.getIndexType());
        rewriter.create<mlir::BranchOp>(loc, conditionBlock, mlir::ValueRange{startConstant});

        conditionBlock->insertBefore(endBlock);
        rewriter.setInsertionPointToStart(conditionBlock);
        auto isLess =
            rewriter.create<mlir::CmpIOp>(loc, mlir::CmpIPredicate::ult, conditionBlock->getArgument(0), tupleSize);
        auto body = new mlir::Block;
        auto unbound = rewriter.create<pylir::Py::ConstantOp>(loc, pylir::Py::UnboundAttr::get(getContext()));
        auto falseConstant = rewriter.create<mlir::ConstantOp>(loc, rewriter.getBoolAttr(false));
        rewriter.create<mlir::CondBranchOp>(loc, isLess, body, endBlock, mlir::ValueRange{unbound, falseConstant});

        body->insertBefore(endBlock);
        rewriter.setInsertionPointToStart(body);
        auto entry = rewriter.create<pylir::Py::TupleIntegerGetItemOp>(loc, tuple, conditionBlock->getArgument(0));
        auto fetch = rewriter.create<pylir::Py::GetAttrOp>(loc, entry, op.attribute());
        auto notFound = new mlir::Block;
        rewriter.create<mlir::CondBranchOp>(loc, fetch.success(), endBlock, fetch->getResults(), notFound,
                                            mlir::ValueRange{});

        notFound->insertBefore(endBlock);
        rewriter.setInsertionPointToStart(notFound);
        auto one = rewriter.create<mlir::ConstantOp>(loc, rewriter.getIndexAttr(1));
        auto nextIter = rewriter.create<mlir::AddIOp>(loc, conditionBlock->getArgument(0), one);
        rewriter.create<mlir::BranchOp>(loc, conditionBlock, mlir::ValueRange{nextIter});

        rewriter.replaceOp(op, endBlock->getArguments());
        return mlir::success();
    }
};

struct LinearContainsPattern : mlir::OpRewritePattern<pylir::Py::LinearContainsOp>
{
    using mlir::OpRewritePattern<pylir::Py::LinearContainsOp>::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(pylir::Py::LinearContainsOp op, mlir::PatternRewriter& rewriter) const override
    {
        auto loc = op.getLoc();
        auto tuple = op.mroTuple();
        auto block = op->getBlock();
        auto endBlock = block->splitBlock(op);
        endBlock->addArgument(rewriter.getI1Type(), loc);
        rewriter.setInsertionPointToEnd(block);
        auto tupleSize = rewriter.create<pylir::Py::TupleIntegerLenOp>(loc, rewriter.getIndexType(), tuple);
        auto startConstant = rewriter.create<mlir::ConstantOp>(loc, rewriter.getIndexAttr(0));
        auto conditionBlock = new mlir::Block;
        conditionBlock->addArgument(rewriter.getIndexType());
        rewriter.create<mlir::BranchOp>(loc, conditionBlock, mlir::ValueRange{startConstant});

        conditionBlock->insertBefore(endBlock);
        rewriter.setInsertionPointToStart(conditionBlock);
        auto isLess =
            rewriter.create<mlir::CmpIOp>(loc, mlir::CmpIPredicate::ult, conditionBlock->getArgument(0), tupleSize);
        auto body = new mlir::Block;
        auto falseConstant = rewriter.create<mlir::ConstantOp>(loc, rewriter.getBoolAttr(false));
        rewriter.create<mlir::CondBranchOp>(loc, isLess, body, endBlock, mlir::ValueRange{falseConstant});

        body->insertBefore(endBlock);
        rewriter.setInsertionPointToStart(body);
        auto entry = rewriter.create<pylir::Py::TupleIntegerGetItemOp>(loc, tuple, conditionBlock->getArgument(0));
        auto isType = rewriter.create<pylir::Py::IsOp>(loc, entry, op.element());
        auto notFound = new mlir::Block;
        auto trueConstant = rewriter.create<mlir::ConstantOp>(loc, rewriter.getBoolAttr(true));
        rewriter.create<mlir::CondBranchOp>(loc, isType, endBlock, mlir::ValueRange{trueConstant}, notFound,
                                            mlir::ValueRange{});

        notFound->insertBefore(endBlock);
        rewriter.setInsertionPointToStart(notFound);
        auto one = rewriter.create<mlir::ConstantOp>(loc, rewriter.getIndexAttr(1));
        auto nextIter = rewriter.create<mlir::AddIOp>(loc, conditionBlock->getArgument(0), one);
        rewriter.create<mlir::BranchOp>(loc, conditionBlock, mlir::ValueRange{nextIter});

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
        auto list = rewriter.create<pylir::Py::MakeListOp>(op.getLoc(), op.arguments(), op.iterExpansion());
        rewriter.replaceOpWithNewOp<pylir::Py::ListToTupleOp>(op, list);
    }

    mlir::LogicalResult match(pylir::Py::MakeTupleOp op) const override
    {
        return mlir::success(!op.iterExpansion().empty());
    }
};

struct TupleExUnrollPattern : mlir::OpRewritePattern<pylir::Py::MakeTupleExOp>
{
    using mlir::OpRewritePattern<pylir::Py::MakeTupleExOp>::OpRewritePattern;

    void rewrite(pylir::Py::MakeTupleExOp op, mlir::PatternRewriter& rewriter) const override
    {
        rewriter.setInsertionPoint(op);
        auto list = rewriter.create<pylir::Py::MakeListExOp>(op.getLoc(), op.arguments(), op.iterExpansion(),
                                                             op.normalDestOperands(), op.unwindDestOperands(),
                                                             op.happyPath(), op.exceptionPath());
        rewriter.replaceOpWithNewOp<pylir::Py::ListToTupleOp>(op, list);
    }

    mlir::LogicalResult match(pylir::Py::MakeTupleExOp op) const override
    {
        return mlir::success(!op.iterExpansion().empty());
    }
};

template <class TargetOp, class InsertOp, class NormalMakeOp = TargetOp>
struct SequenceUnrollPattern : mlir::OpRewritePattern<TargetOp>
{
    using mlir::OpRewritePattern<TargetOp>::OpRewritePattern;

    constexpr static bool hasExceptions = std::disjunction_v<std::is_same<TargetOp, pylir::Py::MakeListExOp>,
                                                             std::is_same<TargetOp, pylir::Py::MakeSetExOp>>;

    void rewrite(TargetOp op, mlir::PatternRewriter& rewriter) const override
    {
        mlir::Block* exceptionPath = nullptr;
        if constexpr (hasExceptions)
        {
            exceptionPath = op.exceptionPath();
        }
        auto block = op->getBlock();
        auto dest = block->splitBlock(op);
        rewriter.setInsertionPointToEnd(block);
        auto loc = op.getLoc();
        auto range = op.iterExpansion().template getAsRange<mlir::IntegerAttr>();
        PYLIR_ASSERT(!range.empty());
        auto begin = range.begin();
        auto prefix = op.getOperands().take_front((*begin).getValue().getZExtValue());
        auto list = rewriter.create<NormalMakeOp>(loc, prefix, rewriter.getI32ArrayAttr({}));
        for (auto iter : llvm::drop_begin(llvm::enumerate(op.getOperands()), (*begin).getValue().getZExtValue()))
        {
            if (begin == range.end() || (*begin).getValue() != iter.index())
            {
                rewriter.create<InsertOp>(loc, list, iter.value());
                continue;
            }
            begin++;
            auto type = rewriter.create<pylir::Py::TypeOfOp>(loc, iter.value());
            auto iterObject = pylir::Py::buildSpecialMethodCall(
                loc, rewriter, "__iter__", type,
                rewriter.create<pylir::Py::MakeTupleOp>(loc, std::vector<pylir::Py::IterArg>{iter.value()}),
                rewriter.create<pylir::Py::ConstantOp>(loc, pylir::Py::DictAttr::get(this->getContext(), {})),
                exceptionPath);

            auto typeMRO = rewriter.create<pylir::Py::GetAttrOp>(loc, type, "__mro__").result();
            auto nextMethod = rewriter.create<pylir::Py::MROLookupOp>(loc, typeMRO, "__next__");
            auto notNextBlock = new mlir::Block;
            auto condition = new mlir::Block;
            rewriter.create<mlir::CondBranchOp>(loc, nextMethod.success(), condition, notNextBlock);

            {
                notNextBlock->insertBefore(dest);
                rewriter.setInsertionPointToStart(notNextBlock);
                auto exception =
                    pylir::Py::buildException(loc, rewriter, pylir::Py::Builtins::TypeError.name, {}, exceptionPath);
                if (exceptionPath)
                {
                    rewriter.create<mlir::BranchOp>(loc, exceptionPath, exception);
                }
                else
                {
                    rewriter.create<pylir::Py::RaiseOp>(loc, exception);
                }
            }

            condition->insertBefore(dest);
            rewriter.setInsertionPointToStart(condition);
            auto exceptionHandler = new mlir::Block;
            exceptionHandler->addArgument(rewriter.getType<pylir::Py::DynamicType>());
            auto next = pylir::Py::buildCall(
                loc, rewriter, nextMethod.result(),
                rewriter.create<pylir::Py::MakeTupleOp>(loc, std::vector<pylir::Py::IterArg>{iterObject}),
                rewriter.create<pylir::Py::ConstantOp>(loc, pylir::Py::DictAttr::get(this->getContext(), {})),
                exceptionHandler);

            rewriter.create<InsertOp>(loc, list, next);
            rewriter.create<mlir::BranchOp>(loc, condition);

            exceptionHandler->insertBefore(dest);
            rewriter.setInsertionPointToStart(exceptionHandler);
            auto exception = exceptionHandler->getArgument(0);
            auto exceptionType = rewriter.create<pylir::Py::TypeOfOp>(loc, exception);
            auto stopIteration =
                rewriter.create<pylir::Py::GetGlobalValueOp>(loc, pylir::Py::Builtins::StopIteration.name);
            auto mro = rewriter.create<pylir::Py::GetAttrOp>(loc, exceptionType, "__mro__").result();
            auto isStopIteration = rewriter.create<pylir::Py::LinearContainsOp>(loc, mro, stopIteration);
            auto reraiseBlock = new mlir::Block;
            auto exitBlock = new mlir::Block;
            rewriter.create<mlir::CondBranchOp>(loc, isStopIteration, exitBlock, reraiseBlock);

            reraiseBlock->insertBefore(dest);
            rewriter.setInsertionPointToStart(reraiseBlock);
            if (exceptionPath)
            {
                rewriter.create<mlir::BranchOp>(loc, exceptionPath, exception);
            }
            else
            {
                rewriter.create<pylir::Py::RaiseOp>(loc, exception);
            }

            exitBlock->insertBefore(dest);
            rewriter.setInsertionPointToStart(exitBlock);
        }
        rewriter.mergeBlocks(dest, rewriter.getBlock());

        if constexpr (hasExceptions)
        {
            rewriter.setInsertionPointAfter(op);
            mlir::Block* happyPath = op.happyPath();
            if (!happyPath->getSinglePredecessor())
            {
                rewriter.template create<mlir::BranchOp>(loc, happyPath);
            }
            else
            {
                rewriter.mergeBlocks(happyPath, op->getBlock(), op.normalDestOperands());
            }
        }
        rewriter.replaceOp(op, {list});
    }

    mlir::LogicalResult match(TargetOp op) const override
    {
        return mlir::success(!op.iterExpansion().empty());
    }
};

struct ExpandPyDialectPass : public pylir::Py::ExpandPyDialectBase<ExpandPyDialectPass>
{
    void runOnFunction() override;
};

void ExpandPyDialectPass::runOnFunction()
{
    mlir::ConversionTarget target(getContext());
    target.addLegalDialect<pylir::Py::PylirPyDialect, mlir::StandardOpsDialect>();
    target.addDynamicallyLegalOp<pylir::Py::MakeTupleOp, pylir::Py::MakeTupleExOp, pylir::Py::MakeListOp,
                                 pylir::Py::MakeListExOp, pylir::Py::MakeSetOp, pylir::Py::MakeSetExOp,
                                 pylir::Py::MakeDictOp, pylir::Py::MakeDictExOp>(
        [](mlir::Operation* op) -> bool
        {
            return llvm::TypeSwitch<mlir::Operation*, bool>(op)
                .Case<pylir::Py::MakeDictOp, pylir::Py::MakeDictExOp>([](auto op)
                                                                      { return op.mappingExpansion().empty(); })
                .Case<pylir::Py::MakeTupleOp, pylir::Py::MakeTupleExOp, pylir::Py::MakeListOp, pylir::Py::MakeListExOp,
                      pylir::Py::MakeSetOp, pylir::Py::MakeSetExOp>([](auto op) { return op.iterExpansion().empty(); })
                .Default(false);
        });
    target.addIllegalOp<pylir::Py::MakeClassOp, pylir::Py::LinearContainsOp, pylir::Py::GetFunctionOp,
                        pylir::Py::MROLookupOp>();

    mlir::RewritePatternSet patterns(&getContext());
    patterns.add<LinearContainsPattern>(&getContext());
    patterns.add<MROLookupPattern>(&getContext());
    patterns.add<GetFunctionPattern>(&getContext());
    patterns.add<TupleUnrollPattern>(&getContext());
    patterns.add<TupleExUnrollPattern>(&getContext());
    patterns.add<SequenceUnrollPattern<pylir::Py::MakeListOp, pylir::Py::ListAppendOp>>(&getContext());
    patterns.add<SequenceUnrollPattern<pylir::Py::MakeListExOp, pylir::Py::ListAppendOp, pylir::Py::MakeListOp>>(
        &getContext());
    if (mlir::failed(mlir::applyPartialConversion(getFunction(), target, std::move(patterns))))
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
