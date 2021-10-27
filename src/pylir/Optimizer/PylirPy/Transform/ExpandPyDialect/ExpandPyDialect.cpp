#include "ExpandPyDialect.hpp"

#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/Transforms/DialectConversion.h>

#include <pylir/Optimizer/PylirPy/IR/PylirPyOps.hpp>
#include <pylir/Optimizer/PylirPy/Transform/PassDetail.hpp>

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
        auto func = rewriter.create<pylir::Py::GetGlobalValueOp>(
            loc, /*TODO: find a proper place to put builtins for both mid and frontend*/ "builtins.function");
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
            if (mlir::isa<pylir::Py::MakeDictOp, pylir::Py::MakeDictExOp>(op))
            {
                return op->getAttrOfType<mlir::ArrayAttr>("mappingExpansion").empty();
            }
            return op->getAttrOfType<mlir::ArrayAttr>("iterExpansion").empty();
        });
    target.addIllegalOp<pylir::Py::MakeClassOp, pylir::Py::LinearContainsOp, pylir::Py::GetFunctionOp,
                        pylir::Py::MROLookupOp>();

    mlir::RewritePatternSet patterns(&getContext());
    patterns.add<LinearContainsPattern>(&getContext());
    patterns.add<MROLookupPattern>(&getContext());
    patterns.add<GetFunctionPattern>(&getContext());
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
