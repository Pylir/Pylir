#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/Transforms/DialectConversion.h>

#include <llvm/ADT/TypeSwitch.h>

#include <pylir/Optimizer/PylirPy/IR/PylirPyOps.hpp>
#include <pylir/Optimizer/PylirPy/Util/Builtins.hpp>
#include <pylir/Optimizer/PylirPy/Util/PyBuilder.hpp>
#include <pylir/Optimizer/PylirPy/Util/Util.hpp>

#include "PassDetail.hpp"
#include "Passes.hpp"

namespace
{

struct CallMethodPattern : mlir::OpRewritePattern<pylir::Py::CallMethodOp>
{
    using mlir::OpRewritePattern<pylir::Py::CallMethodOp>::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(pylir::Py::CallMethodOp op, mlir::PatternRewriter& rewriter) const override
    {
        rewriter.replaceOpWithNewOp<mlir::CallOp>(op, pylir::Py::pylirCallIntrinsic, op.getType(), op.getOperands());
        return mlir::success();
    }
};

struct CallMethodExPattern : mlir::OpRewritePattern<pylir::Py::CallMethodExOp>
{
    using mlir::OpRewritePattern<pylir::Py::CallMethodExOp>::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(pylir::Py::CallMethodExOp op, mlir::PatternRewriter& rewriter) const override
    {
        rewriter.replaceOpWithNewOp<pylir::Py::InvokeOp>(
            op, op.getType(), "$pylir_call", llvm::ArrayRef{op.method(), op.args(), op.keywords()},
            op.normalDestOperands(), op.unwindDestOperands(), op.happyPath(), op.exceptionPath());
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
        auto tupleSize = rewriter.create<pylir::Py::TupleLenOp>(loc, rewriter.getIndexType(), tuple);
        auto startConstant = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 0);
        auto conditionBlock = new mlir::Block;
        conditionBlock->addArgument(rewriter.getIndexType(), loc);
        rewriter.create<mlir::BranchOp>(loc, conditionBlock, mlir::ValueRange{startConstant});

        conditionBlock->insertBefore(endBlock);
        rewriter.setInsertionPointToStart(conditionBlock);
        auto isLess = rewriter.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::ult,
                                                           conditionBlock->getArgument(0), tupleSize);
        auto body = new mlir::Block;
        auto unbound = rewriter.create<pylir::Py::ConstantOp>(loc, pylir::Py::UnboundAttr::get(getContext()));
        auto falseConstant = rewriter.create<mlir::arith::ConstantOp>(loc, rewriter.getBoolAttr(false));
        rewriter.create<mlir::CondBranchOp>(loc, isLess, body, endBlock, mlir::ValueRange{unbound, falseConstant});

        body->insertBefore(endBlock);
        rewriter.setInsertionPointToStart(body);
        auto entry = rewriter.create<pylir::Py::TupleGetItemOp>(loc, tuple, conditionBlock->getArgument(0));
        auto entryType = rewriter.create<pylir::Py::TypeOfOp>(loc, entry);
        auto fetch = rewriter.create<pylir::Py::GetSlotOp>(loc, entry, entryType, op.attribute());
        auto failure = rewriter.create<pylir::Py::IsUnboundValueOp>(loc, fetch);
        auto trueConstant = rewriter.create<mlir::arith::ConstantOp>(loc, rewriter.getBoolAttr(true));
        auto notFound = new mlir::Block;
        rewriter.create<mlir::CondBranchOp>(loc, failure, notFound, endBlock, mlir::ValueRange{fetch, trueConstant});

        notFound->insertBefore(endBlock);
        rewriter.setInsertionPointToStart(notFound);
        auto one = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 1);
        auto nextIter = rewriter.create<mlir::arith::AddIOp>(loc, conditionBlock->getArgument(0), one);
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
        auto tupleSize = rewriter.create<pylir::Py::TupleLenOp>(loc, rewriter.getIndexType(), tuple);
        auto startConstant = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 0);
        auto conditionBlock = new mlir::Block;
        conditionBlock->addArgument(rewriter.getIndexType(), loc);
        rewriter.create<mlir::BranchOp>(loc, conditionBlock, mlir::ValueRange{startConstant});

        conditionBlock->insertBefore(endBlock);
        rewriter.setInsertionPointToStart(conditionBlock);
        auto isLess = rewriter.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::ult,
                                                           conditionBlock->getArgument(0), tupleSize);
        auto body = new mlir::Block;
        auto falseConstant = rewriter.create<mlir::arith::ConstantOp>(loc, rewriter.getBoolAttr(false));
        rewriter.create<mlir::CondBranchOp>(loc, isLess, body, endBlock, mlir::ValueRange{falseConstant});

        body->insertBefore(endBlock);
        rewriter.setInsertionPointToStart(body);
        auto entry = rewriter.create<pylir::Py::TupleGetItemOp>(loc, tuple, conditionBlock->getArgument(0));
        auto isType = rewriter.create<pylir::Py::IsOp>(loc, entry, op.element());
        auto notFound = new mlir::Block;
        auto trueConstant = rewriter.create<mlir::arith::ConstantOp>(loc, rewriter.getBoolAttr(true));
        rewriter.create<mlir::CondBranchOp>(loc, isType, endBlock, mlir::ValueRange{trueConstant}, notFound,
                                            mlir::ValueRange{});

        notFound->insertBefore(endBlock);
        rewriter.setInsertionPointToStart(notFound);
        auto one = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 1);
        auto nextIter = rewriter.create<mlir::arith::AddIOp>(loc, conditionBlock->getArgument(0), one);
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
        mlir::Block* landingPadBlock = nullptr;
        mlir::Block* exceptionHandlerBlock = nullptr;
        if constexpr (hasExceptions)
        {
            landingPadBlock = op.exceptionPath();
            PYLIR_ASSERT(landingPadBlock->getNumSuccessors() == 1);
            exceptionHandlerBlock = landingPadBlock->getSuccessor(0);
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
            auto iterObject = pylir::Py::buildSpecialMethodCall(loc, rewriter, "__iter__", {iter.value()}, {},
                                                                exceptionHandlerBlock, landingPadBlock);
            auto condition = new mlir::Block;
            rewriter.create<mlir::BranchOp>(loc, condition);

            condition->insertBefore(dest);
            rewriter.setInsertionPointToStart(condition);
            auto stopIterationHandler = new mlir::Block;
            stopIterationHandler->addArgument(rewriter.getType<pylir::Py::DynamicType>(), loc);
            auto landingPad = new mlir::Block;
            auto next = pylir::Py::buildSpecialMethodCall(loc, rewriter, "__next__", iterObject, {},
                                                          exceptionHandlerBlock, landingPad);

            rewriter.create<InsertOp>(loc, list, next);
            rewriter.create<mlir::BranchOp>(loc, condition);

            landingPad->insertBefore(dest);
            rewriter.setInsertionPointToStart(landingPad);
            mlir::Value exceptionObject = rewriter.create<pylir::Py::LandingPadOp>(
                loc, mlir::FlatSymbolRefAttr::get(this->getContext(), pylir::Py::Builtins::StopIteration.name));
            rewriter.create<mlir::BranchOp>(loc, stopIterationHandler, exceptionObject);
            stopIterationHandler->insertBefore(dest);
            rewriter.setInsertionPointToStart(stopIterationHandler);
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
    void runOnOperation() override;
};

void ExpandPyDialectPass::runOnOperation()
{
    auto module = getOperation();
    {
        pylir::Py::PyBuilder builder(&getContext());
        builder.setInsertionPointToEnd(module.getBody());
        auto func = builder.create<mlir::FuncOp>(
            pylir::Py::pylirCallIntrinsic,
            builder.getFunctionType({builder.getDynamicType(), builder.getDynamicType(), builder.getDynamicType()},
                                    {builder.getDynamicType()}));
        func.setPrivate();
        builder.setInsertionPointToStart(func.addEntryBlock());

        auto self = func.getArgument(0);
        auto args = func.getArgument(1);
        auto kws = func.getArgument(2);

        auto* condition = new mlir::Block;
        condition->addArgument(builder.getDynamicType(), builder.getCurrentLoc());
        builder.create<mlir::BranchOp>(condition, self);

        func.push_back(condition);
        builder.setInsertionPointToStart(condition);
        self = condition->getArgument(0);
        auto selfType = builder.createTypeOf(self);
        auto mroTuple = builder.createGetSlot(selfType, builder.createTypeRef(), "__mro__");
        auto lookup = builder.createMROLookup(mroTuple, "__call__");
        auto* exitBlock = new mlir::Block;
        exitBlock->addArgument(builder.getDynamicType(), builder.getCurrentLoc());
        auto unbound = builder.createConstant(builder.getUnboundAttr());
        auto* body = new mlir::Block;
        builder.create<mlir::CondBranchOp>(lookup.success(), body, exitBlock, mlir::ValueRange{unbound});

        func.push_back(body);
        builder.setInsertionPointToStart(body);
        auto callableType = builder.createTypeOf(lookup.result());
        auto isFunction = builder.createIs(callableType, builder.createFunctionRef());
        auto* isFunctionBlock = new mlir::Block;
        auto* notFunctionBlock = new mlir::Block;
        builder.create<mlir::CondBranchOp>(isFunction, isFunctionBlock, notFunctionBlock);

        func.push_back(isFunctionBlock);
        builder.setInsertionPointToStart(isFunctionBlock);
        auto fp = builder.createFunctionGetFunction(lookup.result());
        mlir::Value result = builder
                                 .create<mlir::CallIndirectOp>(
                                     fp, mlir::ValueRange{lookup.result(), builder.createTuplePrepend(self, args), kws})
                                 .getResult(0);
        builder.create<mlir::ReturnOp>(result);

        func.push_back(notFunctionBlock);
        builder.setInsertionPointToStart(notFunctionBlock);
        mroTuple = builder.createGetSlot(callableType, builder.createTypeRef(), "__mro__");
        auto getMethod = builder.createMROLookup(mroTuple, "__get__");
        auto* isDescriptor = new mlir::Block;
        builder.create<mlir::CondBranchOp>(getMethod.success(), isDescriptor, condition, lookup.result());

        func.push_back(isDescriptor);
        builder.setInsertionPointToStart(isDescriptor);
        selfType = builder.createTypeOf(self);
        auto tuple = builder.createMakeTuple({self, selfType});
        auto emptyDict = builder.createConstant(builder.getDictAttr());
        result =
            builder.create<mlir::CallOp>(func, mlir::ValueRange{getMethod.result(), tuple, emptyDict}).getResult(0);
        // TODO: check result is not unbound
        builder.create<mlir::BranchOp>(condition, result);

        func.push_back(exitBlock);
        builder.setInsertionPointToStart(exitBlock);
        builder.create<mlir::ReturnOp>(exitBlock->getArgument(0));
    }

    mlir::ConversionTarget target(getContext());
    target.addLegalDialect<pylir::Py::PylirPyDialect, mlir::StandardOpsDialect, mlir::arith::ArithmeticDialect>();
    target.addLegalOp<mlir::FuncOp>();
    target.addDynamicallyLegalOp<pylir::Py::MakeTupleOp, pylir::Py::MakeListOp, pylir::Py::MakeSetOp,
                                 pylir::Py::MakeDictOp>(
        [](mlir::Operation* op) -> bool
        {
            return llvm::TypeSwitch<mlir::Operation*, bool>(op)
                .Case([](pylir::Py::MakeDictOp op) { return op.mappingExpansion().empty(); })
                .Case<pylir::Py::MakeTupleOp, pylir::Py::MakeListOp, pylir::Py::MakeSetOp>(
                    [](auto op) { return op.iterExpansion().empty(); })
                .Default(false);
        });
    target.addIllegalOp<pylir::Py::MakeClassOp, pylir::Py::LinearContainsOp, pylir::Py::CallMethodOp,
                        pylir::Py::CallMethodExOp, pylir::Py::MROLookupOp, pylir::Py::MakeTupleExOp,
                        pylir::Py::MakeListExOp, pylir::Py::MakeSetExOp, pylir::Py::MakeDictExOp>();

    mlir::RewritePatternSet patterns(&getContext());
    patterns.add<LinearContainsPattern>(&getContext());
    patterns.add<MROLookupPattern>(&getContext());
    patterns.add<CallMethodPattern>(&getContext());
    patterns.add<CallMethodExPattern>(&getContext());
    patterns.add<TupleUnrollPattern>(&getContext());
    patterns.add<TupleExUnrollPattern>(&getContext());
    patterns.add<SequenceUnrollPattern<pylir::Py::MakeListOp, pylir::Py::ListAppendOp>>(&getContext());
    patterns.add<SequenceUnrollPattern<pylir::Py::MakeListExOp, pylir::Py::ListAppendOp, pylir::Py::MakeListOp>>(
        &getContext());
    if (mlir::failed(mlir::applyPartialConversion(module, target, std::move(patterns))))
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
