#include <mlir/Dialect/ControlFlow/IR/ControlFlowOps.h>
#include <mlir/Transforms/DialectConversion.h>

#include <llvm/ADT/TypeSwitch.h>

#include <pylir/Optimizer/PylirPy/IR/PylirPyOps.hpp>

#include "PassDetail.hpp"
#include "Passes.hpp"

namespace
{

bool addExceptionHandling(mlir::PatternRewriter& rewriter, mlir::Operation* operation,
                          pylir::Py::TypeSwitchExOp typeSwitch)
{
    mlir::OpBuilder::InsertionGuard guard{rewriter};
    return llvm::TypeSwitch<mlir::Operation*, bool>(operation)
        .Case<mlir::CallIndirectOp, mlir::CallOp>(
            [&](auto callOp)
            {
                auto* endBlock = rewriter.splitBlock(callOp->getBlock(), std::next(mlir::Block::iterator{callOp}));
                rewriter.setInsertionPointAfter(callOp);

                if constexpr (std::is_same_v<decltype(callOp), mlir::CallOp>)
                {
                    rewriter.replaceOpWithNewOp<pylir::Py::InvokeOp>(
                        callOp, callOp.getResultTypes(), callOp.getCalleeAttr(), callOp.operands(),
                        typeSwitch.getNormalDestOperands(), typeSwitch.getUnwindDestOperands(), endBlock,
                        typeSwitch.getExceptionPath());
                }
                else
                {
                    rewriter.replaceOpWithNewOp<pylir::Py::InvokeIndirectOp>(
                        callOp, callOp.getResultTypes(), callOp.getCallee(), callOp.operands(),
                        typeSwitch.getNormalDestOperands(), typeSwitch.getUnwindDestOperands(), endBlock,
                        typeSwitch.getExceptionPath());
                }

                return true;
            })
        .Default(false);
}

template <class Op>
struct TypeSwitchOpConversion : public mlir::OpRewritePattern<Op>
{
    using mlir::OpRewritePattern<Op>::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(Op op, mlir::PatternRewriter& rewriter) const override
    {
        constexpr bool exceptionHandling = std::is_same_v<Op, pylir::Py::TypeSwitchExOp>;

        auto* block = op->getBlock();
        auto* endBlock = rewriter.splitBlock(block, mlir::Block::iterator{op});
        endBlock->addArguments(op.getResultTypes(), llvm::SmallVector(op.getNumResults(), op.getLoc()));

        auto changeTerminators = [&](mlir::Region& region)
        {
            for (auto& iter : region)
            {
                if constexpr (exceptionHandling)
                {
                    for (auto& iter2 : iter)
                    {
                        if (addExceptionHandling(rewriter, &iter2, op))
                        {
                            break;
                        }
                    }
                }
                if (auto yield = mlir::dyn_cast<pylir::Py::YieldOp>(iter.getTerminator()))
                {
                    mlir::OpBuilder::InsertionGuard guard{rewriter};
                    rewriter.setInsertionPoint(yield);
                    rewriter.replaceOpWithNewOp<mlir::cf::BranchOp>(yield, yield.getResults(), endBlock);
                }
            }
        };

        rewriter.setInsertionPointToEnd(block);
        auto typeObject = op.getTypeObject();
        for (auto [typeMatch, region] : llvm::zip(op.getSpecializationTypes(), op.getSpecializations()))
        {
            auto isEqual = rewriter.create<pylir::Py::IsOp>(op.getLoc(), typeObject, typeMatch);
            auto* regionBlock = &region.front();
            auto* continueBlock = new mlir::Block;
            rewriter.create<mlir::cf::CondBranchOp>(op.getLoc(), isEqual, regionBlock, continueBlock);

            changeTerminators(region);
            rewriter.inlineRegionBefore(region, endBlock);

            continueBlock->insertBefore(endBlock);
            rewriter.setInsertionPointToStart(continueBlock);
        }

        rewriter.create<mlir::cf::BranchOp>(op.getLoc(), &op.getGeneric().front());
        changeTerminators(op.getGeneric());
        rewriter.inlineRegionBefore(op.getGeneric(), endBlock);

        rewriter.setInsertionPointToStart(endBlock);
        rewriter.replaceOp(op, endBlock->getArguments());
        if constexpr (exceptionHandling)
        {
            rewriter.create<mlir::cf::BranchOp>(op.getLoc(), op.getHappyPath());
        }
        return mlir::success();
    }
};

struct LowerTypeSwitchPass : public pylir::Py::LowerTypeSwitchBase<LowerTypeSwitchPass>
{
protected:
    void runOnOperation() override
    {
        mlir::ConversionTarget target(getContext());
        target.addIllegalOp<pylir::Py::TypeSwitchOp, pylir::Py::TypeSwitchExOp>();
        target.markUnknownOpDynamicallyLegal([](mlir::Operation*) { return true; });

        mlir::RewritePatternSet patterns(&getContext());
        patterns.insert<TypeSwitchOpConversion<pylir::Py::TypeSwitchOp>>(&getContext());
        patterns.insert<TypeSwitchOpConversion<pylir::Py::TypeSwitchExOp>>(&getContext());
        if (mlir::failed(mlir::applyPartialConversion(getOperation(), target, std::move(patterns))))
        {
            signalPassFailure();
            return;
        }
    }
};
} // namespace

std::unique_ptr<mlir::Pass> pylir::Py::createLowerTypeSwitchPass()
{
    return std::make_unique<LowerTypeSwitchPass>();
}
