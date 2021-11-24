#include <pylir/Optimizer/Conversion/PassDetail.hpp>
#include <pylir/Optimizer/PylirMem/IR/PylirMemDialect.hpp>
#include <pylir/Optimizer/PylirPy/IR/PylirPyDialect.hpp>
#include <pylir/Optimizer/PylirPy/IR/PylirPyOps.hpp>

#include <llvm/ADT/DenseMap.h>

#include <mlir/Transforms/DialectConversion.h>

namespace
{

struct ConstantOpConversion : mlir::OpRewritePattern<pylir::Py::ConstantOp>
{
    ConstantOpConversion(mlir::ModuleOp moduleOp)
        : mlir::OpRewritePattern<pylir::Py::ConstantOp>(moduleOp.getContext()), symbolTable(moduleOp)
    {
    }

    mutable llvm::DenseMap<mlir::Attribute, pylir::Py::GlobalValueOp> globalValues;
    mutable mlir::SymbolTable symbolTable;

    void rewrite(pylir::Py::ConstantOp op, mlir::PatternRewriter& rewriter) const override
    {
        if (auto global = globalValues.lookup(op.constant()))
        {
            rewriter.replaceOpWithNewOp<pylir::Py::ConstantOp>(op, mlir::FlatSymbolRefAttr::get(global));
            return;
        }
        mlir::OpBuilder::InsertionGuard guard{rewriter};
        pylir::Py::GlobalValueOp globalValueOp;
        {
            mlir::OpBuilder builder{getContext()};
            mlir::OperationState state(rewriter.getUnknownLoc(), pylir::Py::GlobalValueOp::getOperationName());
            pylir::Py::GlobalValueOp::build(builder, state, "const$", rewriter.getStringAttr("private"), true,
                                            op.constant().cast<pylir::Py::ObjectAttr>());
            globalValueOp = mlir::cast<pylir::Py::GlobalValueOp>(mlir::Operation::create(state));
        }
        symbolTable.insert(globalValueOp, mlir::Block::iterator{op->getParentOfType<mlir::FuncOp>()});
        globalValues.insert({op.constant(), globalValueOp});
        rewriter.replaceOpWithNewOp<pylir::Py::ConstantOp>(op, mlir::FlatSymbolRefAttr::get(globalValueOp));
    }

    mlir::LogicalResult match(pylir::Py::ConstantOp op) const override
    {
        return mlir::success(op.constant().isa<pylir::Py::ObjectAttr>());
    }
};

struct ConvertPylirPyToPylirMem : pylir::ConvertPylirPyToPylirMemBase<ConvertPylirPyToPylirMem>
{
protected:
    void runOnOperation() override;
};

void ConvertPylirPyToPylirMem::runOnOperation()
{
    mlir::ConversionTarget target(getContext());
    target.addLegalDialect<pylir::Py::PylirPyDialect, pylir::Mem::PylirMemDialect, mlir::StandardOpsDialect,
                           mlir::arith::ArithmeticDialect>();

    target.addIllegalOp<pylir::Py::MakeTupleOp, pylir::Py::MakeListOp, pylir::Py::MakeSetOp, pylir::Py::MakeDictOp,
                        pylir::Py::MakeFuncOp, pylir::Py::MakeFuncOp, pylir::Py::ListToTupleOp,
                        pylir::Py::BoolFromI1Op>();
    target.addDynamicallyLegalOp<pylir::Py::ConstantOp>(
        [](pylir::Py::ConstantOp constantOp) -> bool
        { return constantOp.constant().isa<pylir::Py::UnboundAttr, mlir::SymbolRefAttr>(); });

    mlir::RewritePatternSet patterns(&getContext());
    patterns.insert<ConstantOpConversion>(getOperation());
    if (mlir::failed(mlir::applyPartialConversion(getOperation(), target, std::move(patterns))))
    {
        signalPassFailure();
        return;
    }
}
} // namespace

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> pylir::createConvertPylirPyToPylirMemPass()
{
    return std::make_unique<ConvertPylirPyToPylirMem>();
}
