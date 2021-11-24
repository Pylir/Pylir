#include <pylir/Optimizer/Conversion/PassDetail.hpp>
#include <pylir/Optimizer/PylirMem/IR/PylirMemDialect.hpp>
#include <pylir/Optimizer/PylirMem/IR/PylirMemOps.hpp>
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

struct MakeTupleOpConversion : mlir::OpRewritePattern<pylir::Py::MakeTupleOp>
{
    using mlir::OpRewritePattern<pylir::Py::MakeTupleOp>::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(pylir::Py::MakeTupleOp op, mlir::PatternRewriter& rewriter) const override
    {
        auto tuple = rewriter.create<pylir::Py::ConstantOp>(
            op.getLoc(), mlir::FlatSymbolRefAttr::get(getContext(), pylir::Py::Builtins::Tuple.name));
        auto mem = rewriter.create<pylir::Mem::GCAllocObjectOp>(op.getLoc(), tuple);
        rewriter.replaceOpWithNewOp<pylir::Mem::InitTupleOp>(op, mem, op.arguments(), mlir::Value{});
        return mlir::success();
    }
};

struct MakeListOpConversion : mlir::OpRewritePattern<pylir::Py::MakeListOp>
{
    using mlir::OpRewritePattern<pylir::Py::MakeListOp>::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(pylir::Py::MakeListOp op, mlir::PatternRewriter& rewriter) const override
    {
        auto tuple = rewriter.create<pylir::Py::ConstantOp>(
            op.getLoc(), mlir::FlatSymbolRefAttr::get(getContext(), pylir::Py::Builtins::List.name));
        auto mem = rewriter.create<pylir::Mem::GCAllocObjectOp>(op.getLoc(), tuple);
        rewriter.replaceOpWithNewOp<pylir::Mem::InitListOp>(op, mem, op.arguments(), mlir::Value{});
        return mlir::success();
    }
};

struct MakeSetOpConversion : mlir::OpRewritePattern<pylir::Py::MakeSetOp>
{
    using mlir::OpRewritePattern<pylir::Py::MakeSetOp>::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(pylir::Py::MakeSetOp op, mlir::PatternRewriter& rewriter) const override
    {
        auto set = rewriter.create<pylir::Py::ConstantOp>(
            op.getLoc(), mlir::FlatSymbolRefAttr::get(getContext(), pylir::Py::Builtins::Set.name));
        auto mem = rewriter.create<pylir::Mem::GCAllocObjectOp>(op.getLoc(), set);
        rewriter.replaceOpWithNewOp<pylir::Mem::InitSetOp>(op, mem, op.arguments(), mlir::Value{});
        return mlir::success();
    }
};

struct MakeDictOpConversion : mlir::OpRewritePattern<pylir::Py::MakeDictOp>
{
    using mlir::OpRewritePattern<pylir::Py::MakeDictOp>::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(pylir::Py::MakeDictOp op, mlir::PatternRewriter& rewriter) const override
    {
        auto dict = rewriter.create<pylir::Py::ConstantOp>(
            op.getLoc(), mlir::FlatSymbolRefAttr::get(getContext(), pylir::Py::Builtins::Dict.name));
        auto mem = rewriter.create<pylir::Mem::GCAllocObjectOp>(op.getLoc(), dict);
        auto init = rewriter.replaceOpWithNewOp<pylir::Mem::InitDictOp>(op, mem);
        for (auto [key, value] : llvm::zip(op.keys(), op.values()))
        {
            rewriter.create<pylir::Py::DictSetItemOp>(op.getLoc(), init, key, value);
        }
        return mlir::success();
    }
};

struct MakeFuncOpConversion : mlir::OpRewritePattern<pylir::Py::MakeFuncOp>
{
    using mlir::OpRewritePattern<pylir::Py::MakeFuncOp>::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(pylir::Py::MakeFuncOp op, mlir::PatternRewriter& rewriter) const override
    {
        auto dict = rewriter.create<pylir::Py::ConstantOp>(
            op.getLoc(), mlir::FlatSymbolRefAttr::get(getContext(), pylir::Py::Builtins::Function.name));
        auto mem = rewriter.create<pylir::Mem::GCAllocObjectOp>(op.getLoc(), dict);
        rewriter.replaceOpWithNewOp<pylir::Mem::InitFuncOp>(op, mem, op.functionAttr());
        return mlir::success();
    }
};

struct MakeObjectOpConversion : mlir::OpRewritePattern<pylir::Py::MakeObjectOp>
{
    using mlir::OpRewritePattern<pylir::Py::MakeObjectOp>::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(pylir::Py::MakeObjectOp op, mlir::PatternRewriter& rewriter) const override
    {
        auto mem = rewriter.create<pylir::Mem::GCAllocObjectOp>(op.getLoc(), op.typeObj());
        rewriter.replaceOpWithNewOp<pylir::Mem::InitObjectOp>(op, mem);
        return mlir::success();
    }
};

struct ListToTupleOpConversion : mlir::OpRewritePattern<pylir::Py::ListToTupleOp>
{
    using mlir::OpRewritePattern<pylir::Py::ListToTupleOp>::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(pylir::Py::ListToTupleOp op, mlir::PatternRewriter& rewriter) const override
    {
        auto tuple = rewriter.create<pylir::Py::ConstantOp>(
            op.getLoc(), mlir::FlatSymbolRefAttr::get(getContext(), pylir::Py::Builtins::Tuple.name));
        auto mem = rewriter.create<pylir::Mem::GCAllocObjectOp>(op.getLoc(), tuple);
        rewriter.replaceOpWithNewOp<pylir::Mem::InitTupleFromListOp>(op, mem, op.list(), mlir::Value{});
        return mlir::success();
    }
};

struct BoolFromI1OpConversion : mlir::OpRewritePattern<pylir::Py::BoolFromI1Op>
{
    using mlir::OpRewritePattern<pylir::Py::BoolFromI1Op>::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(pylir::Py::BoolFromI1Op op, mlir::PatternRewriter& rewriter) const override
    {
        auto boolean = rewriter.create<pylir::Py::ConstantOp>(
            op.getLoc(), mlir::FlatSymbolRefAttr::get(getContext(), pylir::Py::Builtins::Bool.name));
        auto mem = rewriter.create<pylir::Mem::GCAllocObjectOp>(op.getLoc(), boolean);
        rewriter.replaceOpWithNewOp<pylir::Mem::InitIntOp>(op, mem, op.input());
        return mlir::success();
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
                        pylir::Py::MakeFuncOp, pylir::Py::MakeObjectOp, pylir::Py::ListToTupleOp,
                        pylir::Py::BoolFromI1Op>();
    target.addDynamicallyLegalOp<pylir::Py::ConstantOp>(
        [](pylir::Py::ConstantOp constantOp) -> bool
        { return constantOp.constant().isa<pylir::Py::UnboundAttr, mlir::SymbolRefAttr>(); });

    mlir::RewritePatternSet patterns(&getContext());
    patterns.insert<ConstantOpConversion>(getOperation());
    patterns.insert<MakeTupleOpConversion>(&getContext());
    patterns.insert<MakeListOpConversion>(&getContext());
    patterns.insert<MakeSetOpConversion>(&getContext());
    patterns.insert<MakeDictOpConversion>(&getContext());
    patterns.insert<MakeFuncOpConversion>(&getContext());
    patterns.insert<MakeObjectOpConversion>(&getContext());
    patterns.insert<ListToTupleOpConversion>(&getContext());
    patterns.insert<BoolFromI1OpConversion>(&getContext());
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
