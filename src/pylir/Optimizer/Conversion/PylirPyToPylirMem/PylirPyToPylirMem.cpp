#include <mlir/Transforms/DialectConversion.h>

#include <llvm/ADT/DenseMap.h>

#include <pylir/Optimizer/Conversion/PassDetail.hpp>
#include <pylir/Optimizer/PylirMem/IR/PylirMemDialect.hpp>
#include <pylir/Optimizer/PylirMem/IR/PylirMemOps.hpp>
#include <pylir/Optimizer/PylirPy/IR/PylirPyDialect.hpp>
#include <pylir/Optimizer/PylirPy/IR/PylirPyOps.hpp>

namespace
{

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

struct TuplePrependOpConversion : mlir::OpRewritePattern<pylir::Py::TuplePrependOp>
{
    using mlir::OpRewritePattern<pylir::Py::TuplePrependOp>::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(pylir::Py::TuplePrependOp op, mlir::PatternRewriter& rewriter) const override
    {
        auto tuple = rewriter.create<pylir::Py::ConstantOp>(
            op.getLoc(), mlir::FlatSymbolRefAttr::get(getContext(), pylir::Py::Builtins::Tuple.name));
        auto mem = rewriter.create<pylir::Mem::GCAllocObjectOp>(op.getLoc(), tuple);
        rewriter.replaceOpWithNewOp<pylir::Mem::InitTuplePrependOp>(op, mem, op.input(), op.tuple(), mlir::Value{});
        return mlir::success();
    }
};

struct TuplePopOpConversion : mlir::OpRewritePattern<pylir::Py::TuplePopFrontOp>
{
    using mlir::OpRewritePattern<pylir::Py::TuplePopFrontOp>::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(pylir::Py::TuplePopFrontOp op, mlir::PatternRewriter& rewriter) const override
    {
        auto tuple = rewriter.create<pylir::Py::ConstantOp>(
            op.getLoc(), mlir::FlatSymbolRefAttr::get(getContext(), pylir::Py::Builtins::Tuple.name));
        auto mem = rewriter.create<pylir::Mem::GCAllocObjectOp>(op.getLoc(), tuple);
        auto zeroI = rewriter.create<mlir::arith::ConstantIndexOp>(op.getLoc(), 0);
        auto result = rewriter.create<pylir::Mem::InitTuplePopFrontOp>(op.getLoc(), mem, op.tuple(), mlir::Value{});
        auto firstElement = rewriter.create<pylir::Py::TupleGetItemOp>(op.getLoc(), op.tuple(), zeroI);
        rewriter.replaceOp(op, {firstElement, result});
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

struct StrConcatOpConversion : mlir::OpRewritePattern<pylir::Py::StrConcatOp>
{
    using mlir::OpRewritePattern<pylir::Py::StrConcatOp>::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(pylir::Py::StrConcatOp op, mlir::PatternRewriter& rewriter) const override
    {
        auto str = rewriter.create<pylir::Py::ConstantOp>(
            op.getLoc(), mlir::FlatSymbolRefAttr::get(getContext(), pylir::Py::Builtins::Str.name));
        auto mem = rewriter.create<pylir::Mem::GCAllocObjectOp>(op.getLoc(), str);
        rewriter.replaceOpWithNewOp<pylir::Mem::InitStrOp>(op, mem, op.strings());
        return mlir::success();
    }
};

struct StrCopyOpConversion : mlir::OpRewritePattern<pylir::Py::StrCopyOp>
{
    using mlir::OpRewritePattern<pylir::Py::StrCopyOp>::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(pylir::Py::StrCopyOp op, mlir::PatternRewriter& rewriter) const override
    {
        auto mem = rewriter.create<pylir::Mem::GCAllocObjectOp>(op.getLoc(), op.typeObject());
        rewriter.replaceOpWithNewOp<pylir::Mem::InitStrOp>(op, mem, op.string());
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
        auto mem = rewriter.create<pylir::Mem::GCAllocObjectOp>(op.getLoc(), op.typeObject());
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

struct IntFromIntegerOpConversion : mlir::OpRewritePattern<pylir::Py::IntFromIntegerOp>
{
    using mlir::OpRewritePattern<pylir::Py::IntFromIntegerOp>::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(pylir::Py::IntFromIntegerOp op, mlir::PatternRewriter& rewriter) const override
    {
        auto integer = rewriter.create<pylir::Py::ConstantOp>(
            op.getLoc(), mlir::FlatSymbolRefAttr::get(getContext(), pylir::Py::Builtins::Int.name));
        auto mem = rewriter.create<pylir::Mem::GCAllocObjectOp>(op.getLoc(), integer);
        rewriter.replaceOpWithNewOp<pylir::Mem::InitIntOp>(op, mem, op.input());
        return mlir::success();
    }
};

struct IntAdOpConversion : mlir::OpRewritePattern<pylir::Py::IntAddOp>
{
    using mlir::OpRewritePattern<pylir::Py::IntAddOp>::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(pylir::Py::IntAddOp op, mlir::PatternRewriter& rewriter) const override
    {
        auto integer = rewriter.create<pylir::Py::ConstantOp>(
            op.getLoc(), mlir::FlatSymbolRefAttr::get(getContext(), pylir::Py::Builtins::Int.name));
        auto mem = rewriter.create<pylir::Mem::GCAllocObjectOp>(op.getLoc(), integer);
        rewriter.replaceOpWithNewOp<pylir::Mem::InitIntAddOp>(op, mem, op.lhs(), op.rhs());
        return mlir::success();
    }
};

struct IntToStrOpConversion : mlir::OpRewritePattern<pylir::Py::IntToStrOp>
{
    using mlir::OpRewritePattern<pylir::Py::IntToStrOp>::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(pylir::Py::IntToStrOp op, mlir::PatternRewriter& rewriter) const override
    {
        auto integer = rewriter.create<pylir::Py::ConstantOp>(
            op.getLoc(), mlir::FlatSymbolRefAttr::get(getContext(), pylir::Py::Builtins::Str.name));
        auto mem = rewriter.create<pylir::Mem::GCAllocObjectOp>(op.getLoc(), integer);
        rewriter.replaceOpWithNewOp<pylir::Mem::InitStrFromIntOp>(op, mem, op.input());
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

    target
        .addIllegalOp<pylir::Py::MakeTupleOp, pylir::Py::MakeListOp, pylir::Py::MakeSetOp, pylir::Py::MakeDictOp,
                      pylir::Py::MakeFuncOp, pylir::Py::MakeObjectOp, pylir::Py::ListToTupleOp, pylir::Py::BoolFromI1Op,
                      pylir::Py::IntFromIntegerOp, pylir::Py::StrConcatOp, pylir::Py::IntToStrOp, pylir::Py::StrCopyOp,
                      pylir::Py::TuplePopFrontOp, pylir::Py::TuplePrependOp, pylir::Py::IntAddOp>();

    mlir::RewritePatternSet patterns(&getContext());
    patterns.insert<MakeTupleOpConversion>(&getContext());
    patterns.insert<MakeListOpConversion>(&getContext());
    patterns.insert<MakeSetOpConversion>(&getContext());
    patterns.insert<MakeDictOpConversion>(&getContext());
    patterns.insert<StrConcatOpConversion>(&getContext());
    patterns.insert<MakeFuncOpConversion>(&getContext());
    patterns.insert<MakeObjectOpConversion>(&getContext());
    patterns.insert<ListToTupleOpConversion>(&getContext());
    patterns.insert<BoolFromI1OpConversion>(&getContext());
    patterns.insert<IntFromIntegerOpConversion>(&getContext());
    patterns.insert<IntToStrOpConversion>(&getContext());
    patterns.insert<StrCopyOpConversion>(&getContext());
    patterns.insert<TuplePrependOpConversion>(&getContext());
    patterns.insert<TuplePopOpConversion>(&getContext());
    patterns.insert<IntAdOpConversion>(&getContext());
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
