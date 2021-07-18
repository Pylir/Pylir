
#include "PylirToLLVM.hpp"

#include <mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Transforms/DialectConversion.h>

#include <llvm/ADT/TypeSwitch.h>

#include <pylir/Optimizer/Dialect/PylirDialect.hpp>
#include <pylir/Optimizer/Dialect/PylirOps.hpp>
#include <pylir/Optimizer/Dialect/PylirTypeObjects.hpp>
#include <pylir/Support/Macros.hpp>

#include "PassDetail.hpp"

namespace
{
mlir::Type getPyObject(mlir::MLIRContext* context)
{
    auto pyObject = mlir::LLVM::LLVMStructType::getIdentified(context, "PyObject");
    if (pyObject.getBody().empty())
    {
        std::vector<mlir::Type> types;
        types.emplace_back(mlir::LLVM::LLVMPointerType::get(pyObject)); // pointer to the type
        pyObject.setBody(types, false);
    }
    return pyObject;
}

mlir::Type getPyBaseObject(mlir::MLIRContext* context)
{
    auto pyBaseObject = mlir::LLVM::LLVMStructType::getIdentified(context, "PyBaseObject");
    if (pyBaseObject.getBody().empty())
    {
        std::vector<mlir::Type> types;
        types.emplace_back(getPyObject(context)); // base
        types.emplace_back(
            mlir::LLVM::LLVMPointerType::get(mlir::IntegerType::get(context, 8))); // opaque pointer to the dict for now
        pyBaseObject.setBody(types, false);
    }
    return pyBaseObject;
}

template <class Self, class Op>
struct SingleOpMatcher : public mlir::ConversionPattern
{
    SingleOpMatcher(llvm::StringRef rootOpName, mlir::MLIRContext* context, mlir::LLVMTypeConverter& typeConverter,
                    mlir::PatternBenefit benefit = 1)
        : mlir::ConversionPattern(rootOpName, benefit, typeConverter, context)
    {
    }

    mlir::LogicalResult match(mlir::Operation* op) const override
    {
        return mlir::success(mlir::isa<Op>(op));
    }

    void rewrite(mlir::Operation* op, llvm::ArrayRef<mlir::Value> operands,
                 mlir::ConversionPatternRewriter& rewriter) const override
    {
        static_cast<const Self*>(this)->Self::rewrite(op, typename Op::Adaptor(operands, op->getAttrDictionary()),
                                                      rewriter);
    }
};

struct ConstantOpConversion : SingleOpMatcher<ConstantOpConversion, pylir::Dialect::ConstantOp>
{
    using SingleOpMatcher::SingleOpMatcher;

    using SingleOpMatcher::rewrite;

    void rewrite(mlir::Operation* op, pylir::Dialect::ConstantOp::Adaptor adaptor,
                 mlir::ConversionPatternRewriter& rewriter) const
    {
        auto typeObject =
            llvm::TypeSwitch<mlir::Attribute, std::string_view>(adaptor.value())
                .Case<pylir::Dialect::IntegerAttr>([](auto) { return pylir::Dialect::longTypeObjectName; })
                .Case<mlir::FlatSymbolRefAttr>([](auto) { return pylir::Dialect::functionTypeObjectName; })
                .Default([](auto) -> std::string_view { PYLIR_UNREACHABLE; });
    }
};

struct GlobalOpConversion : SingleOpMatcher<GlobalOpConversion, pylir::Dialect::GlobalOp>
{
    using SingleOpMatcher::SingleOpMatcher;

    using SingleOpMatcher::rewrite;

    void rewrite(mlir::Operation* op, pylir::Dialect::GlobalOp::Adaptor adaptor,
                 mlir::ConversionPatternRewriter& rewriter) const
    {
        mlir::Type resultType;
        if (!adaptor.constant())
        {
            resultType = mlir::LLVM::LLVMPointerType::get(getPyObject(rewriter.getContext()));
        }
        else
        {
            resultType = typeConverter->convertType(adaptor.type().getValue());
            PYLIR_ASSERT(resultType);
        }
        auto linkage = mlir::LLVM::Linkage::External;
        if (op->hasAttr("linkonce"))
        {
            linkage = mlir::LLVM::Linkage::Linkonce;
        }
        auto newOp = rewriter.create<mlir::LLVM::GlobalOp>(op->getLoc(), resultType, false, linkage,
                                                           adaptor.sym_name().getValue(), mlir::Attribute{});
        if (!adaptor.initializer())
        {
            rewriter.eraseOp(op);
            return;
        }
    }
};
} // namespace

namespace
{
#include <pylir/Optimizer/Conversion/PylirToLLVMPatterns.h.inc>

struct ConvertPylirToLLVMPass : public pylir::Dialect::ConvertPylirToLLVMBase<ConvertPylirToLLVMPass>
{
protected:
    void runOnOperation() override;
};

void ConvertPylirToLLVMPass::runOnOperation()
{
    auto module = getOperation();

    mlir::OwningRewritePatternList patterns;
    mlir::LLVMTypeConverter converter(&getContext());

    converter.addConversion(
        [](pylir::Dialect::UnknownType type) -> llvm::Optional<mlir::Type> {
            return mlir::LLVM::LLVMPointerType::get(getPyObject(type.getContext()));
        });
    converter.addConversion(
        [](pylir::Dialect::KnownTypeObjectType type) -> llvm::Optional<mlir::Type> {
            return mlir::LLVM::LLVMPointerType::get(getPyBaseObject(type.getContext()));
        });

    populateWithGenerated(&getContext(), patterns);
    // patterns.insert<ConstantOpConversion>(pylir::Dialect::ConstantOp::getOperationName(), &getContext(), converter);
    patterns.insert<GlobalOpConversion>(pylir::Dialect::GlobalOp::getOperationName(), &getContext(), converter);
    mlir::populateStdToLLVMConversionPatterns(converter, patterns);

    mlir::ConversionTarget target(getContext());
    target.addLegalDialect<mlir::LLVM::LLVMDialect>();
    target.addIllegalDialect<pylir::Dialect::PylirDialect, mlir::StandardOpsDialect>();
    target.addLegalOp<mlir::ModuleOp, mlir::ModuleTerminatorOp>();
    if (mlir::failed(mlir::applyFullConversion(module, target, std::move(patterns))))
    {
        signalPassFailure();
    }
}
} // namespace

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> pylir::Dialect::createConvertPylirToLLVMPass()
{
    return std::make_unique<ConvertPylirToLLVMPass>();
}
