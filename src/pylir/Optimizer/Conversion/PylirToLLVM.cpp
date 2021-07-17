
#include "PylirToLLVM.hpp"

#include <mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Transforms/DialectConversion.h>

#include <pylir/Optimizer/Dialect/PylirDialect.hpp>
#include <pylir/Optimizer/Dialect/PylirOps.hpp>

#include "PassDetail.hpp"

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
    populateWithGenerated(&getContext(), patterns);
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
