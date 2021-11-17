#include <mlir/Conversion/LLVMCommon/ConversionTarget.h>
#include <mlir/Conversion/LLVMCommon/Pattern.h>
#include <mlir/Conversion/LLVMCommon/TypeConverter.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Transforms/DialectConversion.h>

#include <llvm/ADT/ScopeExit.h>
#include <llvm/ADT/Triple.h>
#include <llvm/ADT/TypeSwitch.h>

#include <pylir/Optimizer/Conversion/PassDetail.hpp>
#include <pylir/Optimizer/PylirMem/IR/PylirMemDialect.hpp>

#include <optional>

#include "WinX64.hpp"
#include "X86_64.hpp"

namespace
{
struct ConvertPylirToLLVMPass : public pylir::ConvertPylirToLLVMBase<ConvertPylirToLLVMPass>
{
protected:
    void runOnOperation() override;
};

void ConvertPylirToLLVMPass::runOnOperation()
{
    auto module = getOperation();
}
} // namespace

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> pylir::Mem::createConvertPylirToLLVMPass()
{
    return std::make_unique<ConvertPylirToLLVMPass>();
}
