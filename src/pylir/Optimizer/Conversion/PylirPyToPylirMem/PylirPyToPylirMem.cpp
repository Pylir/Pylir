#include <pylir/Optimizer/Conversion/PassDetail.hpp>
#include <pylir/Optimizer/PylirMem/IR/PylirMemDialect.hpp>
#include <pylir/Optimizer/PylirPy/IR/PylirPyDialect.hpp>

namespace
{
struct ConvertPylirPyToPylirMem : pylir::ConvertPylirPyToPylirMemBase<ConvertPylirPyToPylirMem>
{
protected:
    void runOnOperation() override;
};

void ConvertPylirPyToPylirMem::runOnOperation() {}
} // namespace

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> pylir::createConvertPylirPyToPylirMemPass()
{
    return std::make_unique<ConvertPylirPyToPylirMem>();
}
