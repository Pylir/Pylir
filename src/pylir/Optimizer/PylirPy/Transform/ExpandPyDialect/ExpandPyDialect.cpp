#include "ExpandPyDialect.hpp"

#include <pylir/Optimizer/PylirPy/Transform/PassDetail.hpp>

namespace
{
struct ExpandPyDialectPass : public pylir::Py::ExpandPyDialectBase<ExpandPyDialectPass>
{
protected:
    void runOnOperation() override;
};

void ExpandPyDialectPass::runOnOperation() {}
} // namespace

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> pylir::Py::createExpandPyDialectPass()
{
    return std::make_unique<ExpandPyDialectPass>();
}
