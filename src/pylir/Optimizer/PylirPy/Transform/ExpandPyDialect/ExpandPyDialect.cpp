#include "ExpandPyDialect.hpp"

#include <mlir/Transforms/DialectConversion.h>

#include <pylir/Optimizer/PylirPy/IR/PylirPyOps.hpp>
#include <pylir/Optimizer/PylirPy/Transform/PassDetail.hpp>

namespace
{
struct ExpandPyDialectPass : public pylir::Py::ExpandPyDialectBase<ExpandPyDialectPass>
{
    void runOnFunction() override;
};

void ExpandPyDialectPass::runOnFunction()
{
    mlir::ConversionTarget target(getContext());
    target.addDynamicallyLegalOp<pylir::Py::MakeTupleOp, pylir::Py::MakeTupleExOp, pylir::Py::MakeListOp,
                                 pylir::Py::MakeListExOp, pylir::Py::MakeSetOp, pylir::Py::MakeSetExOp,
                                 pylir::Py::MakeDictOp, pylir::Py::MakeDictExOp>(
        [](mlir::Operation* op) -> bool
        {
            if (mlir::isa<pylir::Py::MakeDictOp, pylir::Py::MakeDictExOp>(op))
            {
                return op->getAttrOfType<mlir::ArrayAttr>("mappingExpansion").empty();
            }
            return op->getAttrOfType<mlir::ArrayAttr>("iterExpansion").empty();
        });
    target.addIllegalOp<pylir::Py::MakeClassOp>();

    mlir::RewritePatternSet patterns(&getContext());
    if (mlir::failed(mlir::applyPartialConversion(getFunction(), target, std::move(patterns))))
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
