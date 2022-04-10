#include <mlir/IR/Threading.h>
#include <mlir/Pass/PassManager.h>

#include "PassDetail.hpp"
#include "Passes.hpp"
#include "Util/InlinerUtil.hpp"

namespace
{
class Inliner : public pylir::Py::InlinerBase<Inliner>
{
protected:
    void runOnOperation() override
    {

    }
};

} // namespace

std::unique_ptr<mlir::Pass> pylir::Py::createInlinerPass()
{
    return std::make_unique<Inliner>();
}
