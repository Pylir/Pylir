#include "PassDetail.hpp"
#include "Passes.hpp"

namespace
{
struct RemoveAllocasPass : public pylir::Py::RemoveAllocasBase<RemoveAllocasPass>
{
    void runOnFunction() override;
};

void RemoveAllocasPass::runOnFunction() {}
} // namespace

std::unique_ptr<mlir::Pass> pylir::Py::createRemoveAllocasPass()
{
    return std::make_unique<RemoveAllocasPass>();
}
