#include <mlir/Pass/Pass.h>

#include <llvm/ADT/TypeSwitch.h>

#include <pylir/Optimizer/Analysis/MemorySSA.hpp>

#include <memory>

namespace
{
#define GEN_PASS_CLASSES
#include "Passes.h.inc"

class TestMemorySSA : public TestMemorySSABase<TestMemorySSA>
{
protected:
    void runOnOperation() override;
};

void TestMemorySSA::runOnOperation()
{
    for (auto func : getOperation().getOps<mlir::FuncOp>())
    {
        llvm::outs() << getChildAnalysis<pylir::MemorySSA>(func);
    }
}
} // namespace

std::unique_ptr<mlir::Pass> createTestMemorySSA()
{
    return std::make_unique<TestMemorySSA>();
}
