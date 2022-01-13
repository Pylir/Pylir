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
    llvm::DenseMap<const pylir::MemoryAccess*, std::size_t> indices;
    auto printMemAccess = [&](const pylir::MemoryAccess& access)
    {
        llvm::TypeSwitch<const pylir::MemoryAccess*>(&access)
            .Case(
                [&](const pylir::MemoryDef* def)
                {
                    auto thisIndex = indices.insert({&access, indices.size()}).first->second;
                    llvm::outs() << thisIndex << " = MemoryDef(";
                    if (def->getPrevious())
                    {
                        llvm::outs() << indices.insert({def->getPrevious(), indices.size()}).first->second;
                    }
                    else
                    {
                        llvm::outs() << "entry";
                    }
                    llvm::outs() << ')';
                })
            .Case(
                [&](const pylir::MemoryUse* def)
                {
                    llvm::outs() << "MemoryUse(";
                    if (def->getDefinition())
                    {
                        llvm::outs() << indices.insert({def->getDefinition(), indices.size()}).first->second;
                    }
                    else
                    {
                        llvm::outs() << "entry";
                    }
                    llvm::outs() << ") " << def->getAliasResult();
                })
            .Case(
                [&](const pylir::MemoryPhi* def)
                {
                    auto thisIndex = indices.insert({&access, indices.size()}).first->second;
                    llvm::outs() << thisIndex << " = MemoryPhi(";
                    bool first = true;
                    for (auto& iter : def->getIncoming())
                    {
                        if (!first)
                        {
                            llvm::outs() << ", ";
                        }
                        first = false;
                        llvm::outs() << '{';
                        iter.first->printAsOperand(llvm::outs());
                        llvm::outs() << ", " << indices.insert({iter.second, indices.size()}).first->second << "}";
                    }
                    llvm::outs() << ')';
                });
    };

    for (auto func : getOperation().getOps<mlir::FuncOp>())
    {
        auto& memorySSA = getChildAnalysis<pylir::MemorySSA>(func);
        for (auto& block : func.getBlocks())
        {
            if (auto* result = memorySSA.getMemoryAccess(&block))
            {
                llvm::outs() << "// ";
                printMemAccess(*result);
                llvm::outs() << '\n';
                block.printAsOperand(llvm::outs());
                llvm::outs() << '\n';
            }
            for (auto& op : func.getOps())
            {
                auto* result = memorySSA.getMemoryAccess(&op);
                if (!result)
                {
                    continue;
                }
                llvm::outs() << "// ";
                printMemAccess(*result);
                llvm::outs() << '\n';
                llvm::outs() << op << '\n';
            }
        }
    }
}
} // namespace

std::unique_ptr<mlir::Pass> createTestMemorySSA()
{
    return std::make_unique<TestMemorySSA>();
}
