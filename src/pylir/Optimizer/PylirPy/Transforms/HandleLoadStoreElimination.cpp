#include <mlir/IR/Dominance.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>

#include <llvm/ADT/ScopeExit.h>

#include <pylir/Optimizer/PylirPy/IR/PylirPyOps.hpp>
#include <pylir/Optimizer/Transforms/Utils/SSABuilder.hpp>

#include "PassDetail.hpp"
#include "Passes.hpp"

namespace
{

struct BlockData
{
    std::vector<std::pair<pylir::Py::LoadOp, pylir::ValueTracker>> candidates;
};

struct HandleLoadStoreEliminationPass : pylir::Py::HandleLoadStoreEliminationBase<HandleLoadStoreEliminationPass>
{
protected:
    void runOnOperation() override;

private:
    bool optimizeBlock(mlir::Block& block, mlir::Value clobberTracker,
                       llvm::DenseMap<mlir::Block*, BlockData>& blockArgUsages,
                       llvm::DenseMap<mlir::SymbolRefAttr, pylir::SSABuilder::DefinitionsMap>& definitions,
                       pylir::SSABuilder& ssaBuilder);
};

void HandleLoadStoreEliminationPass::runOnOperation()
{
    bool changed = false;
    auto topLevel = getOperation();
    getContext().allowUnregisteredDialects();
    mlir::OperationState state(mlir::UnknownLoc::get(&getContext()), "__clobber_tracker");
    state.addTypes({pylir::Py::DynamicType::get(&getContext())});
    mlir::Operation* clobberTracker = mlir::Operation::create(state);
    auto exit = llvm::make_scope_exit([&] { clobberTracker->erase(); });
    for (auto& region : topLevel->getRegions())
    {
        llvm::DenseMap<mlir::Block*, BlockData> blockArgUsages;
        llvm::DenseMap<mlir::SymbolRefAttr, pylir::SSABuilder::DefinitionsMap> definitions;
        pylir::SSABuilder ssaBuilder([clobberTracker](mlir::BlockArgument) -> mlir::Value
                                     { return clobberTracker->getResult(0); });
        llvm::DenseSet<mlir::Block*> seen;
        for (auto& block : region)
        {
            if (!llvm::all_of(block.getPredecessors(), [&](mlir::Block* pred) { return seen.contains(pred); }))
            {
                ssaBuilder.markOpenBlock(&block);
            }
            changed |= optimizeBlock(block, clobberTracker->getResult(0), blockArgUsages, definitions, ssaBuilder);
            seen.insert(&block);
            for (auto* succ : block.getSuccessors())
            {
                if (ssaBuilder.isOpenBlock(succ))
                {
                    if (llvm::all_of(succ->getPredecessors(), [&](mlir::Block* pred) { return seen.contains(pred); }))
                    {
                        ssaBuilder.sealBlock(succ);
                    }
                }
            }
        }

        for (auto& [block, blockData] : blockArgUsages)
        {
            for (auto& [load, tracker] : blockData.candidates)
            {
                mlir::Value value = tracker;
                if (value == clobberTracker->getResult(0))
                {
                    continue;
                }
                auto blockArg = value.dyn_cast<mlir::BlockArgument>();
                if (!blockArg)
                {
                    // Simplification has lead to the block argument being simplified to a single value which is not
                    // a clobber
                    load.replaceAllUsesWith(value);
                    load->erase();
                    changed = true;
                    continue;
                }
                llvm::SmallVector<mlir::MutableOperandRange> blockArgOperands;
                bool hasClobber = false;
                for (auto* pred : block->getPredecessors())
                {
                    auto terminator = mlir::dyn_cast<mlir::BranchOpInterface>(pred->getTerminator());
                    PYLIR_ASSERT(terminator);
                    auto successors = terminator->getSuccessors();
                    auto index = std::find(successors.begin(), successors.end(), block) - successors.begin();
                    auto ops = terminator.getMutableSuccessorOperands(index);
                    PYLIR_ASSERT(ops);
                    blockArgOperands.emplace_back(*ops);
                    if (static_cast<mlir::OperandRange>(*ops)[blockArg.getArgNumber()] == clobberTracker->getResult(0))
                    {
                        hasClobber = true;
                    }
                }
                if (!hasClobber)
                {
                    changed = true;
                    load.replaceAllUsesWith(value);
                    load->erase();
                    continue;
                }
                // the block argument persistent and one of the inputs is a clobber. The load is therefore not
                // redundant as it has to reload and the block argument should be removed instead
                for (auto& range : blockArgOperands)
                {
                    range.erase(blockArg.getArgNumber());
                }
                tracker = mlir::Value{};
                blockArg.getOwner()->eraseArgument(blockArg.getArgNumber());
            }
        }
    }
    if (!changed)
    {
        markAllAnalysesPreserved();
        return;
    }
    markAnalysesPreserved<mlir::DominanceInfo>();
}

bool HandleLoadStoreEliminationPass::optimizeBlock(
    mlir::Block& block, mlir::Value clobberTracker, llvm::DenseMap<mlir::Block*, BlockData>& blockArgUsages,
    llvm::DenseMap<mlir::SymbolRefAttr, pylir::SSABuilder::DefinitionsMap>& definitions, pylir::SSABuilder& ssaBuilder)
{
    bool changed = false;
    llvm::DenseMap<mlir::SymbolRefAttr, pylir::Py::StoreOp> unusedStores;
    for (auto& op : llvm::make_early_inc_range(block))
    {
        if (auto load = mlir::dyn_cast<pylir::Py::LoadOp>(op))
        {
            unusedStores.erase(load.handleAttr());
            auto& map = definitions[load.handleAttr()];
            bool createdBlockArg;
            auto read = ssaBuilder.readVariable(load->getLoc(), load.getType(), map, &block, &createdBlockArg);
            if (read == clobberTracker)
            {
                // Make a definition to avoid reloads
                map[&block] = load;
                continue;
            }
            if (createdBlockArg)
            {
                map[&block] = load;
                blockArgUsages[&block].candidates.emplace_back(load, read);
                continue;
            }

            m_loadRemoved++;
            changed = true;
            load.replaceAllUsesWith(read);
            load.erase();
            continue;
        }
        if (auto store = mlir::dyn_cast<pylir::Py::StoreOp>(op))
        {
            auto [iter, inserted] = unusedStores.insert({store.handleAttr(), store});
            if (!inserted)
            {
                m_storesRemoved++;
                iter->second->erase();
                iter->second = store;
                changed = true;
            }
            definitions[store.handleAttr()][&block] = store.value();
            continue;
        }
        auto mem = mlir::dyn_cast<mlir::MemoryEffectOpInterface>(op);
        llvm::SmallVector<mlir::MemoryEffects::EffectInstance> effects;
        if (mem)
        {
            mem.getEffects(effects);
        }
        bool mayWrite = false;
        bool mayRead = false;
        for (auto& effect : effects)
        {
            if (effect.getSymbolRef() || effect.getValue())
            {
                continue;
            }
            if (mlir::isa<mlir::MemoryEffects::Write>(effect.getEffect()))
            {
                mayWrite = true;
            }
            if (mlir::isa<mlir::MemoryEffects::Read>(effect.getEffect()))
            {
                mayRead = true;
            }
        }
        if (!mem || mayWrite)
        {
            // Conservatively assume all globals were written
            for (auto& [key, value] : definitions)
            {
                value[&block] = clobberTracker;
            }
        }
        if (!mem || mayRead)
        {
            // Conservatively assume all globals were read
            unusedStores.clear();
        }
    }
    return changed;
}
} // namespace

std::unique_ptr<mlir::Pass> pylir::Py::createHandleLoadStoreEliminationPass()
{
    return std::make_unique<HandleLoadStoreEliminationPass>();
}
