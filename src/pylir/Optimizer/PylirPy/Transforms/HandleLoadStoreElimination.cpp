// Copyright 2022 Markus Böck
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <mlir/IR/Dominance.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>

#include <llvm/ADT/ScopeExit.h>

#include <pylir/Optimizer/PylirPy/IR/PylirPyOps.hpp>
#include <pylir/Optimizer/Transforms/Util/SSABuilder.hpp>
#include <pylir/Optimizer/Transforms/Util/SSAUpdater.hpp>

#include "PassDetail.hpp"
#include "Passes.hpp"

namespace
{

struct BlockData
{
    std::vector<std::pair<pylir::Py::LoadOp, pylir::ValueTracker>> candidates;
};

struct HandleLoadStoreEliminationPass : HandleLoadStoreEliminationBase<HandleLoadStoreEliminationPass>
{
protected:
    void runOnOperation() override;

private:
    bool optimizeBlock(mlir::Block& block, mlir::Value clobberTracker, BlockData& blockArgUsages,
                       llvm::DenseMap<mlir::SymbolRefAttr, pylir::SSABuilder::DefinitionsMap>& definitions,
                       pylir::SSABuilder& ssaBuilder);
};

bool dependsOnClobber(mlir::BlockArgument argument, mlir::Value clobber,
                      llvm::SmallDenseMap<mlir::BlockArgument, bool, 8>& seen)
{
    auto [result, inserted] = seen.insert({argument, false});
    if (!inserted)
    {
        return result->second;
    }
    auto* block = argument.getParentBlock();
    for (auto iter = block->pred_begin(); iter != block->pred_end(); iter++)
    {
        auto branch = llvm::dyn_cast<mlir::BranchOpInterface>((*iter)->getTerminator());
        if (!branch)
        {
            // Conservative result.
            return result->second = true;
        }
        auto operands = branch.getSuccessorOperands(iter.getSuccessorIndex());
        auto op = operands[argument.getArgNumber()];
        if (op == clobber)
        {
            return result->second = true;
        }
        if (auto newArg = op.dyn_cast_or_null<mlir::BlockArgument>(); newArg && dependsOnClobber(newArg, clobber, seen))
        {
            return result->second = true;
        }
    }
    return false;
}

void HandleLoadStoreEliminationPass::runOnOperation()
{
    bool changed = false;
    auto* topLevel = getOperation();
    mlir::OwningOpRef<mlir::UnrealizedConversionCastOp> clobberTracker;
    {
        mlir::OpBuilder builder(&getContext());
        clobberTracker = builder.create<mlir::UnrealizedConversionCastOp>(
            builder.getUnknownLoc(), builder.getType<pylir::Py::DynamicType>(), mlir::ValueRange{});
    }
    for (auto& region : topLevel->getRegions())
    {
        BlockData blockArgUsages;
        llvm::DenseMap<mlir::SymbolRefAttr, pylir::SSABuilder::DefinitionsMap> definitions;
        pylir::SSABuilder ssaBuilder([&clobberTracker](mlir::BlockArgument) -> mlir::Value
                                     { return clobberTracker->getResult(0); });

        pylir::updateSSAinRegion(ssaBuilder, region,
                                 [&](mlir::Block* block) {
                                     changed |= optimizeBlock(*block, clobberTracker->getResult(0), blockArgUsages,
                                                              definitions, ssaBuilder);
                                 });

        llvm::SmallDenseMap<mlir::BlockArgument, bool, 8> blockArgsClobber;
        for (auto& [load, tracker] : blockArgUsages.candidates)
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

            // Only if the block argument does not depend on clobbers is it safe to replae the load with it. Otherwise,
            // a re-load is required.
            if (!dependsOnClobber(blockArg, clobberTracker->getResult(0), blockArgsClobber))
            {
                changed = true;
                load.replaceAllUsesWith(value);
                load->erase();
                continue;
            }
        }

        // Any block args that depend on the clobberTracker, and hence are unused by normal IR, have to now be deleted.
        for (auto& [blockArg, clobbered] : blockArgsClobber)
        {
            if (!clobbered)
            {
                continue;
            }
            for (auto pred = blockArg.getOwner()->pred_begin(); pred != blockArg.getOwner()->pred_end(); pred++)
            {
                auto terminator = mlir::cast<mlir::BranchOpInterface>((*pred)->getTerminator());
                terminator.getSuccessorOperands(pred.getSuccessorIndex()).erase(blockArg.getArgNumber());
            }
            // Other block args that we will be deleting, but haven't yet, might still have a use of this block arg.
            // Drop those.
            blockArg.dropAllUses();
            blockArg.getOwner()->eraseArgument(blockArg.getArgNumber());
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
    mlir::Block& block, mlir::Value clobberTracker, BlockData& blockArgUsages,
    llvm::DenseMap<mlir::SymbolRefAttr, pylir::SSABuilder::DefinitionsMap>& definitions, pylir::SSABuilder& ssaBuilder)
{
    bool changed = false;
    llvm::DenseMap<mlir::SymbolRefAttr, pylir::Py::StoreOp> unusedStores;
    for (auto& op : llvm::make_early_inc_range(block))
    {
        if (auto load = mlir::dyn_cast<pylir::Py::LoadOp>(op))
        {
            unusedStores.erase(load.getHandleAttr());
            auto& map = definitions[load.getHandleAttr()];
            auto read = ssaBuilder.readVariable(load->getLoc(), load.getType(), map, &block);
            if (read == clobberTracker)
            {
                // Make a definition to avoid reloads
                map[&block] = load;
                continue;
            }
            // If it's a block arg we can't immediately replace it as it might still be clobbered from one of the
            // predecessor branches. We record these for now and instead handle them after having traversed every block.
            // Also making a definition in case this load does remain, so that there are no reloads.
            if (read.isa<mlir::BlockArgument>())
            {
                map[&block] = load;
                blockArgUsages.candidates.emplace_back(load, read);
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
            auto [iter, inserted] = unusedStores.insert({store.getHandleAttr(), store});
            if (!inserted)
            {
                m_storesRemoved++;
                iter->second->erase();
                iter->second = store;
                changed = true;
            }
            definitions[store.getHandleAttr()][&block] = store.getValue();
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
