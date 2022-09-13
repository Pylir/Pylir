//  Licensed under the Apache License v2.0 with LLVM Exceptions.
//  See https://llvm.org/LICENSE.txt for license information.
//  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "MemorySSA.hpp"

#include <mlir/IR/Dominance.h>
#include <mlir/IR/ImplicitLocOpBuilder.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>

#include <llvm/ADT/DepthFirstIterator.h>
#include <llvm/ADT/TypeSwitch.h>

#include <pylir/Optimizer/Interfaces/CaptureInterface.hpp>
#include <pylir/Optimizer/Transforms/Util/SSABuilder.hpp>

mlir::Operation* pylir::MemorySSA::getMemoryAccess(mlir::Operation* operation)
{
    return m_results.lookup(operation);
}

namespace
{
mlir::Operation* maybeAddAccess(mlir::ImplicitLocOpBuilder& builder, mlir::Operation* operation, mlir::Value lastDef)
{
    using namespace pylir::MemSSA;
    llvm::SmallVector<mlir::MemoryEffects::EffectInstance> effects;
    auto memoryEffectOpInterface = mlir::dyn_cast<mlir::MemoryEffectOpInterface>(operation);
    if (memoryEffectOpInterface)
    {
        memoryEffectOpInterface.getEffects(effects);
    }
    if (!memoryEffectOpInterface && operation->hasTrait<mlir::OpTrait::HasRecursiveSideEffects>())
    {
        return nullptr;
    }

    llvm::SmallSetVector<llvm::PointerUnion<mlir::Value, mlir::SymbolRefAttr>, 4> read, written;
    for (auto& iter : effects)
    {
        decltype(read)* set = nullptr;
        if (llvm::isa<mlir::MemoryEffects::Write>(iter.getEffect()))
        {
            set = &written;
        }
        else if (llvm::isa<mlir::MemoryEffects::Read>(iter.getEffect()))
        {
            set = &read;
        }
        else
        {
            continue;
        }

        if (iter.getValue())
        {
            set->insert(iter.getValue());
        }
        else if (iter.getSymbolRef())
        {
            set->insert(iter.getSymbolRef());
        }
        else
        {
            set->insert(nullptr);
        }
    }

    if (memoryEffectOpInterface && read.empty() && written.empty())
    {
        return nullptr;
    }

    if (!written.empty())
    {
        return builder.create<MemoryDefOp>(lastDef, operation, written.takeVector(), read.takeVector());
    }

    if (!read.empty())
    {
        return builder.create<MemoryUseOp>(lastDef, operation, read.takeVector());
    }

    // This is the conservative case that may happen if an op does not implement 'MemoryEffectOpInterface'. In this case
    // we do a memory def, specify that all memory is clobbered and all memory was also read.
    return builder.create<MemoryDefOp>(lastDef, operation, llvm::PointerUnion<mlir::Value, mlir::SymbolRefAttr>{},
                                       llvm::PointerUnion<mlir::Value, mlir::SymbolRefAttr>{});
}

void fillRegion(mlir::Region& region, mlir::ImplicitLocOpBuilder& builder,
                llvm::DenseMap<mlir::Block*, mlir::Block*>& blockMapping, pylir::MemorySSA& ssa,
                pylir::SSABuilder& ssaBuilder, pylir::SSABuilder::DefinitionsMap& lastDefs,
                llvm::DenseMap<mlir::Operation*, mlir::Operation*>& results,
                llvm::ArrayRef<mlir::Block*> regionSuccessors)
{
    auto hasUnresolvedPredecessors = [&](mlir::Block* block)
    {
        return llvm::any_of(block->getPredecessors(),
                            [&](mlir::Block* pred)
                            {
                                auto* predMemBlock = blockMapping.lookup(pred);
                                if (!predMemBlock)
                                {
                                    return true;
                                }
                                return !predMemBlock->getParent();
                            });
    };

    for (auto& block : region)
    {
        mlir::Block* memBlock;
        {
            auto [lookup, inserted] = blockMapping.insert({&block, nullptr});
            if (inserted)
            {
                lookup->second = new mlir::Block;
            }
            memBlock = lookup->second;
        }
        // If any of the predecessors have not yet been inserted
        // mark the block as open
        if (hasUnresolvedPredecessors(&block))
        {
            ssaBuilder.markOpenBlock(memBlock);
        }
        ssa.getMemoryRegion().push_back(memBlock);
        builder.setInsertionPointToStart(memBlock);

        mlir::Value lastDef;
        if (memBlock->isEntryBlock())
        {
            lastDef = builder.create<pylir::MemSSA::MemoryLiveOnEntryOp>();
        }
        else
        {
            lastDef = ssaBuilder.readVariable(builder.getLoc(), builder.getType<pylir::MemSSA::DefType>(), lastDefs,
                                              memBlock);
        }
        for (auto& op : block)
        {
            auto* result = maybeAddAccess(builder, &op, lastDef);
            if (result)
            {
                results.insert({&op, result});
                if (auto def = mlir::dyn_cast_or_null<pylir::MemSSA::MemoryDefOp>(result))
                {
                    lastDef = def;
                }
            }
            auto regionBranchOp = mlir::dyn_cast<mlir::RegionBranchOpInterface>(&op);
            if (!regionBranchOp)
            {
                continue;
            }
            lastDefs[memBlock] = lastDef;
            llvm::SmallVector<mlir::RegionSuccessor> successors;
            regionBranchOp.getSuccessorRegions(llvm::None, successors);
            PYLIR_ASSERT(!successors.empty());
            auto* continueRegion = new mlir::Block;

            llvm::SmallVector<mlir::Block*> regionEntries;

            auto getRegionSuccBlocks = [&](const llvm::SmallVector<mlir::RegionSuccessor>& successors)
            {
                return llvm::to_vector(
                    llvm::map_range(successors,
                                    [&](const mlir::RegionSuccessor& succ)
                                    {
                                        if (succ.isParent())
                                        {
                                            return continueRegion;
                                        }
                                        auto result = blockMapping.insert({&succ.getSuccessor()->front(), nullptr});
                                        if (result.second)
                                        {
                                            result.first->second = new mlir::Block;
                                            // We have to conservatively mark region entries as open simply due to
                                            // having absolutely no way to tell whether all their predecessors
                                            // have been visited, due not being able to get a list of all predecessors!
                                            // We only get successors of a region once we are filling it in.
                                            // After we are done filling in all regions of the operation however we can
                                            // be sure all their predecessors have been filled
                                            ssaBuilder.markOpenBlock(result.first->second);
                                            regionEntries.push_back(result.first->second);
                                        }
                                        return result.first->second;
                                    }));
            };

            auto entryBlocks = getRegionSuccBlocks(successors);
            builder.create<pylir::MemSSA::MemoryBranchOp>(llvm::SmallVector<mlir::ValueRange>(entryBlocks.size()),
                                                          entryBlocks);

            llvm::SmallVector<mlir::Region*> workList;

            auto fillWorkList = [&](const llvm::SmallVector<mlir::RegionSuccessor>& successors)
            {
                for (const auto& iter : llvm::reverse(successors))
                {
                    if (!iter.isParent())
                    {
                        workList.push_back(iter.getSuccessor());
                    }
                }
            };

            fillWorkList(successors);

            llvm::DenseSet<mlir::Region*> seen;
            while (!workList.empty())
            {
                auto* succ = workList.pop_back_val();
                if (!seen.insert(succ).second)
                {
                    continue;
                }
                llvm::SmallVector<mlir::RegionSuccessor> subRegionSuccessors;
                regionBranchOp.getSuccessorRegions(succ->getRegionNumber(), subRegionSuccessors);
                fillRegion(*succ, builder, blockMapping, ssa, ssaBuilder, lastDefs, results,
                           getRegionSuccBlocks(subRegionSuccessors));
                fillWorkList(subRegionSuccessors);
            }
            for (auto& iter : regionEntries)
            {
                ssaBuilder.sealBlock(iter);
            }

            ssa.getMemoryRegion().push_back(continueRegion);
            builder.setInsertionPointToStart(continueRegion);
            memBlock = continueRegion;

            lastDef = ssaBuilder.readVariable(builder.getLoc(), builder.getType<pylir::MemSSA::DefType>(), lastDefs,
                                              memBlock);
        }
        lastDefs[memBlock] = lastDef;

        if (block.getTerminator()->hasTrait<mlir::OpTrait::ReturnLike>())
        {
            builder.create<pylir::MemSSA::MemoryBranchOp>(llvm::SmallVector<mlir::ValueRange>(regionSuccessors.size()),
                                                          regionSuccessors);
            continue;
        }

        llvm::SmallVector<mlir::Block*> memSuccessors;
        llvm::SmallVector<mlir::Block*> sealAfter;
        for (auto* succ : block.getSuccessors())
        {
            auto [lookup, inserted] = blockMapping.insert({succ, nullptr});
            if (inserted)
            {
                lookup->second = new mlir::Block;
            }
            else if (lookup->second->getParent())
            {
                // This particular successor seems to have already been filled
                // Check whether filling this block has made all of its predecessors filled
                // and seal it
                if (!hasUnresolvedPredecessors(succ))
                {
                    sealAfter.push_back(lookup->second);
                }
            }
            memSuccessors.push_back(lookup->second);
        }
        builder.create<pylir::MemSSA::MemoryBranchOp>(llvm::SmallVector<mlir::ValueRange>(memSuccessors.size()),
                                                      memSuccessors);
        llvm::for_each(sealAfter, [&](mlir::Block* lookup) { ssaBuilder.sealBlock(lookup); });
    }
}

} // namespace

void pylir::MemorySSA::createIR(mlir::Operation* operation)
{
    mlir::ImplicitLocOpBuilder builder(mlir::UnknownLoc::get(operation->getContext()), operation->getContext());
    m_region = builder.create<MemSSA::MemoryModuleOp>();
    PYLIR_ASSERT(operation->getNumRegions() == 1);
    auto& region = operation->getRegion(0);
    if (region.empty())
    {
        auto* block = new mlir::Block;
        m_region->getBody().push_back(block);
        builder.setInsertionPointToStart(block);
        builder.create<pylir::MemSSA::MemoryLiveOnEntryOp>();
        builder.create<pylir::MemSSA::MemoryBranchOp>(llvm::ArrayRef<mlir::ValueRange>{}, mlir::BlockRange{});
        return;
    }
    SSABuilder::DefinitionsMap lastDefs;
    pylir::SSABuilder ssaBuilder;

    // Insert entry block that has no predecessors
    m_blockMapping.insert({&region.getBlocks().front(), new mlir::Block});
    fillRegion(region, builder, m_blockMapping, *this, ssaBuilder, lastDefs, m_results, {});
}

namespace
{
mlir::Value getLastClobber(llvm::ArrayRef<llvm::PointerUnion<mlir::Value, mlir::SymbolRefAttr>> location,
                           mlir::AliasAnalysis& aliasAnalysis, llvm::ArrayRef<mlir::Value> dominatingDefs)
{
    for (auto def : llvm::reverse(dominatingDefs.drop_front()))
    {
        if (auto blockArg = def.dyn_cast<mlir::BlockArgument>())
        {
            // TODO: Implement optimizations
            return blockArg;
        }
        auto memDef = def.getDefiningOp<pylir::MemSSA::MemoryDefOp>();
        if (llvm::any_of(location,
                         [&](llvm::PointerUnion<mlir::Value, mlir::SymbolRefAttr> ptr)
                         {
                             // If no affected location is specified, conservatively assume it reads the def.
                             if (!ptr)
                             {
                                 return true;
                             }

                             if (auto val = ptr.dyn_cast<mlir::Value>())
                             {
                                 return aliasAnalysis.getModRef(memDef.getInstruction(), val).isMod();
                             }

                             // There is no support for symbol ref attrs in MLIRs alias analysis. For the time being we
                             // assume a symbol ref is always clobbered except if some other symbol is being written to.
                             // Technically speaking, this assumption could not hold if there was some kind of global
                             // symbol alias mechanism/op, but we just assume such a thing does not exist for now.
                             return llvm::any_of(memDef.getWrites(),
                                                 [ptr](llvm::PointerUnion<mlir::Value, mlir::SymbolRefAttr> write)
                                                 {
                                                     if (llvm::isa<mlir::SymbolRefAttr>(write))
                                                     {
                                                         return write == ptr;
                                                     }
                                                     return true;
                                                 });
                         }))
        {
            return memDef;
        }
    }
    return dominatingDefs[0];
}

void optimizeUsesInBlock(mlir::Block* block, mlir::AliasAnalysis& aliasAnalysis,
                         llvm::SmallVectorImpl<mlir::Value>& dominatingDefs)
{
    for (auto& blockArg : block->getArguments())
    {
        dominatingDefs.push_back(blockArg);
    }
    for (auto& access : block->without_terminator())
    {
        auto use = mlir::dyn_cast<pylir::MemSSA::MemoryUseOp>(access);
        if (!use)
        {
            dominatingDefs.push_back(access.getResult(0));
            continue;
        }
        use.getDefinitionMutable().assign(getLastClobber(use.getReads(), aliasAnalysis, dominatingDefs));
    }
}
} // namespace

void pylir::MemorySSA::optimizeUses(mlir::AnalysisManager& analysisManager)
{
    auto& aliasAnalysis = analysisManager.getAnalysis<mlir::AliasAnalysis>();
    auto& dominanceInfo = analysisManager.getAnalysis<mlir::DominanceInfo>();

    llvm::SmallVector<mlir::Value> dominatingDefs;
    if (m_region->getBody().hasOneBlock())
    {
        optimizeUsesInBlock(&m_region->getBody().front(), aliasAnalysis, dominatingDefs);
        return;
    }

    auto& tree = dominanceInfo.getDomTree(&m_region->getBody());
    for (auto* node : llvm::depth_first(tree.getRootNode()))
    {
        auto* block = node->getBlock();
        auto accesses = block->without_terminator();
        if (accesses.empty() && block->getNumArguments() == 0)
        {
            continue;
        }

        // Pop any values that are in blocks that do not dominate the current block
        while (!block->isEntryBlock() && !tree.dominates(dominatingDefs.back().getParentBlock(), block))
        {
            auto* backBlock = dominatingDefs.back().getParentBlock();
            while (dominatingDefs.back().getParentBlock() == backBlock)
            {
                dominatingDefs.pop_back();
            }
        }

        optimizeUsesInBlock(block, aliasAnalysis, dominatingDefs);
    }
}

pylir::MemorySSA::MemorySSA(mlir::Operation* operation, mlir::AnalysisManager& analysisManager)
{
    createIR(operation);
    optimizeUses(analysisManager);
}

void pylir::MemorySSA::dump() const
{
    m_region.get()->dump();
}

void pylir::MemorySSA::print(llvm::raw_ostream& out) const
{
    m_region.get().print(out);
}
