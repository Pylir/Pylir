#include "MemorySSA.hpp"

#include <mlir/Interfaces/SideEffectInterfaces.h>

#include <mlir/IR/ImplicitLocOpBuilder.h>
#include <mlir/IR/Dominance.h>

#include <llvm/ADT/DepthFirstIterator.h>

#include <llvm/ADT/TypeSwitch.h>

#include <pylir/Optimizer/Interfaces/CaptureInterface.hpp>
#include <pylir/Optimizer/Transforms/Utils/SSABuilder.hpp>

mlir::Operation* pylir::MemorySSA::getMemoryAccess(mlir::Operation* operation)
{
    return m_results.lookup(operation);
}

namespace
{
mlir::Operation* maybeAddAccess(mlir::ImplicitLocOpBuilder& builder, pylir::MemorySSA& ssa, mlir::Operation* operation,
                                mlir::Value lastDef)
{
    using namespace pylir::MemSSA;
    mlir::Operation* access = nullptr;
    llvm::SmallVector<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>> effects;
    auto memoryEffectOpInterface = mlir::dyn_cast<mlir::MemoryEffectOpInterface>(operation);
    if (memoryEffectOpInterface)
    {
        memoryEffectOpInterface.getEffects(effects);
    }
    if (memoryEffectOpInterface && effects.empty())
    {
        return nullptr;
    }
    if (llvm::any_of(effects, [](const auto& it) { return llvm::isa<mlir::MemoryEffects::Write>(it.getEffect()); }))
    {
        return builder.create<MemoryDefOp>(lastDef, operation);
    }
    else if (llvm::any_of(effects, [](const auto& it) { return llvm::isa<mlir::MemoryEffects::Read>(it.getEffect()); }))
    {
        access = builder.create<MemoryUseOp>(lastDef, operation);
    }

    auto capturing = mlir::dyn_cast<pylir::CaptureInterface>(operation);
    for (auto& operand : operation->getOpOperands())
    {
        auto opAccess = ssa.getMemoryAccess(operand.get().getDefiningOp());
        if (!opAccess)
        {
            continue;
        }
        // Reads are assumed to not capture until proven otherwise
        if (memoryEffectOpInterface
            && memoryEffectOpInterface.getEffectOnValue<mlir::MemoryEffects::Read>(operand.get()))
        {
            continue;
        }
        if (capturing && !capturing.capturesOperand(operand.getOperandNumber()))
        {
            continue;
        }
        // Conservatively assume it was captured and clobbered
        return builder.create<MemoryDefOp>(lastDef, operation);
    }
    return access;
}

mlir::Value getReadValue(mlir::Operation* op)
{
    auto effectOp = mlir::cast<mlir::MemoryEffectOpInterface>(op);
    llvm::SmallVector<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>> effects;
    effectOp.getEffects(effects);
    auto result = llvm::find_if(effects, [](typename decltype(effects)::const_reference effect)
                                { return mlir::isa<mlir::MemoryEffects::Read>(effect.getEffect()); });
    PYLIR_ASSERT(result != effects.end());
    PYLIR_ASSERT(result->getValue());
    return result->getValue();
}

} // namespace

void pylir::MemorySSA::createIR(mlir::Operation* operation)
{
    mlir::ImplicitLocOpBuilder builder(mlir::UnknownLoc::get(operation->getContext()), operation->getContext());
    m_region = builder.create<MemSSA::MemoryRegionOp>(mlir::FlatSymbolRefAttr::get(operation));
    PYLIR_ASSERT(operation->getNumRegions() == 1);
    auto& region = operation->getRegion(0);
    SSABuilder::DefinitionsMap lastDefs;
    pylir::SSABuilder ssaBuilder;

    auto hasUnresolvedPredecessors = [&](mlir::Block* block)
    {
        return llvm::any_of(block->getPredecessors(),
                            [&](mlir::Block* pred)
                            {
                                auto predMemBlock = m_blockMapping.lookup(pred);
                                if (!predMemBlock)
                                {
                                    return true;
                                }
                                return !predMemBlock->getParent();
                            });
    };

    // Insert entry block that has no predecessors
    m_blockMapping.insert({&region.getBlocks().front(), new mlir::Block});
    for (auto& block : region)
    {
        auto* memBlock = m_blockMapping.find(&block)->second;
        PYLIR_ASSERT(memBlock);
        m_region->body().push_back(memBlock);
        builder.setInsertionPointToStart(memBlock);
        // If any of the predecessors have not yet been inserted
        // mark the block as open
        if (hasUnresolvedPredecessors(&block))
        {
            ssaBuilder.markOpenBlock(memBlock);
        }

        mlir::Value lastDef;
        if (memBlock->isEntryBlock())
        {
            lastDef = builder.create<MemSSA::MemoryLiveOnEntryOp>();
        }
        else
        {
            lastDef = ssaBuilder.readVariable(builder.getType<MemSSA::DefType>(), lastDefs, memBlock);
        }
        for (auto& op : block)
        {
            auto result = maybeAddAccess(builder, *this, &op, lastDef);
            if (!result)
            {
                continue;
            }
            m_results.insert({&op, result});
            if (auto def = mlir::dyn_cast_or_null<MemSSA::MemoryDefOp>(result))
            {
                lastDef = def;
            }
        }
        lastDefs[memBlock] = lastDef;

        llvm::SmallVector<mlir::Block*> memSuccessors;
        llvm::SmallVector<mlir::Block*> sealAfter;
        for (auto* succ : block.getSuccessors())
        {
            auto lookup = m_blockMapping.lookup(succ);
            if (!lookup)
            {
                lookup = new mlir::Block;
                m_blockMapping.insert({succ, lookup});
            }
            else if (lookup->getParent())
            {
                // This particular successor seems to have already been filled
                // Check whether filling this block has made all of its predecessors filled
                // and seal it
                if (!hasUnresolvedPredecessors(succ))
                {
                    sealAfter.push_back(lookup);
                }
            }
            memSuccessors.push_back(lookup);
        }
        builder.create<MemSSA::MemoryBranchOp>(llvm::SmallVector<mlir::ValueRange>(memSuccessors.size()),
                                               memSuccessors);
        llvm::for_each(sealAfter, [&](mlir::Block* lookup) { ssaBuilder.sealBlock(lookup); });
    }
}

namespace
{
mlir::Value getLastClobber(mlir::Value location, mlir::AliasAnalysis& aliasAnalysis,
                           llvm::ArrayRef<mlir::Value> dominatingDefs)
{
    for (auto def : llvm::reverse(dominatingDefs.drop_front()))
    {
        if (auto blockArg = def.dyn_cast<mlir::BlockArgument>())
        {
            // TODO: Implement optimizations
            return blockArg;
        }
        auto memDef = def.getDefiningOp<pylir::MemSSA::MemoryDefOp>();
        auto modRef = aliasAnalysis.getModRef(memDef.instruction(), location);
        if (modRef.isMod())
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
        auto readValue = getReadValue(use.instruction());
        use.definitionMutable().assign(getLastClobber(readValue, aliasAnalysis, dominatingDefs));
    }
}
} // namespace

void pylir::MemorySSA::optimizeUses(mlir::AnalysisManager& analysisManager)
{
    auto& aliasAnalysis = analysisManager.getAnalysis<mlir::AliasAnalysis>();
    auto& dominanceInfo = analysisManager.getAnalysis<mlir::DominanceInfo>();

    llvm::SmallVector<mlir::Value> dominatingDefs;
    if (m_region->body().hasOneBlock())
    {
        optimizeUsesInBlock(&m_region->body().front(), aliasAnalysis, dominatingDefs);
        return;
    }

    auto& tree = dominanceInfo.getDomTree(&m_region->body());
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
