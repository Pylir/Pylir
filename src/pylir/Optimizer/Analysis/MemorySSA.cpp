#include "MemorySSA.hpp"

#include <mlir/Interfaces/SideEffectInterfaces.h>

#include <mlir/IR/ImplicitLocOpBuilder.h>

#include <llvm/ADT/TypeSwitch.h>

#include <pylir/Optimizer/Interfaces/CaptureInterface.hpp>
#include <pylir/Optimizer/Transforms/Utils/SSABuilder.hpp>

mlir::Operation* pylir::MemorySSA::getMemoryAccess(mlir::Operation* operation)
{
    auto result = m_results.find(operation);
    if (result == m_results.end())
    {
        return nullptr;
    }
    return result->second;
}

namespace
{
mlir::Operation* maybeAddAccess(mlir::ImplicitLocOpBuilder& builder, pylir::MemorySSA& ssa, mlir::Operation* operation,
                                mlir::Value lastDef)
{
    using namespace pylir::MemSSA;
    mlir::Operation* access = nullptr;
    auto memoryEffectOpInterface = mlir::dyn_cast<mlir::MemoryEffectOpInterface>(operation);
    if (memoryEffectOpInterface)
    {
        if (memoryEffectOpInterface.hasNoEffect())
        {
            return {};
        }
        if (memoryEffectOpInterface.hasEffect<mlir::MemoryEffects::Write>())
        {
            access = builder.create<MemoryDefOp>(lastDef, operation);
        }
        else if (memoryEffectOpInterface.hasEffect<mlir::MemoryEffects::Read>())
        {
            access = builder.create<MemoryUseOp>(lastDef, operation, mlir::AliasResult::MayAlias);
        }
    }
    // Already identified as a Def
    if (mlir::isa_and_nonnull<MemoryDefOp>(access))
    {
        return access;
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

} // namespace

pylir::MemorySSA::MemorySSA(mlir::Operation* operation, mlir::AnalysisManager& analysisManager)
{
    auto& aliasAnalysis = analysisManager.getAnalysis<mlir::AliasAnalysis>();
    mlir::ImplicitLocOpBuilder builder(mlir::UnknownLoc::get(operation->getContext()), operation->getContext());
    m_region = builder.create<MemSSA::MemoryRegionOp>(mlir::FlatSymbolRefAttr::get(operation));
    PYLIR_ASSERT(operation->getNumRegions() == 1);
    auto& region = operation->getRegion(0);
    llvm::DenseMap<mlir::Block*, mlir::Block*> blockMapping;
    llvm::DenseMap<mlir::Block*, mlir::Value> lastDefs;
    pylir::SSABuilder ssaBuilder;

    auto hasUnresolvedPredecessors = [&](mlir::Block* block)
    {
        return llvm::any_of(block->getPredecessors(),
                            [&](mlir::Block* pred)
                            {
                                auto predMemBlock = blockMapping.lookup(pred);
                                if (!predMemBlock)
                                {
                                    return true;
                                }
                                return !predMemBlock->getParent();
                            });
    };

    // Insert entry block that has no predecessors
    blockMapping.insert({&region.getBlocks().front(), new mlir::Block});
    for (auto& block : region)
    {
        auto* memBlock = blockMapping.find(&block)->second;
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
            auto lookup = blockMapping.lookup(succ);
            if (!lookup)
            {
                lookup = new mlir::Block;
                blockMapping.insert({succ, lookup});
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

void pylir::MemorySSA::dump() const
{
    m_region.get()->dump();
}

void pylir::MemorySSA::print(llvm::raw_ostream& out) const
{
    m_region.get().print(out);
}
