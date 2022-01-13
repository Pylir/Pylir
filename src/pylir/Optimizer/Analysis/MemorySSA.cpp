#include "MemorySSA.hpp"

#include <mlir/Interfaces/SideEffectInterfaces.h>

#include <llvm/ADT/TypeSwitch.h>

#include <pylir/Optimizer/Interfaces/CaptureInterface.hpp>

pylir::MemoryAccess* pylir::MemorySSA::getMemoryAccess(mlir::Operation* operation)
{
    auto result = m_results.find(operation);
    if (result == m_results.end())
    {
        return nullptr;
    }
    return result->second.get();
}

pylir::MemoryPhi* pylir::MemorySSA::getMemoryAccess(mlir::Block* block)
{
    auto result = m_results.find(block);
    if (result == m_results.end())
    {
        return nullptr;
    }
    return llvm::cast<MemoryPhi>(result->second.get());
}

namespace
{
std::unique_ptr<pylir::MemoryAccess> maybeAddAccess(pylir::MemorySSA& ssa, mlir::Operation* operation,
                                                    pylir::MemoryAccess* lastDef)
{
    std::unique_ptr<pylir::MemoryAccess> access;
    auto memoryEffectOpInterface = mlir::dyn_cast<mlir::MemoryEffectOpInterface>(operation);
    if (memoryEffectOpInterface)
    {
        if (memoryEffectOpInterface.hasNoEffect())
        {
            return {};
        }
        if (memoryEffectOpInterface.hasEffect<mlir::MemoryEffects::Write>())
        {
            access = std::make_unique<pylir::MemoryDef>(operation, lastDef);
        }
        else if (memoryEffectOpInterface.hasEffect<mlir::MemoryEffects::Read>())
        {
            access = std::make_unique<pylir::MemoryUse>(operation, lastDef, mlir::AliasResult::MayAlias);
        }
    }
    // Already identified as a Def
    if (mlir::isa_and_nonnull<pylir::MemoryDef>(access.get()))
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
        return std::make_unique<pylir::MemoryDef>(operation, lastDef);
    }
    return access;
}

} // namespace

pylir::MemorySSA::MemorySSA(mlir::Operation* operation, mlir::AnalysisManager& analysisManager)
{
    auto& aliasAnalysis = analysisManager.getAnalysis<mlir::AliasAnalysis>();
    for (auto& region : operation->getRegions())
    {
        MemoryAccess* lastDef = nullptr;
        for (auto& block : region)
        {
            for (auto& op : block)
            {
                auto result = maybeAddAccess(*this, &op, lastDef);
                if (auto* def = mlir::dyn_cast_or_null<pylir::MemoryDef>(result.get()))
                {
                    lastDef = def;
                }
                if (result)
                {
                    m_results.insert({&op, std::move(result)});
                }
            }
        }
    }
}
