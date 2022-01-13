#include "MemorySSA.hpp"

#include <mlir/Interfaces/SideEffectInterfaces.h>

#include <mlir/IR/ImplicitLocOpBuilder.h>

#include <llvm/ADT/TypeSwitch.h>

#include <pylir/Optimizer/Interfaces/CaptureInterface.hpp>

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
    m_region = builder.create<MemSSA::MemoryRegionOp>();
    builder.setInsertionPointToStart(&m_region->body().emplaceBlock());
    for (auto& region : operation->getRegions())
    {
        mlir::Value lastDef;
        for (auto& block : region)
        {
            for (auto& op : block)
            {
                auto result = maybeAddAccess(builder, *this, &op, lastDef);
                if (auto def = mlir::dyn_cast_or_null<MemSSA::MemoryDefOp>(result))
                {
                    lastDef = def;
                }
                if (result)
                {
                    m_results.insert({&op, result});
                }
            }
        }
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
