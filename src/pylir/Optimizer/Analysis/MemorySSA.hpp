#pragma once

#include <mlir/Analysis/AliasAnalysis.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/OwningOpRef.h>
#include <mlir/Pass/AnalysisManager.h>

#include <llvm/ADT/DenseMap.h>

#include <pylir/Support/Macros.hpp>

#include <memory>

#include "MemorySSAIR.hpp"

namespace pylir
{

class MemorySSA
{
    mlir::OwningOpRef<MemSSA::MemoryRegionOp> m_region;
    llvm::DenseMap<mlir::Operation*, mlir::Operation*> m_results;
    llvm::DenseMap<mlir::Block*, mlir::Block*> m_blockMapping;

    void createIR(mlir::Operation* operation);

    void optimizeUses(mlir::AnalysisManager& analysisManager);

public:
    explicit MemorySSA(mlir::Operation* operation, mlir::AnalysisManager& analysisManager);

    mlir::Operation* getMemoryAccess(mlir::Operation* operation);

    mlir::Region& getMemoryRegion()
    {
        return m_region->body();
    }

    void dump() const;

    void print(llvm::raw_ostream& out) const;

    friend llvm::raw_ostream& operator<<(llvm::raw_ostream& os, const MemorySSA& ssa)
    {
        ssa.print(os);
        return os;
    }
};

} // namespace pylir
