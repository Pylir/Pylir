//  Licensed under the Apache License v2.0 with LLVM Exceptions.
//  See https://llvm.org/LICENSE.txt for license information.
//  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <mlir/IR/Dominance.h>

#include <llvm/ADT/DepthFirstIterator.h>

#include <pylir/Optimizer/Analysis/MemorySSA.hpp>
#include <pylir/Optimizer/Interfaces/MemoryFoldInterface.hpp>

#include "Passes.hpp"

namespace pylir {
#define GEN_PASS_DEF_LOADFORWARDINGPASS
#include "pylir/Optimizer/Transforms/Passes.h.inc"
} // namespace pylir

namespace {

struct LoadForwardingPass
    : pylir::impl::LoadForwardingPassBase<LoadForwardingPass> {
protected:
  void runOnOperation() override;

public:
  using Base::Base;
};

void LoadForwardingPass::runOnOperation() {
  auto& memorySSA = getAnalysisManager().getAnalysis<pylir::MemorySSA>();
  auto& aliasAnalysis = getAnalysisManager().getAnalysis<mlir::AliasAnalysis>();
  bool changed = false;

  memorySSA.getMemoryRegion().walk([&](pylir::MemSSA::MemoryUseOp use) {
    auto memoryFold =
        mlir::dyn_cast<pylir::MemoryFoldInterface>(use.getInstruction());
    if (!memoryFold)
      return;

    auto defOp =
        use.getDefinition().getDefiningOp<pylir::MemSSA::MemoryDefOp>();
    if (!defOp)
      return;

    // For every read, check that each is defined by the memDef. In other words,
    // that the memDef definitely writes to each of them (potentially via
    // aliasing effects).
    // TODO: It doesn't have to be necessary that every read is defined, just
    // that at least one is. This would
    //       require communicating to the MemoryFoldInterface which read
    //       definitely writes from given def. Implement this once an op
    //       requires it.
    // TODO: If there is ever an op with multiple writes it'll likely be
    // necessary to communicate to
    //       MemoryFoldInterface, which write affected a read value.
    for (const auto& read : use.getReads()) {
      if (!read) {
        // If the read is just a general read, not to a specific location, there
        // is absolutely no way to figure out whether it definitely reads from
        // the def.
        return;
      }
      if (!llvm::any_of(
              defOp.getWrites(),
              [&](llvm::PointerUnion<mlir::Value, mlir::SymbolRefAttr> ptr) {
                if (!ptr)
                  return false;

                if (auto redVal = llvm::dyn_cast<mlir::Value>(read)) {
                  auto writeVal = llvm::dyn_cast<mlir::Value>(ptr);
                  if (!writeVal)
                    return false;

                  return aliasAnalysis.alias(redVal, writeVal).isMust();
                }
                return llvm::isa<mlir::SymbolRefAttr>(ptr) && ptr == read;
              })) {
        return;
      }
    }

    llvm::SmallVector<mlir::OpFoldResult> results;
    if (mlir::failed(memoryFold.foldUsage(defOp.getInstruction(), results)))
      return;

    changed = true;
    for (auto [foldResult, opResult] :
         llvm::zip(results, memoryFold->getResults())) {
      if (auto value = mlir::dyn_cast<mlir::Value>(foldResult)) {
        opResult.replaceAllUsesWith(value);
        m_localLoadsReplaced++;
      } else if (auto attr = mlir::dyn_cast<mlir::Attribute>(foldResult)) {
        mlir::OpBuilder builder(memoryFold);
        auto* constant = memoryFold->getDialect()->materializeConstant(
            builder, attr, opResult.getType(), memoryFold->getLoc());
        PYLIR_ASSERT(constant);
        opResult.replaceAllUsesWith(constant->getResult(0));
        m_localLoadsReplaced++;
      }
    }
    if (mlir::isOpTriviallyDead(memoryFold)) {
      memoryFold->erase();
      use.erase();
    }
  });

  if (!changed) {
    markAllAnalysesPreserved();
    return;
  }
  markAnalysesPreserved<mlir::DominanceInfo, pylir::MemorySSA>();
}

} // namespace
