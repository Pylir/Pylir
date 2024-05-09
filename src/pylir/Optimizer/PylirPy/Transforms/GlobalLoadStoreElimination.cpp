//  Licensed under the Apache License v2.0 with LLVM Exceptions.
//  See https://llvm.org/LICENSE.txt for license information.
//  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <mlir/IR/Dominance.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>
#include <mlir/Pass/Pass.h>

#include <llvm/ADT/ScopeExit.h>

#include <pylir/Optimizer/PylirPy/IR/PylirPyOps.hpp>
#include <pylir/Optimizer/Transforms/Util/SSABuilder.hpp>
#include <pylir/Optimizer/Transforms/Util/SSAUpdater.hpp>

#include "Passes.hpp"

namespace pylir::Py {
#define GEN_PASS_DEF_GLOBALLOADSTOREELIMINATIONPASS
#include "pylir/Optimizer/PylirPy/Transforms/Passes.h.inc"
} // namespace pylir::Py

namespace {

struct BlockData {
  std::vector<std::pair<pylir::Py::LoadOp, pylir::ValueTracker>> candidates;
};

struct GlobalLoadStoreEliminationPass
    : pylir::Py::impl::GlobalLoadStoreEliminationPassBase<
          GlobalLoadStoreEliminationPass> {
  using Base::Base;

protected:
  void runOnOperation() override;

private:
  bool optimizeBlock(
      mlir::Block& block, mlir::Value clobberTracker, BlockData& blockArgUsages,
      llvm::DenseMap<mlir::SymbolRefAttr, pylir::SSABuilder::DefinitionsMap>&
          definitions,
      pylir::SSABuilder& ssaBuilder);
};

void GlobalLoadStoreEliminationPass::runOnOperation() {
  bool changed = false;
  auto* topLevel = getOperation();
  mlir::OwningOpRef<mlir::UnrealizedConversionCastOp> clobberTracker;
  {
    mlir::OpBuilder builder(&getContext());
    clobberTracker = builder.create<mlir::UnrealizedConversionCastOp>(
        builder.getUnknownLoc(), builder.getType<pylir::Py::DynamicType>(),
        mlir::ValueRange{});
  }
  auto clobberValue = clobberTracker->getResult(0);

  for (auto& region : topLevel->getRegions()) {
    BlockData blockArgUsages;
    llvm::DenseMap<mlir::SymbolRefAttr, pylir::SSABuilder::DefinitionsMap>
        definitions;
    pylir::SSABuilder ssaBuilder(
        [clobberValue](auto&&...) -> mlir::Value { return clobberValue; },
        [clobberValue](mlir::Value lhs, mlir::Value rhs) -> mlir::Value {
          if (llvm::is_contained({lhs, rhs}, clobberValue))
            return clobberValue;

          return nullptr;
        });

    pylir::updateSSAinRegion(ssaBuilder, region, [&](mlir::Block* block) {
      changed |= optimizeBlock(*block, clobberValue, blockArgUsages,
                               definitions, ssaBuilder);
    });

    for (auto& [load, tracker] : blockArgUsages.candidates) {
      mlir::Value value = tracker;
      if (value == clobberValue)
        continue;

      // Simplification has lead to the block argument being simplified to a
      // single value which is not a clobber
      load.replaceAllUsesWith(value);
      load->erase();
      changed = true;
    }
  }
  if (!changed) {
    markAllAnalysesPreserved();
    return;
  }
  markAnalysesPreserved<mlir::DominanceInfo>();
}

bool GlobalLoadStoreEliminationPass::optimizeBlock(
    mlir::Block& block, mlir::Value clobberTracker, BlockData& blockArgUsages,
    llvm::DenseMap<mlir::SymbolRefAttr, pylir::SSABuilder::DefinitionsMap>&
        definitions,
    pylir::SSABuilder& ssaBuilder) {
  bool changed = false;
  llvm::DenseMap<mlir::SymbolRefAttr, pylir::Py::StoreOp> unusedStores;
  for (auto& op : llvm::make_early_inc_range(block)) {
    if (auto load = mlir::dyn_cast<pylir::Py::LoadOp>(op)) {
      unusedStores.erase(load.getGlobalAttr());
      auto& map = definitions[load.getGlobalAttr()];
      auto read =
          ssaBuilder.readVariable(load->getLoc(), load.getType(), map, &block);
      if (read == clobberTracker) {
        // Make a definition to avoid reloads
        map[&block] = load;
        continue;
      }
      // If it's a block arg we can't immediately replace it as it might still
      // be clobbered from one of the predecessor branches. We record these for
      // now and instead handle them after having traversed every block. Also
      // making a definition in case this load does remain, so that there are no
      // reloads.
      if (mlir::isa<mlir::BlockArgument>(read)) {
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
    if (auto store = mlir::dyn_cast<pylir::Py::StoreOp>(op)) {
      auto [iter, inserted] =
          unusedStores.insert({store.getGlobalAttr(), store});
      if (!inserted) {
        m_storesRemoved++;
        iter->second->erase();
        iter->second = store;
        changed = true;
      }
      definitions[store.getGlobalAttr()][&block] = store.getValue();
      continue;
    }
    auto mem = mlir::dyn_cast<mlir::MemoryEffectOpInterface>(op);
    llvm::SmallVector<mlir::MemoryEffects::EffectInstance> effects;
    if (mem) {
      mem.getEffectsOnResource(pylir::Py::GlobalResource::get(), effects);
    }
    bool mayWrite = false;
    bool mayRead = false;
    for (auto& effect : effects) {
      if (mlir::isa<mlir::MemoryEffects::Write>(effect.getEffect()))
        mayWrite = true;

      if (mlir::isa<mlir::MemoryEffects::Read>(effect.getEffect()))
        mayRead = true;
    }
    if (!mem || mayWrite) {
      // Conservatively assume all globals were written
      for (auto& [key, value] : definitions)
        value[&block] = clobberTracker;
    }
    if (!mem || mayRead) {
      // Conservatively assume all globals were read
      unusedStores.clear();
    }
  }
  return changed;
}

} // namespace
