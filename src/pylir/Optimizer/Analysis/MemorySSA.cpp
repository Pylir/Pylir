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

namespace {
void maybeAddAccess(
    mlir::ImplicitLocOpBuilder& builder, mlir::Operation* operation,
    pylir::SSABuilder& ssaBuilder,
    llvm::MapVector<mlir::SideEffects::Resource*,
                    pylir::SSABuilder::DefinitionsMap>& lastDefs) {
  using namespace pylir::MemSSA;
  llvm::SmallVector<mlir::MemoryEffects::EffectInstance> effects;
  auto memoryEffectOpInterface =
      mlir::dyn_cast<mlir::MemoryEffectOpInterface>(operation);
  if (memoryEffectOpInterface)
    memoryEffectOpInterface.getEffects(effects);

  if (!memoryEffectOpInterface &&
      operation->hasTrait<mlir::OpTrait::HasRecursiveMemoryEffects>()) {
    // Ops with recursive side effects contain regions with side effects which
    // will be inlined later anyway. Nothing to do here.
    return;
  }

  struct ReadWrite {
    llvm::SetVector<llvm::PointerUnion<mlir::Value, mlir::SymbolRefAttr>> read,
        written;
  };

  llvm::MapVector<mlir::SideEffects::Resource*, ReadWrite> resourceToEffect;
  for (auto& iter : effects) {
    if (!llvm::isa<mlir::MemoryEffects::Write, mlir::MemoryEffects::Read>(
            iter.getEffect()))
      continue;

    auto& sets = resourceToEffect[iter.getResource()];
    decltype(sets.read)* set = nullptr;
    if (llvm::isa<mlir::MemoryEffects::Write>(iter.getEffect()))
      set = &sets.written;
    else if (llvm::isa<mlir::MemoryEffects::Read>(iter.getEffect()))
      set = &sets.read;

    if (iter.getValue())
      set->insert(iter.getValue());
    else if (iter.getSymbolRef())
      set->insert(iter.getSymbolRef());
    else
      set->insert(nullptr);
  }

  auto getLastDef = [&](mlir::SideEffects::Resource* resource) {
    // If this is the first use of a resource we need to read from the 'nullptr'
    // resource. The 'nullptr' resource is a special sentinel which contains all
    // 'clobber all' operations. Any resources that were not yet known at the
    // time of the clobber all have to do their first use from the 'clobber all'
    // operation. If we didn't special case this, it'd read from 'liveOnEntry'
    // which would technically be incorrect.
    // TODO: Figure out whether this matters. Gut feeling says no.
    auto* res = lastDefs.find(resource);
    if (res == lastDefs.end())
      res =
          lastDefs.insert({nullptr, pylir::SSABuilder::DefinitionsMap{}}).first;

    return ssaBuilder.readVariable(builder.getLoc(), builder.getType<DefType>(),
                                   res->second, builder.getBlock());
  };

  if (resourceToEffect.empty()) {
    if (memoryEffectOpInterface)
      return;

    // This is the conservative case that may happen if an op does not
    // implement 'MemoryEffectOpInterface'. In this case we do a memory def
    // for every resource and specify that all memory is clobbered and all
    // memory was also read. Also writes to the 'clobber all' resource aka
    // 'nullptr' resource for any later occurring new resources.
    bool nullptrSeen = false;
    std::array<mlir::SideEffects::Resource*, 1> array{};
    for (auto* iter1 : llvm::concat<mlir::SideEffects::Resource*>(
             llvm::to_vector(llvm::make_first_range(lastDefs)), array)) {
      if (!iter1) {
        // nullptr may also occur as key in 'lastDefs' already. The 'array'
        // that we concat here is just to be safe in the case 'nullptr'
        // resource has not yet been used. Don't want two defs of 'nullptr'
        // either however, hence this logic.
        if (nullptrSeen)
          continue;

        nullptrSeen = true;
      }

      auto lastDef1 = getLastDef(iter1);
      // Not specifying a memory location here simply means "conservatively
      // assume everything was read/written".
      auto clobberAll = builder.create<MemoryDefOp>(
          lastDef1, operation,
          llvm::PointerUnion<mlir::Value, mlir::SymbolRefAttr>{},
          llvm::PointerUnion<mlir::Value, mlir::SymbolRefAttr>{});
      lastDefs[iter1][builder.getBlock()] = clobberAll;
    }
    return;
  }

  for (auto& [resource, readWrites] : resourceToEffect) {
    auto lastDef = getLastDef(resource);
    if (!readWrites.written.empty()) {
      auto write = builder.create<MemoryDefOp>(lastDef, operation,
                                               readWrites.written.takeVector(),
                                               readWrites.read.takeVector());
      lastDefs[resource][builder.getBlock()] = write;
      continue;
    }

    PYLIR_ASSERT(!readWrites.read.empty());
    builder.create<MemoryUseOp>(lastDef, operation,
                                readWrites.read.takeVector());
  }
}

} // namespace

void pylir::MemorySSA::fillRegion(
    mlir::Region& region, mlir::ImplicitLocOpBuilder& builder,
    SSABuilder& ssaBuilder,
    llvm::MapVector<mlir::SideEffects::Resource*, SSABuilder::DefinitionsMap>&
        lastDefs,
    llvm::ArrayRef<mlir::Block*> regionSuccessors) {
  auto hasUnresolvedPredecessors = [&](mlir::Block* block) {
    return llvm::any_of(block->getPredecessors(), [&](mlir::Block* pred) {
      auto* predMemBlock = m_blockMapping.lookup(pred);
      if (!predMemBlock)
        return true;

      return !predMemBlock->getParent();
    });
  };

  for (auto& block : region) {
    mlir::Block* memBlock;
    {
      auto [lookup, inserted] = m_blockMapping.insert({&block, nullptr});
      if (inserted)
        lookup->second = new mlir::Block;

      memBlock = lookup->second;
    }
    // If any of the predecessors have not yet been inserted
    // mark the block as open
    if (hasUnresolvedPredecessors(&block))
      ssaBuilder.markOpenBlock(memBlock);

    getMemoryRegion().push_back(memBlock);
    builder.setInsertionPointToStart(memBlock);

    for (auto& op : block) {
      maybeAddAccess(builder, &op, ssaBuilder, lastDefs);
      auto regionBranchOp = mlir::dyn_cast<mlir::RegionBranchOpInterface>(&op);
      if (!regionBranchOp)
        continue;

      llvm::SmallVector<mlir::RegionSuccessor> successors;
      regionBranchOp.getSuccessorRegions(mlir::RegionBranchPoint::parent(),
                                         successors);
      PYLIR_ASSERT(!successors.empty());
      auto* continueRegion = new mlir::Block;

      llvm::SmallVector<mlir::Block*> regionEntries;

      auto getRegionSuccBlocks =
          [&](const llvm::SmallVector<mlir::RegionSuccessor>& successors) {
            return llvm::to_vector(llvm::map_range(
                successors, [&](const mlir::RegionSuccessor& succ) {
                  if (succ.isParent())
                    return continueRegion;

                  auto result = m_blockMapping.insert(
                      {&succ.getSuccessor()->front(), nullptr});
                  if (result.second) {
                    result.first->second = new mlir::Block;
                    // We have to conservatively mark region entries as open
                    // simply due to having absolutely no way to tell whether
                    // all their predecessors have been visited, due not being
                    // able to get a list of all predecessors! We only get
                    // successors of a region once we are filling it in. After
                    // we are done filling in all regions of the operation
                    // however we can be sure all their predecessors have been
                    // filled
                    ssaBuilder.markOpenBlock(result.first->second);
                    regionEntries.push_back(result.first->second);
                  }
                  return result.first->second;
                }));
          };

      auto entryBlocks = getRegionSuccBlocks(successors);
      builder.create<pylir::MemSSA::MemoryBranchOp>(
          llvm::SmallVector<mlir::ValueRange>(entryBlocks.size()), entryBlocks);

      llvm::SmallVector<mlir::Region*> workList;

      auto fillWorkList =
          [&](const llvm::SmallVector<mlir::RegionSuccessor>& successors) {
            for (const auto& iter : llvm::reverse(successors))
              if (!iter.isParent())
                workList.push_back(iter.getSuccessor());
          };

      fillWorkList(successors);

      llvm::DenseSet<mlir::Region*> seen;
      while (!workList.empty()) {
        auto* succ = workList.pop_back_val();
        if (!seen.insert(succ).second)
          continue;

        llvm::SmallVector<mlir::RegionSuccessor> subRegionSuccessors;
        regionBranchOp.getSuccessorRegions(succ, subRegionSuccessors);
        fillRegion(*succ, builder, ssaBuilder, lastDefs,
                   getRegionSuccBlocks(subRegionSuccessors));
        fillWorkList(subRegionSuccessors);
      }
      for (auto& iter : regionEntries)
        ssaBuilder.sealBlock(iter);

      getMemoryRegion().push_back(continueRegion);
      builder.setInsertionPointToStart(continueRegion);
      memBlock = continueRegion;
    }

    if (block.getTerminator()->hasTrait<mlir::OpTrait::ReturnLike>()) {
      builder.create<pylir::MemSSA::MemoryBranchOp>(
          llvm::SmallVector<mlir::ValueRange>(regionSuccessors.size()),
          regionSuccessors);
      continue;
    }

    llvm::SmallVector<mlir::Block*> memSuccessors;
    llvm::SmallVector<mlir::Block*> sealAfter;
    for (auto* succ : block.getSuccessors()) {
      auto [lookup, inserted] = m_blockMapping.insert({succ, nullptr});
      if (inserted) {
        lookup->second = new mlir::Block;
      } else if (lookup->second->getParent()) {
        // This particular successor seems to have already been filled
        // Check whether filling this block has made all of its predecessors
        // filled and seal it
        if (!hasUnresolvedPredecessors(succ))
          sealAfter.push_back(lookup->second);
      }
      memSuccessors.push_back(lookup->second);
    }
    builder.create<pylir::MemSSA::MemoryBranchOp>(
        llvm::SmallVector<mlir::ValueRange>(memSuccessors.size()),
        memSuccessors);
    llvm::for_each(sealAfter,
                   [&](mlir::Block* lookup) { ssaBuilder.sealBlock(lookup); });
  }
}

void pylir::MemorySSA::createIR(mlir::Operation* operation) {
  mlir::ImplicitLocOpBuilder builder(
      mlir::UnknownLoc::get(operation->getContext()), operation->getContext());
  m_region = builder.create<MemSSA::MemoryModuleOp>();
  PYLIR_ASSERT(operation->getNumRegions() == 1);
  auto& region = operation->getRegion(0);
  if (region.empty()) {
    auto* block = new mlir::Block;
    m_region->getBody().push_back(block);
    builder.setInsertionPointToStart(block);
    builder.create<pylir::MemSSA::MemoryLiveOnEntryOp>();
    builder.create<pylir::MemSSA::MemoryBranchOp>(
        llvm::ArrayRef<mlir::ValueRange>{}, mlir::BlockRange{});
    return;
  }
  pylir::SSABuilder ssaBuilder(
      [&](mlir::Block*, mlir::Type, mlir::Location) -> mlir::Value {
        if (!m_region->getBody().front().empty())
          if (auto op = mlir::dyn_cast<MemSSA::MemoryLiveOnEntryOp>(
                  m_region->getBody().front().front()))
            return op;

        auto exit = mlir::OpBuilder::InsertionGuard{builder};
        builder.setInsertionPointToStart(&m_region->getBody().front());
        return builder.create<MemSSA::MemoryLiveOnEntryOp>();
      });
  llvm::MapVector<mlir::SideEffects::Resource*, SSABuilder::DefinitionsMap>
      lastDefs;

  // Insert entry block that has no predecessors
  m_blockMapping.insert({&region.getBlocks().front(), new mlir::Block});
  fillRegion(region, builder, ssaBuilder, lastDefs, {});
}

namespace {
mlir::Value getLastClobber(pylir::MemSSA::MemoryUseOp use,
                           mlir::AliasAnalysis& aliasAnalysis) {
  // TODO: Implement optimizations for block args.
  mlir::Value def = use.getDefinition();
  for (; def.getDefiningOp<pylir::MemSSA::MemoryDefOp>();
       def = def.getDefiningOp<pylir::MemSSA::MemoryDefOp>().getClobbered()) {
    auto memDef = def.getDefiningOp<pylir::MemSSA::MemoryDefOp>();
    if (llvm::any_of(
            use.getReads(),
            [&](llvm::PointerUnion<mlir::Value, mlir::SymbolRefAttr> ptr) {
              // If no affected location is specified, conservatively assume it
              // reads the def.
              if (!ptr)
                return true;

              if (auto val = mlir::dyn_cast<mlir::Value>(ptr))
                return aliasAnalysis.getModRef(memDef.getInstruction(), val)
                    .isMod();

              // There is no support for symbol ref attrs in MLIRs alias
              // analysis. For the time being we assume a symbol ref is always
              // clobbered except if some other symbol is being written to.
              // Technically speaking, this assumption could not hold if there
              // was some kind of global symbol alias mechanism/op, but we just
              // assume such a thing does not exist for now.
              return llvm::any_of(
                  memDef.getWrites(),
                  [ptr](llvm::PointerUnion<mlir::Value, mlir::SymbolRefAttr>
                            write) {
                    if (llvm::isa<mlir::SymbolRefAttr>(write))
                      return write == ptr;
                    return true;
                  });
            })) {
      return memDef;
    }
  }
  return def;
}

} // namespace

void pylir::MemorySSA::optimizeUses(mlir::AnalysisManager& analysisManager) {
  auto& aliasAnalysis = analysisManager.getAnalysis<mlir::AliasAnalysis>();
  m_region->walk([&](MemSSA::MemoryUseOp use) {
    use.getDefinitionMutable().assign(getLastClobber(use, aliasAnalysis));
  });
}

pylir::MemorySSA::MemorySSA(mlir::Operation* operation,
                            mlir::AnalysisManager& analysisManager) {
  createIR(operation);
  optimizeUses(analysisManager);
}

void pylir::MemorySSA::dump() const {
  m_region.get()->dump();
}

void pylir::MemorySSA::print(llvm::raw_ostream& out) const {
  m_region.get().print(out);
}
