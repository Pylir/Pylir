//  Licensed under the Apache License v2.0 with LLVM Exceptions.
//  See https://llvm.org/LICENSE.txt for license information.
//  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <mlir/IR/Matchers.h>
#include <mlir/IR/RegionGraphTraits.h>

#include <llvm/ADT/DepthFirstIterator.h>
#include <llvm/ADT/SetOperations.h>
#include <llvm/ADT/SetVector.h>
#include <llvm/Support/Debug.h>

#include <pylir/Optimizer/Interfaces/ConditionalBranchInterface.hpp>
#include <pylir/Optimizer/Interfaces/DialectImplicationPatternsInterface.hpp>
#include <pylir/Support/Functional.hpp>
#include <pylir/Support/Macros.hpp>

#define DEBUG_TYPE "pylir-conditionals-implications"

#include "Passes.hpp"

namespace pylir {
#define GEN_PASS_DEF_CONDITIONALSIMPLICATIONSPASS
#include "pylir/Optimizer/Transforms/Passes.h.inc"
} // namespace pylir

namespace {
class ConditionalsImplicationsPass final
    : public pylir::impl::ConditionalsImplicationsPassBase<
          ConditionalsImplicationsPass> {
  std::optional<mlir::DialectInterfaceCollection<
      pylir::DialectImplicationPatternsInterface>>
      m_interfaces;

  bool simplifyRegion(mlir::Region& region);

public:
  using Base::Base;

protected:
  void runOnOperation() override;

  mlir::LogicalResult initialize(mlir::MLIRContext* context) override {
    m_interfaces.emplace(context);
    return mlir::success();
  }
};

void ConditionalsImplicationsPass::runOnOperation() {
  bool changed = false;
  for (mlir::Region& region : getOperation()->getRegions())
    changed = changed | simplifyRegion(region);

  if (!changed)
    markAllAnalysesPreserved();
}

/// Returns all back-edges within the regions CFG as "from -> to" pairs.
llvm::DenseSet<std::pair<mlir::Block*, mlir::Block*>>
collectBackedges(mlir::Region& region) {
  class DFSState : public llvm::SmallPtrSet<mlir::Block*, 8> {
    llvm::SmallPtrSet<mlir::Block*, 8> m_completed;

  public:
    void completed(mlir::Block* block) {
      m_completed.insert(block);
    }

    [[nodiscard]] bool isCompleted(mlir::Block* block) const {
      return m_completed.contains(block);
    }
  };

  DFSState state;
  llvm::DenseSet<std::pair<mlir::Block*, mlir::Block*>> backedges;
  for (auto iter = llvm::df_ext_begin(&region, state),
            end = llvm::df_ext_end(&region, state);
       iter != end; ++iter) {
    for (mlir::Block* succ : iter->getSuccessors()) {
      // A back edge within a DFS walk is simply an edge to a block whose
      // children have not yet all been visited by the DFS walk. LLVMs
      // abstraction calls this a "not completed" block.
      if (state.contains(succ) && !state.isCompleted(succ))
        backedges.insert({*iter, succ});
    }
  }

  return backedges;
}

/// Custom set, orchestrating when a block should be visited by LLVMs DFS walk.
/// More specifically, it modifies the DFS walk order to turn into a
/// topologically sorted order. Since the CFG may contain loops however, it
/// simply ignores these loops by pretending back-edges do not exist, therefore
/// once again forming a perfect DAG. This leads to no correctness issues in
/// this pass, just potentially missed optimizations.
class TopoSortState : public llvm::SmallPtrSet<mlir::Block*, 8> {
  using Base = llvm::SmallPtrSet<mlir::Block*, 8>;

  llvm::DenseSet<std::pair<mlir::Block*, mlir::Block*>> m_backEdges;

  bool shouldVisit(mlir::Block* block) {
    if (block->getSinglePredecessor())
      return true;

    return llvm::all_of(block->getPredecessors(), [&](mlir::Block* pred) {
      return contains(pred) || m_backEdges.contains({pred, block});
    });
  }

public:
  using iterator = Base::iterator;

  explicit TopoSortState(mlir::Region& region)
      : m_backEdges(collectBackedges(region)) {}

  void completed(mlir::Block*) {}

  std::pair<iterator, bool> insert(mlir::Block* block) {
    if (!shouldVisit(block))
      return {end(), false};

    return Base::insert(block);
  }

  [[nodiscard]] bool isBackedge(mlir::Block* from, mlir::Block* to) const {
    return m_backEdges.contains({from, to});
  }
};

void removeSameBlockSuccessors(
    llvm::SmallVectorImpl<std::pair<mlir::Block*, mlir::Attribute>>&
        implications) {
  if (implications.empty())
    return;

  llvm::sort(implications,
             [](auto& lhs, auto& rhs) { return lhs.first < rhs.first; });

  // Delete all entries for a block, if the block appears more than once. Since
  // the array is rather sorted this is a rather simple state machine where we
  // switch to "delete previous" as soon as the previous element has the same
  // block as the current.
  bool mustDeleteAll = false;
  for (auto* iter = std::next(implications.begin()); iter != implications.end();
       iter++) {
    if (iter->first == std::prev(iter)->first) {
      mustDeleteAll = true;
      iter = implications.erase(std::prev(iter));
    } else if (mustDeleteAll) {
      mustDeleteAll = false;
      iter = implications.erase(std::prev(iter));
    }
  }
  if (mustDeleteAll)
    implications.erase(std::prev(implications.end()));
}

pylir::ImplicationPatternBase*
replaceUseWithConstantPattern(mlir::Value condition, mlir::Attribute value,
                              pylir::PatternAllocator& allocator) {
  struct UseRewriter final : pylir::ImplicationPatternBase {
    mlir::Value condition;
    mlir::Attribute attribute;

    UseRewriter(mlir::Value condition, mlir::Attribute attribute)
        : condition(condition), attribute(attribute) {}

    mlir::LogicalResult
    matchAndRewrite(mlir::Operation* operation,
                    mlir::OpBuilder& builder) const override {
      bool changed = false;
      mlir::Value replacement;
      for (mlir::OpOperand& operand : operation->getOpOperands()) {
        if (operand.get() != condition)
          continue;

        if (!changed) {
          changed = true;
          for (mlir::Dialect* dialect :
               operation->getContext()->getLoadedDialects()) {
            mlir::Operation* constantOp = dialect->materializeConstant(
                builder, attribute, operand.get().getType(),
                operand.get().getLoc());
            if (!constantOp)
              continue;

            replacement = constantOp->getResult(0);
            break;
          }
          PYLIR_ASSERT(
              replacement &&
              "No loaded dialect found to materialize the given constant");
        }
        operand.set(replacement);
      }
      return mlir::success(changed);
    }
  };
  return allocator.allocate<UseRewriter>(condition, value);
}

// Using SetVector since we iterate over all patterns and want deterministic
// iteration order.
using FactList = llvm::SetVector<pylir::ImplicationPatternBase*>;
// Maps "from->to" edge to list of facts propagated through that edge.
using FactMap = llvm::DenseMap<std::pair<mlir::Block*, mlir::Block*>, FactList>;

FactList getActiveFactsFromPredecessors(mlir::Block* block, FactMap& facts,
                                        const TopoSortState& topoSortState) {
  if (block->isEntryBlock())
    return {};

  llvm::SmallVector<FactList*> incomingFacts;
  for (mlir::Block* pred : block->getPredecessors()) {
    // Ignore back-edges. In regular loops, the loop header always dominates the
    // whole loop body, therefore any facts propagated from the header into the
    // loop remain true. Even in an irregular loop, the facts in all cycle entry
    // blocks would meet and not cause any miscompilations.
    if (topoSortState.isBackedge(pred, block))
      continue;

    auto result = facts.find({pred, block});
    if (result == facts.end() || result->second.empty()) {
      // Fast path. If a predecessor does not have any facts then the
      // intersection will always be empty.
      return {};
    }
    incomingFacts.push_back(&result->second);
  }

  // Moving is proper here since we only ever visit an edge once.
  FactList result = std::move(*incomingFacts.front());
  for (FactList* other : llvm::drop_begin(incomingFacts)) {
    for (const auto* iter = result.begin(); iter != result.end();) {
      if (!other->contains(*iter)) {
        iter = result.erase(iter);
        continue;
      }
      iter++;
    }
  }
  return result;
}

bool ConditionalsImplicationsPass::simplifyRegion(mlir::Region& region) {
  if (region.empty())
    return false;

  bool changed = false;

  pylir::PatternAllocator allocator;

  FactMap facts;

  TopoSortState topoSortState(region);
  for (mlir::Block* block : llvm::depth_first_ext(&region, topoSortState)) {
    LLVM_DEBUG({
      llvm::dbgs() << "Visiting ";
      block->printAsOperand(llvm::dbgs());
      llvm::dbgs() << '\n';
    });

    FactList factList =
        getActiveFactsFromPredecessors(block, facts, topoSortState);
    if (!factList.empty()) {
      for (mlir::Operation& iter : *block) {
        for (pylir::ImplicationPatternBase* pattern : factList) {
          mlir::OpBuilder builder(&iter);
          if (mlir::succeeded(pattern->matchAndRewrite(&iter, builder))) {
            m_simplificationsMade++;
            changed = true;
          }
        }
      }
    }

    for (mlir::Block* succ : block->getSuccessors())
      facts[{block, succ}].insert(factList.begin(), factList.end());

    auto interface = mlir::dyn_cast<pylir::ConditionalBranchInterface>(
        block->getTerminator());
    if (!interface)
      continue;

    mlir::Value startCondition = interface.getCondition();
    if (!startCondition)
      continue;

    llvm::SmallVector<std::pair<mlir::Block*, mlir::Attribute>> implications =
        interface.getBranchImplications();
    removeSameBlockSuccessors(implications);

    for (auto [branch, startValue] : implications) {
      auto& patternList = facts[{block, branch}];

      llvm::SmallVector<std::pair<mlir::Value, mlir::Attribute>> workList = {
          {startCondition, startValue}};

      while (!workList.empty()) {
        auto [condition, value] = workList.pop_back_val();

        for (const pylir::DialectImplicationPatternsInterface&
                 dialectInterface : *m_interfaces) {
          dialectInterface.getImplicationPatterns(
              allocator, condition, value,
              /*patternAddCallback=*/
              [&](pylir::ImplicationPatternBase* pattern) {
                patternList.insert(pattern);
              },
              /*implicationAddCallback=*/
              [&](mlir::Value impliedCondition, mlir::Attribute impliedValue) {
                workList.emplace_back(impliedCondition, impliedValue);
              });
        }
        patternList.insert(
            replaceUseWithConstantPattern(condition, value, allocator));
      }
    }
  }

  return changed;
}

} // namespace
