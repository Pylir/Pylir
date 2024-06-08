// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <mlir/IR/RegionGraphTraits.h>
#include <mlir/Pass/Pass.h>

#include <llvm/ADT/DepthFirstIterator.h>

#include <memory>

namespace pylir {
#define GEN_PASS_DEF_DEADCODEELIMINATIONPASS
#include "pylir/Optimizer/Transforms/Passes.h.inc"
} // namespace pylir

using namespace mlir;

namespace {
class DeadCodeEliminationPass final
    : public pylir::impl::DeadCodeEliminationPassBase<DeadCodeEliminationPass> {
public:
  using Base::Base;

protected:
  void runOnOperation() override;
};

} // namespace

void DeadCodeEliminationPass::runOnOperation() {
  bool changed = false;
  getOperation()->walk<WalkOrder::PreOrder>([&](Region* region) {
    if (region->empty())
      return;

    // Perform a DFS walk from the entry block giving us the set of alive basic
    // blocks. The set is filled up by simply iterating through the range.
    llvm::df_iterator_default_set<Block*> reachable;
    llvm::for_each(llvm::depth_first_ext(&region->front(), reachable),
                   [](auto&&) {});

    // Any blocks not reachable can be erased without consequences.
    for (Block& block : llvm::make_early_inc_range(region->getBlocks())) {
      if (reachable.contains(&block))
        continue;

      m_blocksRemoved++;
      block.dropAllDefinedValueUses();
      block.erase();
      changed = true;
    }
  });
  if (!changed)
    markAllAnalysesPreserved();
}
