//  Licensed under the Apache License v2.0 with LLVM Exceptions.
//  See https://llvm.org/LICENSE.txt for license information.
//  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <llvm/ADT/DenseSet.h>
#include <llvm/ADT/STLExtras.h>

#include "SSABuilder.hpp"

namespace pylir {
template <class F>
void updateSSAinRegion(pylir::SSABuilder& ssaBuilder, mlir::Region& region,
                       F f) {
  llvm::DenseSet<mlir::Block*> seen;
  auto allPredSeen = [&](mlir::Block* block) {
    return llvm::all_of(block->getPredecessors(),
                        [&](mlir::Block* pred) { return seen.contains(pred); });
  };

  for (auto& block : region) {
    if (!allPredSeen(&block))
      ssaBuilder.markOpenBlock(&block);

    f(&block);
    seen.insert(&block);
    for (auto* succ : block.getSuccessors())
      if (ssaBuilder.isOpenBlock(succ) && allPredSeen(succ))
        ssaBuilder.sealBlock(succ);
  }
}
} // namespace pylir
