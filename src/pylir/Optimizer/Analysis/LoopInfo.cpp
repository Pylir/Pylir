//  Licensed under the Apache License v2.0 with LLVM Exceptions.
//  See https://llvm.org/LICENSE.txt for license information.
//  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "LoopInfo.hpp"

#include <mlir/IR/Dominance.h>
#include <mlir/IR/RegionGraphTraits.h>
#include <mlir/Pass/AnalysisManager.h>

#include <llvm/ADT/PostOrderIterator.h>

void pylir::LoopInfo::discoverAndMapSubLoops(
    pylir::Loop* loop, llvm::ArrayRef<mlir::Block*> latches,
    mlir::DominanceInfo& domInfo) {
  std::size_t subLoopCount = 0;
  std::size_t blockCount = 0;

  // Traversing through the CFG backwards via a worklist.
  std::vector<mlir::Block*> workList(latches.begin(), latches.end());
  while (!workList.empty()) {
    auto* block = workList.back();
    workList.pop_back();

    auto* subLoop = m_mapping.lookup(block);
    if (!subLoop) {
      if (!domInfo.isReachableFromEntry(block))
        continue;

      // The block has not yet been assigned to a loop
      m_mapping[block] = loop;
      blockCount++;
      if (block == loop->getHeader()) {
        // We have reached the header and hence no more predecessors are part of
        // the loop. Since the header per definition dominates the whole loop,
        // this is inevitable.
        continue;
      }
      workList.insert(workList.end(), block->pred_begin(), block->pred_end());
      continue;
    }

    // We discovered a loop within our loop. Change the surrounding/most top
    // level loop in the hierarchy to be this newly discovered one.
    while (auto* parent = subLoop->getParentLoop())
      subLoop = parent;

    if (subLoop == loop) {
      // Already the topmost one
      continue;
    }

    subLoopCount++;
    blockCount = subLoop->m_blocks.capacity();
    subLoop->m_parentLoop = loop;
    for (auto* pred : subLoop->getHeader()->getPredecessors()) {
      if (m_mapping.lookup(pred) == subLoop) {
        // Ignore backedges, they're already discovered
        continue;
      }
      workList.push_back(pred);
    }
  }
  loop->m_subLoops.reserve(subLoopCount);
  loop->m_blocks.reserve(blockCount);
}

pylir::LoopInfo::LoopInfo(mlir::Operation* operation,
                          mlir::AnalysisManager& analysisManager) {
  auto& region = operation->getRegion(0);
  // DominanceInfo does not allow fetching a dominator tree on a single block
  // region. The entry block of a region can also not be the successor of any
  // other block, that means including itself, hence there are no loops in such
  // a region.
  if (region.hasOneBlock())
    return;

  auto& domInfo = analysisManager.getAnalysis<mlir::DominanceInfo>();
  for (const auto& iter : llvm::post_order(domInfo.getRootNode(&region))) {
    auto* maybeHeader = iter->getBlock();
    llvm::SmallVector<mlir::Block*> latches;
    for (auto* latch : maybeHeader->getPredecessors())
      if (domInfo.dominates(maybeHeader, latch) &&
          domInfo.isReachableFromEntry(latch))
        latches.push_back(latch);

    if (latches.empty())
      continue;

    auto* loop = allocateLoop(maybeHeader);
    discoverAndMapSubLoops(loop, latches, domInfo);
  }

  for (auto* iter : llvm::post_order(&region.front())) {
    auto* loop = m_mapping.lookup(iter);
    if (!loop)
      continue;

    if (loop->getHeader() == iter) {
      if (loop->getParentLoop())
        loop->getParentLoop()->m_subLoops.push_back(loop);
      else
        m_topLevelLoops.push_back(loop);

      // For convenience, especially when printing, inverse the order to get
      // from a post order to a reverse post order
      std::reverse(loop->m_subLoops.begin(), loop->m_subLoops.end());
      // Loop header at first index is already in the right position
      std::reverse(loop->m_blocks.begin() + 1, loop->m_blocks.end());

      // The header does not need to be added to the block set/list of the
      // loop as that is already done in the constructor of loop
      loop = loop->getParentLoop();
    }
    for (; loop; loop = loop->getParentLoop()) {
      loop->m_blocks.push_back(iter);
      loop->m_blockSet.insert(iter);
    }
  }
}

void pylir::Loop::print(llvm::raw_ostream& os, unsigned int depth) const {
  os.indent(depth * 2);
  os << "Loop at depth " << getLoopDepth() << " containing: ";
  bool first = true;
  auto* header = getHeader();
  for (auto* iter : m_blocks) {
    if (first)
      first = false;
    else
      os << ", ";

    iter->printAsOperand(os);

    if (iter == header)
      os << "<header>";
  }

  os << "\n";
  for (auto* iter : m_subLoops)
    iter->print(os, depth + 2);
}

void pylir::LoopInfo::print(llvm::raw_ostream& os) const {
  for (auto* iter : m_topLevelLoops)
    iter->print(os, 0);
}

void pylir::LoopInfo::dump() const {
  print(llvm::outs());
}
