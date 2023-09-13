//  Licensed under the Apache License v2.0 with LLVM Exceptions.
//  See https://llvm.org/LICENSE.txt for license information.
//  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <mlir/IR/Block.h>

#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/GraphTraits.h>
#include <llvm/ADT/SmallPtrSet.h>

#include <memory>
#include <vector>

namespace mlir {
class DominanceInfo;
class AnalysisManager;
} // namespace mlir

namespace pylir {

class Loop {
  /// Parent loop or null if this loop is not nested within any other loop.
  Loop* m_parentLoop = nullptr;
  /// All loops nested within this one.
  std::vector<Loop*> m_subLoops;
  /// List of blocks of this loop. First block is the header of the loop.
  std::vector<mlir::Block*> m_blocks;
  /// Used to query whether a block is contained within this loop.
  llvm::SmallPtrSet<mlir::Block*, 16> m_blockSet;

  friend class LoopInfo;

public:
  explicit Loop(mlir::Block* header) : m_blocks{header}, m_blockSet{header} {}

  ~Loop() {
    for (auto* iter : m_subLoops)
      iter->~Loop();
  }

  Loop(const Loop&) = delete;
  Loop(Loop&&) = delete;
  Loop& operator=(const Loop&) = delete;
  Loop& operator=(Loop&&) = delete;

  [[nodiscard]] std::size_t getLoopDepth() const {
    std::size_t i = 0;
    for (auto* curr = m_parentLoop; curr; curr = curr->m_parentLoop)
      i++;
    return i;
  }

  [[nodiscard]] mlir::Block* getHeader() const {
    return m_blocks.front();
  }

  llvm::ArrayRef<mlir::Block*> getBlocks() const {
    return m_blocks;
  }

  [[nodiscard]] Loop* getParentLoop() const {
    return m_parentLoop;
  }

  using iterator = decltype(m_subLoops)::iterator;

  iterator begin() {
    return m_subLoops.begin();
  }

  iterator end() {
    return m_subLoops.begin();
  }

  bool contains(mlir::Block* block) const {
    return m_blockSet.count(block);
  }

  void print(llvm::raw_ostream& os, unsigned depth = 0) const;
};

class LoopInfo {
  /// Maps a basic block to the inner most loop it is contained in.
  llvm::DenseMap<mlir::Block*, Loop*> m_mapping;
  std::vector<Loop*> m_topLevelLoops;
  llvm::BumpPtrAllocator m_loopAllocator;

  template <class... Args>
  Loop* allocateLoop(Args&&... args) {
    auto* storage = m_loopAllocator.Allocate<Loop>();
    return new (storage) Loop(std::forward<Args>(args)...);
  }

  void discoverAndMapSubLoops(Loop* loop, llvm::ArrayRef<mlir::Block*> latches,
                              mlir::DominanceInfo& domInfo);

public:
  LoopInfo(mlir::Operation* operation, mlir::AnalysisManager& analysisManager);

  ~LoopInfo() {
    for (auto* iter : m_topLevelLoops)
      iter->~Loop();
  }

  LoopInfo(const LoopInfo&) = delete;
  LoopInfo(LoopInfo&& rhs) noexcept
      : m_mapping(std::move(rhs.m_mapping)),
        m_topLevelLoops(std::move(rhs.m_topLevelLoops)),
        m_loopAllocator(std::move(rhs.m_loopAllocator)) {
    rhs.m_topLevelLoops.clear();
  }

  LoopInfo& operator=(const LoopInfo&) = delete;
  LoopInfo& operator=(LoopInfo&& rhs) noexcept {
    m_mapping = std::move(rhs.m_mapping);
    for (auto* iter : m_topLevelLoops)
      iter->~Loop();

    m_topLevelLoops = std::move(rhs.m_topLevelLoops);
    m_loopAllocator = std::move(rhs.m_loopAllocator);
    rhs.m_topLevelLoops.clear();
    return *this;
  }

  Loop* getLoopFor(mlir::Block* block) const {
    return m_mapping.lookup(block);
  }

  llvm::ArrayRef<Loop*> getTopLevelLoops() const {
    return m_topLevelLoops;
  }

  void print(llvm::raw_ostream& os) const;

  void dump() const;
};
} // namespace pylir

namespace llvm {
template <>
struct GraphTraits<pylir::Loop*> {
  using ChildIteratorType = pylir::Loop::iterator;
  using Node = pylir::Loop;
  using NodeRef = Node*;

  static NodeRef getEntryNode(NodeRef l) {
    return l;
  }

  static ChildIteratorType child_begin(NodeRef node) {
    return node->begin();
  }

  static ChildIteratorType child_end(NodeRef node) {
    return node->begin();
  }
};

} // namespace llvm
