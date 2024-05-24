//  Licensed under the Apache License v2.0 with LLVM Exceptions.
//  See https://llvm.org/LICENSE.txt for license information.
//  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <pylir/Runtime/Util/Pages.hpp>
#include <pylir/Support/Macros.hpp>

#include <cstdint>
#include <vector>

namespace pylir::rt {

class PyObject;

class BestFitTree {
  struct Node;

  class BlockHeader {
    BlockHeader* m_previousBlock;
    static_assert(alignof(BlockHeader*) >= 4);

  public:
    std::size_t size;

    BlockHeader(std::size_t size, BlockHeader* previousBlock)
        : m_previousBlock(previousBlock), size(size) {}

    [[nodiscard]] BlockHeader* getPreviousBlock() const noexcept {
      return reinterpret_cast<BlockHeader*>(
          reinterpret_cast<std::uintptr_t>(m_previousBlock) &
          ~std::uintptr_t{0b11});
    }

    void setPreviousBlock(BlockHeader* block) noexcept {
      m_previousBlock = reinterpret_cast<BlockHeader*>(
          (reinterpret_cast<std::uintptr_t>(block)) |
          (reinterpret_cast<std::uintptr_t>(m_previousBlock) & 0b11));
    }

    [[nodiscard]] BlockHeader* getNextBlock() const noexcept {
      return reinterpret_cast<BlockHeader*>(
          const_cast<std::byte*>(reinterpret_cast<const std::byte*>(this) +
                                 sizeof(BlockHeader) + size));
    }

    [[nodiscard]] bool isAllocated() const noexcept {
      return (reinterpret_cast<std::uintptr_t>(m_previousBlock) & 0b11) == 0;
    }

    void markAllocated() noexcept {
      m_previousBlock = getPreviousBlock();
    }

    enum Balance {
      Left = 0b01,
      Equal = 0b10,
      Right = 0b11,
    };

    [[nodiscard]] Balance getBalance() const noexcept {
      PYLIR_ASSERT(!isAllocated());
      return static_cast<Balance>(
          reinterpret_cast<std::uintptr_t>(m_previousBlock) & 0b11);
    }

    void setBalance(Balance balance) noexcept {
      m_previousBlock = reinterpret_cast<BlockHeader*>(
          (reinterpret_cast<std::uintptr_t>(m_previousBlock) &
           ~std::uintptr_t{0b11}) |
          balance);
    }

    Node& emplaceNode() noexcept {
      setBalance(Equal);
      return *new (reinterpret_cast<std::byte*>(this) + sizeof(BlockHeader))
          Node{};
    }

    [[nodiscard]] Node& getNode() noexcept {
      PYLIR_ASSERT(!isAllocated());
      return *reinterpret_cast<Node*>(reinterpret_cast<std::byte*>(this) +
                                      sizeof(BlockHeader));
    }

    [[nodiscard]] std::byte* getCell() noexcept {
      PYLIR_ASSERT(isAllocated());
      return reinterpret_cast<std::byte*>(this) + sizeof(BlockHeader);
    }
  };

  struct Node {
    BlockHeader* left{};
    BlockHeader* right{};
    BlockHeader* parent{};
    // I could reuse left & right for these
    BlockHeader* multiNext{};
    BlockHeader* multiPrevious{};
  };

  static_assert(alignof(Node) <= alignof(BlockHeader));

  std::size_t m_lowerBlockSizeLimit;
  BlockHeader* m_root = nullptr;
  std::vector<PagePtr> m_pages;

  void swapNode(BlockHeader* lhs, BlockHeader* rhs);

  void leftRotate(BlockHeader* parent, BlockHeader* current);

  void rightRotate(BlockHeader* parent, BlockHeader* current);

  void leftRightRotate(BlockHeader* parent, BlockHeader* current);

  void rightLeftRotate(BlockHeader* parent, BlockHeader* current);

  template <BlockHeader* Node::*direction>
  BlockHeader* rotate(BlockHeader* root);

  void insertRebalance(BlockHeader* current);

  void removeRebalance(BlockHeader* current);

  void remove(BlockHeader* current);

  std::pair<BlockHeader*, BlockHeader*> lowerBound(std::size_t size);

  void insert(BlockHeader* blockHeader);

  BlockHeader* doAllocation(BlockHeader* blockHeader, std::size_t size) const;

  void verifyTree();

public:
  explicit BestFitTree(std::size_t lowerBlockSizeLimit)
      : m_lowerBlockSizeLimit(lowerBlockSizeLimit) {
    PYLIR_ASSERT(lowerBlockSizeLimit >= sizeof(Node));
  }

  ~BestFitTree() = default;
  BestFitTree(BestFitTree&&) noexcept = default;
  BestFitTree& operator=(BestFitTree&&) noexcept = default;
  BestFitTree(const BestFitTree&) = delete;
  BestFitTree& operator=(const BestFitTree&) = delete;

  PyObject* alloc(std::size_t size);

  void free(PyObject* object);

  void sweep();

  void finalize();
};

} // namespace pylir::rt
