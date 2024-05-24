//  Licensed under the Apache License v2.0 with LLVM Exceptions.
//  See https://llvm.org/LICENSE.txt for license information.
//  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "BestFitTree.hpp"

#include <pylir/Runtime/Objects/Objects.hpp>
#include <pylir/Support/Util.hpp>

auto pylir::rt::BestFitTree::BestFitTree::lowerBound(std::size_t size)
    -> std::pair<BlockHeader*, BlockHeader*> {
  if (!m_root)
    return {nullptr, nullptr};

  BlockHeader* lowerBound = nullptr;
  BlockHeader* previous = nullptr;
  BlockHeader* current = m_root;
  while (current) {
    previous = current;
    if (current->size < size) {
      current = current->getNode().right;
    } else if (current->size > size) {
      lowerBound = current;
      current = current->getNode().left;
    } else {
      return {current, current};
    }
  }

  return {lowerBound, previous};
}

void pylir::rt::BestFitTree::swapNode(BlockHeader* lhs, BlockHeader* rhs) {
  auto temp = lhs->getBalance();
  lhs->setBalance(rhs->getBalance());
  rhs->setBalance(temp);
  std::swap(rhs->getNode(), lhs->getNode());
  auto fixUpPointees = [this](BlockHeader* header, BlockHeader* previous) {
    auto& node = header->getNode();
    for (auto member : {&Node::left, &Node::right, &Node::parent,
                        &Node::multiNext, &Node::multiPrevious})
      if (node.*member == header)
        node.*member = previous;

    if (node.multiNext)
      node.multiNext->getNode().multiPrevious = header;

    if (node.multiPrevious) {
      node.multiPrevious->getNode().multiNext = header;
      return;
    }
    if (node.left)
      node.left->getNode().parent = header;

    if (node.right)
      node.right->getNode().parent = header;

    auto* parent = node.parent;
    if (!parent) {
      m_root = header;
      return;
    }
    if (parent->getNode().left == previous)
      parent->getNode().left = header;
    else
      parent->getNode().right = header;
  };
  fixUpPointees(lhs, rhs);
  fixUpPointees(rhs, lhs);
}

template <pylir::rt::BestFitTree::BlockHeader* pylir::rt::BestFitTree::Node::*
              direction>
auto pylir::rt::BestFitTree::rotate(BlockHeader* root) -> BlockHeader* {
  constexpr auto otherDirection =
      direction == &Node::left ? &Node::right : &Node::left;
  auto& node = root->getNode();
  auto* parent = node.parent;
  auto* child = node.*otherDirection;
  auto& childNode = child->getNode();
  childNode.parent = parent;
  node.*otherDirection = childNode.*direction;
  if (node.*otherDirection)
    (node.*otherDirection)->getNode().parent = root;

  childNode.*direction = root;
  node.parent = child;
  if (parent) {
    auto& parentNode = parent->getNode();
    if (parentNode.left == root)
      parentNode.left = child;
    else
      parentNode.right = child;

  } else {
    m_root = child;
  }
  return child;
}

void pylir::rt::BestFitTree::leftRotate(BlockHeader* parent,
                                        BlockHeader* current) {
  rotate<&Node::left>(parent);
  if (current->getBalance() == BlockHeader::Equal) {
    parent->setBalance(BlockHeader::Right);
    current->setBalance(BlockHeader::Left);
  } else {
    parent->setBalance(BlockHeader::Equal);
    current->setBalance(BlockHeader::Equal);
  }
}

void pylir::rt::BestFitTree::rightRotate(BlockHeader* parent,
                                         BlockHeader* current) {
  rotate<&Node::right>(parent);
  if (current->getBalance() == BlockHeader::Equal) {
    parent->setBalance(BlockHeader::Left);
    current->setBalance(BlockHeader::Right);
  } else {
    parent->setBalance(BlockHeader::Equal);
    current->setBalance(BlockHeader::Equal);
  }
}

void pylir::rt::BestFitTree::rightLeftRotate(BlockHeader* parent,
                                             BlockHeader* current) {
  auto* newRoot = rotate<&Node::right>(current);
  rotate<&Node::left>(parent);
  switch (newRoot->getBalance()) {
  case BlockHeader::Equal:
    current->setBalance(BlockHeader::Equal);
    parent->setBalance(BlockHeader::Equal);
    break;
  case BlockHeader::Right:
    parent->setBalance(BlockHeader::Left);
    current->setBalance(BlockHeader::Equal);
    break;
  case BlockHeader::Left:
    parent->setBalance(BlockHeader::Equal);
    current->setBalance(BlockHeader::Right);
    break;
  }
  newRoot->setBalance(BlockHeader::Equal);
}

void pylir::rt::BestFitTree::leftRightRotate(BlockHeader* parent,
                                             BlockHeader* current) {
  auto* newRoot = rotate<&Node::left>(current);
  rotate<&Node::right>(parent);
  switch (newRoot->getBalance()) {
  case BlockHeader::Equal:
    current->setBalance(BlockHeader::Equal);
    parent->setBalance(BlockHeader::Equal);
    break;
  case BlockHeader::Right:
    parent->setBalance(BlockHeader::Right);
    current->setBalance(BlockHeader::Equal);
    break;
  case BlockHeader::Left:
    parent->setBalance(BlockHeader::Equal);
    current->setBalance(BlockHeader::Left);
    break;
  }
  newRoot->setBalance(BlockHeader::Equal);
}

void pylir::rt::BestFitTree::remove(BlockHeader* current) {
  {
    auto& node = current->getNode();
    if (auto* subLeaf = node.left; subLeaf && node.right) {
      while (subLeaf->getNode().right)
        subLeaf = subLeaf->getNode().right;

      swapNode(current, subLeaf);
    }
  }
  removeRebalance(current);
  auto& node = current->getNode();
  for (auto member : {&Node::left, &Node::right,
                      static_cast<BlockHeader * Node::*>(nullptr)}) {
    BlockHeader* child = nullptr;
    if (member) {
      child = node.*member;
      if (!child)
        continue;
    }
    auto* parent = node.parent;
    if (parent) {
      if (child)
        child->getNode().parent = parent;

      auto& parentNode = parent->getNode();
      if (parentNode.left == current)
        parentNode.left = child;
      else
        parentNode.right = child;

    } else {
      if (child)
        child->getNode().parent = nullptr;

      m_root = child;
    }
    break;
  }
}

void pylir::rt::BestFitTree::removeRebalance(BlockHeader* current) {
  while (current != m_root) {
    auto& node = current->getNode();
    auto* parent = node.parent;
    auto& parentNode = parent->getNode();
    switch (parent->getBalance()) {
    case BlockHeader::Equal: {
      if (parentNode.right == current)
        parent->setBalance(BlockHeader::Left);
      else
        parent->setBalance(BlockHeader::Right);

      return;
    }
    case BlockHeader::Left: {
      if (parentNode.left == current) {
        parent->setBalance(BlockHeader::Equal);
        break;
      }
      auto balance = parentNode.left->getBalance();
      if (balance == BlockHeader::Right) {
        leftRightRotate(parent, parentNode.left);
        break;
      }
      rightRotate(parent, parentNode.left);
      if (balance == BlockHeader::Equal)
        return;

      break;
    }
    case BlockHeader::Right: {
      if (parentNode.right == current) {
        parent->setBalance(BlockHeader::Equal);
        break;
      }
      auto balance = parentNode.right->getBalance();
      if (balance == BlockHeader::Left) {
        rightLeftRotate(parent, parentNode.right);
        break;
      }
      leftRotate(parent, parentNode.right);
      if (balance == BlockHeader::Equal)
        return;

      break;
    }
    }
    current = parent;
  }
}

void pylir::rt::BestFitTree::insertRebalance(BlockHeader* current) {
  while (current != m_root) {
    auto* parent = current->getNode().parent;
    switch (parent->getBalance()) {
    case BlockHeader::Equal:
      if (parent->getNode().left == current)
        parent->setBalance(BlockHeader::Left);
      else
        parent->setBalance(BlockHeader::Right);

      current = parent;
      continue;
    case BlockHeader::Left: {
      if (parent->getNode().right == current) {
        parent->setBalance(BlockHeader::Equal);
        return;
      }
      if (current->getBalance() == BlockHeader::Right)
        leftRightRotate(parent, current);
      else
        rightRotate(parent, current);

      return;
    }
    case BlockHeader::Right: {
      if (parent->getNode().left == current) {
        parent->setBalance(BlockHeader::Equal);
        return;
      }
      if (current->getBalance() == BlockHeader::Left)
        rightLeftRotate(parent, current);
      else
        leftRotate(parent, current);

      return;
    }
    }
  }
}

void pylir::rt::BestFitTree::insert(BlockHeader* blockHeader) {
  blockHeader->emplaceNode();
  if (!m_root) {
    m_root = blockHeader;
    return;
  }
  auto* parent = lowerBound(blockHeader->size).second;
  if (parent->size == blockHeader->size) {
    blockHeader->getNode().multiNext = parent->getNode().multiNext;
    if (blockHeader->getNode().multiNext)
      blockHeader->getNode().multiNext->getNode().multiPrevious = blockHeader;

    parent->getNode().multiNext = blockHeader;
    blockHeader->getNode().multiPrevious = parent;
    return;
  }
  if (parent->size > blockHeader->size)
    parent->getNode().left = blockHeader;
  else
    parent->getNode().right = blockHeader;

  blockHeader->getNode().parent = parent;
  insertRebalance(blockHeader);
}

auto pylir::rt::BestFitTree::doAllocation(BlockHeader* blockHeader,
                                          std::size_t size) const
    -> BlockHeader* {
  blockHeader->markAllocated();
  auto sizeAligned = pylir::roundUpTo(size, alignof(PyBaseException));
  if (blockHeader->size <=
      m_lowerBlockSizeLimit + sizeof(BlockHeader) + sizeAligned)
    return nullptr;

  auto* splitAddress = blockHeader->getCell() + sizeAligned;
  auto afterSplit = blockHeader->size - sizeAligned - sizeof(BlockHeader);
  blockHeader->size = sizeAligned;
  auto* split = new (splitAddress) BlockHeader(afterSplit, blockHeader);
  split->getNextBlock()->setPreviousBlock(split);
  return split;
}

pylir::rt::PyObject* pylir::rt::BestFitTree::alloc(std::size_t size) {
  auto* result = lowerBound(size).first;
  if (!result) {
    auto& memory = m_pages.emplace_back(pageAllocBytes(size));
    result = new (memory.get())
        BlockHeader(memory.size() - 2 * sizeof(BlockHeader), nullptr);
    // Sentinel for end
    new (memory.get() + memory.size() - sizeof(BlockHeader))
        BlockHeader(0, result);
  } else if (result->getNode().multiNext) {
    auto* temp = result->getNode().multiNext;
    if (auto* next = result->getNode().multiNext = temp->getNode().multiNext)
      next->getNode().multiPrevious = result;

    result = temp;
  } else {
    remove(result);
  }
  auto* split = doAllocation(result, size);
  if (split)
    insert(split);

  return reinterpret_cast<PyObject*>(result->getCell());
}

void pylir::rt::BestFitTree::free(PyObject* object) {
  auto* blockHeader = reinterpret_cast<BlockHeader*>(
      reinterpret_cast<std::byte*>(object) - sizeof(BlockHeader));

  auto* previousBlock = blockHeader->getPreviousBlock();
  if (previousBlock && previousBlock->isAllocated())
    previousBlock = nullptr;

  auto* nextBlock = blockHeader->getNextBlock();
  if (nextBlock->isAllocated())
    nextBlock = nullptr;

  if (!previousBlock && !nextBlock) {
    insert(blockHeader);
    return;
  }
  // Coalesce blocks

  auto unlinkBlock = [this](BlockHeader* block) {
    auto* prev = block->getNode().multiPrevious;
    auto* next = block->getNode().multiNext;
    // if there is no previous then this node is part of the AVL tree
    if (!prev) {
      if (next) {
        // Instead of a remove and then a reinsert, swap with the next node to
        // save a rebalance operation. They have the same key anyways
        swapNode(next, block);
      } else {
        remove(block);
      }
    } else {
      // If we are not part of the AVL tree we can simply remove ourselves from
      // the doubly linked list
      prev->getNode().multiNext = next;
      if (next)
        next->getNode().multiPrevious = prev;
    }
  };

  BlockHeader* leftMostBlock = blockHeader;
  if (previousBlock) {
    unlinkBlock(previousBlock);
    leftMostBlock = previousBlock;
    leftMostBlock->size += sizeof(BlockHeader) + blockHeader->size;
  }
  if (nextBlock) {
    unlinkBlock(nextBlock);
    leftMostBlock->size += sizeof(BlockHeader) + nextBlock->size;
    nextBlock->getNextBlock()->setPreviousBlock(leftMostBlock);
  } else {
    blockHeader->getNextBlock()->setPreviousBlock(leftMostBlock);
  }

  insert(leftMostBlock);
}

void pylir::rt::BestFitTree::verifyTree() {
  auto checkHeight = [](auto& f, BlockHeader* node) -> int {
    if (!node)
      return 0;

    if (node->getNode().left)
      PYLIR_ASSERT(node->getNode().left->getNode().parent == node);

    if (node->getNode().right)
      PYLIR_ASSERT(node->getNode().right->getNode().parent == node);

    auto left = f(f, node->getNode().left);
    auto right = f(f, node->getNode().right);
    auto balance = right - left;
    switch (node->getBalance()) {
    case BlockHeader::Left: PYLIR_ASSERT(balance == -1); break;
    case BlockHeader::Equal: PYLIR_ASSERT(balance == 0); break;
    case BlockHeader::Right: PYLIR_ASSERT(balance == 1); break;
    }
    return 1 + std::max(left, right);
  };
  checkHeight(checkHeight, m_root);
}

void pylir::rt::BestFitTree::finalize() {
  for (auto& iter : m_pages) {
    for (auto* block = reinterpret_cast<BlockHeader*>(iter.get()); block->size;
         block = block->getNextBlock()) {
      if (!block->isAllocated())
        continue;
      auto* object = reinterpret_cast<PyObject*>(block->getCell());
      if (object->getMark<bool>())
        continue;

      destroyPyObject(*object);
    }
  }
}

void pylir::rt::BestFitTree::sweep() {
  for (PagePtr& iter : m_pages) {
    for (auto* block = reinterpret_cast<BlockHeader*>(iter.get()); block->size;
         block = block->getNextBlock()) {
      if (!block->isAllocated())
        continue;

      auto* object = reinterpret_cast<PyObject*>(block->getCell());
      if (object->getMark<bool>()) {
        object->clearMarking();
        continue;
      }
      free(object);
    }
  }
}
