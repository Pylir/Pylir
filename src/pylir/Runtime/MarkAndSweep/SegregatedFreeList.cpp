//  Licensed under the Apache License v2.0 with LLVM Exceptions.
//  See https://llvm.org/LICENSE.txt for license information.
//  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "SegregatedFreeList.hpp"

#include <pylir/Runtime/Objects/Objects.hpp>

#include <cstring>

#include "MarkAndSweep.hpp"

pylir::rt::PyObject* pylir::rt::SegregatedFreeList::nextCell() {
  std::byte* cell;
  if (!m_head) {
    if (!m_pages.empty()) {
      gc.collect();
      if (!m_head)
        cell = m_pages.emplace_back(newPage()).get();
      else
        cell = m_head;

    } else {
      cell = m_pages.emplace_back(newPage()).get();
    }
  } else {
    cell = m_head;
  }
  std::memcpy(&m_head, cell, sizeof(std::byte*));
  return reinterpret_cast<PyObject*>(cell);
}

namespace {
std::byte* getEndCell(const pylir::rt::PagePtr& pagePtr,
                      std::size_t sizeClass) {
  return pagePtr.get() + ((pagePtr.size() / sizeClass) * sizeClass);
}
} // namespace

pylir::rt::PagePtr pylir::rt::SegregatedFreeList::newPage() const {
  auto result = pageAlloc(1);
  auto* end = getEndCell(result, m_sizeClass) - m_sizeClass;
  for (std::byte* begin = result.get(); begin != end; begin += m_sizeClass) {
    std::byte* nextCellAddress = begin + m_sizeClass;
    std::memcpy(begin, &nextCellAddress, sizeof(std::byte*));
  }
  return result;
}

void pylir::rt::SegregatedFreeList::finalize() {
  for (PyObject& object : *this) {
    if (object.getMark<bool>())
      continue;

    destroyPyObject(object);
  }
}

void pylir::rt::SegregatedFreeList::sweep() {
  auto iter = begin();
  for (; iter != end();) {
    PyObject& object = *iter;
    if (object.getMark<bool>()) {
      object.clearMarking();
      iter++;
      continue;
    }
    iter = erase(iter);
  }
}

auto pylir::rt::SegregatedFreeList::erase(iterator iter) -> iterator {
  if (!iter.m_newHead) {
    // If the free list head wasn't discovered previously, then this slot will
    // become the new head.
    m_head = iter.m_newHead = iter.m_byteIter;
    iter.m_previousFreeList = iter.m_newHead;
  } else {
    // Redirect the previous entry in the free list to the new free slot.
    std::memcpy(iter.m_previousFreeList, &iter.m_byteIter, sizeof(std::byte*));
    iter.m_previousFreeList = iter.m_byteIter;
  }
  // Link the new free slot to the next free list slot.
  std::memcpy(iter.m_byteIter, &iter.m_currentFreeList, sizeof(std::byte*));

  PYLIR_ASSERT(!iter.m_currentFreeList ||
               iter.m_previousFreeList < iter.m_currentFreeList);

  ++iter;
  return iter;
}

bool pylir::rt::SegregatedFreeList::iterator::incrementSlot() {
  m_byteIter += m_freeList->m_sizeClass;
  if (m_byteIter == getEndCell(*m_pageIter, m_freeList->m_sizeClass)) {
    m_pageIter++;
    if (m_pageIter == m_freeList->m_pages.end()) {
      m_byteIter = nullptr;
      return true;
    }

    m_byteIter = m_pageIter->get();
  }
  return false;
}

void pylir::rt::SegregatedFreeList::iterator::skipThroughFreeList() {
  if (!m_byteIter)
    return;

  while (m_byteIter == m_currentFreeList) {
    // If no head has been found previously, set the new head to the
    // current one.
    if (!m_newHead)
      m_newHead = m_currentFreeList;

    m_previousFreeList = m_currentFreeList;
    // Forward in the linked list.
    std::memcpy(&m_currentFreeList, m_currentFreeList, sizeof(std::byte*));
    PYLIR_ASSERT(!m_currentFreeList || m_previousFreeList < m_currentFreeList);
    if (incrementSlot())
      return;
  }
}

auto pylir::rt::SegregatedFreeList::iterator::operator++() -> iterator& {
  if (incrementSlot())
    return *this;
  skipThroughFreeList();
  return *this;
}
