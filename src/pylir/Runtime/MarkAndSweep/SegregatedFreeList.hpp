//  Licensed under the Apache License v2.0 with LLVM Exceptions.
//  See https://llvm.org/LICENSE.txt for license information.
//  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <pylir/Runtime/Util/Pages.hpp>

#include <memory>
#include <vector>

namespace pylir::rt {

class PyObject;

class SegregatedFreeList {
  std::size_t m_sizeClass;
  std::byte* m_head = nullptr;
  std::vector<PagePtr> m_pages;

  [[nodiscard]] PagePtr newPage() const;

public:
  explicit SegregatedFreeList(std::size_t sizeClass) : m_sizeClass(sizeClass) {}

  ~SegregatedFreeList() = default;
  SegregatedFreeList(SegregatedFreeList&&) noexcept = default;
  SegregatedFreeList& operator=(SegregatedFreeList&&) noexcept = default;
  SegregatedFreeList(const SegregatedFreeList&) = delete;
  SegregatedFreeList& operator=(const SegregatedFreeList&) = delete;

  PyObject* nextCell();

  void finalize();

  void sweep();

  class iterator {
    SegregatedFreeList* m_freeList = nullptr;
    decltype(m_pages)::iterator m_pageIter{};
    std::byte* m_byteIter = nullptr;
    std::byte* m_currentFreeList = nullptr;
    std::byte* m_previousFreeList = nullptr;
    std::byte* m_newHead = nullptr;

    friend class SegregatedFreeList;

    bool incrementSlot();

    void skipThroughFreeList();

  public:
    iterator() = default;

    iterator(SegregatedFreeList* freeList, decltype(m_pages)::iterator pageIter,
             std::byte* byteIter)
        : m_freeList(freeList), m_pageIter(pageIter), m_byteIter(byteIter),
          m_currentFreeList(freeList->m_head) {
      skipThroughFreeList();
    }

    using difference_type = std::ptrdiff_t;
    using value_type = PyObject;
    using reference = value_type&;
    using pointer = value_type*;

    reference operator*() const {
      return *this->operator->();
    }

    pointer operator->() const {
      return reinterpret_cast<PyObject*>(m_byteIter);
    }

    iterator& operator++();

    iterator operator++(int) {
      iterator temp = *this;
      ++*this;
      return temp;
    }

    bool operator!=(const iterator& rhs) const {
      return std::tuple(m_freeList, m_pageIter, m_byteIter) !=
             std::tuple(rhs.m_freeList, rhs.m_pageIter, rhs.m_byteIter);
    }

    std::byte* getPreviousHead() const {
      return m_previousFreeList;
    }

    std::byte* getNewHead() const {
      return m_newHead;
    }
  };

  iterator begin() {
    return iterator(this, m_pages.begin(),
                    m_pages.empty() ? nullptr : m_pages.begin()->get());
  }

  iterator end() {
    return iterator(this, m_pages.end(), nullptr);
  }

  iterator erase(iterator iter);
};
} // namespace pylir::rt
