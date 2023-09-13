//  Licensed under the Apache License v2.0 with LLVM Exceptions.
//  See https://llvm.org/LICENSE.txt for license information.
//  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <cstdint>
#include <functional>
#include <iterator>

#include "Macros.hpp"

namespace pylir {
template <class ValueType, class CacheObject, auto nextFunc,
          auto containerAccess>
class LazyCacheIterator {
  CacheObject* m_cache{};
  std::size_t m_index{};

  decltype(auto) container() {
    return std::invoke(containerAccess, m_cache);
  }

  [[nodiscard]] decltype(auto) container() const {
    return std::invoke(containerAccess, m_cache);
  }

public:
  using difference_type = std::ptrdiff_t;
  using value_type = ValueType;
  using pointer = const value_type*;
  using reference = const value_type&;
  using iterator_category = std::forward_iterator_tag;

  LazyCacheIterator() = default;

  LazyCacheIterator(CacheObject& cache, std::size_t index)
      : m_cache(&cache), m_index(index) {
    if (index == 0 && container().empty())
      std::invoke(nextFunc, m_cache);
  }

  reference operator*() const {
    return container()[m_index];
  }

  pointer operator->() const {
    return &operator*();
  }

  LazyCacheIterator& operator++() {
    if (m_index + 1 >= container().size())
      std::invoke(nextFunc, m_cache);

    m_index++;
    return *this;
  }

  LazyCacheIterator operator++(int) {
    auto copy = *this;
    operator++();
    return copy;
  }

  bool operator==(const LazyCacheIterator& rhs) const {
    if (m_cache != rhs.m_cache)
      return false;

    bool bothPastEnd =
        m_index >= container().size() && rhs.m_index >= container().size();
    if (bothPastEnd)
      return true;

    return m_index == rhs.m_index;
  }

  bool operator!=(const LazyCacheIterator& rhs) const {
    return !(rhs == *this);
  }

  difference_type operator-(const LazyCacheIterator& rhs) const {
    PYLIR_ASSERT(m_index != static_cast<std::size_t>(-1));
    return m_index - rhs.m_index;
  }

  [[nodiscard]] pointer data() const {
    return container().data() + m_index;
  }

  friend void swap(LazyCacheIterator& lhs, LazyCacheIterator& rhs) {
    std::swap(lhs.m_cache, rhs.m_cache);
    std::swap(lhs.m_index, rhs.m_index);
  }
};

} // namespace pylir
