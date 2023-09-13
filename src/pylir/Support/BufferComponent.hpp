//  Licensed under the Apache License v2.0 with LLVM Exceptions.
//  See https://llvm.org/LICENSE.txt for license information.
//  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <memory>
#include <type_traits>
#include <utility>

namespace pylir {
template <class T, template <class> class Allocator = std::allocator>
class BufferComponent {
  std::size_t m_size{};
  std::size_t m_capacity{};
  T* m_data = nullptr;

  static_assert(typename std::allocator_traits<Allocator<T>>::is_always_equal{},
                "Allocator must be stateless");

  static T* allocate(std::size_t capacity) {
    Allocator<T> a;
    return std::allocator_traits<Allocator<T>>::allocate(a, capacity);
  }

  static void deallocate(T* data, std::size_t capacity) {
    if (data) {
      Allocator<T> a;
      std::allocator_traits<Allocator<T>>::deallocate(a, data, capacity);
    }
  }

  void destruct() {
    std::destroy_n(m_data, m_size);
    deallocate(m_data, m_capacity);
  }

  void maybeGrow() {
    if (m_size + 1 <= m_capacity)
      return;

    reserve(std::max(2 * m_capacity, m_size + 1));
  }

public:
  BufferComponent() = default;

  ~BufferComponent() {
    destruct();
  }

  template <class InputIter>
  BufferComponent(InputIter begin, InputIter end) {
    if constexpr (std::is_same_v<typename std::iterator_traits<
                                     InputIter>::iterator_category,
                                 std::random_access_iterator_tag>)
      reserve(end - begin);

    for (; begin != end; begin++)
      push_back(*begin);
  }

  BufferComponent(const BufferComponent& rhs)
      : m_size(rhs.m_size), m_capacity(rhs.m_capacity),
        m_data(allocate(m_capacity)) {
    std::uninitialized_copy_n(rhs.m_data, m_size, m_data);
  }

  BufferComponent(BufferComponent&& rhs) noexcept
      : m_size(std::exchange(rhs.m_size, 0)),
        m_capacity(std::exchange(rhs.m_capacity, 0)),
        m_data(std::exchange(rhs.m_data, nullptr)) {}

  BufferComponent& operator=(const BufferComponent& rhs) {
    if (this == &rhs)
      return *this;

    if (m_size > rhs.m_size)
      std::destroy_n(m_data + rhs.m_size, m_size - rhs.m_size);

    auto prevSize = m_size;
    reserve(rhs.m_size);
    m_size = rhs.m_size;
    std::copy_n(rhs.m_data, std::min(prevSize, m_size), m_data);
    if (prevSize < rhs.m_size)
      std::uninitialized_copy_n(rhs.m_data + prevSize, rhs.m_size - prevSize,
                                m_data + prevSize);

    return *this;
  }

  BufferComponent& operator=(BufferComponent&& rhs) noexcept {
    destruct();
    m_size = std::exchange(rhs.m_size, 0);
    m_capacity = std::exchange(rhs.m_capacity, 0);
    m_data = std::exchange(rhs.m_data, nullptr);
    return *this;
  }

  [[nodiscard]] std::size_t size() const {
    return m_size;
  }

  void reserve(std::size_t n) {
    if (n <= m_capacity)
      return;

    auto prevCap = m_capacity;
    m_capacity = n;
    auto* newData = allocate(m_capacity);
    std::uninitialized_move_n(m_data, m_size, newData);
    deallocate(m_data, prevCap);
    m_data = newData;
  }

  T* data() {
    return m_data;
  }

  [[nodiscard]] const T* data() const {
    return m_data;
  }

  T& operator[](std::size_t index) {
    return m_data[index];
  }

  const T& operator[](std::size_t index) const {
    return m_data[index];
  }

  T& back() {
    return (*this)[m_size - 1];
  }

  [[nodiscard]] const T& back() const {
    return (*this)[m_size - 1];
  }

  void push_back(T&& value) {
    maybeGrow();
    m_size++;
    new (&m_data[m_size - 1]) T{std::move(value)};
  }

  void push_back(const T& value) {
    maybeGrow();
    m_size++;
    new (&m_data[m_size - 1]) T{value};
  }

  template <class... Args>
  void emplace_back(Args&&... args) {
    maybeGrow();
    m_size++;
    new (&m_data[m_size - 1]) T{std::forward<Args>(args)...};
  }

  void erase(std::size_t index) {
    std::move(m_data + index + 1, m_data + m_size, m_data + index);
    m_size--;
    std::destroy_at(m_data + m_size);
  }

  void clear() {
    std::destroy_n(m_data, m_size);
    m_size = 0;
  }
};

} // namespace pylir
