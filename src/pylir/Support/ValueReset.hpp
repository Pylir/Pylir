//  Licensed under the Apache License v2.0 with LLVM Exceptions.
//  See https://llvm.org/LICENSE.txt for license information.
//  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <type_traits>
#include <utility>

namespace pylir {
template <class T>
class ValueReset {
  static_assert(!std::is_const_v<T>);

  T m_valueAfter;
  T* m_assignedTo;

public:
  template <class U = T>
  ValueReset(T& assignedTo, U valueAfter)
      : m_valueAfter(std::move(valueAfter)), m_assignedTo(&assignedTo) {}

  explicit ValueReset(T&& assignedTo)
      : ValueReset(assignedTo, std::move(assignedTo)) {}

  explicit ValueReset(T& assignedTo) : ValueReset(assignedTo, assignedTo) {}

  ~ValueReset() {
    if (m_assignedTo)
      *m_assignedTo = std::move(m_valueAfter);
  }

  ValueReset(const ValueReset&) = delete;
  ValueReset& operator=(const ValueReset&) = delete;

  ValueReset(ValueReset&& rhs) noexcept
      : m_valueAfter(std::move(rhs.m_valueAfter)),
        m_assignedTo(std::exchange(rhs.m_assignedTo, nullptr)) {}

  ValueReset& operator=(ValueReset&& rhs) noexcept {
    m_assignedTo = std::exchange(rhs.m_assignedTo, nullptr);
    m_valueAfter = std::move(rhs.m_valueAfter);
    return *this;
  }

  /// Returns a reference to the value that will be assigned at destruction.
  T& getValueAfterReset() {
    return m_valueAfter;
  }

  const T& getValueAfterReset() const {
    return m_valueAfter;
  }
};

template <class... Args>
std::tuple<ValueReset<std::decay_t<Args>>...> valueResetMany(Args&&... args) {
  return {ValueReset(std::forward<Args>(args))...};
}

} // namespace pylir
