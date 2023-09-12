//  Licensed under the Apache License v2.0 with LLVM Exceptions.
//  See https://llvm.org/LICENSE.txt for license information.
//  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <functional>
#include <tuple>
#include <utility>

namespace pylir {
namespace detail {
template <class F, class... Args>
class BindFrontImpl {
  F m_f;
  std::tuple<Args...> m_front;

public:
  explicit BindFrontImpl(F f, std::tuple<Args...> args)
      : m_f(std::move(f)), m_front(std::move(args)) {}

  template <class... Last>
  decltype(auto) operator()(Last&&... last) & noexcept(
      std::is_nothrow_invocable_v<F, Args..., Last...>) {
    return std::apply(
        [&](auto&&... members) -> decltype(auto) {
          return std::invoke(m_f, std::forward<decltype(members)>(members)...,
                             std::forward<Last>(last)...);
        },
        m_front);
  }

  template <class... Last>
  decltype(auto) operator()(Last&&... last) const& noexcept(
      std::is_nothrow_invocable_v<F, Args..., Last...>) {
    return std::apply(
        [&](auto&&... members) -> decltype(auto) {
          return std::invoke(m_f, std::forward<decltype(members)>(members)...,
                             std::forward<Last>(last)...);
        },
        m_front);
  }

  template <class... Last>
  decltype(auto) operator()(Last&&... last) && noexcept(
      std::is_nothrow_invocable_v<F, Args..., Last...>) {
    return std::apply(
        [&](auto&&... members) -> decltype(auto) {
          return std::invoke(std::move(m_f),
                             std::forward<decltype(members)>(members)...,
                             std::forward<Last>(last)...);
        },
        std::move(m_front));
  }

  template <class... Last>
  decltype(auto) operator()(Last&&... last) const&& noexcept(
      std::is_nothrow_invocable_v<F, Args..., Last...>) {
    return std::apply(
        [&](auto&&... members) -> decltype(auto) {
          return std::invoke(std::move(m_f),
                             std::forward<decltype(members)>(members)...,
                             std::forward<Last>(last)...);
        },
        std::move(m_front));
  }
};

} // namespace detail

// TODO: Remove in C++20.
template <class F, class... Args>
// NOLINTNEXTLINE(readability-identifier-naming): Intentionally match STL.
auto bind_front(F&& f, Args&&... args) {
  return detail::BindFrontImpl(std::forward<F>(f),
                               std::make_tuple(std::forward<Args>(args)...));
}

} // namespace pylir
