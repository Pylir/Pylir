//  Licensed under the Apache License v2.0 with LLVM Exceptions.
//  See https://llvm.org/LICENSE.txt for license information.
//  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <variant>

#include "Macros.hpp"

namespace pylir {
template <typename T, typename Variant>
constexpr decltype(auto) get(Variant&& variant) noexcept {
  PYLIR_ASSERT(!variant.valueless_by_exception() &&
               std::holds_alternative<T>(variant));
  auto* value = std::get_if<T>(&variant);
  PYLIR_ASSERT(value);
  if constexpr (std::is_lvalue_reference_v<Variant>)
    return *value;
  else
    return std::move(*value);
}

template <std::size_t i, typename Variant>
constexpr decltype(auto) get(Variant&& variant) noexcept {
  PYLIR_ASSERT(!variant.valueless_by_exception() && variant.index() == i);
  auto* value = std::get_if<i>(&variant);
  PYLIR_ASSERT(value);
  if constexpr (std::is_lvalue_reference_v<Variant>)
    return *value;
  else
    return std::move(*value);
}

namespace detail {
template <class... Ts>
struct Overload : Ts... {
  using Ts::operator()...;
};
template <class... Ts>
Overload(Ts...) -> Overload<Ts...>;

template <std::size_t i, class Callable, class Variant>
constexpr decltype(auto)
visitImpl(Callable&& callable, Variant&& variant,
          std::enable_if_t<(i == 0 || i > 12)>* = nullptr) {
  if (variant.index() == i)
    return std::forward<Callable>(callable)(
        pylir::get<i>(std::forward<Variant>(variant)));

  if constexpr (i != 0)
    return visitImpl<i - 1>(std::forward<Callable>(callable),
                            std::forward<Variant>(variant));
  else
    PYLIR_UNREACHABLE;
}

template <std::size_t i, class Callable, class Variant>
constexpr decltype(auto) visitImpl(Callable&& callable, Variant&& variant,
                                   std::enable_if_t<(i == 1)>* = nullptr) {
  switch (variant.index()) {
  case 0:
    return std::forward<Callable>(callable)(
        pylir::get<0>(std::forward<Variant>(variant)));
  case 1:
    return std::forward<Callable>(callable)(
        pylir::get<1>(std::forward<Variant>(variant)));
  default: PYLIR_UNREACHABLE;
  }
}

template <std::size_t i, class Callable, class Variant>
constexpr decltype(auto) visitImpl(Callable&& callable, Variant&& variant,
                                   std::enable_if_t<(i == 2)>* = nullptr) {
  switch (variant.index()) {
  case 0:
    return std::forward<Callable>(callable)(
        pylir::get<0>(std::forward<Variant>(variant)));
  case 1:
    return std::forward<Callable>(callable)(
        pylir::get<1>(std::forward<Variant>(variant)));
  case 2:
    return std::forward<Callable>(callable)(
        pylir::get<2>(std::forward<Variant>(variant)));
  default: PYLIR_UNREACHABLE;
  }
}

template <std::size_t i, class Callable, class Variant>
constexpr decltype(auto) visitImpl(Callable&& callable, Variant&& variant,
                                   std::enable_if_t<(i == 3)>* = nullptr) {
  switch (variant.index()) {
  case 0:
    return std::forward<Callable>(callable)(
        pylir::get<0>(std::forward<Variant>(variant)));
  case 1:
    return std::forward<Callable>(callable)(
        pylir::get<1>(std::forward<Variant>(variant)));
  case 2:
    return std::forward<Callable>(callable)(
        pylir::get<2>(std::forward<Variant>(variant)));
  case 3:
    return std::forward<Callable>(callable)(
        pylir::get<3>(std::forward<Variant>(variant)));
  default: PYLIR_UNREACHABLE;
  }
}

template <std::size_t i, class Callable, class Variant>
constexpr decltype(auto) visitImpl(Callable&& callable, Variant&& variant,
                                   std::enable_if_t<(i == 4)>* = nullptr) {
  switch (variant.index()) {
  case 0:
    return std::forward<Callable>(callable)(
        pylir::get<0>(std::forward<Variant>(variant)));
  case 1:
    return std::forward<Callable>(callable)(
        pylir::get<1>(std::forward<Variant>(variant)));
  case 2:
    return std::forward<Callable>(callable)(
        pylir::get<2>(std::forward<Variant>(variant)));
  case 3:
    return std::forward<Callable>(callable)(
        pylir::get<3>(std::forward<Variant>(variant)));
  case 4:
    return std::forward<Callable>(callable)(
        pylir::get<4>(std::forward<Variant>(variant)));
  default: PYLIR_UNREACHABLE;
  }
}

template <std::size_t i, class Callable, class Variant>
constexpr decltype(auto) visitImpl(Callable&& callable, Variant&& variant,
                                   std::enable_if_t<(i == 5)>* = nullptr) {
  switch (variant.index()) {
  case 0:
    return std::forward<Callable>(callable)(
        pylir::get<0>(std::forward<Variant>(variant)));
  case 1:
    return std::forward<Callable>(callable)(
        pylir::get<1>(std::forward<Variant>(variant)));
  case 2:
    return std::forward<Callable>(callable)(
        pylir::get<2>(std::forward<Variant>(variant)));
  case 3:
    return std::forward<Callable>(callable)(
        pylir::get<3>(std::forward<Variant>(variant)));
  case 4:
    return std::forward<Callable>(callable)(
        pylir::get<4>(std::forward<Variant>(variant)));
  case 5:
    return std::forward<Callable>(callable)(
        pylir::get<5>(std::forward<Variant>(variant)));
  default: PYLIR_UNREACHABLE;
  }
}

template <std::size_t i, class Callable, class Variant>
constexpr decltype(auto) visitImpl(Callable&& callable, Variant&& variant,
                                   std::enable_if_t<(i == 6)>* = nullptr) {
  switch (variant.index()) {
  case 0:
    return std::forward<Callable>(callable)(
        pylir::get<0>(std::forward<Variant>(variant)));
  case 1:
    return std::forward<Callable>(callable)(
        pylir::get<1>(std::forward<Variant>(variant)));
  case 2:
    return std::forward<Callable>(callable)(
        pylir::get<2>(std::forward<Variant>(variant)));
  case 3:
    return std::forward<Callable>(callable)(
        pylir::get<3>(std::forward<Variant>(variant)));
  case 4:
    return std::forward<Callable>(callable)(
        pylir::get<4>(std::forward<Variant>(variant)));
  case 5:
    return std::forward<Callable>(callable)(
        pylir::get<5>(std::forward<Variant>(variant)));
  case 6:
    return std::forward<Callable>(callable)(
        pylir::get<6>(std::forward<Variant>(variant)));
  default: PYLIR_UNREACHABLE;
  }
}

template <std::size_t i, class Callable, class Variant>
constexpr decltype(auto) visitImpl(Callable&& callable, Variant&& variant,
                                   std::enable_if_t<(i == 7)>* = nullptr) {
  switch (variant.index()) {
  case 0:
    return std::forward<Callable>(callable)(
        pylir::get<0>(std::forward<Variant>(variant)));
  case 1:
    return std::forward<Callable>(callable)(
        pylir::get<1>(std::forward<Variant>(variant)));
  case 2:
    return std::forward<Callable>(callable)(
        pylir::get<2>(std::forward<Variant>(variant)));
  case 3:
    return std::forward<Callable>(callable)(
        pylir::get<3>(std::forward<Variant>(variant)));
  case 4:
    return std::forward<Callable>(callable)(
        pylir::get<4>(std::forward<Variant>(variant)));
  case 5:
    return std::forward<Callable>(callable)(
        pylir::get<5>(std::forward<Variant>(variant)));
  case 6:
    return std::forward<Callable>(callable)(
        pylir::get<6>(std::forward<Variant>(variant)));
  case 7:
    return std::forward<Callable>(callable)(
        pylir::get<7>(std::forward<Variant>(variant)));
  default: PYLIR_UNREACHABLE;
  }
}

template <std::size_t i, class Callable, class Variant>
constexpr decltype(auto) visitImpl(Callable&& callable, Variant&& variant,
                                   std::enable_if_t<(i == 8)>* = nullptr) {
  switch (variant.index()) {
  case 0:
    return std::forward<Callable>(callable)(
        pylir::get<0>(std::forward<Variant>(variant)));
  case 1:
    return std::forward<Callable>(callable)(
        pylir::get<1>(std::forward<Variant>(variant)));
  case 2:
    return std::forward<Callable>(callable)(
        pylir::get<2>(std::forward<Variant>(variant)));
  case 3:
    return std::forward<Callable>(callable)(
        pylir::get<3>(std::forward<Variant>(variant)));
  case 4:
    return std::forward<Callable>(callable)(
        pylir::get<4>(std::forward<Variant>(variant)));
  case 5:
    return std::forward<Callable>(callable)(
        pylir::get<5>(std::forward<Variant>(variant)));
  case 6:
    return std::forward<Callable>(callable)(
        pylir::get<6>(std::forward<Variant>(variant)));
  case 7:
    return std::forward<Callable>(callable)(
        pylir::get<7>(std::forward<Variant>(variant)));
  case 8:
    return std::forward<Callable>(callable)(
        pylir::get<8>(std::forward<Variant>(variant)));
  default: PYLIR_UNREACHABLE;
  }
}

template <std::size_t i, class Callable, class Variant>
constexpr decltype(auto) visitImpl(Callable&& callable, Variant&& variant,
                                   std::enable_if_t<(i == 9)>* = nullptr) {
  switch (variant.index()) {
  case 0:
    return std::forward<Callable>(callable)(
        pylir::get<0>(std::forward<Variant>(variant)));
  case 1:
    return std::forward<Callable>(callable)(
        pylir::get<1>(std::forward<Variant>(variant)));
  case 2:
    return std::forward<Callable>(callable)(
        pylir::get<2>(std::forward<Variant>(variant)));
  case 3:
    return std::forward<Callable>(callable)(
        pylir::get<3>(std::forward<Variant>(variant)));
  case 4:
    return std::forward<Callable>(callable)(
        pylir::get<4>(std::forward<Variant>(variant)));
  case 5:
    return std::forward<Callable>(callable)(
        pylir::get<5>(std::forward<Variant>(variant)));
  case 6:
    return std::forward<Callable>(callable)(
        pylir::get<6>(std::forward<Variant>(variant)));
  case 7:
    return std::forward<Callable>(callable)(
        pylir::get<7>(std::forward<Variant>(variant)));
  case 8:
    return std::forward<Callable>(callable)(
        pylir::get<8>(std::forward<Variant>(variant)));
  case 9:
    return std::forward<Callable>(callable)(
        pylir::get<9>(std::forward<Variant>(variant)));
  default: PYLIR_UNREACHABLE;
  }
}

template <std::size_t i, class Callable, class Variant>
constexpr decltype(auto) visitImpl(Callable&& callable, Variant&& variant,
                                   std::enable_if_t<(i == 10)>* = nullptr) {
  switch (variant.index()) {
  case 0:
    return std::forward<Callable>(callable)(
        pylir::get<0>(std::forward<Variant>(variant)));
  case 1:
    return std::forward<Callable>(callable)(
        pylir::get<1>(std::forward<Variant>(variant)));
  case 2:
    return std::forward<Callable>(callable)(
        pylir::get<2>(std::forward<Variant>(variant)));
  case 3:
    return std::forward<Callable>(callable)(
        pylir::get<3>(std::forward<Variant>(variant)));
  case 4:
    return std::forward<Callable>(callable)(
        pylir::get<4>(std::forward<Variant>(variant)));
  case 5:
    return std::forward<Callable>(callable)(
        pylir::get<5>(std::forward<Variant>(variant)));
  case 6:
    return std::forward<Callable>(callable)(
        pylir::get<6>(std::forward<Variant>(variant)));
  case 7:
    return std::forward<Callable>(callable)(
        pylir::get<7>(std::forward<Variant>(variant)));
  case 8:
    return std::forward<Callable>(callable)(
        pylir::get<8>(std::forward<Variant>(variant)));
  case 9:
    return std::forward<Callable>(callable)(
        pylir::get<9>(std::forward<Variant>(variant)));
  case 10:
    return std::forward<Callable>(callable)(
        pylir::get<10>(std::forward<Variant>(variant)));
  default: PYLIR_UNREACHABLE;
  }
}

template <std::size_t i, class Callable, class Variant>
constexpr decltype(auto) visitImpl(Callable&& callable, Variant&& variant,
                                   std::enable_if_t<(i == 11)>* = nullptr) {
  switch (variant.index()) {
  case 0:
    return std::forward<Callable>(callable)(
        pylir::get<0>(std::forward<Variant>(variant)));
  case 1:
    return std::forward<Callable>(callable)(
        pylir::get<1>(std::forward<Variant>(variant)));
  case 2:
    return std::forward<Callable>(callable)(
        pylir::get<2>(std::forward<Variant>(variant)));
  case 3:
    return std::forward<Callable>(callable)(
        pylir::get<3>(std::forward<Variant>(variant)));
  case 4:
    return std::forward<Callable>(callable)(
        pylir::get<4>(std::forward<Variant>(variant)));
  case 5:
    return std::forward<Callable>(callable)(
        pylir::get<5>(std::forward<Variant>(variant)));
  case 6:
    return std::forward<Callable>(callable)(
        pylir::get<6>(std::forward<Variant>(variant)));
  case 7:
    return std::forward<Callable>(callable)(
        pylir::get<7>(std::forward<Variant>(variant)));
  case 8:
    return std::forward<Callable>(callable)(
        pylir::get<8>(std::forward<Variant>(variant)));
  case 9:
    return std::forward<Callable>(callable)(
        pylir::get<9>(std::forward<Variant>(variant)));
  case 10:
    return std::forward<Callable>(callable)(
        pylir::get<10>(std::forward<Variant>(variant)));
  case 11:
    return std::forward<Callable>(callable)(
        pylir::get<11>(std::forward<Variant>(variant)));
  default: PYLIR_UNREACHABLE;
  }
}

template <std::size_t i, class Callable, class Variant>
constexpr decltype(auto) visitImpl(Callable&& callable, Variant&& variant,
                                   std::enable_if_t<(i == 12)>* = nullptr) {
  switch (variant.index()) {
  case 0:
    return std::forward<Callable>(callable)(
        pylir::get<0>(std::forward<Variant>(variant)));
  case 1:
    return std::forward<Callable>(callable)(
        pylir::get<1>(std::forward<Variant>(variant)));
  case 2:
    return std::forward<Callable>(callable)(
        pylir::get<2>(std::forward<Variant>(variant)));
  case 3:
    return std::forward<Callable>(callable)(
        pylir::get<3>(std::forward<Variant>(variant)));
  case 4:
    return std::forward<Callable>(callable)(
        pylir::get<4>(std::forward<Variant>(variant)));
  case 5:
    return std::forward<Callable>(callable)(
        pylir::get<5>(std::forward<Variant>(variant)));
  case 6:
    return std::forward<Callable>(callable)(
        pylir::get<6>(std::forward<Variant>(variant)));
  case 7:
    return std::forward<Callable>(callable)(
        pylir::get<7>(std::forward<Variant>(variant)));
  case 8:
    return std::forward<Callable>(callable)(
        pylir::get<8>(std::forward<Variant>(variant)));
  case 9:
    return std::forward<Callable>(callable)(
        pylir::get<9>(std::forward<Variant>(variant)));
  case 10:
    return std::forward<Callable>(callable)(
        pylir::get<10>(std::forward<Variant>(variant)));
  case 11:
    return std::forward<Callable>(callable)(
        pylir::get<11>(std::forward<Variant>(variant)));
  case 12:
    return std::forward<Callable>(callable)(
        pylir::get<12>(std::forward<Variant>(variant)));
  default: PYLIR_UNREACHABLE;
  }
}

template <class Callable, class Variant>
constexpr decltype(auto) visit(Callable&& callable, Variant&& variant) {
  PYLIR_ASSERT(!variant.valueless_by_exception());
  return visitImpl<std::variant_size_v<std::decay_t<Variant>> - 1>(
      std::forward<Callable>(callable), std::forward<Variant>(variant));
}
} // namespace detail

template <typename Variant, typename... Matchers>
constexpr decltype(auto) match(Variant&& variant, Matchers&&... matchers) {
  PYLIR_ASSERT(!variant.valueless_by_exception());
  return detail::visit(detail::Overload{std::forward<Matchers>(matchers)...},
                       std::forward<Variant>(variant));
}
} // namespace pylir
