//  Licensed under the Apache License v2.0 with LLVM Exceptions.
//  See https://llvm.org/LICENSE.txt for license information.
//  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <llvm/ADT/identity.h>

#include <array>
#include <cstdint>
#include <memory>

#include "Macros.hpp"
#include "Util.hpp"
#include "Variant.hpp"

namespace pylir {
namespace detail {
template <class T, std::size_t i, class First, class... Rest>
[[nodiscard]] constexpr std::size_t getIndex() noexcept {
  if constexpr (std::is_same_v<T, First>)
    return i;
  else if constexpr (sizeof...(Rest) == 0)
    static_assert(
        always_false<T>,
        "Template type was not specified in the sets of possible base classes");
  else
    return getIndex<T, i + 1, Rest...>();
}

} // namespace detail

template <class ConcreteType, class... Args>
class AbstractIntrusiveVariant {
  suitableUInt<sizeof...(Args) - 1> m_index;

  template <class First, class...>
  static First first() {
    PYLIR_UNREACHABLE;
  }

  template <class T>
  class CRTP : public ConcreteType {
  public:
    constexpr CRTP() : ConcreteType(llvm::identity<T>{}) {}

    static bool classof(const ConcreteType* variant) {
      return variant->index() == indexOf<T>();
    }
  };

public:
  template <class T>
  using Base = CRTP<T>;

  template <class T>
  constexpr explicit AbstractIntrusiveVariant(llvm::identity<T>)
      : m_index(std::integral_constant<index_type, indexOf<T>()>::value) {
    static_assert((std::is_same_v<T, Args> || ...),
                  "T must be one of the subclasses specified in Args...");
  }

  using index_type = decltype(m_index);

  template <class T>
  constexpr static index_type indexOf() noexcept {
    return detail::getIndex<T, 0, Args...>();
  }

  [[nodiscard]] constexpr index_type index() const noexcept {
    return m_index;
  }

  template <class... Ts>
  [[nodiscard]] constexpr bool isa() const noexcept {
    return (
        (index() == std::integral_constant<index_type, indexOf<Ts>()>::value) ||
        ...);
  }

  template <class T>
  [[nodiscard]] constexpr const T& cast() const noexcept {
    PYLIR_ASSERT(isa<T>());
    return *static_cast<const T*>(this);
  }

  template <class T>
  [[nodiscard]] constexpr T& cast() noexcept {
    PYLIR_ASSERT(isa<T>());
    return *static_cast<T*>(this);
  }

  template <class T>
  [[nodiscard]] constexpr const T* dyn_cast() const noexcept {
    constexpr auto var = indexOf<T>();
    if (index() != var)
      return nullptr;

    return static_cast<const T*>(this);
  }

  template <class T>
  [[nodiscard]] constexpr T* dyn_cast() noexcept {
    constexpr auto var = indexOf<T>();
    if (index() != var)
      return nullptr;

    return static_cast<T*>(this);
  }

  template <class... F>
  decltype(auto) match(F&&... f) {
    using Callable = decltype(detail::Overload{std::forward<F>(f)...});
    constexpr std::array<
        std::invoke_result_t<Callable, decltype(first<Args...>())&> (*)(
            AbstractIntrusiveVariant&, Callable &&),
        sizeof...(Args)>
        calling = {{+[](AbstractIntrusiveVariant& base,
                        Callable&& callable) -> decltype(auto) {
          return callable(static_cast<Args&>(base));
        }...}};
    return calling[index()](*this, Callable{std::forward<F>(f)...});
  }

  template <class... F>
  decltype(auto) match(F&&... f) const {
    using Callable = decltype(detail::Overload{std::forward<F>(f)...});
    constexpr std::array<
        std::invoke_result_t<Callable, const decltype(first<Args...>())&> (*)(
            const AbstractIntrusiveVariant&, Callable&&),
        sizeof...(Args)>
        calling = {{+[](const AbstractIntrusiveVariant& base,
                        Callable&& callable) -> decltype(auto) {
          return callable(static_cast<const Args&>(base));
        }...}};
    return calling[index()](*this, Callable{std::forward<F>(f)...});
  }
};

namespace detail::AbstractIntrusiveVariant {
template <class Base, class... Args>
auto deduceArgs(::pylir::AbstractIntrusiveVariant<Base, Args...>*)
    -> ::pylir::AbstractIntrusiveVariant<Base, Args...>;

void deduceArgs(...);
} // namespace detail::AbstractIntrusiveVariant

template <class T,
          class U = decltype(detail::AbstractIntrusiveVariant::deduceArgs(
              std::declval<T*>()))>
class IntrusiveVariantDeleter {
  static_assert(
      always_false<T>,
      "Can't delete pointer that isn't a subclass of AbstractIntrusiveVariant");
};

template <class Base, class... SubClasses>
class IntrusiveVariantDeleter<Base,
                              AbstractIntrusiveVariant<Base, SubClasses...>> {
public:
  IntrusiveVariantDeleter() = default;

  template <class T, std::enable_if_t<std::disjunction_v<
                         std::is_same<T, SubClasses>...>>* = nullptr>
  /*implicit*/ IntrusiveVariantDeleter(std::default_delete<T>&&) noexcept {}

  void operator()(Base* pointer) const noexcept {
    constexpr std::array<void (*)(Base*), sizeof...(SubClasses)> deleteFuncs = {
        {+[](Base* ptr) { delete static_cast<SubClasses*>(ptr); }...}};
    deleteFuncs[pointer->index()](pointer);
  }
};

template <class T>
using IntrVarPtr = std::unique_ptr<T, IntrusiveVariantDeleter<T>>;

template <class T,
          class U = decltype(detail::AbstractIntrusiveVariant::deduceArgs(
              std::declval<T*>()))>
struct IsAbstractVariantConcrete : std::false_type {};

template <class T, class... SubClasses>
struct IsAbstractVariantConcrete<T, AbstractIntrusiveVariant<T, SubClasses...>>
    : std::true_type {};

} // namespace pylir
