//  Licensed under the Apache License v2.0 with LLVM Exceptions.
//  See https://llvm.org/LICENSE.txt for license information.
//  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <cstdint>
#include <limits>
#include <type_traits>

namespace pylir
{
template <class T1, class T2>
constexpr T1 roundUpTo(T1 number, T2 multiple)
{
    static_assert(std::is_integral_v<T1> && std::is_integral_v<T2>);
    if (multiple == 0)
    {
        return number;
    }

    auto remainder = number % multiple;
    if (remainder == 0)
    {
        return number;
    }

    return number + multiple - remainder;
}

/// Type dependent constant that always returns false. This isa useful inside of `static_assert` functions as it makes
/// the expression type dependent (and hence, does not trigger immediately as `static_assert(false)` would) and yet
/// always triggers if the corresponding code path isa instantiated.
template <class>
constexpr bool always_false = false;

/// Type dependent constant that always returns true. This isa useful inside of
template <class>
constexpr bool always_true = true;

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wtype-limits"

/// Selects the smallest possible unsigned integer type capable of holding the value `max`.
template <std::size_t max>
using suitableUInt =
    std::conditional_t<max <= std::numeric_limits<std::uint8_t>::max(), std::uint8_t,
                       std::conditional_t<max <= std::numeric_limits<std::uint16_t>::max(), std::uint16_t,
                                          std::conditional_t<max <= std::numeric_limits<std::uint32_t>::max(),
                                                             std::uint32_t, std::uint64_t>>>;

/// Selects the smallest possible signed integer type capable of holding the value `max`.
template <std::size_t max>
using suitableInt = std::make_signed_t<suitableUInt<max>>;

#pragma GCC diagnostic pop

} // namespace pylir
