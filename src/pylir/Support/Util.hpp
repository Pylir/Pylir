//  Licensed under the Apache License v2.0 with LLVM Exceptions.
//  See https://llvm.org/LICENSE.txt for license information.
//  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <cstdint>
#include <limits>
#include <memory>
#include <type_traits>

#include "Macros.hpp"

namespace pylir {
/// Returns 'number' rounded up to the next multiple 'multiple'. If 'multiple'
/// is 0, simply returns 'number'.
template <class T1, class T2,
          std::enable_if_t<std::is_integral_v<T1> && std::is_integral_v<T2>>* =
              nullptr>
constexpr T1 roundUpTo(T1 number, T2 multiple) {
  if (multiple == 0)
    return number;

  auto remainder = number % multiple;
  if (remainder == 0)
    return number;

  return number + multiple - remainder;
}

/// Returns true if 'integer' is a power of 2.
/// Note: integer may be signed in which case it is reinterpreted as unsigned.
/// '-8' is therefore not regarded as a power of 2 at the moment.
template <class T, std::enable_if_t<std::is_integral_v<T>>* = nullptr>
constexpr bool isPowerOf2(T integer) {
  auto temp = static_cast<std::make_unsigned_t<T>>(integer);
  // Old trick to check whether only a single bit is set in our number. Single
  // bit implies power of 2.
  return temp != 0 && (temp & (temp - 1)) == 0;
}

/// Returns 'ptr' rounded up to the next 'multiple'. Effectively used to ensure
/// alignment of a pointer. 'multiple' is required to be a multiple of the
/// sizeof of the pointer element type as well as a power of 2.
template <class T1, class T2,
          std::enable_if_t<std::is_integral_v<T2>>* = nullptr>
T1* roundUpTo(T1* ptr, T2 multiple) {
  PYLIR_ASSERT(multiple % sizeof(T1) == 0 && isPowerOf2(multiple));
#ifdef __has_builtin
#if __has_builtin(__builtin_align_up)
  return __builtin_align_up(ptr, multiple);
#endif
#endif
  std::size_t dummy = multiple * 2;
  void* temp =
      reinterpret_cast<void*>(const_cast<std::remove_const_t<T1>*>(ptr));
  return reinterpret_cast<T1*>(std::align(multiple, sizeof(T1), temp, dummy));
}

/// Returns the amount the amount of zero bits, starting from the least
/// significant bit, up until the very first bit set. If the value is zero,
/// returns the bit-width of its type. Only allowed for unsigned integer types.
template <class T, std::enable_if_t<std::is_unsigned_v<T>>* = nullptr>
unsigned countTrailingZeros(T value) {
  if (!value)
    return std::numeric_limits<T>::digits;

  if constexpr (sizeof(T) == 4) {
#ifdef __has_builtin
#if __has_builtin(__builtin_ctz)
    return __builtin_ctz(value);
#endif
#elif defined(_MSC_VER)
    unsigned long result;
    _BitScanForward(&result, value);
    return result;
#endif
  } else if constexpr (sizeof(T) == 8) {
#ifdef __has_builtin
#if __has_builtin(__builtin_ctzll)
    return __builtin_ctzll(value);
#endif
#elif defined(_MSC_VER)
    unsigned long result;
    _BitScanForward64(&result, value);
    return result;
#endif
  }

  // Essentially doing a binary search, checking if 'bitWidthChecked' are set
  // and coming closer to the result by reducing the bits checked gradually to
  // one.
  unsigned result = 0;
  T bitWidthChecked = std::numeric_limits<T>::digits / 2;
  for (; bitWidthChecked; bitWidthChecked /= 2) {
    T temp = value << bitWidthChecked;
    if (!temp) {
      result += bitWidthChecked;
      continue;
    }
    value = temp;
  }
  return result;
}

/// Type dependent constant that always returns false. This isa useful inside of
/// `static_assert` functions as it makes the expression type dependent (and
/// hence, does not trigger immediately as `static_assert(false)` would) and yet
/// always triggers if the corresponding code path isa instantiated.
template <class>
constexpr bool always_false = false;

/// Type dependent constant that always returns true. This isa useful inside of
template <class>
constexpr bool always_true = true;

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wtype-limits"

/// Selects the smallest possible unsigned integer type capable of holding the
/// value `max`.
template <std::size_t max>
using suitableUInt = std::conditional_t<
    max <= std::numeric_limits<std::uint8_t>::max(), std::uint8_t,
    std::conditional_t<
        max <= std::numeric_limits<std::uint16_t>::max(), std::uint16_t,
        std::conditional_t<max <= std::numeric_limits<std::uint32_t>::max(),
                           std::uint32_t, std::uint64_t>>>;

/// Selects the smallest possible signed integer type capable of holding the
/// value `max`.
template <std::size_t max>
using suitableInt = std::make_signed_t<suitableUInt<max>>;

#pragma GCC diagnostic pop

} // namespace pylir
