//  Licensed under the Apache License v2.0 with LLVM Exceptions.
//  See https://llvm.org/LICENSE.txt for license information.
//  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <type_traits>

namespace pylir {
/// Drop in replacement for std::endian until C++20 is used in the codebase.
/// NOLINTBEGIN(readability-identifier-naming)
enum class endian {
#ifdef _WIN32
  little = 0,
  big = 1,
  native = little
#else
  little = __ORDER_LITTLE_ENDIAN__,
  big = __ORDER_BIG_ENDIAN__,
  native = __BYTE_ORDER__
#endif
};
// NOLINTEND(readability-identifier-naming)

template <class Integral>
[[nodiscard]] Integral swapByteOrder(Integral value) {
  static_assert(std::is_integral_v<Integral>);
  if constexpr (sizeof(value) == 1)
    return value;
  else if constexpr (sizeof(value) == 2)
#ifdef _MSC_VER
    return _byteswap_ushort(value);
#else
    return __builtin_bswap16(value);
#endif
  else if constexpr (sizeof(value) == 4)
#ifdef _MSC_VER
    return _byteswap_ulong(value);
#else
    return __builtin_bswap32(value);
#endif
  else if constexpr (sizeof(value) == 8)
#ifdef _MSC_VER
    return _byteswap_uint64(value);
#else
    return __builtin_bswap64(value);
#endif
}

} // namespace pylir
