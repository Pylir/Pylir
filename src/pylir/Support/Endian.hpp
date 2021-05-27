
#pragma once

#include <type_traits>

namespace pylir
{
// TODO: To be replaced in C++20
enum class endian
{
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

template <class Integral>
[[nodiscard]] Integral swapByteOrder(Integral value)
{
    static_assert(std::is_integral_v<Integral>);
    if constexpr (sizeof(value) == 1)
    {
        return value;
    }
    else if constexpr (sizeof(value) == 2)
    {
#ifdef _MSC_VER
        return _byteswap_ushort(value);
#else
        return __builtin_bswap16(value);
#endif
    }
    else if constexpr (sizeof(value) == 4)
    {
#ifdef _MSC_VER
        return _byteswap_ulong(value);
#else
        return __builtin_bswap32(value);
#endif
    }
    else if constexpr (sizeof(value) == 8)
    {
#ifdef _MSC_VER
        return _byteswap_uint64(value);
#else
        return __builtin_bswap64(value);
#endif
    }
}

} // namespace pylir
