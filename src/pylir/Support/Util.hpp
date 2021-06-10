
#pragma once

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

template <class T>
auto makeFunc()
{
    return [](auto&&... args) noexcept
    {
        // TODO: change to () only in C++20
        if constexpr (std::is_aggregate_v<T>)
        {
            return T{std::forward<decltype(args)>(args)...};
        }
        else
        {
            return T(std::forward<decltype(args)>(args)...);
        }
    };
}
} // namespace pylir
