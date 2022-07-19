//  Copyright 2022 Markus Böck
//
//  Licensed under the Apache License v2.0 with LLVM Exceptions.
//  See https://llvm.org/LICENSE.txt for license information.
//  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <pylir/Support/Macros.hpp>

#include <type_traits>
#include <utility>

namespace pylir::Diag
{
template <class T, class = void>
struct LocationProvider;

template <class T>
struct LocationProvider<T, std::enable_if_t<std::is_integral_v<T>>>
{
    static std::pair<std::size_t, std::size_t> getRange(T value) noexcept
    {
        return {value, value + 1};
    }
};

template <class T, class = void>
struct hasLocationProviderOneArg : std::false_type
{
};

template <class T>
struct hasLocationProviderOneArg<
    T, std::void_t<decltype(LocationProvider<std::decay_t<T>>::getRange(std::declval<std::decay_t<T>>()))>>
    : std::true_type
{
};

template <class T, class = void>
struct hasLocationProviderTwoArg : std::false_type
{
};

template <class T>
struct hasLocationProviderTwoArg<T, std::void_t<decltype(LocationProvider<std::decay_t<T>>::getRange(
                                        std::declval<std::decay_t<T>>(), std::declval<const void*>()))>>
    : std::true_type
{
};

constexpr auto range = [](const auto& value, const void* context = nullptr)
{
    using T = std::decay_t<decltype(value)>;
    if constexpr (hasLocationProviderTwoArg<T>{})
    {
        if (context)
        {
            return LocationProvider<T>::getRange(value, context);
        }
    }
    if constexpr (hasLocationProviderOneArg<T>{})
    {
        return LocationProvider<T>::getRange(value);
    }
    PYLIR_UNREACHABLE;
};

template <class T>
constexpr bool hasLocationProvider_v = std::disjunction_v<hasLocationProviderOneArg<T>, hasLocationProviderTwoArg<T>>;

} // namespace pylir::Diag
