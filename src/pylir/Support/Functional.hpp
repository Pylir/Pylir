#pragma once

#include <functional>
#include <tuple>
#include <utility>

namespace pylir
{
namespace detail
{
template <class F, class... Args>
class BindFrontImpl
{
    F f;
    std::tuple<Args...> front;

public:
    explicit BindFrontImpl(F f, std::tuple<Args...> args) : f(std::move(f)), front(std::move(args)) {}

    template <class... Last>
    decltype(auto) operator()(Last&&... last) & noexcept(std::is_nothrow_invocable_v<F, Args..., Last...>)
    {
        return std::apply(
            [&](auto&&... members) -> decltype(auto)
            { return std::invoke(f, std::forward<decltype(members)>(members)..., std::forward<Last>(last)...); },
            front);
    }

    template <class... Last>
    decltype(auto) operator()(Last&&... last) const& noexcept(std::is_nothrow_invocable_v<F, Args..., Last...>)
    {
        return std::apply(
            [&](auto&&... members) -> decltype(auto)
            { return std::invoke(f, std::forward<decltype(members)>(members)..., std::forward<Last>(last)...); },
            front);
    }

    template <class... Last>
    decltype(auto) operator()(Last&&... last) && noexcept(std::is_nothrow_invocable_v<F, Args..., Last...>)
    {
        return std::apply(
            [&](auto&&... members) -> decltype(auto) {
                return std::invoke(std::move(f), std::forward<decltype(members)>(members)...,
                                   std::forward<Last>(last)...);
            },
            std::move(front));
    }

    template <class... Last>
    decltype(auto) operator()(Last&&... last) const&& noexcept(std::is_nothrow_invocable_v<F, Args..., Last...>)
    {
        return std::apply(
            [&](auto&&... members) -> decltype(auto) {
                return std::invoke(std::move(f), std::forward<decltype(members)>(members)...,
                                   std::forward<Last>(last)...);
            },
            std::move(front));
    }
};

} // namespace detail

template <class F, class... Args>
auto bind_front(F&& f, Args&&... args)
{
    return detail::BindFrontImpl(std::forward<F>(f), std::make_tuple(std::forward<Args>(args)...));
}

} // namespace pylir
