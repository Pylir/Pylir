#pragma once

#include <cstdlib>
#include <utility>

namespace pylir::rt
{
template <class T, class... Args>
T& alloc(Args&&... args)
{
    //TODO: allocate slots
    T* mem = reinterpret_cast<T*>(std::malloc(sizeof(T)));
    return *new (mem) T(std::forward<Args>(args)...);
}
} // namespace pylir::rt
