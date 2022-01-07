#pragma once

#include <cstddef>
#include <cstdlib>
#include <type_traits>

namespace pylir::rt
{
class PyObject;

struct PyObjectHasher
{
    std::size_t operator()(PyObject* object) const noexcept;
};

struct PyObjectEqual
{
    bool operator()(PyObject* lhs, PyObject* rhs) const noexcept;
};

template <class T>
struct MallocAllocator
{
    using pointer = T*;
    using value_type = T;
    using const_void_pointer = const void*;
    using size_type = std::size_t;
    using is_always_equal = std::true_type;

    pointer allocate(std::size_t n) noexcept
    {
        return (pointer)std::malloc(sizeof(T) * n);
    }

    void deallocate(pointer p, std::size_t) noexcept
    {
        free(p);
    }

    constexpr bool operator==(const MallocAllocator&)
    {
        return true;
    }

    constexpr bool operator!=(const MallocAllocator&)
    {
        return false;
    }
};

} // namespace pylir::rt
