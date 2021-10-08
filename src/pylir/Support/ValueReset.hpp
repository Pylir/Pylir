#pragma once

#include <type_traits>
#include <utility>

namespace pylir
{
template <class T>
class ValueReset
{
    static_assert(!std::is_const_v<T>);

    T m_valueAfter;
    T* m_assignedTo;

public:
    template <class U = T>
    ValueReset(T& assignedTo, U valueAfter) : m_valueAfter(valueAfter), m_assignedTo(&assignedTo)
    {
    }

    explicit ValueReset(T& assignedTo) : ValueReset(assignedTo, assignedTo) {}

    ~ValueReset()
    {
        if (m_assignedTo)
        {
            *m_assignedTo = m_valueAfter;
        }
    }

    ValueReset(const ValueReset&) = delete;
    ValueReset& operator=(const ValueReset&) = delete;

    ValueReset(ValueReset&& rhs) noexcept
    {
        m_assignedTo = std::exchange(rhs.m_assignedTo, nullptr);
        m_valueAfter = rhs.m_valueAfter;
    }

    ValueReset& operator=(ValueReset&& rhs) noexcept
    {
        m_assignedTo = std::exchange(rhs.m_assignedTo, nullptr);
        m_valueAfter = rhs.m_valueAfter;
        return *this;
    }
};

} // namespace pylir
