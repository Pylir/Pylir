
#pragma once

#include <cstdint>
#include <iterator>
#include <vector>

#include "Macros.hpp"

namespace pylir
{
template <class ValueType, class NextFunc>
class LazyCacheIterator
{
    const std::vector<ValueType>* m_cache;
    std::size_t m_index;
    NextFunc m_nextFunc; // TODO: No unique address in C++20

public:
    using difference_type = std::ptrdiff_t;
    using value_type = ValueType;
    using pointer = const value_type*;
    using reference = const value_type&;
    using iterator_category = std::forward_iterator_tag;

    LazyCacheIterator() = default;

    LazyCacheIterator(std::vector<ValueType>& cache, std::size_t index, NextFunc nextFunc = NextFunc{})
        : m_cache(&cache), m_index(index), m_nextFunc(std::move(nextFunc))
    {
        if (index == 0 && m_cache->empty())
        {
            m_nextFunc();
        }
    }

    reference operator*() const
    {
        return (*m_cache)[m_index];
    }

    LazyCacheIterator& operator++()
    {
        if (m_index + 1 >= m_cache->size())
        {
            m_nextFunc();
        }
        m_index++;
        return *this;
    }

    LazyCacheIterator operator++(int)
    {
        auto copy = *this;
        operator++();
        return copy;
    }

    bool operator==(const LazyCacheIterator& rhs) const
    {
        if (m_cache != rhs.m_cache)
        {
            return false;
        }
        bool bothPastEnd = m_index >= m_cache->size() && rhs.m_index >= m_cache->size();
        if (bothPastEnd)
        {
            return true;
        }
        return m_index == rhs.m_index;
    }

    bool operator!=(const LazyCacheIterator& rhs) const
    {
        return !(rhs == *this);
    }

    difference_type operator-(const LazyCacheIterator& rhs) const
    {
        PYLIR_ASSERT(m_index != static_cast<std::size_t>(-1));
        return m_index - rhs.m_index;
    }

    friend void swap(LazyCacheIterator& lhs, LazyCacheIterator& rhs)
    {
        std::swap(lhs.m_cache, rhs.m_cache);
        std::swap(lhs.m_index, rhs.m_index);
        std::swap(lhs.m_nextFunc, rhs.m_nextFunc);
    }
};

} // namespace pylir
