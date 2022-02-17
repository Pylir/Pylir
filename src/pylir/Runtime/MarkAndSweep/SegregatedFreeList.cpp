#include "SegregatedFreeList.hpp"

#include <cstring>

pylir::rt::PyObject* pylir::rt::SegregatedFreeList::nextCell()
{
    std::byte* cell;
    if (!m_head)
    {
        cell = m_pages.emplace_back(newPage()).get();
    }
    else
    {
        cell = m_head;
    }
    std::memcpy(&m_head, cell, sizeof(std::byte*));
    return reinterpret_cast<PyObject*>(cell);
}

namespace
{
std::byte* getEndCell(const pylir::rt::PagePtr& pagePtr, std::size_t sizeClass)
{
    return pagePtr.get() + ((pagePtr.size() / sizeClass) * sizeClass);
}
} // namespace

pylir::rt::PagePtr pylir::rt::SegregatedFreeList::newPage() const
{
    auto result = pageAlloc(1);
    auto* end = getEndCell(result, m_sizeClass) - m_sizeClass;
    for (std::byte* begin = result.get(); begin != end; begin += m_sizeClass)
    {
        std::byte* nextCellAddress = begin + m_sizeClass;
        std::memcpy(begin, &nextCellAddress, sizeof(std::byte*));
    }
    return result;
}

pylir::rt::SegregatedFreeList::~SegregatedFreeList()
{
    auto iter = m_pages.begin();
    for (; iter != m_pages.end(); iter++)
    {
        auto* end = getEndCell(*iter, m_sizeClass);
        if (m_head >= iter->get() && m_head < end)
        {
            break;
        }
        for (std::byte* begin = iter->get(); begin != end; begin += m_sizeClass)
        {
            destroyPyObject(*reinterpret_cast<PyObject*>(begin));
        }
    }
    for (; iter != m_pages.end(); iter++)
    {
        auto* end = getEndCell(*iter, m_sizeClass);
        for (std::byte* begin = iter->get(); begin != end; begin += m_sizeClass)
        {
            if (begin == m_head)
            {
                std::memcpy(&m_head, begin, sizeof(std::byte*));
                continue;
            }
            destroyPyObject(*reinterpret_cast<PyObject*>(begin));
        }
    }
}
