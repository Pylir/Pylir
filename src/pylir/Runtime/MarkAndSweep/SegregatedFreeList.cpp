//  Licensed under the Apache License v2.0 with LLVM Exceptions.
//  See https://llvm.org/LICENSE.txt for license information.
//  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "SegregatedFreeList.hpp"

#include <cstring>

#include "MarkAndSweep.hpp"

pylir::rt::PyObject* pylir::rt::SegregatedFreeList::nextCell()
{
    std::byte* cell;
    if (!m_head)
    {
        if (!m_pages.empty())
        {
            gc.collect();
            if (!m_head)
            {
                cell = m_pages.emplace_back(newPage()).get();
            }
            else
            {
                cell = m_head;
            }
        }
        else
        {
            cell = m_pages.emplace_back(newPage()).get();
        }
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

void pylir::rt::SegregatedFreeList::sweep()
{
    std::byte* newHead = nullptr;
    std::byte* lastFree = nullptr;
    auto iter = m_pages.begin();
    for (; iter != m_pages.end(); iter++)
    {
        auto* end = getEndCell(*iter, m_sizeClass);
        for (std::byte* begin = iter->get(); begin != end; begin += m_sizeClass)
        {
            if (begin == m_head)
            {
                if (lastFree)
                {
                    std::memcpy(lastFree, &m_head, sizeof(std::byte*));
                }
                else
                {
                    newHead = m_head;
                }
                lastFree = m_head;
                std::memcpy(&m_head, m_head, sizeof(std::byte*));
                continue;
            }
            auto* object = reinterpret_cast<PyObject*>(begin);
            if (object->getMark<bool>())
            {
                object->clearMarking();
                continue;
            }
            destroyPyObject(*object);
            if (!newHead)
            {
                newHead = reinterpret_cast<std::byte*>(object);
                lastFree = newHead;
            }
            else
            {
                std::memcpy(lastFree, &object, sizeof(std::byte*));
                lastFree = reinterpret_cast<std::byte*>(object);
            }
        }
    }
    if (lastFree)
    {
        std::byte* nullPointer = nullptr;
        std::memcpy(lastFree, &nullPointer, sizeof(std::byte*));
    }
    m_head = newHead;
}
