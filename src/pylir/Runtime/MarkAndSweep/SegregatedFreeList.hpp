#pragma once

#include <pylir/Runtime/Objects.hpp>
#include <pylir/Runtime/Pages.hpp>

#include <memory>
#include <vector>

namespace pylir::rt
{

class SegregatedFreeList
{
    std::size_t m_sizeClass;
    std::byte* m_head = nullptr;
    std::vector<PagePtr> m_pages;

    PagePtr newPage() const;

public:
    explicit SegregatedFreeList(std::size_t sizeClass) : m_sizeClass(sizeClass) {}

    ~SegregatedFreeList();

    PyObject* nextCell();

    void sweep();
};
} // namespace pylir::rt
