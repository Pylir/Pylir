#pragma once

#include <pylir/Support/Util.hpp>

#include <cstddef>
#include <memory>

namespace pylir::rt
{
std::size_t getPageSize();

struct PageDeleter
{
    std::size_t size;

    void operator()(std::byte* page) const noexcept;
};

class PagePtr : private std::unique_ptr<std::byte, PageDeleter>
{
public:
    PagePtr(std::byte* memory, std::size_t length)
        : std::unique_ptr<std::byte, PageDeleter>{memory, PageDeleter{length}}
    {
    }

    using std::unique_ptr<std::byte, PageDeleter>::get;
    using std::unique_ptr<std::byte, PageDeleter>::operator*;

    std::size_t size() const noexcept
    {
        return get_deleter().size;
    }
};

PagePtr pageAlloc(std::size_t pageCount);

inline PagePtr pageAllocBytes(std::size_t bytes)
{
    auto pageSize = getPageSize();
    auto div = bytes / pageSize;
    bool rest = bytes % pageSize;
    return pageAlloc(div + rest);
}
} // namespace pylir::rt
