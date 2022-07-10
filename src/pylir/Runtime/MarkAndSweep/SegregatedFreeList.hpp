// Copyright 2022 Markus Böck
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

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

    [[nodiscard]] PagePtr newPage() const;

public:
    explicit SegregatedFreeList(std::size_t sizeClass) : m_sizeClass(sizeClass) {}

    ~SegregatedFreeList();
    SegregatedFreeList(SegregatedFreeList&&) noexcept = default;
    SegregatedFreeList& operator=(SegregatedFreeList&&) noexcept = default;
    SegregatedFreeList(const SegregatedFreeList&) = delete;
    SegregatedFreeList& operator=(const SegregatedFreeList&) = delete;

    PyObject* nextCell();

    void sweep();
};
} // namespace pylir::rt
