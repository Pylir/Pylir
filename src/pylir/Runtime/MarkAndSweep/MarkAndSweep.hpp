// Copyright 2022 Markus Böck
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <pylir/Runtime/Objects.hpp>

#include "BestFitTree.hpp"
#include "SegregatedFreeList.hpp"

namespace pylir::rt
{
class MarkAndSweep
{
    SegregatedFreeList m_unit2{2 * alignof(std::max_align_t)};
    SegregatedFreeList m_unit4{4 * alignof(std::max_align_t)};
    SegregatedFreeList m_unit6{6 * alignof(std::max_align_t)};
    SegregatedFreeList m_unit8{8 * alignof(std::max_align_t)};
    BestFitTree m_tree{8 * alignof(std::max_align_t)};

public:
    PyObject* alloc(std::size_t count);

    void collect();
};

extern MarkAndSweep gc;

} // namespace pylir::rt
