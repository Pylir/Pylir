//  Licensed under the Apache License v2.0 with LLVM Exceptions.
//  See https://llvm.org/LICENSE.txt for license information.
//  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "BestFitTree.hpp"
#include "SegregatedFreeList.hpp"

namespace pylir::rt {
class MarkAndSweep {
  // Maximum useful alignment on the target as determined by the compiler. This
  // is what is used in libunwind for the exception object. The alignment of
  // `std::maxalign_t` with clang-cl is notably less.
  struct MaxAligned {
  } __attribute__((__aligned__));

  SegregatedFreeList m_unit2{2 * alignof(MaxAligned)};
  SegregatedFreeList m_unit4{4 * alignof(MaxAligned)};
  SegregatedFreeList m_unit6{6 * alignof(MaxAligned)};
  SegregatedFreeList m_unit8{8 * alignof(MaxAligned)};
  BestFitTree m_tree{8 * alignof(MaxAligned)};

public:
  ~MarkAndSweep() {
    m_unit2.finalize();
    m_unit4.finalize();
    m_unit6.finalize();
    m_unit8.finalize();
    m_tree.finalize();
  }

  PyObject* alloc(std::size_t count);

  void collect();
};

extern MarkAndSweep gc;

} // namespace pylir::rt
