//  Licensed under the Apache License v2.0 with LLVM Exceptions.
//  See https://llvm.org/LICENSE.txt for license information.
//  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <pylir/Runtime/Objects.hpp>

#include <cstddef>
#include <cstdint>
#include <cstdlib>

#include "MarkAndSweep.hpp"

extern "C" void* pylir_gc_alloc(std::size_t size)
{
    return pylir::rt::gc.alloc(size);
}
