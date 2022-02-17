#include <pylir/Runtime/Objects.hpp>

#include <cstddef>
#include <cstdint>
#include <cstdlib>

#include "MarkAndSweep.hpp"

extern "C" void* pylir_gc_alloc(std::size_t size)
{
    return pylir::rt::gc.alloc(size);
}
