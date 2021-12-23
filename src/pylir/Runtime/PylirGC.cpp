#include "API.hpp"

#include <cstdlib>

void* pylir_gc_alloc(size_t size)
{
    // TODO: GC :)
    return std::malloc(size);
}
