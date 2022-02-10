
#include <cstdlib>

#include "API.hpp"

void* pylir_gc_alloc(size_t size)
{
    // TODO: GC :)
    return std::malloc(size);
}
