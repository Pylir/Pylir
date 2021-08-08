#include "Pylir.h"

#include <stdlib.h>

void* pylir_gc_alloc(size_t size)
{
    // TODO: GC :)
    return malloc(size);
}
