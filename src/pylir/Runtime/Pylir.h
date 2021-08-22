
#pragma once

#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C"
{
#endif

    struct PylirTypeObject;

    typedef struct PylirObject
    {
        PylirTypeObject* m_type;
    } PylirObject;

    void* pylir_gc_alloc(size_t size);

    extern PylirTypeObject* pylir_integer_type_object;

    typedef struct PylirInteger
    {
        PylirObject base;
        size_t count;
        /* size_t values[]; */
    } PylirInteger;

    PylirInteger* pylir_integer_from_size_t(size_t value);

    PylirInteger* pylir_integer_mul(PylirInteger* lhs, PylirInteger* rhs);

    // In MLIR IR this has no sign. In the runtime it is signed. Basically a signed version of size_t
    typedef ptrdiff_t index;

    typedef struct PylirIntegerToIndexResult
    {
        index value;
        bool overflow;
    } PylirIntegerToIndexResult;

    PylirIntegerToIndexResult pylir_integer_to_index(PylirInteger* value);

#ifdef __cplusplus
}
#endif
