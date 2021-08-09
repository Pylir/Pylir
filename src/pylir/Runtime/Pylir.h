
#pragma once

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

    typedef struct PylirIntegerValue
    {
        void* opaque;
    } PylirIntegerValue;

    PylirIntegerValue pylir_integer_from_size_t(size_t value);

    PylirIntegerValue pylir_integer_mul(PylirIntegerValue lhs, PylirIntegerValue rhs);

    typedef struct PylirInteger
    {
        PylirObject base;
        PylirIntegerValue value;
    } PylirInteger;

#ifdef __cplusplus
}
#endif
