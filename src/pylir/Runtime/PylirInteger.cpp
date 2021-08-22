#include "Pylir.h"

#include "PylirCPP.hpp"

// TODO Big Integer

PylirInteger* pylir_integer_from_size_t(size_t value)
{
    auto object = pylir::rt::allocVar<PylirInteger>(pylir_integer_type_object, 0, sizeof(size_t));

    object->count = value;

    return object;
}

PylirInteger* pylir_integer_mul(PylirInteger* lhs, PylirInteger* rhs)
{
    auto object = pylir::rt::allocVar<PylirInteger>(pylir_integer_type_object, 0, sizeof(size_t));

    object->count = lhs->count * rhs->count;

    return object;
}

PylirIntegerToIndexResult pylir_integer_to_index(PylirInteger* value)
{
    return {static_cast<index>(value->count), false};
}
