#include "Pylir.h"

#include "PylirCPP.hpp"

// TODO Big Integer

PylirInteger* pylir_integer_from_size_t(PylirTypeObject* typeObject, size_t value)
{
    auto object = pylir::rt::allocVar<PylirInteger>(typeObject, 0, sizeof(size_t));

    object->m_count = value;

    return object;
}

PylirInteger* pylir_integer_mul(PylirInteger* lhs, PylirInteger* rhs)
{
    auto object = pylir::rt::allocVar<PylirInteger>(&pylir_integer_type_object, 0, sizeof(size_t));

    object->m_count = lhs->m_count * rhs->m_count;

    return object;
}

PylirIntegerToIndexResult pylir_integer_to_index(PylirInteger* value)
{
    return {static_cast<index>(value->m_count), false};
}
