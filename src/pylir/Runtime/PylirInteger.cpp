#include "Pylir.h"

// TODO Big Integer

PylirIntegerValue pylir_integer_from_size_t(size_t value)
{
    return {reinterpret_cast<void*>(value)};
}

PylirIntegerValue pylir_integer_mul(PylirIntegerValue lhs, PylirIntegerValue rhs)
{
    return {reinterpret_cast<void*>(reinterpret_cast<size_t>(lhs.opaque) * reinterpret_cast<size_t>(rhs.opaque))};
}
