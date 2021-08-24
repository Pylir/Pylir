#include "Pylir.h"

#include <algorithm>

#include "PylirCPP.hpp"

PylirString* pylir_string_from_utf32(PylirTypeObject* typeObject, const uint32_t* data, size_t count)
{
    auto* object = pylir::rt::allocVar<PylirString>(typeObject, count, sizeof(uint32_t));

    std::copy(data, data + count, pylir::rt::begin(object));

    return object;
}
