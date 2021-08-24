
#pragma once

#include <cstdint>

#include "Pylir.h"

namespace pylir::rt
{
template <class T>
T* alloc(PylirTypeObject* typeObject)
{
    auto object = reinterpret_cast<T*>(pylir_gc_alloc(sizeof(T)));
    reinterpret_cast<PylirObject*>(object)->m_type = typeObject;
    return object;
}

template <class T>
T* allocVar(PylirTypeObject* typeObject, size_t count, size_t elementSize)
{
    auto object = reinterpret_cast<T*>(
        pylir_gc_alloc(sizeof(T) + count * elementSize + (typeObject->m_dictPtr != 0 ? sizeof(PylirObject*) : 0)));
    auto& [base, size] = *object;
    base.m_type = typeObject;
    size = count;
    return object;
}

inline std::uint32_t* begin(PylirString* string)
{
    return reinterpret_cast<std::uint32_t*>(reinterpret_cast<char*>(string) + sizeof(PylirString));
}

inline std::uint32_t* end(PylirString* string)
{
    return begin(string) + string->m_count;
}

} // namespace pylir::rt
