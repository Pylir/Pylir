
#pragma once

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
    auto object = reinterpret_cast<T*>(pylir_gc_alloc(sizeof(T) + count * elementSize));
    auto& [base, size] = *object;
    base.m_type = typeObject;
    size = count;
    return object;
}

} // namespace pylir::rt
