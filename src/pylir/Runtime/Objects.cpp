// Copyright 2022 Markus Böck
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "Objects.hpp"

#include <pylir/Support/Macros.hpp>

using namespace pylir::rt;

static_assert(std::is_standard_layout_v<PyObject>);
static_assert(std::is_standard_layout_v<PyTypeObject>);
static_assert(std::is_standard_layout_v<PyFunction>);
static_assert(std::is_standard_layout_v<PyList>);
static_assert(std::is_standard_layout_v<PyTuple>);
static_assert(std::is_standard_layout_v<PyString>);
static_assert(std::is_standard_layout_v<PyDict>);
static_assert(std::is_standard_layout_v<PyInt>);
static_assert(std::is_standard_layout_v<PyBaseException>);

PyObject* PyObject::getSlot(int index)
{
    return reinterpret_cast<PyObject**>(this)[type(*this).m_offset + index];
}

PyObject* PyObject::getSlot(std::string_view name)
{
    auto& slotsTuple = type(*this).getSlot(PyTypeObject::Slots)->cast<PyTuple>();
    for (std::size_t i = 0; i < slotsTuple.len(); i++)
    {
        auto& str = slotsTuple.getItem(i).cast<PyString>();
        if (str == name)
        {
            return getSlot(i);
        }
    }
    return nullptr;
}

void PyObject::setSlot(int index, PyObject& object)
{
    reinterpret_cast<PyObject**>(this)[type(*this).m_offset + index] = &object;
}

void pylir::rt::destroyPyObject(PyObject& object)
{
    if (auto* integer = object.dyn_cast<pylir::rt::PyInt>())
    {
        integer->~PyInt();
    }
    else if (auto* str = object.dyn_cast<pylir::rt::PyString>())
    {
        str->~PyString();
    }
    else if (auto* dict = object.dyn_cast<pylir::rt::PyDict>())
    {
        dict->~PyDict();
    }
    // All other types are trivially destructible
}

bool pylir::rt::isinstance(PyObject& object, PyTypeObject& typeObject)
{
    auto& mro = type(object).getMROTuple();
    return std::find(mro.begin(), mro.end(), &typeObject) != mro.end();
}

bool PyObject::operator==(PyObject& other)
{
    if (this == &other)
    {
        return true;
    }
    PyObject& eqFunc = *type(*this).methodLookup(PyTypeObject::Eq);
    return Builtins::Bool(eqFunc(*this, other)).cast<PyInt>().boolean();
}

PyObject* PyObject::mroLookup(int index)
{
    auto& mro = type(*this).getMROTuple();
    for (auto* iter : mro)
    {
        if (auto* slot = iter->getSlot(index))
        {
            return slot;
        }
    }
    return nullptr;
}

PyObject* PyObject::methodLookup(int index)
{
    auto* overload = mroLookup(index);
    if (!overload)
    {
        return nullptr;
    }
    if (overload->isa<PyFunction>())
    {
        return overload;
    }
    if (auto* getter = overload->mroLookup(PyTypeObject::Get))
    {
        overload = &(*getter)(*this, type(*this));
    }
    return overload;
}
