// Copyright 2022 Markus Böck
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "MarkAndSweep.hpp"

#include <pylir/Runtime/Globals.hpp>
#include <pylir/Runtime/Stack.hpp>
#include <pylir/Support/Util.hpp>

// Anything below 65535 would do basically
pylir::rt::MarkAndSweep pylir::rt::gc __attribute__((init_priority(200)));

pylir::rt::PyObject* pylir::rt::MarkAndSweep::alloc(std::size_t count)
{
    count = pylir::roundUpTo(count, alignof(PyBaseException));
    switch (count / alignof(PyBaseException))
    {
        case 1:
        case 2: return m_unit2.nextCell();
        case 3:
        case 4: return m_unit4.nextCell();
        case 5:
        case 6: return m_unit6.nextCell();
        case 7:
        case 8: return m_unit8.nextCell();
        default: return m_tree.alloc(count);
    }
}

namespace
{

void mark(pylir::rt::PyObject* object)
{
    object->setMark(true);
}

template <class F>
void introspectObject(pylir::rt::PyObject* object, F f)
{
    if (auto* tuple = object->dyn_cast<pylir::rt::PyTuple>())
    {
        for (auto* iter : *tuple)
        {
            if (iter)
            {
                f(iter);
            }
        }
        return;
    }
    if (auto* list = object->dyn_cast<pylir::rt::PyList>())
    {
        f(list->getTuple());
    }
    else if (auto* dict = object->dyn_cast<pylir::rt::PyDict>())
    {
        for (auto& [key, value] : *dict)
        {
            if (key)
            {
                f(key);
            }
            if (value)
            {
                f(value);
            }
        }
    }
    auto* slots = type(*object).getSlot(pylir::rt::PyTypeObject::Slots);
    auto slotCount = slots ? slots->cast<pylir::rt::PyTuple>().len() : 0;
    for (std::size_t i = 0; i < slotCount; i++)
    {
        if (auto* slot = object->getSlot(i))
        {
            f(slot);
        }
    }
}

void mark(std::uintptr_t stackLowerBound, std::uintptr_t stackUpperBound, std::vector<pylir::rt::PyObject*>&& workList)
{
    while (!workList.empty())
    {
        auto* top = workList.back();
        workList.pop_back();
        introspectObject(top,
                         [&](pylir::rt::PyObject* subObject)
                         {
                             auto address = reinterpret_cast<std::uintptr_t>(subObject);
                             if ((address >= stackLowerBound && address <= stackUpperBound) || isGlobal(subObject)
                                 || subObject->getMark<bool>())
                             {
                                 return;
                             }
                             mark(subObject);
                             workList.push_back(subObject);
                         });
    }
}

} // namespace

void pylir::rt::MarkAndSweep::collect()
{
    std::vector<PyObject*> roots;
    auto [stackLower, stackUpper] = collectStackRoots(roots);
    auto handles = getHandles();
    roots.reserve(roots.size() + handles.size());
    for (auto& iter : handles)
    {
        if (*iter)
        {
            roots.push_back(*iter);
        }
    }
    for (auto iter = roots.begin(); iter != roots.end();)
    {
        auto address = reinterpret_cast<std::uintptr_t>(*iter);
        if ((address >= stackLower && address <= stackUpper) || isGlobal(*iter) || (*iter)->getMark<bool>())
        {
            iter = roots.erase(iter);
        }
        else
        {
            mark(*iter);
            iter++;
        }
    }
    for (const auto& iter : getCollections())
    {
        introspectObject(iter,
                         [&](PyObject* subObject)
                         {
                             if (isGlobal(subObject) || subObject->getMark<bool>())
                             {
                                 return;
                             }
                             mark(subObject);
                             roots.push_back(subObject);
                         });
    }
    mark(stackLower, stackUpper, std::move(roots));
    m_unit2.sweep();
    m_unit4.sweep();
    m_unit6.sweep();
    m_unit8.sweep();
    m_tree.sweep();
}
