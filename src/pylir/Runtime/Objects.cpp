#include "Objects.hpp"

#include <pylir/Support/Macros.hpp>

using namespace pylir::rt;

static_assert(std::is_standard_layout_v<PyObject>);
static_assert(std::is_standard_layout_v<PyTypeObject>);
static_assert(std::is_standard_layout_v<PyFunction>);
static_assert(std::is_standard_layout_v<PySequence>);
static_assert(std::is_standard_layout_v<PyString>);
static_assert(std::is_standard_layout_v<PyDict>);
static_assert(std::is_standard_layout_v<PyInt>);
static_assert(std::is_standard_layout_v<PyBaseException>);

PyObject* PyObject::getSlot(int index)
{
    return reinterpret_cast<PyObject**>(this)[m_type->m_offset + index];
}

PyObject* PyObject::getSlot(std::string_view name)
{
    PySequence& slotsTuple = type(*this).getSlot(PyTypeObject::__slots__)->cast<PySequence>();
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

bool pylir::rt::isinstance(PyObject& object, PyObject& typeObject)
{
    auto& mro = type(object).getSlot(PyTypeObject::__mro__)->cast<PySequence>();
    return std::find(mro.begin(), mro.end(), &typeObject) != mro.end();
}

bool PyObject::operator==(PyObject& other)
{
    PyObject& eqFunc = *type(*this).methodLookup(PyTypeObject::__eq__);
    PyObject& boolean = eqFunc(*this, other);
    if (!type(boolean).is(Builtins::Bool))
    {
        // TODO: TypeError
    }
    return boolean.cast<PyInt>().boolean();
}

PyObject* PyObject::mroLookup(int index)
{
    auto& mro = type(*this).getSlot(PyTypeObject::__mro__)->cast<PySequence>();
    for (auto* iter : mro)
    {
        if (auto slot = iter->getSlot(index))
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
    if (auto* getter = overload->mroLookup(PyTypeObject::__get__))
    {
        overload = &(*getter)(*this, type(*this));
    }
    return overload;
}
