#include "Objects.hpp"

#include <pylir/Support/Macros.hpp>

using namespace pylir::rt;

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
    PyObject& eqFunc = *type(*this).getSlot(PyTypeObject::__eq__);
    PyObject& boolean = eqFunc(*this, other);
    if (!type(boolean).is(Builtins::Bool))
    {
        // TODO: TypeError
    }
    return boolean.cast<PyInt>().boolean();
}
