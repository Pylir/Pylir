#include "Objects.hpp"

#include <pylir/Support/Macros.hpp>

using namespace pylir::rt;

class StackTuple : public PySequence
{
public:
    StackTuple(PyObject** data, std::size_t size)
        : PySequence(reinterpret_cast<PyTypeObject*>(&Builtin::Tuple), {size, size, data})
    {
    }
};

PyObject* PyObject::getSlot(int index)
{
    return reinterpret_cast<PyObject**>(this + 1)[m_type->getOffset() + index];
}

PyObject* PyObject::getSlot(std::string_view name)
{
    PySequence* slotsTuple = getType()->getSlot(PyTypeObject::__slots__)->cast<PySequence>();
    for (std::size_t i = 0; i < slotsTuple->len(); i++)
    {
        auto* str = slotsTuple->getItem(i)->cast<PyString>();
        if (*str == name)
        {
            return getSlot(i);
        }
    }
    return nullptr;
}

PyObject* PyObject::call(PySequence* args, PyDict* keywords)
{
    auto current = this;
    while (current && !current->isa<PyFunction>())
    {
        current = current->getType()->getSlot(PyTypeObject::__call__);
    }
    if (!current)
    {
        // TODO: exception
    }
    auto* function = current->cast<PyFunction>();
    return function->call(args, keywords);
}

PyObject* PyObject::call(std::initializer_list<PyObject*> args)
{
    StackTuple tuple(const_cast<PyObject**>(args.begin()), args.size());
    PyDict dict;
    return call(&tuple, &dict);
}

std::size_t PyObjectHasher::operator()(PyObject* object) const noexcept
{
    auto* hashFunction = object->getType()->getSlot(PyTypeObject::__hash__);
    PYLIR_ASSERT(hashFunction);
    auto* integer = hashFunction->call({object})->dyn_cast<PyInt>();
    if (!integer)
    {
        // TODO: something
    }
    return integer->to<std::size_t>();
}

bool PyObjectEqual::operator()(PyObject* lhs, PyObject* rhs) const noexcept
{
    auto* eqFunction = lhs->getType()->getSlot(PyTypeObject::__eq__);
    PYLIR_ASSERT(eqFunction);
    auto* boolean = eqFunction->call({lhs, rhs})->dyn_cast<PyInt>();
    if (!boolean) // TODO: probably need to check its EXACTLY a boolean
    {
    }
    return boolean->boolean();
}

PyObject* PyDict::tryGetItem(PyObject* key)
{
    auto result = m_table.find(key);
    if (result == m_table.end())
    {
        return nullptr;
    }
    return result->value;
}
