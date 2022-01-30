#include "Support.hpp"

using namespace pylir::rt;

#include "Objects.hpp"

std::size_t PyObjectHasher::operator()(PyObject* object) const noexcept
{
    // TODO: use hash
    auto* hashFunction = type(*object).getSlot(PyTypeObject::__hash__);
    PYLIR_ASSERT(hashFunction);
    auto* integer = (*hashFunction)(*object).dyn_cast<PyInt>();
    if (!integer)
    {
        // TODO: something
    }
    return integer->to<std::size_t>();
}

bool PyObjectEqual::operator()(PyObject* lhs, PyObject* rhs) const noexcept
{
    if (lhs == rhs)
    {
        return true;
    }
    return *lhs == *rhs;
}