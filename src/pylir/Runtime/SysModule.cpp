#include "Objects.hpp"

using namespace pylir::rt;

#include <iostream>

namespace
{

PyObject& exceptHookImpl(PyFunction& function, PySequence& args, PyDict& keywords)
{
    // TODO: check arguments
    auto& exceptionType = args.getItem(0);
    auto& exception = args.getItem(1);
    auto type = exceptionType.getSlot(PyTypeObject::Slots::__name__)->cast<PyString>().view();
    if (type.substr(0, sizeof("builtins")) == "builtins.")
    {
        type = type.substr(sizeof("builtins"));
    }
    std::cerr << type << ": ";
    std::cerr << Builtins::Str(exception).cast<PyString>().view() << std::endl;
    return Builtins::None;
}

StaticInstance<Builtins::Str> functionName("__excepthook__");
StaticInstance<Builtins::Str> qualFunctionName("sys.__excepthook__");

} // namespace

// TODO: fill slots
StaticInstance<Builtins::Function> exceptHook asm("sys.__excepthook__")(
    {
        {PyFunction::__name__, functionName},
        {PyFunction::__qualname__, qualFunctionName},
        {PyFunction::__defaults__, Builtins::None},
        {PyFunction::__kwdefaults__, Builtins::None},
        {PyFunction::__closure__, Builtins::None},
    },
    exceptHookImpl);
