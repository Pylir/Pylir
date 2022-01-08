#include "Objects.hpp"

using namespace pylir::rt;

namespace
{

PyObject& exceptHookImpl(PyFunction& function, PySequence& args, PyDict& keywords)
{
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
