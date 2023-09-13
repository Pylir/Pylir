//  Licensed under the Apache License v2.0 with LLVM Exceptions.
//  See https://llvm.org/LICENSE.txt for license information.
//  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <pylir/Runtime/Objects/Objects.hpp>

using namespace pylir::rt;

#include <iostream>

namespace {

PyObject& exceptHookImpl(PyFunction& /*function*/, PyTuple& args,
                         PyDict& /*keywords*/) {
  // TODO: check arguments
  auto& exceptionType = args.getItem(0);
  auto& exception = args.getItem(1);
  auto type =
      exceptionType.getSlot(PyTypeObject::Slots::Name)->cast<PyString>().view();
  if (type.substr(0, sizeof("builtins")) == "builtins.")
    type = type.substr(sizeof("builtins"));

  std::cerr << type << ": ";
  std::cerr << Builtins::Str(exception).cast<PyString>().view() << std::endl;
  return Builtins::None;
}

StaticInstance<Builtins::Str> qualFunctionName("sys.__excepthook__");

} // namespace

StaticInstance<Builtins::Function> exceptHook asm("sys.__excepthook__")(
    {
        {PyFunction::QualName, qualFunctionName},
        {PyFunction::Defaults, Builtins::None},
        {PyFunction::KwDefaults, Builtins::None},
        {PyFunction::Closure, Builtins::None},
    },
    exceptHookImpl);
