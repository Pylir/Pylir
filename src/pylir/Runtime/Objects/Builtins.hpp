//  Licensed under the Apache License v2.0 with LLVM Exceptions.
//  See https://llvm.org/LICENSE.txt for license information.
//  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <cstddef>
#include <cstdint>
#include <initializer_list>

namespace pylir::rt {
class PyObject;
class PyTypeObject;
class PyFunction;
class PyInt;
class PyString;
class PyTuple;
class PyList;
class PyDict;
class PyBaseException;

namespace Builtins {
// 'asm' directive takes the name very literally, going as far as ignoring the
// platform mangling. We'll therefore have to do it manually.
#if defined(__APPLE__) || (defined(__i386__) && defined(_WIN32))
#define MANGLING "_"
#else
#define MANGLING ""
#endif
#define BUILTIN(name, symbol, _, Type) extern Type name asm(MANGLING symbol);
#include <pylir/Interfaces/BuiltinsModule.def>

#define COMPILER_BUILTIN_TERNARY_OP(name, slotName)                       \
  extern "C" PyObject& pylir##slotName(PyObject& first, PyObject& second, \
                                       PyObject& third);

#define COMPILER_BUILTIN_BIN_OP(name, slotName) \
  extern "C" PyObject& pylir##slotName(PyObject& lhs, PyObject& rhs);
#define COMPILER_BUILTIN_UNARY_OP(name, slotName) \
  extern "C" PyObject& pylir##slotName(PyObject& val);

#include <pylir/Interfaces/CompilerBuiltins.def>

#undef MANGLING
} // namespace Builtins

template <PyTypeObject&>
struct PyTypeTraits;

template <>
struct PyTypeTraits<Builtins::Type> {
  using instanceType = PyTypeObject;
  constexpr static std::size_t slotCount =
      std::initializer_list<int>{
#define TYPE_SLOT(x, ...) 0,
#include <pylir/Interfaces/Slots.def>
      }
          .size();
};

template <>
struct PyTypeTraits<Builtins::Function> {
  using instanceType = PyFunction;
  constexpr static std::size_t slotCount =
      std::initializer_list<int>{
#define FUNCTION_SLOT(x, ...) 0,
#include <pylir/Interfaces/Slots.def>
      }
          .size();
};

#define NO_SLOT_TYPE(name, instance)            \
  template <>                                   \
  struct PyTypeTraits<Builtins::name> {         \
    using instanceType = instance;              \
    constexpr static std::size_t slotCount = 0; \
  }

NO_SLOT_TYPE(Int, PyInt);
NO_SLOT_TYPE(Bool, PyInt);
// TODO: NO_SLOT_TYPE(Float, PyFloat);
NO_SLOT_TYPE(Str, PyString);
NO_SLOT_TYPE(Tuple, PyTuple);
NO_SLOT_TYPE(List, PyList);
// TODO: NO_SLOT_TYPE(Set, PySet);
NO_SLOT_TYPE(Dict, PyDict);
#undef NO_SLOT_TYPE

namespace details {
constexpr static std::size_t BaseExceptionSlotCount =
    std::initializer_list<int>{
#define BASEEXCEPTION_SLOT(x, ...) 0,
#include <pylir/Interfaces/Slots.def>
    }
        .size();
} // namespace details

#define BUILTIN_EXCEPTION(name, ...)                                          \
  template <>                                                                 \
  struct PyTypeTraits<Builtins::name> {                                       \
    using instanceType = PyBaseException;                                     \
    constexpr static std::size_t slotCount = details::BaseExceptionSlotCount; \
  };
#include <pylir/Interfaces/BuiltinsModule.def>

} // namespace pylir::rt
