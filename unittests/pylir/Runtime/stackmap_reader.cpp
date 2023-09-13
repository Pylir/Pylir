// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <catch2/catch_test_macros.hpp>

#include <pylir/Runtime/GC/Stack.hpp>
#include <pylir/Runtime/Objects/Objects.hpp>

#include <array>
#include <iostream>

#include "catch2/matchers/catch_matchers.hpp"
#include "catch2/matchers/catch_matchers_vector.hpp"

// Called by stackmap_source.ll.in to escape the pointers and extend their
// lifetime beyond the closure call.
extern "C" void pylir_test_stack_escape(pylir::rt::PyObject& o) {
  std::cerr << ' ' << &o;
}

// Function defined in stackmap_source.ll.in and compiled by pylir to generate
// the stackmap. The reason this function has so many parameters is so that the
// chance of at least one being put in a callee saved register is very high,
// regardless of architecture. That way we should have full test coverage of
// roots on the stack, regardless of precise location (spilled, callee save
// register, alloca).
extern "C" void pylir_test_stack_read(
    void* closure, void (*closureCall)(void*), pylir::rt::PyObject& a,
    pylir::rt::PyObject& b, pylir::rt::PyObject& c, pylir::rt::PyObject& d,
    pylir::rt::PyObject& e, pylir::rt::PyObject& f, pylir::rt::PyObject& g,
    pylir::rt::PyObject& h, pylir::rt::PyObject& i, pylir::rt::PyObject& j,
    pylir::rt::PyObject& k, pylir::rt::PyObject& l, pylir::rt::PyObject& m,
    pylir::rt::PyObject& n, pylir::rt::PyObject& o, pylir::rt::PyObject& p,
    pylir::rt::PyObject** vec);

TEST_CASE("Stackmap reader") {
  std::array<pylir::rt::PyString, 16> array = {
      pylir::rt::PyString("zero"),       pylir::rt::PyString("first"),
      pylir::rt::PyString("second"),     pylir::rt::PyString("third"),
      pylir::rt::PyString("fourth"),     pylir::rt::PyString("fifth"),
      pylir::rt::PyString("sixth"),      pylir::rt::PyString("seventh"),
      pylir::rt::PyString("eighth"),     pylir::rt::PyString("ninth"),
      pylir::rt::PyString("tenth"),      pylir::rt::PyString("eleventh"),
      pylir::rt::PyString("twelfth"),    pylir::rt::PyString("thirteenth"),
      pylir::rt::PyString("fourteenth"), pylir::rt::PyString("fifteenth")};
  auto el1 = pylir::rt::PyString("seventeenth");
  auto el2 = pylir::rt::PyString("eighteenth");
  std::array<pylir::rt::PyObject*, 2> vec = {&el1, &el2};
  std::vector<std::string> result;
  auto impl = [&]() {
    std::vector<pylir::rt::PyObject*> roots;
    pylir::rt::collectStackRoots(roots);
    for (auto* iter : roots) {
      result.emplace_back(iter->cast<pylir::rt::PyString>().view());
    }
  };

  pylir_test_stack_read(
      reinterpret_cast<void*>(&impl),
      +[](void* lambda) { (*reinterpret_cast<decltype(impl)*>(lambda))(); },
      array[0], array[1], array[2], array[3], array[4], array[5], array[6],
      array[7], array[8], array[9], array[10], array[11], array[12], array[13],
      array[14], array[15], vec.data());

  CHECK_THAT(result, Catch::Matchers::UnorderedEquals<std::string>({
                         "zero",      "first",       "second",     "third",
                         "fourth",    "fifth",       "sixth",      "seventh",
                         "eighth",    "ninth",       "tenth",      "eleventh",
                         "twelfth",   "thirteenth",  "fourteenth", "fifteenth",
                         "sixteenth", "seventeenth", "eighteenth",
                     }));
}
