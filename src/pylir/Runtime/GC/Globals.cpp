//  Licensed under the Apache License v2.0 with LLVM Exceptions.
//  See https://llvm.org/LICENSE.txt for license information.
//  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "Globals.hpp"

extern "C" pylir::rt::PyObject*** const pylir_roots_start;
extern "C" pylir::rt::PyObject*** const pylir_roots_end;
extern "C" pylir::rt::PyObject** const pylir_collections_start;
extern "C" pylir::rt::PyObject** const pylir_collections_end;
extern "C" pylir::rt::PyObject** const pylir_constants_start;
extern "C" pylir::rt::PyObject** const pylir_constants_end;

bool pylir::rt::isGlobal(PyObject* object) {
  struct Ranges {
    pylir::rt::PyObject* constMin;
    pylir::rt::PyObject* constMax;
    pylir::rt::PyObject* collMin;
    pylir::rt::PyObject* collMax;
  };
  static auto ranges = [] {
    auto [constMin, constMax] =
        std::minmax_element(pylir_constants_start, pylir_constants_end);
    auto [colMin, colMax] =
        std::minmax_element(pylir_collections_start, pylir_collections_end);
    return Ranges{*constMin, *constMax, *colMin, *colMax};
  }();
  return (object >= ranges.constMin && object <= ranges.constMax) ||
         (object >= ranges.collMin && object <= ranges.collMax);
}

tcb::span<pylir::rt::PyObject**> pylir::rt::getHandles() {
  return {pylir_roots_start, pylir_roots_end};
}

tcb::span<pylir::rt::PyObject*> pylir::rt::getCollections() {
  return {pylir_collections_start, pylir_collections_end};
}
