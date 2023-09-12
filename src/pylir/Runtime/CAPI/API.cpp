//  Licensed under the Apache License v2.0 with LLVM Exceptions.
//  See https://llvm.org/LICENSE.txt for license information.
//  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "API.hpp"

#include <pylir/Runtime/GC/GC.hpp>

#include <iostream>
#include <string_view>

using namespace pylir::rt;

PyObject* pylir_dict_lookup(PyDict& dict, PyObject& key, std::size_t hash) {
  return dict.tryGetItem(key, hash);
}

void pylir_dict_insert(PyDict& dict, PyObject& key, std::size_t hash,
                       PyObject& value) {
  dict.setItem(key, hash, value);
}

void pylir_dict_insert_unique(PyDict& dict, PyObject& key, std::size_t hash,
                              PyObject& value) {
  dict.setItemUnique(key, hash, value);
}

void pylir_dict_erase(PyDict& dict, PyObject& key, std::size_t hash) {
  dict.delItem(key, hash);
}

std::size_t pylir_str_hash(PyString& string) {
  return std::hash<std::string_view>{}(string.view());
}

void pylir_print(PyString& string) {
  std::cout << string.view();
}

extern "C" void* pylir_gc_alloc(std::size_t size) {
  return pylir::rt::gc.alloc(size);
}
