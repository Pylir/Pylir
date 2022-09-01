// Copyright 2022 Markus Böck
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "API.hpp"

#include <iostream>
#include <string_view>

using namespace pylir::rt;

PyObject* pylir_dict_lookup(PyDict& dict, PyObject& key, std::size_t hash)
{
    return dict.tryGetItem(key, hash);
}

void pylir_dict_insert(PyDict& dict, PyObject& key, std::size_t hash, PyObject& value)
{
    dict.setItem(key, hash, value);
}

void pylir_dict_insert_unique(PyDict& dict, PyObject& key, std::size_t hash, PyObject& value)
{
    dict.setItemUnique(key, hash, value);
}

void pylir_dict_erase(PyDict& dict, PyObject& key, std::size_t hash)
{
    dict.delItem(key, hash);
}

std::size_t pylir_str_hash(PyString& string)
{
    return std::hash<std::string_view>{}(string.view());
}

void pylir_print(PyString& string)
{
    std::cout << string.view();
}
