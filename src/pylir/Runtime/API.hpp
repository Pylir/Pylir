// Copyright 2022 Markus Böck
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <cstddef>

#include "Objects.hpp"

extern "C" std::size_t pylir_str_hash(pylir::rt::PyString& string);

extern "C" pylir::rt::PyObject* pylir_dict_lookup(pylir::rt::PyDict& dict, pylir::rt::PyObject& key);

extern "C" void pylir_dict_insert(pylir::rt::PyDict& dict, pylir::rt::PyObject& key, pylir::rt::PyObject& value);

extern "C" void pylir_dict_erase(pylir::rt::PyDict& dict, pylir::rt::PyObject& key);

extern "C" void pylir_print(pylir::rt::PyString& string);

extern "C" void pylir_raise(pylir::rt::PyBaseException& exception);
