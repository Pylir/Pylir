
#pragma once

#include <cstddef>

#include "Objects.hpp"

extern "C" std::size_t pylir_str_hash(pylir::rt::PyString& string);

extern "C" pylir::rt::PyObject* pylir_dict_lookup(pylir::rt::PyDict& dict, pylir::rt::PyObject& key);

extern "C" void pylir_dict_insert(pylir::rt::PyDict& dict, pylir::rt::PyObject& key, pylir::rt::PyObject& value);

extern "C" void pylir_dict_erase(pylir::rt::PyDict& dict, pylir::rt::PyObject& key);

extern "C" void pylir_print(pylir::rt::PyString& string);

extern "C" void pylir_raise(pylir::rt::PyBaseException& exception);

struct IntGetResult
{
    std::size_t value;
    bool valid;
};

extern "C" IntGetResult pylir_int_get(mp_int* mpInt, std::size_t bytes);
