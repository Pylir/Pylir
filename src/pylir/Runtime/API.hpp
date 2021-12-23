
#pragma once

#include <cstddef>

#include "Objects.hpp"

extern "C" void* pylir_gc_alloc(std::size_t);

extern "C" std::size_t pylir_str_hash(pylir::rt::PyString* string);

extern "C" pylir::rt::PyObject* pylir_dict_lookup(pylir::rt::PyDict* dict, pylir::rt::PyObject* key);

extern "C" void pylir_dict_insert(pylir::rt::PyDict* dict, pylir::rt::PyObject* key, pylir::rt::PyObject* value);
