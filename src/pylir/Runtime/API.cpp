#include "API.hpp"

#include <iostream>
#include <string_view>

using namespace pylir::rt;

PyObject* pylir_dict_lookup(PyDict& dict, PyObject& key)
{
    return dict.tryGetItem(key);
}

void pylir_dict_insert(PyDict& dict, PyObject& key, PyObject& value)
{
    dict.setItem(key, value);
}

void pylir_dict_erase(PyDict& dict, PyObject& key)
{
    dict.delItem(key);
}

std::size_t pylir_str_hash(PyString& string)
{
    return std::hash<std::string_view>{}(string.view());
}

IntGetResult pylir_int_get(mp_int* mpInt, std::size_t bytes)
{
    mp_int max;
    (void)mp_init_u64(&max, std::numeric_limits<std::uint64_t>::max());
    if (mp_cmp_mag(mpInt, &max) == MP_GT)
    {
        mp_clear(&max);
        return {0, false};
    }
    mp_clear(&max);
    auto value = mp_get_u64(mpInt);
    if (bytes == 8)
    {
        return {value, true};
    }
    return {value, value <= (1ull << (8 * bytes))};
}

void pylir_print(PyString& string)
{
    std::cout << string.view();
}
