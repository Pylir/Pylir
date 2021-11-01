
#pragma once

#include <array>
#include <string_view>

namespace pylir::Py::Builtins
{

struct Builtin
{
    std::string_view name;
    bool isPublic;
};

#define BUILTIN(x, s, isPublic) constexpr Builtin x = {s, isPublic};
#include "Builtins.def"

constexpr std::array allBuiltins = {
#define BUILTIN(x, ...) x,
#include "Builtins.def"
};
} // namespace pylir::Py::Builtins
