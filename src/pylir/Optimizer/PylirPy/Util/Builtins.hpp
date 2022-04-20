// Copyright 2022 Markus Böck
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

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

#define BUILTIN(x, s, isPublic, ...) constexpr Builtin x = {s, isPublic};
#include <pylir/Interfaces/Builtins.def>

constexpr std::array allBuiltins = {
#define BUILTIN(x, ...) x,
#include <pylir/Interfaces/Builtins.def>
};
} // namespace pylir::Py::Builtins
