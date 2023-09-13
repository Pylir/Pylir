//  Licensed under the Apache License v2.0 with LLVM Exceptions.
//  See https://llvm.org/LICENSE.txt for license information.
//  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <llvm/ADT/StringRef.h>

#include <array>

namespace pylir::Builtins {

struct Builtin {
  llvm::StringLiteral name;
  bool isPublic;
};

#define COMPILER_BUILTIN(cppName, intrName) \
  constexpr Builtin Pylir##cppName = {#intrName, false};
#include "CompilerBuiltins.def"

#define BUILTIN(x, s, isPublic, ...) constexpr Builtin x = {s, isPublic};
#include "BuiltinsModule.def"

[[maybe_unused]] constexpr std::array allBuiltins = {
#define BUILTIN(x, ...) x,
#include "BuiltinsModule.def"

#define COMPILER_BUILTIN(cppName, ...) Pylir##cppName,
#include "CompilerBuiltins.def"
};

enum class TypeSlots {
#define TYPE_SLOT(pythonName, cppName) cppName,
#include "Slots.def"
};

enum class FunctionSlots {
#define FUNCTION_SLOT(pythonName, cppName) cppName,
#include "Slots.def"
};

enum class BaseExceptionSlots {
#define BASEEXCEPTION_SLOT(pythonName, cppName) cppName,
#include "Slots.def"
};
} // namespace pylir::Builtins
