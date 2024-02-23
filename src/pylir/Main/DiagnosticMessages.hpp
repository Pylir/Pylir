//  Licensed under the Apache License v2.0 with LLVM Exceptions.
//  See https://llvm.org/LICENSE.txt for license information.
//  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <fmt/compile.h>

namespace pylir::Diag {

constexpr auto N_WONT_BE_EMITTED_WHEN_ONLY_CHECKING_SYNTAX =
    FMT_STRING("{} won't be emitted when only checking syntax");

constexpr auto O4_MAY_ENABLE_LTO_COMPILER_MIGHT_OUTPUT_LLVM_IR = FMT_STRING(
    "'-O4' may enable LTO. Compiler might output LLVM IR instead of an {}");

constexpr auto LTO_ENABLED_COMPILER_WILL_OUTPUT_LLVM_IR =
    FMT_STRING("LTO enabled. Compiler will output LLVM IR instead of an {}");

constexpr auto EXPECTED_ONLY_ONE_INPUT_FILE =
    FMT_STRING("expected only one input file");

constexpr auto NO_INPUT_FILE = FMT_STRING("no input file");

constexpr auto FAILED_TO_OPEN_FILE_N = FMT_STRING("failed to open file '{}'");

constexpr auto FAILED_TO_OPEN_OUTPUT_FILE_N_FOR_WRITING =
    FMT_STRING("failed to open output file '{}' for writing");

constexpr auto FAILED_TO_ACCESS_FILE_N =
    FMT_STRING("failed to access file '{}'");

constexpr auto FAILED_TO_FIND_MODULE_N =
    FMT_STRING("failed to find module '{}'");

constexpr auto FAILED_TO_READ_FILE_N = FMT_STRING("failed to read file '{}'");

constexpr auto FAILED_TO_CREATE_TEMPORARY_FILE_N =
    FMT_STRING("failed to create temporary file '{}'");

constexpr auto FAILED_TO_DISCARD_TEMPORARY_FILE_N =
    FMT_STRING("failed to discard temporary file '{}'");

constexpr auto FAILED_TO_RENAME_TEMPORARY_FILE_N_TO_N =
    FMT_STRING("failed to rename temporary file '{}' to '{}'");

constexpr auto FAILED_TO_KEEP_TEMPORARY_FILE_N =
    FMT_STRING("failed to rename temporary file '{}'");

constexpr auto FAILED_TO_FIND_LINKER = FMT_STRING("failed to find linker");

constexpr auto ATTEMPTED_N = FMT_STRING("attempted {}");

constexpr auto OUTPUT_CANNOT_BE_STDOUT_WHEN_WRITING_DEPENDENCY_FILE =
    FMT_STRING("output cannot be stdout when writing dependency file");

constexpr auto COULD_NOT_FIND_TARGET_N =
    FMT_STRING("could not find target '{}'");

constexpr auto UNSUPPORTED_TARGET_N = FMT_STRING("unsupported target '{}'");

constexpr auto INVALID_OPTIMIZATION_LEVEL_N =
    FMT_STRING("invalid optimization level '{}'");

constexpr auto TARGET_N_DOES_NOT_SUPPORT_COMPILING_TO_N =
    FMT_STRING("target '{}' does not support compiling to {}");

constexpr auto UNKNOWN_SANITIZER_N = FMT_STRING("unknown sanitizer '{}'");

constexpr auto ADDRESS_AND_THREAD_SANITIZERS_ARE_INCOMPATIBLE_WITH_EACH_OTHER =
    FMT_STRING(
        "'address' and 'thread' sanitizers are incompatible with each other");

} // namespace pylir::Diag
