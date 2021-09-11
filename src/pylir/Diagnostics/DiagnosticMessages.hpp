
#pragma once

#include <fmt/compile.h>

namespace pylir::Diag
{
constexpr auto UNEXPECTED_EOF_WHILE_PARSING = FMT_STRING("unexpected EOF while parsing");

constexpr auto UNEXPECTED_CHARACTER_AFTER_LINE_CONTINUATION_CHARACTER =
    FMT_STRING("unexpected character after line continuation character");

constexpr auto UNEXPECTED_CHARACTER_N = FMT_STRING("unexpected character '{}'");

constexpr auto EXPECTED_END_OF_LITERAL = FMT_STRING("expected end of literal");

constexpr auto NEWLINE_NOT_ALLOWED_IN_LITERAL = FMT_STRING("newline not allowed in literal");

constexpr auto EXPECTED_OPEN_BRACE_AFTER_BACKSLASH_N = FMT_STRING("expected '{{' after \\N");

constexpr auto UNICODE_NAME_N_NOT_FOUND = FMT_STRING("unicode name '{}' not found");

constexpr auto EXPECTED_N_MORE_HEX_CHARACTERS = FMT_STRING("expected {} more hex characters");

constexpr auto U_PLUS_N_IS_NOT_A_VALID_UNICODE_CODEPOINT = FMT_STRING("U+{:X} is not a valid unicode codepoint");

constexpr auto ONLY_ASCII_VALUES_ARE_ALLOWED_IN_BYTE_LITERALS =
    FMT_STRING("only ascii values are allowed in byte literals");

constexpr auto USE_HEX_OR_OCTAL_ESCAPES_INSTEAD = FMT_STRING("use hex or octal escapes instead");

constexpr auto INVALID_NUMBER_PREFIX_N = FMT_STRING("invalid number prefix '{}'");

constexpr auto UNDERSCORE_ONLY_ALLOWED_BETWEEN_DIGITS = FMT_STRING("underscore only allowed between digits");

constexpr auto NUMBER_WITH_LEADING_ZEROS_NOT_ALLOWED = FMT_STRING("number with leading zeros not allowed");

constexpr auto REMOVE_LEADING_ZEROS = FMT_STRING("remove leading zeroes");

constexpr auto INVALID_INTEGER_SUFFIX = FMT_STRING("invalid integer suffix '{}'");

constexpr auto EXPECTED_DIGITS_FOR_THE_EXPONENT = FMT_STRING("expected digits for the exponent");

constexpr auto EXPECTED_N = FMT_STRING("expected {:q^}");

constexpr auto EXPECTED_N_BEFORE_N = FMT_STRING("expected {:q^} before {:q^}");

constexpr auto EXPECTED_N_INSTEAD_OF_N = FMT_STRING("expected {:q^} instead of {:q^}");

constexpr auto INVALID_INDENTATION_N = FMT_STRING("invalid indentation {}");

constexpr auto NEXT_CLOSEST_INDENTATION_N = FMT_STRING("next closest indent {}");

constexpr auto CANNOT_ASSIGN_TO_N = FMT_STRING("cannot assign to {}");

constexpr auto CANNOT_ASSIGN_TO_RESULT_OF_N = FMT_STRING("cannot assign to result of {}");

constexpr auto CANNOT_ASSIGN_TO_RESULT_OF_OPERATOR_N = FMT_STRING("cannot assign to result of operator {:q^}");

constexpr auto UNKNOWN_FEATURE_N = FMT_STRING("unknown feature '{}'");

constexpr auto CANNOT_EMIT_LLVM_IR_AND_MLIR_IR_AT_THE_SAME_TIME =
    FMT_STRING("cannot emit LLVM IR and MLIR IR at the same time");

constexpr auto LLVM_IR_WONT_BE_EMITTED_WHEN_ONLY_CHECKING_SYNTAX =
    FMT_STRING("LLVM IR won't be emitted when only checking syntax");

constexpr auto MLIR_IR_WONT_BE_EMITTED_WHEN_ONLY_CHECKING_SYNTAX =
    FMT_STRING("MLIR IR won't be emitted when only checking syntax");

constexpr auto ASSEMBLY_WONT_BE_EMITTED_WHEN_ONLY_CHECKING_SYNTAX =
    FMT_STRING("Assembly won't be emitted when only checking syntax");

constexpr auto FAILED_TO_OPEN_FILE_N = FMT_STRING("failed to open file '{}'");

constexpr auto FAILED_TO_ACCESS_FILE_N = FMT_STRING("failed to access file '{}'");

constexpr auto FAILED_TO_READ_FILE_N = FMT_STRING("failed to read file '{}'");

constexpr auto COULD_NOT_FIND_TARGET_N = FMT_STRING("could not find target '{}'");

constexpr auto INVALID_OPTIMIZATION_LEVEL_N = FMT_STRING("invalid optimization level '{}'");

constexpr auto TARGET_N_DOES_NOT_SUPPORT_COMPILING_TO_N = FMT_STRING("target '{}' does not support compiling to {}");

constexpr auto DECLARATION_OF_GLOBAL_N_CONFLICTS_WITH_LOCAL_VARIABLE =
    FMT_STRING("declaration of global '{}' conflicts with local variable");

constexpr auto DECLARATION_OF_GLOBAL_N_CONFLICTS_WITH_FREE_VARIABLE =
    FMT_STRING("declaration of global '{}' conflicts with free variable");

constexpr auto DECLARATION_OF_NONLOCAL_N_CONFLICTS_WITH_LOCAL_VARIABLE =
    FMT_STRING("declaration of nonlocal '{}' conflicts with local variable");

constexpr auto DECLARATION_OF_NONLOCAL_N_CONFLICTS_WITH_FREE_VARIABLE =
    FMT_STRING("declaration of nonlocal '{}' conflicts with free variable");

constexpr auto LOCAL_VARIABLE_BOUND_HERE = FMT_STRING("local variable bound here");

constexpr auto FREE_VARIABLE_BOUND_HERE = FMT_STRING("free variable bound here");

constexpr auto COULD_NOT_FIND_VARIABLE_N_IN_OUTER_SCOPES = FMT_STRING("could not find variable '{}' in outer scopes");

} // namespace pylir::Diag
