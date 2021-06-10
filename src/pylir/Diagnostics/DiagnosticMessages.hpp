
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

constexpr auto EXPECTED_N_INSTEAD_OF_N = FMT_STRING("expected {:q^} instead of {:q^}");

} // namespace pylir::Diag
