
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

} // namespace pylir::Diag
