
#pragma once

#include <string_view>

namespace pylir::Diag
{
constexpr std::string_view UNEXPECTED_EOF_WHILE_PARSING = "unexpected EOF while parsing";

constexpr std::string_view UNEXPECTED_CHARACTER_AFTER_LINE_CONTINUATION_CHARACTER =
    "unexpected character after line continuation character";

constexpr std::string_view UNEXPECTED_CHARACTER_N = "unexpected character '{}'";

constexpr std::string_view EXPECTED_END_OF_LITERAL = "expected end of literal";

constexpr std::string_view NEWLINE_NOT_ALLOWED_IN_LITERAL = "newline not allowed in literal";

} // namespace pylir::Diag
