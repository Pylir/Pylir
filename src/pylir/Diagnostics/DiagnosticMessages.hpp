
#pragma once

#include <string_view>

namespace pylir::Diag
{
constexpr std::string_view UNEXPECTED_EOF_WHILE_PARSING = "unexpected EOF while parsing";

constexpr std::string_view UNEXPECTED_CHARACTER_AFTER_LINE_CONTINUATION_CHARACTER =
    "unexpected character after line continuation character";
} // namespace pylir::Diag
