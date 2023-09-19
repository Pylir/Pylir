//  Licensed under the Apache License v2.0 with LLVM Exceptions.
//  See https://llvm.org/LICENSE.txt for license information.
//  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "Text.hpp"

#include <utf8proc.h>

std::optional<pylir::Text::Encoding>
pylir::Text::checkForBOM(std::string_view bytes) {
  constexpr auto startsWith = [](std::string_view view, const auto& prefix) {
    return view.substr(0, prefix.size()) ==
           std::string_view(prefix.data(), prefix.size());
  };
  // TODO: use starts_with in C++20
  if (startsWith(bytes, UTF8_BOM))
    return Encoding::UTF8;

  if (startsWith(bytes, UTF32BE_BOM))
    return Encoding::UTF32BE;

  if (startsWith(bytes, UTF32LE_BOM))
    return Encoding::UTF32LE;

  if (startsWith(bytes, UTF16BE_BOM))
    return Encoding::UTF16BE;

  if (startsWith(bytes, UTF16LE_BOM))
    return Encoding::UTF16LE;

  return std::nullopt;
}

std::optional<pylir::Text::Encoding>
pylir::Text::readBOM(std::string_view& bytes) {
  auto result = checkForBOM(bytes);
  if (result) {
    switch (*result) {
    case Encoding::UTF8: bytes.remove_prefix(UTF8_BOM.size()); break;
    case Encoding::UTF16LE:
    case Encoding::UTF16BE:
      static_assert(UTF16LE_BOM.size() == UTF16BE_BOM.size());
      bytes.remove_prefix(UTF16LE_BOM.size());
      break;
    case Encoding::UTF32LE:
    case Encoding::UTF32BE:
      static_assert(UTF32LE_BOM.size() == UTF32BE_BOM.size());
      bytes.remove_prefix(UTF32LE_BOM.size());
      break;
    }
  }
  return result;
}

constexpr static std::array<char, 4> REPLACEMENT_CHARACTER_UTF8 = {
    static_cast<char>(0xEF), static_cast<char>(0xBF), static_cast<char>(0xBD),
    0};

std::array<char, 4> pylir::Text::toUTF8(std::string_view& utf8, bool* legal) {
  utf8proc_int32_t result;
  auto size =
      utf8proc_iterate(reinterpret_cast<const utf8proc_uint8_t*>(utf8.data()),
                       utf8.size(), &result);
  if (legal)
    *legal = result > 0;

  if (result <= 0) {
    if (!utf8.empty()) {
      // Consume just one byte on error
      utf8.remove_prefix(1);
    }
    return REPLACEMENT_CHARACTER_UTF8;
  }
  std::array<char, 4> array{};
  std::copy(utf8.begin(), utf8.begin() + size, array.begin());
  return array;
}

std::array<char, 4> pylir::Text::toUTF8(std::u16string_view& utf16,
                                        bool* legal) {
  char32_t codepoint = toUTF32(utf16, legal);
  return toUTF8(codepoint, legal);
}

std::array<char, 4> pylir::Text::toUTF8(char32_t utf32, bool* legal) {
  std::array<utf8proc_uint8_t, 4> array{};
  auto valid = utf8proc_codepoint_valid(utf32);
  if (legal)
    *legal = valid;

  if (valid)
    utf8proc_encode_char(utf32, array.data());
  else
    return REPLACEMENT_CHARACTER_UTF8;

  return {static_cast<char>(array[0]), static_cast<char>(array[1]),
          static_cast<char>(array[2]), static_cast<char>(array[3])};
}

constexpr static std::array<char16_t, 2> REPLACEMENT_CHARACTER_UTF16 = {0xFFFD,
                                                                        0};

std::array<char16_t, 2> pylir::Text::toUTF16(std::u16string_view& utf16,
                                             bool* legal) {
  // Using the convert function, just for verification purposes
  bool ok;
  auto copy = utf16;
  (void)toUTF32(utf16, &ok);
  if (legal)
    *legal = ok;

  if (!ok)
    return REPLACEMENT_CHARACTER_UTF16;

  std::array<char16_t, 2> result{};
  std::copy(copy.data(), utf16.data(), result.begin());
  return result;
}

std::array<char16_t, 2> pylir::Text::toUTF16(std::string_view& utf8,
                                             bool* legal) {
  char32_t codepoint = toUTF32(utf8, legal);
  return toUTF16(codepoint, legal);
}

std::array<char16_t, 2> pylir::Text::toUTF16(char32_t utf32, bool* legal) {
  if (!utf8proc_codepoint_valid(utf32)) {
    if (legal)
      *legal = false;

    return REPLACEMENT_CHARACTER_UTF16;
  }
  if (legal)
    *legal = true;

  if (utf32 <= 0xFFFF)
    return {static_cast<char16_t>(utf32), 0};

  // https://unicode.org/faq/utf_bom.html
  constexpr char32_t leadOffset = 0xD800 - (0x10000 >> 10);

  std::array<char16_t, 2> result{};
  result[0] = leadOffset + (utf32 >> 10);
  result[1] = 0xDC00 + (utf32 & 0x3FF);
  return result;
}

constexpr static char32_t REPLACEMENT_CHARACTER_UTF32 = 0xFFFD;

char32_t pylir::Text::toUTF32(std::string_view& utf8, bool* legal) {
  utf8proc_int32_t result;
  auto size =
      utf8proc_iterate(reinterpret_cast<const utf8proc_uint8_t*>(utf8.data()),
                       utf8.size(), &result);
  if (legal)
    *legal = result > 0;

  if (result <= 0) {
    if (!utf8.empty())
      utf8.remove_prefix(1);

    return REPLACEMENT_CHARACTER_UTF32;
  }
  utf8.remove_prefix(size);
  return result;
}

char32_t pylir::Text::toUTF32(std::u16string_view& utf16, bool* legal) {
  if (legal)
    *legal = true;

  if (utf16.empty()) {
    if (legal)
      *legal = false;

    return REPLACEMENT_CHARACTER_UTF32;
  }
  char16_t first = utf16.front();
  utf16.remove_prefix(1);
  constexpr char32_t utF16LowSurrogateStart = 0xDC00;
  constexpr char32_t utF16LowSurrogateEnd = 0xDFFF;
  constexpr char32_t utF16HighSurrogateStart = 0xD800;
  constexpr char32_t utF16HighSurrogateEnd = 0xDBFF;
  if (first < utF16HighSurrogateStart || first > utF16HighSurrogateEnd) {
    if (first >= utF16LowSurrogateStart && first <= utF16LowSurrogateEnd) {
      if (legal)
        *legal = false;

      return REPLACEMENT_CHARACTER_UTF32;
    }
    return first;
  }
  if (utf16.empty()) {
    if (legal)
      *legal = false;

    return REPLACEMENT_CHARACTER_UTF32;
  }
  char16_t second = utf16.front();
  utf16.remove_prefix(1);
  if (second < utF16LowSurrogateStart || second > utF16LowSurrogateEnd) {
    if (legal)
      *legal = false;

    return REPLACEMENT_CHARACTER_UTF32;
  }
  return ((first - utF16HighSurrogateStart) << 10) +
         (second - utF16LowSurrogateStart) + 0x0010000;
}

char32_t pylir::Text::toUTF32(char32_t utf32, bool* legal) {
  // Just verify by converting to another format
  bool ok = utf8proc_codepoint_valid(utf32);
  if (legal)
    *legal = ok;

  if (!ok)
    return REPLACEMENT_CHARACTER_UTF32;

  return utf32;
}

std::string pylir::Text::toUTF8String(std::u16string_view utf16, bool* legal) {
  if (legal)
    *legal = true;

  std::string result;
  result.reserve(utf16.size() * 2);
  while (!utf16.empty()) {
    bool ok;
    auto utf8 = toUTF8(utf16, &ok);
    if (legal)
      *legal = *legal && ok;

    for (auto character : utf8) {
      if (!character)
        break;
      result += character;
    }
  }
  return result;
}

std::string pylir::Text::toUTF8String(std::u32string_view utf32, bool* legal) {
  if (legal)
    *legal = true;

  std::string result;
  result.reserve(utf32.size() * 4);
  for (auto codepoint : utf32) {
    bool ok;
    auto utf8 = toUTF8(codepoint, &ok);
    if (legal)
      *legal = *legal && ok;

    for (auto character : utf8) {
      if (!character)
        break;
      result += character;
    }
  }
  return result;
}

std::u32string pylir::Text::toUTF32String(std::string_view utf8, bool* legal) {
  if (legal)
    *legal = true;

  std::u32string result;
  result.reserve(utf8.size() / 2);
  while (!utf8.empty()) {
    bool ok;
    result += toUTF32(utf8, &ok);
    if (legal)
      *legal = *legal && ok;
  }
  return result;
}

bool pylir::Text::isWhitespace(char32_t codepoint) {
  const auto* properties = utf8proc_get_property(codepoint);
  return properties->category == UTF8PROC_CATEGORY_ZS ||
         properties->bidi_class == UTF8PROC_BIDI_CLASS_WS ||
         properties->bidi_class == UTF8PROC_BIDI_CLASS_B ||
         properties->bidi_class == UTF8PROC_BIDI_CLASS_S;
}

bool pylir::Text::isValidCodepoint(char32_t codepoint) {
  return utf8proc_codepoint_valid(codepoint);
}

std::string pylir::Text::normalize(std::string_view utf8,
                                   pylir::Text::Normalization normalization) {
  [[maybe_unused]] bool ok;
  auto u32 = toUTF32String(utf8, &ok);
  PYLIR_ASSERT(ok);
  return toUTF8String(normalize(u32, normalization));
}

std::u32string
pylir::Text::normalize(std::u32string_view utf32,
                       pylir::Text::Normalization normalization) {
  utf8proc_option_t option;
  switch (normalization) {
  case Normalization::NFD:
    option =
        static_cast<utf8proc_option_t>(UTF8PROC_STABLE | UTF8PROC_DECOMPOSE);
    break;
  case Normalization::NFC:
    option = static_cast<utf8proc_option_t>(UTF8PROC_STABLE | UTF8PROC_COMPOSE);
    break;
  case Normalization::NFKD:
    option = static_cast<utf8proc_option_t>(
        UTF8PROC_STABLE | UTF8PROC_DECOMPOSE | UTF8PROC_COMPAT);
    break;
  case Normalization::NFKC:
    option = static_cast<utf8proc_option_t>(UTF8PROC_STABLE | UTF8PROC_COMPOSE |
                                            UTF8PROC_COMPAT);
    break;
  default: PYLIR_UNREACHABLE;
  }
  std::vector<utf8proc_int32_t> buffer(utf32.size());
  std::size_t actuallyWritten = 0;
  for (auto codepoint : utf32) {
    do {
      auto size = utf8proc_decompose_char(
          codepoint, buffer.data() + actuallyWritten,
          buffer.size() - actuallyWritten, option, nullptr);
      PYLIR_ASSERT(size >= 0);
      if (static_cast<std::size_t>(size) > buffer.size() - actuallyWritten) {
        buffer.resize(
            std::max<std::size_t>(buffer.size() * 2, buffer.size() + size));
        continue;
      }
      actuallyWritten += size;
      break;
    } while (true);
  }
  return {buffer.begin(), buffer.begin() + actuallyWritten};
}

std::size_t pylir::Text::consoleWidth(char32_t codepoint) {
  return utf8proc_charwidth(codepoint);
}
