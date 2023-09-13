//  Licensed under the Apache License v2.0 with LLVM Exceptions.
//  See https://llvm.org/LICENSE.txt for license information.
//  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <functional>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "Endian.hpp"
#include "LazyCacheIterator.hpp"
#include "Macros.hpp"

namespace pylir::Text {
enum class Encoding { UTF8, UTF16LE, UTF16BE, UTF32LE, UTF32BE };

constexpr std::array<char, 3> UTF8_BOM = {
    static_cast<char>(0xEF), static_cast<char>(0xBB), static_cast<char>(0xBF)};
constexpr std::array<char, 2> UTF16LE_BOM = {static_cast<char>(0xFF),
                                             static_cast<char>(0xFE)};
constexpr std::array<char, 2> UTF16BE_BOM = {static_cast<char>(0xFE),
                                             static_cast<char>(0xFF)};
constexpr std::array<char, 4> UTF32LE_BOM = {static_cast<char>(0xFF),
                                             static_cast<char>(0xFE), 0, 0};
constexpr std::array<char, 4> UTF32BE_BOM = {0, 0, static_cast<char>(0xFE),
                                             static_cast<char>(0xFF)};

/**
 * Checks if the start of the view contains a BOM indicating UTF-8, UTF-16 or
 * UTF-32.
 *
 * @param bytes view into a list of bytes. No encoding in particular is assumed
 * yet
 * @return The encoding for the BOM or an empty optional if no BOM is contained
 */
std::optional<Encoding> checkForBOM(std::string_view bytes);

/**
 * Like checkForBOM, but also advances the string_view past the BOM if present
 *
 * @param bytes
 * @return
 */
std::optional<Encoding> readBOM(std::string_view& bytes);

/**
 * Does no conversion, but instead checks whether the input is proper UTF-8.
 * The utf8 source is 1 to 4 bytes. Read bytes are consumed.
 *
 * @param utf8 UTF8 source
 * @param legal if not null, set to false when either 1) the utf-8 source was
 * exhausted or 2) the resulting codepoint(s) are illegal.
 *
 * @return Input, or unicode replacement character on error
 */
std::array<char, 4> toUTF8(std::string_view& utf8, bool* legal = nullptr);

/**
 * Converts a single unicode codepoint from UTF-16 to UTF-8.
 * The utf16 source is therefore 1 to 2 bytes. Read bytes are consumed.
 *
 * @param utf16 UTF16 source
 * @param legal if not null, set to false when either 1) the utf-16 source was
 * exhausted or 2) the resulting codepoint(s) are illegal. The unicode
 * replacement character is returned in either case.
 *
 * @return the codepoint in UTF8. Index 0 is guaranteed to have a value, Index 1
 * to 3 are 0 if not needed
 */
std::array<char, 4> toUTF8(std::u16string_view& utf16, bool* legal = nullptr);

/**
 * Converts a single unicode codepoint from UTF-32 to UTF-8.
 *
 * @param utf32 UTF32 source
 * @param legal if not null, set to false when the resulting codepoint(s) are
 * illegal. The unicode replacement character is returned in that case.
 *
 * @return the codepoint in UTF8. Index 0 is guaranteed to have a value, Index 1
 * to 3 are 0 if not needed
 */
std::array<char, 4> toUTF8(char32_t utf32, bool* legal = nullptr);

std::string toUTF8String(std::u16string_view utf16, bool* legal = nullptr);

std::string toUTF8String(std::u32string_view utf32, bool* legal = nullptr);

/**
 * Does no conversion, but instead checks whether the input is proper UTF-16.
 * The utf16 source is 1 to 2 bytes. Read bytes are consumed.
 *
 * @param utf16 UTF16 source
 * @param legal if not null, set to false when either 1) the utf-16 source was
 * exhausted or 2) the resulting codepoint(s) are illegal. The unicode
 * replacement character is returned in either case.
 *
 * @return Input, or unicode replacement character on error
 */
std::array<char16_t, 2> toUTF16(std::u16string_view& utf16,
                                bool* legal = nullptr);

/**
 * Converts a single unicode codepoint from UTF-8 to UTF-16.
 * The utf8 source is therefore 1 to 4 bytes. Read bytes are consumed
 *
 * @param utf8 UTF8 source
 * @param legal if not null, set to false when either 1) the utf-8 source was
 * exhausted or 2) the resulting codepoint(s) are illegal. The unicode
 * replacement character is returned in either case.
 *
 * @return the codepoint in UTF16. Index 0 is guaranteed to have a value, Index
 * 1 is 0 if not needed
 */
std::array<char16_t, 2> toUTF16(std::string_view& utf8, bool* legal = nullptr);

/**
 * Converts a single unicode codepoint from UTF-32 to UTF-16
 *
 * @param utf32 UTF-32 source
 * @param legal if not null, set to false if the resulting UTF-16 codepoint is
 * illegal. The unicode replacement character is returned in that case.
 *
 * @return the codepoint in UTF16. Index 0 is guaranteed to have a value, Index
 * 1 is 0 if not needed
 */
std::array<char16_t, 2> toUTF16(char32_t utf32, bool* legal = nullptr);

/**
 * Converts a single unicode codepoint from UTF-8 to UTF-32.
 * The utf8 source is therefore 1 to 4 bytes. Read bytes are consumed.
 *
 * @param utf8 UTF8 source
 * @param legal if not null, set to false when either 1) the utf-8 source was
 * exhausted or 2) the resulting codepoint(s) are illegal
 *
 * @return the codepoint in UTF32
 */
char32_t toUTF32(std::string_view& utf8, bool* legal = nullptr);

/**
 * Converts a single unicode codepoint from UTF-16 to UTF-32.
 * The utf8 source is therefore 1 to 2 bytes. Read bytes are consumed.
 *
 * @param utf16 UTF16 source
 * @param legal if not null, set to false when either 1) the utf-16 source was
 * exhausted or 2) the resulting codepoint(s) are illegal. The unicode
 * replacement character is returned in either case.
 *
 * @return the codepoint in UTF32
 */
char32_t toUTF32(std::u16string_view& utf16, bool* legal = nullptr);

/**
 * Does no conversion, but instead checks whether the input is proper UTF-32
 *
 * @param utf16 UTF16 source
 * @param legal if not null, set to false when the resulting codepoint(s) are
 * illegal.
 * @return Input, or unicode replacement character on error
 */
char32_t toUTF32(char32_t utf32, bool* legal = nullptr);

std::u32string toUTF32String(std::string_view utf8, bool* legal = nullptr);

std::u32string toUTF32String(std::u16string_view utf16, bool* legal = nullptr);

template <class Source, class Target>
class Transcoder;

template <class Target>
class Transcoder<void, Target> {
  static_assert(std::is_same_v<char, Target> ||
                std::is_same_v<char16_t, Target> ||
                std::is_same_v<char32_t, Target>);

  std::string_view m_source;
  Encoding m_encoding;

  class Iterator {
    std::string_view m_source;
    std::array<Target, sizeof(char32_t) / sizeof(Target)> m_data;
    Encoding m_encoding{};

    void transcodeNext() {
      switch (m_encoding) {
      case Encoding::UTF8:
        if constexpr (std::is_same_v<char, Target>) {
          auto utf8 = toUTF8(m_source);
          std::copy_if(utf8.begin(), utf8.end(), m_data.begin(),
                       [](auto value) -> bool { return value; });
        } else if constexpr (std::is_same_v<char16_t, Target>) {
          auto utf16 = toUTF16(m_source);
          std::copy_if(utf16.begin(), utf16.end(), m_data.begin(),
                       [](auto value) -> bool { return value; });
        } else if constexpr (std::is_same_v<char32_t, Target>) {
          m_data.front() = toUTF32(m_source);
        }
        break;
      case Encoding::UTF16LE:
      case Encoding::UTF16BE: {
        std::array<char16_t, 2> temp{};
        auto sizeAvailable = std::min<std::size_t>(4, m_source.size()) &
                             ~static_cast<std::size_t>(1);
        if (sizeAvailable == 0) {
          sizeAvailable = m_source.size();
        }
        std::memcpy(temp.data(), m_source.data(), sizeAvailable);
        auto viewSize = (std::size_t)std::ceil(sizeAvailable / 2.0);
        if constexpr (endian::native == endian::big) {
          if (m_encoding == Encoding::UTF16LE)
            std::transform(temp.begin(), temp.begin() + viewSize, temp.begin(),
                           swapByteOrder<char16_t>);

        } else if constexpr (endian::native == endian::little) {
          if (m_encoding == Encoding::UTF16BE)
            std::transform(temp.begin(), temp.begin() + viewSize, temp.begin(),
                           swapByteOrder<char16_t>);
        }
        auto view = std::u16string_view(temp.data(), viewSize);
        if constexpr (std::is_same_v<char, Target>) {
          auto utf8 = toUTF8(view);
          std::copy_if(utf8.begin(), utf8.end(), m_data.begin(),
                       [](auto value) -> bool { return value; });
        } else if constexpr (std::is_same_v<char16_t, Target>) {
          auto utf16 = toUTF16(view);
          std::copy_if(utf16.begin(), utf16.end(), m_data.begin(),
                       [](auto value) -> bool { return value; });
        } else if constexpr (std::is_same_v<char32_t, Target>) {
          m_data.front() = toUTF32(view);
        }
        m_source.remove_prefix(std::min<std::size_t>(
            m_source.size(), (viewSize - view.size()) * 2));
        break;
      }
      case Encoding::UTF32LE:
      case Encoding::UTF32BE:
        char32_t value;
        std::memcpy(&value, m_source.data(),
                    std::min<std::size_t>(4, m_source.size()));
        if constexpr (endian::native == endian::big) {
          if (m_encoding == Encoding::UTF32LE)
            value = swapByteOrder(value);

        } else if constexpr (endian::native == endian::little) {
          if (m_encoding == Encoding::UTF32BE)
            value = swapByteOrder(value);
        }
        if constexpr (std::is_same_v<char, Target>) {
          auto utf8 = toUTF8(value);
          std::copy_if(utf8.begin(), utf8.end(), m_data.begin(),
                       [](auto value) -> bool { return value; });
        } else if constexpr (std::is_same_v<char16_t, Target>) {
          auto utf16 = toUTF16(value);
          std::copy_if(utf16.begin(), utf16.end(), m_data.begin(),
                       [](auto value) -> bool { return value; });
        } else if constexpr (std::is_same_v<char32_t, Target>) {
          m_data.front() = toUTF32(value);
        }
        m_source.remove_prefix(std::min<std::size_t>(4, m_source.size()));
        break;
      }
    }

  public:
    using value_type = Target;
    using reference = const value_type&;
    using difference_type = std::ptrdiff_t;
    using pointer = const value_type*;
    using iterator_category = std::forward_iterator_tag;

    Iterator() = default;

    Iterator(std::string_view source, Encoding encoding)
        : m_source(source), m_encoding(encoding) {
      if (!source.empty())
        transcodeNext();
      else
        std::fill(m_data.begin(), m_data.end(), 0);
    }

    reference operator*() const {
      return m_data[0];
    }

    Iterator& operator++() {
      m_data.front() = 0;
      std::copy_backward(m_data.begin() + 1, m_data.end(), m_data.begin());
      if (!m_data.front() && !m_source.empty())
        transcodeNext();

      return *this;
    }

    Iterator operator++(int) {
      auto copy = *this;
      operator++();
      return copy;
    }

    bool operator==(const Iterator& rhs) const {
      return m_source.data() == rhs.m_source.data() && m_data == rhs.m_data;
    }

    bool operator!=(const Iterator& rhs) const {
      return !(rhs == *this);
    }

    friend void swap(Iterator& lhs, Iterator& rhs) {
      std::swap(lhs.m_source, rhs.m_source);
      std::swap(lhs.m_data, rhs.m_data);
      std::swap(lhs.m_index, rhs.m_index);
    }
  };

public:
  Transcoder(std::string_view source, Encoding encoding)
      : m_source(source), m_encoding(encoding) {}

  using value_type = Target;
  using reference = const value_type&;
  using const_reference = reference;
  using iterator = Iterator;
  using const_iterator = iterator;
  using difference_type = std::ptrdiff_t;
  using size_type = std::size_t;

  [[nodiscard]] iterator begin() const {
    return iterator(m_source, m_encoding);
  }

  const_iterator cbegin() {
    return begin();
  }

  iterator end() {
    return iterator(m_source.substr(m_source.size()), m_encoding);
  }

  const_iterator cend() {
    return end();
  }
};

bool isWhitespace(char32_t codepoint);

bool isValidCodepoint(char32_t codepoint);

std::size_t consoleWidth(char32_t codepoint);

std::optional<char32_t> fromName(std::string_view utf8name);

enum class Normalization { NFD, NFC, NFKD, NFKC };

std::string normalize(std::string_view utf8, Normalization normalization);

std::u32string normalize(std::u32string_view utf32,
                         Normalization normalization);

} // namespace pylir::Text
