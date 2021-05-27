
#pragma once

#include <algorithm>
#include <array>
#include <cstring>
#include <functional>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "Endian.hpp"
#include "Macros.hpp"

namespace pylir::Text
{
enum class Encoding
{
    UTF8,
    UTF16LE,
    UTF16BE,
    UTF32LE,
    UTF32BE
};

constexpr std::array<char, 3> UTF8_BOM = {static_cast<char>(0xEF), static_cast<char>(0xBB), static_cast<char>(0xBF)};
constexpr std::array<char, 2> UTF16LE_BOM = {static_cast<char>(0xFF), static_cast<char>(0xFE)};
constexpr std::array<char, 2> UTF16BE_BOM = {static_cast<char>(0xFE), static_cast<char>(0xFF)};
constexpr std::array<char, 4> UTF32LE_BOM = {static_cast<char>(0xFF), static_cast<char>(0xFE), 0, 0};
constexpr std::array<char, 4> UTF32BE_BOM = {0, 0, static_cast<char>(0xFE), static_cast<char>(0xFF)};

/**
 * Checks if the start of the view contains a BOM indicating UTF-8, UTF-16 or UTF-32.
 *
 * @param bytes view into a list of bytes. No encoding in particular is assumed yet
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
 * @param legal if not null, set to false when either 1) the utf-8 source was exhausted or 2) the resulting
 *              codepoint(s) are illegal.
 *
 * @return Input, or unicode replacement character on error
 */
std::array<char, 4> toUTF8(std::string_view& utf8, bool* legal = nullptr);

/**
 * Converts a single unicode codepoint from UTF-16 to UTF-8.
 * The utf16 source is therefore 1 to 2 bytes. Read bytes are consumed.
 *
 * @param utf16 UTF16 source
 * @param legal if not null, set to false when either 1) the utf-16 source was exhausted or 2) the resulting
 *              codepoint(s) are illegal. The unicode replacement character is returned in either case.
 *
 * @return the codepoint in UTF8. Index 0 is guaranteed to have a value, Index 1 to 3 are 0 if not needed
 */
std::array<char, 4> toUTF8(std::u16string_view& utf16, bool* legal = nullptr);

/**
 * Converts a single unicode codepoint from UTF-32 to UTF-8.
 *
 * @param utf32 UTF32 source
 * @param legal if not null, set to false when the resulting codepoint(s) are illegal.
 *              The unicode replacement character is returned in that case.
 *
 * @return the codepoint in UTF8. Index 0 is guaranteed to have a value, Index 1 to 3 are 0 if not needed
 */
std::array<char, 4> toUTF8(char32_t utf32, bool* legal = nullptr);

/**
 * Does no conversion, but instead checks whether the input is proper UTF-16.
 * The utf16 source is 1 to 2 bytes. Read bytes are consumed.
 *
 * @param utf16 UTF16 source
 * @param legal if not null, set to false when either 1) the utf-16 source was exhausted or 2) the resulting
 *              codepoint(s) are illegal. The unicode replacement character is returned in either case.
 *
 * @return Input, or unicode replacement character on error
 */
std::array<char16_t, 2> toUTF16(std::u16string_view& utf16, bool* legal = nullptr);

/**
 * Converts a single unicode codepoint from UTF-8 to UTF-16.
 * The utf8 source is therefore 1 to 4 bytes. Read bytes are consumed
 *
 * @param utf8 UTF8 source
 * @param legal if not null, set to false when either 1) the utf-8 source was exhausted or 2) the resulting codepoint(s)
 *              are illegal. The unicode replacement character is returned in either case.
 *
 * @return the codepoint in UTF16. Index 0 is guaranteed to have a value, Index 1 is 0 if not needed
 */
std::array<char16_t, 2> toUTF16(std::string_view& utf8, bool* legal = nullptr);

/**
 * Converts a single unicode codepoint from UTF-32 to UTF-16
 *
 * @param utf32 UTF-32 source
 * @param legal if not null, set to false if the resulting UTF-16 codepoint is illegal.
 *              The unicode replacement character is returned in that case.
 *
 * @return the codepoint in UTF16. Index 0 is guaranteed to have a value, Index 1 is 0 if not needed
 */
std::array<char16_t, 2> toUTF16(char32_t utf32, bool* legal = nullptr);

/**
 * Converts a single unicode codepoint from UTF-8 to UTF-32.
 * The utf8 source is therefore 1 to 4 bytes. Read bytes are consumed.
 *
 * @param utf8 UTF8 source
 * @param legal if not null, set to false when either 1) the utf-8 source was exhausted or 2) the resulting codepoint(s)
 *              are illegal
 *
 * @return the codepoint in UTF32
 */
char32_t toUTF32(std::string_view& utf8, bool* legal = nullptr);

/**
 * Converts a single unicode codepoint from UTF-16 to UTF-32.
 * The utf8 source is therefore 1 to 2 bytes. Read bytes are consumed.
 *
 * @param utf16 UTF16 source
 * @param legal if not null, set to false when either 1) the utf-16 source was exhausted or 2) the resulting
 *              codepoint(s) are illegal. The unicode replacement character is returned in either case.
 *
 * @return the codepoint in UTF32
 */
char32_t toUTF32(std::u16string_view& utf16, bool* legal = nullptr);

/**
 * Does no conversion, but instead checks whether the input is proper UTF-32
 *
 * @param utf16 UTF16 source
 * @param legal if not null, set to false when the resulting codepoint(s) are illegal.
 * @return Input, or unicode replacement character on error
 */
char32_t toUTF32(char32_t utf32, bool* legal = nullptr);

template <class Source, class Target>
class Transcoder;

template <class Target>
class Transcoder<void, Target>
{
    static_assert(std::is_same_v<char, Target> || std::is_same_v<char16_t, Target> || std::is_same_v<char32_t, Target>);

    std::string_view m_source;
    Encoding m_encoding;
    std::vector<Target> m_results;

    class Iterator
    {
        Transcoder* m_transcoder;
        std::size_t m_index;

    public:
        using difference_type = std::ptrdiff_t;
        using value_type = const Target;
        using pointer = value_type*;
        using reference = value_type&;
        using iterator_category = std::bidirectional_iterator_tag;

        Iterator() = default;

        Iterator(Transcoder& transcoder, std::size_t index) : m_transcoder(&transcoder), m_index(index) {}

        reference operator*() const
        {
            return m_transcoder->m_results[m_index];
        }

        Iterator& operator++()
        {
            if (m_index + 1 >= m_transcoder->m_results.size())
            {
                if (!m_transcoder->transcodeNext())
                {
                    m_index = -1;
                    return *this;
                }
            }
            m_index++;
            return *this;
        }

        Iterator operator++(int)
        {
            auto copy = *this;
            operator++();
            return copy;
        }

        Iterator operator--()
        {
            m_index--;
            return *this;
        }

        Iterator& operator--(int)
        {
            auto copy = *this;
            operator--();
            return copy;
        }

        bool operator==(const Iterator& rhs) const
        {
            return m_transcoder == rhs.m_transcoder && m_index == rhs.m_index;
        }

        bool operator!=(const Iterator& rhs) const
        {
            return !(rhs == *this);
        }

        friend void swap(Iterator& lhs, Iterator& rhs)
        {
            std::swap(lhs.m_transcoder, rhs.m_transcoder);
            std::swap(lhs.m_index, rhs.m_index);
        }
    };

    friend class Iterator;

    bool transcodeNext()
    {
        if (m_source.empty())
        {
            return false;
        }
        switch (m_encoding)
        {
            case Encoding::UTF8:
                if constexpr (std::is_same_v<char, Target>)
                {
                    auto utf8 = toUTF8(m_source);
                    std::copy_if(utf8.begin(), utf8.end(), std::back_inserter(m_results),
                                 [](auto value) -> bool { return value; });
                }
                else if constexpr (std::is_same_v<char16_t, Target>)
                {
                    auto utf16 = toUTF16(m_source);
                    std::copy_if(utf16.begin(), utf16.end(), std::back_inserter(m_results),
                                 [](auto value) -> bool { return value; });
                }
                else if constexpr (std::is_same_v<char32_t, Target>)
                {
                    m_results.emplace_back(toUTF32(m_source));
                }
                break;
            case Encoding::UTF16LE:
            case Encoding::UTF16BE:
            {
                std::array<char16_t, 2> temp{};
                auto sizeAvailable = std::min<std::size_t>(4, m_source.size()) % 2;
                std::memcpy(temp.data(), m_source.data(), sizeAvailable);
                auto viewSize = sizeAvailable / 2;
                if (endian::native == endian::big && m_encoding == Encoding::UTF16LE)
                {
                    std::transform(temp.begin(), temp.begin() + viewSize, temp.begin(), swapByteOrder<char16_t>);
                }
                else if (endian::native == endian::little && m_encoding == Encoding::UTF16BE)
                {
                    std::transform(temp.begin(), temp.begin() + viewSize, temp.begin(), swapByteOrder<char16_t>);
                }
                auto view = std::u16string_view(temp.data(), viewSize);
                if constexpr (std::is_same_v<char, Target>)
                {
                    auto utf8 = toUTF8(view);
                    std::copy_if(utf8.begin(), utf8.end(), std::back_inserter(m_results),
                                 [](auto value) -> bool { return value; });
                }
                else if constexpr (std::is_same_v<char16_t, Target>)
                {
                    auto utf16 = toUTF16(view);
                    std::copy_if(utf16.begin(), utf16.end(), std::back_inserter(m_results),
                                 [](auto value) -> bool { return value; });
                }
                else if constexpr (std::is_same_v<char32_t, Target>)
                {
                    m_results.emplace_back(toUTF32(view));
                }
                m_source.remove_suffix(viewSize - view.size());
                break;
            }
            case Encoding::UTF32LE:
            case Encoding::UTF32BE:
                char32_t value;
                std::memcpy(&value, m_source.data(), std::min<std::size_t>(4, m_source.size()));
                if (endian::native == endian::big && m_encoding == Encoding::UTF16LE)
                {
                    value = swapByteOrder(value);
                }
                else if (endian::native == endian::little && m_encoding == Encoding::UTF16BE)
                {
                    value = swapByteOrder(value);
                }
                if constexpr (std::is_same_v<char, Target>)
                {
                    auto utf8 = toUTF8(value);
                    std::copy_if(utf8.begin(), utf8.end(), std::back_inserter(m_results),
                                 [](auto value) -> bool { return value; });
                }
                else if constexpr (std::is_same_v<char16_t, Target>)
                {
                    auto utf16 = toUTF16(value);
                    std::copy_if(utf16.begin(), utf16.end(), std::back_inserter(m_results),
                                 [](auto value) -> bool { return value; });
                }
                else if constexpr (std::is_same_v<char32_t, Target>)
                {
                    m_results.emplace_back(toUTF32(value));
                }
                break;
        }
        return true;
    }

public:
    Transcoder(std::string_view source, Encoding encoding) : m_source(source), m_encoding(encoding) {}

    using value_type = const Target;
    using reference = value_type&;
    using const_reference = reference;
    using iterator = Iterator;
    using const_iterator = iterator;
    using difference_type = std::ptrdiff_t;
    using size_type = std::size_t;

    iterator begin()
    {
        if (m_results.empty())
        {
            if (!transcodeNext())
            {
                return end();
            }
        }
        return Iterator(*this, 0);
    }

    const_iterator cbegin()
    {
        return begin();
    }

    iterator end()
    {
        return Iterator(*this, -1);
    }

    const_iterator cend()
    {
        return end();
    }
};

} // namespace pylir::Text
