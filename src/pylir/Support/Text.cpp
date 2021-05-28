
#include "Text.hpp"

#include <llvm/Support/ConvertUTF.h>
#include <llvm/Support/UnicodeCharRanges.h>

#include <utf8proc/utf8proc.h>

std::optional<pylir::Text::Encoding> pylir::Text::checkForBOM(std::string_view bytes)
{
    constexpr auto startsWith = [](std::string_view view, const auto& prefix)
    { return view.substr(0, prefix.size()) == std::string_view(prefix.data(), prefix.size()); };
    // TODO: use starts_with in C++20
    if (startsWith(bytes, UTF8_BOM))
    {
        return Encoding::UTF8;
    }
    if (startsWith(bytes, UTF16BE_BOM))
    {
        return Encoding::UTF16BE;
    }
    if (startsWith(bytes, UTF16LE_BOM))
    {
        return Encoding::UTF16BE;
    }
    if (startsWith(bytes, UTF32BE_BOM))
    {
        return Encoding::UTF32BE;
    }
    if (startsWith(bytes, UTF32LE_BOM))
    {
        return Encoding::UTF32LE;
    }
    return std::nullopt;
}

std::optional<pylir::Text::Encoding> pylir::Text::readBOM(std::string_view& bytes)
{
    auto result = checkForBOM(bytes);
    if (result)
    {
        switch (*result)
        {
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

constexpr static std::array<char, 4> REPLACEMENT_CHARACTER_UTF8 = {static_cast<char>(0xEF), static_cast<char>(0xBF),
                                                                   static_cast<char>(0xBD), 0};

std::array<char, 4> pylir::Text::toUTF8(std::string_view& utf8, bool* legal)
{
    auto size = llvm::getNumBytesForUTF8(utf8.front());
    auto result = llvm::isLegalUTF8Sequence(
        reinterpret_cast<const llvm::UTF8*>(utf8.data()),
        reinterpret_cast<const llvm::UTF8*>(utf8.data() + std::min<std::size_t>(size, utf8.size())));
    if (legal)
    {
        *legal = result;
    }
    if (!result)
    {
        // Consume just one byte on error
        utf8.remove_prefix(1);
        return REPLACEMENT_CHARACTER_UTF8;
    }
    std::array<char, 4> array{};
    std::copy(utf8.begin(), utf8.begin() + size, array.begin());
    return array;
}

namespace
{
template <class T>
bool resultOk(llvm::ConversionResult result, T* targetBegin, T* targetNow)
{
    return result == llvm::conversionOK || result == llvm::targetExhausted
           || (result == llvm::sourceExhausted && targetNow - targetBegin > 0);
}
} // namespace

std::array<char, 4> pylir::Text::toUTF8(std::u16string_view& utf16, bool* legal)
{
    std::array<llvm::UTF8, 4> array{};
    std::array<llvm::UTF16, 2> llvmUtf16;
    std::copy(utf16.begin(), utf16.begin() + std::min<std::size_t>(2, utf16.size()), llvmUtf16.begin());
    const auto* sourceBegin = llvmUtf16.data();
    auto* targetBegin = array.data();
    auto result = llvm::ConvertUTF16toUTF8(&sourceBegin, sourceBegin + std::min<std::size_t>(2, utf16.size()),
                                           &targetBegin, array.data() + array.size(), llvm::strictConversion);
    bool ok = resultOk(result, array.data(), targetBegin);
    if (legal)
    {
        *legal = ok;
    }
    if (!ok)
    {
        utf16.remove_prefix(1);
        return REPLACEMENT_CHARACTER_UTF8;
    }
    utf16.remove_prefix(sourceBegin - llvmUtf16.data());
    std::array<char, 4> charArray;
    std::copy(array.begin(), array.end(), charArray.begin());
    return charArray;
}

std::array<char, 4> pylir::Text::toUTF8(char32_t utf32, bool* legal)
{
    std::array<char, 4> array{};
    auto* targetBegin = array.data();
    auto result = llvm::ConvertCodePointToUTF8(utf32, targetBegin);
    if (legal)
    {
        *legal = result;
    }
    if (!result)
    {
        return REPLACEMENT_CHARACTER_UTF8;
    }
    return array;
}

constexpr static std::array<char16_t, 2> REPLACEMENT_CHARACTER_UTF16 = {0xFFFD, 0};

std::array<char16_t, 2> pylir::Text::toUTF16(std::u16string_view& utf16, bool* legal)
{
    // Using the convert function, just for verification purposes
    bool ok;
    auto copy = utf16;
    (void)toUTF8(utf16, &ok);
    if (legal)
    {
        *legal = ok;
    }
    if (!ok)
    {
        return REPLACEMENT_CHARACTER_UTF16;
    }
    std::array<char16_t, 2> result;
    std::copy(copy.data(), utf16.data(), result.begin());
    return result;
}

std::array<char16_t, 2> pylir::Text::toUTF16(std::string_view& utf8, bool* legal)
{
    std::array<llvm::UTF8, 4> llvmUtf8{};
    std::copy(utf8.begin(), utf8.begin() + std::min<std::size_t>(4, utf8.size()), llvmUtf8.begin());
    std::array<llvm::UTF16, 2> llvmUtf16;
    const auto* sourceBegin = llvmUtf8.data();
    auto* targetBegin = llvmUtf16.data();
    auto result = llvm::ConvertUTF8toUTF16(&sourceBegin, sourceBegin + std::min<std::size_t>(4, utf8.size()),
                                           &targetBegin, llvmUtf16.data() + llvmUtf16.size(), llvm::strictConversion);
    bool ok = resultOk(result, llvmUtf16.data(), targetBegin);
    if (legal)
    {
        *legal = ok;
    }
    if (!ok)
    {
        utf8.remove_prefix(1);
        return REPLACEMENT_CHARACTER_UTF16;
    }
    std::array<char16_t, 2> charArray;
    std::copy(llvmUtf16.begin(), llvmUtf16.end(), charArray.begin());
    utf8.remove_prefix(sourceBegin - llvmUtf8.data());
    return charArray;
}

std::array<char16_t, 2> pylir::Text::toUTF16(char32_t utf32, bool* legal)
{
    std::array<llvm::UTF16, 2> llvmUtf16;
    llvm::UTF32 source = utf32;
    const auto* sourceBegin = &source;
    auto* targetBegin = llvmUtf16.data();
    auto result = llvm::ConvertUTF32toUTF16(&sourceBegin, sourceBegin + 1, &targetBegin,
                                            llvmUtf16.data() + llvmUtf16.size(), llvm::strictConversion);
    bool ok = resultOk(result, llvmUtf16.data(), targetBegin);
    if (legal)
    {
        *legal = ok;
    }
    if (!ok)
    {
        return REPLACEMENT_CHARACTER_UTF16;
    }
    std::array<char16_t, 2> charArray;
    std::copy(llvmUtf16.begin(), llvmUtf16.end(), charArray.begin());
    return charArray;
}

constexpr static char32_t REPLACEMENT_CHARACTER_UTF32 = 0xFFFD;

char32_t pylir::Text::toUTF32(std::string_view& utf8, bool* legal)
{
    std::array<llvm::UTF8, 4> llvmUtf8{};
    std::copy(utf8.begin(), utf8.begin() + std::min<std::size_t>(4, utf8.size()), llvmUtf8.begin());
    const auto* sourceBegin = llvmUtf8.data();
    llvm::UTF32 llvmUTF32;
    auto* targetBegin = &llvmUTF32;
    auto result = llvm::ConvertUTF8toUTF32(&sourceBegin, sourceBegin + std::min<std::size_t>(4, utf8.size()),
                                           &targetBegin, targetBegin + 1, llvm::strictConversion);
    bool ok = resultOk(result, &llvmUTF32, targetBegin);
    if (legal)
    {
        *legal = ok;
    }
    if (!ok)
    {
        utf8.remove_prefix(1);
        return REPLACEMENT_CHARACTER_UTF32;
    }
    utf8.remove_prefix(sourceBegin - llvmUtf8.data());
    return llvmUTF32;
}

char32_t pylir::Text::toUTF32(std::u16string_view& utf16, bool* legal)
{
    std::array<llvm::UTF16, 2> llvmUtf16{};
    std::copy(utf16.begin(), utf16.begin() + std::min<std::size_t>(2, utf16.size()), llvmUtf16.begin());
    const auto* sourceBegin = llvmUtf16.data();
    llvm::UTF32 llvmUTF32;
    auto* targetBegin = &llvmUTF32;
    auto result = llvm::ConvertUTF16toUTF32(&sourceBegin, sourceBegin + std::min<std::size_t>(2, utf16.size()),
                                            &targetBegin, targetBegin + 1, llvm::strictConversion);
    bool ok = resultOk(result, &llvmUTF32, targetBegin);
    if (legal)
    {
        *legal = ok;
    }
    if (!ok)
    {
        utf16.remove_prefix(1);
        return REPLACEMENT_CHARACTER_UTF32;
    }
    utf16.remove_prefix(sourceBegin - llvmUtf16.data());
    return llvmUTF32;
}

char32_t pylir::Text::toUTF32(char32_t utf32, bool* legal)
{
    // Just verify by converting to another format
    bool ok = utf8proc_codepoint_valid(utf32);
    if (legal)
    {
        *legal = ok;
    }
    if (!ok)
    {
        return REPLACEMENT_CHARACTER_UTF32;
    }
    return utf32;
}

std::string pylir::Text::toUTF8String(std::u16string_view utf16, bool* legal)
{
    if (legal)
    {
        *legal = true;
    }
    std::string result;
    result.reserve(utf16.size() * 2);
    while (!utf16.empty())
    {
        bool ok;
        auto utf8 = toUTF8(utf16, &ok);
        if (legal)
        {
            *legal = *legal && ok;
        }
        for (auto character : utf8)
        {
            if (!character)
            {
                break;
            }
            result += character;
        }
    }
    return result;
}

std::string pylir::Text::toUTF8String(std::u32string_view utf32, bool* legal)
{
    if (legal)
    {
        *legal = true;
    }
    std::string result;
    result.reserve(utf32.size() * 4);
    for (auto codepoint : utf32)
    {
        bool ok;
        auto utf8 = toUTF8(codepoint, &ok);
        if (legal)
        {
            *legal = *legal && ok;
        }
        for (auto character : utf8)
        {
            if (!character)
            {
                break;
            }
            result += character;
        }
    }
    return result;
}

std::u32string pylir::Text::toUTF32String(std::string_view utf8, bool* legal)
{
    if (legal)
    {
        *legal = true;
    }
    std::u32string result;
    result.reserve(utf8.size() / 2);
    while (!utf8.empty())
    {
        bool ok;
        result += toUTF32(utf8, &ok);
        if (legal)
        {
            *legal = *legal && ok;
        }
    }
    return result;
}

bool pylir::Text::isWhitespace(char32_t codepoint)
{
    auto properties = utf8proc_get_property(codepoint);
    return properties->category == UTF8PROC_CATEGORY_ZS || properties->bidi_class == UTF8PROC_BIDI_CLASS_WS
           || properties->bidi_class == UTF8PROC_BIDI_CLASS_B || properties->bidi_class == UTF8PROC_BIDI_CLASS_S;
}

std::string pylir::Text::normalize(std::string_view utf8, pylir::Text::Normalization normalization)
{
    [[maybe_unused]] bool ok;
    auto u32 = toUTF32String(utf8, &ok);
    PYLIR_ASSERT(ok);
    return toUTF8String(normalize(u32, normalization));
}

std::u32string pylir::Text::normalize(std::u32string_view utf32, pylir::Text::Normalization normalization)
{
    utf8proc_option_t option;
    switch (normalization)
    {
        case Normalization::NFD: option = static_cast<utf8proc_option_t>(UTF8PROC_STABLE | UTF8PROC_DECOMPOSE); break;
        case Normalization::NFC: option = static_cast<utf8proc_option_t>(UTF8PROC_STABLE | UTF8PROC_COMPOSE); break;
        case Normalization::NFKD:
            option = static_cast<utf8proc_option_t>(UTF8PROC_STABLE | UTF8PROC_DECOMPOSE | UTF8PROC_COMPAT);
            break;
        case Normalization::NFKC:
            option = static_cast<utf8proc_option_t>(UTF8PROC_STABLE | UTF8PROC_COMPOSE | UTF8PROC_COMPAT);
            break;
        default: PYLIR_UNREACHABLE;
    }
    std::vector<utf8proc_int32_t> buffer(utf32.size());
    std::size_t actuallyWritten = 0;
    for (auto codepoint : utf32)
    {
        do
        {
            auto size = utf8proc_decompose_char(codepoint, buffer.data() + actuallyWritten,
                                                buffer.size() - actuallyWritten, option, nullptr);
            PYLIR_ASSERT(size >= 0);
            if (static_cast<std::size_t>(size) > buffer.size() - actuallyWritten)
            {
                buffer.resize(std::max<std::size_t>(buffer.size() * 2, buffer.size() + size));
                continue;
            }
            actuallyWritten += size;
            break;
        } while (true);
    }
    return std::u32string(buffer.begin(), buffer.begin() + actuallyWritten);
}
