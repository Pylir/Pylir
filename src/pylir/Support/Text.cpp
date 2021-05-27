
#include "Text.hpp"

#include <llvm/Support/ConvertUTF.h>

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

std::array<char, 4> pylir::Text::toUTF8(std::u16string_view& utf16, bool* legal)
{
    std::array<llvm::UTF8, 4> array{};
    std::array<llvm::UTF16, 2> llvmUtf16;
    std::copy(utf16.begin(), utf16.begin() + std::min<std::size_t>(2, utf16.size()), llvmUtf16.begin());
    const auto* sourceBegin = llvmUtf16.data();
    auto* targetBegin = array.data();
    auto result = llvm::ConvertUTF16toUTF8(&sourceBegin, sourceBegin + std::min<std::size_t>(2, utf16.size()),
                                           &targetBegin, array.data() + array.size(), llvm::strictConversion);
    bool ok = result == llvm::conversionOK || result == llvm::targetExhausted;
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
        *legal = result == llvm::conversionOK;
    }
    if (result != llvm::conversionOK)
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
    bool ok = result == llvm::conversionOK || result == llvm::targetExhausted;
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
    bool ok = result == llvm::conversionOK || result == llvm::targetExhausted;
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
    bool ok = result == llvm::conversionOK || result == llvm::targetExhausted;
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
    bool ok = result == llvm::conversionOK || result == llvm::targetExhausted;
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
    bool ok;
    (void)toUTF16(utf32, &ok);
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
