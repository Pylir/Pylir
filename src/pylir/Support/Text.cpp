
#include "Text.hpp"

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
