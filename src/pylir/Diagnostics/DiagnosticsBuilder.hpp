#pragma once

#include <fmt/format.h>

#include "Document.hpp"

namespace pylir::Diag
{
enum class colour : uint32_t
{
    alice_blue = 0xF0F8FF,
    antique_white = 0xFAEBD7,
    aqua = 0x00FFFF,
    aquamarine = 0x7FFFD4,
    azure = 0xF0FFFF,
    beige = 0xF5F5DC,
    bisque = 0xFFE4C4,
    black = 0x000000,
    blanched_almond = 0xFFEBCD,
    blue = 0x0000FF,
    blue_violet = 0x8A2BE2,
    brown = 0xA52A2A,
    burly_wood = 0xDEB887,
    cadet_blue = 0x5F9EA0,
    chartreuse = 0x7FFF00,
    chocolate = 0xD2691E,
    coral = 0xFF7F50,
    cornflower_blue = 0x6495ED,
    cornsilk = 0xFFF8DC,
    crimson = 0xDC143C,
    cyan = 0x00FFFF,
    dark_blue = 0x00008B,
    dark_cyan = 0x008B8B,
    dark_golden_rod = 0xB8860B,
    dark_gray = 0xA9A9A9,
    dark_green = 0x006400,
    dark_khaki = 0xBDB76B,
    dark_magenta = 0x8B008B,
    dark_olive_green = 0x556B2F,
    dark_orange = 0xFF8C00,
    dark_orchid = 0x9932CC,
    dark_red = 0x8B0000,
    dark_salmon = 0xE9967A,
    dark_sea_green = 0x8FBC8F,
    dark_slate_blue = 0x483D8B,
    dark_slate_gray = 0x2F4F4F,
    dark_turquoise = 0x00CED1,
    dark_violet = 0x9400D3,
    deep_pink = 0xFF1493,
    deep_sky_blue = 0x00BFFF,
    dim_gray = 0x696969,
    dodger_blue = 0x1E90FF,
    fire_brick = 0xB22222,
    floral_white = 0xFFFAF0,
    forest_green = 0x228B22,
    fuchsia = 0xFF00FF,
    gainsboro = 0xDCDCDC,
    ghost_white = 0xF8F8FF,
    gold = 0xFFD700,
    golden_rod = 0xDAA520,
    gray = 0x808080,
    green = 0x008000,
    green_yellow = 0xADFF2F,
    honey_dew = 0xF0FFF0,
    hot_pink = 0xFF69B4,
    indian_red = 0xCD5C5C,
    indigo = 0x4B0082,
    ivory = 0xFFFFF0,
    khaki = 0xF0E68C,
    lavender = 0xE6E6FA,
    lavender_blush = 0xFFF0F5,
    lawn_green = 0x7CFC00,
    lemon_chiffon = 0xFFFACD,
    light_blue = 0xADD8E6,
    light_coral = 0xF08080,
    light_cyan = 0xE0FFFF,
    light_golden_rod_yellow = 0xFAFAD2,
    light_gray = 0xD3D3D3,
    light_green = 0x90EE90,
    light_pink = 0xFFB6C1,
    light_salmon = 0xFFA07A,
    light_sea_green = 0x20B2AA,
    light_sky_blue = 0x87CEFA,
    light_slate_gray = 0x778899,
    light_steel_blue = 0xB0C4DE,
    light_yellow = 0xFFFFE0,
    lime = 0x00FF00,
    lime_green = 0x32CD32,
    linen = 0xFAF0E6,
    magenta = 0xFF00FF,
    maroon = 0x800000,
    medium_aquamarine = 0x66CDAA,
    medium_blue = 0x0000CD,
    medium_orchid = 0xBA55D3,
    medium_purple = 0x9370DB,
    medium_sea_green = 0x3CB371,
    medium_slate_blue = 0x7B68EE,
    medium_spring_green = 0x00FA9A,
    medium_turquoise = 0x48D1CC,
    medium_violet_red = 0xC71585,
    midnight_blue = 0x191970,
    mint_cream = 0xF5FFFA,
    misty_rose = 0xFFE4E1,
    moccasin = 0xFFE4B5,
    navajo_white = 0xFFDEAD,
    navy = 0x000080,
    old_lace = 0xFDF5E6,
    olive = 0x808000,
    olive_drab = 0x6B8E23,
    orange = 0xFFA500,
    orange_red = 0xFF4500,
    orchid = 0xDA70D6,
    pale_golden_rod = 0xEEE8AA,
    pale_green = 0x98FB98,
    pale_turquoise = 0xAFEEEE,
    pale_violet_red = 0xDB7093,
    papaya_whip = 0xFFEFD5,
    peach_puff = 0xFFDAB9,
    peru = 0xCD853F,
    pink = 0xFFC0CB,
    plum = 0xDDA0DD,
    powder_blue = 0xB0E0E6,
    purple = 0x800080,
    rebecca_purple = 0x663399,
    red = 0xFF0000,
    rosy_brown = 0xBC8F8F,
    royal_blue = 0x4169E1,
    saddle_brown = 0x8B4513,
    salmon = 0xFA8072,
    sandy_brown = 0xF4A460,
    sea_green = 0x2E8B57,
    sea_shell = 0xFFF5EE,
    sienna = 0xA0522D,
    silver = 0xC0C0C0,
    sky_blue = 0x87CEEB,
    slate_blue = 0x6A5ACD,
    slate_gray = 0x708090,
    snow = 0xFFFAFA,
    spring_green = 0x00FF7F,
    steel_blue = 0x4682B4,
    tan = 0xD2B48C,
    teal = 0x008080,
    thistle = 0xD8BFD8,
    tomato = 0xFF6347,
    turquoise = 0x40E0D0,
    violet = 0xEE82EE,
    wheat = 0xF5DEB3,
    white = 0xFFFFFF,
    white_smoke = 0xF5F5F5,
    yellow = 0xFFFF00,
    yellow_green = 0x9ACD32
};

constexpr auto ERROR_COLOUR = colour::red;

constexpr auto WARNING_COLOUR = colour::magenta;

constexpr auto NOTE_COLOUR = colour::cyan;

constexpr auto INSERT_COLOUR = colour::lime_green;

constexpr auto ERROR_COMPLY = colour::peru;

constexpr auto WARNING_COMPLY = colour::peru;

constexpr auto NOTE_COMPLY = colour::peru;

enum class emphasis : uint8_t
{
    bold = 1,
    italic = 1 << 1,
    underline = 1 << 2,
    strikethrough = 1 << 3
};

enum Severity
{
    Warning,
    Error,
    Note
};

template <class T, class = void>
struct LocationProvider;

template <class T>
struct LocationProvider<T, std::enable_if_t<std::is_integral_v<T>>>
{
    static std::size_t getPos(T value) noexcept
    {
        return value;
    }

    static std::pair<std::size_t, std::size_t> getRange(T value) noexcept
    {
        return {value, value + 1};
    }
};

template <class T, class = void>
struct hasLocationProvider : std::false_type
{
};

template <class T>
struct hasLocationProvider<T, std::void_t<decltype(LocationProvider<std::decay_t<T>>::getPos)>> : std::true_type
{
};

class DiagnosticsBuilder
{
    struct Label
    {
        std::size_t start;
        std::size_t end;
        std::optional<std::string> labelText;
        std::optional<colour> optionalColour;
        std::optional<emphasis> optionalEmphasis;
    };

    struct Message
    {
        Document* document;
        std::size_t location;
        std::string message;
        std::vector<Label> labels;
    };
    std::vector<Message> m_messages;

    static std::string printLine(std::size_t width, std::size_t lineNumber, pylir::Diag::Document& document,
                                 std::vector<Label> labels);

    std::string emitMessage(const Message& message, Severity severity) const;

public:
    template <class T, class S, class... Args>
    DiagnosticsBuilder(Document& document, const T& location, const S& message, Args&&... args)
        : m_messages{Message{&document,
                             LocationProvider<std::decay_t<T>>::getPos(location),
                             fmt::format(message, std::forward<Args>(args)...),
                             {}}}
    {
    }

    template <class T, class U>
    auto addLabel(const T& start, const U& end, std::optional<std::string>&& labelText = std::nullopt,
                  std::optional<colour>&& colour = std::nullopt,
                  std::optional<emphasis>&& emphasis =
                      std::nullopt) & -> std::enable_if_t<hasLocationProvider<T>{} && hasLocationProvider<U>{},
                                                          DiagnosticsBuilder&>
    {
        m_messages.back().labels.push_back({LocationProvider<std::decay_t<T>>::getPos(start),
                                            LocationProvider<std::decay_t<U>>::getPos(end), std::move(labelText),
                                            std::move(colour), std::move(emphasis)});
        return *this;
    }

    template <class T, class U>
    [[nodiscard]] auto addLabel(const T& start, const U& end, std::optional<std::string>&& labelText = std::nullopt,
                                std::optional<colour>&& colour = std::nullopt,
                                std::optional<emphasis>&& emphasis = std::nullopt) && -> std::
        enable_if_t<hasLocationProvider<T>{} && hasLocationProvider<U>{}, DiagnosticsBuilder&&>
    {
        return std::move(addLabel(start, end, std::move(labelText), std::move(colour), std::move(emphasis)));
    }

    template <class T>
    auto addLabel(const T& pos, std::optional<std::string>&& labelText = std::nullopt,
                  std::optional<colour>&& colour = std::nullopt,
                  std::optional<emphasis>&& emphasis =
                      std::nullopt) & -> std::enable_if_t<hasLocationProvider<T>{}, DiagnosticsBuilder&>
    {
        auto [start, end] = LocationProvider<std::decay_t<T>>::getRange(pos);
        return addLabel(start, end, std::move(labelText), std::move(colour), std::move(emphasis));
    }

    template <class T>
    [[nodiscard]] auto addLabel(const T& start, std::optional<std::string>&& labelText = std::nullopt,
                                std::optional<colour>&& colour = std::nullopt,
                                std::optional<emphasis>&& emphasis =
                                    std::nullopt) && -> std::enable_if_t<hasLocationProvider<T>{}, DiagnosticsBuilder&&>
    {
        return std::move(addLabel(start, std::move(labelText), std::move(colour), std::move(emphasis)));
    }

    template <class T, class S, class... Args>
    auto addNote(const T& location, const S& message,
                 Args&&... args) & -> std::enable_if_t<hasLocationProvider<T>{}, DiagnosticsBuilder&>
    {
        return addNote(*m_messages.back().document, LocationProvider<std::decay_t<T>>::getPos(location),
                       fmt::format(message, std::forward<Args>(args)...));
    }

    template <class T, class S, class... Args>
    auto addNote(Document& document, const T& location, const S& message,
                 Args&&... args) & -> std::enable_if_t<hasLocationProvider<T>{}, DiagnosticsBuilder&>
    {
        m_messages.push_back({&document,
                              LocationProvider<std::decay_t<T>>::getPos(location),
                              fmt::format(message, std::forward<Args>(args)...),
                              {}});
        return *this;
    }

    template <class T, class S, class... Args>
    [[nodiscard]] auto addNote(const T& location, const S& message,
                               Args&&... args) && -> std::enable_if_t<hasLocationProvider<T>{}, DiagnosticsBuilder&&>
    {
        return std::move(addNote(*m_messages.back().document, location, message, std::forward<Args>(args)...));
    }

    template <class T, class S, class... Args>
    [[nodiscard]] auto addNote(Document& document, const T& location, const S& message,
                               Args&&... args) && -> std::enable_if_t<hasLocationProvider<T>{}, DiagnosticsBuilder&&>
    {
        return std::move(addNote(document, location, message, std::forward<Args>(args)...));
    }

    std::string emit(Severity severity) const
    {
        auto begin = m_messages.begin();
        std::string result = emitMessage(*begin++, severity);
        std::for_each(begin, m_messages.end(), [&](auto& message) { result += emitMessage(message, Severity::Note); });
        return result;
    }

    std::string emitError() const
    {
        return emit(Severity::Error);
    }

    std::string emitWarning() const
    {
        return emit(Severity::Warning);
    }
};

} // namespace pylir::Diag
