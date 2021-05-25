
#pragma once

#include <array>
#include <optional>
#include <string>
#include <string_view>

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



} // namespace pylir::Text
