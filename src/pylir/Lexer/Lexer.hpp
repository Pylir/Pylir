
#pragma once

#include <pylir/Diagnostics/DiagnosticsBuilder.hpp>
#include <pylir/Support/LazyCacheIterator.hpp>
#include <pylir/Support/Macros.hpp>
#include <pylir/Support/Text.hpp>

#include <cstdint>
#include <optional>
#include <string_view>
#include <vector>

#include "Token.hpp"

namespace pylir
{
class Lexer
{
    int m_fileId;
    std::string_view m_source;
    std::vector<Token> m_tokens;
    std::vector<int> m_lineStarts{0};
    Text::Encoding m_encoding;
    std::optional<Text::Transcoder<void, char32_t>> m_transcoder; // Not really optional, but have to late init :/
    Text::Transcoder<void, char32_t>::iterator m_current;

    bool parseNext();

public:
    using value_type = Token;
    using reference = const Token&;
    using const_reference = reference;
    using iterator = LazyCacheIterator<value_type, Lexer, &Lexer::parseNext, &Lexer::m_tokens>;
    using const_iterator = iterator;
    using difference_type = iterator::difference_type;
    using size_type = std::size_t;

    explicit Lexer(std::string_view source, int fileId = 0);

    Lexer(const Lexer&) = delete;
    Lexer& operator=(const Lexer&) = delete;

    Lexer(Lexer&&) noexcept = default;
    Lexer& operator=(Lexer&&) noexcept = default;

    iterator begin()
    {
        return iterator(*this, 0);
    }

    const_iterator cbegin()
    {
        return begin();
    }

    iterator end()
    {
        return iterator(*this, -1);
    }

    const_iterator cend()
    {
        return end();
    }

    Diag::DiagnosticsBuilder createDiagnosticsBuilder(std::size_t location, std::string_view message);
};
} // namespace pylir
