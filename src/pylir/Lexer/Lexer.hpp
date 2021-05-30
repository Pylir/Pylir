
#pragma once

#include <pylir/Diagnostics/DiagnosticsBuilder.hpp>
#include <pylir/Diagnostics/Document.hpp>
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
    std::vector<Token> m_tokens;
    Diag::Document* m_document;
    Diag::Document::iterator m_current;
    std::function<void(Diag::DiagnosticsBuilder&& diagnosticsBuilder)> m_diagCallback;

    static void outputToStderr(Diag::DiagnosticsBuilder&& diagnosticsBuilder);

    bool parseNext();

public:
    using value_type = Token;
    using reference = const Token&;
    using const_reference = reference;
    using iterator = LazyCacheIterator<value_type, Lexer, &Lexer::parseNext, &Lexer::m_tokens>;
    using const_iterator = iterator;
    using difference_type = iterator::difference_type;
    using size_type = std::size_t;

    explicit Lexer(Diag::Document& document, int fileId = 0,
                   std::function<void(Diag::DiagnosticsBuilder&& diagnosticsBuilder)> diagCallback = outputToStderr);

    Lexer(const Lexer&) = delete;
    Lexer& operator=(const Lexer&) = delete;

    Lexer(Lexer&&) noexcept = default;
    Lexer& operator=(Lexer&&) noexcept = default;

    [[nodiscard]] iterator begin()
    {
        return iterator(*this, 0);
    }

    [[nodiscard]] const_iterator cbegin()
    {
        return begin();
    }

    [[nodiscard]] iterator end()
    {
        return iterator(*this, -1);
    }

    [[nodiscard]] const_iterator cend()
    {
        return end();
    }

    [[nodiscard]] Diag::DiagnosticsBuilder createDiagnosticsBuilder(std::size_t location, std::string_view message);
};
} // namespace pylir
