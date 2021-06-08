
#pragma once

#include <pylir/Diagnostics/DiagnosticsBuilder.hpp>
#include <pylir/Diagnostics/Document.hpp>
#include <pylir/Lexer/Lexer.hpp>

#include <tl/expected.hpp>

namespace pylir
{
class Parser
{
    Lexer m_lexer;
    Lexer::iterator m_current;

public:

    explicit Parser(
        Diag::Document& document, int fileId = 0,
        std::function<void(Diag::DiagnosticsBuilder&& diagnosticsBuilder)> callBack = [](auto&&) {})
        : m_lexer(document, fileId, std::move(callBack)), m_current(m_lexer.begin())
    {
    }
};
} // namespace pylir
