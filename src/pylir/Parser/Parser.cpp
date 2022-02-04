#include "Parser.hpp"

#include <pylir/Diagnostics/DiagnosticMessages.hpp>

#include "Visitor.hpp"

void pylir::Parser::addToNamespace(const pylir::Token& token)
{
    PYLIR_ASSERT(token.getTokenType() == TokenType::Identifier);
    auto identifierToken = IdentifierToken{token};
    addToNamespace(identifierToken);
}

void pylir::Parser::addToNamespace(const IdentifierToken& token)
{
    if (m_namespace.empty())
    {
        m_globals.insert(token);
        return;
    }
    auto result = m_namespace.back().identifiers.find(token);
    if (result == m_namespace.back().identifiers.end() || result->second == Scope::Kind::Unknown)
    {
        m_namespace.back().identifiers.insert_or_assign(result, token, Scope::Kind::Local);
    }
}

void pylir::Parser::addToNamespace(const Syntax::TargetList& targetList)
{
    class TargetVisitor : public Syntax::Visitor<TargetVisitor>
    {
    public:
        std::function<void(const pylir::IdentifierToken&)> callback;

        using Visitor::visit;

        void visit(const Syntax::Target& target)
        {
            if (const auto* identifier = std::get_if<IdentifierToken>(&target.variant))
            {
                callback(*identifier);
            }
            Visitor::visit(target);
        }
    } visitor{{},
              [&](const IdentifierToken& token)
              { addToNamespace(token); }};
    visitor.visit(targetList);
}

tl::expected<pylir::Token, std::string> pylir::Parser::expect(pylir::TokenType tokenType)
{
    if (m_current == m_lexer.end())
    {
        return tl::unexpected{
            createDiagnosticsBuilder(m_document->getText().size(), Diag::EXPECTED_N, tokenType)
                .addLabel(m_document->getText().size(), fmt::format("{}", tokenType), Diag::ERROR_COLOUR)
                .emitError()};
    }
    if (m_current->getTokenType() == TokenType::SyntaxError)
    {
        return tl::unexpected{pylir::get<std::string>(m_current->getValue())};
    }
    else if (m_current->getTokenType() != tokenType)
    {
        return tl::unexpected{
            createDiagnosticsBuilder(*m_current, Diag::EXPECTED_N_INSTEAD_OF_N, tokenType, m_current->getTokenType())
                .addLabel(*m_current, fmt::format("{}", tokenType), Diag::ERROR_COLOUR, Diag::emphasis::strikethrough)
                .emitError()};
    }
    else
    {
        return *m_current++;
    }
}

bool pylir::Parser::lookaheadEquals(tcb::span<const TokenType> tokens)
{
    Lexer::iterator end;
    std::size_t count = 0;
    for (end = m_current; end != m_lexer.end() && count != tokens.size(); end++, count++)
        ;
    if (count != tokens.size())
    {
        return false;
    }
    return std::equal(m_current, end, tokens.begin(),
                      [](const Token& token, TokenType tokenType) { return token.getTokenType() == tokenType; });
}
