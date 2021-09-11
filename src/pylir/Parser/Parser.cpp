#include "Parser.hpp"

#include <pylir/Diagnostics/DiagnosticMessages.hpp>

#include "Visitor.hpp"

void pylir::Parser::addToLocals(const pylir::Token& token)
{
    PYLIR_ASSERT(token.getTokenType() == TokenType::Identifier);
    if (m_inClass || m_namespace.empty() || m_namespace.back().freeVariables.count(IdentifierToken{token}))
    {
        return;
    }
    m_namespace.back().locals.insert(IdentifierToken{token});
}

void pylir::Parser::addToLocals(const Syntax::TargetList& targetList)
{
    if (m_inClass || m_namespace.empty())
    {
        return;
    }
    class TargetVisitor : public Syntax::Visitor<TargetVisitor>
    {
    public:
        std::function<void(const pylir::IdentifierToken&)> callback;

        using Visitor::visit;

        void visit(const Syntax::Target& target)
        {
            if (auto* identifier = std::get_if<IdentifierToken>(&target.variant))
            {
                callback(*identifier);
            }
            Visitor::visit(target);
        }
    } visitor{{},
              [&](const IdentifierToken& token)
              {
                  if (m_namespace.back().freeVariables.count(token))
                  {
                      return;
                  }
                  m_namespace.back().locals.insert(token);
              }};
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
    else if (m_current->getTokenType() == TokenType::SyntaxError)
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
