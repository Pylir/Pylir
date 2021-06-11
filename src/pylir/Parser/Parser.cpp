#include "Parser.hpp"

#include <pylir/Diagnostics/DiagnosticMessages.hpp>
#include <pylir/Support/Util.hpp>

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
        return tl::unexpected{std::get<std::string>(m_current->getValue())};
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

tl::expected<pylir::Syntax::Atom, std::string> pylir::Parser::parseAtom()
{
    if (m_current == m_lexer.end())
    {
        return tl::unexpected{
            createDiagnosticsBuilder(m_document->getText().size(), Diag::EXPECTED_N, "identifier, number or enclosure")
                .addLabel(m_document->getText().size(), std::nullopt, Diag::ERROR_COLOUR)
                .emitError()};
    }

    switch (m_current->getTokenType())
    {
        case TokenType::SyntaxError: return tl::unexpected{std::get<std::string>(m_current->getValue())};
        case TokenType::Identifier:
        {
            return Syntax::Atom{Syntax::Atom::Identifier{*m_current}};
        }
        case TokenType::StringLiteral:
        case TokenType::ByteLiteral:
        case TokenType::IntegerLiteral:
        case TokenType::FloatingPointLiteral:
        case TokenType::ComplexLiteral:
        {
            return Syntax::Atom{Syntax::Atom::Literal{*m_current}};
        }
        case TokenType::OpenParentheses:
        case TokenType::OpenBrace:
        case TokenType::OpenSquareBracket:
        {
            auto enclosure = parseEnclosure();
            if (!enclosure)
            {
                return tl::unexpected{std::move(enclosure).error()};
            }
            return Syntax::Atom{std::make_unique<Syntax::Enclosure>(std::move(*enclosure))};
        }
        default:
            return tl::unexpected{
                createDiagnosticsBuilder(*m_current, Diag::EXPECTED_N_INSTEAD_OF_N, "identifier, number or enclosure",
                                         m_current->getTokenType())
                    .addLabel(*m_current, std::nullopt, Diag::ERROR_COLOUR, Diag::emphasis::strikethrough)
                    .emitError()};
    }
}

tl::expected<pylir::Syntax::Enclosure, std::string> pylir::Parser::parseEnclosure()
{
    if (m_current == m_lexer.end())
    {
        return tl::unexpected{createDiagnosticsBuilder(m_document->getText().size(), Diag::EXPECTED_N,
                                                       fmt::format("{:q}, {:q} or {:q}", TokenType::OpenParentheses,
                                                                   TokenType::OpenSquareBracket, TokenType::OpenBrace))
                                  .addLabel(m_document->getText().size(), std::nullopt, Diag::ERROR_COLOUR)
                                  .emitError()};
    }
    switch (m_current->getTokenType())
    {
        case TokenType::OpenParentheses:
        {
            // TODO:
        }
        case TokenType::OpenBrace:
        {
            // TODO:
        }
        case TokenType::OpenSquareBracket:
        {
            // TODO:
        }
        case TokenType::SyntaxError: return tl::unexpected{std::get<std::string>(m_current->getValue())};
        default:
            return tl::unexpected{
                createDiagnosticsBuilder(*m_current, Diag::EXPECTED_N_INSTEAD_OF_N,
                                         fmt::format("{:q}, {:q} or {:q}", TokenType::OpenParentheses,
                                                     TokenType::OpenSquareBracket, TokenType::OpenBrace),
                                         m_current->getTokenType())
                    .addLabel(*m_current, std::nullopt, Diag::ERROR_COLOUR, Diag::emphasis::strikethrough)
                    .emitError()};
    }
}

tl::expected<pylir::Syntax::AttributeRef, std::string>
    pylir::Parser::parseAttributeRef(std::unique_ptr<Syntax::Primary>&& primary)
{
    auto dot = expect(TokenType::Dot);
    if (!dot)
    {
        return tl::unexpected{std::move(dot).error()};
    }
    auto identifier = expect(TokenType::Identifier);
    if (!identifier)
    {
        return tl::unexpected{std::move(identifier).error()};
    }
    return Syntax::AttributeRef{std::move(primary), *std::move(dot), *std::move(identifier)};
}

tl::expected<pylir::Syntax::Subscription, std::string>
    pylir::Parser::parseSubscription(std::unique_ptr<Syntax::Primary>&& primary)
{
    auto squareBracket = expect(TokenType::OpenSquareBracket);
    if (!squareBracket)
    {
        return tl::unexpected{std::move(squareBracket).error()};
    }
    auto list = parseCommaList(pylir::bind_front(&Parser::parseExpression, this));
    if (!list)
    {
        return tl::unexpected{std::move(list).error()};
    }
    auto closeSquareBracket = expect(TokenType::CloseSquareBracket);
    if (!closeSquareBracket)
    {
        return tl::unexpected{std::move(closeSquareBracket).error()};
    }
    return Syntax::Subscription{std::move(primary), *std::move(squareBracket), std::move(*list),
                                std::move(*closeSquareBracket)};
}

tl::expected<pylir::Syntax::Primary, std::string> pylir::Parser::parsePrimary()
{
    return tl::unexpected{std::string{}};
}
