#include "Parser.hpp"

#include <pylir/Diagnostics/DiagnosticMessages.hpp>
#include <pylir/Support/Util.hpp>

constexpr auto makeTuple = [](auto&&... args) { return std::make_tuple(std::forward<decltype(args)>(args)...); };

constexpr auto tupleAppend = [](auto&& tuple, auto&&... args)
{
    return std::tuple_cat(std::forward<decltype(tuple)>(tuple), std::make_tuple(std::forward<decltype(args)>(args)...));
};

template <class T>
constexpr auto makeFunc()
{
    return [](auto&&... args) noexcept
    {
        // TODO: change to () only in C++20
        if constexpr (std::is_aggregate_v<T>)
        {
            return T{std::forward<decltype(args)>(args)...};
        }
        else
        {
            return T(std::forward<decltype(args)>(args)...);
        }
    };
}

template <class T>
constexpr auto makeFromTuple()
{
    return [](auto&& tuple) noexcept { return std::apply(makeFunc<T>(), std::forward<decltype(tuple)>(tuple)); };
}

constexpr auto apply = [](auto&& func, auto&& tuple)
{ return std::apply(std::forward<decltype(func)>(func), std::forward<decltype(tuple)>(tuple)); };

tl::expected<pylir::Lexer::iterator, std::string> pylir::Parser::expect(pylir::TokenType tokenType)
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
        return m_current++;
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
            return parseEnclosure()
                .map([](Syntax::Enclosure&& enclosure)
                     { return std::make_unique<Syntax::Enclosure>(std::move(enclosure)); })
                .map(makeFunc<Syntax::Atom>());
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
    /*return expect(TokenType::Dot)
        .and_then([this](Lexer::iterator dot)
                  { return expect(TokenType::Identifier).map(pylir::bind_front(makeTuple, dot)); })
        .map(pylir::bind_front(apply, pylir::bind_front(makeFunc<Syntax::AttributeRef>(), primary)));*/
}

tl::expected<pylir::Syntax::Subscription, std::string>
    pylir::Parser::parseSubscription(std::unique_ptr<Syntax::Primary>&& primary)
{
    return expect(TokenType::OpenSquareBracket)
        .and_then(
            [this](Lexer::iterator openSquareBracket)
            {
                return parseCommaList(pylir::bind_front(&Parser::parseExpression, this))
                    .map(pylir::bind_front(makeTuple, *openSquareBracket));
            })
        .and_then(
            [&](auto&& tuple)
            {
                return expect(TokenType::CloseSquareBracket)
                    .map(&Lexer::iterator::operator*)
                    .map(pylir::bind_front(tupleAppend, std::move(tuple)))
                    .map(
                        [&](auto&& tuple)
                        {
                            return std::apply(
                                [&](auto&&... args) {
                                    return Syntax::Subscription{std::move(primary),
                                                                std::forward<decltype(args)>(args)...};
                                },
                                std::move(tuple));
                        });
            });
}

tl::expected<pylir::Syntax::Primary, std::string> pylir::Parser::parsePrimary()
{
    return tl::unexpected{std::string{}};
}
