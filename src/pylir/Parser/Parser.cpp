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

tl::expected<pylir::Syntax::Slicing, std::string>
    pylir::Parser::parseSlicing(std::unique_ptr<Syntax::Primary>&& primary)
{
    auto squareBracket = expect(TokenType::OpenSquareBracket);
    if (!squareBracket)
    {
        return tl::unexpected{std::move(squareBracket).error()};
    }
    auto list = parseCommaList(
        [&]() -> tl::expected<std::variant<Syntax::Slicing::ProperSlice, Syntax::Expression>, std::string>
        {
            auto first = parseExpression();
            if (!first)
            {
                return tl::unexpected{std::move(first).error()};
            }
            if (m_current == m_lexer.end() || m_current->getTokenType() != TokenType::Colon)
            {
                return *std::move(first);
            }
            auto firstColon = *m_current++;
            std::unique_ptr<Syntax::Expression> upperBound;
            if (m_current != m_lexer.end() && Syntax::firstInExpression(m_current->getTokenType()))
            {
                auto temp = parseExpression();
                if (!temp)
                {
                    return tl::unexpected{std::move(temp).error()};
                }
                upperBound = std::make_unique<Syntax::Expression>(*std::move(temp));
            }
            auto secondColumn = expect(TokenType::Colon);
            if (!secondColumn)
            {
                return tl::unexpected{std::move(secondColumn).error()};
            }
            std::unique_ptr<Syntax::Expression> stride;
            if (m_current != m_lexer.end() && Syntax::firstInExpression(m_current->getTokenType()))
            {
                auto temp = parseExpression();
                if (!temp)
                {
                    return tl::unexpected{std::move(temp).error()};
                }
                stride = std::make_unique<Syntax::Expression>(*std::move(temp));
            }
            return Syntax::Slicing::ProperSlice{std::make_unique<Syntax::Expression>(std::move(*first)),
                                                std::move(firstColon), std::move(upperBound), std::move(*secondColumn),
                                                std::move(stride)};
        },
        &Syntax::firstInExpression);
    if (!list)
    {
        return tl::unexpected{std::move(list).error()};
    }
    auto closeSquareBracket = expect(TokenType::CloseSquareBracket);
    if (!closeSquareBracket)
    {
        return tl::unexpected{std::move(closeSquareBracket).error()};
    }
    return Syntax::Slicing{std::move(primary), *std::move(squareBracket), {}, std::move(*closeSquareBracket)};
}

tl::expected<pylir::Syntax::Call, std::string> pylir::Parser::parseCall(std::unique_ptr<Syntax::Primary>&& primary)
{
    auto openParenth = expect(TokenType::OpenParentheses);
    if (!openParenth)
    {
        return tl::unexpected{std::move(openParenth).error()};
    }
    if (m_current == m_lexer.end() || m_current->getTokenType() == TokenType::CloseParentheses)
    {
        auto closeParenth = expect(TokenType::CloseParentheses);
        if (!closeParenth)
        {
            return tl::unexpected{std::move(closeParenth).error()};
        }
        return Syntax::Call{std::move(primary), std::move(*openParenth), std::monostate{}, std::move(*closeParenth)};
    }
    // If it's a star, power of or an "identifier =", it's definitely an argument list, not a comprehension
    std::optional<Syntax::AssignmentExpression> firstAssignment;
    if (m_current->getTokenType() != TokenType::Star && m_current->getTokenType() != TokenType::PowerOf
        && !lookaheadEquals(std::array{TokenType::Identifier, TokenType::Assignment}))
    {
        // Otherwise parse an Assignment expression
        auto assignment = parseAssignmentExpression();
        if (!assignment)
        {
            return tl::unexpected{std::move(assignment).error()};
        }
        if (m_current != m_lexer.end() && Syntax::firstInCompFor(m_current->getTokenType()))
        {
            // We are in a comprehension!
            auto comprehension = parseComprehension(std::move(*assignment));
            if (!comprehension)
            {
                return tl::unexpected{std::move(comprehension).error()};
            }
            auto closeParenth = expect(TokenType::CloseParentheses);
            if (!closeParenth)
            {
                return tl::unexpected{std::move(closeParenth).error()};
            }
            return Syntax::Call{std::move(primary), std::move(*openParenth),
                                std::make_unique<Syntax::Comprehension>(std::move(*comprehension)),
                                std::move(*closeParenth)};
        }
        firstAssignment = std::move(*assignment);
    }

    std::optional<Token> trailingComma;

    std::optional<Token> firstComma;

    // If we had an assignment expression or the first token is a Star we are parsing positional arguments
    std::optional<Syntax::Call::PositionalArguments> positionalArguments;
    if (firstAssignment || m_current->getTokenType() == TokenType::Star)
    {
        Syntax::Call::PositionalItem firstItem;
        if (firstAssignment)
        {
            firstItem.variant = std::make_unique<Syntax::AssignmentExpression>(std::move(*firstAssignment));
        }
        else
        {
            PYLIR_ASSERT(m_current->getTokenType() == TokenType::Star);
            auto star = *m_current++;
            auto expression = parseExpression();
            if (!expression)
            {
                return tl::unexpected{std::move(expression).error()};
            }
            firstItem.variant = Syntax::Call::PositionalItem::Star{
                std::move(star), std::make_unique<Syntax::Expression>(std::move(*expression))};
        }
        std::vector<std::pair<Token, Syntax::Call::PositionalItem>> rest;
        while (m_current != m_lexer.end() && m_current->getTokenType() == TokenType::Comma)
        {
            auto comma = *m_current++;
            // If after the comma is **, then are going to parse keyword arguments
            // If after the comma is "identifier =" then we are going to parse starred_and_keywords
            // In both cases it's the end of the positional arguments and the firstComma has been consumed already
            if ((m_current != m_lexer.end() && m_current->getTokenType() == TokenType::PowerOf)
                || lookaheadEquals(std::array{TokenType::Identifier, TokenType::Assignment}))
            {
                firstComma = std::move(comma);
                break;
            }

            if (m_current != m_lexer.end() && m_current->getTokenType() == TokenType::Star)
            {
                auto star = *m_current++;
                auto expression = parseExpression();
                if (!expression)
                {
                    return tl::unexpected{std::move(expression).error()};
                }
                rest.emplace_back(std::move(comma),
                                  Syntax::Call::PositionalItem{Syntax::Call::PositionalItem::Star{
                                      std::move(star), std::make_unique<Syntax::Expression>(std::move(*expression))}});
                continue;
            }

            if (m_current != m_lexer.end() && m_current->getTokenType() == TokenType::CloseParentheses)
            {
                // We are the end of the call expression and there was a trailing comma
                trailingComma = std::move(comma);
                break;
            }

            auto assignment = parseAssignmentExpression();
            if (!assignment)
            {
                return tl::unexpected{std::move(assignment).error()};
            }
            rest.emplace_back(
                std::move(comma),
                Syntax::Call::PositionalItem{std::make_unique<Syntax::AssignmentExpression>(std::move(*assignment))});
        }
        positionalArguments = Syntax::Call::PositionalArguments{std::move(firstItem), std::move(rest)};
    }

    std::optional<Token> secondComma;

    // We wouldn't have left the positional arguments if it was a star, as positional arguments accept those too
    // so we only land here if "identifier =" is spotted
    std::optional<Syntax::Call::StarredAndKeywords> starredAndKeywords;
    if (lookaheadEquals(std::array{TokenType::Identifier, TokenType::Assignment}))
    {
        std::optional<Syntax::Call::KeywordItem> item;
        {
            auto identifier = *m_current++;
            auto assignment = *m_current++;
            auto expression = parseExpression();
            if (!expression)
            {
                return tl::unexpected{std::move(expression).error()};
            }
            item = Syntax::Call::KeywordItem{std::move(identifier), std::move(assignment),
                                             std::make_unique<Syntax::Expression>(std::move(*expression))};
        }
        std::vector<std::pair<Token, Syntax::Call::StarredAndKeywords::Variant>> rest;
        while (m_current != m_lexer.end() && m_current->getTokenType() == TokenType::Comma)
        {
            auto comma = *m_current++;
            // If there's a ** then keyword arguments have started and we need to bail
            if (m_current != m_lexer.end() && m_current->getTokenType() == TokenType::PowerOf)
            {
                if (!firstComma)
                {
                    firstComma = std::move(comma);
                }
                else
                {
                    secondComma = std::move(comma);
                }
                break;
            }

            if (m_current != m_lexer.end() && m_current->getTokenType() == TokenType::CloseParentheses)
            {
                // We are the end of the call expression and there was a trailing comma
                trailingComma = std::move(comma);
                break;
            }

            if (m_current != m_lexer.end() && m_current->getTokenType() == TokenType::Star)
            {
                auto star = *m_current++;
                auto expression = parseExpression();
                if (!expression)
                {
                    return tl::unexpected{std::move(expression).error()};
                }
                rest.emplace_back(std::move(comma),
                                  Syntax::Call::StarredAndKeywords::Expression{
                                      std::move(star), std::make_unique<Syntax::Expression>(std::move(*expression))});
                continue;
            }

            auto identifier = expect(TokenType::Identifier);
            if (!identifier)
            {
                return tl::unexpected{std::move(identifier).error()};
            }
            auto assignment = expect(TokenType::Assignment);
            if (!assignment)
            {
                return tl::unexpected{std::move(assignment).error()};
            }
            auto expression = parseExpression();
            if (!expression)
            {
                return tl::unexpected{std::move(expression).error()};
            }
            rest.emplace_back(std::move(comma),
                              Syntax::Call::KeywordItem{std::move(*identifier), std::move(*assignment),
                                                        std::make_unique<Syntax::Expression>(std::move(*expression))});
        }
        starredAndKeywords = Syntax::Call::StarredAndKeywords{std::move(*item), std::move(rest)};
    }

    std::optional<Syntax::Call::KeywordArguments> keywordArguments;
    if (m_current != m_lexer.end() && m_current->getTokenType() == TokenType::PowerOf)
    {
        std::optional<Syntax::Call::KeywordArguments::Expression> item;
        {
            auto stars = *m_current++;
            auto expression = parseExpression();
            if (!expression)
            {
                return tl::unexpected{std::move(expression).error()};
            }
            item = Syntax::Call::KeywordArguments::Expression{
                std::move(stars), std::make_unique<Syntax::Expression>(std::move(*expression))};
        }
        std::vector<std::pair<Token, Syntax::Call::KeywordArguments::Variant>> rest;
        while (m_current != m_lexer.end() && m_current->getTokenType() == TokenType::Comma)
        {
            auto comma = *m_current++;
            if (m_current != m_lexer.end() && m_current->getTokenType() == TokenType::CloseParentheses)
            {
                // We are the end of the call expression and there was a trailing comma
                trailingComma = std::move(comma);
                break;
            }

            if (m_current != m_lexer.end() && m_current->getTokenType() == TokenType::PowerOf)
            {
                auto star = *m_current++;
                auto expression = parseExpression();
                if (!expression)
                {
                    return tl::unexpected{std::move(expression).error()};
                }
                rest.emplace_back(std::move(comma),
                                  Syntax::Call::KeywordArguments::Expression{
                                      std::move(star), std::make_unique<Syntax::Expression>(std::move(*expression))});
                continue;
            }

            auto identifier = expect(TokenType::Identifier);
            if (!identifier)
            {
                return tl::unexpected{std::move(identifier).error()};
            }
            auto assignment = expect(TokenType::Assignment);
            if (!assignment)
            {
                return tl::unexpected{std::move(assignment).error()};
            }
            auto expression = parseExpression();
            if (!expression)
            {
                return tl::unexpected{std::move(expression).error()};
            }
            rest.emplace_back(std::move(comma),
                              Syntax::Call::KeywordItem{std::move(*identifier), std::move(*assignment),
                                                        std::make_unique<Syntax::Expression>(std::move(*expression))});
        }
        keywordArguments = Syntax::Call::KeywordArguments{std::move(*item), std::move(rest)};
    }

    auto closeParenth = expect(TokenType::CloseParentheses);
    if (!closeParenth)
    {
        return tl::unexpected{std::move(closeParenth).error()};
    }
    return Syntax::Call{std::move(primary), std::move(*openParenth),
                        std::pair{Syntax::Call::ArgumentList{std::move(positionalArguments), std::move(firstComma),
                                                             std::move(starredAndKeywords), std::move(secondComma),
                                                             std::move(keywordArguments)},
                                  std::move(trailingComma)},
                        std::move(*closeParenth)};
}

tl::expected<pylir::Syntax::Primary, std::string> pylir::Parser::parsePrimary()
{
    // First must always be an atom, all others are left recursive
    auto atom = parseAtom();
    if (!atom)
    {
        return tl::unexpected{std::move(atom).error()};
    }
    Syntax::Primary current{std::move(*atom)};
    if (m_current == m_lexer.end())
    {
        return {std::move(current)};
    }
    while (m_current != m_lexer.end()
           && (m_current->getTokenType() == TokenType::Dot || m_current->getTokenType() == TokenType::OpenParentheses
               || m_current->getTokenType() == TokenType::OpenSquareBracket))
    {
        switch (m_current->getTokenType())
        {
            case TokenType::Dot:
            {
                auto attributeRef = parseAttributeRef(std::make_unique<Syntax::Primary>(std::move(current)));
                if (!attributeRef)
                {
                    return tl::unexpected{std::move(attributeRef).error()};
                }
                current = Syntax::Primary{std::move(*attributeRef)};
                break;
            }
            case TokenType::OpenSquareBracket:
            {
                // The grammar for a slice list that has 0 proper slices, and the one for subscription are identical
                auto slice = parseSlicing(std::make_unique<Syntax::Primary>(std::move(current)));
                if (!slice)
                {
                    return tl::unexpected{std::move(slice).error()};
                }
                // If there are no proper slices, then it's a Subscript and we'll convert
                if (std::holds_alternative<Syntax::Expression>(*slice->sliceList.firstExpr)
                    && std::all_of(slice->sliceList.remainingExpr.begin(), slice->sliceList.remainingExpr.end(),
                                   [](auto&& variant)
                                   { return std::holds_alternative<Syntax::Expression>(*variant.second); }))
                {
                    Syntax::ExpressionList expressionList;
                    expressionList.firstExpr = std::make_unique<Syntax::Expression>(
                        std::move(std::get<Syntax::Expression>(*slice->sliceList.firstExpr)));
                    expressionList.trailingComma = std::move(slice->sliceList.trailingComma);

                    expressionList.remainingExpr.reserve(slice->sliceList.remainingExpr.size());
                    std::transform(std::move_iterator(slice->sliceList.remainingExpr.begin()),
                                   std::move_iterator(slice->sliceList.remainingExpr.end()),
                                   std::back_inserter(expressionList.remainingExpr),
                                   [](auto&& pair)
                                   {
                                       return std::pair{std::move(pair.first),
                                                        std::make_unique<Syntax::Expression>(
                                                            std::move(std::get<Syntax::Expression>(*pair.second)))};
                                   });

                    current = Syntax::Primary{
                        Syntax::Subscription{std::move(slice->primary), std::move(slice->openSquareBracket),
                                             std::move(expressionList), std::move(slice->closeSquareBracket)}};
                }
                else
                {
                    current = Syntax::Primary{std::move(*slice)};
                }
                break;
            }
            case TokenType::OpenParentheses:
            {
                auto call = parseCall(std::make_unique<Syntax::Primary>(std::move(current)));
                if (!call)
                {
                    return tl::unexpected{std::move(call).error()};
                }
                current = Syntax::Primary{std::move(*call)};
                break;
            }
            default: PYLIR_UNREACHABLE;
        }
    }
    return {std::move(current)};
}

tl::expected<pylir::Syntax::CommaList<pylir::Syntax::Expression>, std::string> pylir::Parser::parseExpressionList()
{
    return parseCommaList(pylir::bind_front(&Parser::parseExpression, this), &Syntax::firstInExpression);
}

tl::expected<pylir::Syntax::AssignmentExpression, std::string> pylir::Parser::parseAssignmentExpression()
{
    std::optional<std::pair<Token, Token>> prefix;
    if (m_current != m_lexer.end() && std::next(m_current) != m_lexer.end()
        && std::equal(m_current, std::next(m_current, 2), std::array{TokenType::Identifier, TokenType::Walrus}.begin(),
                      [](const Token& token, TokenType rhs) { return token.getTokenType() == rhs; }))
    {
        prefix.emplace(*m_current, *std::next(m_current));
        std::advance(m_current, 2);
    }
    auto expression = parseExpression();
    if (!expression)
    {
        return tl::unexpected{std::move(expression).error()};
    }
    return Syntax::AssignmentExpression{std::move(prefix),
                                        std::make_unique<Syntax::Expression>(std::move(*expression))};
}
