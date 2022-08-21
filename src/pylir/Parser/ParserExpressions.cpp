// Copyright 2022 Markus Böck
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "Parser.hpp"

#include <pylir/Diagnostics/DiagnosticMessages.hpp>
#include <pylir/Support/Functional.hpp>
#include <pylir/Support/Variant.hpp>

tl::expected<pylir::Syntax::Yield, std::string> pylir::Parser::parseYieldExpression()
{
    if (!m_inFunc)
    {
        return tl::unexpected{
            createError(*m_current, Diag::OCCURRENCE_OF_YIELD_OUTSIDE_OF_FUNCTION)
                                  .addLabel(*m_current)
                                  .emit()};
    }
    auto yield = expect(TokenType::YieldKeyword);
    if (!yield)
    {
        return tl::unexpected{std::move(yield).error()};
    }

    if (!peekedIs(TokenType::FromKeyword) && !peekedIs(firstInExpression))
    {
        return Syntax::Yield{{}, std::move(*yield), std::nullopt, nullptr};
    }

    if (auto from = maybeConsume(TokenType::FromKeyword))
    {
        auto expression = parseExpression();
        if (!expression)
        {
            return tl::unexpected{std::move(expression).error()};
        }
        return Syntax::Yield{{}, std::move(*yield), std::move(from), std::move(*expression)};
    }
    auto list = parseExpressionList();
    if (!list)
    {
        return tl::unexpected{std::move(list).error()};
    }
    return Syntax::Yield{{}, std::move(*yield), std::nullopt, std::move(*list)};
}

tl::expected<pylir::IntrVarPtr<pylir::Syntax::Expression>, std::string> pylir::Parser::parseAtom()
{
    if (m_current == m_lexer.end())
    {
        return tl::unexpected{
            createError(m_document->getText().size(), Diag::EXPECTED_N, "identifier, number or enclosure")
                .addLabel(m_document->getText().size())
                .emit()};
    }

    switch (m_current->getTokenType())
    {
        case TokenType::SyntaxError: return tl::unexpected{pylir::get<std::string>((*m_current++).getValue())};
        case TokenType::Identifier:
        {
            auto token = *m_current++;
            if (!m_namespace.empty())
            {
                // emplace will only insert if it is not already contained. So it will only be marked as unknown
                // if we didn't know it's kind already
                m_namespace.back().identifiers.insert({IdentifierToken{token}, Syntax::Scope::Kind::Unknown});
            }
            return make_node<Syntax::Atom>(token);
        }
        case TokenType::StringLiteral:
        case TokenType::ByteLiteral:
        case TokenType::IntegerLiteral:
        case TokenType::FloatingPointLiteral:
        case TokenType::ComplexLiteral:
        case TokenType::TrueKeyword:
        case TokenType::FalseKeyword:
        case TokenType::NoneKeyword: return make_node<Syntax::Atom>(*m_current++);
        case TokenType::OpenParentheses:
        case TokenType::OpenBrace:
        case TokenType::OpenSquareBracket: return parseEnclosure();
        default:
            return tl::unexpected{createError(*m_current, Diag::EXPECTED_N_INSTEAD_OF_N,
                                              "identifier, number or enclosure", m_current->getTokenType())
                                      .addLabel(*m_current, Diag::flags::strikethrough)
                                      .emit()};
    }
}

tl::expected<pylir::IntrVarPtr<pylir::Syntax::Expression>, std::string> pylir::Parser::parseEnclosure()
{
    if (m_current == m_lexer.end())
    {
        return tl::unexpected{createError(m_document->getText().size(), Diag::EXPECTED_N,
                                          fmt::format("{:q}, {:q} or {:q}", TokenType::OpenParentheses,
                                                      TokenType::OpenSquareBracket, TokenType::OpenBrace))
                                  .addLabel(m_document->getText().size())
                                  .emit()};
    }
    switch (m_current->getTokenType())
    {
        case TokenType::OpenParentheses:
        {
            auto openParenth = *m_current++;
            if (m_current == m_lexer.end() || peekedIs(TokenType::CloseParentheses))
            {
                auto closeParentheses = expect(TokenType::CloseParentheses);
                if (!closeParentheses)
                {
                    return tl::unexpected{std::move(closeParentheses).error()};
                }
                return make_node<Syntax::TupleConstruct>(openParenth, std::vector<Syntax::StarredItem>{},
                                                         *closeParentheses);
            }
            if (m_current->getTokenType() == TokenType::YieldKeyword)
            {
                auto yield = parseYieldExpression();
                if (!yield)
                {
                    return tl::unexpected{std::move(yield).error()};
                }
                auto closeParentheses = expect(TokenType::CloseParentheses);
                if (!closeParentheses)
                {
                    return tl::unexpected{std::move(closeParentheses).error()};
                }
                return std::make_unique<Syntax::Yield>(std::move(*yield));
            }

            if (firstInStarredItem(m_current->getTokenType())
                && (!firstInExpression(m_current->getTokenType())
                    || lookaheadEquals(std::array{TokenType::Identifier, TokenType::Walrus})))
            {
                auto starredExpression = parseStarredExpression();
                if (!starredExpression)
                {
                    return tl::unexpected{std::move(starredExpression).error()};
                }
                auto closeParentheses = expect(TokenType::CloseParentheses);
                if (!closeParentheses)
                {
                    return tl::unexpected{std::move(closeParentheses).error()};
                }
                return starredExpression;
            }

            auto expression = parseExpression();
            if (!expression)
            {
                return tl::unexpected{std::move(expression).error()};
            }
            if (!peekedIs(firstInCompFor))
            {
                auto starredExpression = parseStarredExpression(std::move(*expression));
                if (!starredExpression)
                {
                    return tl::unexpected{std::move(starredExpression).error()};
                }
                auto closeParentheses = expect(TokenType::CloseParentheses);
                if (!closeParentheses)
                {
                    return tl::unexpected{std::move(closeParentheses).error()};
                }
                return starredExpression;
            }

            auto compFor = parseCompFor();
            if (!compFor)
            {
                return tl::unexpected{std::move(compFor).error()};
            }
            auto closeParentheses = expect(TokenType::CloseParentheses);
            if (!closeParentheses)
            {
                return tl::unexpected{std::move(closeParentheses).error()};
            }
            return make_node<Syntax::Generator>(std::move(openParenth), std::move(*expression), std::move(*compFor),
                                                std::move(*closeParentheses));
        }
        case TokenType::OpenBrace:
        {
            auto openBrace = *m_current++;
            if (m_current == m_lexer.end() || peekedIs(TokenType::CloseBrace))
            {
                auto closeBrace = expect(TokenType::CloseBrace);
                if (!closeBrace)
                {
                    return tl::unexpected{std::move(closeBrace).error()};
                }
                return make_node<Syntax::DictDisplay>(
                    std::move(openBrace), std::vector<Syntax::DictDisplay::KeyDatum>{}, std::move(*closeBrace));
            }

            if (peekedIs(TokenType::Star) || lookaheadEquals(std::array{TokenType::Identifier, TokenType::Walrus}))
            {
                auto starredList = parseStarredList();
                if (!starredList)
                {
                    return tl::unexpected{std::move(starredList).error()};
                }
                auto closeBrace = expect(TokenType::CloseBrace);
                if (!closeBrace)
                {
                    return tl::unexpected{std::move(closeBrace).error()};
                }
                return make_node<Syntax::SetDisplay>(std::move(openBrace), std::move(*starredList),
                                                     std::move(*closeBrace));
            }

            std::optional<Syntax::DictDisplay::KeyDatum> keyDatum;
            if (m_current->getTokenType() != TokenType::PowerOf)
            {
                auto expression = parseExpression();
                if (!expression)
                {
                    return tl::unexpected{std::move(expression).error()};
                }
                if (!peekedIs(TokenType::Colon))
                {
                    // We are 100% in a Set.
                    if (peekedIs(firstInCompFor))
                    {
                        auto comprehension = parseComprehension(std::move(*expression));
                        if (!comprehension)
                        {
                            return tl::unexpected{std::move(comprehension).error()};
                        }
                        auto closeBrace = expect(TokenType::CloseBrace);
                        if (!closeBrace)
                        {
                            return tl::unexpected{std::move(closeBrace).error()};
                        }
                        return make_node<Syntax::SetDisplay>(std::move(openBrace), std::move(*comprehension),
                                                             std::move(*closeBrace));
                    }
                    auto starredList = parseStarredList(Syntax::StarredItem{std::nullopt, std::move(*expression)});
                    if (!starredList)
                    {
                        return tl::unexpected{std::move(starredList).error()};
                    }
                    auto closeBrace = expect(TokenType::CloseBrace);
                    if (!closeBrace)
                    {
                        return tl::unexpected{std::move(closeBrace).error()};
                    }
                    return make_node<Syntax::SetDisplay>(std::move(openBrace), std::move(*starredList),
                                                         std::move(*closeBrace));
                }
                auto colon = *m_current++;
                auto secondExpression = parseExpression();
                if (!secondExpression)
                {
                    return tl::unexpected{std::move(secondExpression).error()};
                }
                if (peekedIs(firstInCompFor))
                {
                    auto compFor = parseCompFor();
                    if (!compFor)
                    {
                        return tl::unexpected{std::move(compFor).error()};
                    }
                    auto closeBrace = expect(TokenType::CloseBrace);
                    if (!closeBrace)
                    {
                        return tl::unexpected{std::move(closeBrace).error()};
                    }
                    return make_node<Syntax::DictDisplay>(
                        std::move(openBrace),
                        Syntax::DictDisplay::DictComprehension{std::move(*expression), std::move(colon),
                                                               std::move(*secondExpression), std::move(*compFor)},
                        std::move(*closeBrace));
                }
                keyDatum = Syntax::DictDisplay::KeyDatum{std::move(*expression), std::move(colon),
                                                         std::move(*secondExpression)};
            }

            auto keyDatumList = parseCommaList(
                [&]() -> tl::expected<Syntax::DictDisplay::KeyDatum, std::string>
                {
                    if (auto powerOf = maybeConsume(TokenType::PowerOf))
                    {
                        auto orExpr = parseOrExpr();
                        if (!orExpr)
                        {
                            return tl::unexpected{std::move(orExpr).error()};
                        }
                        return Syntax::DictDisplay::KeyDatum{std::move(*orExpr), std::move(*powerOf), nullptr};
                    }
                    auto first = parseExpression();
                    if (!first)
                    {
                        return tl::unexpected{std::move(first).error()};
                    }
                    auto colon = expect(TokenType::Colon);
                    if (!colon)
                    {
                        return tl::unexpected{std::move(colon).error()};
                    }
                    auto second = parseExpression();
                    if (!second)
                    {
                        return tl::unexpected{std::move(second).error()};
                    }
                    return Syntax::DictDisplay::KeyDatum{std::move(*first), std::move(*colon), std::move(*second)};
                },
                [&](TokenType type) { return firstInExpression(type) || type == TokenType::PowerOf; },
                std::move(keyDatum));
            if (!keyDatumList)
            {
                return tl::unexpected{std::move(keyDatumList).error()};
            }
            auto closeBrace = expect(TokenType::CloseBrace);
            if (!closeBrace)
            {
                return tl::unexpected{std::move(closeBrace).error()};
            }
            return make_node<Syntax::DictDisplay>(std::move(openBrace), std::move(*keyDatumList),
                                                  std::move(*closeBrace));
        }
        case TokenType::OpenSquareBracket:
        {
            auto openSquareBracket = *m_current++;
            if (m_current == m_lexer.end() || peekedIs(TokenType::CloseSquareBracket))
            {
                auto closeSquare = expect(TokenType::CloseSquareBracket);
                if (!closeSquare)
                {
                    return tl::unexpected{std::move(closeSquare).error()};
                }
                return make_node<Syntax::ListDisplay>(std::move(openSquareBracket), std::vector<Syntax::StarredItem>{},
                                                      std::move(*closeSquare));
            }
            if (firstInStarredItem(m_current->getTokenType()) && !firstInComprehension(m_current->getTokenType()))
            {
                auto starredList = parseStarredList();
                if (!starredList)
                {
                    return tl::unexpected{std::move(starredList).error()};
                }
                auto closeSquare = expect(TokenType::CloseSquareBracket);
                if (!closeSquare)
                {
                    return tl::unexpected{std::move(closeSquare).error()};
                }
                return make_node<Syntax::ListDisplay>(std::move(openSquareBracket), std::move(*starredList),
                                                      std::move(*closeSquare));
            }

            auto assignment = parseAssignmentExpression();
            if (!assignment)
            {
                return tl::unexpected{std::move(assignment).error()};
            }
            if (!peekedIs(firstInCompFor))
            {
                auto starredList = parseStarredList(Syntax::StarredItem{std::nullopt, std::move(*assignment)});
                if (!starredList)
                {
                    return tl::unexpected{std::move(starredList).error()};
                }
                auto closeSquare = expect(TokenType::CloseSquareBracket);
                if (!closeSquare)
                {
                    return tl::unexpected{std::move(closeSquare).error()};
                }
                return make_node<Syntax::ListDisplay>(std::move(openSquareBracket), std::move(*starredList),
                                                      std::move(*closeSquare));
            }

            auto comprehension = parseComprehension(std::move(*assignment));
            if (!comprehension)
            {
                return tl::unexpected{std::move(comprehension).error()};
            }
            auto closeSquare = expect(TokenType::CloseSquareBracket);
            if (!closeSquare)
            {
                return tl::unexpected{std::move(closeSquare).error()};
            }
            return make_node<Syntax::ListDisplay>(std::move(openSquareBracket), std::move(*comprehension),
                                                  std::move(*closeSquare));
        }
        case TokenType::SyntaxError: return tl::unexpected{pylir::get<std::string>(m_current->getValue())};
        default:
            return tl::unexpected{createError(*m_current, Diag::EXPECTED_N_INSTEAD_OF_N,
                                              fmt::format("{:q}, {:q} or {:q}", TokenType::OpenParentheses,
                                                          TokenType::OpenSquareBracket, TokenType::OpenBrace),
                                              m_current->getTokenType())
                    .addLabel(*m_current, Diag::flags::strikethrough)
                    .emit()};
    }
}

tl::expected<pylir::Syntax::AttributeRef, std::string>
    pylir::Parser::parseAttributeRef(IntrVarPtr<Syntax::Expression>&& expression)
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
    return Syntax::AttributeRef{{}, std::move(expression), *std::move(dot), IdentifierToken{*std::move(identifier)}};
}

tl::expected<pylir::IntrVarPtr<pylir::Syntax::Expression>, std::string>
    pylir::Parser::parseSlicingOrSubscription(IntrVarPtr<Syntax::Expression>&& expression)
{
    auto squareBracket = expect(TokenType::OpenSquareBracket);
    if (!squareBracket)
    {
        return tl::unexpected{std::move(squareBracket).error()};
    }
    auto list = parseCommaList(
        [&]() -> tl::expected<IntrVarPtr<Syntax::Expression>, std::string>
        {
            IntrVarPtr<Syntax::Expression> lowerBound;
            if (m_current->getTokenType() != TokenType::Colon)
            {
                auto first = parseExpression();
                if (!first)
                {
                    return tl::unexpected{std::move(first).error()};
                }
                if (m_current == m_lexer.end() || m_current->getTokenType() != TokenType::Colon)
                {
                    return first;
                }
                lowerBound = std::move(*first);
            }
            auto firstColon = *m_current++;
            IntrVarPtr<Syntax::Expression> upperBound;
            if (peekedIs(firstInExpression))
            {
                auto temp = parseExpression();
                if (!temp)
                {
                    return tl::unexpected{std::move(temp).error()};
                }
                upperBound = std::move(*temp);
            }
            auto secondColumn = expect(TokenType::Colon);
            if (!secondColumn)
            {
                return tl::unexpected{std::move(secondColumn).error()};
            }
            IntrVarPtr<Syntax::Expression> stride;
            if (peekedIs(firstInExpression))
            {
                auto temp = parseExpression();
                if (!temp)
                {
                    return tl::unexpected{std::move(temp).error()};
                }
                stride = std::move(*temp);
            }
            return make_node<Syntax::Slice>(std::move(lowerBound), std::move(firstColon), std::move(upperBound),
                                            std::move(*secondColumn), std::move(stride));
        },
        &firstInExpression);
    if (!list)
    {
        return tl::unexpected{std::move(list).error()};
    }
    auto closeSquareBracket = expect(TokenType::CloseSquareBracket);
    if (!closeSquareBracket)
    {
        return tl::unexpected{std::move(closeSquareBracket).error()};
    }
    if (list->size() != 1)
    {
        std::vector<Syntax::StarredItem> starredItems(list->size());
        std::transform(std::move_iterator(list->begin()), std::move_iterator(list->end()), starredItems.begin(),
                       [](IntrVarPtr<Syntax::Expression>&& expr) {
                           return Syntax::StarredItem{std::nullopt, std::move(expr)};
                       });

        return make_node<Syntax::Subscription>(
            std::move(expression), std::move(*squareBracket),
            make_node<Syntax::TupleConstruct>(std::nullopt, std::move(starredItems), std::nullopt),
            std::move(*closeSquareBracket));
    }
    return make_node<Syntax::Subscription>(std::move(expression), std::move(*squareBracket), std::move(list->front()),
                                           std::move(*closeSquareBracket));
}

tl::expected<std::vector<pylir::Syntax::Argument>, std::string>
    pylir::Parser::parseArgumentList(IntrVarPtr<Syntax::Expression>&& firstAssignment)
{
    std::vector<pylir::Syntax::Argument> arguments;
    if (firstAssignment)
    {
        arguments.push_back({std::nullopt, std::nullopt, std::move(firstAssignment)});
    }
    std::optional<std::size_t> firstKeywordIndex;
    std::optional<std::size_t> firstMappingExpansionIndex;
    while (arguments.empty() || peekedIs(TokenType::Comma))
    {
        if (!arguments.empty())
        {
            // Some productions using argument_list allow a trailing comma afterwards. We can't always allow this and
            // hence need to let the caller handle it. We therefore only consume the comma if the thing afterwards
            // may be parsed as argument as well.
            if (std::next(m_current) == m_lexer.end()
                || (!firstInExpression(std::next(m_current)->getTokenType())
                    && std::next(m_current)->getTokenType() != TokenType::Star
                    && std::next(m_current)->getTokenType() != TokenType::PowerOf))
            {
                break;
            }
            m_current++;
        }
        std::optional<Token> expansionOrEqual;
        std::optional<IdentifierToken> keywordName;
        switch (m_current->getTokenType())
        {
            case TokenType::PowerOf:
            case TokenType::Star: expansionOrEqual = *m_current++; break;
            case TokenType::Identifier:
            {
                if (std::next(m_current) != m_lexer.end()
                    && std::next(m_current)->getTokenType() == TokenType::Assignment)
                {
                    keywordName = IdentifierToken{*m_current++};
                    expansionOrEqual = *m_current++;
                }
                break;
            }
            default: break;
        }
        auto expression = !expansionOrEqual ? parseAssignmentExpression() : parseExpression();
        if (!expression)
        {
            return tl::unexpected{std::move(expression).error()};
        }

        // Remember the indices of both the first keyword argument and the first mapping expansion.
        if (keywordName && !firstKeywordIndex)
        {
            firstKeywordIndex = arguments.size();
        }
        else if (!firstMappingExpansionIndex && expansionOrEqual
                 && expansionOrEqual->getTokenType() == TokenType::PowerOf)
        {
            firstMappingExpansionIndex = arguments.size();
        }

        if (!expansionOrEqual && (firstKeywordIndex || firstMappingExpansionIndex))
        {
            // We diagnose whichever one of the two cases happened first.
            if (!firstMappingExpansionIndex || (firstKeywordIndex && firstKeywordIndex < firstMappingExpansionIndex))
            {
                return tl::unexpected{
                    createError(**expression, Diag::POSITIONAL_ARGUMENT_NOT_ALLOWED_FOLLOWING_KEYWORD_ARGUMENTS)
                        .addLabel(**expression)
                        .addNote(*arguments[*firstKeywordIndex].maybeName, Diag::FIRST_KEYWORD_ARGUMENT_N_HERE,
                                 arguments[*firstKeywordIndex].maybeName->getValue())
                        .addLabel(*arguments[*firstKeywordIndex].maybeName)
                        .emit()};
            }
            return tl::unexpected{
                createError(**expression, Diag::POSITIONAL_ARGUMENT_NOT_ALLOWED_FOLLOWING_DICTIONARY_UNPACKING)
                    .addLabel(**expression)
                    .addNote(arguments[*firstMappingExpansionIndex], Diag::FIRST_DICTIONARY_UNPACKING_HERE)
                    .addLabel(arguments[*firstMappingExpansionIndex])
                    .emit()};
        }

        if (expansionOrEqual && expansionOrEqual->getTokenType() == TokenType::Star && firstMappingExpansionIndex)
        {
            return tl::unexpected{
                createError(**expression, Diag::ITERABLE_UNPACKING_NOT_ALLOWED_FOLLOWING_DICTIONARY_UNPACKING)
                    .addLabel(**expression)
                    .addNote(arguments[*firstMappingExpansionIndex], Diag::FIRST_DICTIONARY_UNPACKING_HERE)
                    .addLabel(arguments[*firstMappingExpansionIndex])
                    .emit()};
        }
        arguments.push_back({std::move(keywordName), std::move(expansionOrEqual), std::move(*expression)});
    }
    return arguments;
}

tl::expected<pylir::Syntax::Call, std::string> pylir::Parser::parseCall(IntrVarPtr<Syntax::Expression>&& expression)
{
    auto openParenth = expect(TokenType::OpenParentheses);
    if (!openParenth)
    {
        return tl::unexpected{std::move(openParenth).error()};
    }
    if (m_current == m_lexer.end() || peekedIs(TokenType::CloseParentheses))
    {
        auto closeParenth = expect(TokenType::CloseParentheses);
        if (!closeParenth)
        {
            return tl::unexpected{std::move(closeParenth).error()};
        }
        return Syntax::Call{{},
                            std::move(expression),
                            std::move(*openParenth),
                            std::vector<Syntax::Argument>{},
                            std::move(*closeParenth)};
    }
    // If it's a star, power of or an "identifier =", it's definitely an argument list, not a comprehension
    IntrVarPtr<Syntax::Expression> firstAssignment;
    if (peekedIsNot({TokenType::Star, TokenType::PowerOf})
        && !lookaheadEquals(std::array{TokenType::Identifier, TokenType::Assignment}))
    {
        // Otherwise parse an Assignment expression
        auto assignment = parseAssignmentExpression();
        if (!assignment)
        {
            return tl::unexpected{std::move(assignment).error()};
        }
        if (peekedIs(firstInCompFor))
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
            return Syntax::Call{{},
                                std::move(expression),
                                std::move(*openParenth),
                                std::move(*comprehension),
                                std::move(*closeParenth)};
        }
        firstAssignment = std::move(*assignment);
    }

    auto argumentList = parseArgumentList(std::move(firstAssignment));
    if (!argumentList)
    {
        return tl::unexpected{std::move(argumentList).error()};
    }
    maybeConsume(TokenType::Comma);

    auto closeParenth = expect(TokenType::CloseParentheses);
    if (!closeParenth)
    {
        return tl::unexpected{std::move(closeParenth).error()};
    }
    return Syntax::Call{
        {}, std::move(expression), std::move(*openParenth), std::move(*argumentList), std::move(*closeParenth)};
}

tl::expected<pylir::IntrVarPtr<pylir::Syntax::Expression>, std::string> pylir::Parser::parsePrimary()
{
    // First must always be an atom, all others are left recursive
    auto atom = parseAtom();
    if (!atom)
    {
        return tl::unexpected{std::move(atom).error()};
    }
    IntrVarPtr<Syntax::Expression> current{std::move(*atom)};
    if (m_current == m_lexer.end())
    {
        return {std::move(current)};
    }
    while (peekedIs({TokenType::Dot, TokenType::OpenParentheses, TokenType::OpenSquareBracket}))
    {
        switch (m_current->getTokenType())
        {
            case TokenType::Dot:
            {
                auto attributeRef = parseAttributeRef(std::move(current));
                if (!attributeRef)
                {
                    return tl::unexpected{std::move(attributeRef).error()};
                }
                current = std::make_unique<Syntax::AttributeRef>(std::move(*attributeRef));
                break;
            }
            case TokenType::OpenSquareBracket:
            {
                auto newCurrent = parseSlicingOrSubscription(std::move(current));
                if (!newCurrent)
                {
                    return tl::unexpected{std::move(newCurrent).error()};
                }
                current = std::move(*newCurrent);
                break;
            }
            case TokenType::OpenParentheses:
            {
                auto call = parseCall(std::move(current));
                if (!call)
                {
                    return tl::unexpected{std::move(call).error()};
                }
                current = std::make_unique<Syntax::Call>(std::move(*call));
                break;
            }
            default: PYLIR_UNREACHABLE;
        }
    }
    return {std::move(current)};
}

tl::expected<pylir::IntrVarPtr<pylir::Syntax::Expression>, std::string> pylir::Parser::parseExpressionList()
{
    std::vector<IntrVarPtr<Syntax::Expression>> expression;
    bool lastWasComma = true;
    do
    {
        auto expr = parseExpression();
        if (!expr)
        {
            return tl::unexpected{std::move(expr).error()};
        }
        expression.push_back(std::move(*expr));
        if (!maybeConsume(TokenType::Comma))
        {
            lastWasComma = false;
            break;
        }
    } while (peekedIs(firstInExpression));
    if (expression.size() == 1 && !lastWasComma)
    {
        return std::move(expression.front());
    }
    std::vector<Syntax::StarredItem> items(expression.size());
    std::transform(std::move_iterator(expression.begin()), std::move_iterator(expression.end()), items.begin(),
                   [](IntrVarPtr<Syntax::Expression>&& expr) {
                       return Syntax::StarredItem{std::nullopt, std::move(expr)};
                   });
    return make_node<Syntax::TupleConstruct>(std::nullopt, std::move(items), std::nullopt);
}

tl::expected<pylir::IntrVarPtr<pylir::Syntax::Expression>, std::string> pylir::Parser::parseAssignmentExpression()
{
    if (!lookaheadEquals(std::array{TokenType::Identifier, TokenType::Walrus}))
    {
        return parseExpression();
    }

    IdentifierToken variable{*m_current++};
    addToNamespace(variable);
    BaseToken walrus = *m_current++;
    auto expression = parseExpression();
    if (!expression)
    {
        return tl::unexpected{std::move(expression).error()};
    }
    return make_node<Syntax::Assignment>(std::move(variable), walrus, std::move(*expression));
}

tl::expected<pylir::Syntax::UnaryOp, std::string> pylir::Parser::parseAwaitExpr()
{
    auto await = expect(TokenType::AwaitKeyword);
    if (!await)
    {
        return tl::unexpected{std::move(await).error()};
    }
    auto primary = parsePrimary();
    if (!primary)
    {
        return tl::unexpected{std::move(primary).error()};
    }
    return Syntax::UnaryOp{{}, std::move(*await), std::move(*primary)};
}

tl::expected<pylir::IntrVarPtr<pylir::Syntax::Expression>, std::string> pylir::Parser::parsePower()
{
    IntrVarPtr<Syntax::Expression> expression;
    if (peekedIs(TokenType::AwaitKeyword))
    {
        auto await = parseAwaitExpr();
        if (!await)
        {
            return tl::unexpected{std::move(await).error()};
        }
        expression = std::make_unique<Syntax::UnaryOp>(std::move(*await));
    }
    else
    {
        auto primary = parsePrimary();
        if (!primary)
        {
            return tl::unexpected{std::move(primary).error()};
        }
        expression = std::move(*primary);
    }
    auto powerOf = maybeConsume(TokenType::PowerOf);
    if (!powerOf)
    {
        return expression;
    }
    auto uExpr = parseUExpr();
    if (!uExpr)
    {
        return tl::unexpected{std::move(uExpr).error()};
    }
    return make_node<Syntax::BinOp>(std::move(expression), std::move(*powerOf), std::move(*uExpr));
}

tl::expected<pylir::IntrVarPtr<pylir::Syntax::Expression>, std::string> pylir::Parser::parseUExpr()
{
    std::vector<Token> unaries;
    while (auto unary = maybeConsume({TokenType::Minus, TokenType::Plus, TokenType::BitNegate}))
    {
        unaries.push_back(std::move(*unary));
    }
    auto power = parsePower();
    if (!power)
    {
        return tl::unexpected{std::move(power).error()};
    }
    IntrVarPtr<Syntax::Expression> current{std::move(*power)};
    std::reverse(unaries.begin(), unaries.end());
    for (Token& token : unaries)
    {
        current = make_node<Syntax::UnaryOp>(std::move(token), std::move(current));
    }
    return current;
}

tl::expected<pylir::IntrVarPtr<pylir::Syntax::Expression>, std::string> pylir::Parser::parseMExpr()
{
    auto first = parseUExpr();
    if (!first)
    {
        return tl::unexpected{std::move(first).error()};
    }
    IntrVarPtr<Syntax::Expression> current{std::move(*first)};
    while (auto op = maybeConsume(
               {TokenType::Star, TokenType::AtSign, TokenType::IntDivide, TokenType::Divide, TokenType::Remainder}))
    {
        if (op->getTokenType() == TokenType::AtSign)
        {
            auto rhs = parseMExpr();
            if (!rhs)
            {
                return tl::unexpected{std::move(rhs).error()};
            }
            current = make_node<Syntax::BinOp>(std::move(current), std::move(*op), std::move(*rhs));
            continue;
        }

        auto rhs = parseUExpr();
        if (!rhs)
        {
            return tl::unexpected{std::move(rhs).error()};
        }
        current = make_node<Syntax::BinOp>(std::move(current), std::move(*op), std::move(*rhs));
    }
    return current;
}

tl::expected<pylir::IntrVarPtr<pylir::Syntax::Expression>, std::string> pylir::Parser::parseAExpr()
{
    return parseGenericBinOp<&Parser::parseMExpr, TokenType::Minus, TokenType::Plus>();
}

tl::expected<pylir::IntrVarPtr<pylir::Syntax::Expression>, std::string> pylir::Parser::parseShiftExpr()
{
    return parseGenericBinOp<&Parser::parseAExpr, TokenType::ShiftLeft, TokenType::ShiftRight>();
}

tl::expected<pylir::IntrVarPtr<pylir::Syntax::Expression>, std::string> pylir::Parser::parseAndExpr()
{
    return parseGenericBinOp<&Parser::parseShiftExpr, TokenType::BitAnd>();
}

tl::expected<pylir::IntrVarPtr<pylir::Syntax::Expression>, std::string> pylir::Parser::parseXorExpr()
{
    return parseGenericBinOp<&Parser::parseAndExpr, TokenType::BitXor>();
}

tl::expected<pylir::IntrVarPtr<pylir::Syntax::Expression>, std::string> pylir::Parser::parseOrExpr()
{
    return parseGenericBinOp<&Parser::parseXorExpr, TokenType::BitOr>();
}

tl::expected<pylir::IntrVarPtr<pylir::Syntax::Expression>, std::string> pylir::Parser::parseComparison()
{
    auto first = parseOrExpr();
    if (!first)
    {
        return tl::unexpected{std::move(first).error()};
    }
    IntrVarPtr<Syntax::Expression> current{std::move(*first)};
    std::vector<std::pair<Syntax::Comparison::Operator, IntrVarPtr<Syntax::Expression>>> rest;
    while (auto op = maybeConsume({TokenType::LessThan, TokenType::LessOrEqual, TokenType::GreaterThan,
                                   TokenType::GreaterOrEqual, TokenType::NotEqual, TokenType::Equal,
                                   TokenType::IsKeyword, TokenType::NotKeyword, TokenType::InKeyword}))
    {
        std::optional<Token> second;
        switch (op->getTokenType())
        {
            case TokenType::IsKeyword: second = maybeConsume(TokenType::NotKeyword); break;
            case TokenType::NotKeyword:
            {
                auto in = expect(TokenType::InKeyword);
                if (!in)
                {
                    return tl::unexpected{std::move(in).error()};
                }
                second = std::move(*in);
                break;
            }
            default: break;
        }
        auto rhs = parseOrExpr();
        if (!rhs)
        {
            return tl::unexpected{std::move(rhs).error()};
        }
        rest.emplace_back(Syntax::Comparison::Operator{std::move(*op), std::move(second)}, std::move(*rhs));
    }
    if (rest.empty())
    {
        return current;
    }
    return make_node<Syntax::Comparison>(std::move(current), std::move(rest));
}

tl::expected<pylir::IntrVarPtr<pylir::Syntax::Expression>, std::string> pylir::Parser::parseNotTest()
{
    auto end = std::find_if_not(m_current, m_lexer.end(),
                                [](const Token& token) { return token.getTokenType() == TokenType::NotKeyword; });
    std::vector<Token> nots(m_current, end);
    m_current = end;
    std::reverse(nots.begin(), nots.end());
    auto comp = parseComparison();
    if (!comp)
    {
        return tl::unexpected{std::move(comp).error()};
    }
    IntrVarPtr<Syntax::Expression> current{std::move(*comp)};
    for (Token& token : nots)
    {
        current = make_node<Syntax::UnaryOp>(std::move(token), std::move(current));
    }
    return {std::move(current)};
}

tl::expected<pylir::IntrVarPtr<pylir::Syntax::Expression>, std::string> pylir::Parser::parseAndTest()
{
    return parseGenericBinOp<&Parser::parseNotTest, TokenType::AndKeyword>();
}

tl::expected<pylir::IntrVarPtr<pylir::Syntax::Expression>, std::string> pylir::Parser::parseOrTest()
{
    return parseGenericBinOp<&Parser::parseAndTest, TokenType::OrKeyword>();
}

tl::expected<pylir::IntrVarPtr<pylir::Syntax::Expression>, std::string> pylir::Parser::parseConditionalExpression()
{
    auto orTest = parseOrTest();
    if (!orTest)
    {
        return tl::unexpected{std::move(orTest).error()};
    }
    auto ifKeyword = maybeConsume(TokenType::IfKeyword);
    if (!ifKeyword)
    {
        return orTest;
    }
    auto condition = parseOrTest();
    if (!condition)
    {
        return tl::unexpected{std::move(condition).error()};
    }
    auto elseKeyword = expect(TokenType::ElseKeyword);
    if (!elseKeyword)
    {
        return tl::unexpected{std::move(elseKeyword).error()};
    }
    auto other = parseExpression();
    if (!other)
    {
        return tl::unexpected{std::move(other).error()};
    }
    return make_node<Syntax::Conditional>(std::move(*orTest), std::move(*ifKeyword), std::move(*condition),
                                          std::move(*elseKeyword), std::move(*other));
}

tl::expected<pylir::IntrVarPtr<pylir::Syntax::Expression>, std::string> pylir::Parser::parseExpression()
{
    if (!peekedIs(TokenType::LambdaKeyword))
    {
        return parseConditionalExpression();
    }

    auto lambda = parseLambdaExpression();
    if (!lambda)
    {
        return tl::unexpected{std::move(lambda).error()};
    }
    return std::make_unique<Syntax::Lambda>(std::move(*lambda));
}

tl::expected<pylir::Syntax::Lambda, std::string> pylir::Parser::parseLambdaExpression()
{
    auto keyword = expect(TokenType::LambdaKeyword);
    if (!keyword)
    {
        return tl::unexpected{std::move(keyword).error()};
    }
    std::vector<Syntax::Parameter> parameterList;
    if (peekedIsNot(TokenType::Colon))
    {
        auto parsedParameterList = parseParameterList();
        if (!parsedParameterList)
        {
            return tl::unexpected{std::move(parsedParameterList).error()};
        }
        parameterList = std::move(*parsedParameterList);
    }
    auto colon = expect(TokenType::Colon);
    if (!colon)
    {
        return tl::unexpected{std::move(colon).error()};
    }
    auto expression = parseExpression();
    if (!expression)
    {
        return tl::unexpected{std::move(expression).error()};
    }
    return Syntax::Lambda{{}, std::move(*keyword), std::move(parameterList), std::move(*colon), std::move(*expression)};
}

tl::expected<pylir::Syntax::Comprehension, std::string>
    pylir::Parser::parseComprehension(IntrVarPtr<Syntax::Expression>&& assignmentExpression)
{
    auto compFor = parseCompFor();
    if (!compFor)
    {
        return tl::unexpected{std::move(compFor).error()};
    }
    return Syntax::Comprehension{std::move(assignmentExpression), std::move(*compFor)};
}

tl::expected<pylir::Syntax::CompFor, std::string> pylir::Parser::parseCompFor()
{
    std::optional<Token> awaitToken = maybeConsume(TokenType::AwaitKeyword);
    auto forToken = expect(TokenType::ForKeyword);
    if (!forToken)
    {
        return tl::unexpected{std::move(forToken).error()};
    }
    auto targetList = parseTargetList(*forToken);
    if (!targetList)
    {
        return tl::unexpected{std::move(targetList).error()};
    }
    addToNamespace(**targetList);
    auto inToken = expect(TokenType::InKeyword);
    if (!inToken)
    {
        return tl::unexpected{std::move(inToken).error()};
    }
    auto orTest = parseOrTest();
    if (!orTest)
    {
        return tl::unexpected{std::move(orTest).error()};
    }
    if (!peekedIs({TokenType::ForKeyword, TokenType::IfKeyword, TokenType::AwaitKeyword}))
    {
        return Syntax::CompFor{std::move(awaitToken), std::move(*forToken), std::move(*targetList),
                               std::move(*inToken),   std::move(*orTest),   std::monostate{}};
    }
    std::variant<std::monostate, std::unique_ptr<Syntax::CompFor>, std::unique_ptr<Syntax::CompIf>> trail;
    if (m_current->getTokenType() == TokenType::IfKeyword)
    {
        auto compIf = parseCompIf();
        if (!compIf)
        {
            return tl::unexpected{std::move(compIf).error()};
        }
        trail = std::make_unique<Syntax::CompIf>(std::move(*compIf));
    }
    else
    {
        auto compFor = parseCompFor();
        if (!compFor)
        {
            return tl::unexpected{std::move(compFor).error()};
        }
        trail = std::make_unique<Syntax::CompFor>(std::move(*compFor));
    }
    return Syntax::CompFor{std::move(awaitToken), std::move(*forToken), std::move(*targetList),
                           std::move(*inToken),   std::move(*orTest),   std::move(trail)};
}

tl::expected<pylir::Syntax::CompIf, std::string> pylir::Parser::parseCompIf()
{
    auto ifToken = expect(TokenType::IfKeyword);
    if (!ifToken)
    {
        return tl::unexpected{std::move(ifToken).error()};
    }
    auto orTest = parseOrTest();
    if (!orTest)
    {
        return tl::unexpected{std::move(orTest).error()};
    }
    if (!peekedIs({TokenType::ForKeyword, TokenType::IfKeyword, TokenType::AwaitKeyword}))
    {
        return Syntax::CompIf{std::move(*ifToken), std::move(*orTest), std::monostate{}};
    }
    std::variant<std::monostate, std::unique_ptr<Syntax::CompFor>, std::unique_ptr<Syntax::CompIf>> trail;
    if (m_current->getTokenType() == TokenType::IfKeyword)
    {
        auto compIf = parseCompIf();
        if (!compIf)
        {
            return tl::unexpected{std::move(compIf).error()};
        }
        trail = std::make_unique<Syntax::CompIf>(std::move(*compIf));
    }
    else
    {
        auto compFor = parseCompFor();
        if (!compFor)
        {
            return tl::unexpected{std::move(compFor).error()};
        }
        trail = std::make_unique<Syntax::CompFor>(std::move(*compFor));
    }
    return Syntax::CompIf{std::move(*ifToken), std::move(*orTest), std::move(trail)};
}

tl::expected<pylir::IntrVarPtr<pylir::Syntax::Expression>, std::string>
    pylir::Parser::parseStarredExpression(IntrVarPtr<Syntax::Expression>&& firstItem)
{
    if (peekedIsNot(TokenType::Star) && !firstItem)
    {
        auto expression = parseAssignmentExpression();
        if (!expression)
        {
            return tl::unexpected{std::move(expression).error()};
        }
        firstItem = std::move(*expression);
    }
    std::vector<Syntax::StarredItem> items;
    if (firstItem)
    {
        if (!maybeConsume(TokenType::Comma))
        {
            return std::move(firstItem);
        }
        items.push_back(Syntax::StarredItem{std::nullopt, std::move(firstItem)});
    }
    while (peekedIs(firstInStarredItem))
    {
        auto item = parseStarredItem();
        if (!item)
        {
            return tl::unexpected{std::move(item).error()};
        }
        // If a comma doesn't follow, then it's the last optional trailing starred_item
        items.emplace_back(std::move(*item));
        if (!maybeConsume(TokenType::Comma))
        {
            // if there were no leading expressions (aka no commas) and it is an expansion (with a star), then it's
            // a syntax error as those are only possible when commas are involved (to form a tuple).
            // TODO: Better error message
            if (items.size() == 1 && item->maybeStar)
            {
                return tl::unexpected{expect(TokenType::Comma).error()};
            }
            break;
        }
    }
    return make_node<Syntax::TupleConstruct>(std::nullopt, std::move(items), std::nullopt);
}

tl::expected<pylir::Syntax::StarredItem, std::string> pylir::Parser::parseStarredItem()
{
    if (auto star = maybeConsume(TokenType::Star))
    {
        auto expression = parseOrExpr();
        if (!expression)
        {
            return tl::unexpected{std::move(expression).error()};
        }
        return Syntax::StarredItem{std::move(star), std::move(*expression)};
    }
    auto assignment = parseAssignmentExpression();
    if (!assignment)
    {
        return tl::unexpected{std::move(assignment).error()};
    }
    return Syntax::StarredItem{std::nullopt, std::move(*assignment)};
}

tl::expected<std::vector<pylir::Syntax::StarredItem>, std::string>
    pylir::Parser::parseStarredList(std::optional<Syntax::StarredItem>&& firstItem)
{
    return parseCommaList(pylir::bind_front(&Parser::parseStarredItem, this), &firstInStarredItem,
                          std::move(firstItem));
}
