#include "Parser.hpp"

#include <pylir/Diagnostics/DiagnosticMessages.hpp>
#include <pylir/Support/Functional.hpp>

tl::expected<pylir::Syntax::Target, std::string> pylir::Parser::parseTarget()
{
    if (m_current == m_lexer.end())
    {
        return tl::unexpected{
            createDiagnosticsBuilder(m_document->getText().size(), Diag::EXPECTED_N, TokenType::Identifier)
                .addLabel(m_document->getText().size(), std::nullopt, Diag::ERROR_COLOUR)
                .emitError()};
    }
    switch (m_current->getTokenType())
    {
        case TokenType::Identifier:
        {
            auto augTarget = parseAugTarget();
            if (!augTarget)
            {
                return tl::unexpected{std::move(augTarget).error()};
            }
            return pylir::match(
                std::move(augTarget->variant), [](const auto&) -> Syntax::Target { PYLIR_UNREACHABLE; },
                [](Syntax::AttributeRef&& attributeRef) -> Syntax::Target { return {std::move(attributeRef)}; },
                [](Syntax::Subscription&& subscription) -> Syntax::Target { return {std::move(subscription)}; },
                [](Syntax::Slicing&& slicing) -> Syntax::Target { return {std::move(slicing)}; },
                [](IdentifierToken&& identifier) -> Syntax::Target { return {std::move(identifier)}; });
        }
        case TokenType::Star:
        {
            auto star = *m_current++;
            auto target = parseTarget();
            if (!target)
            {
                return tl::unexpected{std::move(target).error()};
            }
            return Syntax::Target{std::pair{star, std::make_unique<Syntax::Target>(std::move(*target))}};
        }
        case TokenType::OpenParentheses:
        {
            auto open = *m_current++;
            if (m_current == m_lexer.end() || m_current->getTokenType() == TokenType::CloseParentheses)
            {
                auto close = expect(TokenType::CloseParentheses);
                if (!close)
                {
                    return tl::unexpected{std::move(close).error()};
                }
                return Syntax::Target{Syntax::Target::Parenth{open, std::nullopt, *close}};
            }
            auto list = parseTargetList();
            if (!list)
            {
                return tl::unexpected{std::move(list).error()};
            }
            auto close = expect(TokenType::CloseParentheses);
            if (!close)
            {
                return tl::unexpected{std::move(close).error()};
            }
            return Syntax::Target{Syntax::Target::Parenth{open, std::move(*list), *close}};
        }
        case TokenType::OpenSquareBracket:
        {
            auto open = *m_current++;
            if (m_current == m_lexer.end() || m_current->getTokenType() == TokenType::CloseSquareBracket)
            {
                auto close = expect(TokenType::CloseSquareBracket);
                if (!close)
                {
                    return tl::unexpected{std::move(close).error()};
                }
                return Syntax::Target{Syntax::Target::Square{open, std::nullopt, *close}};
            }
            auto list = parseTargetList();
            if (!list)
            {
                return tl::unexpected{std::move(list).error()};
            }
            auto close = expect(TokenType::CloseSquareBracket);
            if (!close)
            {
                return tl::unexpected{std::move(close).error()};
            }
            return Syntax::Target{Syntax::Target::Square{open, std::move(*list), *close}};
        }
        case TokenType::SyntaxError: return tl::unexpected{pylir::get<std::string>(m_current->getValue())};
        default:
        {
            return tl::unexpected{createDiagnosticsBuilder(*m_current, Diag::EXPECTED_N_INSTEAD_OF_N,
                                                           TokenType::Identifier, m_current->getTokenType())
                                      .addLabel(*m_current, std::nullopt, Diag::ERROR_COLOUR)
                                      .emitError()};
        }
    }
}

tl::expected<pylir::Syntax::TargetList, std::string>
    pylir::Parser::parseTargetList(std::optional<Syntax::Target>&& firstItem)
{
    return parseCommaList(pylir::bind_front(&Parser::parseTarget, this), Syntax::firstInTarget, std::move(firstItem));
}

tl::expected<pylir::Syntax::AssignmentStmt, std::string>
    pylir::Parser::parseAssignmentStmt(std::optional<Syntax::Target>&& firstItem)
{
    std::vector<std::pair<Syntax::TargetList, BaseToken>> targets;
    do
    {
        auto targetList = parseTargetList(std::move(firstItem));
        if (!targetList)
        {
            return tl::unexpected{std::move(targetList).error()};
        }
        firstItem.reset();
        auto assignment = expect(TokenType::Assignment);
        if (!assignment)
        {
            return tl::unexpected{std::move(assignment).error()};
        }
        targets.emplace_back(std::move(*targetList), std::move(*assignment));
    } while (m_current != m_lexer.end() && Syntax::firstInTarget(m_current->getTokenType()));
    if (m_current != m_lexer.end() && m_current->getTokenType() == TokenType::YieldKeyword)
    {
        auto yieldExpr = parseYieldExpression();
        if (!yieldExpr)
        {
            return tl::unexpected{std::move(yieldExpr).error()};
        }
        return Syntax::AssignmentStmt{std::move(targets), std::move(*yieldExpr)};
    }

    auto starredExpression = parseStarredExpression();
    if (!starredExpression)
    {
        return tl::unexpected{std::move(starredExpression).error()};
    }
    return Syntax::AssignmentStmt{std::move(targets), std::move(*starredExpression)};
}

tl::expected<pylir::Syntax::AugTarget, std::string> pylir::Parser::parseAugTarget()
{
    auto identifier = expect(TokenType::Identifier);
    if (!identifier)
    {
        return tl::unexpected{std::move(identifier).error()};
    }
    if (m_current == m_lexer.end()
        || (m_current->getTokenType() != TokenType::Dot && m_current->getTokenType() != TokenType::OpenSquareBracket))
    {
        return Syntax::AugTarget{IdentifierToken{std::move(*identifier)}};
    }
    Syntax::Primary current{Syntax::Atom{IdentifierToken{std::move(*identifier)}}};
    do
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
                auto newCurrent = parseSlicingOrSubscription(std::make_unique<Syntax::Primary>(std::move(current)));
                if (!newCurrent)
                {
                    return tl::unexpected{std::move(newCurrent).error()};
                }
                current = std::move(*newCurrent);
                break;
            }
            default: PYLIR_UNREACHABLE;
        }
    } while (
        m_current != m_lexer.end()
        && (m_current->getTokenType() == TokenType::Dot || m_current->getTokenType() == TokenType::OpenSquareBracket));
    return pylir::match(
        std::move(current.variant), [](const auto&) -> Syntax::AugTarget { PYLIR_UNREACHABLE; },
        [](Syntax::AttributeRef&& attributeRef) -> Syntax::AugTarget { return {std::move(attributeRef)}; },
        [](Syntax::Subscription&& subscription) -> Syntax::AugTarget { return {std::move(subscription)}; },
        [](Syntax::Slicing&& slicing) -> Syntax::AugTarget { return {std::move(slicing)}; },
        [](IdentifierToken&& identifier) -> Syntax::AugTarget { return {std::move(identifier)}; });
}

tl::expected<pylir::Syntax::SimpleStmt, std::string> pylir::Parser::parseSimpleSmt()
{
    if (m_current == m_lexer.end())
    {
        return tl::unexpected{createDiagnosticsBuilder(m_document->getText().size(), Diag::EXPECTED_N, "statement")
                                  .addLabel(m_document->getText().size(), std::nullopt, Diag::ERROR_COLOUR)
                                  .emitError()};
    }
    switch (m_current->getTokenType())
    {
        case TokenType::AssertKeyword:
        {
            auto assertStmt = parseAssertStmt();
            if (!assertStmt)
            {
                return tl::unexpected{std::move(assertStmt).error()};
            }
            return Syntax::SimpleStmt{std::move(*assertStmt)};
        }
        case TokenType::PassKeyword: return Syntax::SimpleStmt{Syntax::PassStmt{*m_current++}};
        case TokenType::BreakKeyword: return Syntax::SimpleStmt{Syntax::BreakStmt{*m_current++}};
        case TokenType::ContinueKeyword: return Syntax::SimpleStmt{Syntax::ContinueStmt{*m_current++}};
        case TokenType::DelKeyword:
        {
            auto delKeyword = *m_current++;
            auto targetList = parseTargetList();
            if (!targetList)
            {
                return tl::unexpected{std::move(targetList).error()};
            }
            return Syntax::SimpleStmt{Syntax::DelStmt{delKeyword, std::move(*targetList)}};
        }
        case TokenType::ReturnKeyword:
        {
            auto returnKeyword = *m_current++;
            if (m_current == m_lexer.end() || !Syntax::firstInExpression(m_current->getTokenType()))
            {
                return Syntax::SimpleStmt{Syntax::ReturnStmt{returnKeyword, std::nullopt}};
            }
            auto expressionList = parseExpressionList();
            if (!expressionList)
            {
                return tl::unexpected{std::move(expressionList).error()};
            }
            return Syntax::SimpleStmt{Syntax::ReturnStmt{returnKeyword, std::move(*expressionList)}};
        }
        case TokenType::YieldKeyword:
        {
            auto yieldExpr = parseYieldExpression();
            if (!yieldExpr)
            {
                return tl::unexpected{std::move(yieldExpr).error()};
            }
            return Syntax::SimpleStmt{Syntax::YieldStmt{std::move(*yieldExpr)}};
        }
        case TokenType::RaiseKeyword:
        {
            auto raise = *m_current++;
            if (m_current == m_lexer.end() || !Syntax::firstInExpression(m_current->getTokenType()))
            {
                return Syntax::SimpleStmt{Syntax::RaiseStmt{raise, std::nullopt}};
            }
            auto expression = parseExpression();
            if (!expression)
            {
                return tl::unexpected{std::move(expression).error()};
            }
            if (m_current == m_lexer.end() || m_current->getTokenType() != TokenType::FromKeyword)
            {
                return Syntax::SimpleStmt{Syntax::RaiseStmt{raise, std::pair{std::move(*expression), std::nullopt}}};
            }
            auto from = *m_current++;
            auto source = parseExpression();
            if (!source)
            {
                return tl::unexpected{std::move(source).error()};
            }
            return Syntax::SimpleStmt{
                Syntax::RaiseStmt{raise, std::pair{std::move(*expression), std::pair{from, std::move(*source)}}}};
        }
        case TokenType::GlobalKeyword:
        case TokenType::NonlocalKeyword:
        {
            auto keyword = *m_current++;
            auto identifier = expect(TokenType::Identifier);
            if (!identifier)
            {
                return tl::unexpected{std::move(identifier).error()};
            }
            std::vector<std::pair<BaseToken, IdentifierToken>> rest;
            while (m_current != m_lexer.end() && m_current->getTokenType() == TokenType::Comma)
            {
                auto comma = *m_current++;
                auto another = expect(TokenType::Identifier);
                if (!another)
                {
                    return tl::unexpected{std::move(another).error()};
                }
                rest.emplace_back(comma, std::move(*another));
            }
            if (keyword.getTokenType() == TokenType::NonlocalKeyword)
            {
                return Syntax::SimpleStmt{
                    Syntax::NonLocalStmt{keyword, IdentifierToken{std::move(*identifier)}, std::move(rest)}};
            }
            else
            {
                return Syntax::SimpleStmt{
                    Syntax::GlobalStmt{keyword, IdentifierToken{std::move(*identifier)}, std::move(rest)}};
            }
        }
        case TokenType::SyntaxError: return tl::unexpected{pylir::get<std::string>(m_current->getValue())};
        default:
            // Starred expression is a super set of both `target` and `augtarget`.
            auto starredExpression = parseStarredExpression();
            if (!starredExpression)
            {
                return tl::unexpected{std::move(starredExpression).error()};
            }

            if (m_current == m_lexer.end())
            {
                return Syntax::SimpleStmt{std::move(*starredExpression)};
            }

            switch (m_current->getTokenType())
            {
                case TokenType::Comma: // A target list can have a trailing comma,
                case TokenType::Assignment:
                {
                    // If an assignment follows, check whether the starred expression could be a target list
                }
                case TokenType::PlusAssignment:
                case TokenType::Colon:
                case TokenType::MinusAssignment:
                case TokenType::TimesAssignment:
                case TokenType::AtAssignment:
                case TokenType::DivideAssignment:
                case TokenType::IntDivideAssignment:
                case TokenType::RemainderAssignment:
                case TokenType::PowerOfAssignment:
                case TokenType::ShiftRightAssignment:
                case TokenType::ShiftLeftAssignment:
                case TokenType::BitAndAssignment:
                case TokenType::BitXorAssignment:
                case TokenType::BitOrAssignment:
                {
                    // needs to be an augtarget then
                }
                default: return Syntax::SimpleStmt{std::move(*starredExpression)};
            }
    }
}

tl::expected<pylir::Syntax::AssertStmt, std::string> pylir::Parser::parseAssertStmt()
{
    auto assertKeyword = expect(TokenType::AssertKeyword);
    if (!assertKeyword)
    {
        return tl::unexpected{std::move(assertKeyword).error()};
    }
    auto expression = parseExpression();
    if (!expression)
    {
        return tl::unexpected{std::move(expression).error()};
    }
    if (m_current == m_lexer.end() || m_current->getTokenType() == TokenType::Comma)
    {
        return Syntax::AssertStmt{std::move(*assertKeyword), std::move(*expression), std::nullopt};
    }
    auto comma = *m_current++;
    auto message = parseExpression();
    if (!message)
    {
        return tl::unexpected{std::move(message).error()};
    }
    return Syntax::AssertStmt{std::move(*assertKeyword), std::move(*expression),
                              std::pair{std::move(comma), std::move(*message)}};
}

tl::expected<pylir::Syntax::AugTarget, std::string>
    pylir::Parser::convertToAug(Syntax::StarredExpression&& starredExpression)
{
    struct Visitor
    {
        using Ret = tl::expected<pylir::Syntax::AugTarget, std::string>;

        Ret visit(Syntax::OrTest&& expression) {}

        Ret visit(Syntax::ConditionalExpression&& expression)
        {
            if (expression.suffix)
            {
                // TODO: Error
            }
            return visit(std::move(expression.value));
        }

        Ret visit(Syntax::Expression&& expression)
        {
            return pylir::match(
                std::move(expression.variant),
                [&](Syntax::ConditionalExpression&& conditionalExpression)
                { return visit(std::move(conditionalExpression)); },
                [&](std::unique_ptr<Syntax::LambdaExpression>&& lambdaExpression) -> Ret
                {
                    // TODO: Error
                });
        }

        Ret visit(Syntax::AssignmentExpression&& assignmentExpression)
        {
            if (assignmentExpression.identifierAndWalrus)
            {
                // TODO: Error
            }
            return visit(std::move(*assignmentExpression.expression));
        }

        Ret visit(Syntax::StarredExpression&& starredExpression)
        {
            return pylir::match(
                std::move(starredExpression.variant),
                [&](Syntax::Expression&& expression) { return visit(std::move(expression)); },
                [&](Syntax::StarredExpression::Items&& items) -> Ret
                {
                    if (!items.leading.empty())
                    {
                        // TODO: Error
                    }
                    if (!items.last)
                    {
                        // TODO: Error
                    }
                    return pylir::match(
                        std::move(items.last->variant),
                        [&](Syntax::AssignmentExpression&& assignmentExpression)
                        { return visit(std::move(assignmentExpression)); },
                        [&](std::pair<BaseToken, Syntax::OrExpr>&&) -> Ret
                        {
                            // TODO: Error
                        });
                });
        }
    } visitor;
    return visitor.visit(std::move(starredExpression));
}
