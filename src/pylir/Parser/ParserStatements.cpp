// Copyright 2022 Markus Böck
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

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
    pylir::Parser::parseAssignmentStmt(std::optional<Syntax::TargetList>&& firstItem)
{
    std::vector<std::pair<Syntax::TargetList, BaseToken>> targets;
    bool hadFirst = firstItem.has_value();
    if (firstItem)
    {
        auto assignment = expect(TokenType::Assignment);
        if (!assignment)
        {
            return tl::unexpected{std::move(assignment).error()};
        }
        targets.emplace_back(std::move(*firstItem), std::move(*assignment));
    }
    std::optional<Syntax::StarredExpression> leftOverStarredExpression;
    do
    {
        if (hadFirst && (m_current == m_lexer.end() || !Syntax::firstInTarget(m_current->getTokenType())))
        {
            break;
        }
        auto starredExpression = parseStarredExpression();
        if (!starredExpression)
        {
            return tl::unexpected{std::move(starredExpression).error()};
        }
        if (m_current == m_lexer.end() || m_current->getTokenType() != TokenType::Assignment)
        {
            leftOverStarredExpression = std::move(*starredExpression);
            break;
        }
        auto assignment = *m_current++;
        auto targetList = convertToTargetList(std::move(*starredExpression), assignment);
        if (!targetList)
        {
            return tl::unexpected{std::move(targetList).error()};
        }
        addToNamespace(*targetList);
        targets.emplace_back(std::move(*targetList), assignment);
    } while (m_current != m_lexer.end() && Syntax::firstInTarget(m_current->getTokenType()));
    if (leftOverStarredExpression)
    {
        return Syntax::AssignmentStmt{std::move(targets), std::move(*leftOverStarredExpression)};
    }
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

tl::expected<pylir::Syntax::SimpleStmt, std::string> pylir::Parser::parseSimpleStmt()
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
        case TokenType::BreakKeyword:
            if (!m_inLoop)
            {
                return tl::unexpected{createDiagnosticsBuilder(*m_current, Diag::OCCURRENCE_OF_N_OUTSIDE_OF_LOOP,
                                                               m_current->getTokenType())
                                          .addLabel(*m_current, std::nullopt, Diag::ERROR_COLOUR)
                                          .emitError()};
            }
            return Syntax::SimpleStmt{Syntax::BreakStmt{*m_current++}};
        case TokenType::ContinueKeyword:
            if (!m_inLoop)
            {
                return tl::unexpected{createDiagnosticsBuilder(*m_current, Diag::OCCURRENCE_OF_N_OUTSIDE_OF_LOOP,
                                                               m_current->getTokenType())
                                          .addLabel(*m_current, std::nullopt, Diag::ERROR_COLOUR)
                                          .emitError()};
            }
            return Syntax::SimpleStmt{Syntax::ContinueStmt{*m_current++}};
        case TokenType::DelKeyword:
        {
            auto delKeyword = *m_current++;
            auto targetList = parseTargetList();
            if (!targetList)
            {
                return tl::unexpected{std::move(targetList).error()};
            }
            addToNamespace(*targetList);
            return Syntax::SimpleStmt{Syntax::DelStmt{delKeyword, std::move(*targetList)}};
        }
        case TokenType::ReturnKeyword:
        {
            if (!m_inFunc)
            {
                return tl::unexpected{
                    createDiagnosticsBuilder(*m_current, Diag::OCCURRENCE_OF_RETURN_OUTSIDE_OF_FUNCTION)
                        .addLabel(*m_current, std::nullopt, Diag::ERROR_COLOUR)
                        .emitError()};
            }
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
                auto handleToken = [&](const IdentifierToken& nonLocal) -> tl::expected<void, std::string>
                {
                    if (m_namespace.empty())
                    {
                        return {};
                    }
                    if (auto result = m_namespace.back().identifiers.find(nonLocal);
                        result != m_namespace.back().identifiers.end())
                    {
                        switch (result->second)
                        {
                            case Scope::Kind::Local:
                                return tl::unexpected{
                                    createDiagnosticsBuilder(
                                        nonLocal, Diag::DECLARATION_OF_NONLOCAL_N_CONFLICTS_WITH_LOCAL_VARIABLE,
                                        nonLocal.getValue())
                                        .addLabel(nonLocal, std::nullopt, Diag::ERROR_COLOUR)
                                        .addNote(result->first, Diag::LOCAL_VARIABLE_N_BOUND_HERE, nonLocal.getValue())
                                        .addLabel(result->first, std::nullopt, Diag::NOTE_COLOUR)
                                        .emitError()};
                            case Scope::Kind::Global:
                                return tl::unexpected{
                                    createDiagnosticsBuilder(
                                        nonLocal, Diag::DECLARATION_OF_NONLOCAL_N_CONFLICTS_WITH_GLOBAL_VARIABLE,
                                        nonLocal.getValue())
                                        .addLabel(nonLocal, std::nullopt, Diag::ERROR_COLOUR)
                                        .addNote(result->first, Diag::GLOBAL_VARIABLE_N_BOUND_HERE, nonLocal.getValue())
                                        .addLabel(result->first, std::nullopt, Diag::NOTE_COLOUR)
                                        .emitError()};
                            case Scope::Kind::Unknown:
                                return tl::unexpected{
                                    createDiagnosticsBuilder(nonLocal, Diag::NONLOCAL_N_USED_PRIOR_TO_DECLARATION,
                                                             nonLocal.getValue())
                                        .addLabel(nonLocal, std::nullopt, Diag::ERROR_COLOUR)
                                        .addNote(result->first, Diag::N_USED_HERE, nonLocal.getValue())
                                        .addLabel(result->first, std::nullopt, Diag::NOTE_COLOUR)
                                        .emitError()};
                            case Scope::Kind::NonLocal: break;
                        }
                    }
                    return {};
                };
                if (auto error = handleToken(IdentifierToken{*identifier}); !error)
                {
                    return tl::unexpected{error.error()};
                }
                if (!m_namespace.empty())
                {
                    m_namespace.back().identifiers.emplace(IdentifierToken{*identifier}, Scope::Kind::NonLocal);
                }
                for (auto& iter : rest)
                {
                    if (auto error = handleToken(iter.second); !error)
                    {
                        return tl::unexpected{error.error()};
                    }
                    if (!m_namespace.empty())
                    {
                        m_namespace.back().identifiers.emplace(iter.second, Scope::Kind::NonLocal);
                    }
                }
                return Syntax::SimpleStmt{
                    Syntax::NonLocalStmt{keyword, IdentifierToken{std::move(*identifier)}, std::move(rest)}};
            }

            if (!m_namespace.empty())
            {
                auto handleToken = [&](const IdentifierToken& global) -> tl::expected<void, std::string>
                {
                    if (auto result = m_namespace.back().identifiers.find(global);
                        result != m_namespace.back().identifiers.end())
                    {
                        switch (result->second)
                        {
                            case Scope::Kind::Local:
                                return tl::unexpected{
                                    createDiagnosticsBuilder(
                                        global, Diag::DECLARATION_OF_GLOBAL_N_CONFLICTS_WITH_LOCAL_VARIABLE,
                                        global.getValue())
                                        .addLabel(global, std::nullopt, Diag::ERROR_COLOUR)
                                        .addNote(result->first, Diag::LOCAL_VARIABLE_N_BOUND_HERE, global.getValue())
                                        .addLabel(result->first, std::nullopt, Diag::NOTE_COLOUR)
                                        .emitError()};
                            case Scope::Kind::NonLocal:
                                return tl::unexpected{
                                    createDiagnosticsBuilder(
                                        global, Diag::DECLARATION_OF_GLOBAL_N_CONFLICTS_WITH_NONLOCAL_VARIABLE,
                                        global.getValue())
                                        .addLabel(global, std::nullopt, Diag::ERROR_COLOUR)
                                        .addNote(result->first, Diag::NONLOCAL_VARIABLE_N_BOUND_HERE, global.getValue())
                                        .addLabel(result->first, std::nullopt, Diag::NOTE_COLOUR)
                                        .emitError()};
                            case Scope::Kind::Unknown:
                                return tl::unexpected{createDiagnosticsBuilder(global,
                                                                               Diag::GLOBAL_N_USED_PRIOR_TO_DECLARATION,
                                                                               global.getValue())
                                                          .addLabel(global, std::nullopt, Diag::ERROR_COLOUR)
                                                          .addNote(result->first, Diag::N_USED_HERE, global.getValue())
                                                          .addLabel(result->first, std::nullopt, Diag::NOTE_COLOUR)
                                                          .emitError()};
                            case Scope::Kind::Global: break;
                        }
                    }
                    return {};
                };
                if (auto error = handleToken(IdentifierToken{*identifier}); !error)
                {
                    return tl::unexpected{std::move(error).error()};
                }
                m_globals.insert(IdentifierToken{*identifier});
                m_namespace.back().identifiers.emplace(IdentifierToken{*identifier}, Scope::Kind::Global);
                for (auto& iter : rest)
                {
                    if (auto error = handleToken(iter.second); !error)
                    {
                        return tl::unexpected{std::move(error).error()};
                    }
                    m_globals.insert(iter.second);
                    m_namespace.back().identifiers.emplace(iter.second, Scope::Kind::Global);
                }
            }
            else
            {
                m_globals.insert(IdentifierToken{*identifier});
                for (auto& iter : rest)
                {
                    m_globals.insert(iter.second);
                }
            }
            return Syntax::SimpleStmt{
                Syntax::GlobalStmt{keyword, IdentifierToken{std::move(*identifier)}, std::move(rest)}};
        }
        case TokenType::FromKeyword:
        case TokenType::ImportKeyword:
        {
            auto import = parseImportStmt();
            if (!import)
            {
                return tl::unexpected{std::move(import).error()};
            }
            if (auto* fromImportAs = std::get_if<Syntax::ImportStmt::FromImportList>(&import->variant);
                fromImportAs && fromImportAs->relativeModule.dots.empty() && fromImportAs->relativeModule.module
                && fromImportAs->relativeModule.module->leading.empty()
                && fromImportAs->relativeModule.module->lastIdentifier.getValue() == "__future__")
            {
                auto check = [&](const IdentifierToken& identifierToken) -> tl::expected<void, std::string>
                {
#define HANDLE_FEATURE(x)                 \
    if (identifierToken.getValue() == #x) \
    {                                     \
        return {};                        \
    }
#define HANDLE_REQUIRED_FEATURE(x)        \
    if (identifierToken.getValue() == #x) \
    {                                     \
        m_##x = true;                     \
        return {};                        \
    }
#include "Features.def"
                    return tl::unexpected{
                        createDiagnosticsBuilder(identifierToken, Diag::UNKNOWN_FEATURE_N, identifierToken.getValue())
                            .addLabel(identifierToken, std::nullopt, Diag::ERROR_COLOUR)
                            .emitError()};
                };
                if (auto result = check(fromImportAs->identifier); !result)
                {
                    return tl::unexpected{std::move(result).error()};
                }
                for (auto& iter : fromImportAs->rest)
                {
                    if (auto result = check(iter.identifier); !result)
                    {
                        return tl::unexpected{std::move(result).error()};
                    }
                }
                return Syntax::SimpleStmt{Syntax::FutureStmt{
                    fromImportAs->from,
                    fromImportAs->relativeModule.module->lastIdentifier,
                    fromImportAs->import,
                    fromImportAs->openParenth,
                    std::move(fromImportAs->identifier),
                    std::move(fromImportAs->name),
                    std::move(fromImportAs->rest),
                    fromImportAs->comma,
                    fromImportAs->closeParenth,
                }};
            }
            return Syntax::SimpleStmt{std::move(*import)};
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
                case TokenType::Assignment:
                {
                    // If an assignment follows, check whether the starred expression could be a target list
                    auto targetList = convertToTargetList(std::move(*starredExpression), *m_current);
                    if (!targetList)
                    {
                        return tl::unexpected{std::move(targetList).error()};
                    }
                    addToNamespace(*targetList);
                    auto assignmentStmt = parseAssignmentStmt(std::move(*targetList));
                    if (!assignmentStmt)
                    {
                        return tl::unexpected{std::move(assignmentStmt).error()};
                    }
                    return Syntax::SimpleStmt{std::move(*assignmentStmt)};
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
                    auto augTarget = convertToAug(std::move(*starredExpression), *m_current);
                    if (!augTarget)
                    {
                        return tl::unexpected{std::move(augTarget).error()};
                    }
                    if (m_current->getTokenType() == TokenType::Colon)
                    {
                        auto colon = *m_current++;
                        auto expression = parseExpression();
                        if (!expression)
                        {
                            return tl::unexpected{std::move(expression).error()};
                        }
                        if (m_current == m_lexer.end() || m_current->getTokenType() != TokenType::Assignment)
                        {
                            return Syntax::SimpleStmt{Syntax::AnnotatedAssignmentSmt{
                                std::move(*augTarget), colon, std::move(*expression), std::nullopt}};
                        }
                        auto assignment = *m_current++;
                        if (m_current != m_lexer.end() && m_current->getTokenType() == TokenType::YieldKeyword)
                        {
                            auto yield = parseYieldExpression();
                            if (!yield)
                            {
                                return tl::unexpected{std::move(yield).error()};
                            }
                            return Syntax::SimpleStmt{
                                Syntax::AnnotatedAssignmentSmt{std::move(*augTarget), colon, std::move(*expression),
                                                               std::pair{assignment, std::move(*yield)}}};
                        }
                        auto starred = parseStarredExpression();
                        if (!starred)
                        {
                            return tl::unexpected{std::move(starred).error()};
                        }
                        return Syntax::SimpleStmt{
                            Syntax::AnnotatedAssignmentSmt{std::move(*augTarget), colon, std::move(*expression),
                                                           std::pair{assignment, std::move(*starred)}}};
                    }
                    auto augOp = *m_current++;
                    if (m_current != m_lexer.end() && m_current->getTokenType() == TokenType::YieldKeyword)
                    {
                        auto yield = parseYieldExpression();
                        if (!yield)
                        {
                            return tl::unexpected{std::move(yield).error()};
                        }
                        return Syntax::SimpleStmt{
                            Syntax::AugmentedAssignmentStmt{std::move(*augTarget), augOp, std::move(*yield)}};
                    }
                    auto expressionList = parseExpressionList();
                    if (!expressionList)
                    {
                        return tl::unexpected{std::move(expressionList).error()};
                    }
                    return Syntax::SimpleStmt{
                        Syntax::AugmentedAssignmentStmt{std::move(*augTarget), augOp, std::move(*expressionList)}};
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
    if (m_current == m_lexer.end() || m_current->getTokenType() != TokenType::Comma)
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

namespace
{
using namespace pylir;

template <class T>
struct Visitor
{
    Parser& parser;
    const BaseToken& assignOp;

    template <class... Args>
    T construct(Args&&... args)
    {
        if constexpr (std::is_same_v<T, Syntax::AugTarget>)
        {
            return T{std::forward<Args>(args)...};
        }
        else if constexpr (std::is_same_v<T, Syntax::TargetList>)
        {
            Syntax::Target target{std::forward<Args>(args)...};
            return Syntax::TargetList{std::make_unique<Syntax::Target>(std::move(target)), {}, std::nullopt};
        }
    }

    using Ret = tl::expected<T, std::string>;

    template <class ThisClass, class Getter>
    auto visitBinaryOp(ThisClass&& thisClass, Getter getter)
        -> std::enable_if_t<!std::is_lvalue_reference_v<ThisClass>, Ret>
    {
        return pylir::match(
            std::move(thisClass.variant),
            [&](std::unique_ptr<typename ThisClass::BinOp>&& binOp) -> Ret
            {
                auto& [lhs, token, rhs] = *binOp;
                return tl::unexpected{parser
                                          .createDiagnosticsBuilder(token, Diag::CANNOT_ASSIGN_TO_RESULT_OF_OPERATOR_N,
                                                                    std::invoke(getter, token))
                                          .addLabel(token, std::nullopt, Diag::ERROR_COLOUR)
                                          .addLabel(assignOp, std::nullopt, Diag::ERROR_COMPLY)
                                          .emitError()};
            },
            [&](auto&& other) -> std::enable_if_t<!std::is_lvalue_reference_v<decltype(other)>, Ret> {
                return visit(std::forward<decltype(other)>(other));
            });
    }

    template <class Thing = T>
    auto visit(Syntax::StarredItem&& expression) -> std::enable_if_t<std::is_same_v<Thing, Syntax::TargetList>, Ret>
    {
        return pylir::match(
            std::move(expression.variant),
            [&](Syntax::AssignmentExpression&& assignmentExpression) { return visit(std::move(assignmentExpression)); },
            [&](std::pair<BaseToken, Syntax::OrExpr>&& pair) -> Ret
            {
                auto target = visit(std::move(pair.second));
                if (!target)
                {
                    return target;
                }
                return construct(std::pair{pair.first, std::move(target->firstExpr)});
            });
    }

    template <class Thing = T>
    auto visit(Syntax::Enclosure&& expression) -> std::enable_if_t<std::is_same_v<T, Thing>, Ret>
    {
        return pylir::match(
            std::move(expression.variant),
            [&](Syntax::Enclosure::ParenthForm&& parenthForm) -> Ret
            {
                if (!parenthForm.expression)
                {
                    return construct(
                        Syntax::Target::Parenth{parenthForm.openParenth, std::nullopt, parenthForm.closeParenth});
                }
                auto list = visit(std::move(*parenthForm.expression));
                if (!list)
                {
                    return list;
                }
                return construct(
                    Syntax::Target::Parenth{parenthForm.openParenth, std::move(*list), parenthForm.closeParenth});
            },
            [&](Syntax::Enclosure::DictDisplay&&) -> Ret
            {
                return tl::unexpected{
                    parser.createDiagnosticsBuilder(expression, Diag::CANNOT_ASSIGN_TO_N, "dictionary display")
                        .addLabel(expression, std::nullopt, Diag::ERROR_COLOUR)
                        .addLabel(assignOp, std::nullopt, Diag::ERROR_COMPLY)
                        .emitError()};
            },
            [&](Syntax::Enclosure::SetDisplay&&) -> Ret
            {
                return tl::unexpected{
                    parser.createDiagnosticsBuilder(expression, Diag::CANNOT_ASSIGN_TO_N, "set display")
                        .addLabel(expression, std::nullopt, Diag::ERROR_COLOUR)
                        .addLabel(assignOp, std::nullopt, Diag::ERROR_COMPLY)
                        .emitError()};
            },
            [&](Syntax::Enclosure::ListDisplay&& listDisplay) -> Ret
            {
                return pylir::match(
                    std::move(listDisplay.variant),
                    [&](std::monostate) -> Ret {
                        return construct(
                            Syntax::Target::Square{listDisplay.openSquare, std::nullopt, listDisplay.closeSquare});
                    },
                    [&](Syntax::Comprehension&&) -> Ret
                    {
                        return tl::unexpected{
                            parser.createDiagnosticsBuilder(expression, Diag::CANNOT_ASSIGN_TO_N, "list display")
                                .addLabel(expression, std::nullopt, Diag::ERROR_COLOUR)
                                .addLabel(assignOp, std::nullopt, Diag::ERROR_COMPLY)
                                .emitError()};
                    },
                    [&](Syntax::StarredList&& starredList) -> Ret
                    {
                        Syntax::TargetList targetList{};
                        auto first = visit(std::move(*starredList.firstExpr));
                        if (!first)
                        {
                            return first;
                        }
                        targetList.firstExpr = std::move(first->firstExpr);
                        for (auto& [comma, item] : starredList.remainingExpr)
                        {
                            auto other = visit(std::move(*item));
                            if (!other)
                            {
                                return other;
                            }
                            targetList.remainingExpr.emplace_back(comma, std::move(other->firstExpr));
                        }
                        targetList.trailingComma = starredList.trailingComma;
                        return construct(Syntax::Target::Square{listDisplay.openSquare, std::move(targetList),
                                                                listDisplay.closeSquare});
                    });
            },
            [&](Syntax::Enclosure::YieldAtom&&) -> Ret
            {
                return tl::unexpected{
                    parser.createDiagnosticsBuilder(expression, Diag::CANNOT_ASSIGN_TO_N, "yield expression")
                        .addLabel(expression, std::nullopt, Diag::ERROR_COLOUR)
                        .addLabel(assignOp, std::nullopt, Diag::ERROR_COMPLY)
                        .emitError()};
            },
            [&](Syntax::Enclosure::GeneratorExpression&&) -> Ret
            {
                return tl::unexpected{
                    parser.createDiagnosticsBuilder(expression, Diag::CANNOT_ASSIGN_TO_N, "generator expression")
                        .addLabel(expression, std::nullopt, Diag::ERROR_COLOUR)
                        .addLabel(assignOp, std::nullopt, Diag::ERROR_COMPLY)
                        .emitError()};
            });
    }

    Ret visit(Syntax::Atom&& expression)
    {
        return pylir::match(
            std::move(expression.variant),
            [&](IdentifierToken&& identifierToken) -> Ret { return construct(std::move(identifierToken)); },
            [&](Syntax::Atom::Literal&& literal) -> Ret
            {
                return tl::unexpected{
                    parser.createDiagnosticsBuilder(literal.token, Diag::CANNOT_ASSIGN_TO_N, "literal")
                        .addLabel(literal.token, std::nullopt, Diag::ERROR_COLOUR)
                        .addLabel(assignOp, std::nullopt, Diag::ERROR_COMPLY)
                        .emitError()};
            },
            [&](std::unique_ptr<Syntax::Enclosure>&& enclosure) -> Ret
            {
                if constexpr (std::is_same_v<Syntax::AugTarget, T>)
                {
                    return tl::unexpected{
                        parser.createDiagnosticsBuilder(*enclosure, Diag::CANNOT_ASSIGN_TO_N, "enclosure")
                            .addLabel(*enclosure, std::nullopt, Diag::ERROR_COLOUR)
                            .addLabel(assignOp, std::nullopt, Diag::ERROR_COMPLY)
                            .emitError()};
                }
                else
                {
                    return visit(std::move(*enclosure));
                }
            });
    }

    Ret visit(Syntax::Primary&& expression)
    {
        return pylir::match(
            std::move(expression.variant), [&](Syntax::Atom&& atom) { return visit(std::move(atom)); },
            [&](Syntax::Call&& call) -> Ret
            {
                return tl::unexpected{
                    parser.createDiagnosticsBuilder(call.openParentheses, Diag::CANNOT_ASSIGN_TO_RESULT_OF_N, "call")
                        .addLabel(call.openParentheses, call, std::nullopt, Diag::ERROR_COLOUR)
                        .addLabel(assignOp, std::nullopt, Diag::ERROR_COMPLY)
                        .emitError()};
            },
            [&](auto&& value) -> std::enable_if_t<std::is_rvalue_reference_v<decltype(value)>, Ret> {
                return construct(std::forward<decltype(value)>(value));
            });
    }

    Ret visit(Syntax::AwaitExpr&& expression)
    {
        return tl::unexpected{
            parser
                .createDiagnosticsBuilder(expression.awaitToken, Diag::CANNOT_ASSIGN_TO_RESULT_OF_N, "await expression")
                .addLabel(expression.awaitToken, std::nullopt, Diag::ERROR_COLOUR)
                .addLabel(assignOp, std::nullopt, Diag::ERROR_COMPLY)
                .emitError()};
    }

    Ret visit(Syntax::Power&& expression)
    {
        if (expression.rightHand)
        {
            return tl::unexpected{parser
                                      .createDiagnosticsBuilder(expression.rightHand->first,
                                                                Diag::CANNOT_ASSIGN_TO_RESULT_OF_OPERATOR_N,
                                                                TokenType::PowerOf)
                                      .addLabel(expression.rightHand->first, std::nullopt, Diag::ERROR_COLOUR)
                                      .addLabel(assignOp, std::nullopt, Diag::ERROR_COMPLY)
                                      .emitError()};
        }
        return pylir::match(std::move(expression.variant), [&](auto&& value) { return visit(std::forward<decltype(value)>(value)); });
    }

    Ret visit(Syntax::UExpr&& expression)
    {
        return pylir::match(
            std::move(expression.variant), [&](Syntax::Power&& power) { return visit(std::move(power)); },
            [&](std::pair<Token, std::unique_ptr<Syntax::UExpr>>&& pair) -> Ret
            {
                return tl::unexpected{
                    parser
                        .createDiagnosticsBuilder(pair.first, Diag::CANNOT_ASSIGN_TO_RESULT_OF_N,
                                                  fmt::format("unary operator {:q}", pair.first.getTokenType()))
                        .addLabel(pair.first, std::nullopt, Diag::ERROR_COLOUR)
                        .addLabel(assignOp, std::nullopt, Diag::ERROR_COMPLY)
                        .emitError()};
            });
    }

    Ret visit(Syntax::MExpr&& expression)
    {
        return pylir::match(
            std::move(expression.variant),
            [&](auto&& binOp) -> std::enable_if_t<std::is_rvalue_reference_v<decltype(binOp)>, Ret>
            {
                auto& [lhs, token, rhs] = *binOp;
                TokenType tokenType;
                if constexpr (std::is_same_v<Token, std::decay_t<decltype(token)>>)
                {
                    tokenType = token.getTokenType();
                }
                else
                {
                    tokenType = TokenType::AtSign;
                }
                return tl::unexpected{
                    parser.createDiagnosticsBuilder(token, Diag::CANNOT_ASSIGN_TO_RESULT_OF_OPERATOR_N, tokenType)
                        .addLabel(token, std::nullopt, Diag::ERROR_COLOUR)
                        .addLabel(assignOp, std::nullopt, Diag::ERROR_COMPLY)
                        .emitError()};
            },
            [&](Syntax::UExpr&& other) { return visit(std::move(other)); });
    }

    Ret visit(Syntax::AExpr&& expression)
    {
        return visitBinaryOp(std::move(expression), &Token::getTokenType);
    }

    Ret visit(Syntax::ShiftExpr&& expression)
    {
        return visitBinaryOp(std::move(expression), &Token::getTokenType);
    }

    Ret visit(Syntax::AndExpr&& expression)
    {
        return visitBinaryOp(std::move(expression), [](auto&&) { return TokenType::BitAnd; });
    }

    Ret visit(Syntax::XorExpr&& expression)
    {
        return visitBinaryOp(std::move(expression), [](auto&&) { return TokenType::BitXor; });
    }

    Ret visit(Syntax::OrExpr&& expression)
    {
        return visitBinaryOp(std::move(expression), [](auto&&) { return TokenType::BitOr; });
    }

    Ret visit(Syntax::Comparison&& expression)
    {
        if (!expression.rest.empty())
        {
            auto builder = parser.createDiagnosticsBuilder(expression.rest.front().first.firstToken,
                                                           Diag::CANNOT_ASSIGN_TO_RESULT_OF_N, "comparison");
            for (auto& [pair, other] : expression.rest)
            {
                if (pair.secondToken)
                {
                    builder.addLabel(pair.firstToken, *pair.secondToken, std::nullopt, Diag::ERROR_COLOUR);
                }
                else
                {
                    builder.addLabel(pair.firstToken, std::nullopt, Diag::ERROR_COLOUR);
                }
            }
            return tl::unexpected{builder.addLabel(assignOp, std::nullopt, Diag::ERROR_COMPLY).emitError()};
        }
        return visit(std::move(expression.left));
    }

    Ret visit(Syntax::NotTest&& expression)
    {
        return pylir::match(
            std::move(expression.variant),
            [&](Syntax::Comparison&& comparison) { return visit(std::move(comparison)); },
            [&](std::pair<BaseToken, std::unique_ptr<Syntax::NotTest>>&& pair) -> Ret
            {
                return tl::unexpected{parser
                                          .createDiagnosticsBuilder(pair.first,
                                                                    Diag::CANNOT_ASSIGN_TO_RESULT_OF_OPERATOR_N,
                                                                    TokenType::NotKeyword)
                                          .addLabel(pair.first, std::nullopt, Diag::ERROR_COLOUR)
                                          .addLabel(assignOp, std::nullopt, Diag::ERROR_COMPLY)
                                          .emitError()};
            });
    }

    Ret visit(Syntax::AndTest&& expression)
    {
        return visitBinaryOp(std::move(expression), [](auto&&) { return TokenType ::AndKeyword; });
    }

    Ret visit(Syntax::OrTest&& expression)
    {
        return visitBinaryOp(std::move(expression), [](auto&&) { return TokenType::OrKeyword; });
    }

    Ret visit(Syntax::ConditionalExpression&& expression)
    {
        if (expression.suffix)
        {
            return tl::unexpected{parser
                                      .createDiagnosticsBuilder(expression.suffix->ifToken,
                                                                Diag::CANNOT_ASSIGN_TO_RESULT_OF_N,
                                                                "conditional expression")
                                      .addLabel(expression.suffix->ifToken, *expression.suffix->elseValue, std::nullopt,
                                                Diag::ERROR_COLOUR)
                                      .addLabel(assignOp, std::nullopt, Diag::ERROR_COMPLY)
                                      .emitError()};
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
                return tl::unexpected{parser
                                          .createDiagnosticsBuilder(*lambdaExpression,
                                                                    Diag::CANNOT_ASSIGN_TO_RESULT_OF_N,
                                                                    "lambda expression")
                                          .addLabel(*lambdaExpression, std::nullopt, Diag::ERROR_COLOUR)
                                          .addLabel(assignOp, std::nullopt, Diag::ERROR_COMPLY)
                                          .emitError()};
            });
    }

    Ret visit(Syntax::AssignmentExpression&& assignmentExpression)
    {
        if (assignmentExpression.identifierAndWalrus)
        {
            return tl::unexpected{
                parser
                    .createDiagnosticsBuilder(assignmentExpression.identifierAndWalrus->second,
                                              Diag::CANNOT_ASSIGN_TO_RESULT_OF_OPERATOR_N, TokenType::Walrus)
                    .addLabel(assignmentExpression.identifierAndWalrus->second, std::nullopt, Diag::ERROR_COLOUR)
                    .addLabel(assignOp, std::nullopt, Diag::ERROR_COMPLY)
                    .emitError()};
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
                if constexpr (std::is_same_v<Syntax::AugTarget, T>)
                {
                    if (!items.leading.empty())
                    {
                        auto& last = items.last ? *items.last : items.leading.back().first;
                        return tl::unexpected{
                            parser
                                .createDiagnosticsBuilder(items.leading.front().first, Diag::CANNOT_ASSIGN_TO_N,
                                                          "multiple values")
                                .addLabel(items.leading.front().first, last, std::nullopt, Diag::ERROR_COLOUR)
                                .addLabel(assignOp, std::nullopt, Diag::ERROR_COMPLY)
                                .emitError()};
                    }
                    if (!items.last)
                    {
                        return tl::unexpected{parser
                                                  .createDiagnosticsBuilder(assignOp, Diag::EXPECTED_N_BEFORE_N,
                                                                            "identifier", "assignment")
                                                  .addLabel(assignOp, std::nullopt, Diag::ERROR_COLOUR)
                                                  .emitError()};
                    }
                    return pylir::match(
                        std::move(items.last->variant),
                        [&](Syntax::AssignmentExpression&& assignmentExpression)
                        { return visit(std::move(assignmentExpression)); },
                        [&](std::pair<BaseToken, Syntax::OrExpr>&&) -> Ret { PYLIR_UNREACHABLE; });
                }
                else
                {
                    if (items.leading.empty() && !items.last)
                    {
                        return tl::unexpected{parser
                                                  .createDiagnosticsBuilder(assignOp, Diag::EXPECTED_N_BEFORE_N,
                                                                            "identifier", "assignment")
                                                  .addLabel(assignOp, std::nullopt, Diag::ERROR_COLOUR)
                                                  .emitError()};
                    }
                    Syntax::TargetList targetList{};
                    std::optional<BaseToken> comma;
                    for (auto& iter : items.leading)
                    {
                        auto target = visit(std::move(iter.first));
                        if (!target)
                        {
                            return target;
                        }
                        if (!targetList.firstExpr)
                        {
                            targetList.firstExpr = std::move(target->firstExpr);
                        }
                        else
                        {
                            targetList.remainingExpr.emplace_back(*comma, std::move(target->firstExpr));
                        }
                        comma = iter.second;
                    }
                    if (!items.last)
                    {
                        targetList.trailingComma = comma;
                    }
                    else
                    {
                        auto target = visit(std::move(*items.last));
                        if (!target)
                        {
                            return target;
                        }
                        if (!targetList.firstExpr)
                        {
                            targetList.firstExpr = std::move(target->firstExpr);
                        }
                        else
                        {
                            targetList.remainingExpr.emplace_back(*comma, std::move(target->firstExpr));
                        }
                    }
                    return targetList;
                }
            });
    }
};
} // namespace

tl::expected<pylir::Syntax::AugTarget, std::string>
    pylir::Parser::convertToAug(Syntax::StarredExpression&& starredExpression, const BaseToken& assignOp)
{
    Visitor<Syntax::AugTarget> visitor{*this, assignOp};
    return visitor.visit(std::move(starredExpression));
}

tl::expected<pylir::Syntax::TargetList, std::string>
    pylir::Parser::convertToTargetList(Syntax::StarredExpression&& starredExpression, const BaseToken& assignOp)
{
    Visitor<Syntax::TargetList> visitor{*this, assignOp};
    return visitor.visit(std::move(starredExpression));
}

tl::expected<Syntax::ImportStmt, std::string> pylir::Parser::parseImportStmt()
{
    auto parseModule = [&]() -> tl::expected<Syntax::ImportStmt::Module, std::string>
    {
        std::vector<std::pair<IdentifierToken, BaseToken>> leading;
        do
        {
            auto identifier = expect(TokenType::Identifier);
            if (!identifier)
            {
                return tl::unexpected{std::move(identifier).error()};
            }
            if (m_current == m_lexer.end() || m_current->getTokenType() != TokenType::Dot)
            {
                return Syntax::ImportStmt::Module{std::move(leading), IdentifierToken{std::move(*identifier)}};
            }
            leading.emplace_back(IdentifierToken{std::move(*identifier)}, *m_current++);
        } while (true);
    };

    auto parseRelativeModule = [&]() -> tl::expected<Syntax::ImportStmt::RelativeModule, std::string>
    {
        std::vector<BaseToken> dots;
        for (; m_current != m_lexer.end() && m_current->getTokenType() == TokenType::Dot; m_current++)
        {
            dots.emplace_back(*m_current);
        }
        if (!dots.empty() && (m_current == m_lexer.end() || m_current->getTokenType() != TokenType::Identifier))
        {
            return Syntax::ImportStmt::RelativeModule{std::move(dots), std::nullopt};
        }
        auto module = parseModule();
        if (!module)
        {
            return tl::unexpected{std::move(module).error()};
        }
        return Syntax::ImportStmt::RelativeModule{std::move(dots), std::move(*module)};
    };
    if (m_current == m_lexer.end())
    {
        return tl::unexpected{
            createDiagnosticsBuilder(m_document->getText().size(), Diag::EXPECTED_N,
                                     fmt::format("{:q} or {:q}", TokenType::ImportKeyword, TokenType::FromKeyword))
                .addLabel(m_document->getText().size(), std::nullopt, Diag::ERROR_COLOUR)
                .emitError()};
    }
    switch (m_current->getTokenType())
    {
        case TokenType::ImportKeyword:
        {
            auto import = *m_current++;
            auto module = parseModule();
            if (!module)
            {
                return tl::unexpected{std::move(module).error()};
            }
            std::optional<std::pair<BaseToken, IdentifierToken>> name;
            if (m_current != m_lexer.end() && m_current->getTokenType() == TokenType::AsKeyword)
            {
                auto as = *m_current++;
                auto identifier = expect(TokenType::Identifier);
                if (!identifier)
                {
                    return tl::unexpected{std::move(identifier).error()};
                }
                name.emplace(as, IdentifierToken{std::move(*identifier)});
            }
            std::vector<Syntax::ImportStmt::ImportAsAs::Further> rest;
            while (m_current != m_lexer.end() && m_current->getTokenType() == TokenType::Comma)
            {
                auto comma = *m_current++;
                auto nextModule = parseModule();
                if (!nextModule)
                {
                    return tl::unexpected{std::move(nextModule).error()};
                }
                std::optional<std::pair<BaseToken, IdentifierToken>> nextName;
                if (m_current != m_lexer.end() && m_current->getTokenType() == TokenType::AsKeyword)
                {
                    auto as = *m_current++;
                    auto identifier = expect(TokenType::Identifier);
                    if (!identifier)
                    {
                        return tl::unexpected{std::move(identifier).error()};
                    }
                    nextName.emplace(as, IdentifierToken{std::move(*identifier)});
                }
                rest.push_back({comma, std::move(*nextModule), std::move(nextName)});
            }
            return Syntax::ImportStmt{
                Syntax::ImportStmt::ImportAsAs{import, std::move(*module), std::move(name), std::move(rest)}};
        }
        case TokenType::FromKeyword:
        {
            auto from = *m_current++;
            auto relative = parseRelativeModule();
            if (!relative)
            {
                return tl::unexpected{std::move(relative).error()};
            }
            auto import = expect(TokenType::ImportKeyword);
            if (!import)
            {
                return tl::unexpected{std::move(import).error()};
            }
            if (m_current != m_lexer.end() && m_current->getTokenType() == TokenType::Star)
            {
                auto star = *m_current++;
                return Syntax::ImportStmt{Syntax::ImportStmt::FromImportAll{from, std::move(*relative), *import, star}};
            }
            std::optional<BaseToken> openParenth;
            if (m_current != m_lexer.end() && m_current->getTokenType() == TokenType::OpenParentheses)
            {
                openParenth = *m_current++;
            }
            auto identifier = expect(TokenType::Identifier);
            if (!identifier)
            {
                return tl::unexpected{std::move(identifier).error()};
            }
            std::optional<std::pair<BaseToken, IdentifierToken>> name;
            if (m_current != m_lexer.end() && m_current->getTokenType() == TokenType::AsKeyword)
            {
                auto as = *m_current++;
                auto identifier = expect(TokenType::Identifier);
                if (!identifier)
                {
                    return tl::unexpected{std::move(identifier).error()};
                }
                name.emplace(as, IdentifierToken{std::move(*identifier)});
            }
            std::vector<Syntax::ImportStmt::FromImportList::Further> rest;
            std::optional<BaseToken> trailingComma;
            while (m_current != m_lexer.end() && m_current->getTokenType() == TokenType::Comma)
            {
                auto comma = *m_current++;
                if (openParenth
                    && (m_current == m_lexer.end() || m_current->getTokenType() == TokenType::CloseParentheses))
                {
                    trailingComma = comma;
                    break;
                }
                auto imported = expect(TokenType::Identifier);
                if (!imported)
                {
                    return tl::unexpected{std::move(imported).error()};
                }
                std::optional<std::pair<BaseToken, IdentifierToken>> nextName;
                if (m_current != m_lexer.end() && m_current->getTokenType() == TokenType::AsKeyword)
                {
                    auto as = *m_current++;
                    auto identifier = expect(TokenType::Identifier);
                    if (!identifier)
                    {
                        return tl::unexpected{std::move(identifier).error()};
                    }
                    nextName.emplace(as, IdentifierToken{std::move(*identifier)});
                }
                rest.push_back({comma, IdentifierToken{std::move(*imported)}, std::move(nextName)});
            }
            std::optional<BaseToken> closeParentheses;
            if (openParenth)
            {
                auto close = expect(TokenType::CloseParentheses);
                if (!close)
                {
                    return tl::unexpected{std::move(close).error()};
                }
                closeParentheses = *close;
            }
            return Syntax::ImportStmt{Syntax::ImportStmt::FromImportList{
                from, std::move(*relative), *import, openParenth, IdentifierToken{std::move(*identifier)},
                std::move(name), std::move(rest), trailingComma, closeParentheses}};
        }
        case TokenType::SyntaxError: return tl::unexpected{pylir::get<std::string>(m_current->getValue())};
        default:
            return tl::unexpected{
                createDiagnosticsBuilder(*m_current, Diag::EXPECTED_N_INSTEAD_OF_N,
                                         fmt::format("{:q} or {:q}", TokenType::ImportKeyword, TokenType::FromKeyword),
                                         m_current->getTokenType())
                    .addLabel(*m_current, std::nullopt, Diag::ERROR_COLOUR)
                    .emitError()};
    }
}
