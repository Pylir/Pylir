// Copyright 2022 Markus Böck
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "Parser.hpp"

#include <pylir/Diagnostics/DiagnosticMessages.hpp>
#include <pylir/Support/Functional.hpp>

tl::expected<pylir::Syntax::AssignmentStmt, std::string>
    pylir::Parser::parseAssignmentStmt(IntrVarPtr<Syntax::Target>&& firstItem)
{
    std::vector<std::pair<IntrVarPtr<Syntax::Target>, Token>> targets;
    bool hadFirst = firstItem != nullptr;
    if (firstItem)
    {
        auto assignment = expect(TokenType::Assignment);
        if (!assignment)
        {
            return tl::unexpected{std::move(assignment).error()};
        }
        targets.emplace_back(std::move(firstItem), std::move(*assignment));
    }
    IntrVarPtr<Syntax::Expression> leftOverStarredExpression;
    do
    {
        if (hadFirst && !peekedIs(firstInTarget))
        {
            break;
        }
        auto starredExpression = parseStarredExpression();
        if (!starredExpression)
        {
            return tl::unexpected{std::move(starredExpression).error()};
        }
        auto assignment = maybeConsume(TokenType::Assignment);
        if (!assignment)
        {
            leftOverStarredExpression = std::move(*starredExpression);
            break;
        }
        auto error = checkTarget(**starredExpression, *assignment);
        if (!error)
        {
            return tl::unexpected{std::move(error).error()};
        }
        addToNamespace(**starredExpression);
        targets.emplace_back(std::move(*starredExpression), *assignment);
    } while (peekedIs(firstInTarget));
    if (leftOverStarredExpression)
    {
        return Syntax::AssignmentStmt{{}, std::move(targets), nullptr, std::move(leftOverStarredExpression)};
    }
    if (peekedIs(TokenType::YieldKeyword))
    {
        auto yieldExpr = parseYieldExpression();
        if (!yieldExpr)
        {
            return tl::unexpected{std::move(yieldExpr).error()};
        }
        return Syntax::AssignmentStmt{
            {}, std::move(targets), nullptr, std::make_unique<Syntax::Yield>(std::move(*yieldExpr))};
    }

    auto starredExpression = parseStarredExpression();
    if (!starredExpression)
    {
        return tl::unexpected{std::move(starredExpression).error()};
    }
    return Syntax::AssignmentStmt{{}, std::move(targets), nullptr, std::move(*starredExpression)};
}

tl::expected<pylir::IntrVarPtr<pylir::Syntax::SimpleStmt>, std::string> pylir::Parser::parseSimpleStmt()
{
    if (m_current == m_lexer.end())
    {
        return make_node<Syntax::ExpressionStmt>(nullptr);
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
            return std::make_unique<Syntax::AssertStmt>(std::move(*assertStmt));
        }
        case TokenType::PassKeyword: return make_node<Syntax::SingleTokenStmt>(*m_current++);
        case TokenType::BreakKeyword:
        case TokenType::ContinueKeyword:
            if (!m_inLoop)
            {
                return tl::unexpected{createDiagnosticsBuilder(*m_current, Diag::OCCURRENCE_OF_N_OUTSIDE_OF_LOOP,
                                                               m_current->getTokenType())
                                          .addLabel(*m_current, std::nullopt, Diag::ERROR_COLOUR)
                                          .emitError()};
            }
            return make_node<Syntax::SingleTokenStmt>(*m_current++);
        case TokenType::DelKeyword:
        {
            auto delKeyword = *m_current++;
            auto targetList = parseTargetList(delKeyword);
            if (!targetList)
            {
                return tl::unexpected{std::move(targetList).error()};
            }
            addToNamespace(**targetList);
            return make_node<Syntax::DelStmt>(delKeyword, std::move(*targetList));
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
            if (!peekedIs(firstInExpression))
            {
                return make_node<Syntax::ReturnStmt>(returnKeyword, nullptr);
            }
            auto expressionList = parseExpressionList();
            if (!expressionList)
            {
                return tl::unexpected{std::move(expressionList).error()};
            }
            return make_node<Syntax::ReturnStmt>(returnKeyword, std::move(*expressionList));
        }
        case TokenType::YieldKeyword:
        {
            auto yieldExpr = parseYieldExpression();
            if (!yieldExpr)
            {
                return tl::unexpected{std::move(yieldExpr).error()};
            }
            return make_node<Syntax::ExpressionStmt>(std::make_unique<Syntax::Yield>(std::move(*yieldExpr)));
        }
        case TokenType::RaiseKeyword:
        {
            auto raise = *m_current++;
            if (!peekedIs(firstInExpression))
            {
                return make_node<Syntax::RaiseStmt>(raise, nullptr, nullptr);
            }
            auto expression = parseExpression();
            if (!expression)
            {
                return tl::unexpected{std::move(expression).error()};
            }
            if (!maybeConsume(TokenType::FromKeyword))
            {
                return make_node<Syntax::RaiseStmt>(raise, std::move(*expression), nullptr);
            }
            auto source = parseExpression();
            if (!source)
            {
                return tl::unexpected{std::move(source).error()};
            }
            return make_node<Syntax::RaiseStmt>(raise, std::move(*expression), std::move(*source));
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
            std::vector<IdentifierToken> identifiers;
            identifiers.emplace_back(std::move(*identifier));
            while (maybeConsume(TokenType::Comma))
            {
                auto another = expect(TokenType::Identifier);
                if (!another)
                {
                    return tl::unexpected{std::move(another).error()};
                }
                identifiers.emplace_back(std::move(*another));
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

                for (auto& iter : identifiers)
                {
                    if (auto error = handleToken(iter); !error)
                    {
                        return tl::unexpected{error.error()};
                    }
                    if (!m_namespace.empty())
                    {
                        m_namespace.back().identifiers.emplace(iter, Scope::Kind::NonLocal);
                    }
                }
                return make_node<Syntax::GlobalOrNonLocalStmt>(keyword, std::move(identifiers));
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
                for (auto& iter : identifiers)
                {
                    if (auto error = handleToken(iter); !error)
                    {
                        return tl::unexpected{std::move(error).error()};
                    }
                    m_globals.insert(iter);
                    m_namespace.back().identifiers.emplace(iter, Scope::Kind::Global);
                }
            }
            else
            {
                for (auto& iter : identifiers)
                {
                    m_globals.insert(iter);
                }
            }
            return make_node<Syntax::GlobalOrNonLocalStmt>(keyword, std::move(identifiers));
        }
        case TokenType::FromKeyword:
        case TokenType::ImportKeyword:
        {
            auto import = parseImportStmt();
            if (!import)
            {
                return tl::unexpected{std::move(import).error()};
            }
            if (auto* fromImportAs = std::get_if<Syntax::ImportStmt::FromImport>(&import->variant);
                fromImportAs && fromImportAs->relativeModule.dots.empty() && fromImportAs->relativeModule.module
                && fromImportAs->relativeModule.module->identifiers.size() == 1
                && fromImportAs->relativeModule.module->identifiers.back().getValue() == "__future__")
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
                for (auto& iter : fromImportAs->imports)
                {
                    if (auto result = check(iter.first); !result)
                    {
                        return tl::unexpected{std::move(result).error()};
                    }
                }
                return make_node<Syntax::FutureStmt>(fromImportAs->from,
                                                     fromImportAs->relativeModule.module->identifiers.front(),
                                                     fromImportAs->import, std::move(fromImportAs->imports));
            }
            return std::make_unique<Syntax::ImportStmt>(std::move(*import));
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
                return make_node<Syntax::ExpressionStmt>(std::move(*starredExpression));
            }

            switch (m_current->getTokenType())
            {
                case TokenType::Assignment:
                {
                    // If an assignment follows, check whether the starred expression could be a target list
                    auto targetList = checkTarget(**starredExpression, *m_current);
                    if (!targetList)
                    {
                        return tl::unexpected{std::move(targetList).error()};
                    }
                    addToNamespace(**starredExpression);
                    auto assignmentStmt = parseAssignmentStmt(std::move(*starredExpression));
                    if (!assignmentStmt)
                    {
                        return tl::unexpected{std::move(assignmentStmt).error()};
                    }
                    return std::make_unique<Syntax::AssignmentStmt>(std::move(*assignmentStmt));
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
                    auto augTarget = checkAug(**starredExpression, *m_current);
                    if (!augTarget)
                    {
                        return tl::unexpected{std::move(augTarget).error()};
                    }
                    if (auto colon = maybeConsume(TokenType::Colon))
                    {
                        auto expression = parseExpression();
                        if (!expression)
                        {
                            return tl::unexpected{std::move(expression).error()};
                        }
                        std::vector<std::pair<IntrVarPtr<Syntax::Target>, Token>> vector;
                        vector.emplace_back(std::move(*starredExpression), std::move(*colon));
                        if (!maybeConsume(TokenType::Assignment))
                        {
                            return make_node<Syntax::AssignmentStmt>(std::move(vector), std::move(*expression),
                                                                     nullptr);
                        }
                        if (peekedIs(TokenType::YieldKeyword))
                        {
                            auto yield = parseYieldExpression();
                            if (!yield)
                            {
                                return tl::unexpected{std::move(yield).error()};
                            }
                            return make_node<Syntax::AssignmentStmt>(
                                std::move(vector), std::move(*expression),
                                std::make_unique<Syntax::Yield>(std::move(*yield)));
                        }
                        auto starred = parseStarredExpression();
                        if (!starred)
                        {
                            return tl::unexpected{std::move(starred).error()};
                        }
                        return make_node<Syntax::AssignmentStmt>(std::move(vector), std::move(*expression),
                                                                 std::move(*starred));
                    }
                    auto augOp = *m_current++;
                    std::vector<std::pair<IntrVarPtr<Syntax::Target>, Token>> vector;
                    vector.emplace_back(std::move(*starredExpression), augOp);
                    if (peekedIs(TokenType::YieldKeyword))
                    {
                        auto yield = parseYieldExpression();
                        if (!yield)
                        {
                            return tl::unexpected{std::move(yield).error()};
                        }
                        return make_node<Syntax::AssignmentStmt>(std::move(vector), nullptr,
                                                                 std::make_unique<Syntax::Yield>(std::move(*yield)));
                    }
                    auto expressionList = parseExpressionList();
                    if (!expressionList)
                    {
                        return tl::unexpected{std::move(expressionList).error()};
                    }
                    return make_node<Syntax::AssignmentStmt>(std::move(vector), nullptr, std::move(*expressionList));
                }
                default: return make_node<Syntax::ExpressionStmt>(std::move(*starredExpression));
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
    if (!maybeConsume(TokenType::Comma))
    {
        return Syntax::AssertStmt{{}, std::move(*assertKeyword), std::move(*expression), nullptr};
    }
    auto message = parseExpression();
    if (!message)
    {
        return tl::unexpected{std::move(message).error()};
    }
    return Syntax::AssertStmt{{}, std::move(*assertKeyword), std::move(*expression), std::move(*message)};
}

namespace
{
using namespace pylir;

struct Visitor
{
    Parser& parser;
    const Token& assignOp;
    bool augmented;

    // Disallows implicit conversions
    template <class T, std::enable_if_t<std::is_same_v<T, Syntax::Expression>>* = nullptr>
    std::optional<std::string> visit(const T& expression)
    {
        return expression.match([&](const auto& sub) { return visit(sub); });
    }

    std::optional<std::string> visit(const Syntax::AttributeRef&)
    {
        return std::nullopt;
    }

    std::optional<std::string> visit(const Syntax::Subscription&)
    {
        return std::nullopt;
    }

    std::optional<std::string> visit(const Syntax::Slice&)
    {
        return std::nullopt;
    }

    std::optional<std::string> visit(const Syntax::TupleConstruct& tupleConstruct)
    {
        if (augmented)
        {
            switch (tupleConstruct.items.size())
            {
                case 0:
                    return parser
                        .createDiagnosticsBuilder(tupleConstruct, Diag::OPERATOR_N_CANNOT_ASSIGN_TO_EMPTY_TUPLE,
                                                  assignOp.getTokenType())
                        .addLabel(tupleConstruct, std::nullopt, Diag::ERROR_COLOUR)
                        .addLabel(assignOp, std::nullopt, Diag::ERROR_COMPLY)
                        .emitError();
                case 1:
                    return parser
                        .createDiagnosticsBuilder(tupleConstruct,
                                                  Diag::OPERATOR_N_CANNOT_ASSIGN_TO_SINGLE_TUPLE_ELEMENT,
                                                  assignOp.getTokenType())
                        .addLabel(tupleConstruct, std::nullopt, Diag::ERROR_COLOUR)
                        .addLabel(assignOp, std::nullopt, Diag::ERROR_COMPLY)
                        .emitError();
                default:
                    return parser
                        .createDiagnosticsBuilder(tupleConstruct, Diag::OPERATOR_N_CANNOT_ASSIGN_TO_MULTIPLE_VARIABLES,
                                                  assignOp.getTokenType())
                        .addLabel(tupleConstruct, std::nullopt, Diag::ERROR_COLOUR)
                        .addLabel(assignOp, std::nullopt, Diag::ERROR_COMPLY)
                        .emitError();
            }
        }
        for (const auto& iter : tupleConstruct.items)
        {
            if (auto error = visit(*iter.expression))
            {
                return error;
            }
        }
        return std::nullopt;
    }

    std::optional<std::string> visit(const Syntax::DictDisplay& expression)
    {
        return parser.createDiagnosticsBuilder(expression, Diag::CANNOT_ASSIGN_TO_N, "dictionary display")
            .addLabel(expression, std::nullopt, Diag::ERROR_COLOUR)
            .addLabel(assignOp, std::nullopt, Diag::ERROR_COMPLY)
            .emitError();
    }

    std::optional<std::string> visit(const Syntax::SetDisplay& expression)
    {
        return parser.createDiagnosticsBuilder(expression, Diag::CANNOT_ASSIGN_TO_N, "set display")
            .addLabel(expression, std::nullopt, Diag::ERROR_COLOUR)
            .addLabel(assignOp, std::nullopt, Diag::ERROR_COMPLY)
            .emitError();
    }

    std::optional<std::string> visit(const Syntax::ListDisplay& expression)
    {
        if (std::holds_alternative<Syntax::Comprehension>(expression.variant) || augmented)
        {
            return parser.createDiagnosticsBuilder(expression, Diag::CANNOT_ASSIGN_TO_N, "list display")
                .addLabel(expression, std::nullopt, Diag::ERROR_COLOUR)
                .addLabel(assignOp, std::nullopt, Diag::ERROR_COMPLY)
                .emitError();
        }
        for (const auto& iter : pylir::get<std::vector<Syntax::StarredItem>>(expression.variant))
        {
            if (auto error = visit(*iter.expression))
            {
                return error;
            }
        }
        return std::nullopt;
    }

    std::optional<std::string> visit(const Syntax::Yield& expression)
    {
        return parser.createDiagnosticsBuilder(expression, Diag::CANNOT_ASSIGN_TO_N, "yield expression")
            .addLabel(expression, std::nullopt, Diag::ERROR_COLOUR)
            .addLabel(assignOp, std::nullopt, Diag::ERROR_COMPLY)
            .emitError();
    }

    std::optional<std::string> visit(const Syntax::Generator& expression)
    {
        return parser.createDiagnosticsBuilder(expression, Diag::CANNOT_ASSIGN_TO_N, "generator expression")
            .addLabel(expression, std::nullopt, Diag::ERROR_COLOUR)
            .addLabel(assignOp, std::nullopt, Diag::ERROR_COMPLY)
            .emitError();
    }

    std::optional<std::string> visit(const Syntax::BinOp& binOp)
    {
        return parser
            .createDiagnosticsBuilder(binOp.operation, Diag::CANNOT_ASSIGN_TO_RESULT_OF_OPERATOR_N,
                                      binOp.operation.getTokenType())
            .addLabel(binOp.operation, std::nullopt, Diag::ERROR_COLOUR)
            .addLabel(assignOp, std::nullopt, Diag::ERROR_COMPLY)
            .emitError();
    }

    std::optional<std::string> visit(const Syntax::Lambda& lambda)
    {
        return parser.createDiagnosticsBuilder(lambda, Diag::CANNOT_ASSIGN_TO_RESULT_OF_N, "lambda expression")
            .addLabel(lambda, std::nullopt, Diag::ERROR_COLOUR)
            .addLabel(assignOp, std::nullopt, Diag::ERROR_COMPLY)
            .emitError();
    }

    std::optional<std::string> visit(const Syntax::Atom& expression)
    {
        switch (expression.token.getTokenType())
        {
            case TokenType::StringLiteral:
            case TokenType::IntegerLiteral:
            case TokenType::FloatingPointLiteral:
            case TokenType::NoneKeyword:
            case TokenType::FalseKeyword:
            case TokenType::TrueKeyword:
            case TokenType::ByteLiteral:
            case TokenType::ComplexLiteral:
                return parser.createDiagnosticsBuilder(expression.token, Diag::CANNOT_ASSIGN_TO_N, "literal")
                    .addLabel(expression.token, std::nullopt, Diag::ERROR_COLOUR)
                    .addLabel(assignOp, std::nullopt, Diag::ERROR_COMPLY)
                    .emitError();
            default: return std::nullopt;
        }
    }

    std::optional<std::string> visit(const Syntax::Call& call)
    {
        return parser.createDiagnosticsBuilder(call.openParenth, Diag::CANNOT_ASSIGN_TO_RESULT_OF_N, "call")
            .addLabel(call.openParenth, call, std::nullopt, Diag::ERROR_COLOUR)
            .addLabel(assignOp, std::nullopt, Diag::ERROR_COMPLY)
            .emitError();
    }

    std::optional<std::string> visit(const Syntax::UnaryOp& expression)
    {
        return parser
            .createDiagnosticsBuilder(expression.operation, Diag::CANNOT_ASSIGN_TO_RESULT_OF_UNARY_OPERATOR_N,
                                      expression.operation.getTokenType())
            .addLabel(expression.operation, std::nullopt, Diag::ERROR_COLOUR)
            .addLabel(assignOp, std::nullopt, Diag::ERROR_COMPLY)
            .emitError();
    }

    std::optional<std::string> visit(const Syntax::Comparison& comparison)
    {
        // It's better looking, but still a technically arbitrary decision, but we'll emit the diagnostic only once
        // for the very last use of a comparison operator.
        const auto& back = comparison.rest.back().first;
        if (back.secondToken)
        {
            return parser
                .createDiagnosticsBuilder(back.firstToken, Diag::CANNOT_ASSIGN_TO_RESULT_OF_N,
                                          fmt::format(FMT_STRING("'{} {}'"), back.firstToken.getTokenType(),
                                                      back.secondToken->getTokenType()))
                .addLabel(back.firstToken, *back.secondToken, std::nullopt, Diag::ERROR_COLOUR)
                .addLabel(assignOp, std::nullopt, Diag::ERROR_COMPLY)
                .emitError();
        }

        return parser
            .createDiagnosticsBuilder(back.firstToken, Diag::CANNOT_ASSIGN_TO_RESULT_OF_OPERATOR_N,
                                      back.firstToken.getTokenType())
            .addLabel(back.firstToken, std::nullopt, Diag::ERROR_COLOUR)
            .addLabel(assignOp, std::nullopt, Diag::ERROR_COMPLY)
            .emitError();
    }

    std::optional<std::string> visit(const Syntax::Conditional& expression)
    {
        return parser
            .createDiagnosticsBuilder(expression.ifToken, Diag::CANNOT_ASSIGN_TO_RESULT_OF_N, "conditional expression")
            .addLabel(expression.ifToken, *expression.elseValue, std::nullopt, Diag::ERROR_COLOUR)
            .addLabel(assignOp, std::nullopt, Diag::ERROR_COMPLY)
            .emitError();
    }

    std::optional<std::string> visit(const Syntax::Assignment& assignment)
    {
        return parser
            .createDiagnosticsBuilder(assignment.walrus, Diag::CANNOT_ASSIGN_TO_RESULT_OF_OPERATOR_N, TokenType::Walrus)
            .addLabel(assignment.walrus, std::nullopt, Diag::ERROR_COLOUR)
            .addLabel(assignOp, std::nullopt, Diag::ERROR_COMPLY)
            .emitError();
    }
};
} // namespace

tl::expected<pylir::IntrVarPtr<Syntax::Target>, std::string>
    pylir::Parser::parseTarget(const pylir::Token& assignmentLikeToken)
{
    auto expression = parsePrimary();
    if (!expression)
    {
        return tl::unexpected{std::move(expression).error()};
    }
    auto error = checkTarget(**expression, assignmentLikeToken);
    if (!error)
    {
        return tl::unexpected{std::move(error).error()};
    }
    return expression;
}

tl::expected<pylir::IntrVarPtr<pylir::Syntax::Target>, std::string>
    pylir::Parser::parseTargetList(const pylir::Token& assignmentLikeToken)
{
    IntrVarPtr<Syntax::Target> target;
    {
        bool lastWasComma = false;
        std::vector<IntrVarPtr<Syntax::Target>> subTargets;
        while (subTargets.empty() || (m_current != m_lexer.end() && m_current->getTokenType() == TokenType::Comma))
        {
            if (!subTargets.empty())
            {
                m_current++;
                if (!firstInTarget(m_current->getTokenType()))
                {
                    lastWasComma = true;
                    break;
                }
            }
            auto other = parsePrimary();
            if (!other)
            {
                return tl::unexpected{std::move(other).error()};
            }
            subTargets.emplace_back(std::move(*other));
        }
        if (lastWasComma || subTargets.size() > 1)
        {
            std::vector<Syntax::StarredItem> starredItems(subTargets.size());
            std::transform(std::move_iterator(subTargets.begin()), std::move_iterator(subTargets.end()),
                           starredItems.begin(),
                           [](IntrVarPtr<Syntax::Target>&& target) {
                               return Syntax::StarredItem{std::nullopt, std::move(target)};
                           });
            target = make_node<Syntax::TupleConstruct>(std::nullopt, std::move(starredItems), std::nullopt);
        }
        else
        {
            target = std::move(subTargets[0]);
        }
    }

    auto error = checkTarget(*target, assignmentLikeToken);
    if (!error)
    {
        return tl::unexpected{std::move(error).error()};
    }
    return target;
}

tl::expected<void, std::string> pylir::Parser::checkAug(const Syntax::Expression& starredExpression,
                                                        const Token& assignOp)
{
    Visitor visitor{*this, assignOp, true};
    auto error = visitor.visit(starredExpression);
    if (error)
    {
        return tl::unexpected{std::move(*error)};
    }
    return {};
}

tl::expected<void, std::string> pylir::Parser::checkTarget(const Syntax::Expression& starredExpression,
                                                           const Token& assignOp)
{
    Visitor visitor{*this, assignOp, false};
    auto error = visitor.visit(starredExpression);
    if (error)
    {
        return tl::unexpected{std::move(*error)};
    }
    return {};
}

tl::expected<Syntax::ImportStmt, std::string> pylir::Parser::parseImportStmt()
{
    auto parseModule = [&]() -> tl::expected<Syntax::ImportStmt::Module, std::string>
    {
        std::vector<IdentifierToken> identifiers;
        do
        {
            auto identifier = expect(TokenType::Identifier);
            if (!identifier)
            {
                return tl::unexpected{std::move(identifier).error()};
            }
            identifiers.emplace_back(std::move(*identifier));
            if (!maybeConsume(TokenType::Dot))
            {
                return Syntax::ImportStmt::Module{std::move(identifiers)};
            }
        } while (true);
    };

    auto parseRelativeModule = [&]() -> tl::expected<Syntax::ImportStmt::RelativeModule, std::string>
    {
        std::vector<BaseToken> dots;
        while (auto dot = maybeConsume(TokenType::Dot))
        {
            dots.emplace_back(*dot);
        }
        if (!dots.empty() && !peekedIs(TokenType::Identifier))
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

            bool first = true;
            std::vector<std::pair<Syntax::ImportStmt::Module, std::optional<IdentifierToken>>> modules;
            while (first || maybeConsume(TokenType::Comma))
            {
                if (first)
                {
                    first = false;
                }
                auto nextModule = parseModule();
                if (!nextModule)
                {
                    return tl::unexpected{std::move(nextModule).error()};
                }
                std::optional<IdentifierToken> nextName;
                if (maybeConsume(TokenType::AsKeyword))
                {
                    auto identifier = expect(TokenType::Identifier);
                    if (!identifier)
                    {
                        return tl::unexpected{std::move(identifier).error()};
                    }
                    nextName.emplace(std::move(*identifier));
                }
                modules.emplace_back(std::move(*nextModule), std::move(nextName));
            }
            return Syntax::ImportStmt{{}, Syntax::ImportStmt::ImportAs{std::move(import), std::move(modules)}};
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
            if (auto star = maybeConsume(TokenType::Star))
            {
                return Syntax::ImportStmt{{},
                                          Syntax::ImportStmt::ImportAll{from, std::move(*relative), *import, *star}};
            }

            std::optional<BaseToken> openParenth = maybeConsume(TokenType::OpenParentheses);

            bool first = true;
            std::vector<std::pair<IdentifierToken, std::optional<IdentifierToken>>> imports;
            while (first || maybeConsume(TokenType::Comma))
            {
                if (first)
                {
                    first = false;
                }
                else if (openParenth && peekedIs(TokenType::CloseParentheses))
                {
                    break;
                }
                auto imported = expect(TokenType::Identifier);
                if (!imported)
                {
                    return tl::unexpected{std::move(imported).error()};
                }
                std::optional<IdentifierToken> nextName;
                if (maybeConsume(TokenType::AsKeyword))
                {
                    auto identifier = expect(TokenType::Identifier);
                    if (!identifier)
                    {
                        return tl::unexpected{std::move(identifier).error()};
                    }
                    nextName.emplace(std::move(*identifier));
                }
                imports.emplace_back(std::move(*imported), std::move(nextName));
            }
            if (openParenth)
            {
                auto close = expect(TokenType::CloseParentheses);
                if (!close)
                {
                    return tl::unexpected{std::move(close).error()};
                }
            }
            return Syntax::ImportStmt{
                {}, Syntax::ImportStmt::FromImport{from, std::move(*relative), *import, std::move(imports)}};
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
