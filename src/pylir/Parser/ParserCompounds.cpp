// Copyright 2022 Markus Böck
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "Parser.hpp"

#include <llvm/ADT/ScopeExit.h>

#include <pylir/Diagnostics/DiagnosticMessages.hpp>
#include <pylir/Support/Functional.hpp>
#include <pylir/Support/ValueReset.hpp>

#include "Visitor.hpp"

std::optional<pylir::Syntax::FileInput> pylir::Parser::parseFileInput()
{
    decltype(Syntax::Suite::statements) vector;
    while (true)
    {
        while (maybeConsume(TokenType::Newline))
        {
        }
        if (m_current == m_lexer.end())
        {
            break;
        }
        auto statement = parseStatement();
        if (!statement)
        {
            return std::nullopt;
        }
        vector.insert(vector.end(), std::move_iterator(statement->begin()), std::move_iterator(statement->end()));
    }
    return Syntax::FileInput{{std::move(vector)}, {m_globals.begin(), m_globals.end()}};
}

std::optional<decltype(pylir::Syntax::Suite::statements)> pylir::Parser::parseStatement()
{
    decltype(pylir::Syntax::Suite::statements) result;
    if (peekedIs(firstInCompoundStmt))
    {
        auto compound = parseCompoundStmt();
        if (!compound)
        {
            return std::nullopt;
        }
        result.emplace_back(std::move(*compound));
        return result;
    }

    auto stmtList = parseStmtList();
    if (!stmtList)
    {
        return std::nullopt;
    }
    auto newLine = expect(TokenType::Newline);
    if (!newLine)
    {
        return std::nullopt;
    }
    result.insert(result.end(), std::move_iterator(stmtList->begin()), std::move_iterator(stmtList->end()));
    return result;
}

std::optional<std::vector<pylir::IntrVarPtr<pylir::Syntax::SimpleStmt>>> pylir::Parser::parseStmtList()
{
    std::vector<IntrVarPtr<Syntax::SimpleStmt>> statements;
    while (peekedIs(TokenType::SemiColon) || peekedIs(firstInSimpleStmt))
    {
        while (maybeConsume(TokenType::SemiColon))
        {
        }
        if (m_current == m_lexer.end())
        {
            return statements;
        }
        auto simpleStmt = parseSimpleStmt();
        if (!simpleStmt)
        {
            return std::nullopt;
        }
        statements.push_back(std::move(*simpleStmt));
        if (!peekedIs(TokenType::SemiColon))
        {
            return statements;
        }
    }
    return statements;
}

std::optional<pylir::IntrVarPtr<pylir::Syntax::CompoundStmt>> pylir::Parser::parseCompoundStmt()
{
    if (m_current == m_lexer.end())
    {
        createError(endOfFileLoc(), Diag::EXPECTED_N, "statement").addLabel(endOfFileLoc());
        return std::nullopt;
    }
    switch (m_current->getTokenType())
    {
        case TokenType::IfKeyword:
        {
            auto ifStmt = parseIfStmt();
            if (!ifStmt)
            {
                return std::nullopt;
            }
            return std::make_unique<Syntax::IfStmt>(std::move(*ifStmt));
        }
        case TokenType::ForKeyword:
        {
            auto forStmt = parseForStmt();
            if (!forStmt)
            {
                return std::nullopt;
            }
            return std::make_unique<Syntax::ForStmt>(std::move(*forStmt));
        }
        case TokenType::TryKeyword:
        {
            auto tryStmt = parseTryStmt();
            if (!tryStmt)
            {
                return std::nullopt;
            }
            return std::make_unique<Syntax::TryStmt>(std::move(*tryStmt));
        }
        case TokenType::WithKeyword:
        {
            auto withStmt = parseWithStmt();
            if (!withStmt)
            {
                return std::nullopt;
            }
            return std::make_unique<Syntax::WithStmt>(std::move(*withStmt));
        }
        case TokenType::WhileKeyword:
        {
            auto whileStmt = parseWhileStmt();
            if (!whileStmt)
            {
                return std::nullopt;
            }
            return std::make_unique<Syntax::WhileStmt>(std::move(*whileStmt));
        }
        case TokenType::DefKeyword:
        {
            auto funcDef = parseFuncDef({}, std::nullopt);
            if (!funcDef)
            {
                return std::nullopt;
            }
            return std::make_unique<Syntax::FuncDef>(std::move(*funcDef));
        }
        case TokenType::ClassKeyword:
        {
            auto classDef = parseClassDef({});
            if (!classDef)
            {
                return std::nullopt;
            }
            return std::make_unique<Syntax::ClassDef>(std::move(*classDef));
        }
        case TokenType::AtSign:
        {
            std::vector<Syntax::Decorator> decorators;
            do
            {
                auto at = *m_current++;
                auto assignment = parseAssignmentExpression();
                if (!assignment)
                {
                    return std::nullopt;
                }
                auto newline = expect(TokenType::Newline);
                if (!newline)
                {
                    return std::nullopt;
                }
                decorators.push_back({at, std::move(*assignment), *newline});
            } while (peekedIs(TokenType::AtSign));
            if (auto async = maybeConsume(TokenType::AsyncKeyword))
            {
                auto func = parseFuncDef(std::move(decorators), std::move(*async));
                if (!func)
                {
                    return std::nullopt;
                }
                return std::make_unique<Syntax::FuncDef>(std::move(*func));
            }
            if (m_current == m_lexer.end())
            {
                createError(endOfFileLoc(), Diag::EXPECTED_N, "class or function").addLabel(endOfFileLoc());
                return std::nullopt;
            }
            switch (m_current->getTokenType())
            {
                case TokenType::DefKeyword:
                {
                    auto func = parseFuncDef(std::move(decorators), std::nullopt);
                    if (!func)
                    {
                        return std::nullopt;
                    }
                    return std::make_unique<Syntax::FuncDef>(std::move(*func));
                }
                case TokenType::ClassKeyword:
                {
                    auto clazz = parseClassDef(std::move(decorators));
                    if (!clazz)
                    {
                        return std::nullopt;
                    }
                    return std::make_unique<Syntax::ClassDef>(std::move(*clazz));
                }
                case TokenType::SyntaxError: return std::nullopt;
                default:
                {
                    createError(*m_current, Diag::EXPECTED_N_INSTEAD_OF_N, "class or function",
                                m_current->getTokenType())
                        .addLabel(*m_current);
                    return std::nullopt;
                }
            }
        }
        case TokenType::AsyncKeyword:
        {
            auto async = *m_current++;
            if (m_current == m_lexer.end())
            {
                createError(endOfFileLoc(), Diag::EXPECTED_N, "'for', 'with' or function").addLabel(endOfFileLoc());
                return std::nullopt;
            }
            switch (m_current->getTokenType())
            {
                case TokenType::DefKeyword:
                {
                    auto func = parseFuncDef({}, async);
                    if (!func)
                    {
                        return std::nullopt;
                    }
                    return std::make_unique<Syntax::FuncDef>(std::move(*func));
                }
                case TokenType::ForKeyword:
                {
                    auto forStmt = parseForStmt();
                    if (!forStmt)
                    {
                        return std::nullopt;
                    }
                    forStmt->maybeAsyncKeyword = async;
                    return std::make_unique<Syntax::ForStmt>(std::move(*forStmt));
                }
                case TokenType::WithKeyword:
                {
                    auto withStmt = parseWithStmt();
                    if (!withStmt)
                    {
                        return std::nullopt;
                    }
                    withStmt->maybeAsyncKeyword = async;
                    return std::make_unique<Syntax::WithStmt>(std::move(*withStmt));
                }
                case TokenType::SyntaxError: return std::nullopt;
                default:
                {
                    createError(*m_current, Diag::EXPECTED_N_INSTEAD_OF_N, "'for', 'with' or function",
                                m_current->getTokenType())
                        .addLabel(*m_current);
                    return std::nullopt;
                }
            }
        }
        case TokenType::SyntaxError: return std::nullopt;
        default:
        {
            createError(*m_current, Diag::EXPECTED_N_INSTEAD_OF_N, "statement", m_current->getTokenType())
                .addLabel(*m_current);
            return std::nullopt;
        }
    }
}

std::optional<pylir::Syntax::IfStmt::Else> pylir::Parser::parseElse()
{
    auto elseKeyowrd = *m_current++;
    auto elseColon = expect(TokenType::Colon);
    if (!elseColon)
    {
        return std::nullopt;
    }
    auto elseSuite = parseSuite();
    if (!elseSuite)
    {
        return std::nullopt;
    }
    return Syntax::IfStmt::Else{elseKeyowrd, *elseColon, std::make_unique<Syntax::Suite>(std::move(*elseSuite))};
}

std::optional<pylir::Syntax::IfStmt> pylir::Parser::parseIfStmt()
{
    auto ifKeyword = expect(TokenType::IfKeyword);
    if (!ifKeyword)
    {
        return std::nullopt;
    }
    auto assignment = parseAssignmentExpression();
    if (!assignment)
    {
        return std::nullopt;
    }
    auto colon = expect(TokenType::Colon);
    if (!colon)
    {
        return std::nullopt;
    }
    auto suite = parseSuite();
    if (!suite)
    {
        return std::nullopt;
    }
    std::vector<Syntax::IfStmt::Elif> elifs;
    while (auto elif = maybeConsume(TokenType::ElifKeyword))
    {
        auto condition = parseAssignmentExpression();
        if (!condition)
        {
            return std::nullopt;
        }
        auto elifColon = expect(TokenType::Colon);
        if (!elifColon)
        {
            return std::nullopt;
        }
        auto elIfSuite = parseSuite();
        if (!elIfSuite)
        {
            return std::nullopt;
        }
        elifs.push_back(
            {*elif, std::move(*condition), *elifColon, std::make_unique<Syntax::Suite>(std::move(*elIfSuite))});
    }
    std::optional<Syntax::IfStmt::Else> elseSection;
    if (peekedIs(TokenType::ElseKeyword))
    {
        auto parsedElse = parseElse();
        if (!parsedElse)
        {
            return std::nullopt;
        }
        elseSection = std::move(*parsedElse);
    }
    return Syntax::IfStmt{{},
                          *ifKeyword,
                          std::move(*assignment),
                          *colon,
                          std::make_unique<Syntax::Suite>(std::move(*suite)),
                          std::move(elifs),
                          std::move(elseSection)};
}

std::optional<pylir::Syntax::WhileStmt> pylir::Parser::parseWhileStmt()
{
    auto whileKeyword = expect(TokenType::WhileKeyword);
    if (!whileKeyword)
    {
        return std::nullopt;
    }
    auto condition = parseAssignmentExpression();
    if (!condition)
    {
        return std::nullopt;
    }
    auto colon = expect(TokenType::Colon);
    if (!colon)
    {
        return std::nullopt;
    }
    std::optional reset = pylir::ValueReset(m_inLoop);
    m_inLoop = true;
    auto suite = parseSuite();
    if (!suite)
    {
        return std::nullopt;
    }
    std::optional<Syntax::IfStmt::Else> elseSection;
    if (peekedIs(TokenType::ElseKeyword))
    {
        reset.reset();
        auto parsedElse = parseElse();
        if (!parsedElse)
        {
            return std::nullopt;
        }
        elseSection = std::move(*parsedElse);
    }
    return Syntax::WhileStmt{{},
                             *whileKeyword,
                             std::move(*condition),
                             *colon,
                             std::make_unique<Syntax::Suite>(std::move(*suite)),
                             std::move(elseSection)};
}

std::optional<pylir::Syntax::ForStmt> pylir::Parser::parseForStmt()
{
    auto forKeyword = expect(TokenType::ForKeyword);
    if (!forKeyword)
    {
        return std::nullopt;
    }
    auto targetList = parseTargetList(*forKeyword);
    if (!targetList)
    {
        return std::nullopt;
    }
    addToNamespace(**targetList);
    auto inKeyword = expect(TokenType::InKeyword);
    if (!inKeyword)
    {
        return std::nullopt;
    }
    auto expressionList = parseExpressionList();
    if (!expressionList)
    {
        return std::nullopt;
    }
    auto colon = expect(TokenType::Colon);
    if (!colon)
    {
        return std::nullopt;
    }
    std::optional reset = pylir::ValueReset(m_inLoop);
    m_inLoop = true;
    auto suite = parseSuite();
    if (!suite)
    {
        return std::nullopt;
    }
    std::optional<Syntax::IfStmt::Else> elseSection;
    if (peekedIs(TokenType::ElseKeyword))
    {
        reset.reset();
        auto parsedElse = parseElse();
        if (!parsedElse)
        {
            return std::nullopt;
        }
        elseSection = std::move(*parsedElse);
    }
    return Syntax::ForStmt{{},
                           std::nullopt,
                           *forKeyword,
                           std::move(*targetList),
                           *inKeyword,
                           std::move(*expressionList),
                           *colon,
                           std::make_unique<Syntax::Suite>(std::move(*suite)),
                           std::move(elseSection)};
}

std::optional<pylir::Syntax::TryStmt> pylir::Parser::parseTryStmt()
{
    auto tryKeyword = expect(TokenType::TryKeyword);
    if (!tryKeyword)
    {
        return std::nullopt;
    }
    auto colon = expect(TokenType::Colon);
    if (!colon)
    {
        return std::nullopt;
    }
    auto suite = parseSuite();
    if (!suite)
    {
        return std::nullopt;
    }
    if (auto finallyKeyword = maybeConsume(TokenType::FinallyKeyword))
    {
        auto finallyColon = expect(TokenType::Colon);
        if (!finallyColon)
        {
            return std::nullopt;
        }
        auto finallySuite = parseSuite();
        if (!finallySuite)
        {
            return std::nullopt;
        }
        return Syntax::TryStmt{{},
                               *tryKeyword,
                               *colon,
                               std::make_unique<Syntax::Suite>(std::move(*suite)),
                               {},
                               std::nullopt,
                               std::nullopt,
                               Syntax::TryStmt::Finally{*finallyKeyword, *finallyColon,
                                                        std::make_unique<Syntax::Suite>(std::move(*finallySuite))}};
    }

    std::optional<Syntax::TryStmt::ExceptAll> catchAll;
    std::vector<Syntax::TryStmt::ExceptArgs> exceptSections;
    do
    {
        auto exceptKeyword = expect(TokenType::ExceptKeyword);
        if (!exceptKeyword)
        {
            return std::nullopt;
        }
        if (catchAll)
        {
            createError(catchAll->exceptKeyword, Diag::EXCEPT_CLAUSE_WITHOUT_EXPRESSION_MUST_COME_LAST)
                .addLabel(catchAll->exceptKeyword, Diag::flags::bold);
            return std::nullopt;
        }
        if (auto exceptColon = maybeConsume(TokenType::Colon))
        {
            auto exceptSuite = parseSuite();
            if (!exceptSuite)
            {
                return std::nullopt;
            }
            catchAll = {*exceptKeyword, *exceptColon, std::make_unique<Syntax::Suite>(std::move(*exceptSuite))};
            continue;
        }
        auto expression = parseExpression();
        if (!expression)
        {
            return std::nullopt;
        }
        std::optional<IdentifierToken> name;
        if (maybeConsume(TokenType::AsKeyword))
        {
            auto id = expect(TokenType::Identifier);
            if (!id)
            {
                return std::nullopt;
            }
            addToNamespace(*id);
            name.emplace(std::move(*id));
        }
        auto exceptColon = expect(TokenType::Colon);
        if (!exceptColon)
        {
            return std::nullopt;
        }
        auto exceptSuite = parseSuite();
        if (!exceptSuite)
        {
            return std::nullopt;
        }
        exceptSections.push_back({*exceptKeyword, std::move(*expression), std::move(name), *exceptColon,
                                  std::make_unique<Syntax::Suite>(std::move(*exceptSuite))});
    } while (peekedIs(TokenType::ExceptKeyword));

    std::optional<Syntax::IfStmt::Else> elseSection;
    if (peekedIs(TokenType::ElseKeyword))
    {
        auto parsedElse = parseElse();
        if (!parsedElse)
        {
            return std::nullopt;
        }
        elseSection = std::move(*parsedElse);
    }

    std::optional<Syntax::TryStmt::Finally> finally;
    if (auto finallyKeyword = maybeConsume(TokenType::FinallyKeyword))
    {
        auto finallyColon = expect(TokenType::Colon);
        if (!finallyColon)
        {
            return std::nullopt;
        }
        auto finallySuite = parseSuite();
        if (!finallySuite)
        {
            return std::nullopt;
        }
        finally = Syntax::TryStmt::Finally{*finallyKeyword, *finallyColon,
                                           std::make_unique<Syntax::Suite>(std::move(*finallySuite))};
    }
    return Syntax::TryStmt{{},
                           *tryKeyword,
                           *colon,
                           std::make_unique<Syntax::Suite>(std::move(*suite)),
                           std::move(exceptSections),
                           std::move(catchAll),
                           std::move(elseSection),
                           std::move(finally)};
}

std::optional<pylir::Syntax::WithStmt> pylir::Parser::parseWithStmt()
{
    auto withKeyword = expect(TokenType::WithKeyword);
    if (!withKeyword)
    {
        return std::nullopt;
    }

    std::vector<Syntax::WithStmt::WithItem> withItems;
    while (withItems.empty() || maybeConsume(TokenType::Comma))
    {
        auto expression = parseExpression();
        if (!expression)
        {
            return std::nullopt;
        }
        IntrVarPtr<Syntax::Target> name;
        if (auto asKeyword = maybeConsume(TokenType::AsKeyword))
        {
            auto target = parseTarget(*asKeyword);
            if (!target)
            {
                return std::nullopt;
            }
            name = std::move(*target);
        }
        withItems.push_back({std::move(*expression), std::move(name)});
    }

    auto colon = expect(TokenType::Colon);
    if (!colon)
    {
        return std::nullopt;
    }
    auto suite = parseSuite();
    if (!suite)
    {
        return std::nullopt;
    }
    return Syntax::WithStmt{{},           std::nullopt,
                            *withKeyword, std::move(withItems),
                            *colon,       std::make_unique<Syntax::Suite>(std::move(*suite))};
}

std::optional<pylir::Syntax::Suite> pylir::Parser::parseSuite()
{
    decltype(Syntax::Suite::statements) statements;
    if (maybeConsume(TokenType::Newline))
    {
        if (!maybeConsume(TokenType::Indent))
        {
            // stmt_list was empty, and hence a newline immediately followed with no indent after.
            return Syntax::Suite{};
        }

        do
        {
            auto statement = parseStatement();
            if (!statement)
            {
                return std::nullopt;
            }
            statements.insert(statements.end(), std::move_iterator(statement->begin()),
                              std::move_iterator(statement->end()));
        } while (peekedIsNot(TokenType::Dedent));
        auto dedent = expect(TokenType::Dedent);
        if (!dedent)
        {
            return std::nullopt;
        }
        return Syntax::Suite{std::move(statements)};
    }

    auto statementList = parseStmtList();
    if (!statementList)
    {
        return std::nullopt;
    }
    auto newline = expect(TokenType::Newline);
    if (!newline)
    {
        return std::nullopt;
    }
    statements.insert(statements.end(), std::move_iterator(statementList->begin()),
                      std::move_iterator(statementList->end()));
    return Syntax::Suite{std::move(statements)};
}

std::optional<std::vector<pylir::Syntax::Parameter>> pylir::Parser::parseParameterList()
{
    std::vector<pylir::Syntax::Parameter> parameters;
    Syntax::Parameter::Kind currentKind = Syntax::Parameter::Normal;

    std::optional<BaseToken> seenPositionalOnly;
    std::optional<std::size_t> seenDefaultParam;
    std::variant<std::monostate, std::size_t, BaseToken> seenPosRest;
    std::optional<std::size_t> seenKwRest;
    while (parameters.empty() || maybeConsume(TokenType::Comma))
    {
        if (!parameters.empty())
        {
            if (m_current == m_lexer.end())
            {
                return parameters;
            }
        }

        std::optional<Token> stars;
        switch (m_current->getTokenType())
        {
            case TokenType::Divide:
            {
                if (parameters.empty())
                {
                    createError(*m_current, Diag::AT_LEAST_ONE_PARAMETER_REQUIRED_BEFORE_POSITIONAL_ONLY_INDICATOR)
                        .addLabel(*m_current);
                    return std::nullopt;
                }
                if (!seenPositionalOnly)
                {
                    seenPositionalOnly = *m_current++;
                    for (auto& iter : parameters)
                    {
                        iter.kind = Syntax::Parameter::PosOnly;
                    }
                    continue;
                }
                createError(*m_current, Diag::POSITIONAL_ONLY_INDICATOR_MAY_ONLY_APPEAR_ONCE)
                    .addLabel(*m_current)
                    .addNote(*seenPositionalOnly, Diag::PREVIOUS_OCCURRENCE_HERE)
                    .addLabel(*seenPositionalOnly);
                return std::nullopt;
            }
            case TokenType::Star:
                stars = *m_current++;
                if (currentKind == Syntax::Parameter::Normal)
                {
                    currentKind = Syntax::Parameter::KeywordOnly;
                    if (!peekedIs({TokenType::Comma, TokenType::Identifier}))
                    {
                        return parameters;
                    }
                    if (m_current->getTokenType() == TokenType::Comma)
                    {
                        seenPosRest = *stars;
                        continue;
                    }
                }
                break;
            case TokenType::PowerOf: currentKind = Syntax::Parameter::KeywordRest; stars = *m_current++;
            case TokenType::Identifier: break;
            default: return parameters;
        }

        auto identifier = expect(TokenType::Identifier);
        if (!identifier)
        {
            return std::nullopt;
        }

        IntrVarPtr<Syntax::Expression> maybeType;
        IntrVarPtr<Syntax::Expression> maybeDefault;
        if (maybeConsume(TokenType::Colon))
        {
            auto type = parseExpression();
            if (!type)
            {
                return std::nullopt;
            }
            maybeType = std::move(*type);
        }
        if (!stars && maybeConsume(TokenType::Assignment))
        {
            auto defaultVal = parseExpression();
            if (!defaultVal)
            {
                return std::nullopt;
            }
            maybeDefault = std::move(*defaultVal);
        }
        if (!maybeDefault && seenDefaultParam)
        {
            createError(*identifier,
                        Diag::NO_DEFAULT_ARGUMENT_FOR_PARAMETER_N_FOLLOWING_PARAMETERS_WITH_DEFAULT_ARGUMENTS,
                        pylir::get<std::string>(identifier->getValue()))
                .addLabel(*identifier)
                .addNote(parameters[*seenDefaultParam], Diag::PARAMETER_N_WITH_DEFAULT_ARGUMENT_HERE,
                         parameters[*seenDefaultParam].name.getValue())
                .addLabel(parameters[*seenDefaultParam]);
            return std::nullopt;
        }
        if (maybeDefault)
        {
            seenDefaultParam = parameters.size();
        }
        if (seenKwRest)
        {
            createError(*identifier, Diag::NO_MORE_PARAMETERS_ALLOWED_AFTER_EXCESS_KEYWORD_PARAMETER_N,
                        pylir::get<std::string>(identifier->getValue()), parameters[*seenKwRest].name.getValue())
                .addLabel(*identifier)
                .addNote(parameters[*seenKwRest], Diag::EXCESS_KEYWORD_PARAMETER_N_HERE,
                         parameters[*seenKwRest].name.getValue())
                .addLabel(parameters[*seenKwRest]);
            return std::nullopt;
        }

        if (!stars)
        {
            parameters.push_back({currentKind, stars, IdentifierToken(std::move(*identifier)), std::move(maybeType),
                                  std::move(maybeDefault)});
            continue;
        }

        if (stars->getTokenType() == TokenType::Star)
        {
            if (auto* token = std::get_if<BaseToken>(&seenPosRest))
            {
                createError(*identifier, Diag::STARRED_PARAMETER_NOT_ALLOWED_AFTER_KEYWORD_ONLY_INDICATOR)
                    .addLabel(*identifier)
                    .addNote(*token, Diag::KEYWORD_ONLY_INDICATOR_HERE)
                    .addLabel(*token);
                return std::nullopt;
            }
            if (auto* index = std::get_if<std::size_t>(&seenPosRest))
            {
                createError(*identifier, Diag::ONLY_ONE_STARRED_PARAMETER_ALLOWED)
                    .addLabel(*identifier)
                    .addNote(parameters[*index], Diag::STARRED_PARAMETER_N_HERE, parameters[*index].name.getValue())
                    .addLabel(parameters[*index]);
                return std::nullopt;
            }

            seenPosRest = parameters.size();
            parameters.push_back({Syntax::Parameter::PosRest, stars, IdentifierToken(std::move(*identifier)),
                                  std::move(maybeType), std::move(maybeDefault)});
            continue;
        }

        seenKwRest = parameters.size();
        parameters.push_back({Syntax::Parameter::KeywordRest, stars, IdentifierToken(std::move(*identifier)),
                              std::move(maybeType), std::move(maybeDefault)});
    }
    return parameters;
}

namespace
{
class NamespaceVisitor : public pylir::Syntax::Visitor<NamespaceVisitor>
{
    template <class T>
    void finishNamespace(T& def)
    {
        for (auto& [id, kind] : def.scope.identifiers)
        {
            switch (kind)
            {
                case pylir::Syntax::Scope::NonLocal:
                {
                    if (std::none_of(scopes.begin(), scopes.end(),
                                     [&id = id](const pylir::Syntax::Scope* scope) -> bool
                                     { return scope->identifiers.count(id); }))
                    {
                        onError(id);
                        break;
                    }
                    break;
                }
                case pylir::Syntax::Scope::Unknown:
                {
                    for (auto& iter : llvm::reverse(scopes))
                    {
                        auto res = iter->identifiers.find(id);
                        if (res != iter->identifiers.end())
                        {
                            switch (res->second)
                            {
                                case pylir::Syntax::Scope::Global: kind = pylir::Syntax::Scope::Global; break;
                                case pylir::Syntax::Scope::Unknown: continue;
                                case pylir::Syntax::Scope::Cell:
                                case pylir::Syntax::Scope::Local:
                                case pylir::Syntax::Scope::NonLocal: kind = pylir::Syntax::Scope::NonLocal; break;
                            }
                            break;
                        }
                    }
                    break;
                }
                default: break;
            }
        }

        // add any non locals from nested functions except if they are local to this function aka the referred
        // to local
        for (auto& [id, kind] : def.scope.identifiers)
        {
            if (kind != pylir::Syntax::Scope::NonLocal)
            {
                continue;
            }
            pylir::match(
                parentDef,
                [&id = id, this](std::monostate)
                {
                    if (!maybeTopLevelFunction)
                    {
                        return;
                    }
                    // Special case for the very top level function that is finishing the namespace.
                    // its scope is the very first one
                    auto iter = maybeTopLevelFunction->identifiers.insert({id, pylir::Syntax::Scope::NonLocal}).first;
                    if (iter->second == pylir::Syntax::Scope::Local)
                    {
                        iter->second = pylir::Syntax::Scope::Cell;
                    }
                },
                [&id = id](pylir::Syntax::FuncDef* funcDef)
                {
                    auto iter = funcDef->scope.identifiers.insert({id, pylir::Syntax::Scope::NonLocal}).first;
                    if (iter->second == pylir::Syntax::Scope::Local)
                    {
                        iter->second = pylir::Syntax::Scope::Cell;
                    }
                },
                [&id = id](pylir::Syntax::ClassDef* classDef) {
                    classDef->scope.identifiers.insert({id, pylir::Syntax::Scope::NonLocal});
                });
        }
    }

    pylir::Syntax::Scope* maybeTopLevelFunction;
    std::vector<const pylir::Syntax::Scope*> scopes;
    std::function<void(const pylir::IdentifierToken&)> onError;
    std::variant<std::monostate, pylir::Syntax::FuncDef*, pylir::Syntax::ClassDef*> parentDef;

public:
    NamespaceVisitor(pylir::Syntax::Scope* maybeScope, std::function<void(const pylir::IdentifierToken&)>&& onError)
        : maybeTopLevelFunction(maybeScope), onError(std::move(onError))
    {
        if (maybeScope)
        {
            scopes.push_back(maybeScope);
        }
    }

    using Visitor::visit;

    // TODO: consider having a non `const` version, using something different from Syntax::Visitor.
    //       SOMETHING to get rid of const_cast here

    void visit(const pylir::Syntax::ClassDef& classDef)
    {
        auto& def = const_cast<pylir::Syntax::ClassDef&>(classDef);
        {
            pylir::ValueReset reset(parentDef);
            parentDef = &def;
            Visitor::visit(classDef);
        }
        finishNamespace(def);
    }

    void visit(const pylir::Syntax::FuncDef& funcDef)
    {
        auto& def = const_cast<pylir::Syntax::FuncDef&>(funcDef);
        {
            scopes.push_back(&funcDef.scope);
            auto exit = llvm::make_scope_exit([&] { scopes.pop_back(); });
            pylir::ValueReset reset(parentDef);
            parentDef = &def;
            Visitor::visit(funcDef);
        }
        finishNamespace(def);
    }
};

} // namespace

void pylir::Parser::finishNamespace(Syntax::Suite& suite, Syntax::Scope* maybeScope) const
{
    NamespaceVisitor visitor(
        maybeScope, [&](const IdentifierToken& token)
        { createError(token, Diag::COULD_NOT_FIND_VARIABLE_N_IN_OUTER_SCOPES, token.getValue()).addLabel(token); });
    visitor.visit(suite);
}

std::optional<pylir::Syntax::FuncDef> pylir::Parser::parseFuncDef(std::vector<Syntax::Decorator>&& decorators,
                                                                  std::optional<BaseToken>&& asyncKeyword)
{
    auto defKeyword = expect(TokenType::DefKeyword);
    if (!defKeyword)
    {
        return std::nullopt;
    }
    auto funcName = expect(TokenType::Identifier);
    if (!funcName)
    {
        return std::nullopt;
    }
    addToNamespace(*funcName);
    auto openParenth = expect(TokenType::OpenParentheses);
    if (!openParenth)
    {
        return std::nullopt;
    }
    std::vector<Syntax::Parameter> parameterList;
    if (peekedIsNot(TokenType::CloseParentheses))
    {
        auto parsedParameterList = parseParameterList();
        if (!parsedParameterList)
        {
            return std::nullopt;
        }
        parameterList = std::move(*parsedParameterList);
    }
    auto closeParenth = expect(TokenType::CloseParentheses);
    if (!closeParenth)
    {
        return std::nullopt;
    }
    IntrVarPtr<Syntax::Expression> suffix;
    if (maybeConsume(TokenType::Arrow))
    {
        auto expression = parseExpression();
        if (!expression)
        {
            return std::nullopt;
        }
        suffix = std::move(*expression);
    }
    auto colon = expect(TokenType::Colon);
    if (!colon)
    {
        return std::nullopt;
    }
    m_namespace.emplace_back();
    std::optional exit = llvm::make_scope_exit([&] { m_namespace.pop_back(); });

    // add parameters to local variables
    for (auto& iter : parameterList)
    {
        addToNamespace(iter.name);
    }

    pylir::ValueReset resetLoop(m_inLoop);
    pylir::ValueReset resetFunc(m_inFunc);
    m_inLoop = false;
    m_inFunc = true;
    auto suite = parseSuite();
    if (!suite)
    {
        return std::nullopt;
    }

    auto scope = std::move(m_namespace.back());
    exit.reset();
    if (m_namespace.empty())
    {
        // this indicates that this funcdef is at global scope or only nested in classes. We now need to resolve any
        // nonlocals inside any nested funcDefs and figure out whether any unknowns are nonlocal or global

        for (auto& [iter, kind] : scope.identifiers)
        {
            if (kind != Syntax::Scope::NonLocal)
            {
                continue;
            }
            createError(iter, Diag::COULD_NOT_FIND_VARIABLE_N_IN_OUTER_SCOPES, iter.getValue()).addLabel(iter);
        }
        finishNamespace(*suite, &scope);
    }

    return Syntax::FuncDef{{},
                           std::move(decorators),
                           asyncKeyword,
                           *defKeyword,
                           IdentifierToken{std::move(*funcName)},
                           *openParenth,
                           std::move(parameterList),
                           *closeParenth,
                           std::move(suffix),
                           *colon,
                           std::make_unique<Syntax::Suite>(std::move(*suite)),
                           std::move(scope)};
}

std::optional<pylir::Syntax::ClassDef> pylir::Parser::parseClassDef(std::vector<Syntax::Decorator>&& decorators)
{
    auto classKeyword = expect(TokenType::ClassKeyword);
    if (!classKeyword)
    {
        return std::nullopt;
    }
    auto className = expect(TokenType::Identifier);
    if (!className)
    {
        return std::nullopt;
    }
    addToNamespace(*className);
    std::optional<Syntax::ClassDef::Inheritance> inheritance;
    if (auto open = maybeConsume(TokenType::OpenParentheses))
    {
        std::vector<Syntax::Argument> argumentList;
        if (peekedIsNot(TokenType::CloseParentheses))
        {
            auto parsedArgumentList = parseArgumentList();
            if (!parsedArgumentList)
            {
                return std::nullopt;
            }
            argumentList = std::move(*parsedArgumentList);
        }
        auto close = expect(TokenType::CloseParentheses);
        if (!close)
        {
            return std::nullopt;
        }
        inheritance = Syntax::ClassDef::Inheritance{*open, std::move(argumentList), *close};
    }
    auto colon = expect(TokenType::Colon);
    if (!colon)
    {
        return std::nullopt;
    }
    m_namespace.emplace_back();
    std::optional exit = llvm::make_scope_exit([&] { m_namespace.pop_back(); });
    pylir::ValueReset resetLoop(m_inLoop, m_inLoop);
    pylir::ValueReset resetFunc(m_inFunc, m_inFunc);
    m_inLoop = false;
    m_inFunc = false;
    auto suite = parseSuite();
    if (!suite)
    {
        return std::nullopt;
    }

    auto scope = std::move(m_namespace.back());
    exit.reset();
    if (m_namespace.empty())
    {
        for (auto& [iter, kind] : scope.identifiers)
        {
            if (kind != Syntax::Scope::NonLocal)
            {
                continue;
            }
            createError(iter, Diag::COULD_NOT_FIND_VARIABLE_N_IN_OUTER_SCOPES, iter.getValue()).addLabel(iter);
        }
        finishNamespace(*suite);
    }
    return Syntax::ClassDef{{},
                            std::move(decorators),
                            *classKeyword,
                            IdentifierToken{std::move(*className)},
                            std::move(inheritance),
                            *colon,
                            std::make_unique<Syntax::Suite>(std::move(*suite)),
                            std::move(scope)};
}
