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

tl::expected<pylir::Syntax::FileInput, std::string> pylir::Parser::parseFileInput()
{
    decltype(Syntax::Suite::statements) vector;
    while (true)
    {
        while (m_current != m_lexer.end() && m_current->getTokenType() == TokenType::Newline)
        {
            m_current++;
        }
        if (m_current == m_lexer.end())
        {
            break;
        }
        auto statement = parseStatement();
        if (!statement)
        {
            return tl::unexpected{std::move(statement).error()};
        }
        vector.insert(vector.end(), std::move_iterator(statement->begin()), std::move_iterator(statement->end()));
    }
    return Syntax::FileInput{{std::move(vector)}, {m_globals.begin(), m_globals.end()}};
}

tl::expected<decltype(pylir::Syntax::Suite::statements), std::string> pylir::Parser::parseStatement()
{
    decltype(pylir::Syntax::Suite::statements) result;
    if (m_current != m_lexer.end() && firstInCompoundStmt(m_current->getTokenType()))
    {
        auto compound = parseCompoundStmt();
        if (!compound)
        {
            return tl::unexpected{std::move(compound).error()};
        }
        result.emplace_back(std::move(*compound));
        return result;
    }

    auto stmtList = parseStmtList();
    if (!stmtList)
    {
        return tl::unexpected{std::move(stmtList).error()};
    }
    auto newLine = expect(TokenType::Newline);
    if (!newLine)
    {
        return tl::unexpected{std::move(newLine).error()};
    }
    result.insert(result.end(), std::move_iterator(stmtList->begin()), std::move_iterator(stmtList->end()));
    return result;
}

tl::expected<std::vector<pylir::IntrVarPtr<pylir::Syntax::SimpleStmt>>, std::string> pylir::Parser::parseStmtList()
{
    std::vector<IntrVarPtr<Syntax::SimpleStmt>> statements;
    while (m_current != m_lexer.end()
           && (m_current->getTokenType() == TokenType::SemiColon || firstInSimpleStmt(m_current->getTokenType())))
    {
        while (m_current != m_lexer.end() && m_current->getTokenType() == TokenType::SemiColon)
        {
            m_current++;
        }
        if (m_current == m_lexer.end())
        {
            return statements;
        }
        auto simpleStmt = parseSimpleStmt();
        if (!simpleStmt)
        {
            return tl::unexpected{std::move(simpleStmt).error()};
        }
        statements.push_back(std::move(*simpleStmt));
        if (m_current == m_lexer.end()
            || m_current->getTokenType() != TokenType::SemiColon)
        {
            return statements;
        }
    }
    return statements;
}

tl::expected<pylir::IntrVarPtr<pylir::Syntax::CompoundStmt>, std::string> pylir::Parser::parseCompoundStmt()
{
    if (m_current == m_lexer.end())
    {
        return tl::unexpected{createDiagnosticsBuilder(m_document->getText().size(), Diag::EXPECTED_N, "statement")
                                  .addLabel(m_document->getText().size(), std::nullopt, Diag::ERROR_COLOUR)
                                  .emitError()};
    }
    switch (m_current->getTokenType())
    {
        case TokenType::IfKeyword:
        {
            auto ifStmt = parseIfStmt();
            if (!ifStmt)
            {
                return tl::unexpected{std::move(ifStmt).error()};
            }
            return std::make_unique<Syntax::IfStmt>(std::move(*ifStmt));
        }
        case TokenType::ForKeyword:
        {
            auto forStmt = parseForStmt();
            if (!forStmt)
            {
                return tl::unexpected{std::move(forStmt).error()};
            }
            return std::make_unique<Syntax::ForStmt>(std::move(*forStmt));
        }
        case TokenType::TryKeyword:
        {
            auto tryStmt = parseTryStmt();
            if (!tryStmt)
            {
                return tl::unexpected{std::move(tryStmt).error()};
            }
            return std::make_unique<Syntax::TryStmt>(std::move(*tryStmt));
        }
        case TokenType::WithKeyword:
        {
            auto withStmt = parseWithStmt();
            if (!withStmt)
            {
                return tl::unexpected{std::move(withStmt).error()};
            }
            return std::make_unique<Syntax::WithStmt>(std::move(*withStmt));
        }
        case TokenType::WhileKeyword:
        {
            auto whileStmt = parseWhileStmt();
            if (!whileStmt)
            {
                return tl::unexpected{std::move(whileStmt).error()};
            }
            return std::make_unique<Syntax::WhileStmt>(std::move(*whileStmt));
        }
        case TokenType::DefKeyword:
        {
            auto funcDef = parseFuncDef({}, std::nullopt);
            if (!funcDef)
            {
                return tl::unexpected{std::move(funcDef).error()};
            }
            return std::make_unique<Syntax::FuncDef>(std::move(*funcDef));
        }
        case TokenType::ClassKeyword:
        {
            auto classDef = parseClassDef({});
            if (!classDef)
            {
                return tl::unexpected{std::move(classDef).error()};
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
                    return tl::unexpected{std::move(assignment).error()};
                }
                auto newline = expect(TokenType::Newline);
                if (!newline)
                {
                    return tl::unexpected{std::move(newline).error()};
                }
                decorators.push_back({at, std::move(*assignment), *newline});
            } while (m_current != m_lexer.end() && m_current->getTokenType() == TokenType::AtSign);
            if (m_current != m_lexer.end() && m_current->getTokenType() == TokenType::AsyncKeyword)
            {
                auto async = *m_current++;
                auto func = parseFuncDef(std::move(decorators), std::move(async));
                if (!func)
                {
                    return tl::unexpected{std::move(func).error()};
                }
                return std::make_unique<Syntax::FuncDef>(std::move(*func));
            }
            if (m_current == m_lexer.end())
            {
                return tl::unexpected{
                    createDiagnosticsBuilder(m_document->getText().size(), Diag::EXPECTED_N, "class or function")
                        .addLabel(m_document->getText().size(), std::nullopt, Diag::ERROR_COLOUR)
                        .emitError()};
            }
            switch (m_current->getTokenType())
            {
                case TokenType::DefKeyword:
                {
                    auto func = parseFuncDef(std::move(decorators), std::nullopt);
                    if (!func)
                    {
                        return tl::unexpected{std::move(func).error()};
                    }
                    return std::make_unique<Syntax::FuncDef>(std::move(*func));
                }
                case TokenType::ClassKeyword:
                {
                    auto clazz = parseClassDef(std::move(decorators));
                    if (!clazz)
                    {
                        return tl::unexpected{std::move(clazz).error()};
                    }
                    return std::make_unique<Syntax::ClassDef>(std::move(*clazz));
                }
                case TokenType::SyntaxError: return tl::unexpected{pylir::get<std::string>(m_current->getValue())};
                default:
                {
                    return tl::unexpected{createDiagnosticsBuilder(*m_current, Diag::EXPECTED_N_INSTEAD_OF_N,
                                                                   "class or function", m_current->getTokenType())
                                              .addLabel(*m_current, std::nullopt, Diag::ERROR_COLOUR)
                                              .emitError()};
                }
            }
        }
        case TokenType::AsyncKeyword:
        {
            auto async = *m_current++;
            if (m_current == m_lexer.end())
            {
                return tl::unexpected{createDiagnosticsBuilder(m_document->getText().size(), Diag::EXPECTED_N,
                                                               "'for', 'with' or function")
                                          .addLabel(m_document->getText().size(), std::nullopt, Diag::ERROR_COLOUR)
                                          .emitError()};
            }
            switch (m_current->getTokenType())
            {
                case TokenType::DefKeyword:
                {
                    auto func = parseFuncDef({}, async);
                    if (!func)
                    {
                        return tl::unexpected{std::move(func).error()};
                    }
                    return std::make_unique<Syntax::FuncDef>(std::move(*func));
                }
                case TokenType::ForKeyword:
                {
                    auto forStmt = parseForStmt();
                    if (!forStmt)
                    {
                        return tl::unexpected{std::move(forStmt).error()};
                    }
                    forStmt->maybeAsyncKeyword = async;
                    return std::make_unique<Syntax::ForStmt>(std::move(*forStmt));
                }
                case TokenType::WithKeyword:
                {
                    auto withStmt = parseWithStmt();
                    if (!withStmt)
                    {
                        return tl::unexpected{std::move(withStmt).error()};
                    }
                    withStmt->maybeAsyncKeyword = async;
                    return std::make_unique<Syntax::WithStmt>(std::move(*withStmt));
                }
                case TokenType::SyntaxError: return tl::unexpected{pylir::get<std::string>(m_current->getValue())};
                default:
                {
                    return tl::unexpected{createDiagnosticsBuilder(*m_current, Diag::EXPECTED_N_INSTEAD_OF_N,
                                                                   "'for', 'with' or function",
                                                                   m_current->getTokenType())
                                              .addLabel(*m_current, std::nullopt, Diag::ERROR_COLOUR)
                                              .emitError()};
                }
            }
        }
        case TokenType::SyntaxError: return tl::unexpected{pylir::get<std::string>(m_current->getValue())};
        default:
        {
            return tl::unexpected{createDiagnosticsBuilder(*m_current, Diag::EXPECTED_N_INSTEAD_OF_N, "statement",
                                                           m_current->getTokenType())
                                      .addLabel(*m_current, std::nullopt, Diag::ERROR_COLOUR)
                                      .emitError()};
        }
    }
}

tl::expected<pylir::Syntax::IfStmt::Else, std::string> pylir::Parser::parseElse()
{
    auto elseKeyowrd = *m_current++;
    auto elseColon = expect(TokenType::Colon);
    if (!elseColon)
    {
        return tl::unexpected{std::move(elseColon).error()};
    }
    auto elseSuite = parseSuite();
    if (!elseSuite)
    {
        return tl::unexpected{std::move(elseSuite).error()};
    }
    return Syntax::IfStmt::Else{elseKeyowrd, *elseColon, std::make_unique<Syntax::Suite>(std::move(*elseSuite))};
}

tl::expected<pylir::Syntax::IfStmt, std::string> pylir::Parser::parseIfStmt()
{
    auto ifKeyword = expect(TokenType::IfKeyword);
    if (!ifKeyword)
    {
        return tl::unexpected{std::move(ifKeyword).error()};
    }
    auto assignment = parseAssignmentExpression();
    if (!assignment)
    {
        return tl::unexpected{std::move(assignment).error()};
    }
    auto colon = expect(TokenType::Colon);
    if (!colon)
    {
        return tl::unexpected{std::move(colon).error()};
    }
    auto suite = parseSuite();
    if (!suite)
    {
        return tl::unexpected{std::move(suite).error()};
    }
    std::vector<Syntax::IfStmt::Elif> elifs;
    while (m_current != m_lexer.end() && m_current->getTokenType() == TokenType::ElifKeyword)
    {
        auto elif = *m_current++;
        auto condition = parseAssignmentExpression();
        if (!condition)
        {
            return tl::unexpected{std::move(condition).error()};
        }
        auto elifColon = expect(TokenType::Colon);
        if (!elifColon)
        {
            return tl::unexpected{std::move(elifColon).error()};
        }
        auto elIfSuite = parseSuite();
        if (!elIfSuite)
        {
            return tl::unexpected{std::move(elIfSuite).error()};
        }
        elifs.push_back(
            {elif, std::move(*condition), *elifColon, std::make_unique<Syntax::Suite>(std::move(*elIfSuite))});
    }
    std::optional<Syntax::IfStmt::Else> elseSection;
    if (m_current != m_lexer.end() && m_current->getTokenType() == TokenType::ElseKeyword)
    {
        auto parsedElse = parseElse();
        if (!parsedElse)
        {
            return tl::unexpected{std::move(parsedElse).error()};
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

tl::expected<pylir::Syntax::WhileStmt, std::string> pylir::Parser::parseWhileStmt()
{
    auto whileKeyword = expect(TokenType::WhileKeyword);
    if (!whileKeyword)
    {
        return tl::unexpected{std::move(whileKeyword).error()};
    }
    auto condition = parseAssignmentExpression();
    if (!condition)
    {
        return tl::unexpected{std::move(condition).error()};
    }
    auto colon = expect(TokenType::Colon);
    if (!colon)
    {
        return tl::unexpected{std::move(colon).error()};
    }
    std::optional reset = pylir::ValueReset(m_inLoop);
    m_inLoop = true;
    auto suite = parseSuite();
    if (!suite)
    {
        return tl::unexpected{std::move(suite).error()};
    }
    std::optional<Syntax::IfStmt::Else> elseSection;
    if (m_current != m_lexer.end() && m_current->getTokenType() == TokenType::ElseKeyword)
    {
        reset.reset();
        auto parsedElse = parseElse();
        if (!parsedElse)
        {
            return tl::unexpected{std::move(parsedElse).error()};
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

tl::expected<pylir::Syntax::ForStmt, std::string> pylir::Parser::parseForStmt()
{
    auto forKeyword = expect(TokenType::ForKeyword);
    if (!forKeyword)
    {
        return tl::unexpected{std::move(forKeyword).error()};
    }
    auto targetList = parseTargetList(*forKeyword);
    if (!targetList)
    {
        return tl::unexpected{std::move(targetList).error()};
    }
    addToNamespace(**targetList);
    auto inKeyword = expect(TokenType::InKeyword);
    if (!inKeyword)
    {
        return tl::unexpected{std::move(inKeyword).error()};
    }
    auto expressionList = parseExpressionList();
    if (!expressionList)
    {
        return tl::unexpected{std::move(expressionList).error()};
    }
    auto colon = expect(TokenType::Colon);
    if (!colon)
    {
        return tl::unexpected{std::move(colon).error()};
    }
    std::optional reset = pylir::ValueReset(m_inLoop);
    m_inLoop = true;
    auto suite = parseSuite();
    if (!suite)
    {
        return tl::unexpected{std::move(suite).error()};
    }
    std::optional<Syntax::IfStmt::Else> elseSection;
    if (m_current != m_lexer.end() && m_current->getTokenType() == TokenType::ElseKeyword)
    {
        reset.reset();
        auto parsedElse = parseElse();
        if (!parsedElse)
        {
            return tl::unexpected{std::move(parsedElse).error()};
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

tl::expected<pylir::Syntax::TryStmt, std::string> pylir::Parser::parseTryStmt()
{
    auto tryKeyword = expect(TokenType::TryKeyword);
    if (!tryKeyword)
    {
        return tl::unexpected{std::move(tryKeyword).error()};
    }
    auto colon = expect(TokenType::Colon);
    if (!colon)
    {
        return tl::unexpected{std::move(colon).error()};
    }
    auto suite = parseSuite();
    if (!suite)
    {
        return tl::unexpected{std::move(suite).error()};
    }
    if (m_current != m_lexer.end() && m_current->getTokenType() == TokenType::FinallyKeyword)
    {
        auto finallyKeyword = *m_current++;
        auto finallyColon = expect(TokenType::Colon);
        if (!finallyColon)
        {
            return tl::unexpected{std::move(finallyColon).error()};
        }
        auto finallySuite = parseSuite();
        if (!finallySuite)
        {
            return tl::unexpected{std::move(finallySuite).error()};
        }
        return Syntax::TryStmt{{},
                               *tryKeyword,
                               *colon,
                               std::make_unique<Syntax::Suite>(std::move(*suite)),
                               {},
                               std::nullopt,
                               std::nullopt,
                               Syntax::TryStmt::Finally{finallyKeyword, *finallyColon,
                                                        std::make_unique<Syntax::Suite>(std::move(*finallySuite))}};
    }

    std::optional<Syntax::TryStmt::ExceptAll> catchAll;
    std::vector<Syntax::TryStmt::ExceptArgs> exceptSections;
    do
    {
        auto exceptKeyword = expect(TokenType::ExceptKeyword);
        if (!exceptKeyword)
        {
            return tl::unexpected{std::move(exceptKeyword).error()};
        }
        if (catchAll)
        {
            return tl::unexpected{
                createDiagnosticsBuilder(catchAll->exceptKeyword, Diag::EXCEPT_CLAUSE_WITHOUT_EXPRESSION_MUST_COME_LAST)
                    .addLabel(catchAll->exceptKeyword, std::nullopt, Diag::ERROR_COLOUR, Diag::emphasis::bold)
                    .emitError()};
        }
        if (m_current == m_lexer.end() || m_current->getTokenType() == TokenType::Colon)
        {
            auto exceptColon = expect(TokenType::Colon);
            if (!exceptColon)
            {
                return tl::unexpected{std::move(exceptColon).error()};
            }
            auto exceptSuite = parseSuite();
            if (!exceptSuite)
            {
                return tl::unexpected{std::move(exceptSuite).error()};
            }
            catchAll = {*exceptKeyword, *exceptColon, std::make_unique<Syntax::Suite>(std::move(*exceptSuite))};
            continue;
        }
        auto expression = parseExpression();
        if (!expression)
        {
            return tl::unexpected{std::move(expression).error()};
        }
        std::optional<IdentifierToken> name;
        if (m_current != m_lexer.end() && m_current->getTokenType() == TokenType::AsKeyword)
        {
            auto asKeyword = *m_current++;
            auto id = expect(TokenType::Identifier);
            if (!id)
            {
                return tl::unexpected{std::move(id).error()};
            }
            addToNamespace(*id);
            name.emplace(std::move(*id));
        }
        auto exceptColon = expect(TokenType::Colon);
        if (!exceptColon)
        {
            return tl::unexpected{std::move(exceptColon).error()};
        }
        auto exceptSuite = parseSuite();
        if (!exceptSuite)
        {
            return tl::unexpected{std::move(exceptSuite).error()};
        }
        exceptSections.push_back({*exceptKeyword, std::move(*expression), std::move(name), *exceptColon,
                                  std::make_unique<Syntax::Suite>(std::move(*exceptSuite))});
    } while (m_current != m_lexer.end() && m_current->getTokenType() == TokenType::ExceptKeyword);

    std::optional<Syntax::IfStmt::Else> elseSection;
    if (m_current != m_lexer.end() && m_current->getTokenType() == TokenType::ElseKeyword)
    {
        auto parsedElse = parseElse();
        if (!parsedElse)
        {
            return tl::unexpected{std::move(parsedElse).error()};
        }
        elseSection = std::move(*parsedElse);
    }

    std::optional<Syntax::TryStmt::Finally> finally;
    if (m_current != m_lexer.end() && m_current->getTokenType() == TokenType::FinallyKeyword)
    {
        auto finallyKeyword = *m_current++;
        auto finallyColon = expect(TokenType::Colon);
        if (!finallyColon)
        {
            return tl::unexpected{std::move(finallyColon).error()};
        }
        auto finallySuite = parseSuite();
        if (!finallySuite)
        {
            return tl::unexpected{std::move(finallySuite).error()};
        }
        finally = Syntax::TryStmt::Finally{finallyKeyword, *finallyColon,
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

tl::expected<pylir::Syntax::WithStmt, std::string> pylir::Parser::parseWithStmt()
{
    auto withKeyword = expect(TokenType::WithKeyword);
    if (!withKeyword)
    {
        return tl::unexpected{std::move(withKeyword).error()};
    }

    bool first = true;
    std::vector<Syntax::WithStmt::WithItem> withItems;
    while (first || (m_current != m_lexer.end() && m_current->getTokenType() == TokenType::Comma))
    {
        if (first)
        {
            first = false;
        }
        else
        {
            m_current++;
        }
        auto expression = parseExpression();
        if (!expression)
        {
            return tl::unexpected{std::move(expression).error()};
        }
        IntrVarPtr<Syntax::Target> name;
        if (m_current != m_lexer.end() && m_current->getTokenType() == TokenType::AsKeyword)
        {
            auto asKeyword = *m_current++;
            auto target = parseTarget(asKeyword);
            if (!target)
            {
                return tl::unexpected{std::move(target).error()};
            }
            name = std::move(*target);
        }
        withItems.push_back({std::move(*expression), std::move(name)});
    }

    auto colon = expect(TokenType::Colon);
    if (!colon)
    {
        return tl::unexpected{std::move(colon).error()};
    }
    auto suite = parseSuite();
    if (!suite)
    {
        return tl::unexpected{std::move(suite).error()};
    }
    return Syntax::WithStmt{{},           std::nullopt,
                            *withKeyword, std::move(withItems),
                            *colon,       std::make_unique<Syntax::Suite>(std::move(*suite))};
}

tl::expected<pylir::Syntax::Suite, std::string> pylir::Parser::parseSuite()
{
    decltype(Syntax::Suite::statements) statements;
    if (m_current != m_lexer.end() && m_current->getTokenType() == TokenType::Newline)
    {
        m_current++;
        if (m_current == m_lexer.end() || m_current->getTokenType() != TokenType::Indent)
        {
            // stmt_list was empty, and hence a newline immediately followed with no indent after.
            return Syntax::Suite{};
        }

        m_current++;
        do
        {
            auto statement = parseStatement();
            if (!statement)
            {
                return tl::unexpected{std::move(statement).error()};
            }
            statements.insert(statements.end(), std::move_iterator(statement->begin()),
                              std::move_iterator(statement->end()));
        } while (m_current != m_lexer.end() && m_current->getTokenType() != TokenType::Dedent);
        auto dedent = expect(TokenType::Dedent);
        if (!dedent)
        {
            return tl::unexpected{std::move(dedent).error()};
        }
        return Syntax::Suite{std::move(statements)};
    }

    auto statementList = parseStmtList();
    if (!statementList)
    {
        return tl::unexpected{std::move(statementList).error()};
    }
    auto newline = expect(TokenType::Newline);
    if (!newline)
    {
        return tl::unexpected{std::move(newline).error()};
    }
    statements.insert(statements.end(), std::move_iterator(statementList->begin()),
                      std::move_iterator(statementList->end()));
    return Syntax::Suite{std::move(statements)};
}

tl::expected<std::vector<pylir::Syntax::Parameter>, std::string> pylir::Parser::parseParameterList()
{
    std::vector<pylir::Syntax::Parameter> parameters;
    Syntax::Parameter::Kind currentKind = Syntax::Parameter::Normal;

    std::optional<BaseToken> seenPositionalOnly;
    std::optional<std::size_t> seenDefaultParam;
    std::variant<std::monostate, std::size_t, BaseToken> seenPosRest;
    std::optional<std::size_t> seenKwRest;
    while (parameters.empty() || (m_current != m_lexer.end() && m_current->getTokenType() == TokenType::Comma))
    {
        if (!parameters.empty())
        {
            m_current++;
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
                    return tl::unexpected{
                        createDiagnosticsBuilder(*m_current,
                                                 Diag::AT_LEAST_ONE_PARAMETER_REQUIRED_BEFORE_POSITIONAL_ONLY_INDICATOR)
                            .addLabel(*m_current, std::nullopt, Diag::ERROR_COLOUR)
                            .emitError()};
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
                return tl::unexpected{
                    createDiagnosticsBuilder(*m_current, Diag::POSITIONAL_ONLY_INDICATOR_MAY_ONLY_APPEAR_ONCE)
                        .addLabel(*m_current, std::nullopt, Diag::ERROR_COLOUR)
                        .addNote(*seenPositionalOnly, Diag::PREVIOUS_OCCURRENCE_HERE)
                        .addLabel(*seenPositionalOnly, std::nullopt, Diag::NOTE_COLOUR)
                        .emitError()};
            }
            case TokenType::Star:
                stars = *m_current++;
                if (currentKind == Syntax::Parameter::Normal)
                {
                    currentKind = Syntax::Parameter::KeywordOnly;
                    if (m_current == m_lexer.end())
                    {
                        return parameters;
                    }
                    if (m_current->getTokenType() != TokenType::Comma
                        && m_current->getTokenType() != TokenType::Identifier)
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
            return tl::unexpected{std::move(identifier).error()};
        }

        IntrVarPtr<Syntax::Expression> maybeType;
        IntrVarPtr<Syntax::Expression> maybeDefault;
        if (m_current != m_lexer.end() && m_current->getTokenType() == TokenType::Colon)
        {
            m_current++;
            auto type = parseExpression();
            if (!type)
            {
                return tl::unexpected{std::move(type).error()};
            }
            maybeType = std::move(*type);
        }
        if (!stars && m_current != m_lexer.end() && m_current->getTokenType() == TokenType::Assignment)
        {
            m_current++;
            auto defaultVal = parseExpression();
            if (!defaultVal)
            {
                return tl::unexpected{std::move(defaultVal).error()};
            }
            maybeDefault = std::move(*defaultVal);
        }
        if (!maybeDefault && seenDefaultParam)
        {
            return tl::unexpected{
                createDiagnosticsBuilder(
                    *identifier, Diag::NO_DEFAULT_ARGUMENT_FOR_PARAMETER_N_FOLLOWING_PARAMETERS_WITH_DEFAULT_ARGUMENTS,
                    pylir::get<std::string>(identifier->getValue()))
                    .addLabel(*identifier, std::nullopt, Diag::ERROR_COLOUR)
                    .addNote(parameters[*seenDefaultParam], Diag::PARAMETER_N_WITH_DEFAULT_ARGUMENT_HERE,
                             parameters[*seenDefaultParam].name.getValue())
                    .addLabel(parameters[*seenDefaultParam], std::nullopt, Diag::NOTE_COLOUR)
                    .emitError()};
        }
        if (maybeDefault)
        {
            seenDefaultParam = parameters.size();
        }
        if (seenKwRest)
        {
            return tl::unexpected{
                createDiagnosticsBuilder(*identifier, Diag::NO_MORE_PARAMETERS_ALLOWED_AFTER_EXCESS_KEYWORD_PARAMETER_N,
                                         pylir::get<std::string>(identifier->getValue()),
                                         parameters[*seenKwRest].name.getValue())
                    .addLabel(*identifier, std::nullopt, Diag::ERROR_COLOUR)
                    .addNote(parameters[*seenKwRest], Diag::EXCESS_KEYWORD_PARAMETER_N_HERE,
                             parameters[*seenKwRest].name.getValue())
                    .addLabel(parameters[*seenKwRest], std::nullopt, Diag::NOTE_COLOUR)
                    .emitError()};
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
                return tl::unexpected{createDiagnosticsBuilder(
                                          *identifier, Diag::STARRED_PARAMETER_NOT_ALLOWED_AFTER_KEYWORD_ONLY_INDICATOR)
                                          .addLabel(*identifier, std::nullopt, Diag::ERROR_COLOUR)
                                          .addNote(*token, Diag::KEYWORD_ONLY_INDICATOR_HERE)
                                          .addLabel(*token, std::nullopt, Diag::NOTE_COLOUR)
                                          .emitError()};
            }
            if (auto* index = std::get_if<std::size_t>(&seenPosRest))
            {
                return tl::unexpected{
                    createDiagnosticsBuilder(*identifier, Diag::ONLY_ONE_STARRED_PARAMETER_ALLOWED)
                        .addLabel(*identifier, std::nullopt, Diag::ERROR_COLOUR)
                        .addNote(parameters[*index], Diag::STARRED_PARAMETER_N_HERE, parameters[*index].name.getValue())
                        .addLabel(parameters[*index], std::nullopt, Diag::NOTE_COLOUR)
                        .emitError()};
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
        if (error)
        {
            return;
        }

        for (auto& iter : def.nonLocalVariables)
        {
            if (std::none_of(scopes.begin(), scopes.end(),
                             [&](const pylir::IdentifierSet* set) -> bool { return set->count(iter); }))
            {
                error = onError(iter);
                break;
            }
        }

        for (auto& iter : def.unknown)
        {
            if (std::any_of(scopes.begin(), scopes.end(),
                            [&](const pylir::IdentifierSet* set) -> bool { return set->count(iter); })
                || globals.count(iter))
            {
                def.nonLocalVariables.insert(iter);
            }
        }
        def.unknown.clear();

        // add any non locals from nested functions except if they are local to this function aka the referred
        // to local
        for (auto& iter : def.nonLocalVariables)
        {
            pylir::match(
                parentDef,
                [&](std::monostate)
                {
                    // Any non locals going up that haven't caused an error are from the very top level function at
                    // global scope
                    closures.insert(iter);
                },
                [&](pylir::Syntax::FuncDef* funcDef)
                {
                    if (auto result = funcDef->localVariables.find(iter); result == funcDef->localVariables.end())
                    {
                        funcDef->nonLocalVariables.insert(iter);
                    }
                    else
                    {
                        funcDef->localVariables.erase(result);
                        funcDef->closures.insert(iter);
                    }
                },
                [&](pylir::Syntax::ClassDef* classDef) { classDef->nonLocalVariables.insert(iter); });
        }
    }

    std::vector<const pylir::IdentifierSet*> scopes;
    std::function<std::string(const pylir::IdentifierToken&)> onError;
    std::variant<std::monostate, pylir::Syntax::FuncDef*, pylir::Syntax::ClassDef*> parentDef;
    const pylir::IdentifierSet& globals;

public:
    pylir::IdentifierSet closures;
    std::optional<std::string> error;

    NamespaceVisitor(std::vector<const pylir::IdentifierSet*>&& scopes,
                     std::function<std::string(const pylir::IdentifierToken&)>&& onError,
                     const pylir::IdentifierSet& globals)
        : scopes(std::move(scopes)), onError(std::move(onError)), globals(globals)
    {
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
        scopes.push_back(&funcDef.localVariables);
        auto exit = llvm::make_scope_exit([&] { scopes.pop_back(); });
        auto& def = const_cast<pylir::Syntax::FuncDef&>(funcDef);
        {
            pylir::ValueReset reset(parentDef);
            parentDef = &def;
            Visitor::visit(funcDef);
        }
        finishNamespace(def);
    }
};

} // namespace

tl::expected<pylir::IdentifierSet, std::string>
    pylir::Parser::finishNamespace(pylir::Syntax::Suite& suite, const IdentifierSet& nonLocals,
                                   std::vector<const pylir::IdentifierSet*> scopes)
{
    if (auto first = nonLocals.begin(); first != nonLocals.end())
    {
        return tl::unexpected{
            createDiagnosticsBuilder(*first, Diag::COULD_NOT_FIND_VARIABLE_N_IN_OUTER_SCOPES, first->getValue())
                .addLabel(*first, std::nullopt, Diag::ERROR_COLOUR)
                .emitError()};
    }
    NamespaceVisitor visitor{std::move(scopes),
                             [&](const IdentifierToken& token)
                             {
                                 return createDiagnosticsBuilder(token, Diag::COULD_NOT_FIND_VARIABLE_N_IN_OUTER_SCOPES,
                                                                 token.getValue())
                                     .addLabel(token, std::nullopt, Diag::ERROR_COLOUR)
                                     .emitError();
                             },
                             m_globals};
    visitor.visit(suite);
    if (visitor.error)
    {
        return tl::unexpected{std::move(*visitor.error)};
    }
    return std::move(visitor.closures);
}

tl::expected<pylir::Syntax::FuncDef, std::string>
    pylir::Parser::parseFuncDef(std::vector<Syntax::Decorator>&& decorators, std::optional<BaseToken>&& asyncKeyword)
{
    auto defKeyword = expect(TokenType::DefKeyword);
    if (!defKeyword)
    {
        return tl::unexpected{std::move(defKeyword).error()};
    }
    auto funcName = expect(TokenType::Identifier);
    if (!funcName)
    {
        return tl::unexpected{std::move(funcName).error()};
    }
    addToNamespace(*funcName);
    auto openParenth = expect(TokenType::OpenParentheses);
    if (!openParenth)
    {
        return tl::unexpected{std::move(openParenth).error()};
    }
    std::vector<Syntax::Parameter> parameterList;
    if (m_current != m_lexer.end() && m_current->getTokenType() != TokenType::CloseParentheses)
    {
        auto parsedParameterList = parseParameterList();
        if (!parsedParameterList)
        {
            return tl::unexpected{std::move(parsedParameterList).error()};
        }
        parameterList = std::move(*parsedParameterList);
    }
    auto closeParenth = expect(TokenType::CloseParentheses);
    if (!closeParenth)
    {
        return tl::unexpected{std::move(closeParenth).error()};
    }
    IntrVarPtr<Syntax::Expression> suffix;
    if (m_current != m_lexer.end() && m_current->getTokenType() == TokenType::Arrow)
    {
        m_current++;
        auto expression = parseExpression();
        if (!expression)
        {
            return tl::unexpected{std::move(expression).error()};
        }
        suffix = std::move(*expression);
    }
    auto colon = expect(TokenType::Colon);
    if (!colon)
    {
        return tl::unexpected{std::move(colon).error()};
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
    IdentifierSet locals;
    IdentifierSet nonLocals;
    IdentifierSet closures;
    IdentifierSet unknowns;

    for (auto& [token, kind] : m_namespace.back().identifiers)
    {
        switch (kind)
        {
            case Scope::Kind::Local: locals.insert(token); break;
            case Scope::Kind::NonLocal: nonLocals.insert(token); break;
            case Scope::Kind::Unknown: unknowns.insert(token); break;
            default: break;
        }
    }

    exit.reset();
    if (!suite)
    {
        return tl::unexpected{std::move(suite).error()};
    }

    if (m_namespace.empty())
    {
        // this indicates that this funcdef is at global scope or only nested in classes. We now need to resolve any
        // nonlocals inside any nested funcDefs and figure out whether any unknowns are nonlocal or global

        // Unknowns of this funcDef can't be nonlocal as only variables from the global namespace could be in use. Clear
        // it. CodeGen will issue NameErrors if need be
        unknowns.clear();

        auto error = finishNamespace(*suite, nonLocals, {&locals});
        if (!error)
        {
            return tl::unexpected{std::move(error.error())};
        }
        closures = std::move(*error);
        for (const auto& iter : closures)
        {
            locals.erase(iter);
        }
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
                           std::move(locals),
                           std::move(nonLocals),
                           std::move(closures),
                           std::move(unknowns)};
}

tl::expected<pylir::Syntax::ClassDef, std::string>
    pylir::Parser::parseClassDef(std::vector<Syntax::Decorator>&& decorators)
{
    auto classKeyword = expect(TokenType::ClassKeyword);
    if (!classKeyword)
    {
        return tl::unexpected{std::move(classKeyword).error()};
    }
    auto className = expect(TokenType::Identifier);
    if (!className)
    {
        return tl::unexpected{std::move(className).error()};
    }
    addToNamespace(*className);
    std::optional<Syntax::ClassDef::Inheritance> inheritance;
    if (m_current != m_lexer.end() && m_current->getTokenType() == TokenType::OpenParentheses)
    {
        auto open = *m_current++;
        std::vector<Syntax::Argument> argumentList;
        if (m_current != m_lexer.end() && m_current->getTokenType() != TokenType::CloseParentheses)
        {
            auto parsedArgumentList = parseArgumentList();
            if (!parsedArgumentList)
            {
                return tl::unexpected{std::move(parsedArgumentList).error()};
            }
            argumentList = std::move(*parsedArgumentList);
        }
        auto close = expect(TokenType::CloseParentheses);
        if (!close)
        {
            return tl::unexpected{std::move(close).error()};
        }
        inheritance = Syntax::ClassDef::Inheritance{open, std::move(argumentList), *close};
    }
    auto colon = expect(TokenType::Colon);
    if (!colon)
    {
        return tl::unexpected{std::move(colon).error()};
    }
    m_namespace.emplace_back();
    std::optional exit = llvm::make_scope_exit([&] { m_namespace.pop_back(); });
    m_namespace.back().classScope = true;
    pylir::ValueReset resetLoop(m_inLoop, m_inLoop);
    pylir::ValueReset resetFunc(m_inFunc, m_inFunc);
    m_inLoop = false;
    m_inFunc = false;
    auto suite = parseSuite();
    IdentifierSet nonLocals;
    IdentifierSet locals;
    IdentifierSet unknowns;

    for (auto& [token, kind] : m_namespace.back().identifiers)
    {
        switch (kind)
        {
            case Scope::Kind::NonLocal: nonLocals.insert(token); break;
            case Scope::Kind::Local: locals.insert(token); break;
            case Scope::Kind::Unknown: unknowns.insert(token); break;
            default: break;
        }
    }

    if (!suite)
    {
        return tl::unexpected{std::move(suite).error()};
    }
    exit.reset();
    if (m_namespace.empty())
    {
        // unknowns can't be nonlocal because locals don't exist at global scope anymore
        unknowns.clear();
        if (auto error = finishNamespace(*suite, nonLocals); !error)
        {
            return tl::unexpected{std::move(error.error())};
        }
    }
    return Syntax::ClassDef{{},
                            std::move(decorators),
                            *classKeyword,
                            IdentifierToken{std::move(*className)},
                            std::move(inheritance),
                            *colon,
                            std::make_unique<Syntax::Suite>(std::move(*suite)),
                            std::move(locals),
                            std::move(nonLocals),
                            std::move(unknowns)};
}
