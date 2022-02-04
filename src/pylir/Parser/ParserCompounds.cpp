#include "Parser.hpp"

#include <llvm/ADT/ScopeExit.h>

#include <pylir/Diagnostics/DiagnosticMessages.hpp>
#include <pylir/Support/Functional.hpp>
#include <pylir/Support/ValueReset.hpp>

#include "Visitor.hpp"

tl::expected<pylir::Syntax::FileInput, std::string> pylir::Parser::parseFileInput()
{
    std::vector<std::variant<BaseToken, Syntax::Statement>> statements;
    while (true)
    {
        for (; m_current != m_lexer.end() && m_current->getTokenType() == TokenType::Newline; m_current++)
        {
            statements.emplace_back(*m_current);
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
        statements.emplace_back(std::move(*statement));
    }
    return Syntax::FileInput{std::move(statements), {m_globals.begin(), m_globals.end()}};
}

tl::expected<pylir::Syntax::Statement, std::string> pylir::Parser::parseStatement()
{
    if (m_current != m_lexer.end() && Syntax::firstInCompoundStmt(m_current->getTokenType()))
    {
        auto compound = parseCompoundStmt();
        if (!compound)
        {
            return tl::unexpected{std::move(compound).error()};
        }
        return Syntax::Statement{std::move(*compound)};
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
    return Syntax::Statement{Syntax::Statement::SingleLine{std::move(*stmtList), *newLine}};
}

tl::expected<pylir::Syntax::StmtList, std::string> pylir::Parser::parseStmtList()
{
    return parseCommaList(pylir::bind_front(&Parser::parseSimpleStmt, this), Syntax::firstInSimpleStmt, std::nullopt,
                          TokenType::SemiColon);
}

tl::expected<pylir::Syntax::CompoundStmt, std::string> pylir::Parser::parseCompoundStmt()
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
            return Syntax::CompoundStmt{std::move(*ifStmt)};
        }
        case TokenType::ForKeyword:
        {
            auto forStmt = parseForStmt();
            if (!forStmt)
            {
                return tl::unexpected{std::move(forStmt).error()};
            }
            return Syntax::CompoundStmt{std::move(*forStmt)};
        }
        case TokenType::TryKeyword:
        {
            auto tryStmt = parseTryStmt();
            if (!tryStmt)
            {
                return tl::unexpected{std::move(tryStmt).error()};
            }
            return Syntax::CompoundStmt{std::move(*tryStmt)};
        }
        case TokenType::WithKeyword:
        {
            auto withStmt = parseWithStmt();
            if (!withStmt)
            {
                return tl::unexpected{std::move(withStmt).error()};
            }
            return Syntax::CompoundStmt{std::move(*withStmt)};
        }
        case TokenType::WhileKeyword:
        {
            auto whileStmt = parseWhileStmt();
            if (!whileStmt)
            {
                return tl::unexpected{std::move(whileStmt).error()};
            }
            return Syntax::CompoundStmt{std::move(*whileStmt)};
        }
        case TokenType::DefKeyword:
        {
            auto funcDef = parseFuncDef({}, std::nullopt);
            if (!funcDef)
            {
                return tl::unexpected{std::move(funcDef).error()};
            }
            return Syntax::CompoundStmt{std::move(*funcDef)};
        }
        case TokenType::ClassKeyword:
        {
            auto classDef = parseClassDef({});
            if (!classDef)
            {
                return tl::unexpected{std::move(classDef).error()};
            }
            return Syntax::CompoundStmt{std::move(*classDef)};
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
                return Syntax::CompoundStmt{std::move(*func)};
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
                    return Syntax::CompoundStmt{std::move(*func)};
                }
                case TokenType::ClassKeyword:
                {
                    auto func = parseClassDef(std::move(decorators));
                    if (!func)
                    {
                        return tl::unexpected{std::move(func).error()};
                    }
                    return Syntax::CompoundStmt{std::move(*func)};
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
                    return Syntax::CompoundStmt{std::move(*func)};
                }
                case TokenType::ForKeyword:
                {
                    auto forStmt = parseForStmt();
                    if (!forStmt)
                    {
                        return tl::unexpected{std::move(forStmt).error()};
                    }
                    return Syntax::CompoundStmt{Syntax::AsyncForStmt{async, std::move(*forStmt)}};
                }
                case TokenType::WithKeyword:
                {
                    auto withStmt = parseWithStmt();
                    if (!withStmt)
                    {
                        return tl::unexpected{std::move(withStmt).error()};
                    }
                    return Syntax::CompoundStmt{Syntax::AsyncWithStmt{async, std::move(*withStmt)}};
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
    return Syntax::IfStmt{*ifKeyword,       std::move(*assignment),
                          *colon,           std::make_unique<Syntax::Suite>(std::move(*suite)),
                          std::move(elifs), std::move(elseSection)};
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
    return Syntax::WhileStmt{*whileKeyword, std::move(*condition), *colon,
                             std::make_unique<Syntax::Suite>(std::move(*suite)), std::move(elseSection)};
}

tl::expected<pylir::Syntax::ForStmt, std::string> pylir::Parser::parseForStmt()
{
    auto forKeyword = expect(TokenType::ForKeyword);
    if (!forKeyword)
    {
        return tl::unexpected{std::move(forKeyword).error()};
    }
    auto targetList = parseTargetList();
    if (!targetList)
    {
        return tl::unexpected{std::move(targetList).error()};
    }
    addToNamespace(*targetList);
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
    return Syntax::ForStmt{*forKeyword,
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
        return Syntax::TryStmt{*tryKeyword,
                               *colon,
                               std::make_unique<Syntax::Suite>(std::move(*suite)),
                               {},
                               std::nullopt,
                               Syntax::TryStmt::Finally{finallyKeyword, *finallyColon,
                                                        std::make_unique<Syntax::Suite>(std::move(*finallySuite))}};
    }

    std::optional<Token> catchAll;
    bool catchAllIssuedAlready = false;
    std::vector<Syntax::TryStmt::Except> exceptSections;
    do
    {
        auto exceptKeyword = expect(TokenType::ExceptKeyword);
        if (!exceptKeyword)
        {
            return tl::unexpected{std::move(exceptKeyword).error()};
        }
        if (catchAll && !catchAllIssuedAlready)
        {
            // Don't issue this more than once
            catchAllIssuedAlready = true;
            return tl::unexpected{
                createDiagnosticsBuilder(*catchAll, Diag::EXCEPT_CLAUSE_WITHOUT_EXPRESSION_MUST_COME_LAST)
                    .addLabel(*catchAll, std::nullopt, Diag::ERROR_COLOUR, Diag::emphasis::bold)
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
            exceptSections.push_back(
                {*exceptKeyword, std::nullopt, *exceptColon, std::make_unique<Syntax::Suite>(std::move(*exceptSuite))});
            catchAll = *exceptKeyword;
            continue;
        }
        auto expression = parseExpression();
        if (!expression)
        {
            return tl::unexpected{std::move(expression).error()};
        }
        std::optional<std::pair<BaseToken, IdentifierToken>> name;
        if (m_current != m_lexer.end() && m_current->getTokenType() == TokenType::AsKeyword)
        {
            auto asKeyword = *m_current++;
            auto id = expect(TokenType::Identifier);
            if (!id)
            {
                return tl::unexpected{std::move(id).error()};
            }
            addToNamespace(*id);
            name.emplace(asKeyword, IdentifierToken{*id});
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
        exceptSections.push_back({*exceptKeyword, std::pair{std::move(*expression), std::move(name)}, *exceptColon,
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
    return Syntax::TryStmt{*tryKeyword,
                           *colon,
                           std::make_unique<Syntax::Suite>(std::move(*suite)),
                           std::move(exceptSections),
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
    std::optional<Syntax::WithStmt::WithItem> firstItem;
    {
        auto expression = parseExpression();
        if (!expression)
        {
            return tl::unexpected{std::move(expression).error()};
        }
        std::optional<std::pair<BaseToken, Syntax::Target>> name;
        if (m_current != m_lexer.end() && m_current->getTokenType() == TokenType::AsKeyword)
        {
            auto asKeyword = *m_current++;
            auto target = parseTarget();
            if (!target)
            {
                return tl::unexpected{std::move(target).error()};
            }
            name.emplace(asKeyword, std::move(*target));
        }
        firstItem = Syntax::WithStmt::WithItem{std::move(*expression), std::move(name)};
    }

    std::vector<std::pair<BaseToken, Syntax::WithStmt::WithItem>> withItems;
    while (m_current != m_lexer.end() && m_current->getTokenType() == TokenType::Comma)
    {
        auto comma = *m_current++;
        auto expression = parseExpression();
        if (!expression)
        {
            return tl::unexpected{std::move(expression).error()};
        }
        std::optional<std::pair<BaseToken, Syntax::Target>> name;
        if (m_current != m_lexer.end() && m_current->getTokenType() == TokenType::AsKeyword)
        {
            auto asKeyword = *m_current++;
            auto target = parseTarget();
            if (!target)
            {
                return tl::unexpected{std::move(target).error()};
            }
            name.emplace(asKeyword, std::move(*target));
        }
        withItems.emplace_back(comma, Syntax::WithStmt::WithItem{std::move(*expression), std::move(name)});
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
    return Syntax::WithStmt{*withKeyword, std::move(*firstItem), std::move(withItems), *colon,
                            std::make_unique<Syntax::Suite>(std::move(*suite))};
}

tl::expected<pylir::Syntax::Suite, std::string> pylir::Parser::parseSuite()
{
    if (m_current != m_lexer.end() && m_current->getTokenType() == TokenType::Newline)
    {
        auto newline = *m_current++;
        auto indent = expect(TokenType::Indent);
        if (!indent)
        {
            return tl::unexpected{std::move(indent).error()};
        }
        std::vector<Syntax::Statement> statements;
        do
        {
            auto statement = parseStatement();
            if (!statement)
            {
                return tl::unexpected{std::move(statement).error()};
            }
            statements.push_back(std::move(*statement));
        } while (m_current != m_lexer.end() && m_current->getTokenType() != TokenType::Dedent);
        auto dedent = expect(TokenType::Dedent);
        if (!dedent)
        {
            return tl::unexpected{std::move(dedent).error()};
        }
        return Syntax::Suite{Syntax::Suite::MultiLine{newline, *indent, std::move(statements), *dedent}};
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
    return Syntax::Suite{Syntax::Suite::SingleLine{std::move(*statementList), *newline}};
}

tl::expected<pylir::Syntax::ParameterList, std::string> pylir::Parser::parseParameterList()
{
    auto parseParameter = [&]() -> tl::expected<Syntax::ParameterList::Parameter, std::string>
    {
        auto identifier = expect(TokenType::Identifier);
        if (!identifier)
        {
            return tl::unexpected{std::move(identifier).error()};
        }
        std::optional<std::pair<BaseToken, Syntax::Expression>> annotation;
        if (m_current != m_lexer.end() && m_current->getTokenType() == TokenType::Colon)
        {
            auto colon = *m_current++;
            auto expression = parseExpression();
            if (!expression)
            {
                return tl::unexpected{std::move(expression).error()};
            }
            annotation = std::pair{colon, std::move(*expression)};
        }
        return Syntax::ParameterList::Parameter{IdentifierToken{*identifier}, std::move(annotation)};
    };

    auto parseDefParameter = [&]() -> tl::expected<Syntax::ParameterList::DefParameter, std::string>
    {
        auto parameter = parseParameter();
        if (!parameter)
        {
            return tl::unexpected{std::move(parameter).error()};
        }
        std::optional<std::pair<BaseToken, Syntax::Expression>> defaultArg;
        if (m_current != m_lexer.end() && m_current->getTokenType() == TokenType::Assignment)
        {
            auto assignment = *m_current++;
            auto expression = parseExpression();
            if (!expression)
            {
                return tl::unexpected{std::move(expression).error()};
            }
            defaultArg = std::pair{assignment, std::move(*expression)};
        }
        return Syntax::ParameterList::DefParameter{std::move(*parameter), std::move(defaultArg)};
    };

    auto parseParameterListStarArgs = [&]() -> tl::expected<Syntax::ParameterList::StarArgs, std::string>
    {
        PYLIR_ASSERT(m_current != m_lexer.end());
        if (m_current->getTokenType() == TokenType::PowerOf)
        {
            auto doubleStar = *m_current++;
            auto parameter = parseParameter();
            if (!parameter)
            {
                return tl::unexpected{std::move(parameter).error()};
            }
            std::optional<BaseToken> comma;
            if (m_current != m_lexer.end() && m_current->getTokenType() == TokenType::Comma)
            {
                comma = *m_current++;
            }
            return Syntax::ParameterList::StarArgs{
                Syntax::ParameterList::StarArgs::DoubleStar{doubleStar, std::move(*parameter), comma}};
        }
        PYLIR_ASSERT(m_current->getTokenType() == TokenType::Star);
        auto star = *m_current++;
        std::optional<Syntax::ParameterList::Parameter> parameter;
        if (m_current != m_lexer.end() && m_current->getTokenType() == TokenType::Identifier)
        {
            auto parsedParameter = parseParameter();
            if (!parsedParameter)
            {
                return tl::unexpected{std::move(parsedParameter).error()};
            }
            parameter = std::move(*parsedParameter);
        }
        std::vector<std::pair<BaseToken, Syntax::ParameterList::DefParameter>> defParameters;
        while (lookaheadEquals(std::array{TokenType::Comma, TokenType::Identifier}))
        {
            auto comma = *m_current++;
            auto defParameter = parseDefParameter();
            if (!defParameter)
            {
                return tl::unexpected{std::move(defParameter).error()};
            }
            defParameters.emplace_back(comma, std::move(*defParameter));
        }
        if (!lookaheadEquals(std::array{TokenType::Comma, TokenType::PowerOf}))
        {
            if (m_current != m_lexer.end() && m_current->getTokenType() == TokenType::Comma)
            {
                return Syntax::ParameterList::StarArgs{Syntax::ParameterList::StarArgs::Star{
                    star, std::move(parameter), std::move(defParameters),
                    Syntax::ParameterList::StarArgs::Star::Further{*m_current++, std::nullopt}}};
            }
            return Syntax::ParameterList::StarArgs{Syntax::ParameterList::StarArgs::Star{
                star, std::move(parameter), std::move(defParameters), std::nullopt}};
        }
        auto comma = *m_current++;
        auto powerOf = *m_current++;
        auto expandParameter = parseParameter();
        if (!expandParameter)
        {
            return tl::unexpected{std::move(expandParameter).error()};
        }
        std::optional<BaseToken> trailingComma;
        if (m_current != m_lexer.end() && m_current->getTokenType() == TokenType::Comma)
        {
            trailingComma = *m_current++;
        }
        return Syntax::ParameterList::StarArgs{Syntax::ParameterList::StarArgs::Star{
            star, std::move(parameter), std::move(defParameters),
            Syntax::ParameterList::StarArgs::Star::Further{
                comma,
                Syntax::ParameterList::StarArgs::DoubleStar{powerOf, std::move(*expandParameter), trailingComma}}}};
    };

    std::optional<IdentifierToken> firstDefaultParameter;

    auto parseParameterListNoPosOnlyPrefix = [&]() -> tl::expected<Syntax::ParameterList::NoPosOnly, std::string>
    {
        if (m_current != m_lexer.end()
            && (m_current->getTokenType() == TokenType::Star || m_current->getTokenType() == TokenType::PowerOf))
        {
            auto starArgs = parseParameterListStarArgs();
            if (!starArgs)
            {
                return tl::unexpected{std::move(starArgs).error()};
            }
            return Syntax::ParameterList::NoPosOnly{std::move(*starArgs)};
        }

        auto first = parseDefParameter();
        if (!first)
        {
            return tl::unexpected{std::move(first).error()};
        }
        if (!firstDefaultParameter && first->defaultArg)
        {
            firstDefaultParameter = first->parameter.identifier;
        }
        else if (firstDefaultParameter && !first->defaultArg.has_value())
        {
            return tl::unexpected{
                createDiagnosticsBuilder(
                    *first, Diag::NO_DEFAULT_ARGUMENT_FOR_PARAMETER_N_FOLLOWING_PARAMETERS_WITH_DEFAULT_ARGUMENTS,
                    first->parameter.identifier.getValue())
                    .addLabel(first->parameter.identifier, std::nullopt, Diag::ERROR_COLOUR)
                    .addNote(*firstDefaultParameter, Diag::PARAMETER_N_WITH_DEFAULT_ARGUMENT_HERE,
                             firstDefaultParameter->getValue())
                    .addLabel(*firstDefaultParameter, std::nullopt, Diag::NOTE_COLOUR)
                    .emitError()};
        }
        std::vector<std::pair<BaseToken, Syntax::ParameterList::DefParameter>> defParameters;
        while (lookaheadEquals(std::array{TokenType::Comma, TokenType::Identifier}))
        {
            auto comma = *m_current++;
            auto defParameter = parseDefParameter();
            if (!defParameter)
            {
                return tl::unexpected{std::move(defParameter).error()};
            }
            if (!firstDefaultParameter && defParameter->defaultArg.has_value())
            {
                firstDefaultParameter = defParameter->parameter.identifier;
            }
            else if (firstDefaultParameter && !defParameter->defaultArg.has_value())
            {
                return tl::unexpected{
                    createDiagnosticsBuilder(
                        *defParameter,
                        Diag::NO_DEFAULT_ARGUMENT_FOR_PARAMETER_N_FOLLOWING_PARAMETERS_WITH_DEFAULT_ARGUMENTS,
                        defParameter->parameter.identifier.getValue())
                        .addLabel(defParameter->parameter.identifier, std::nullopt, Diag::ERROR_COLOUR)
                        .addNote(*firstDefaultParameter, Diag::PARAMETER_N_WITH_DEFAULT_ARGUMENT_HERE,
                                 firstDefaultParameter->getValue())
                        .addLabel(*firstDefaultParameter, std::nullopt, Diag::NOTE_COLOUR)
                        .emitError()};
            }
            defParameters.emplace_back(comma, std::move(*defParameter));
        }
        return Syntax::ParameterList::NoPosOnly{
            Syntax::ParameterList::NoPosOnly::DefParams{std::move(*first), std::move(defParameters), std::nullopt}};
    };

    auto prefix = parseParameterListNoPosOnlyPrefix();
    if (!prefix)
    {
        return tl::unexpected{std::move(prefix).error()};
    }
    if (std::holds_alternative<Syntax::ParameterList::StarArgs>(prefix->variant))
    {
        return Syntax::ParameterList{std::move(*prefix)};
    }
    auto& defParams = pylir::get<Syntax::ParameterList::NoPosOnly::DefParams>(prefix->variant);

    if (!lookaheadEquals(std::array{TokenType::Comma, TokenType::Divide}))
    {
        if (m_current == m_lexer.end() || m_current->getTokenType() != TokenType::Comma)
        {
            return Syntax::ParameterList{std::move(*prefix)};
        }
        auto comma = *m_current++;
        std::optional<Syntax::ParameterList::StarArgs> starArgs;
        if (m_current != m_lexer.end() && m_current->getTokenType() == TokenType::Star)
        {
            auto parsedStarArgs = parseParameterListStarArgs();
            if (!parsedStarArgs)
            {
                return tl::unexpected{std::move(parsedStarArgs).error()};
            }
            starArgs = std::move(*parsedStarArgs);
        }
        defParams.suffix = std::pair{comma, std::move(starArgs)};
        return Syntax::ParameterList{std::move(*prefix)};
    }

    auto comma = *m_current++;
    auto slash = *m_current++;
    if (m_current == m_lexer.end() || m_current->getTokenType() != TokenType::Comma)
    {
        return Syntax::ParameterList{Syntax::ParameterList::PosOnly{
            std::move(defParams.first), std::move(defParams.rest), comma, slash, std::nullopt}};
    }
    auto suffixComma = *m_current++;
    if (m_current == m_lexer.end()
        || (m_current->getTokenType() != TokenType::Identifier && m_current->getTokenType() != TokenType::Star
            && m_current->getTokenType() != TokenType::PowerOf))
    {
        return Syntax::ParameterList{Syntax::ParameterList::PosOnly{
            std::move(defParams.first), std::move(defParams.rest), comma, slash, std::pair{suffixComma, std::nullopt}}};
    }
    auto secondNoPosOnly = parseParameterListNoPosOnlyPrefix();
    if (!secondNoPosOnly)
    {
        return tl::unexpected{std::move(secondNoPosOnly).error()};
    }
    if (std::holds_alternative<Syntax::ParameterList::StarArgs>(secondNoPosOnly->variant))
    {
        return Syntax::ParameterList{
            Syntax::ParameterList::PosOnly{std::move(defParams.first), std::move(defParams.rest), comma, slash,
                                           std::pair{suffixComma, std::move(*secondNoPosOnly)}}};
    }

    if (m_current == m_lexer.end() || m_current->getTokenType() != TokenType::Comma)
    {
        return Syntax::ParameterList{
            Syntax::ParameterList::PosOnly{std::move(defParams.first), std::move(defParams.rest), comma, slash,
                                           std::pair{suffixComma, std::move(*secondNoPosOnly)}}};
    }
    auto& secondDefParams = pylir::get<Syntax::ParameterList::NoPosOnly::DefParams>(secondNoPosOnly->variant);

    auto trailingComma = *m_current++;
    std::optional<Syntax::ParameterList::StarArgs> starArgs;
    if (m_current != m_lexer.end() && m_current->getTokenType() == TokenType::Star)
    {
        auto parsedStarArgs = parseParameterListStarArgs();
        if (!parsedStarArgs)
        {
            return tl::unexpected{std::move(parsedStarArgs).error()};
        }
        starArgs = std::move(*parsedStarArgs);
    }
    secondDefParams.suffix = std::pair{trailingComma, std::move(starArgs)};
    return Syntax::ParameterList{Syntax::ParameterList::PosOnly{std::move(defParams.first), std::move(defParams.rest),
                                                                comma, slash,
                                                                std::pair{suffixComma, std::move(*secondNoPosOnly)}}};
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
    std::optional<Syntax::ParameterList> parameterList;
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
    std::optional<std::pair<BaseToken, Syntax::Expression>> suffix;
    if (m_current != m_lexer.end() && m_current->getTokenType() == TokenType::Arrow)
    {
        auto arrow = *m_current++;
        auto expression = parseExpression();
        if (!expression)
        {
            return tl::unexpected{std::move(expression).error()};
        }
        suffix = std::pair{arrow, std::move(*expression)};
    }
    auto colon = expect(TokenType::Colon);
    if (!colon)
    {
        return tl::unexpected{std::move(colon).error()};
    }
    m_namespace.emplace_back();
    std::optional exit = llvm::make_scope_exit([&] { m_namespace.pop_back(); });
    // add parameters to local variables

    if (parameterList)
    {
        class ParamVisitor : public Syntax::Visitor<ParamVisitor>
        {
        public:
            std::function<void(const IdentifierToken&)> callback;

            using Visitor::visit;

            void visit(const Syntax::ParameterList::Parameter& parameter)
            {
                callback(parameter.identifier);
            }
        } visitor{{}, [&](const IdentifierToken& token) { addToNamespace(token); }};
        visitor.visit(*parameterList);
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

    return Syntax::FuncDef{std::move(decorators),
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
        std::optional<Syntax::ArgumentList> argumentList;
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
    return Syntax::ClassDef{std::move(decorators),
                            *classKeyword,
                            IdentifierToken{std::move(*className)},
                            std::move(inheritance),
                            *colon,
                            std::make_unique<Syntax::Suite>(std::move(*suite)),
                            std::move(locals),
                            std::move(nonLocals),
                            std::move(unknowns)};
}
