#include "Parser.hpp"

#include <pylir/Diagnostics/DiagnosticMessages.hpp>
#include <pylir/Support/Functional.hpp>

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
    return Syntax::FileInput{std::move(statements)};
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
    auto suite = parseSuite();
    if (!suite)
    {
        return tl::unexpected{std::move(suite).error()};
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
    auto suite = parseSuite();
    if (!suite)
    {
        return tl::unexpected{std::move(suite).error()};
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

}
