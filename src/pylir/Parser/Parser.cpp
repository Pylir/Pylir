// Copyright 2022 Markus Böck
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "Parser.hpp"

#include <pylir/Diagnostics/DiagnosticMessages.hpp>

#include "Visitor.hpp"

bool pylir::Parser::firstInExpression(TokenType tokenType)
{
    switch (tokenType)
    {
        case TokenType::LambdaKeyword:
        case TokenType::Minus:
        case TokenType::Plus:
        case TokenType::BitNegate:
        case TokenType::AwaitKeyword:
        case TokenType::StringLiteral:
        case TokenType::ByteLiteral:
        case TokenType::IntegerLiteral:
        case TokenType::FloatingPointLiteral:
        case TokenType::ComplexLiteral:
        case TokenType::Identifier:
        case TokenType::TrueKeyword:
        case TokenType::FalseKeyword:
        case TokenType::NoneKeyword:
        case TokenType::NotKeyword:
        case TokenType::OpenParentheses:
        case TokenType::OpenSquareBracket:
        case TokenType::OpenBrace: return true;
        default: break;
    }
    return false;
}

bool pylir::Parser::firstInTarget(TokenType tokenType)
{
    switch (tokenType)
    {
        case TokenType::OpenParentheses:
        case TokenType::OpenSquareBracket:
        case TokenType::OpenBrace:
        case TokenType::Identifier:
        case TokenType::Star:
        case TokenType::StringLiteral:
        case TokenType::ByteLiteral:
        case TokenType::IntegerLiteral:
        case TokenType::FloatingPointLiteral:
        case TokenType::ComplexLiteral: return true;
        default: return false;
    }
}

bool pylir::Parser::firstInSimpleStmt(TokenType tokenType)
{
    switch (tokenType)
    {
        case TokenType::AssertKeyword:
        case TokenType::PassKeyword:
        case TokenType::DelKeyword:
        case TokenType::ReturnKeyword:
        case TokenType::YieldKeyword:
        case TokenType::RaiseKeyword:
        case TokenType::BreakKeyword:
        case TokenType::ContinueKeyword:
        case TokenType::ImportKeyword:
        case TokenType::FromKeyword:
        case TokenType::GlobalKeyword:
        case TokenType::NonlocalKeyword:
        case TokenType::Star: return true;
        default: return firstInAssignmentExpression(tokenType);
    }
}

bool pylir::Parser::firstInCompoundStmt(TokenType tokenType)
{
    switch (tokenType)
    {
        case TokenType::IfKeyword:
        case TokenType::WhileKeyword:
        case TokenType::ForKeyword:
        case TokenType::TryKeyword:
        case TokenType::WithKeyword:
        case TokenType::AtSign:
        case TokenType::AsyncKeyword:
        case TokenType::DefKeyword:
        case TokenType::ClassKeyword: return true;
        default: return false;
    }
}

void pylir::Parser::addToNamespace(const pylir::Token& token)
{
    PYLIR_ASSERT(token.getTokenType() == TokenType::Identifier);
    auto identifierToken = IdentifierToken{token};
    addToNamespace(identifierToken);
}

void pylir::Parser::addToNamespace(const IdentifierToken& token)
{
    if (m_namespace.empty())
    {
        m_globals.insert(token);
        return;
    }
    auto result = m_namespace.back().identifiers.insert({token, Syntax::Scope::Local}).first;
    if (result->second == Syntax::Scope::Unknown)
    {
        result->second = Syntax::Scope::Local;
    }
}

void pylir::Parser::addToNamespace(const Syntax::Target& target)
{
    class TargetVisitor : public Syntax::Visitor<TargetVisitor>
    {
    public:
        std::function<void(const pylir::IdentifierToken&)> callback;

        using Visitor::visit;

        void visit(const Syntax::Atom& atom)
        {
            if (atom.token.getTokenType() == TokenType::Identifier)
            {
                callback(IdentifierToken(atom.token));
            }
            Visitor::visit(atom);
        }
    } visitor{{},
              [&](const IdentifierToken& token)
              { addToNamespace(token); }};
    visitor.visit(target);
}

tl::expected<pylir::Token, std::string> pylir::Parser::expect(pylir::TokenType tokenType)
{
    if (m_current == m_lexer.end())
    {
        return tl::unexpected{createDiagnosticsBuilder(m_document->getText().size(), Diag::EXPECTED_N, tokenType)
                                  .addLabel(m_document->getText().size(), fmt::format("{}", tokenType))
                                  .emit()};
    }
    if (m_current->getTokenType() == TokenType::SyntaxError)
    {
        return tl::unexpected{pylir::get<std::string>(m_current->getValue())};
    }
    if (m_current->getTokenType() != tokenType)
    {
        return tl::unexpected{
            createDiagnosticsBuilder(*m_current, Diag::EXPECTED_N_INSTEAD_OF_N, tokenType, m_current->getTokenType())
                .addLabel(*m_current, fmt::format("{}", tokenType), Diag::flags::strikethrough)
                .emit()};
    }

    return *m_current++;
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
