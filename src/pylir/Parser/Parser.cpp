//  Licensed under the Apache License v2.0 with LLVM Exceptions.
//  See https://llvm.org/LICENSE.txt for license information.
//  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "Parser.hpp"

#include <pylir/Diagnostics/DiagnosticMessages.hpp>

#include "Visitor.hpp"

bool pylir::Parser::firstInExpression(TokenType tokenType) {
  switch (tokenType) {
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

bool pylir::Parser::firstInTarget(TokenType tokenType) {
  switch (tokenType) {
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

bool pylir::Parser::firstInSimpleStmt(TokenType tokenType) {
  switch (tokenType) {
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

bool pylir::Parser::firstInCompoundStmt(TokenType tokenType) {
  switch (tokenType) {
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

std::optional<pylir::Token> pylir::Parser::expect(pylir::TokenType tokenType) {
  if (m_current == m_lexer.end()) {
    createError(endOfFileLoc(), Diag::EXPECTED_N, tokenType)
        .addHighlight(endOfFileLoc(), fmt::format("{}", tokenType));
    return std::nullopt;
  }
  if (m_current->getTokenType() == TokenType::SyntaxError)
    return std::nullopt;

  if (m_current->getTokenType() != tokenType) {
    createError(*m_current, Diag::EXPECTED_N_INSTEAD_OF_N, tokenType,
                m_current->getTokenType())
        .addHighlight(*m_current, fmt::format("{}", tokenType),
                      Diag::flags::strikethrough);
    return std::nullopt;
  }

  return *m_current++;
}

bool pylir::Parser::lookaheadEquals(tcb::span<const TokenType> tokens) {
  Lexer::iterator end;
  std::size_t count = 0;
  for (end = m_current; end != m_lexer.end() && count != tokens.size();
       end++, count++)
    ;

  if (count != tokens.size())
    return false;

  return std::equal(m_current, end, tokens.begin(),
                    [](const Token& token, TokenType tokenType) {
                      return token.getTokenType() == tokenType;
                    });
}
