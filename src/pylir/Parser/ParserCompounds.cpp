//  Licensed under the Apache License v2.0 with LLVM Exceptions.
//  See https://llvm.org/LICENSE.txt for license information.
//  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "Parser.hpp"

#include <pylir/Diagnostics/DiagnosticMessages.hpp>
#include <pylir/Support/Functional.hpp>

#include "SemanticAnalysis.hpp"
#include "Visitor.hpp"

std::optional<pylir::Syntax::FileInput> pylir::Parser::parseFileInput() {
  decltype(Syntax::Suite::statements) vector;
  while (true) {
    while (maybeConsume(TokenType::Newline))
      ;
    if (m_current == m_lexer.end())
      break;

    auto statement = parseStatement();
    if (!statement)
      return std::nullopt;

    vector.insert(vector.end(), std::move_iterator(statement->begin()),
                  std::move_iterator(statement->end()));
  }
  auto fileInput = Syntax::FileInput{{std::move(vector)}, {}};
  SemanticAnalysis(m_lexer.getDiagManager()).visit(fileInput);
  return fileInput;
}

std::optional<decltype(pylir::Syntax::Suite::statements)>
pylir::Parser::parseStatement() {
  decltype(pylir::Syntax::Suite::statements) result;
  if (peekedIs(firstInCompoundStmt)) {
    auto compound = parseCompoundStmt();
    if (!compound)
      return std::nullopt;

    result.emplace_back(std::move(*compound));
    return result;
  }

  auto stmtList = parseStmtList();
  if (!stmtList)
    return std::nullopt;

  auto newLine = expect(TokenType::Newline);
  if (!newLine)
    return std::nullopt;

  result.insert(result.end(), std::move_iterator(stmtList->begin()),
                std::move_iterator(stmtList->end()));
  return result;
}

std::optional<std::vector<pylir::IntrVarPtr<pylir::Syntax::SimpleStmt>>>
pylir::Parser::parseStmtList() {
  std::vector<IntrVarPtr<Syntax::SimpleStmt>> statements;
  while (peekedIs(TokenType::SemiColon) || peekedIs(firstInSimpleStmt)) {
    while (maybeConsume(TokenType::SemiColon))
      ;
    if (m_current == m_lexer.end())
      return statements;

    auto simpleStmt = parseSimpleStmt();
    if (!simpleStmt)
      return std::nullopt;

    statements.push_back(std::move(*simpleStmt));
    if (!peekedIs(TokenType::SemiColon))
      return statements;
  }
  return statements;
}

std::optional<pylir::IntrVarPtr<pylir::Syntax::CompoundStmt>>
pylir::Parser::parseCompoundStmt() {
  if (m_current == m_lexer.end()) {
    createError(endOfFileLoc(), Diag::EXPECTED_N, "statement")
        .addHighlight(endOfFileLoc());
    return std::nullopt;
  }
  switch (m_current->getTokenType()) {
  case TokenType::IfKeyword: {
    auto ifStmt = parseIfStmt();
    if (!ifStmt)
      return std::nullopt;

    return std::make_unique<Syntax::IfStmt>(std::move(*ifStmt));
  }
  case TokenType::ForKeyword: {
    auto forStmt = parseForStmt();
    if (!forStmt)
      return std::nullopt;

    return std::make_unique<Syntax::ForStmt>(std::move(*forStmt));
  }
  case TokenType::TryKeyword: {
    auto tryStmt = parseTryStmt();
    if (!tryStmt)
      return std::nullopt;

    return std::make_unique<Syntax::TryStmt>(std::move(*tryStmt));
  }
  case TokenType::WithKeyword: {
    auto withStmt = parseWithStmt();
    if (!withStmt)
      return std::nullopt;

    return std::make_unique<Syntax::WithStmt>(std::move(*withStmt));
  }
  case TokenType::WhileKeyword: {
    auto whileStmt = parseWhileStmt();
    if (!whileStmt)
      return std::nullopt;

    return std::make_unique<Syntax::WhileStmt>(std::move(*whileStmt));
  }
  case TokenType::DefKeyword: {
    auto funcDef = parseFuncDef({}, std::nullopt);
    if (!funcDef)
      return std::nullopt;

    return std::make_unique<Syntax::FuncDef>(std::move(*funcDef));
  }
  case TokenType::ClassKeyword: {
    auto classDef = parseClassDef({});
    if (!classDef)
      return std::nullopt;

    return std::make_unique<Syntax::ClassDef>(std::move(*classDef));
  }
  case TokenType::AtSign: {
    std::vector<Syntax::Decorator> decorators;
    do {
      auto at = *m_current++;
      auto assignment = parseAssignmentExpression();
      if (!assignment)
        return std::nullopt;

      auto newline = expect(TokenType::Newline);
      if (!newline)
        return std::nullopt;

      decorators.push_back({at, std::move(*assignment), *newline});
    } while (peekedIs(TokenType::AtSign));
    if (auto async = maybeConsume(TokenType::AsyncKeyword)) {
      auto func = parseFuncDef(std::move(decorators), std::move(*async));
      if (!func)
        return std::nullopt;

      return std::make_unique<Syntax::FuncDef>(std::move(*func));
    }
    if (m_current == m_lexer.end()) {
      createError(endOfFileLoc(), Diag::EXPECTED_N, "class or function")
          .addHighlight(endOfFileLoc());
      return std::nullopt;
    }
    switch (m_current->getTokenType()) {
    case TokenType::DefKeyword: {
      auto func = parseFuncDef(std::move(decorators), std::nullopt);
      if (!func)
        return std::nullopt;

      return std::make_unique<Syntax::FuncDef>(std::move(*func));
    }
    case TokenType::ClassKeyword: {
      auto clazz = parseClassDef(std::move(decorators));
      if (!clazz)
        return std::nullopt;

      return std::make_unique<Syntax::ClassDef>(std::move(*clazz));
    }
    case TokenType::SyntaxError: return std::nullopt;
    default: {
      createError(*m_current, Diag::EXPECTED_N_INSTEAD_OF_N,
                  "class or function", m_current->getTokenType())
          .addHighlight(*m_current);
      return std::nullopt;
    }
    }
  }
  case TokenType::AsyncKeyword: {
    auto async = *m_current++;
    if (m_current == m_lexer.end()) {
      createError(endOfFileLoc(), Diag::EXPECTED_N, "'for', 'with' or function")
          .addHighlight(endOfFileLoc());
      return std::nullopt;
    }
    switch (m_current->getTokenType()) {
    case TokenType::DefKeyword: {
      auto func = parseFuncDef({}, async);
      if (!func)
        return std::nullopt;
      return std::make_unique<Syntax::FuncDef>(std::move(*func));
    }
    case TokenType::ForKeyword: {
      auto forStmt = parseForStmt();
      if (!forStmt)
        return std::nullopt;
      forStmt->maybeAsyncKeyword = async;
      return std::make_unique<Syntax::ForStmt>(std::move(*forStmt));
    }
    case TokenType::WithKeyword: {
      auto withStmt = parseWithStmt();
      if (!withStmt)
        return std::nullopt;
      withStmt->maybeAsyncKeyword = async;
      return std::make_unique<Syntax::WithStmt>(std::move(*withStmt));
    }
    case TokenType::SyntaxError: return std::nullopt;
    default: {
      createError(*m_current, Diag::EXPECTED_N_INSTEAD_OF_N,
                  "'for', 'with' or function", m_current->getTokenType())
          .addHighlight(*m_current);
      return std::nullopt;
    }
    }
  }
  case TokenType::SyntaxError: return std::nullopt;
  default: {
    createError(*m_current, Diag::EXPECTED_N_INSTEAD_OF_N, "statement",
                m_current->getTokenType())
        .addHighlight(*m_current);
    return std::nullopt;
  }
  }
}

std::optional<pylir::Syntax::IfStmt::Else> pylir::Parser::parseElse() {
  auto elseKeyowrd = *m_current++;
  auto elseColon = expect(TokenType::Colon);
  if (!elseColon)
    return std::nullopt;
  auto elseSuite = parseSuite();
  if (!elseSuite)
    return std::nullopt;
  return Syntax::IfStmt::Else{
      elseKeyowrd, *elseColon,
      std::make_unique<Syntax::Suite>(std::move(*elseSuite))};
}

std::optional<pylir::Syntax::IfStmt> pylir::Parser::parseIfStmt() {
  auto ifKeyword = expect(TokenType::IfKeyword);
  if (!ifKeyword)
    return std::nullopt;
  auto assignment = parseAssignmentExpression();
  if (!assignment)
    return std::nullopt;
  auto colon = expect(TokenType::Colon);
  if (!colon)
    return std::nullopt;
  auto suite = parseSuite();
  if (!suite)
    return std::nullopt;
  std::vector<Syntax::IfStmt::Elif> elifs;
  while (auto elif = maybeConsume(TokenType::ElifKeyword)) {
    auto condition = parseAssignmentExpression();
    if (!condition)
      return std::nullopt;
    auto elifColon = expect(TokenType::Colon);
    if (!elifColon)
      return std::nullopt;
    auto elIfSuite = parseSuite();
    if (!elIfSuite)
      return std::nullopt;
    elifs.push_back({*elif, std::move(*condition), *elifColon,
                     std::make_unique<Syntax::Suite>(std::move(*elIfSuite))});
  }
  std::optional<Syntax::IfStmt::Else> elseSection;
  if (peekedIs(TokenType::ElseKeyword)) {
    auto parsedElse = parseElse();
    if (!parsedElse)
      return std::nullopt;
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

std::optional<pylir::Syntax::WhileStmt> pylir::Parser::parseWhileStmt() {
  auto whileKeyword = expect(TokenType::WhileKeyword);
  if (!whileKeyword)
    return std::nullopt;
  auto condition = parseAssignmentExpression();
  if (!condition)
    return std::nullopt;
  auto colon = expect(TokenType::Colon);
  if (!colon)
    return std::nullopt;
  auto suite = parseSuite();
  if (!suite)
    return std::nullopt;
  std::optional<Syntax::IfStmt::Else> elseSection;
  if (peekedIs(TokenType::ElseKeyword)) {
    auto parsedElse = parseElse();
    if (!parsedElse)
      return std::nullopt;
    elseSection = std::move(*parsedElse);
  }
  return Syntax::WhileStmt{{},
                           *whileKeyword,
                           std::move(*condition),
                           *colon,
                           std::make_unique<Syntax::Suite>(std::move(*suite)),
                           std::move(elseSection)};
}

std::optional<pylir::Syntax::ForStmt> pylir::Parser::parseForStmt() {
  auto forKeyword = expect(TokenType::ForKeyword);
  if (!forKeyword)
    return std::nullopt;
  auto targetList = parseTargetList(*forKeyword);
  if (!targetList)
    return std::nullopt;
  auto inKeyword = expect(TokenType::InKeyword);
  if (!inKeyword)
    return std::nullopt;
  auto expressionList = parseExpressionList();
  if (!expressionList)
    return std::nullopt;
  auto colon = expect(TokenType::Colon);
  if (!colon)
    return std::nullopt;
  auto suite = parseSuite();
  if (!suite)
    return std::nullopt;
  std::optional<Syntax::IfStmt::Else> elseSection;
  if (peekedIs(TokenType::ElseKeyword)) {
    auto parsedElse = parseElse();
    if (!parsedElse)
      return std::nullopt;
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

std::optional<pylir::Syntax::TryStmt> pylir::Parser::parseTryStmt() {
  auto tryKeyword = expect(TokenType::TryKeyword);
  if (!tryKeyword)
    return std::nullopt;
  auto colon = expect(TokenType::Colon);
  if (!colon)
    return std::nullopt;
  auto suite = parseSuite();
  if (!suite)
    return std::nullopt;
  if (auto finallyKeyword = maybeConsume(TokenType::FinallyKeyword)) {
    auto finallyColon = expect(TokenType::Colon);
    if (!finallyColon)
      return std::nullopt;
    auto finallySuite = parseSuite();
    if (!finallySuite)
      return std::nullopt;
    return Syntax::TryStmt{
        {},
        *tryKeyword,
        *colon,
        std::make_unique<Syntax::Suite>(std::move(*suite)),
        {},
        std::nullopt,
        std::nullopt,
        Syntax::TryStmt::Finally{
            *finallyKeyword, *finallyColon,
            std::make_unique<Syntax::Suite>(std::move(*finallySuite))}};
  }

  std::optional<Syntax::TryStmt::ExceptAll> catchAll;
  std::vector<Syntax::TryStmt::ExceptArgs> exceptSections;
  do {
    auto exceptKeyword = expect(TokenType::ExceptKeyword);
    if (!exceptKeyword)
      return std::nullopt;
    if (catchAll) {
      createError(catchAll->exceptKeyword,
                  Diag::EXCEPT_CLAUSE_WITHOUT_EXPRESSION_MUST_COME_LAST)
          .addHighlight(catchAll->exceptKeyword, Diag::flags::bold);
      return std::nullopt;
    }
    if (auto exceptColon = maybeConsume(TokenType::Colon)) {
      auto exceptSuite = parseSuite();
      if (!exceptSuite)
        return std::nullopt;
      catchAll = {*exceptKeyword, *exceptColon,
                  std::make_unique<Syntax::Suite>(std::move(*exceptSuite))};
      continue;
    }
    auto expression = parseExpression();
    if (!expression)
      return std::nullopt;
    std::optional<IdentifierToken> name;
    if (maybeConsume(TokenType::AsKeyword)) {
      auto id = expect(TokenType::Identifier);
      if (!id)
        return std::nullopt;
      name.emplace(std::move(*id));
    }
    auto exceptColon = expect(TokenType::Colon);
    if (!exceptColon)
      return std::nullopt;
    auto exceptSuite = parseSuite();
    if (!exceptSuite)
      return std::nullopt;
    exceptSections.push_back(
        {*exceptKeyword, std::move(*expression), std::move(name), *exceptColon,
         std::make_unique<Syntax::Suite>(std::move(*exceptSuite))});
  } while (peekedIs(TokenType::ExceptKeyword));

  std::optional<Syntax::IfStmt::Else> elseSection;
  if (peekedIs(TokenType::ElseKeyword)) {
    auto parsedElse = parseElse();
    if (!parsedElse)
      return std::nullopt;
    elseSection = std::move(*parsedElse);
  }

  std::optional<Syntax::TryStmt::Finally> finally;
  if (auto finallyKeyword = maybeConsume(TokenType::FinallyKeyword)) {
    auto finallyColon = expect(TokenType::Colon);
    if (!finallyColon)
      return std::nullopt;
    auto finallySuite = parseSuite();
    if (!finallySuite)
      return std::nullopt;
    finally = Syntax::TryStmt::Finally{
        *finallyKeyword, *finallyColon,
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

std::optional<pylir::Syntax::WithStmt> pylir::Parser::parseWithStmt() {
  auto withKeyword = expect(TokenType::WithKeyword);
  if (!withKeyword)
    return std::nullopt;

  std::vector<Syntax::WithStmt::WithItem> withItems;
  while (withItems.empty() || maybeConsume(TokenType::Comma)) {
    auto expression = parseExpression();
    if (!expression)
      return std::nullopt;
    IntrVarPtr<Syntax::Target> name;
    if (auto asKeyword = maybeConsume(TokenType::AsKeyword)) {
      auto target = parseTarget(*asKeyword);
      if (!target)
        return std::nullopt;
      name = std::move(*target);
    }
    withItems.push_back({std::move(*expression), std::move(name)});
  }

  auto colon = expect(TokenType::Colon);
  if (!colon)
    return std::nullopt;
  auto suite = parseSuite();
  if (!suite)
    return std::nullopt;
  return Syntax::WithStmt{
      {},           std::nullopt,
      *withKeyword, std::move(withItems),
      *colon,       std::make_unique<Syntax::Suite>(std::move(*suite))};
}

std::optional<pylir::Syntax::Suite> pylir::Parser::parseSuite() {
  decltype(Syntax::Suite::statements) statements;
  if (maybeConsume(TokenType::Newline)) {
    if (!maybeConsume(TokenType::Indent)) {
      // stmt_list was empty, and hence a newline immediately followed with no
      // indent after.
      return Syntax::Suite{};
    }

    do {
      auto statement = parseStatement();
      if (!statement)
        return std::nullopt;
      statements.insert(statements.end(),
                        std::move_iterator(statement->begin()),
                        std::move_iterator(statement->end()));
    } while (peekedIsNot(TokenType::Dedent));
    auto dedent = expect(TokenType::Dedent);
    if (!dedent)
      return std::nullopt;
    return Syntax::Suite{std::move(statements)};
  }

  auto statementList = parseStmtList();
  if (!statementList)
    return std::nullopt;
  auto newline = expect(TokenType::Newline);
  if (!newline)
    return std::nullopt;
  statements.insert(statements.end(),
                    std::move_iterator(statementList->begin()),
                    std::move_iterator(statementList->end()));
  return Syntax::Suite{std::move(statements)};
}

std::optional<std::vector<pylir::Syntax::Parameter>>
pylir::Parser::parseParameterList() {
  std::vector<pylir::Syntax::Parameter> parameters;
  Syntax::Parameter::Kind currentKind = Syntax::Parameter::Normal;

  std::optional<BaseToken> seenPositionalOnly;
  std::optional<std::size_t> seenDefaultParam;
  std::variant<std::monostate, std::size_t, BaseToken> seenPosRest;
  std::optional<std::size_t> seenKwRest;

  bool first = true;
  while (first || maybeConsume(TokenType::Comma)) {
    if (first) {
      first = false;
    } else if (m_current == m_lexer.end()) {
      return parameters;
    }

    std::optional<Token> stars;
    switch (m_current->getTokenType()) {
    case TokenType::Divide: {
      if (parameters.empty()) {
        createError(
            *m_current,
            Diag::
                AT_LEAST_ONE_PARAMETER_REQUIRED_BEFORE_POSITIONAL_ONLY_INDICATOR)
            .addHighlight(*m_current);
        return std::nullopt;
      }
      if (!seenPositionalOnly) {
        seenPositionalOnly = *m_current++;
        for (auto& iter : parameters) {
          iter.kind = Syntax::Parameter::PosOnly;
        }
        continue;
      }
      createError(*m_current,
                  Diag::POSITIONAL_ONLY_INDICATOR_MAY_ONLY_APPEAR_ONCE)
          .addHighlight(*m_current)
          .addNote(*seenPositionalOnly, Diag::PREVIOUS_OCCURRENCE_HERE)
          .addHighlight(*seenPositionalOnly);
      return std::nullopt;
    }
    case TokenType::Star:
      stars = *m_current++;
      if (currentKind == Syntax::Parameter::Normal) {
        currentKind = Syntax::Parameter::KeywordOnly;
        if (!peekedIs({TokenType::Comma, TokenType::Identifier})) {
          return parameters;
        }
        if (m_current->getTokenType() == TokenType::Comma) {
          seenPosRest = *stars;
          continue;
        }
      }
      break;
    case TokenType::PowerOf:
      currentKind = Syntax::Parameter::KeywordRest;
      stars = *m_current++;
    case TokenType::Identifier: break;
    default: return parameters;
    }

    auto identifier = expect(TokenType::Identifier);
    if (!identifier)
      return std::nullopt;

    IntrVarPtr<Syntax::Expression> maybeType;
    IntrVarPtr<Syntax::Expression> maybeDefault;
    std::optional<BaseToken> maybeColon = maybeConsume(TokenType::Colon);
    if (maybeColon) {
      auto type = parseExpression();
      if (!type)
        return std::nullopt;
      maybeType = std::move(*type);
    }
    if (!stars && maybeConsume(TokenType::Assignment)) {
      auto defaultVal = parseExpression();
      if (!defaultVal)
        return std::nullopt;
      maybeDefault = std::move(*defaultVal);
    }
    if (currentKind != Syntax::Parameter::KeywordRest && !maybeDefault &&
        seenDefaultParam) {
      createError(
          *identifier,
          Diag::
              NO_DEFAULT_ARGUMENT_FOR_PARAMETER_N_FOLLOWING_PARAMETERS_WITH_DEFAULT_ARGUMENTS,
          pylir::get<std::string>(identifier->getValue()))
          .addHighlight(*identifier)
          .addNote(parameters[*seenDefaultParam],
                   Diag::PARAMETER_N_WITH_DEFAULT_ARGUMENT_HERE,
                   parameters[*seenDefaultParam].name.getValue())
          .addHighlight(parameters[*seenDefaultParam]);
      return std::nullopt;
    }
    if (maybeDefault) {
      seenDefaultParam = parameters.size();
    }
    if (seenKwRest) {
      createError(
          *identifier,
          Diag::NO_MORE_PARAMETERS_ALLOWED_AFTER_EXCESS_KEYWORD_PARAMETER_N,
          pylir::get<std::string>(identifier->getValue()),
          parameters[*seenKwRest].name.getValue())
          .addHighlight(*identifier)
          .addNote(parameters[*seenKwRest],
                   Diag::EXCESS_KEYWORD_PARAMETER_N_HERE,
                   parameters[*seenKwRest].name.getValue())
          .addHighlight(parameters[*seenKwRest]);
      return std::nullopt;
    }

    if (!stars) {
      parameters.push_back({currentKind, stars,
                            IdentifierToken(std::move(*identifier)), maybeColon,
                            std::move(maybeType), std::move(maybeDefault)});
      continue;
    }

    if (stars->getTokenType() == TokenType::Star) {
      if (auto* token = std::get_if<BaseToken>(&seenPosRest)) {
        createError(
            *identifier,
            Diag::STARRED_PARAMETER_NOT_ALLOWED_AFTER_KEYWORD_ONLY_INDICATOR)
            .addHighlight(*identifier)
            .addNote(*token, Diag::KEYWORD_ONLY_INDICATOR_HERE)
            .addHighlight(*token);
        return std::nullopt;
      }
      if (auto* index = std::get_if<std::size_t>(&seenPosRest)) {
        createError(*identifier, Diag::ONLY_ONE_STARRED_PARAMETER_ALLOWED)
            .addHighlight(*identifier)
            .addNote(parameters[*index], Diag::STARRED_PARAMETER_N_HERE,
                     parameters[*index].name.getValue())
            .addHighlight(parameters[*index]);
        return std::nullopt;
      }

      seenPosRest = parameters.size();
      parameters.push_back({Syntax::Parameter::PosRest, stars,
                            IdentifierToken(std::move(*identifier)), maybeColon,
                            std::move(maybeType), std::move(maybeDefault)});
      continue;
    }

    seenKwRest = parameters.size();
    parameters.push_back({Syntax::Parameter::KeywordRest, stars,
                          IdentifierToken(std::move(*identifier)), maybeColon,
                          std::move(maybeType), std::move(maybeDefault)});
  }
  return parameters;
}

std::optional<pylir::Syntax::FuncDef>
pylir::Parser::parseFuncDef(std::vector<Syntax::Decorator>&& decorators,
                            std::optional<BaseToken>&& asyncKeyword) {
  auto defKeyword = expect(TokenType::DefKeyword);
  if (!defKeyword)
    return std::nullopt;
  auto funcName = expect(TokenType::Identifier);
  if (!funcName)
    return std::nullopt;
  auto openParenth = expect(TokenType::OpenParentheses);
  if (!openParenth)
    return std::nullopt;
  std::vector<Syntax::Parameter> parameterList;
  if (peekedIsNot(TokenType::CloseParentheses)) {
    auto parsedParameterList = parseParameterList();
    if (!parsedParameterList)
      return std::nullopt;
    parameterList = std::move(*parsedParameterList);
  }
  auto closeParenth = expect(TokenType::CloseParentheses);
  if (!closeParenth)
    return std::nullopt;
  IntrVarPtr<Syntax::Expression> suffix;
  if (maybeConsume(TokenType::Arrow)) {
    auto expression = parseExpression();
    if (!expression)
      return std::nullopt;
    suffix = std::move(*expression);
  }
  auto colon = expect(TokenType::Colon);
  if (!colon)
    return std::nullopt;

  auto suite = parseSuite();
  if (!suite)
    return std::nullopt;

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
                         {}};
}

std::optional<pylir::Syntax::ClassDef>
pylir::Parser::parseClassDef(std::vector<Syntax::Decorator>&& decorators) {
  auto classKeyword = expect(TokenType::ClassKeyword);
  if (!classKeyword)
    return std::nullopt;
  auto className = expect(TokenType::Identifier);
  if (!className)
    return std::nullopt;
  std::optional<Syntax::ClassDef::Inheritance> inheritance;
  if (auto open = maybeConsume(TokenType::OpenParentheses)) {
    std::vector<Syntax::Argument> argumentList;
    if (peekedIsNot(TokenType::CloseParentheses)) {
      auto parsedArgumentList = parseArgumentList();
      if (!parsedArgumentList)
        return std::nullopt;
      argumentList = std::move(*parsedArgumentList);
    }
    auto close = expect(TokenType::CloseParentheses);
    if (!close)
      return std::nullopt;
    inheritance =
        Syntax::ClassDef::Inheritance{*open, std::move(argumentList), *close};
  }
  auto colon = expect(TokenType::Colon);
  if (!colon)
    return std::nullopt;

  auto suite = parseSuite();
  if (!suite)
    return std::nullopt;

  return Syntax::ClassDef{{},
                          std::move(decorators),
                          *classKeyword,
                          IdentifierToken{std::move(*className)},
                          std::move(inheritance),
                          *colon,
                          std::make_unique<Syntax::Suite>(std::move(*suite)),
                          {}};
}
