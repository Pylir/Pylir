//  Licensed under the Apache License v2.0 with LLVM Exceptions.
//  See https://llvm.org/LICENSE.txt for license information.
//  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "Parser.hpp"

#include <llvm/ADT/ScopeExit.h>
#include <llvm/ADT/StringExtras.h>

#include <pylir/Diagnostics/DiagnosticMessages.hpp>
#include <pylir/Support/Functional.hpp>
#include <pylir/Support/Variant.hpp>

namespace {
/// Checks whether 'expression' is a reference to an intrinsic. An intrinsic
/// consists of a series of attribute references resulting in the syntax:
/// "pylir" `.` "intr" { `.` identifier }.
/// Returns an empty optional if the expression is not an intrinsic reference.
std::optional<pylir::Syntax::Intrinsic>
checkForIntrinsic(const pylir::Syntax::Expression& expression) {
  using namespace pylir;
  using namespace pylir::Syntax;

  // Collect all the chained attribute references and their identifiers up until
  // the atom.
  llvm::SmallVector<IdentifierToken> identifiers;
  const Expression* current = &expression;
  while (const auto* ref = current->dyn_cast<AttributeRef>()) {
    identifiers.push_back(ref->identifier);
    current = ref->object.get();
  }

  // If its not an atom or not an identifier its not an intrinsic.
  const auto* atom = current->dyn_cast<Atom>();
  if (!atom || atom->token.getTokenType() != TokenType::Identifier)
    return std::nullopt;

  identifiers.emplace_back(atom->token);
  std::reverse(identifiers.begin(), identifiers.end());
  // Intrinsics always start with 'pylir' and 'intr'.
  if (identifiers.size() < 2 || identifiers[0].getValue() != "pylir" ||
      identifiers[1].getValue() != "intr")
    return std::nullopt;

  std::string name = llvm::join(
      llvm::map_range(identifiers, std::mem_fn(&IdentifierToken::getValue)),
      ".");
  return Intrinsic{{}, std::move(name), std::move(identifiers)};
}

} // namespace

std::optional<pylir::Syntax::Yield> pylir::Parser::parseYieldExpression() {
  auto yield = expect(TokenType::YieldKeyword);
  if (!yield)
    return std::nullopt;

  if (!peekedIs(TokenType::FromKeyword) && !peekedIs(firstInExpression))
    return Syntax::Yield{{}, std::move(*yield), std::nullopt, nullptr};

  if (auto from = maybeConsume(TokenType::FromKeyword)) {
    auto expression = parseExpression();
    if (!expression)
      return std::nullopt;
    return Syntax::Yield{
        {}, std::move(*yield), std::move(from), std::move(*expression)};
  }
  auto list = parseExpressionList();
  if (!list)
    return std::nullopt;
  return Syntax::Yield{{}, std::move(*yield), std::nullopt, std::move(*list)};
}

std::optional<pylir::IntrVarPtr<pylir::Syntax::Expression>>
pylir::Parser::parseAtom() {
  if (m_current == m_lexer.end()) {
    createError(endOfFileLoc(), Diag::EXPECTED_N,
                "identifier, number or enclosure")
        .addHighlight(endOfFileLoc());
    return std::nullopt;
  }

  switch (m_current->getTokenType()) {
  case TokenType::SyntaxError: return std::nullopt;
  case TokenType::Identifier: {
    auto token = *m_current++;
    return makeNode<Syntax::Atom>(token);
  }
  case TokenType::StringLiteral:
  case TokenType::ByteLiteral:
  case TokenType::IntegerLiteral:
  case TokenType::FloatingPointLiteral:
  case TokenType::ComplexLiteral:
  case TokenType::TrueKeyword:
  case TokenType::FalseKeyword:
  case TokenType::NoneKeyword: return makeNode<Syntax::Atom>(*m_current++);
  case TokenType::OpenParentheses:
  case TokenType::OpenBrace:
  case TokenType::OpenSquareBracket: return parseEnclosure();
  default:
    createError(*m_current, Diag::EXPECTED_N_INSTEAD_OF_N,
                "identifier, number or enclosure", m_current->getTokenType())
        .addHighlight(*m_current, Diag::flags::strikethrough);
    return std::nullopt;
  }
}

std::optional<pylir::IntrVarPtr<pylir::Syntax::Expression>>
pylir::Parser::parseEnclosure() {
  if (m_current == m_lexer.end()) {
    createError(endOfFileLoc(), Diag::EXPECTED_N,
                fmt::format("{:q}, {:q} or {:q}", TokenType::OpenParentheses,
                            TokenType::OpenSquareBracket, TokenType::OpenBrace))
        .addHighlight(endOfFileLoc());
    return std::nullopt;
  }
  switch (m_current->getTokenType()) {
  case TokenType::OpenParentheses: {
    auto openParenth = *m_current++;
    if (m_current == m_lexer.end() || peekedIs(TokenType::CloseParentheses)) {
      auto closeParentheses = expect(TokenType::CloseParentheses);
      if (!closeParentheses)
        return std::nullopt;
      return makeNode<Syntax::TupleConstruct>(
          openParenth, std::vector<Syntax::StarredItem>{}, *closeParentheses);
    }
    if (m_current->getTokenType() == TokenType::YieldKeyword) {
      auto yield = parseYieldExpression();
      if (!yield)
        return std::nullopt;
      auto closeParentheses = expect(TokenType::CloseParentheses);
      if (!closeParentheses)
        return std::nullopt;
      return std::make_unique<Syntax::Yield>(std::move(*yield));
    }

    if (firstInStarredItem(m_current->getTokenType()) &&
        (!firstInExpression(m_current->getTokenType()) ||
         lookaheadEquals(
             std::array{TokenType::Identifier, TokenType::Walrus}))) {
      auto starredExpression = parseStarredExpression();
      if (!starredExpression)
        return std::nullopt;
      auto closeParentheses = expect(TokenType::CloseParentheses);
      if (!closeParentheses)
        return std::nullopt;
      return starredExpression;
    }

    auto expression = parseExpression();
    if (!expression)
      return std::nullopt;
    if (!peekedIs(firstInCompFor)) {
      auto starredExpression = parseStarredExpression(std::move(*expression));
      if (!starredExpression)
        return std::nullopt;
      auto closeParentheses = expect(TokenType::CloseParentheses);
      if (!closeParentheses)
        return std::nullopt;
      return starredExpression;
    }

    auto compFor = parseCompFor();
    if (!compFor)
      return std::nullopt;
    auto closeParentheses = expect(TokenType::CloseParentheses);
    if (!closeParentheses)
      return std::nullopt;
    return makeNode<Syntax::Generator>(
        std::move(openParenth), std::move(*expression), std::move(*compFor),
        std::move(*closeParentheses));
  }
  case TokenType::OpenBrace: {
    auto openBrace = *m_current++;
    if (m_current == m_lexer.end() || peekedIs(TokenType::CloseBrace)) {
      auto closeBrace = expect(TokenType::CloseBrace);
      if (!closeBrace)
        return std::nullopt;
      return makeNode<Syntax::DictDisplay>(
          std::move(openBrace), std::vector<Syntax::DictDisplay::KeyDatum>{},
          std::move(*closeBrace));
    }

    if (peekedIs(TokenType::Star) ||
        lookaheadEquals(std::array{TokenType::Identifier, TokenType::Walrus})) {
      auto starredList = parseStarredList();
      if (!starredList)
        return std::nullopt;
      auto closeBrace = expect(TokenType::CloseBrace);
      if (!closeBrace)
        return std::nullopt;
      return makeNode<Syntax::SetDisplay>(std::move(openBrace),
                                          std::move(*starredList),
                                          std::move(*closeBrace));
    }

    std::optional<Syntax::DictDisplay::KeyDatum> keyDatum;
    if (m_current->getTokenType() != TokenType::PowerOf) {
      auto expression = parseExpression();
      if (!expression)
        return std::nullopt;
      if (!peekedIs(TokenType::Colon)) {
        // We are 100% in a Set.
        if (peekedIs(firstInCompFor)) {
          auto comprehension = parseComprehension(std::move(*expression));
          if (!comprehension)
            return std::nullopt;
          auto closeBrace = expect(TokenType::CloseBrace);
          if (!closeBrace)
            return std::nullopt;
          return makeNode<Syntax::SetDisplay>(std::move(openBrace),
                                              std::move(*comprehension),
                                              std::move(*closeBrace));
        }
        auto starredList = parseStarredList(
            Syntax::StarredItem{std::nullopt, std::move(*expression)});
        if (!starredList)
          return std::nullopt;
        auto closeBrace = expect(TokenType::CloseBrace);
        if (!closeBrace)
          return std::nullopt;
        return makeNode<Syntax::SetDisplay>(std::move(openBrace),
                                            std::move(*starredList),
                                            std::move(*closeBrace));
      }
      auto colon = *m_current++;
      auto secondExpression = parseExpression();
      if (!secondExpression)
        return std::nullopt;
      if (peekedIs(firstInCompFor)) {
        auto compFor = parseCompFor();
        if (!compFor)
          return std::nullopt;
        auto closeBrace = expect(TokenType::CloseBrace);
        if (!closeBrace)
          return std::nullopt;
        return makeNode<Syntax::DictDisplay>(
            std::move(openBrace),
            Syntax::DictDisplay::DictComprehension{
                std::move(*expression), std::move(colon),
                std::move(*secondExpression), std::move(*compFor)},
            std::move(*closeBrace));
      }
      keyDatum = Syntax::DictDisplay::KeyDatum{std::move(*expression),
                                               std::move(colon),
                                               std::move(*secondExpression)};
    }

    auto keyDatumList = parseCommaList(
        [&]() -> std::optional<Syntax::DictDisplay::KeyDatum> {
          if (auto powerOf = maybeConsume(TokenType::PowerOf)) {
            auto orExpr = parseOrExpr();
            if (!orExpr)
              return std::nullopt;
            return Syntax::DictDisplay::KeyDatum{std::move(*orExpr),
                                                 std::move(*powerOf), nullptr};
          }
          auto first = parseExpression();
          if (!first)
            return std::nullopt;
          auto colon = expect(TokenType::Colon);
          if (!colon)
            return std::nullopt;
          auto second = parseExpression();
          if (!second)
            return std::nullopt;
          return Syntax::DictDisplay::KeyDatum{
              std::move(*first), std::move(*colon), std::move(*second)};
        },
        [&](TokenType type) {
          return firstInExpression(type) || type == TokenType::PowerOf;
        },
        std::move(keyDatum));
    if (!keyDatumList)
      return std::nullopt;
    auto closeBrace = expect(TokenType::CloseBrace);
    if (!closeBrace)
      return std::nullopt;
    return makeNode<Syntax::DictDisplay>(
        std::move(openBrace), std::move(*keyDatumList), std::move(*closeBrace));
  }
  case TokenType::OpenSquareBracket: {
    auto openSquareBracket = *m_current++;
    if (m_current == m_lexer.end() || peekedIs(TokenType::CloseSquareBracket)) {
      auto closeSquare = expect(TokenType::CloseSquareBracket);
      if (!closeSquare)
        return std::nullopt;
      return makeNode<Syntax::ListDisplay>(std::move(openSquareBracket),
                                           std::vector<Syntax::StarredItem>{},
                                           std::move(*closeSquare));
    }
    if (firstInStarredItem(m_current->getTokenType()) &&
        !firstInComprehension(m_current->getTokenType())) {
      auto starredList = parseStarredList();
      if (!starredList)
        return std::nullopt;
      auto closeSquare = expect(TokenType::CloseSquareBracket);
      if (!closeSquare)
        return std::nullopt;
      return makeNode<Syntax::ListDisplay>(std::move(openSquareBracket),
                                           std::move(*starredList),
                                           std::move(*closeSquare));
    }

    auto assignment = parseAssignmentExpression();
    if (!assignment)
      return std::nullopt;
    if (!peekedIs(firstInCompFor)) {
      auto starredList = parseStarredList(
          Syntax::StarredItem{std::nullopt, std::move(*assignment)});
      if (!starredList)
        return std::nullopt;
      auto closeSquare = expect(TokenType::CloseSquareBracket);
      if (!closeSquare)
        return std::nullopt;
      return makeNode<Syntax::ListDisplay>(std::move(openSquareBracket),
                                           std::move(*starredList),
                                           std::move(*closeSquare));
    }

    auto comprehension = parseComprehension(std::move(*assignment));
    if (!comprehension)
      return std::nullopt;
    auto closeSquare = expect(TokenType::CloseSquareBracket);
    if (!closeSquare)
      return std::nullopt;
    return makeNode<Syntax::ListDisplay>(std::move(openSquareBracket),
                                         std::move(*comprehension),
                                         std::move(*closeSquare));
  }
  case TokenType::SyntaxError: return std::nullopt;
  default:
    createError(*m_current, Diag::EXPECTED_N_INSTEAD_OF_N,
                fmt::format("{:q}, {:q} or {:q}", TokenType::OpenParentheses,
                            TokenType::OpenSquareBracket, TokenType::OpenBrace),
                m_current->getTokenType())
        .addHighlight(*m_current, Diag::flags::strikethrough);
    return std::nullopt;
  }
}

std::optional<pylir::Syntax::AttributeRef>
pylir::Parser::parseAttributeRef(IntrVarPtr<Syntax::Expression>&& expression) {
  auto dot = expect(TokenType::Dot);
  if (!dot)
    return std::nullopt;
  auto identifier = expect(TokenType::Identifier);
  if (!identifier)
    return std::nullopt;
  return Syntax::AttributeRef{{},
                              std::move(expression),
                              *std::move(dot),
                              IdentifierToken{*std::move(identifier)}};
}

std::optional<pylir::IntrVarPtr<pylir::Syntax::Expression>>
pylir::Parser::parseSlicingOrSubscription(
    IntrVarPtr<Syntax::Expression>&& expression) {
  auto squareBracket = expect(TokenType::OpenSquareBracket);
  if (!squareBracket)
    return std::nullopt;
  auto list = parseCommaList(
      [&]() -> std::optional<IntrVarPtr<Syntax::Expression>> {
        IntrVarPtr<Syntax::Expression> lowerBound;
        if (m_current->getTokenType() != TokenType::Colon) {
          auto first = parseExpression();
          if (!first)
            return std::nullopt;
          if (m_current == m_lexer.end() ||
              m_current->getTokenType() != TokenType::Colon)
            return first;

          lowerBound = std::move(*first);
        }
        auto firstColon = *m_current++;
        IntrVarPtr<Syntax::Expression> upperBound;
        if (peekedIs(firstInExpression)) {
          auto temp = parseExpression();
          if (!temp)
            return std::nullopt;
          upperBound = std::move(*temp);
        }
        auto secondColumn = expect(TokenType::Colon);
        if (!secondColumn)
          return std::nullopt;
        IntrVarPtr<Syntax::Expression> stride;
        if (peekedIs(firstInExpression)) {
          auto temp = parseExpression();
          if (!temp)
            return std::nullopt;
          stride = std::move(*temp);
        }
        return makeNode<Syntax::Slice>(
            std::move(lowerBound), std::move(firstColon), std::move(upperBound),
            std::move(*secondColumn), std::move(stride));
      },
      &firstInExpression);
  if (!list)
    return std::nullopt;
  auto closeSquareBracket = expect(TokenType::CloseSquareBracket);
  if (!closeSquareBracket)
    return std::nullopt;
  if (list->size() != 1) {
    std::vector<Syntax::StarredItem> starredItems(list->size());
    std::transform(std::move_iterator(list->begin()),
                   std::move_iterator(list->end()), starredItems.begin(),
                   [](IntrVarPtr<Syntax::Expression>&& expr) {
                     return Syntax::StarredItem{std::nullopt, std::move(expr)};
                   });

    return makeNode<Syntax::Subscription>(
        std::move(expression), std::move(*squareBracket),
        makeNode<Syntax::TupleConstruct>(std::nullopt, std::move(starredItems),
                                         std::nullopt),
        std::move(*closeSquareBracket));
  }
  return makeNode<Syntax::Subscription>(
      std::move(expression), std::move(*squareBracket),
      std::move(list->front()), std::move(*closeSquareBracket));
}

std::optional<std::vector<pylir::Syntax::Argument>>
pylir::Parser::parseArgumentList(
    IntrVarPtr<Syntax::Expression>&& firstAssignment) {
  std::vector<pylir::Syntax::Argument> arguments;
  if (firstAssignment)
    arguments.push_back(
        {std::nullopt, std::nullopt, std::move(firstAssignment)});

  std::optional<std::size_t> firstKeywordIndex;
  std::optional<std::size_t> firstMappingExpansionIndex;
  while (arguments.empty() || peekedIs(TokenType::Comma)) {
    if (!arguments.empty()) {
      // Some productions using argument_list allow a trailing comma afterwards.
      // We can't always allow this and hence need to let the caller handle it.
      // We therefore only consume the comma if the thing afterwards may be
      // parsed as argument as well.
      if (std::next(m_current) == m_lexer.end() ||
          (!firstInExpression(std::next(m_current)->getTokenType()) &&
           std::next(m_current)->getTokenType() != TokenType::Star &&
           std::next(m_current)->getTokenType() != TokenType::PowerOf))
        break;

      m_current++;
    }
    std::optional<Token> expansionOrEqual;
    std::optional<IdentifierToken> keywordName;
    switch (m_current->getTokenType()) {
    case TokenType::PowerOf:
    case TokenType::Star: expansionOrEqual = *m_current++; break;
    case TokenType::Identifier: {
      if (std::next(m_current) != m_lexer.end() &&
          std::next(m_current)->getTokenType() == TokenType::Assignment) {
        keywordName = IdentifierToken{*m_current++};
        expansionOrEqual = *m_current++;
      }
      break;
    }
    default: break;
    }
    auto expression =
        !expansionOrEqual ? parseAssignmentExpression() : parseExpression();
    if (!expression)
      return std::nullopt;

    // Remember the indices of both the first keyword argument and the first
    // mapping expansion.
    if (keywordName && !firstKeywordIndex)
      firstKeywordIndex = arguments.size();
    else if (!firstMappingExpansionIndex && expansionOrEqual &&
             expansionOrEqual->getTokenType() == TokenType::PowerOf)
      firstMappingExpansionIndex = arguments.size();

    if (!expansionOrEqual &&
        (firstKeywordIndex || firstMappingExpansionIndex)) {
      // We diagnose whichever one of the two cases happened first.
      if (!firstMappingExpansionIndex ||
          (firstKeywordIndex &&
           firstKeywordIndex < firstMappingExpansionIndex)) {
        createError(
            **expression,
            Diag::POSITIONAL_ARGUMENT_NOT_ALLOWED_FOLLOWING_KEYWORD_ARGUMENTS)
            .addHighlight(**expression)
            .addNote(*arguments[*firstKeywordIndex].maybeName,
                     Diag::FIRST_KEYWORD_ARGUMENT_N_HERE,
                     arguments[*firstKeywordIndex].maybeName->getValue())
            .addHighlight(*arguments[*firstKeywordIndex].maybeName);
        return std::nullopt;
      }
      createError(
          **expression,
          Diag::POSITIONAL_ARGUMENT_NOT_ALLOWED_FOLLOWING_DICTIONARY_UNPACKING)
          .addHighlight(**expression)
          .addNote(arguments[*firstMappingExpansionIndex],
                   Diag::FIRST_DICTIONARY_UNPACKING_HERE)
          .addHighlight(arguments[*firstMappingExpansionIndex]);
      return std::nullopt;
    }

    if (expansionOrEqual &&
        expansionOrEqual->getTokenType() == TokenType::Star &&
        firstMappingExpansionIndex) {
      createError(
          **expression,
          Diag::ITERABLE_UNPACKING_NOT_ALLOWED_FOLLOWING_DICTIONARY_UNPACKING)
          .addHighlight(**expression)
          .addNote(arguments[*firstMappingExpansionIndex],
                   Diag::FIRST_DICTIONARY_UNPACKING_HERE)
          .addHighlight(arguments[*firstMappingExpansionIndex]);
      return std::nullopt;
    }
    arguments.push_back({std::move(keywordName), std::move(expansionOrEqual),
                         std::move(*expression)});
  }
  return arguments;
}

std::optional<pylir::Syntax::Call>
pylir::Parser::parseCall(IntrVarPtr<Syntax::Expression>&& expression) {
  auto openParenth = expect(TokenType::OpenParentheses);
  if (!openParenth)
    return std::nullopt;
  if (m_current == m_lexer.end() || peekedIs(TokenType::CloseParentheses)) {
    auto closeParenth = expect(TokenType::CloseParentheses);
    if (!closeParenth)
      return std::nullopt;
    return Syntax::Call{{},
                        std::move(expression),
                        std::move(*openParenth),
                        std::vector<Syntax::Argument>{},
                        std::move(*closeParenth)};
  }
  // If it's a star, power of or an "identifier =", it's definitely an argument
  // list, not a comprehension
  IntrVarPtr<Syntax::Expression> firstAssignment;
  if (peekedIsNot({TokenType::Star, TokenType::PowerOf}) &&
      !lookaheadEquals(
          std::array{TokenType::Identifier, TokenType::Assignment})) {
    // Otherwise parse an Assignment expression
    auto assignment = parseAssignmentExpression();
    if (!assignment)
      return std::nullopt;
    if (peekedIs(firstInCompFor)) {
      // We are in a comprehension!
      auto comprehension = parseComprehension(std::move(*assignment));
      if (!comprehension)
        return std::nullopt;
      auto closeParenth = expect(TokenType::CloseParentheses);
      if (!closeParenth)
        return std::nullopt;
      return Syntax::Call{{},
                          std::move(expression),
                          std::move(*openParenth),
                          std::move(*comprehension),
                          std::move(*closeParenth)};
    }
    firstAssignment = std::move(*assignment);
  }

  auto argumentList = parseArgumentList(std::move(firstAssignment));
  if (!argumentList)
    return std::nullopt;
  maybeConsume(TokenType::Comma);

  auto closeParenth = expect(TokenType::CloseParentheses);
  if (!closeParenth)
    return std::nullopt;
  return Syntax::Call{{},
                      std::move(expression),
                      std::move(*openParenth),
                      std::move(*argumentList),
                      std::move(*closeParenth)};
}

std::optional<pylir::IntrVarPtr<pylir::Syntax::Expression>>
pylir::Parser::parsePrimary() {
  // First must always be an atom, all others are left recursive
  auto atom = parseAtom();
  if (!atom)
    return std::nullopt;
  IntrVarPtr<Syntax::Expression> current{std::move(*atom)};
  if (m_current == m_lexer.end())
    return {std::move(current)};

  // Checks whether 'current' refers to an intrinsic and transforms it to that
  // node. Should be called after all attribute refs have been parsed.
  auto performIntrinsicCheck = [&] {
    if (std::optional<Syntax::Intrinsic> intrinsic =
            checkForIntrinsic(*current))
      current = std::make_unique<Syntax::Intrinsic>(std::move(*intrinsic));
  };

  while (peekedIs({TokenType::Dot, TokenType::OpenParentheses,
                   TokenType::OpenSquareBracket})) {
    switch (m_current->getTokenType()) {
    case TokenType::Dot: {
      auto attributeRef = parseAttributeRef(std::move(current));
      if (!attributeRef)
        return std::nullopt;
      current =
          std::make_unique<Syntax::AttributeRef>(std::move(*attributeRef));
      break;
    }
    case TokenType::OpenSquareBracket: {
      performIntrinsicCheck();

      auto newCurrent = parseSlicingOrSubscription(std::move(current));
      if (!newCurrent)
        return std::nullopt;
      current = std::move(*newCurrent);
      break;
    }
    case TokenType::OpenParentheses: {
      performIntrinsicCheck();

      auto call = parseCall(std::move(current));
      if (!call)
        return std::nullopt;
      current = std::make_unique<Syntax::Call>(std::move(*call));
      break;
    }
    default: PYLIR_UNREACHABLE;
    }
  }

  performIntrinsicCheck();

  return {std::move(current)};
}

std::optional<pylir::IntrVarPtr<pylir::Syntax::Expression>>
pylir::Parser::parseExpressionList() {
  std::vector<IntrVarPtr<Syntax::Expression>> expression;
  bool lastWasComma = true;
  do {
    auto expr = parseExpression();
    if (!expr)
      return std::nullopt;
    expression.push_back(std::move(*expr));
    if (!maybeConsume(TokenType::Comma)) {
      lastWasComma = false;
      break;
    }
  } while (peekedIs(firstInExpression));
  if (expression.size() == 1 && !lastWasComma)
    return std::move(expression.front());

  std::vector<Syntax::StarredItem> items(expression.size());
  std::transform(std::move_iterator(expression.begin()),
                 std::move_iterator(expression.end()), items.begin(),
                 [](IntrVarPtr<Syntax::Expression>&& expr) {
                   return Syntax::StarredItem{std::nullopt, std::move(expr)};
                 });
  return makeNode<Syntax::TupleConstruct>(std::nullopt, std::move(items),
                                          std::nullopt);
}

std::optional<pylir::IntrVarPtr<pylir::Syntax::Expression>>
pylir::Parser::parseAssignmentExpression() {
  if (!lookaheadEquals(std::array{TokenType::Identifier, TokenType::Walrus}))
    return parseExpression();

  IdentifierToken variable{*m_current++};
  BaseToken walrus = *m_current++;
  auto expression = parseExpression();
  if (!expression)
    return std::nullopt;
  return makeNode<Syntax::Assignment>(std::move(variable), walrus,
                                      std::move(*expression));
}

std::optional<pylir::Syntax::UnaryOp> pylir::Parser::parseAwaitExpr() {
  auto await = expect(TokenType::AwaitKeyword);
  if (!await)
    return std::nullopt;
  auto primary = parsePrimary();
  if (!primary)
    return std::nullopt;
  return Syntax::UnaryOp{{}, std::move(*await), std::move(*primary)};
}

std::optional<pylir::IntrVarPtr<pylir::Syntax::Expression>>
pylir::Parser::parsePower() {
  IntrVarPtr<Syntax::Expression> expression;
  if (peekedIs(TokenType::AwaitKeyword)) {
    auto await = parseAwaitExpr();
    if (!await)
      return std::nullopt;
    expression = std::make_unique<Syntax::UnaryOp>(std::move(*await));
  } else {
    auto primary = parsePrimary();
    if (!primary)
      return std::nullopt;
    expression = std::move(*primary);
  }
  auto powerOf = maybeConsume(TokenType::PowerOf);
  if (!powerOf)
    return expression;

  auto uExpr = parseUExpr();
  if (!uExpr)
    return std::nullopt;
  return makeNode<Syntax::BinOp>(std::move(expression), std::move(*powerOf),
                                 std::move(*uExpr));
}

std::optional<pylir::IntrVarPtr<pylir::Syntax::Expression>>
pylir::Parser::parseUExpr() {
  std::vector<Token> unaries;
  while (auto unary = maybeConsume(
             {TokenType::Minus, TokenType::Plus, TokenType::BitNegate}))
    unaries.push_back(std::move(*unary));

  auto power = parsePower();
  if (!power)
    return std::nullopt;
  IntrVarPtr<Syntax::Expression> current{std::move(*power)};
  std::reverse(unaries.begin(), unaries.end());
  for (Token& token : unaries)
    current = makeNode<Syntax::UnaryOp>(std::move(token), std::move(current));

  return current;
}

std::optional<pylir::IntrVarPtr<pylir::Syntax::Expression>>
pylir::Parser::parseMExpr() {
  auto first = parseUExpr();
  if (!first)
    return std::nullopt;
  IntrVarPtr<Syntax::Expression> current{std::move(*first)};
  while (auto op = maybeConsume({TokenType::Star, TokenType::AtSign,
                                 TokenType::IntDivide, TokenType::Divide,
                                 TokenType::Remainder})) {
    if (op->getTokenType() == TokenType::AtSign) {
      auto rhs = parseMExpr();
      if (!rhs)
        return std::nullopt;
      current = makeNode<Syntax::BinOp>(std::move(current), std::move(*op),
                                        std::move(*rhs));
      continue;
    }

    auto rhs = parseUExpr();
    if (!rhs)
      return std::nullopt;
    current = makeNode<Syntax::BinOp>(std::move(current), std::move(*op),
                                      std::move(*rhs));
  }
  return current;
}

std::optional<pylir::IntrVarPtr<pylir::Syntax::Expression>>
pylir::Parser::parseAExpr() {
  return parseGenericBinOp<&Parser::parseMExpr, TokenType::Minus,
                           TokenType::Plus>();
}

std::optional<pylir::IntrVarPtr<pylir::Syntax::Expression>>
pylir::Parser::parseShiftExpr() {
  return parseGenericBinOp<&Parser::parseAExpr, TokenType::ShiftLeft,
                           TokenType::ShiftRight>();
}

std::optional<pylir::IntrVarPtr<pylir::Syntax::Expression>>
pylir::Parser::parseAndExpr() {
  return parseGenericBinOp<&Parser::parseShiftExpr, TokenType::BitAnd>();
}

std::optional<pylir::IntrVarPtr<pylir::Syntax::Expression>>
pylir::Parser::parseXorExpr() {
  return parseGenericBinOp<&Parser::parseAndExpr, TokenType::BitXor>();
}

std::optional<pylir::IntrVarPtr<pylir::Syntax::Expression>>
pylir::Parser::parseOrExpr() {
  return parseGenericBinOp<&Parser::parseXorExpr, TokenType::BitOr>();
}

std::optional<pylir::IntrVarPtr<pylir::Syntax::Expression>>
pylir::Parser::parseComparison() {
  auto first = parseOrExpr();
  if (!first)
    return std::nullopt;
  IntrVarPtr<Syntax::Expression> current{std::move(*first)};
  std::vector<
      std::pair<Syntax::Comparison::Operator, IntrVarPtr<Syntax::Expression>>>
      rest;
  while (auto op = maybeConsume(
             {TokenType::LessThan, TokenType::LessOrEqual,
              TokenType::GreaterThan, TokenType::GreaterOrEqual,
              TokenType::NotEqual, TokenType::Equal, TokenType::IsKeyword,
              TokenType::NotKeyword, TokenType::InKeyword})) {
    std::optional<Token> second;
    switch (op->getTokenType()) {
    case TokenType::IsKeyword:
      second = maybeConsume(TokenType::NotKeyword);
      break;
    case TokenType::NotKeyword: {
      auto in = expect(TokenType::InKeyword);
      if (!in)
        return std::nullopt;
      second = std::move(*in);
      break;
    }
    default: break;
    }
    auto rhs = parseOrExpr();
    if (!rhs)
      return std::nullopt;
    rest.emplace_back(
        Syntax::Comparison::Operator{std::move(*op), std::move(second)},
        std::move(*rhs));
  }
  if (rest.empty())
    return current;

  return makeNode<Syntax::Comparison>(std::move(current), std::move(rest));
}

std::optional<pylir::IntrVarPtr<pylir::Syntax::Expression>>
pylir::Parser::parseNotTest() {
  auto end = std::find_if_not(m_current, m_lexer.end(), [](const Token& token) {
    return token.getTokenType() == TokenType::NotKeyword;
  });
  std::vector<Token> nots(m_current, end);
  m_current = end;
  std::reverse(nots.begin(), nots.end());
  auto comp = parseComparison();
  if (!comp)
    return std::nullopt;
  IntrVarPtr<Syntax::Expression> current{std::move(*comp)};
  for (Token& token : nots)
    current = makeNode<Syntax::UnaryOp>(std::move(token), std::move(current));

  return {std::move(current)};
}

std::optional<pylir::IntrVarPtr<pylir::Syntax::Expression>>
pylir::Parser::parseAndTest() {
  return parseGenericBinOp<&Parser::parseNotTest, TokenType::AndKeyword>();
}

std::optional<pylir::IntrVarPtr<pylir::Syntax::Expression>>
pylir::Parser::parseOrTest() {
  return parseGenericBinOp<&Parser::parseAndTest, TokenType::OrKeyword>();
}

std::optional<pylir::IntrVarPtr<pylir::Syntax::Expression>>
pylir::Parser::parseConditionalExpression() {
  auto orTest = parseOrTest();
  if (!orTest)
    return std::nullopt;
  auto ifKeyword = maybeConsume(TokenType::IfKeyword);
  if (!ifKeyword)
    return orTest;

  auto condition = parseOrTest();
  if (!condition)
    return std::nullopt;
  auto elseKeyword = expect(TokenType::ElseKeyword);
  if (!elseKeyword)
    return std::nullopt;
  auto other = parseExpression();
  if (!other)
    return std::nullopt;
  return makeNode<Syntax::Conditional>(
      std::move(*orTest), std::move(*ifKeyword), std::move(*condition),
      std::move(*elseKeyword), std::move(*other));
}

std::optional<pylir::IntrVarPtr<pylir::Syntax::Expression>>
pylir::Parser::parseExpression() {
  if (!peekedIs(TokenType::LambdaKeyword))
    return parseConditionalExpression();

  auto lambda = parseLambdaExpression();
  if (!lambda)
    return std::nullopt;
  return std::make_unique<Syntax::Lambda>(std::move(*lambda));
}

std::optional<pylir::Syntax::Lambda> pylir::Parser::parseLambdaExpression() {
  auto keyword = expect(TokenType::LambdaKeyword);
  if (!keyword)
    return std::nullopt;
  std::vector<Syntax::Parameter> parameterList;
  if (peekedIsNot(TokenType::Colon)) {
    auto parsedParameterList = parseParameterList();
    if (!parsedParameterList)
      return std::nullopt;
    parameterList = std::move(*parsedParameterList);
  }

  // There is an ambiguity in the grammar here due to the parameter list before.
  // In particular: `lambda x: int` could be parsed as either a lambda with a
  // parameter x returning the value `int` or a lambda with parameter x of type
  // `int`, but with the lambda missing a colon expression after. In short, we
  // can't decide beforehand whether the `:` is part of the type of the last
  // parameter or whether it is the lambda body. Since the parameter list parses
  // eagerly we'll just have to detect that particular case by checking whether
  // the last parameter had a type and no default arg, and then extract those
  // from the param to use them as lambda bodies.
  bool expressionIsInParameterList = !parameterList.empty() &&
                                     parameterList.back().maybeType &&
                                     !parameterList.back().maybeDefault;
  std::optional<BaseToken> colon;
  if (!expressionIsInParameterList) {
    colon = expect(TokenType::Colon);
    if (!colon)
      return std::nullopt;
  } else {
    colon = parameterList.back().maybeColon;
    parameterList.back().maybeColon.reset();
  }

  IntrVarPtr<Syntax::Expression> expression;
  if (!expressionIsInParameterList) {
    auto temp = parseExpression();
    if (!temp)
      return std::nullopt;
    expression = std::move(*temp);
  } else {
    expression = std::move(parameterList.back().maybeType);
  }

  return Syntax::Lambda{{},     std::move(*keyword),   std::move(parameterList),
                        *colon, std::move(expression), {}};
}

std::optional<pylir::Syntax::Comprehension> pylir::Parser::parseComprehension(
    IntrVarPtr<Syntax::Expression>&& assignmentExpression) {
  auto compFor = parseCompFor();
  if (!compFor)
    return std::nullopt;
  return Syntax::Comprehension{std::move(assignmentExpression),
                               std::move(*compFor)};
}

std::optional<pylir::Syntax::CompFor> pylir::Parser::parseCompFor() {
  std::optional<Token> awaitToken = maybeConsume(TokenType::AwaitKeyword);
  auto forToken = expect(TokenType::ForKeyword);
  if (!forToken)
    return std::nullopt;
  auto targetList = parseTargetList(*forToken);
  if (!targetList)
    return std::nullopt;
  auto inToken = expect(TokenType::InKeyword);
  if (!inToken)
    return std::nullopt;
  auto orTest = parseOrTest();
  if (!orTest)
    return std::nullopt;
  if (!peekedIs({TokenType::ForKeyword, TokenType::IfKeyword,
                 TokenType::AwaitKeyword}))
    return Syntax::CompFor{std::move(awaitToken),  std::move(*forToken),
                           std::move(*targetList), std::move(*inToken),
                           std::move(*orTest),     std::monostate{}};

  std::variant<std::monostate, std::unique_ptr<Syntax::CompFor>,
               std::unique_ptr<Syntax::CompIf>>
      trail;
  if (m_current->getTokenType() == TokenType::IfKeyword) {
    auto compIf = parseCompIf();
    if (!compIf)
      return std::nullopt;
    trail = std::make_unique<Syntax::CompIf>(std::move(*compIf));
  } else {
    auto compFor = parseCompFor();
    if (!compFor)
      return std::nullopt;
    trail = std::make_unique<Syntax::CompFor>(std::move(*compFor));
  }
  return Syntax::CompFor{std::move(awaitToken),  std::move(*forToken),
                         std::move(*targetList), std::move(*inToken),
                         std::move(*orTest),     std::move(trail)};
}

std::optional<pylir::Syntax::CompIf> pylir::Parser::parseCompIf() {
  auto ifToken = expect(TokenType::IfKeyword);
  if (!ifToken)
    return std::nullopt;
  auto orTest = parseOrTest();
  if (!orTest)
    return std::nullopt;
  if (!peekedIs({TokenType::ForKeyword, TokenType::IfKeyword,
                 TokenType::AwaitKeyword}))
    return Syntax::CompIf{std::move(*ifToken), std::move(*orTest),
                          std::monostate{}};

  std::variant<std::monostate, std::unique_ptr<Syntax::CompFor>,
               std::unique_ptr<Syntax::CompIf>>
      trail;
  if (m_current->getTokenType() == TokenType::IfKeyword) {
    auto compIf = parseCompIf();
    if (!compIf)
      return std::nullopt;
    trail = std::make_unique<Syntax::CompIf>(std::move(*compIf));
  } else {
    auto compFor = parseCompFor();
    if (!compFor)
      return std::nullopt;
    trail = std::make_unique<Syntax::CompFor>(std::move(*compFor));
  }
  return Syntax::CompIf{std::move(*ifToken), std::move(*orTest),
                        std::move(trail)};
}

std::optional<pylir::IntrVarPtr<pylir::Syntax::Expression>>
pylir::Parser::parseStarredExpression(
    IntrVarPtr<Syntax::Expression>&& firstItem) {
  if (peekedIsNot(TokenType::Star) && !firstItem) {
    auto expression = parseAssignmentExpression();
    if (!expression)
      return std::nullopt;
    firstItem = std::move(*expression);
  }
  std::vector<Syntax::StarredItem> items;
  if (firstItem) {
    if (!maybeConsume(TokenType::Comma))
      return std::move(firstItem);

    items.push_back(Syntax::StarredItem{std::nullopt, std::move(firstItem)});
  }
  while (peekedIs(firstInStarredItem)) {
    auto item = parseStarredItem();
    if (!item)
      return std::nullopt;
    // If a comma doesn't follow, then it's the last optional trailing
    // starred_item
    items.emplace_back(std::move(*item));
    if (!maybeConsume(TokenType::Comma)) {
      // if there were no leading expressions (aka no commas) and it is an
      // expansion (with a star), then it's a syntax error as those are only
      // possible when commas are involved (to form a tuple).
      // TODO: Better error message
      if (items.size() == 1 && item->maybeStar) {
        expect(TokenType::Comma);
        return std::nullopt;
      }
      break;
    }
  }
  return makeNode<Syntax::TupleConstruct>(std::nullopt, std::move(items),
                                          std::nullopt);
}

std::optional<pylir::Syntax::StarredItem> pylir::Parser::parseStarredItem() {
  if (auto star = maybeConsume(TokenType::Star)) {
    auto expression = parseOrExpr();
    if (!expression)
      return std::nullopt;
    return Syntax::StarredItem{std::move(star), std::move(*expression)};
  }
  auto assignment = parseAssignmentExpression();
  if (!assignment)
    return std::nullopt;
  return Syntax::StarredItem{std::nullopt, std::move(*assignment)};
}

std::optional<std::vector<pylir::Syntax::StarredItem>>
pylir::Parser::parseStarredList(
    std::optional<Syntax::StarredItem>&& firstItem) {
  return parseCommaList(pylir::bind_front(&Parser::parseStarredItem, this),
                        &firstInStarredItem, std::move(firstItem));
}
