//  Licensed under the Apache License v2.0 with LLVM Exceptions.
//  See https://llvm.org/LICENSE.txt for license information.
//  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "Parser.hpp"

#include <pylir/Diagnostics/DiagnosticMessages.hpp>
#include <pylir/Support/Functional.hpp>

std::optional<pylir::Syntax::AssignmentStmt>
pylir::Parser::parseAssignmentStmt(IntrVarPtr<Syntax::Target>&& firstItem) {
  std::vector<std::pair<IntrVarPtr<Syntax::Target>, Token>> targets;
  bool hadFirst = firstItem != nullptr;
  if (firstItem) {
    auto assignment = expect(TokenType::Assignment);
    if (!assignment)
      return std::nullopt;
    targets.emplace_back(std::move(firstItem), std::move(*assignment));
  }
  IntrVarPtr<Syntax::Expression> leftOverStarredExpression;
  do {
    if (hadFirst && !peekedIs(firstInTarget))
      break;
    auto starredExpression = parseStarredExpression();
    if (!starredExpression)
      return std::nullopt;
    auto assignment = maybeConsume(TokenType::Assignment);
    if (!assignment) {
      leftOverStarredExpression = std::move(*starredExpression);
      break;
    }
    checkTarget(**starredExpression, *assignment);
    targets.emplace_back(std::move(*starredExpression), *assignment);
  } while (peekedIs(firstInTarget));
  if (leftOverStarredExpression)
    return Syntax::AssignmentStmt{
        {}, std::move(targets), nullptr, std::move(leftOverStarredExpression)};

  if (peekedIs(TokenType::YieldKeyword)) {
    auto yieldExpr = parseYieldExpression();
    if (!yieldExpr)
      return std::nullopt;
    return Syntax::AssignmentStmt{
        {},
        std::move(targets),
        nullptr,
        std::make_unique<Syntax::Yield>(std::move(*yieldExpr))};
  }

  auto starredExpression = parseStarredExpression();
  if (!starredExpression)
    return std::nullopt;
  return Syntax::AssignmentStmt{
      {}, std::move(targets), nullptr, std::move(*starredExpression)};
}

std::optional<pylir::IntrVarPtr<pylir::Syntax::SimpleStmt>>
pylir::Parser::parseSimpleStmt() {
  if (m_current == m_lexer.end())
    return makeNode<Syntax::ExpressionStmt>(nullptr);

  switch (m_current->getTokenType()) {
  case TokenType::AssertKeyword: {
    auto assertStmt = parseAssertStmt();
    if (!assertStmt)
      return std::nullopt;
    return std::make_unique<Syntax::AssertStmt>(std::move(*assertStmt));
  }
  case TokenType::PassKeyword:
  case TokenType::BreakKeyword:
  case TokenType::ContinueKeyword:
    return makeNode<Syntax::SingleTokenStmt>(*m_current++);
  case TokenType::DelKeyword: {
    auto delKeyword = *m_current++;
    auto targetList = parseTargetList(delKeyword, true);
    if (!targetList)
      return std::nullopt;
    return makeNode<Syntax::DelStmt>(delKeyword, std::move(*targetList));
  }
  case TokenType::ReturnKeyword: {
    auto returnKeyword = *m_current++;
    if (!peekedIs(firstInExpression))
      return makeNode<Syntax::ReturnStmt>(returnKeyword, nullptr);

    auto expressionList = parseExpressionList();
    if (!expressionList)
      return std::nullopt;
    return makeNode<Syntax::ReturnStmt>(returnKeyword,
                                        std::move(*expressionList));
  }
  case TokenType::YieldKeyword: {
    auto yieldExpr = parseYieldExpression();
    if (!yieldExpr)
      return std::nullopt;
    return makeNode<Syntax::ExpressionStmt>(
        std::make_unique<Syntax::Yield>(std::move(*yieldExpr)));
  }
  case TokenType::RaiseKeyword: {
    auto raise = *m_current++;
    if (!peekedIs(firstInExpression))
      return makeNode<Syntax::RaiseStmt>(raise, nullptr, nullptr);

    auto expression = parseExpression();
    if (!expression)
      return std::nullopt;
    if (!maybeConsume(TokenType::FromKeyword))
      return makeNode<Syntax::RaiseStmt>(raise, std::move(*expression),
                                         nullptr);

    auto source = parseExpression();
    if (!source)
      return std::nullopt;
    return makeNode<Syntax::RaiseStmt>(raise, std::move(*expression),
                                       std::move(*source));
  }
  case TokenType::GlobalKeyword:
  case TokenType::NonlocalKeyword: {
    auto keyword = *m_current++;
    auto identifier = expect(TokenType::Identifier);
    if (!identifier)
      return std::nullopt;
    std::vector<IdentifierToken> identifiers;
    identifiers.emplace_back(std::move(*identifier));
    while (maybeConsume(TokenType::Comma)) {
      auto another = expect(TokenType::Identifier);
      if (!another)
        return std::nullopt;
      identifiers.emplace_back(std::move(*another));
    }
    if (keyword.getTokenType() == TokenType::NonlocalKeyword)
      return makeNode<Syntax::GlobalOrNonLocalStmt>(keyword,
                                                    std::move(identifiers));

    return makeNode<Syntax::GlobalOrNonLocalStmt>(keyword,
                                                  std::move(identifiers));
  }
  case TokenType::FromKeyword:
  case TokenType::ImportKeyword: {
    auto import = parseImportStmt();
    if (!import)
      return std::nullopt;
    if (auto* fromImportAs =
            std::get_if<Syntax::ImportStmt::FromImport>(&import->variant);
        fromImportAs && fromImportAs->relativeModule.dots.empty() &&
        fromImportAs->relativeModule.module &&
        fromImportAs->relativeModule.module->identifiers.size() == 1 &&
        fromImportAs->relativeModule.module->identifiers.back().getValue() ==
            "__future__") {
      auto check = [&](const IdentifierToken& identifierToken) {
#define HANDLE_FEATURE(x)               \
  if (identifierToken.getValue() == #x) \
    return;

#define HANDLE_REQUIRED_FEATURE(x)        \
  if (identifierToken.getValue() == #x) { \
    m_##x = true;                         \
    return;                               \
  }
#include "Features.def"
        createError(identifierToken, Diag::UNKNOWN_FEATURE_N,
                    identifierToken.getValue())
            .addHighlight(identifierToken);
      };
      llvm::for_each(llvm::make_first_range(fromImportAs->imports), check);
      return makeNode<Syntax::FutureStmt>(
          fromImportAs->from,
          fromImportAs->relativeModule.module->identifiers.front(),
          fromImportAs->import, std::move(fromImportAs->imports));
    }
    return std::make_unique<Syntax::ImportStmt>(std::move(*import));
  }
  case TokenType::SyntaxError: return std::nullopt;
  default:
    // Starred expression is a super set of both `target` and `augtarget`.
    auto starredExpression = parseStarredExpression();
    if (!starredExpression)
      return std::nullopt;

    if (m_current == m_lexer.end()) {
      return makeNode<Syntax::ExpressionStmt>(std::move(*starredExpression));
    }

    switch (m_current->getTokenType()) {
    case TokenType::Assignment: {
      // If an assignment follows, check whether the starred expression could be
      // a target list
      checkTarget(**starredExpression, *m_current);
      auto assignmentStmt = parseAssignmentStmt(std::move(*starredExpression));
      if (!assignmentStmt)
        return std::nullopt;
      return std::make_unique<Syntax::AssignmentStmt>(
          std::move(*assignmentStmt));
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
    case TokenType::BitOrAssignment: {
      // needs to be an augtarget then
      checkAug(**starredExpression, *m_current);
      if (auto colon = maybeConsume(TokenType::Colon)) {
        auto expression = parseExpression();
        if (!expression)
          return std::nullopt;
        std::vector<std::pair<IntrVarPtr<Syntax::Target>, Token>> vector;
        vector.emplace_back(std::move(*starredExpression), std::move(*colon));
        if (!maybeConsume(TokenType::Assignment))
          return makeNode<Syntax::AssignmentStmt>(
              std::move(vector), std::move(*expression), nullptr);

        if (peekedIs(TokenType::YieldKeyword)) {
          auto yield = parseYieldExpression();
          if (!yield)
            return std::nullopt;
          return makeNode<Syntax::AssignmentStmt>(
              std::move(vector), std::move(*expression),
              std::make_unique<Syntax::Yield>(std::move(*yield)));
        }
        auto starred = parseStarredExpression();
        if (!starred)
          return std::nullopt;
        return makeNode<Syntax::AssignmentStmt>(
            std::move(vector), std::move(*expression), std::move(*starred));
      }
      auto augOp = *m_current++;
      std::vector<std::pair<IntrVarPtr<Syntax::Target>, Token>> vector;
      vector.emplace_back(std::move(*starredExpression), augOp);
      if (peekedIs(TokenType::YieldKeyword)) {
        auto yield = parseYieldExpression();
        if (!yield)
          return std::nullopt;
        return makeNode<Syntax::AssignmentStmt>(
            std::move(vector), nullptr,
            std::make_unique<Syntax::Yield>(std::move(*yield)));
      }
      auto expressionList = parseExpressionList();
      if (!expressionList)
        return std::nullopt;
      return makeNode<Syntax::AssignmentStmt>(std::move(vector), nullptr,
                                              std::move(*expressionList));
    }
    default:
      return makeNode<Syntax::ExpressionStmt>(std::move(*starredExpression));
    }
  }
}

std::optional<pylir::Syntax::AssertStmt> pylir::Parser::parseAssertStmt() {
  auto assertKeyword = expect(TokenType::AssertKeyword);
  if (!assertKeyword)
    return std::nullopt;
  auto expression = parseExpression();
  if (!expression)
    return std::nullopt;
  if (!maybeConsume(TokenType::Comma))
    return Syntax::AssertStmt{
        {}, std::move(*assertKeyword), std::move(*expression), nullptr};

  auto message = parseExpression();
  if (!message)
    return std::nullopt;
  return Syntax::AssertStmt{{},
                            std::move(*assertKeyword),
                            std::move(*expression),
                            std::move(*message)};
}

namespace {
using namespace pylir;

struct Visitor {
  Parser& parser;
  const Token& assignOp;
  enum Mode { Augmented, Assignment, Del } mode;

  // Disallows implicit conversions
  template <class T,
            std::enable_if_t<std::is_same_v<T, Syntax::Expression>>* = nullptr>
  void visit(const T& expression) {
    return expression.match([&](const auto& sub) { return visit(sub); });
  }

  void visit(const Syntax::AttributeRef&) {}

  void visit(const Syntax::Subscription&) {}

  void visit(const Syntax::Slice&) {}

  void visit(llvm::ArrayRef<Syntax::StarredItem> starredItems) {
    PYLIR_ASSERT(mode != Augmented);
    const Syntax::StarredItem* prevStar = nullptr;
    bool multipleIterableErrorTriggered = false;
    for (const auto& iter : starredItems) {
      if (iter.maybeStar) {
        if (mode == Del) {
          parser
              .createError(*iter.maybeStar,
                           Diag::CANNOT_DELETE_ITERABLE_UNPACKING)
              .addHighlight(*iter.maybeStar, *iter.expression)
              .addHighlight(assignOp, Diag::flags::secondaryColour);
        } else if (prevStar) {
          // No need to emit this error more than once.
          if (!multipleIterableErrorTriggered) {
            multipleIterableErrorTriggered = true;
            parser
                .createError(
                    *iter.maybeStar,
                    Diag::ONLY_ONE_ITERABLE_UNPACKING_POSSIBLE_IN_ASSIGNMENT)
                .addHighlight(*iter.maybeStar, *iter.expression)
                .addHighlight(assignOp, Diag::flags::secondaryColour)
                .addNote(*prevStar->maybeStar, Diag::PREVIOUS_OCCURRENCE_HERE)
                .addHighlight(*prevStar);
          }
        } else {
          prevStar = &iter;
        }
      }
      visit(*iter.expression);
    }
  }

  void visit(const Syntax::TupleConstruct& tupleConstruct) {
    if (mode == Augmented) {
      switch (tupleConstruct.items.size()) {
      case 0:
        parser
            .createError(tupleConstruct,
                         Diag::OPERATOR_N_CANNOT_ASSIGN_TO_EMPTY_TUPLE,
                         assignOp.getTokenType())
            .addHighlight(tupleConstruct)
            .addHighlight(assignOp, Diag::flags::secondaryColour);
        break;
      case 1:
        parser
            .createError(tupleConstruct,
                         Diag::OPERATOR_N_CANNOT_ASSIGN_TO_SINGLE_TUPLE_ELEMENT,
                         assignOp.getTokenType())
            .addHighlight(tupleConstruct)
            .addHighlight(assignOp, Diag::flags::secondaryColour);
        break;
      default:
        parser
            .createError(tupleConstruct,
                         Diag::OPERATOR_N_CANNOT_ASSIGN_TO_MULTIPLE_VARIABLES,
                         assignOp.getTokenType())
            .addHighlight(tupleConstruct)
            .addHighlight(assignOp, Diag::flags::secondaryColour);
        break;
      }
      return;
    }
    visit(tupleConstruct.items);
  }

  void visit(const Syntax::DictDisplay& expression) {
    parser.createError(expression, Diag::CANNOT_ASSIGN_TO_N, "dictionary")
        .addHighlight(expression)
        .addHighlight(assignOp, Diag::flags::secondaryColour);
  }

  void visit(const Syntax::SetDisplay& expression) {
    parser.createError(expression, Diag::CANNOT_ASSIGN_TO_N, "set")
        .addHighlight(expression)
        .addHighlight(assignOp, Diag::flags::secondaryColour);
  }

  void visit(const Syntax::ListDisplay& expression) {
    if (std::holds_alternative<Syntax::Comprehension>(expression.variant) ||
        mode == Augmented) {
      parser
          .createError(expression, Diag::CANNOT_ASSIGN_TO_N,
                       "list comprehension")
          .addHighlight(expression)
          .addHighlight(assignOp, Diag::flags::secondaryColour);
      return;
    }
    visit(pylir::get<std::vector<Syntax::StarredItem>>(expression.variant));
  }

  void visit(const Syntax::Intrinsic& intrinsic) {
    parser.createError(intrinsic, Diag::CANNOT_ASSIGN_TO_N, "intrinsic")
        .addHighlight(intrinsic)
        .addHighlight(assignOp, Diag::flags::secondaryColour);
  }

  void visit(const Syntax::Yield& expression) {
    parser.createError(expression, Diag::CANNOT_ASSIGN_TO_N, "yield expression")
        .addHighlight(expression)
        .addHighlight(assignOp, Diag::flags::secondaryColour);
  }

  void visit(const Syntax::Generator& expression) {
    parser
        .createError(expression, Diag::CANNOT_ASSIGN_TO_N,
                     "generator expression")
        .addHighlight(expression)
        .addHighlight(assignOp, Diag::flags::secondaryColour);
  }

  void visit(const Syntax::BinOp& binOp) {
    parser
        .createError(binOp.operation,
                     Diag::CANNOT_ASSIGN_TO_RESULT_OF_OPERATOR_N,
                     binOp.operation.getTokenType())
        .addHighlight(binOp.operation)
        .addHighlight(assignOp, Diag::flags::secondaryColour);
  }

  void visit(const Syntax::Lambda& lambda) {
    parser
        .createError(lambda, Diag::CANNOT_ASSIGN_TO_RESULT_OF_N,
                     "lambda expression")
        .addHighlight(lambda)
        .addHighlight(assignOp, Diag::flags::secondaryColour);
  }

  void visit(const Syntax::Atom& expression) {
    switch (expression.token.getTokenType()) {
    case TokenType::StringLiteral:
    case TokenType::IntegerLiteral:
    case TokenType::FloatingPointLiteral:
    case TokenType::NoneKeyword:
    case TokenType::FalseKeyword:
    case TokenType::TrueKeyword:
    case TokenType::ByteLiteral:
    case TokenType::ComplexLiteral:
      parser.createError(expression.token, Diag::CANNOT_ASSIGN_TO_N, "literal")
          .addHighlight(expression.token)
          .addHighlight(assignOp, Diag::flags::secondaryColour);
    default: break;
    }
  }

  void visit(const Syntax::Call& call) {
    parser
        .createError(call.openParenth, Diag::CANNOT_ASSIGN_TO_RESULT_OF_N,
                     "call")
        .addHighlight(call.openParenth, call)
        .addHighlight(assignOp, Diag::flags::secondaryColour);
  }

  void visit(const Syntax::UnaryOp& expression) {
    parser
        .createError(expression.operation,
                     Diag::CANNOT_ASSIGN_TO_RESULT_OF_UNARY_OPERATOR_N,
                     expression.operation.getTokenType())
        .addHighlight(expression.operation)
        .addHighlight(assignOp, Diag::flags::secondaryColour);
  }

  void visit(const Syntax::Comparison& comparison) {
    // It's better looking, but still a technically arbitrary decision, but
    // we'll emit the diagnostic only once for the very last use of a comparison
    // operator.
    const auto& back = comparison.rest.back().first;
    if (back.secondToken) {
      parser
          .createError(back.firstToken, Diag::CANNOT_ASSIGN_TO_RESULT_OF_N,
                       fmt::format(FMT_STRING("'{} {}'"),
                                   back.firstToken.getTokenType(),
                                   back.secondToken->getTokenType()))
          .addHighlight(back.firstToken, *back.secondToken)
          .addHighlight(assignOp, Diag::flags::secondaryColour);
      return;
    }

    parser
        .createError(back.firstToken,
                     Diag::CANNOT_ASSIGN_TO_RESULT_OF_OPERATOR_N,
                     back.firstToken.getTokenType())
        .addHighlight(back.firstToken)
        .addHighlight(assignOp, Diag::flags::secondaryColour);
  }

  void visit(const Syntax::Conditional& expression) {
    parser
        .createError(expression.ifToken, Diag::CANNOT_ASSIGN_TO_RESULT_OF_N,
                     "conditional expression")
        .addHighlight(expression.ifToken, *expression.elseValue)
        .addHighlight(assignOp, Diag::flags::secondaryColour);
  }

  void visit(const Syntax::Assignment& assignment) {
    parser
        .createError(assignment.walrus,
                     Diag::CANNOT_ASSIGN_TO_RESULT_OF_OPERATOR_N,
                     TokenType::Walrus)
        .addHighlight(assignment.walrus)
        .addHighlight(assignOp, Diag::flags::secondaryColour);
  }
};
} // namespace

std::optional<pylir::IntrVarPtr<Syntax::Target>>
pylir::Parser::parseTarget(const pylir::Token& assignmentLikeToken) {
  auto expression = parsePrimary();
  if (!expression)
    return std::nullopt;
  checkTarget(**expression, assignmentLikeToken);
  return expression;
}

std::optional<pylir::IntrVarPtr<pylir::Syntax::Target>>
pylir::Parser::parseTargetList(const pylir::Token& assignmentLikeToken,
                               bool delStmt) {
  IntrVarPtr<Syntax::Target> target;
  {
    bool lastWasComma = false;
    std::vector<IntrVarPtr<Syntax::Target>> subTargets;
    while (subTargets.empty() ||
           (m_current != m_lexer.end() &&
            m_current->getTokenType() == TokenType::Comma)) {
      if (!subTargets.empty()) {
        m_current++;
        if (!firstInTarget(m_current->getTokenType())) {
          lastWasComma = true;
          break;
        }
      }
      auto other = parsePrimary();
      if (!other)
        return std::nullopt;
      subTargets.emplace_back(std::move(*other));
    }
    if (lastWasComma || subTargets.size() > 1) {
      std::vector<Syntax::StarredItem> starredItems(subTargets.size());
      std::transform(
          std::move_iterator(subTargets.begin()),
          std::move_iterator(subTargets.end()), starredItems.begin(),
          [](IntrVarPtr<Syntax::Target>&& target) {
            return Syntax::StarredItem{std::nullopt, std::move(target)};
          });
      target = makeNode<Syntax::TupleConstruct>(
          std::nullopt, std::move(starredItems), std::nullopt);
    } else {
      target = std::move(subTargets[0]);
    }
  }

  checkTarget(*target, assignmentLikeToken, delStmt);
  return target;
}

void pylir::Parser::checkAug(const Syntax::Expression& starredExpression,
                             const Token& assignOp) {
  Visitor{*this, assignOp, Visitor::Augmented}.visit(starredExpression);
}

void pylir::Parser::checkTarget(const Syntax::Expression& starredExpression,
                                const Token& assignOp, bool deleteStmt) {
  Visitor{*this, assignOp, deleteStmt ? Visitor::Del : Visitor::Assignment}
      .visit(starredExpression);
}

std::optional<Syntax::ImportStmt> pylir::Parser::parseImportStmt() {
  auto parseModule = [&]() -> std::optional<Syntax::ImportStmt::Module> {
    std::vector<IdentifierToken> identifiers;
    do {
      auto identifier = expect(TokenType::Identifier);
      if (!identifier)
        return std::nullopt;
      identifiers.emplace_back(std::move(*identifier));
      if (!maybeConsume(TokenType::Dot))
        return Syntax::ImportStmt::Module{std::move(identifiers)};

    } while (true);
  };

  auto parseRelativeModule =
      [&]() -> std::optional<Syntax::ImportStmt::RelativeModule> {
    std::vector<BaseToken> dots;
    while (auto dot = maybeConsume(TokenType::Dot))
      dots.emplace_back(*dot);

    if (!dots.empty() && !peekedIs(TokenType::Identifier))
      return Syntax::ImportStmt::RelativeModule{std::move(dots), std::nullopt};

    auto module = parseModule();
    if (!module)
      return std::nullopt;
    return Syntax::ImportStmt::RelativeModule{std::move(dots),
                                              std::move(*module)};
  };
  if (m_current == m_lexer.end()) {
    createError(endOfFileLoc(), Diag::EXPECTED_N,
                fmt::format("{:q} or {:q}", TokenType::ImportKeyword,
                            TokenType::FromKeyword))
        .addHighlight(endOfFileLoc());
    return std::nullopt;
  }
  switch (m_current->getTokenType()) {
  case TokenType::ImportKeyword: {
    auto import = *m_current++;

    bool first = true;
    std::vector<
        std::pair<Syntax::ImportStmt::Module, std::optional<IdentifierToken>>>
        modules;
    while (first || maybeConsume(TokenType::Comma)) {
      if (first)
        first = false;

      auto nextModule = parseModule();
      if (!nextModule)
        return std::nullopt;
      std::optional<IdentifierToken> nextName;
      if (maybeConsume(TokenType::AsKeyword)) {
        auto identifier = expect(TokenType::Identifier);
        if (!identifier)
          return std::nullopt;
        nextName.emplace(std::move(*identifier));
      }
      modules.emplace_back(std::move(*nextModule), std::move(nextName));
    }
    return Syntax::ImportStmt{
        {},
        Syntax::ImportStmt::ImportAs{std::move(import), std::move(modules)}};
  }
  case TokenType::FromKeyword: {
    auto from = *m_current++;
    auto relative = parseRelativeModule();
    if (!relative)
      return std::nullopt;
    auto import = expect(TokenType::ImportKeyword);
    if (!import)
      return std::nullopt;
    if (auto star = maybeConsume(TokenType::Star)) {
      return Syntax::ImportStmt{
          {},
          Syntax::ImportStmt::ImportAll{from, std::move(*relative), *import,
                                        *star}};
    }

    std::optional<BaseToken> openParenth =
        maybeConsume(TokenType::OpenParentheses);

    bool first = true;
    std::vector<std::pair<IdentifierToken, std::optional<IdentifierToken>>>
        imports;
    while (first || maybeConsume(TokenType::Comma)) {
      if (first)
        first = false;
      else if (openParenth && peekedIs(TokenType::CloseParentheses))
        break;
      auto imported = expect(TokenType::Identifier);
      if (!imported)
        return std::nullopt;
      std::optional<IdentifierToken> nextName;
      if (maybeConsume(TokenType::AsKeyword)) {
        auto identifier = expect(TokenType::Identifier);
        if (!identifier)
          return std::nullopt;
        nextName.emplace(std::move(*identifier));
      }
      imports.emplace_back(std::move(*imported), std::move(nextName));
    }
    if (openParenth) {
      auto close = expect(TokenType::CloseParentheses);
      if (!close)
        return std::nullopt;
    }
    return Syntax::ImportStmt{
        {},
        Syntax::ImportStmt::FromImport{from, std::move(*relative), *import,
                                       std::move(imports)}};
  }
  case TokenType::SyntaxError: return std::nullopt;
  default:
    createError(*m_current, Diag::EXPECTED_N_INSTEAD_OF_N,
                fmt::format("{:q} or {:q}", TokenType::ImportKeyword,
                            TokenType::FromKeyword),
                m_current->getTokenType())
        .addHighlight(*m_current);
    return std::nullopt;
  }
}
