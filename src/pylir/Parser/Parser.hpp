//  Licensed under the Apache License v2.0 with LLVM Exceptions.
//  See https://llvm.org/LICENSE.txt for license information.
//  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <pylir/Diagnostics/DiagnosticsBuilder.hpp>
#include <pylir/Diagnostics/Document.hpp>
#include <pylir/Lexer/Lexer.hpp>

#include <unordered_map>
#include <unordered_set>

#include <tcb/span.hpp>

#include "Syntax.hpp"

namespace pylir {
class Parser {
#define HANDLE_FEATURE(x)
#define HANDLE_REQUIRED_FEATURE(x) bool m_##x : 1;
#include "Features.def"
  Lexer m_lexer;
  Lexer::iterator m_current;

  /// Attempts to peek ahead in the token stream and returns the token if it
  /// matches the given predicate. If it does not match, or there are no more
  /// tokens it returns an empty optional.
  template <class P>
  [[nodiscard]] std::optional<Token> peekedIs(P predicate) {
    if (m_current == m_lexer.end())
      return std::nullopt;

    if (predicate(m_current->getTokenType()))
      return *m_current;

    return std::nullopt;
  }

  /// Attempts to peek ahead in the token stream and returns the token if it is
  /// one of the listed token types. If it does not match, or there are no more
  /// tokens it returns an empty optional.
  [[nodiscard]] std::optional<Token>
  peekedIs(std::initializer_list<TokenType> tokenTypes) {
    return peekedIs(
        [=](TokenType type) { return llvm::is_contained(tokenTypes, type); });
  }

  /// Attempts to peek ahead in the token stream and returns the token if it
  /// matches the token type. If it does not match, or there are no more tokens
  /// it returns an empty optional.
  [[nodiscard]] std::optional<Token> peekedIs(TokenType tokenType) {
    return peekedIs({tokenType});
  }

  /// Checks whether the next token does not match the predicate. This differs
  /// from `!peekedIs(predicate)' as it still returns false if the token stream
  /// is empty.
  template <class P>
  [[nodiscard]] bool peekedIsNot(P predicate) {
    if (m_current == m_lexer.end())
      return false;

    return !peekedIs(predicate);
  }

  /// Checks whether the next token is none of the given token types. This
  /// differs from `!peekedIs(tokenTypes)' as it still returns false if the
  /// token stream is empty.
  [[nodiscard]] bool peekedIsNot(std::initializer_list<TokenType> tokenTypes) {
    if (m_current == m_lexer.end())
      return false;

    return !peekedIs(tokenTypes);
  }

  /// Checks whether the next token is not of the given token type. This differs
  /// from `!peekedIs(tokenType)' as it still returns false if the token stream
  /// is empty.
  [[nodiscard]] bool peekedIsNot(TokenType tokenType) {
    if (m_current == m_lexer.end())
      return false;

    return !peekedIs(tokenType);
  }

  /// Attempts to peek ahead in the token stream and checks if the token matches
  /// the given predicate. If it does it returns the token and advances the
  /// token stream to the next token. If it does not match, or there are no more
  /// tokens it returns an empty optional.
  std::optional<Token>
  maybeConsume(llvm::function_ref<bool(TokenType)> predicate) {
    auto result = peekedIs(predicate);
    if (result)
      m_current++;

    return result;
  }

  /// Attempts to peek ahead in the token stream and checks if the token is one
  /// of the given token types. If it does it returns the token and advances the
  /// token stream to the next token. If it does not match, or there are no more
  /// tokens it returns an empty optional.
  std::optional<Token>
  maybeConsume(std::initializer_list<TokenType> tokenTypes) {
    auto result = peekedIs(tokenTypes);
    if (result)
      m_current++;

    return result;
  }

  /// Attempts to peek ahead in the token stream and checks if the token is the
  /// given token type. If it does it returns the token and advances the token
  /// stream to the next token. If it does not match, or there are no more
  /// tokens it returns an empty optional.
  std::optional<Token> maybeConsume(TokenType tokenType) {
    auto result = peekedIs(tokenType);
    if (result)
      m_current++;

    return result;
  }

  [[nodiscard]] std::pair<std::size_t, std::size_t> endOfFileLoc() const {
    return m_lexer.getDiagManager().getDocument().getEndOfFileLoc();
  }

  std::optional<Token> expect(TokenType tokenType);

  template <class T, class... Args>
  static std::unique_ptr<T> makeNode(Args&&... args) {
    return std::unique_ptr<T>(new T{{}, std::forward<Args>(args)...});
  }

  bool lookaheadEquals(tcb::span<const TokenType> tokens);

  template <class ParseFunc, class CheckFunc>
  auto parseCommaList(
      ParseFunc parseFunc, CheckFunc checkFunc,
      std::optional<typename std::invoke_result_t<ParseFunc>::value_type>&&
          optionalFirst = std::nullopt,
      TokenType tokenType = TokenType::Comma)
      -> std::optional<
          std::vector<typename std::invoke_result_t<ParseFunc>::value_type>> {
    using T = typename std::invoke_result_t<ParseFunc>::value_type;
    if (!optionalFirst) {
      auto first = parseFunc();
      if (!first)
        return std::nullopt;

      optionalFirst = std::move(*first);
    }
    std::vector<T> rest;
    rest.push_back(std::move(*optionalFirst));
    while (maybeConsume(tokenType)) {
      if (!checkFunc(m_current->getTokenType()))
        break;

      auto other = parseFunc();
      if (!other)
        return std::nullopt;

      rest.emplace_back(std::move(*other));
    }
    return rest;
  }

  template <auto parseLesser, TokenType... allowed>
  std::optional<IntrVarPtr<Syntax::Expression>> parseGenericBinOp() {
    auto first = (this->*parseLesser)();
    if (!first)
      return std::nullopt;

    IntrVarPtr<Syntax::Expression> current{std::move(*first)};
    while (auto op = maybeConsume({allowed...})) {
      auto rhs = (this->*parseLesser)();
      if (!rhs)
        return std::nullopt;

      current = makeNode<Syntax::BinOp>(std::move(current), std::move(*op),
                                        std::move(*rhs));
    }
    return std::move(current);
  }

  void checkAug(const Syntax::Expression& expression, const Token& assignOp);

  void checkTarget(const Syntax::Expression& expression, const Token& assignOp,
                   bool deleteStmt = false);

  static bool firstInAssignmentExpression(TokenType tokenType) {
    return firstInExpression(tokenType);
  }

  static bool firstInExpression(TokenType tokenType);

  static bool firstInStarredItem(TokenType tokenType) {
    return tokenType == TokenType::Star ||
           firstInAssignmentExpression(tokenType);
  }

  static bool firstInComprehension(TokenType tokenType) {
    return firstInAssignmentExpression(tokenType);
  }

  static bool firstInCompFor(TokenType tokenType) {
    switch (tokenType) {
    case TokenType::ForKeyword:
    case TokenType::AsyncKeyword: return true;
    default: return false;
    }
  }

  static bool firstInTarget(TokenType tokenType);

  static bool firstInSimpleStmt(TokenType tokenType);

  static bool firstInCompoundStmt(TokenType tokenType);

public:
  explicit Parser(Diag::DiagnosticsDocManager<>& diagManager)
      :
#define HANDLE_FEATURE(x)
#define HANDLE_REQUIRED_FEATURE(x) m_##x{true},
#include "Features.def"
        m_lexer(diagManager), m_current(m_lexer.begin()) {
  }

  template <class T, class S, class... Args>
  [[nodiscard]] auto createError(const T& location, const S& message,
                                 Args&&... args) const {
    return Diag::DiagnosticsBuilder(m_lexer.getDiagManager(),
                                    Diag::Severity::Error, location, message,
                                    std::forward<Args>(args)...);
  }

  /**
   * yield_expression ::=  "yield" [expression_list | "from" expression]
   */
  std::optional<Syntax::Yield> parseYieldExpression();

  /**
   * atom ::=  identifier | literal | enclosure
   *
   * literal ::=  stringliteral | bytesliteral | integer | floatnumber |
   * imagnumber | None | True | False
   */
  std::optional<IntrVarPtr<Syntax::Expression>> parseAtom();

  /**
   * attributeref ::=  primary "." identifier
   */
  std::optional<Syntax::AttributeRef>
  parseAttributeRef(IntrVarPtr<Syntax::Expression>&& expression);

  /**
   * argument_list        ::=  positional_arguments ["," starred_and_keywords]
   * ["," keywords_arguments] |  starred_and_keywords ["," keywords_arguments]
   *                        |  keywords_arguments
   * positional_arguments ::=  positional_item { "," positional_item }
   * positional_item      ::=  assignment_expression | "*" expression
   * starred_and_keywords ::=  ("*" expression | keyword_item)
   *                           { "," "*" expression | "," keyword_item }
   * keywords_arguments   ::=  (keyword_item | "**" expression)
   *                           { "," keyword_item | "," "**" expression }
   * keyword_item ::=  identifier "=" expression
   */
  std::optional<std::vector<Syntax::Argument>>
  parseArgumentList(IntrVarPtr<Syntax::Expression>&& firstAssignment = nullptr);

  /**
   * call ::=  primary "(" [argument_list [","] | comprehension] ")"
   */
  std::optional<Syntax::Call>
  parseCall(IntrVarPtr<Syntax::Expression>&& expression);

  /**
   * primary ::=  atom | attributeref | subscription | slicing | call
   */
  std::optional<IntrVarPtr<Syntax::Expression>> parsePrimary();

  /**
   * subscription ::=  primary "[" expression_list "]"
   * slicing      ::=  primary "[" slice_list "]"
   * slice_list   ::=  slice_item { "," slice_item } [","]
   * slice_item   ::=  expression | proper_slice
   * proper_slice ::=  [lower_bound] ":" [upper_bound] [ ":" [stride] ]
   * lower_bound  ::=  expression
   * upper_bound  ::=  expression
   * stride       ::=  expression
   */
  std::optional<IntrVarPtr<Syntax::Expression>>
  parseSlicingOrSubscription(IntrVarPtr<Syntax::Expression>&& expression);

  /**
   * await_expr ::=  "await" primary
   */
  std::optional<Syntax::UnaryOp> parseAwaitExpr();

  /**
   * power ::= (await_expr | primary) ["**" u_expr]
   */
  std::optional<IntrVarPtr<Syntax::Expression>> parsePower();

  /**
   * u_expr ::=  power
   *          |  "-" u_expr
   *          |  "+" u_expr
   *          |  "~" u_expr
   */
  std::optional<IntrVarPtr<Syntax::Expression>> parseUExpr();

  /**
   * m_expr ::=  u_expr
   *          |  m_expr "*" u_expr
   *          |  m_expr "@" m_expr
   *          |  m_expr "//" u_expr
   *          |  m_expr "/" u_expr
   *          |  m_expr "%" u_expr
   */
  std::optional<IntrVarPtr<Syntax::Expression>> parseMExpr();

  /**
   * a_expr ::=  m_expr
   *          |  a_expr "+" m_expr
   *          |  a_expr "-" m_expr
   */
  std::optional<IntrVarPtr<Syntax::Expression>> parseAExpr();

  /**
   * shift_expr ::=  a_expr
   *              |  shift_expr ("<<" | ">>") a_expr
   */
  std::optional<IntrVarPtr<Syntax::Expression>> parseShiftExpr();

  /**
   * and_expr ::=  shift_expr
   *            |  and_expr "&" shift_expr
   */
  std::optional<IntrVarPtr<Syntax::Expression>> parseAndExpr();

  /**
   * xor_expr ::=  and_expr
   *            |  xor_expr "^" and_expr
   */
  std::optional<IntrVarPtr<Syntax::Expression>> parseXorExpr();

  /**
   * or_expr  ::=  xor_expr
   *            | or_expr "|" xor_expr
   */
  std::optional<IntrVarPtr<Syntax::Expression>> parseOrExpr();

  /**
   * comparison    ::=  or_expr { comp_operator or_expr }
   * comp_operator ::=  "<" | ">" | "==" | ">=" | "<=" | "!=" | "is" ["not"] |
   * ["not"] "in"
   */
  std::optional<IntrVarPtr<Syntax::Expression>> parseComparison();

  /**
   * not_test ::=  comparison
   *            |  "not" not_test
   */
  std::optional<IntrVarPtr<Syntax::Expression>> parseNotTest();

  /**
   * and_test ::=  not_test
   *            |  and_test "and" not_test
   */
  std::optional<IntrVarPtr<Syntax::Expression>> parseAndTest();

  /**
   * or_test  ::=  and_test
   *            |  or_test "or" and_test
   */
  std::optional<IntrVarPtr<Syntax::Expression>> parseOrTest();

  /**
   * assignment_expression ::=  [identifier ":="] expression
   */
  std::optional<IntrVarPtr<Syntax::Expression>> parseAssignmentExpression();

  /**
   * conditional_expression ::=  or_test ["if" or_test "else" expression]
   */
  std::optional<IntrVarPtr<Syntax::Expression>> parseConditionalExpression();

  /**
   * expression ::=  conditional_expression
   *              |  lambda_expr
   */
  std::optional<IntrVarPtr<Syntax::Expression>> parseExpression();

  /**
   * expression_list ::=  expression { "," expression } [","]
   */
  std::optional<IntrVarPtr<Syntax::Expression>> parseExpressionList();

  /**
   * lambda_expr ::=  "lambda" [parameter_list] ":" expression
   */
  std::optional<Syntax::Lambda> parseLambdaExpression();

  /**
   * starred_expression ::=  expression
   *                      |  { starred_item "," } [starred_item]
   *
   * Note: While this is the formal syntax described in the python reference, we
   * parse the second alternative with the last item not being optional unless
   * at least one comma has been parsed. Empty starred_expressions are instead
   * handled at it's uses grammar rules.
   */
  std::optional<IntrVarPtr<Syntax::Expression>>
  parseStarredExpression(IntrVarPtr<Syntax::Expression>&& firstItem = nullptr);

  /**
   * starred_item ::=  assignment_expression
   *                |  "*" or_expr
   */
  std::optional<Syntax::StarredItem> parseStarredItem();

  std::optional<std::vector<Syntax::StarredItem>> parseStarredList(
      std::optional<Syntax::StarredItem>&& firstItem = std::nullopt);

  /**
   * comp_for ::=  ["async"] "for" target_list "in" or_test [comp_iter]
   */
  std::optional<Syntax::CompFor> parseCompFor();

  /**
   * comp_if ::=  "if" or_test [comp_iter]
   */
  std::optional<Syntax::CompIf> parseCompIf();

  /**
   * comprehension ::=  assignment_expression comp_for
   */
  std::optional<Syntax::Comprehension>
  parseComprehension(IntrVarPtr<Syntax::Expression>&& assignmentExpression);

  /**
   * enclosure            ::=  parenth_form | list_display | dict_display |
   * set_display |  generator_expression | yield_atom
   *
   * parenth_form         ::=  "(" [starred_expression] ")"
   *
   * list_display         ::=  "[" [starred_list | comprehension] "]"
   *
   * set_display          ::=  "{" (starred_list | comprehension) "}"
   *
   * dict_display         ::=  "{" [key_datum_list | dict_comprehension] "}"
   * key_datum_list       ::=  key_datum { "," key_datum } [","]
   * key_datum            ::=  expression ":" expression
   *                        |  "**" or_expr
   * dict_comprehension   ::=  expression ":" expression comp_for
   *
   * generator_expression ::=  "(" expression comp_for ")"
   *
   * yield_atom           ::=  "(" yield_expression ")"
   */
  std::optional<IntrVarPtr<Syntax::Expression>> parseEnclosure();

  /**
   * target      ::=  identifier
   *               |  "(" [target_list] ")"
   *               |  "[" [target_list] "]"
   *               |  attributeref
   *               |  subscription
   *               |  slicing
   *               |  "*" target
   *
   * target_list ::= target { "," target } [","]
   *
   * Undocumented, but CPython seems to only allow "*" target once. Any further
   * stars as prefix are rejected. This makes target a strict subset of
   * starred_expression.
   */
  std::optional<IntrVarPtr<Syntax::Target>>
  parseTarget(const Token& assignmentLikeToken);

  std::optional<IntrVarPtr<Syntax::Target>>
  parseTargetList(const Token& assignmentLikeToken, bool delStmt = false);

  /**
   * assignment_stmt ::=  target_list "=" { target_list "=" }
   * (starred_expression | yield_expression)
   */
  std::optional<Syntax::AssignmentStmt>
  parseAssignmentStmt(IntrVarPtr<Syntax::Target>&& firstItem = nullptr);

  /**
   * assert_stmt ::=  "assert" expression ["," expression]
   */
  std::optional<Syntax::AssertStmt> parseAssertStmt();

  /**
   * import_stmt     ::= "import" module ["as" identifier]
   *                     { "," module ["as" identifier] }
   *                   | "from" relative_module "import" identifier
   *                     ["as" identifier] { "," identifier ["as" identifier] }
   *                   | "from" relative_module "import"
   *                     "(" identifier ["as" identifier] { "," identifier
   *                         ["as" identifier] } [","] ")"
   *                   | "from" relative_module "import" "*"
   *
   * module ::= { identifier "." } identifier
   * relative_module ::=  { "." } module |  "." { "." }
   */
  std::optional<Syntax::ImportStmt> parseImportStmt();

  /**
   * simple_stmt               ::=  expression_stmt
   *                             |  assert_stmt
   *                             |  assignment_stmt
   *                             |  augmented_assignment_stmt
   *                             |  annotated_assignment_stmt
   *                             |  pass_stmt
   *                             |  del_stmt
   *                             |  return_stmt
   *                             |  yield_stmt
   *                             |  raise_stmt
   *                             |  break_stmt
   *                             |  continue_stmt
   *                             |  import_stmt
   *                             |  future_stmt
   *                             |  global_stmt
   *                             |  nonlocal_stmt
   *
   * expression_stmt           ::=  starred_expression
   *
   * nonlocal_stmt             ::=  "nonlocal" identifier { "," identifier }
   *
   * global_stmt               ::=  "global" identifier { "," identifier }
   *
   * future_stmt               ::=  "from" "__future__" "import" feature ["as"
   * identifier] { "," feature ["as" identifier] } |  "from" "__future__"
   * "import" "(" feature ["as" identifier] { "," feature ["as" identifier] }
   * [","] ")" feature                   ::=  identifier
   *
   * continue_stmt             ::=  "continue"
   *
   * break_stmt                ::=  "break"
   *
   * raise_stmt                ::=  "raise" [expression ["from" expression]]
   *
   * yield_stmt                ::=  yield_expression
   *
   * return_stmt               ::=  "return" [expression_list]
   *
   * del_stmt                  ::=  "del" target_list
   *
   * pass_stmt                 ::=  "pass"
   *
   * annotated_assignment_stmt ::=  augtarget ":" expression ["="
   * (starred_expression | yield_expression)]
   *
   * augmented_assignment_stmt ::=  augtarget augop (expression_list |
   * yield_expression) augop                     ::=  "+=" | "-=" | "*=" | "@="
   * | "/=" | "//=" | "%=" | "**=" |  ">>=" | "<<=" | "&=" | "^=" | "|="
   * augtarget                 ::=  identifier | attributeref | subscription |
   * slicing
   */
  std::optional<IntrVarPtr<Syntax::SimpleStmt>> parseSimpleStmt();

  /**
   * compound_stmt ::=  if_stmt
   *                 |  while_stmt
   *                 |  for_stmt
   *                 |  try_stmt
   *                 |  with_stmt
   *                 |  funcdef
   *                 |  classdef
   *                 |  async_with_stmt
   *                 |  async_for_stmt
   *                 |  async_funcdef
   *
   * async_with_stmt ::=  "async" with_stmt
   *
   * async_for_stmt ::=  "async" for_stmt
   */
  std::optional<IntrVarPtr<Syntax::CompoundStmt>> parseCompoundStmt();

  /**
   * stmt_list ::=  simple_stmt { ";" simple_stmt } [";"]
   *
   *                Note: we parse this as [simple_stmt] { ";" [simple_stmt] }
   * [";"]. That way the expression statement inside of simple_stmt can never be
   * empty.
   */
  std::optional<std::vector<IntrVarPtr<Syntax::SimpleStmt>>> parseStmtList();

  /**
   * statement ::=  stmt_list NEWLINE | compound_stmt
   */
  std::optional<decltype(Syntax::Suite::statements)> parseStatement();

  /**
   * suite ::=  stmt_list NEWLINE | NEWLINE INDENT statement { statement }
   * DEDENT
   */
  std::optional<Syntax::Suite> parseSuite();

  /**
   * file_input ::=  { NEWLINE | statement }
   */
  std::optional<Syntax::FileInput> parseFileInput();

  std::optional<Syntax::IfStmt::Else> parseElse();

  /**
   * if_stmt ::=  "if" assignment_expression ":" suite ("elif"
   * assignment_expression ":" suite)* ["else" ":" suite]
   */
  std::optional<Syntax::IfStmt> parseIfStmt();

  /**
   * while_stmt ::=  "while" assignment_expression ":" suite ["else" ":" suite]
   */
  std::optional<Syntax::WhileStmt> parseWhileStmt();

  /**
   * for_stmt ::=  "for" target_list "in" expression_list ":" suite ["else" ":"
   * suite]
   */
  std::optional<Syntax::ForStmt> parseForStmt();

  /**
   * try_stmt  ::=  try1_stmt | try2_stmt
   * try1_stmt ::=  "try" ":" suite
   *                { "except" expression ["as" identifier] ":" suite }
   *                ["except" ":" suite]
   *                ["else" ":" suite]
   *                ["finally" ":" suite]
   * try2_stmt ::=  "try" ":" suite "finally" ":" suite
   */
  std::optional<Syntax::TryStmt> parseTryStmt();

  /**
   * with_stmt ::=  "with" with_item { "," with_item } ":" suite
   * with_item ::=  expression ["as" target]
   */
  std::optional<Syntax::WithStmt> parseWithStmt();

  /**
   * parameter_list            ::=  defparameter { "," defparameter } "," "/"
   * ["," [parameter_list_no_posonly]] |  parameter_list_no_posonly
   * parameter_list_no_posonly ::=  defparameter { "," defparameter } [","
   * [parameter_list_starargs]] |  parameter_list_starargs
   * parameter_list_starargs   ::=  "*" [parameter] {"," defparameter } [","
   * ["**" parameter [","]]] |  "**" parameter [","] parameter ::=  identifier
   * [":" expression] defparameter              ::=  parameter ["=" expression]
   */
  std::optional<std::vector<Syntax::Parameter>> parseParameterList();

  /**
   * decorator     ::=  "@" assignment_expression NEWLINE
   *
   * funcdef       ::=  [decorators] "def" funcname "(" [parameter_list] ")"
   * ["->" expression] ":" suite funcname      ::=  identifier
   *
   * async_funcdef ::=  [decorators] "async" "def" funcname "(" [parameter_list]
   * ")" ["->" expression] ":" suite
   */
  std::optional<Syntax::FuncDef>
  parseFuncDef(std::vector<Syntax::Decorator>&& decorators,
               std::optional<BaseToken>&& asyncKeyword);

  /**
   * classdef    ::=  [decorators] "class" classname [inheritance] ":" suite
   * inheritance ::=  "(" [argument_list] ")"
   * classname   ::=  identifier
   */
  std::optional<Syntax::ClassDef>
  parseClassDef(std::vector<Syntax::Decorator>&& decorators);
};
} // namespace pylir
