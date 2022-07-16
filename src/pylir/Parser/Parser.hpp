// Copyright 2022 Markus Böck
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <pylir/Diagnostics/DiagnosticsBuilder.hpp>
#include <pylir/Diagnostics/Document.hpp>
#include <pylir/Lexer/Lexer.hpp>

#include <unordered_map>
#include <unordered_set>

#include <tcb/span.hpp>
#include <tl/expected.hpp>

#include "Syntax.hpp"

namespace pylir
{
class Parser
{
    Lexer m_lexer;
    Lexer::iterator m_current;

#define HANDLE_FEATURE(x)
#define HANDLE_REQUIRED_FEATURE(x) bool m_##x : 1;
#include "Features.def"
    const Diag::Document* m_document;

    struct Scope
    {
        enum class Kind
        {
            Local,
            NonLocal,
            Global,
            Unknown
        };

        bool classScope{};
        IdentifierMap<Kind> identifiers;
    };
    std::vector<Scope> m_namespace;
    IdentifierSet m_globals;
    bool m_inLoop = false;
    bool m_inFunc = false;

    tl::expected<Token, std::string> expect(TokenType tokenType);

    void addToNamespace(const Token& token);

    void addToNamespace(const IdentifierToken& token);

    void addToNamespace(const Syntax::TargetList& targetList);

    tl::expected<IdentifierSet, std::string> finishNamespace(pylir::Syntax::Suite& suite,
                                                             const IdentifierSet& nonLocals,
                                                             std::vector<const pylir::IdentifierSet*> scopes = {});

    bool lookaheadEquals(tcb::span<const TokenType> tokens);

    template <class ParseFunc, class CheckFunc>
    auto parseCommaList(
        ParseFunc parseFunc, CheckFunc checkFunc,
        std::optional<typename std::invoke_result_t<ParseFunc>::value_type>&& optionalFirst = std::nullopt,
        TokenType tokenType = TokenType::Comma)
        -> tl::expected<Syntax::CommaList<typename std::invoke_result_t<ParseFunc>::value_type>, std::string>
    {
        using T = typename std::invoke_result_t<ParseFunc>::value_type;
        if (!optionalFirst)
        {
            auto first = parseFunc();
            if (!first)
            {
                return tl::unexpected{std::move(first).error()};
            }
            optionalFirst = std::move(*first);
        }
        std::vector<std::pair<BaseToken, std::unique_ptr<T>>> rest;
        std::optional<BaseToken> last;
        while (m_current != m_lexer.end() && m_current->getTokenType() == tokenType)
        {
            auto comma = m_current++;
            if (!checkFunc(m_current->getTokenType()))
            {
                last = *comma;
                break;
            }
            auto other = parseFunc();
            if (!other)
            {
                return tl::unexpected{std::move(other).error()};
            }
            rest.emplace_back(*comma, std::make_unique<T>(*std::move(other)));
        }
        return Syntax::CommaList<T>{std::make_unique<T>(std::move(*optionalFirst)), std::move(rest), last};
    }

    template <class T, auto parseLesser, TokenType... allowed>
    tl::expected<T, std::string> parseGenericBinOp()
    {
        auto first = (this->*parseLesser)();
        if (!first)
        {
            return tl::unexpected{std::move(first).error()};
        }
        T current{std::move(*first)};
        while (m_current != m_lexer.end() && ((m_current->getTokenType() == allowed) || ...))
        {
            auto op = *m_current++;
            auto rhs = (this->*parseLesser)();
            if (!rhs)
            {
                return tl::unexpected{std::move(rhs).error()};
            }
            current.variant = std::make_unique<typename T::BinOp>(
                typename T::BinOp{std::make_unique<T>(std::move(current)), std::move(op), std::move(*rhs)});
        }
        return {std::move(current)};
    }

    tl::expected<Syntax::AugTarget, std::string> convertToAug(Syntax::StarredExpression&& starredExpression,
                                                              const BaseToken& assignOp);

    tl::expected<Syntax::TargetList, std::string> convertToTargetList(Syntax::StarredExpression&& starredExpression,
                                                                      const BaseToken& assignOp);

    static bool firstInAssignmentExpression(TokenType tokenType)
    {
        return firstInExpression(tokenType);
    }

    static bool firstInExpression(TokenType tokenType);

    static bool firstInStarredItem(TokenType tokenType)
    {
        return tokenType == TokenType::Star || firstInAssignmentExpression(tokenType);
    }

    static bool firstInComprehension(TokenType tokenType)
    {
        return firstInAssignmentExpression(tokenType);
    }

    static bool firstInCompFor(TokenType tokenType)
    {
        switch (tokenType)
        {
            case TokenType::ForKeyword:
            case TokenType::AsyncKeyword: return true;
            default: return false;
        }
    }

    static bool firstInTarget(TokenType tokenType);

    static bool firstInSimpleStmt(TokenType tokenType);

    static bool firstInCompoundStmt(TokenType tokenType);

public:
    explicit Parser(
        const Diag::Document& document, int fileId = 0,
        std::function<void(Diag::DiagnosticsBuilder&& diagnosticsBuilder)> callBack = [](auto&&) {})
        : m_lexer(document, fileId, std::move(callBack)),
          m_current(m_lexer.begin()),
#define HANDLE_FEATURE(x)
#define HANDLE_REQUIRED_FEATURE(x) m_##x{true},
#include "Features.def"
          m_document(&document)
    {
    }

    template <class T, class S, class... Args>
    [[nodiscard]] Diag::DiagnosticsBuilder createDiagnosticsBuilder(const T& location, const S& message,
                                                                    Args&&... args) const
    {
        return Diag::DiagnosticsBuilder(*m_document, location, message, std::forward<Args>(args)...);
    }

    /**
     * yield_expression ::=  "yield" [expression_list | "from" expression]
     */
    tl::expected<Syntax::YieldExpression, std::string> parseYieldExpression();

    /**
     * atom ::=  identifier | literal | enclosure
     *
     * literal ::=  stringliteral | bytesliteral | integer | floatnumber | imagnumber | None | True | False
     */
    tl::expected<Syntax::Atom, std::string> parseAtom();

    /**
     * attributeref ::=  primary "." identifier
     */
    tl::expected<Syntax::AttributeRef, std::string> parseAttributeRef(std::unique_ptr<Syntax::Primary>&& primary);

    /**
     * slicing      ::=  primary "[" slice_list "]"
     * slice_list   ::=  slice_item { "," slice_item } [","]
     * slice_item   ::=  expression | proper_slice
     * proper_slice ::=  [lower_bound] ":" [upper_bound] [ ":" [stride] ]
     * lower_bound  ::=  expression
     * upper_bound  ::=  expression
     * stride       ::=  expression
     */
    tl::expected<Syntax::Slicing, std::string> parseSlicing(std::unique_ptr<Syntax::Primary>&& primary);

    /**
     * argument_list        ::=  positional_arguments ["," starred_and_keywords] ["," keywords_arguments]
     *                        |  starred_and_keywords ["," keywords_arguments]
     *                        |  keywords_arguments
     * positional_arguments ::=  positional_item { "," positional_item }
     * positional_item      ::=  assignment_expression | "*" expression
     * starred_and_keywords ::=  ("*" expression | keyword_item) { "," "*" expression | "," keyword_item }
     * keywords_arguments   ::=  (keyword_item | "**" expression) { "," keyword_item | "," "**" expression }
     * keyword_item         ::=  identifier "=" expression
     */
    tl::expected<Syntax::ArgumentList, std::string>
        parseArgumentList(std::optional<Syntax::AssignmentExpression>&& firstAssignment = std::nullopt);

    /**
     * call ::=  primary "(" [argument_list [","] | comprehension] ")"
     */
    tl::expected<Syntax::Call, std::string> parseCall(std::unique_ptr<Syntax::Primary>&& primary);

    /**
     * primary ::=  atom | attributeref | subscription | slicing | call
     */
    tl::expected<Syntax::Primary, std::string> parsePrimary();

    /**
     * subscription ::=  primary "[" expression_list "]"
     */
    tl::expected<Syntax::Primary, std::string> parseSlicingOrSubscription(std::unique_ptr<Syntax::Primary>&& primary);

    /**
     * await_expr ::=  "await" primary
     */
    tl::expected<Syntax::AwaitExpr, std::string> parseAwaitExpr();

    /**
     * power ::= (await_expr | primary) ["**" u_expr]
     */
    tl::expected<Syntax::Power, std::string> parsePower();

    /**
     * u_expr ::=  power
     *          |  "-" u_expr
     *          |  "+" u_expr
     *          |  "~" u_expr
     */
    tl::expected<Syntax::UExpr, std::string> parseUExpr();

    /**
     * m_expr ::=  u_expr
     *          |  m_expr "*" u_expr
     *          |  m_expr "@" m_expr
     *          |  m_expr "//" u_expr
     *          |  m_expr "/" u_expr
     *          |  m_expr "%" u_expr
     */
    tl::expected<Syntax::MExpr, std::string> parseMExpr();

    /**
     * a_expr ::=  m_expr
     *          |  a_expr "+" m_expr
     *          |  a_expr "-" m_expr
     */
    tl::expected<Syntax::AExpr, std::string> parseAExpr();

    /**
     * shift_expr ::=  a_expr
     *              |  shift_expr ("<<" | ">>") a_expr
     */
    tl::expected<Syntax::ShiftExpr, std::string> parseShiftExpr();

    /**
     * and_expr ::=  shift_expr
     *            |  and_expr "&" shift_expr
     */
    tl::expected<Syntax::AndExpr, std::string> parseAndExpr();

    /**
     * xor_expr ::=  and_expr
     *            |  xor_expr "^" and_expr
     */
    tl::expected<Syntax::XorExpr, std::string> parseXorExpr();

    /**
     * or_expr  ::=  xor_expr
     *            | or_expr "|" xor_expr
     */
    tl::expected<Syntax::OrExpr, std::string> parseOrExpr();

    /**
     * comparison    ::=  or_expr { comp_operator or_expr }
     * comp_operator ::=  "<" | ">" | "==" | ">=" | "<=" | "!=" | "is" ["not"] | ["not"] "in"
     */
    tl::expected<Syntax::Comparison, std::string> parseComparison();

    /**
     * not_test ::=  comparison
     *            |  "not" not_test
     */
    tl::expected<Syntax::NotTest, std::string> parseNotTest();

    /**
     * and_test ::=  not_test
     *            |  and_test "and" not_test
     */
    tl::expected<Syntax::AndTest, std::string> parseAndTest();

    /**
     * or_test  ::=  and_test
     *            |  or_test "or" and_test
     */
    tl::expected<Syntax::OrTest, std::string> parseOrTest();

    /**
     * assignment_expression ::=  [identifier ":="] expression
     */
    tl::expected<Syntax::AssignmentExpression, std::string> parseAssignmentExpression();

    /**
     * conditional_expression ::=  or_test ["if" or_test "else" expression]
     */
    tl::expected<Syntax::ConditionalExpression, std::string> parseConditionalExpression();

    /**
     * expression ::=  conditional_expression
     *              |  lambda_expr
     */
    tl::expected<Syntax::Expression, std::string> parseExpression();

    tl::expected<Syntax::CommaList<Syntax::Expression>, std::string> parseExpressionList();

    /**
     * lambda_expr ::=  "lambda" [parameter_list] ":" expression
     */
    tl::expected<Syntax::LambdaExpression, std::string> parseLambdaExpression();

    /**
     * starred_expression ::=  expression
     *                      |  { starred_item "," } [starred_item]
     */
    tl::expected<Syntax::StarredExpression, std::string>
        parseStarredExpression(std::optional<Syntax::AssignmentExpression>&& firstItem = std::nullopt);

    /**
     * starred_item ::=  assignment_expression
     *                |  "*" or_expr
     */
    tl::expected<Syntax::StarredItem, std::string> parseStarredItem();

    tl::expected<Syntax::StarredList, std::string>
        parseStarredList(std::optional<Syntax::StarredItem>&& firstItem = std::nullopt);

    /**
     * comp_for ::=  ["async"] "for" target_list "in" or_test [comp_iter]
     */
    tl::expected<Syntax::CompFor, std::string> parseCompFor();

    /**
     * comp_if ::=  "if" or_test [comp_iter]
     */
    tl::expected<Syntax::CompIf, std::string> parseCompIf();

    /**
     * comprehension ::=  assignment_expression comp_for
     */
    tl::expected<Syntax::Comprehension, std::string>
        parseComprehension(Syntax::AssignmentExpression&& assignmentExpression);

    /**
     * enclosure            ::=  parenth_form | list_display | dict_display | set_display
     *                        |  generator_expression | yield_atom
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
    tl::expected<Syntax::Enclosure, std::string> parseEnclosure();

    /**
     * target          ::=  identifier
     *                   | "(" [target_list] ")"
     *                   | "[" [target_list] "]"
     *                   | attributeref
     *                   | subscription
     *                   | slicing
     *                   | "*" target
     *
     * Undocumented, but CPython seems to only allows "*" target once. Any further stars as prefix are rejected.
     * This makes target a strict subset of starred_expression.
     */
    tl::expected<Syntax::Target, std::string> parseTarget();

    tl::expected<Syntax::TargetList, std::string>
        parseTargetList(std::optional<Syntax::Target>&& firstItem = std::nullopt);

    /**
     * assignment_stmt ::=  target_list "=" { target_list "=" } (starred_expression | yield_expression)
     */
    tl::expected<Syntax::AssignmentStmt, std::string>
        parseAssignmentStmt(std::optional<Syntax::TargetList>&& firstItem = std::nullopt);

    /**
     * augtarget ::=  identifier | attributeref | subscription | slicing
     */
    tl::expected<Syntax::AugTarget, std::string> parseAugTarget();

    /**
     * assert_stmt ::=  "assert" expression ["," expression]
     */
    tl::expected<Syntax::AssertStmt, std::string> parseAssertStmt();

    /**
     * import_stmt     ::=  "import" module ["as" identifier] { "," module ["as" identifier] }
     *                   |  "from" relative_module "import" identifier ["as" identifier]
     *                        { "," identifier ["as" identifier] }
     *                   |  "from" relative_module "import" "(" identifier ["as" identifier]
     *                        { "," identifier ["as" identifier] } [","] ")"
     *                   |  "from" relative_module "import" "*"
     * module          ::=  { identifier "." } identifier
     * relative_module ::=  { "." } module
     *                   |  "." { "." }
     */
    tl::expected<Syntax::ImportStmt, std::string> parseImportStmt();

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
     * nonlocal_stmt             ::=  "nonlocal" identifier { "," identifier }
     *
     * global_stmt               ::=  "global" identifier { "," identifier }
     *
     * future_stmt               ::=  "from" "__future__" "import" feature ["as" identifier]
     *                                  { "," feature ["as" identifier] }
     *                             |  "from" "__future__" "import" "(" feature ["as" identifier]
     *                                  { "," feature ["as" identifier] } [","] ")"
     * feature                   ::=  identifier
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
     * annotated_assignment_stmt ::=  augtarget ":" expression ["=" (starred_expression | yield_expression)]
     *
     * augmented_assignment_stmt ::=  augtarget augop (expression_list | yield_expression)
     * augop                     ::=  "+=" | "-=" | "*=" | "@=" | "/=" | "//=" | "%=" | "**="
     *                             |  ">>=" | "<<=" | "&=" | "^=" | "|="
     */
    tl::expected<Syntax::SimpleStmt, std::string> parseSimpleStmt();

    /**
     * file_input ::=  { NEWLINE | statement }
     */
    tl::expected<Syntax::FileInput, std::string> parseFileInput();

    /**
     * statement ::=  stmt_list NEWLINE | compound_stmt
     */
    tl::expected<Syntax::Statement, std::string> parseStatement();

    /**
     * stmt_list ::=  simple_stmt (";" simple_stmt)* [";"]
     */
    tl::expected<Syntax::StmtList, std::string> parseStmtList();

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
    tl::expected<Syntax::CompoundStmt, std::string> parseCompoundStmt();

    /**
     * suite ::=  stmt_list NEWLINE | NEWLINE INDENT statement { statement } DEDENT
     */
    tl::expected<Syntax::Suite, std::string> parseSuite();

    tl::expected<Syntax::IfStmt::Else, std::string> parseElse();

    /**
     * if_stmt ::=  "if" assignment_expression ":" suite ("elif" assignment_expression ":" suite)* ["else" ":" suite]
     */
    tl::expected<Syntax::IfStmt, std::string> parseIfStmt();

    /**
     * while_stmt ::=  "while" assignment_expression ":" suite ["else" ":" suite]
     */
    tl::expected<Syntax::WhileStmt, std::string> parseWhileStmt();

    /**
     * for_stmt ::=  "for" target_list "in" expression_list ":" suite ["else" ":" suite]
     */
    tl::expected<Syntax::ForStmt, std::string> parseForStmt();

    /**
     * try_stmt  ::=  try1_stmt | try2_stmt
     * try1_stmt ::=  "try" ":" suite
     *                { "except" expression ["as" identifier] ":" suite }
     *                ["except" ":" suite]
     *                ["else" ":" suite]
     *                ["finally" ":" suite]
     * try2_stmt ::=  "try" ":" suite "finally" ":" suite
     */
    tl::expected<Syntax::TryStmt, std::string> parseTryStmt();

    /**
     * with_stmt ::=  "with" with_item { "," with_item } ":" suite
     * with_item ::=  expression ["as" target]
     */
    tl::expected<Syntax::WithStmt, std::string> parseWithStmt();

    /**
     * parameter_list            ::=  defparameter ("," defparameter)* "," "/" ["," [parameter_list_no_posonly]]
     *                             |  parameter_list_no_posonly
     * parameter_list_no_posonly ::=  defparameter { "," defparameter } ["," [parameter_list_starargs]]
     *                             |  parameter_list_starargs
     * parameter_list_starargs   ::=  "*" [parameter] ("," defparameter)* ["," ["**" parameter [","]]]
     *                             |  "**" parameter [","]
     * parameter                 ::=  identifier [":" expression]
     * defparameter              ::=  parameter ["=" expression]
     */
    tl::expected<Syntax::ParameterList, std::string> parseParameterList();

    /**
     * decorator     ::=  "@" assignment_expression NEWLINE
     *
     * funcdef       ::=  [decorators] "def" funcname "(" [parameter_list] ")" ["->" expression] ":" suite
     * funcname      ::=  identifier
     *
     * async_funcdef ::=  [decorators] "async" "def" funcname "(" [parameter_list] ")" ["->" expression] ":" suite
     */
    tl::expected<Syntax::FuncDef, std::string> parseFuncDef(std::vector<Syntax::Decorator>&& decorators,
                                                            std::optional<BaseToken>&& asyncKeyword);

    /**
     * classdef    ::=  [decorators] "class" classname [inheritance] ":" suite
     * inheritance ::=  "(" [argument_list] ")"
     * classname   ::=  identifier
     */
    tl::expected<Syntax::ClassDef, std::string> parseClassDef(std::vector<Syntax::Decorator>&& decorators);
};
} // namespace pylir
