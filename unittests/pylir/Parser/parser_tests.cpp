// Copyright 2022 Markus Böck
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <catch2/catch.hpp>

#include <pylir/Diagnostics/DiagnosticMessages.hpp>
#include <pylir/Parser/Dumper.hpp>
#include <pylir/Parser/Parser.hpp>

#include <iostream>

#define PARSER_EMITS(source, ...)                                       \
    [](std::string str)                                                 \
    {                                                                   \
        std::string error;                                              \
        pylir::Diag::DiagnosticsManager manager(                        \
            [&error](pylir::Diag::DiagnosticsBuilderBase&& base)        \
            {                                                           \
                llvm::errs() << base;                                   \
                llvm::raw_string_ostream(error) << base;                \
            });                                                         \
        pylir::Diag::Document document(std::move(str));                 \
        auto docManager = manager.createSubDiagnosticManager(document); \
        pylir::Parser(docManager).parseFileInput();                     \
        CHECK_THAT(error, Catch::Contains(fmt::format(__VA_ARGS__)));   \
    }(source)

using namespace Catch::Matchers;

TEST_CASE("Parse try statement", "[Parser]")
{
    PARSER_EMITS("try:\n"
                 "    pass\n"
                 "except:\n"
                 "    pass\n"
                 "except int:\n"
                 "    pass\n",
                 pylir::Diag::EXCEPT_CLAUSE_WITHOUT_EXPRESSION_MUST_COME_LAST);
}

TEST_CASE("Parse break continue statement", "[Parser")
{
    PARSER_EMITS("break", pylir::Diag::OCCURRENCE_OF_N_OUTSIDE_OF_LOOP, "'break'");
    PARSER_EMITS("continue", pylir::Diag::OCCURRENCE_OF_N_OUTSIDE_OF_LOOP, "'continue'");
    PARSER_EMITS("while True:\n    def foo():\n        break\n"
                 "",
                 pylir::Diag::OCCURRENCE_OF_N_OUTSIDE_OF_LOOP, "'break'");
    PARSER_EMITS("while True:\n    def foo():\n        continue\n"
                 "",
                 pylir::Diag::OCCURRENCE_OF_N_OUTSIDE_OF_LOOP, "'continue'");
    PARSER_EMITS("while True:\n    class Foo:\n        break\n"
                 "",
                 pylir::Diag::OCCURRENCE_OF_N_OUTSIDE_OF_LOOP, "'break'");
    PARSER_EMITS("while True:\n    class Foo:\n        continue\n"
                 "",
                 pylir::Diag::OCCURRENCE_OF_N_OUTSIDE_OF_LOOP, "'continue'");
    PARSER_EMITS("while True:\n"
                 "  pass\n"
                 "else:\n"
                 "  break\n",
                 pylir::Diag::OCCURRENCE_OF_N_OUTSIDE_OF_LOOP, "'break'");
    PARSER_EMITS("while True:\n"
                 "  pass\n"
                 "else:\n"
                 "  continue\n",
                 pylir::Diag::OCCURRENCE_OF_N_OUTSIDE_OF_LOOP, "'continue'");
}

TEST_CASE("Parse return statement", "[Parser]")
{
    PARSER_EMITS("return", pylir::Diag::OCCURRENCE_OF_RETURN_OUTSIDE_OF_FUNCTION);
    PARSER_EMITS("class Foo:\n"
                 "    return",
                 pylir::Diag::OCCURRENCE_OF_RETURN_OUTSIDE_OF_FUNCTION);
    PARSER_EMITS("def foo():\n"
                 "    class Foo:\n"
                 "        return",
                 pylir::Diag::OCCURRENCE_OF_RETURN_OUTSIDE_OF_FUNCTION);
}

TEST_CASE("Parse yield expression", "[Parser]")
{
    PARSER_EMITS("yield 5", pylir::Diag::OCCURRENCE_OF_YIELD_OUTSIDE_OF_FUNCTION);
    PARSER_EMITS("class Foo:\n"
                 "    yield 5",
                 pylir::Diag::OCCURRENCE_OF_YIELD_OUTSIDE_OF_FUNCTION);
    PARSER_EMITS("def foo():\n"
                 "    class Foo:\n"
                 "        yield 5",
                 pylir::Diag::OCCURRENCE_OF_YIELD_OUTSIDE_OF_FUNCTION);
}

TEST_CASE("Parse assignment statement", "[Parser]")
{
    PARSER_EMITS("= 3", pylir::Diag::EXPECTED_N_INSTEAD_OF_N, "newline", "'='");
    PARSER_EMITS("(a := 3) = 3", pylir::Diag::CANNOT_ASSIGN_TO_RESULT_OF_OPERATOR_N, pylir::TokenType::Walrus);
    PARSER_EMITS("(lambda: 3) = 3", pylir::Diag::CANNOT_ASSIGN_TO_RESULT_OF_N, "lambda expression");
    PARSER_EMITS("(3 if True else 5) = 3", pylir::Diag::CANNOT_ASSIGN_TO_RESULT_OF_N, "conditional expression");
    PARSER_EMITS("(3 and 5) = 3", pylir::Diag::CANNOT_ASSIGN_TO_RESULT_OF_OPERATOR_N, pylir::TokenType::AndKeyword);
    PARSER_EMITS("(not 3) = 3", pylir::Diag::CANNOT_ASSIGN_TO_RESULT_OF_UNARY_OPERATOR_N, pylir::TokenType::NotKeyword);
    PARSER_EMITS("(3 != 5) = 3", pylir::Diag::CANNOT_ASSIGN_TO_RESULT_OF_OPERATOR_N, pylir::TokenType::NotEqual);
    PARSER_EMITS("(-3) = 3", pylir::Diag::CANNOT_ASSIGN_TO_RESULT_OF_UNARY_OPERATOR_N, pylir::TokenType::Minus);
    PARSER_EMITS("(2**8) = 3", pylir::Diag::CANNOT_ASSIGN_TO_RESULT_OF_OPERATOR_N, pylir::TokenType::PowerOf);
    PARSER_EMITS("(await foo()) = 3", pylir::Diag::CANNOT_ASSIGN_TO_RESULT_OF_UNARY_OPERATOR_N,
                 pylir::TokenType::AwaitKeyword);
    PARSER_EMITS("(foo()) = 3", pylir::Diag::CANNOT_ASSIGN_TO_RESULT_OF_N, "call");
    PARSER_EMITS("5 = 3", pylir::Diag::CANNOT_ASSIGN_TO_N, "literal");
    PARSER_EMITS("{} = 3", pylir::Diag::CANNOT_ASSIGN_TO_N, "dictionary display");
    PARSER_EMITS("{5} = 3", pylir::Diag::CANNOT_ASSIGN_TO_N, "set display");
    PARSER_EMITS("[5 for c in f] = 3", pylir::Diag::CANNOT_ASSIGN_TO_N, "list display");
    PARSER_EMITS("[5] = 3", pylir::Diag::CANNOT_ASSIGN_TO_N, "literal");
    PARSER_EMITS("def foo():(yield 5) = 3", pylir::Diag::CANNOT_ASSIGN_TO_N, "yield expression");
    PARSER_EMITS("(c for c in f) = 3", pylir::Diag::CANNOT_ASSIGN_TO_N, "generator expression");
}

TEST_CASE("Parse augmented assignment statement", "[Parser]")
{
    PARSER_EMITS("+= 3", pylir::Diag::EXPECTED_N_INSTEAD_OF_N, pylir::TokenType::Newline,
                 pylir::TokenType::PlusAssignment);
    PARSER_EMITS("a,b += 3", pylir::Diag::OPERATOR_N_CANNOT_ASSIGN_TO_MULTIPLE_VARIABLES,
                 pylir::TokenType::PlusAssignment);
    PARSER_EMITS("*b += 3", pylir::Diag::EXPECTED_N_INSTEAD_OF_N, pylir::TokenType::Comma,
                 pylir::TokenType::PlusAssignment);
}

TEST_CASE("Parse namespaces", "[Parser]")
{
    PARSER_EMITS("def foo():\n"
                 "    a = 3\n"
                 "    nonlocal a\n",
                 pylir::Diag::DECLARATION_OF_NONLOCAL_N_CONFLICTS_WITH_LOCAL_VARIABLE, "a");
    PARSER_EMITS("def foo():\n"
                 "    a = 3\n\n"
                 "\n    def inner():\n        global a\n        nonlocal a\n"
                 "",
                 pylir::Diag::DECLARATION_OF_NONLOCAL_N_CONFLICTS_WITH_GLOBAL_VARIABLE, "a");
    PARSER_EMITS("def foo():\n"
                 "    nonlocal a\n",
                 pylir::Diag::COULD_NOT_FIND_VARIABLE_N_IN_OUTER_SCOPES, "a");
    PARSER_EMITS("a = 0\n"
                 "def foo():\n"
                 "    nonlocal a\n",
                 pylir::Diag::COULD_NOT_FIND_VARIABLE_N_IN_OUTER_SCOPES, "a");
    PARSER_EMITS("def foo():\n"
                 "    a = 3\n\n"
                 "\n    def inner():\n        nonlocal a\n        global a\n"
                 "",
                 pylir::Diag::DECLARATION_OF_GLOBAL_N_CONFLICTS_WITH_NONLOCAL_VARIABLE, "a");
    PARSER_EMITS("def foo():\n"
                 "    a\n"
                 "    global a\n",
                 pylir::Diag::GLOBAL_N_USED_PRIOR_TO_DECLARATION, "a");
    PARSER_EMITS("def outer():\n    a = 3\n\n    def foo():\n"
                 "        a\n"
                 "        nonlocal a\n",
                 pylir::Diag::NONLOCAL_N_USED_PRIOR_TO_DECLARATION, "a");

    // All these things should work just as well if inside a class
    PARSER_EMITS("class Foo:\n"
                 "  def foo():\n"
                 "    a = 3\n"
                 "    nonlocal a\n",
                 pylir::Diag::DECLARATION_OF_NONLOCAL_N_CONFLICTS_WITH_LOCAL_VARIABLE, "a");
    PARSER_EMITS("class Foo:\n"
                 "  def foo():\n"
                 "    a = 3\n\n"
                 "\n    def inner():\n        global a\n        nonlocal a\n"
                 "",
                 pylir::Diag::DECLARATION_OF_NONLOCAL_N_CONFLICTS_WITH_GLOBAL_VARIABLE, "a");
    PARSER_EMITS("class Foo:\n"
                 "  def foo():\n"
                 "    nonlocal a\n",
                 pylir::Diag::COULD_NOT_FIND_VARIABLE_N_IN_OUTER_SCOPES, "a");
    PARSER_EMITS("a = 0\n"
                 "class Foo:\n"
                 "  def foo():\n"
                 "    nonlocal a\n",
                 pylir::Diag::COULD_NOT_FIND_VARIABLE_N_IN_OUTER_SCOPES, "a");
    PARSER_EMITS("class Foo:\n"
                 "  def foo():\n"
                 "    a = 3\n\n"
                 "\n    def inner():\n        nonlocal a\n        global a\n"
                 "",
                 pylir::Diag::DECLARATION_OF_GLOBAL_N_CONFLICTS_WITH_NONLOCAL_VARIABLE, "a");
    PARSER_EMITS("class Foo:\n"
                 "  def foo():\n"
                 "    a\n"
                 "    global a\n",
                 pylir::Diag::GLOBAL_N_USED_PRIOR_TO_DECLARATION, "a");
    PARSER_EMITS("class Foo:\n"
                 "  def outer():\n    a = 3\n\n    def foo():\n"
                 "        a\n"
                 "        nonlocal a\n",
                 pylir::Diag::NONLOCAL_N_USED_PRIOR_TO_DECLARATION, "a");
}

TEST_CASE("Parse function definition", "[Parser]")
{
    PARSER_EMITS("def foo(a = 3,c):\n"
                 "  pass",
                 pylir::Diag::NO_DEFAULT_ARGUMENT_FOR_PARAMETER_N_FOLLOWING_PARAMETERS_WITH_DEFAULT_ARGUMENTS, "c");
    PARSER_EMITS("def foo(a = 3, /, c):\n"
                 "  pass",
                 pylir::Diag::NO_DEFAULT_ARGUMENT_FOR_PARAMETER_N_FOLLOWING_PARAMETERS_WITH_DEFAULT_ARGUMENTS, "c");
}

TEST_CASE("Parser argument list", "[Parser]")
{
    PARSER_EMITS("foo(a = 3, c)", pylir::Diag::POSITIONAL_ARGUMENT_NOT_ALLOWED_FOLLOWING_KEYWORD_ARGUMENTS);
    PARSER_EMITS("foo(**a, c)", pylir::Diag::POSITIONAL_ARGUMENT_NOT_ALLOWED_FOLLOWING_DICTIONARY_UNPACKING);
    PARSER_EMITS("foo(a = 3, **b, c)", pylir::Diag::POSITIONAL_ARGUMENT_NOT_ALLOWED_FOLLOWING_KEYWORD_ARGUMENTS);
    PARSER_EMITS("foo(**b, *c)", pylir::Diag::ITERABLE_UNPACKING_NOT_ALLOWED_FOLLOWING_DICTIONARY_UNPACKING);
}

namespace
{
void parse(std::string_view source)
{
    pylir::Diag::DiagnosticsManager manager;
    pylir::Diag::Document document(std::string{source});
    auto docManager = manager.createSubDiagnosticManager(document);
    pylir::Parser parser(docManager);

    parser.parseFileInput();
}
} // namespace

TEST_CASE("Parser fuzzer discoveries", "[Parser]")
{
    parse("5(**~5(**10,j)p");
}
