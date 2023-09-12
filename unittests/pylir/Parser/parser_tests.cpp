//  Licensed under the Apache License v2.0 with LLVM Exceptions.
//  See https://llvm.org/LICENSE.txt for license information.
//  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <pylir/Diagnostics/DiagnosticMessages.hpp>
#include <pylir/Parser/Dumper.hpp>
#include <pylir/Parser/Parser.hpp>

#define PARSER_EMITS(source, ...)                                             \
  [](std::string_view str) {                                                  \
    std::string error;                                                        \
    pylir::Diag::DiagnosticsManager manager(                                  \
        [&error](pylir::Diag::Diagnostic&& base) {                            \
          llvm::errs() << base;                                               \
          llvm::raw_string_ostream(error) << base;                            \
        });                                                                   \
    pylir::Diag::Document document(str);                                      \
    auto docManager = manager.createSubDiagnosticManager(document);           \
    pylir::Parser(docManager).parseFileInput();                               \
    CHECK_THAT(error,                                                         \
               Catch::Matchers::ContainsSubstring(fmt::format(__VA_ARGS__))); \
  }(source)

using namespace Catch::Matchers;

TEST_CASE("Parse assignment statement", "[Parser]") {
  PARSER_EMITS("= 3", pylir::Diag::EXPECTED_N_INSTEAD_OF_N, "newline", "'='");
}

TEST_CASE("Parse augmented assignment statement", "[Parser]") {
  PARSER_EMITS("+= 3", pylir::Diag::EXPECTED_N_INSTEAD_OF_N,
               pylir::TokenType::Newline, pylir::TokenType::PlusAssignment);
  PARSER_EMITS("*b += 3", pylir::Diag::EXPECTED_N_INSTEAD_OF_N,
               pylir::TokenType::Comma, pylir::TokenType::PlusAssignment);
}

TEST_CASE("Parse namespaces", "[Parser]") {
  PARSER_EMITS(
      "def foo():\n"
      "    a = 3\n"
      "    nonlocal a\n",
      pylir::Diag::DECLARATION_OF_NONLOCAL_N_CONFLICTS_WITH_LOCAL_VARIABLE,
      "a");
  PARSER_EMITS(
      "def foo():\n"
      "    a = 3\n\n"
      "\n    def inner():\n        global a\n        nonlocal a\n"
      "",
      pylir::Diag::DECLARATION_OF_NONLOCAL_N_CONFLICTS_WITH_GLOBAL_VARIABLE,
      "a");
  PARSER_EMITS("def foo():\n"
               "    nonlocal a\n",
               pylir::Diag::COULD_NOT_FIND_VARIABLE_N_IN_OUTER_SCOPES, "a");
  PARSER_EMITS("a = 0\n"
               "def foo():\n"
               "    nonlocal a\n",
               pylir::Diag::COULD_NOT_FIND_VARIABLE_N_IN_OUTER_SCOPES, "a");
  PARSER_EMITS(
      "def foo():\n"
      "    a = 3\n\n"
      "\n    def inner():\n        nonlocal a\n        global a\n"
      "",
      pylir::Diag::DECLARATION_OF_GLOBAL_N_CONFLICTS_WITH_NONLOCAL_VARIABLE,
      "a");
  PARSER_EMITS("def foo():\n"
               "    a\n"
               "    global a\n",
               pylir::Diag::GLOBAL_N_USED_PRIOR_TO_DECLARATION, "a");
  PARSER_EMITS("def outer():\n    a = 3\n\n    def foo():\n"
               "        a\n"
               "        nonlocal a\n",
               pylir::Diag::NONLOCAL_N_USED_PRIOR_TO_DECLARATION, "a");

  // All these things should work just as well if inside a class
  PARSER_EMITS(
      "class Foo:\n"
      "  def foo():\n"
      "    a = 3\n"
      "    nonlocal a\n",
      pylir::Diag::DECLARATION_OF_NONLOCAL_N_CONFLICTS_WITH_LOCAL_VARIABLE,
      "a");
  PARSER_EMITS(
      "class Foo:\n"
      "  def foo():\n"
      "    a = 3\n\n"
      "\n    def inner():\n        global a\n        nonlocal a\n"
      "",
      pylir::Diag::DECLARATION_OF_NONLOCAL_N_CONFLICTS_WITH_GLOBAL_VARIABLE,
      "a");
  PARSER_EMITS("class Foo:\n"
               "  def foo():\n"
               "    nonlocal a\n",
               pylir::Diag::COULD_NOT_FIND_VARIABLE_N_IN_OUTER_SCOPES, "a");
  PARSER_EMITS("a = 0\n"
               "class Foo:\n"
               "  def foo():\n"
               "    nonlocal a\n",
               pylir::Diag::COULD_NOT_FIND_VARIABLE_N_IN_OUTER_SCOPES, "a");
  PARSER_EMITS(
      "class Foo:\n"
      "  def foo():\n"
      "    a = 3\n\n"
      "\n    def inner():\n        nonlocal a\n        global a\n"
      "",
      pylir::Diag::DECLARATION_OF_GLOBAL_N_CONFLICTS_WITH_NONLOCAL_VARIABLE,
      "a");
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

TEST_CASE("Parse function definition", "[Parser]") {
  PARSER_EMITS(
      "def foo(a = 3,c):\n"
      "  pass",
      pylir::Diag::
          NO_DEFAULT_ARGUMENT_FOR_PARAMETER_N_FOLLOWING_PARAMETERS_WITH_DEFAULT_ARGUMENTS,
      "c");
  PARSER_EMITS(
      "def foo(a = 3, /, c):\n"
      "  pass",
      pylir::Diag::
          NO_DEFAULT_ARGUMENT_FOR_PARAMETER_N_FOLLOWING_PARAMETERS_WITH_DEFAULT_ARGUMENTS,
      "c");
}

TEST_CASE("Parser argument list", "[Parser]") {
  PARSER_EMITS(
      "foo(a = 3, c)",
      pylir::Diag::POSITIONAL_ARGUMENT_NOT_ALLOWED_FOLLOWING_KEYWORD_ARGUMENTS);
  PARSER_EMITS(
      "foo(**a, c)",
      pylir::Diag::
          POSITIONAL_ARGUMENT_NOT_ALLOWED_FOLLOWING_DICTIONARY_UNPACKING);
  PARSER_EMITS(
      "foo(a = 3, **b, c)",
      pylir::Diag::POSITIONAL_ARGUMENT_NOT_ALLOWED_FOLLOWING_KEYWORD_ARGUMENTS);
  PARSER_EMITS(
      "foo(**b, *c)",
      pylir::Diag::
          ITERABLE_UNPACKING_NOT_ALLOWED_FOLLOWING_DICTIONARY_UNPACKING);
}

namespace {
void parse(std::string_view source) {
  pylir::Diag::DiagnosticsManager manager;
  pylir::Diag::Document document(source);
  auto docManager = manager.createSubDiagnosticManager(document);
  pylir::Parser parser(docManager);

  parser.parseFileInput();
}
} // namespace

TEST_CASE("Parser fuzzer discoveries", "[Parser]") {
  parse("5(**~5(**10,j)p");
}
