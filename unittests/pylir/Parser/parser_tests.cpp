#include <catch2/catch.hpp>

#include <pylir/Diagnostics/DiagnosticMessages.hpp>
#include <pylir/Parser/Dumper.hpp>
#include <pylir/Parser/Parser.hpp>

#include <iostream>

#define PARSER_EMITS(source, ...)                                         \
    [](std::string str)                                                   \
    {                                                                     \
        pylir::Diag::Document document(str);                              \
        pylir::Parser parser(document);                                   \
        auto fileInput = parser.parseFileInput();                         \
        if (!fileInput)                                                   \
        {                                                                 \
            auto& error = fileInput.error();                              \
            std::cerr << error;                                           \
            CHECK_THAT(error, Catch::Contains(fmt::format(__VA_ARGS__))); \
            return;                                                       \
        }                                                                 \
        FAIL("No error emitted");                                         \
    }(source)

using namespace Catch::Matchers;

TEST_CASE("Parse assignment statement", "[Parser]")
{
    PARSER_EMITS("= 3", pylir::Diag::EXPECTED_N_BEFORE_N, "identifier", "assignment");
    PARSER_EMITS("(a := 3) = 3", pylir::Diag::CANNOT_ASSIGN_TO_RESULT_OF_OPERATOR_N, pylir::TokenType::Walrus);
    PARSER_EMITS("(lambda: 3) = 3", pylir::Diag::CANNOT_ASSIGN_TO_RESULT_OF_N, "lambda expression");
    PARSER_EMITS("(3 if True else 5) = 3", pylir::Diag::CANNOT_ASSIGN_TO_RESULT_OF_N, "conditional expression");
    PARSER_EMITS("(3 and 5) = 3", pylir::Diag::CANNOT_ASSIGN_TO_RESULT_OF_OPERATOR_N, pylir::TokenType::AndKeyword);
    PARSER_EMITS("(not 3) = 3", pylir::Diag::CANNOT_ASSIGN_TO_RESULT_OF_OPERATOR_N, pylir::TokenType::NotKeyword);
    PARSER_EMITS("(3 != 5) = 3", pylir::Diag::CANNOT_ASSIGN_TO_RESULT_OF_N, "comparison");
    PARSER_EMITS("(-3) = 3", pylir::Diag::CANNOT_ASSIGN_TO_RESULT_OF_N,
                 fmt::format("unary operator {:q}", pylir::TokenType::Minus));
    PARSER_EMITS("(2**8) = 3", pylir::Diag::CANNOT_ASSIGN_TO_RESULT_OF_OPERATOR_N, pylir::TokenType::PowerOf);
    PARSER_EMITS("(await foo()) = 3", pylir::Diag::CANNOT_ASSIGN_TO_RESULT_OF_N, "await expression");
    PARSER_EMITS("(foo()) = 3", pylir::Diag::CANNOT_ASSIGN_TO_RESULT_OF_N, "call");
    PARSER_EMITS("5 = 3", pylir::Diag::CANNOT_ASSIGN_TO_N, "literal");
    PARSER_EMITS("{} = 3", pylir::Diag::CANNOT_ASSIGN_TO_N, "dictionary display");
    PARSER_EMITS("{5} = 3", pylir::Diag::CANNOT_ASSIGN_TO_N, "set display");
    PARSER_EMITS("[5 for c in f] = 3", pylir::Diag::CANNOT_ASSIGN_TO_N, "list display");
    PARSER_EMITS("[5] = 3", pylir::Diag::CANNOT_ASSIGN_TO_N, "literal");
    PARSER_EMITS("(yield 5) = 3", pylir::Diag::CANNOT_ASSIGN_TO_N, "yield expression");
    PARSER_EMITS("(c for c in f) = 3", pylir::Diag::CANNOT_ASSIGN_TO_N, "generator expression");
}

TEST_CASE("Parse augmented assignment statement", "[Parser]")
{
    PARSER_EMITS("+= 3", pylir::Diag::EXPECTED_N_BEFORE_N, "identifier", "assignment");
    PARSER_EMITS("a,b += 3", pylir::Diag::CANNOT_ASSIGN_TO_N, "multiple values");
    PARSER_EMITS("*b += 3", pylir::Diag::CANNOT_ASSIGN_TO_N, "starred item");
    PARSER_EMITS("(b) += 3", pylir::Diag::CANNOT_ASSIGN_TO_N, "enclosure");
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
