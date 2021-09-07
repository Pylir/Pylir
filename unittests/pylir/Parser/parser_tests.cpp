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
