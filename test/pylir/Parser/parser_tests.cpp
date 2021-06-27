#include <catch2/catch.hpp>

#include <pylir/Diagnostics/DiagnosticMessages.hpp>
#include <pylir/Parser/Dumper.hpp>
#include <pylir/Parser/Parser.hpp>

#include <iostream>

#define dumpExpression(source)                                \
    []() -> std::string                                       \
    {                                                         \
        pylir::Diag::Document document(source);               \
        pylir::Parser parser(document);                       \
        auto assignment = parser.parseAssignmentExpression(); \
        if (!assignment)                                      \
        {                                                     \
            FAIL(assignment.error());                         \
        }                                                     \
        pylir::Dumper dumper;                                 \
        return dumper.dump(*assignment);                      \
    }()

#define dumpStatement(source)                       \
    []() -> std::string                             \
    {                                               \
        pylir::Diag::Document document(source);     \
        pylir::Parser parser(document);             \
        auto simpleStmt = parser.parseSimpleStmt(); \
        if (!simpleStmt)                            \
        {                                           \
            FAIL(simpleStmt.error());               \
        }                                           \
        pylir::Dumper dumper;                       \
        return dumper.dump(*simpleStmt);            \
    }()

#define dumpAll(source)                            \
    []() -> std::string                            \
    {                                              \
        pylir::Diag::Document document(source);    \
        pylir::Parser parser(document);            \
        auto simpleStmt = parser.parseFileInput(); \
        if (!simpleStmt)                           \
        {                                          \
            FAIL(simpleStmt.error());              \
        }                                          \
        pylir::Dumper dumper;                      \
        return dumper.dump(*simpleStmt);           \
    }()

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

TEST_CASE("Parse atom", "[Parser]")
{
    SECTION("Identifier")
    {
        CHECK_THAT(dumpExpression("test"), Contains("atom test"));
    }
    SECTION("String")
    {
        CHECK_THAT(dumpExpression("\"\"\"dwadw\nwdawdw\"\"\""), Contains("atom 'dwadw\\nwdawdw'"));
    }
    SECTION("Byte")
    {
        CHECK_THAT(dumpExpression("b\"dwadw\\nwda\\077dw\""), Contains("atom b'dwadw\\nwda?dw'"));
    }
    SECTION("Integer")
    {
        CHECK_THAT(dumpExpression("5"), Contains("atom 5"));
    }
    SECTION("Float")
    {
        CHECK_THAT(dumpExpression(".5"), Contains("atom 0.5"));
    }
    SECTION("Keywords")
    {
        CHECK_THAT(dumpExpression("None"), Contains("atom None"));
        CHECK_THAT(dumpExpression("True"), Contains("atom True"));
        CHECK_THAT(dumpExpression("False"), Contains("atom False"));
    }
}

TEST_CASE("Parse Enclosure", "[Parser]")
{
    SECTION("Parenth")
    {
        CHECK_THAT(dumpExpression("()"), Contains("parenth empty"));
        CHECK_THAT(dumpExpression("(5)"), Contains("parenth\n"
                                                   "`-atom 5"));
        CHECK_THAT(dumpExpression("(5,)"), Contains("parenth\n"
                                                    "`-starred expression\n"
                                                    "  `-atom 5"));
        CHECK_THAT(dumpExpression("(5,3)"), Contains("parenth\n"
                                                     "`-starred expression\n"
                                                     "  |-atom 5\n"
                                                     "  `-atom 3"));
    }
    SECTION("yield")
    {
        CHECK_THAT(dumpExpression("(yield from 5)"), Contains("yieldatom\n"
                                                              "`-yield from\n"
                                                              "  `-atom 5"));
        CHECK_THAT(dumpExpression("(yield)"), Contains("yieldatom\n"
                                                       "`-yield empty"));
        CHECK_THAT(dumpExpression("(yield 5)"), Contains("yieldatom\n"
                                                         "`-yield list\n"
                                                         "  `-atom 5"));
        CHECK_THAT(dumpExpression("(yield 5,5,)"), Contains("yieldatom\n"
                                                            "`-yield list\n"
                                                            "  |-atom 5\n"
                                                            "  `-atom 5"));
    }
    SECTION("List display")
    {
        CHECK_THAT(dumpExpression("[]"), Contains("list display empty"));
        CHECK_THAT(dumpExpression("[5]"), Contains("list display\n"
                                                   "`-atom 5"));
        CHECK_THAT(dumpExpression("[5,3]"), Contains("list display\n"
                                                     "`-starred list\n"
                                                     "  |-atom 5\n"
                                                     "  `-atom 3"));
        CHECK_THAT(dumpExpression("[*5,3]"), Contains("list display\n"
                                                      "`-starred list\n"
                                                      "  |-starred item\n"
                                                      "  | `-atom 5\n"
                                                      "  `-atom 3"));
    }
    SECTION("set display")
    {
        CHECK_THAT(dumpExpression("{5}"), Contains("set display\n"
                                                   "`-atom 5"));
        CHECK_THAT(dumpExpression("{5,3}"), Contains("set display\n"
                                                     "`-starred list\n"
                                                     "  |-atom 5\n"
                                                     "  `-atom 3"));
        CHECK_THAT(dumpExpression("{*5,3}"), Contains("set display\n"
                                                      "`-starred list\n"
                                                      "  |-starred item\n"
                                                      "  | `-atom 5\n"
                                                      "  `-atom 3"));
    }
    SECTION("dict display")
    {
        CHECK_THAT(dumpExpression("{}"), Contains("dict display empty"));
        CHECK_THAT(dumpExpression("{5:3}"), Contains("dict display\n"
                                                     "`-key\n"
                                                     "  |-atom 5\n"
                                                     "  `-atom 3"));
        CHECK_THAT(dumpExpression("{5:3,3:2}"), Contains("dict display\n"
                                                         "`-key datum list\n"
                                                         "  |-key\n"
                                                         "  | |-atom 5\n"
                                                         "  | `-atom 3\n"
                                                         "  `-key\n"
                                                         "    |-atom 3\n"
                                                         "    `-atom 2"));
        CHECK_THAT(dumpExpression("{**5,3:2}"), Contains("dict display\n"
                                                         "`-key datum list\n"
                                                         "  |-datum\n"
                                                         "  | `-atom 5\n"
                                                         "  `-key\n"
                                                         "    |-atom 3\n"
                                                         "    `-atom 2"));
    }
}

TEST_CASE("Parse attribute ref", "[Parser]")
{
    CHECK_THAT(dumpExpression("a.b"), Contains("attribute b\n"
                                               "`-atom a"));
}

TEST_CASE("Parse subscription", "[Parser]")
{
    CHECK_THAT(dumpExpression("a[b]"), Contains("subscription\n"
                                                "|-primary: atom a\n"
                                                "`-index: atom b"));
    CHECK_THAT(dumpExpression("a[b,c]"), Contains("subscription\n"
                                                  "|-primary: atom a\n"
                                                  "`-index: expression list\n"
                                                  "  |-atom b\n"
                                                  "  `-atom c"));
}

TEST_CASE("Parse slicing", "[Parser]")
{
    CHECK_THAT(dumpExpression("a[b::]"), Contains("slicing\n"
                                                  "|-primary: atom a\n"
                                                  "`-index: proper slice\n"
                                                  "  `-lowerBound: atom b"));
    CHECK_THAT(dumpExpression("a[:c:]"), Contains("slicing\n"
                                                  "|-primary: atom a\n"
                                                  "`-index: proper slice\n"
                                                  "  `-upperBound: atom c"));
    CHECK_THAT(dumpExpression("a[::c]"), Contains("slicing\n"
                                                  "|-primary: atom a\n"
                                                  "`-index: proper slice\n"
                                                  "  `-stride: atom c"));
    CHECK_THAT(dumpExpression("a[b:c:d]"), Contains("slicing\n"
                                                    "|-primary: atom a\n"
                                                    "`-index: proper slice\n"
                                                    "  |-lowerBound: atom b\n"
                                                    "  |-upperBound: atom c\n"
                                                    "  `-stride: atom d"));
    CHECK_THAT(dumpExpression("a[b:c:d,3]"), Contains("slicing\n"
                                                      "|-primary: atom a\n"
                                                      "`-index: proper slice list\n"
                                                      "  |-proper slice\n"
                                                      "  | |-lowerBound: atom b\n"
                                                      "  | |-upperBound: atom c\n"
                                                      "  | `-stride: atom d\n"
                                                      "  `-atom 3"));
}

TEST_CASE("Parse calls", "[Parser]")
{
    CHECK_THAT(dumpExpression("a()"), Contains("call\n"
                                               "`-atom a"));
    CHECK_THAT(dumpExpression("a(b)"), Contains("call\n"
                                                "|-atom a\n"
                                                "`-argument list\n"
                                                "  `-positional arguments\n"
                                                "    `-atom b"));
    CHECK_THAT(dumpExpression("a(b)"), Contains("call\n"
                                                "|-atom a\n"
                                                "`-argument list\n"
                                                "  `-positional arguments\n"
                                                "    `-atom b"));
    CHECK_THAT(dumpExpression("a(b,c)"), Contains("call\n"
                                                  "|-atom a\n"
                                                  "`-argument list\n"
                                                  "  `-positional arguments\n"
                                                  "    |-atom b\n"
                                                  "    `-atom c"));
    CHECK_THAT(dumpExpression("a(b,*c)"), Contains("call\n"
                                                   "|-atom a\n"
                                                   "`-argument list\n"
                                                   "  `-positional arguments\n"
                                                   "    |-atom b\n"
                                                   "    `-starred\n"
                                                   "      `-atom c"));
    CHECK_THAT(dumpExpression("a(b,c = 3)"), Contains("call\n"
                                                      "|-atom a\n"
                                                      "`-argument list\n"
                                                      "  |-positional arguments\n"
                                                      "  | `-atom b\n"
                                                      "  `-starred keywords\n"
                                                      "    `-keyword item c\n"
                                                      "      `-atom 3"));
    CHECK_THAT(dumpExpression("a(b,c = 3,*b)"), Contains("call\n"
                                                         "|-atom a\n"
                                                         "`-argument list\n"
                                                         "  |-positional arguments\n"
                                                         "  | `-atom b\n"
                                                         "  `-starred keywords\n"
                                                         "    |-keyword item c\n"
                                                         "    | `-atom 3\n"
                                                         "    `-starred expression\n"
                                                         "      `-atom b"));
    CHECK_THAT(dumpExpression("a(**b,c = 3)"), Contains("call\n"
                                                        "|-atom a\n"
                                                        "`-argument list\n"
                                                        "  `-keyword arguments\n"
                                                        "    |-mapped expression\n"
                                                        "    | `-atom b\n"
                                                        "    `-keyword item c\n"
                                                        "      `-atom 3"));
}

TEST_CASE("Parse await expression", "[Parser]")
{
    CHECK_THAT(dumpExpression("await 5"), Contains("await expression\n"
                                                   "`-atom 5"));
}

TEST_CASE("Parse unary expression", "[Parser]")
{
    CHECK_THAT(dumpExpression("-5"), Contains("unary '-'\n"
                                              "`-atom 5"));
    CHECK_THAT(dumpExpression("+5"), Contains("unary '+'\n"
                                              "`-atom 5"));
    CHECK_THAT(dumpExpression("~5"), Contains("unary '~'\n"
                                              "`-atom 5"));
    CHECK_THAT(dumpExpression("+-5"), Contains("unary '+'\n"
                                               "`-unary '-'\n"
                                               "  `-atom 5"));
}

TEST_CASE("Parse power expression", "[Parser]")
{
    CHECK_THAT(dumpExpression("2 ** 5"), Contains("power\n"
                                                  "|-base: atom 2\n"
                                                  "`-exponent: atom 5"));
    CHECK_THAT(dumpExpression("await 2 ** 5"), Contains("power\n"
                                                        "|-base: await expression\n"
                                                        "| `-atom 2\n"
                                                        "`-exponent: atom 5"));
}

TEST_CASE("Parse mexpr", "[Parser]")
{
    CHECK_THAT(dumpExpression("2 * 5"), Contains("mexpr '*'\n"
                                                 "|-lhs: atom 2\n"
                                                 "`-rhs: atom 5"));
    CHECK_THAT(dumpExpression("2 @ 5"), Contains("mexpr '@'\n"
                                                 "|-lhs: atom 2\n"
                                                 "`-rhs: atom 5"));
    CHECK_THAT(dumpExpression("2 // 5"), Contains("mexpr '//'\n"
                                                  "|-lhs: atom 2\n"
                                                  "`-rhs: atom 5"));
    CHECK_THAT(dumpExpression("2 / 5"), Contains("mexpr '/'\n"
                                                 "|-lhs: atom 2\n"
                                                 "`-rhs: atom 5"));
    CHECK_THAT(dumpExpression("2 % 5"), Contains("mexpr '%'\n"
                                                 "|-lhs: atom 2\n"
                                                 "`-rhs: atom 5"));
    CHECK_THAT(dumpExpression("2 @ 5 * 3"), Contains("mexpr '@'\n"
                                                     "|-lhs: atom 2\n"
                                                     "`-rhs: mexpr '*'\n"
                                                     "  |-lhs: atom 5\n"
                                                     "  `-rhs: atom 3"));
    CHECK_THAT(dumpExpression("2 / 5 * 3"), Contains("mexpr '*'\n"
                                                     "|-lhs: mexpr '/'\n"
                                                     "| |-lhs: atom 2\n"
                                                     "| `-rhs: atom 5\n"
                                                     "`-rhs: atom 3"));
}

TEST_CASE("Parse aexpr", "[Parser]")
{
    CHECK_THAT(dumpExpression("2 + 5"), Contains("aexpr '+'\n"
                                                 "|-lhs: atom 2\n"
                                                 "`-rhs: atom 5"));
    CHECK_THAT(dumpExpression("2 - 5"), Contains("aexpr '-'\n"
                                                 "|-lhs: atom 2\n"
                                                 "`-rhs: atom 5"));
}

TEST_CASE("Parse shiftExpr", "[Parser]")
{
    CHECK_THAT(dumpExpression("2 << 5"), Contains("shiftExpr '<<'\n"
                                                  "|-lhs: atom 2\n"
                                                  "`-rhs: atom 5"));
    CHECK_THAT(dumpExpression("2 >> 5"), Contains("shiftExpr '>>'\n"
                                                  "|-lhs: atom 2\n"
                                                  "`-rhs: atom 5"));
}

TEST_CASE("Parse andExpr", "[Parser]")
{
    CHECK_THAT(dumpExpression("2 & 5"), Contains("andExpr '&'\n"
                                                 "|-lhs: atom 2\n"
                                                 "`-rhs: atom 5"));
}

TEST_CASE("Parse xorExpr", "[Parser]")
{
    CHECK_THAT(dumpExpression("2 ^ 5"), Contains("xorExpr '^'\n"
                                                 "|-lhs: atom 2\n"
                                                 "`-rhs: atom 5"));
}

TEST_CASE("Parse orExpr", "[Parser]")
{
    CHECK_THAT(dumpExpression("2 | 5"), Contains("orExpr '|'\n"
                                                 "|-lhs: atom 2\n"
                                                 "`-rhs: atom 5"));
}

TEST_CASE("Parse comparison", "[Parser]")
{
    CHECK_THAT(dumpExpression("2 < 5"), Contains("comparison\n"
                                                 "|-lhs: atom 2\n"
                                                 "`-'<': atom 5"));
    CHECK_THAT(dumpExpression("2 > 5"), Contains("comparison\n"
                                                 "|-lhs: atom 2\n"
                                                 "`-'>': atom 5"));
    CHECK_THAT(dumpExpression("2 <= 5"), Contains("comparison\n"
                                                  "|-lhs: atom 2\n"
                                                  "`-'<=': atom 5"));
    CHECK_THAT(dumpExpression("2 >= 5"), Contains("comparison\n"
                                                  "|-lhs: atom 2\n"
                                                  "`-'>=': atom 5"));
    CHECK_THAT(dumpExpression("2 == 5"), Contains("comparison\n"
                                                  "|-lhs: atom 2\n"
                                                  "`-'==': atom 5"));
    CHECK_THAT(dumpExpression("2 != 5"), Contains("comparison\n"
                                                  "|-lhs: atom 2\n"
                                                  "`-'!=': atom 5"));
    CHECK_THAT(dumpExpression("2 is 5"), Contains("comparison\n"
                                                  "|-lhs: atom 2\n"
                                                  "`-'is': atom 5"));
    CHECK_THAT(dumpExpression("2 is not 5"), Contains("comparison\n"
                                                      "|-lhs: atom 2\n"
                                                      "`-'is' 'not': atom 5"));
    CHECK_THAT(dumpExpression("2 in 5"), Contains("comparison\n"
                                                  "|-lhs: atom 2\n"
                                                  "`-'in': atom 5"));
    CHECK_THAT(dumpExpression("2 not in 5"), Contains("comparison\n"
                                                      "|-lhs: atom 2\n"
                                                      "`-'not' 'in': atom 5"));
}

TEST_CASE("Parse not test", "[Parser]")
{
    CHECK_THAT(dumpExpression("not 2"), Contains("notTest\n"
                                                 "`-atom 2"));
    CHECK_THAT(dumpExpression("not not 2"), Contains("notTest\n"
                                                     "`-notTest\n"
                                                     "  `-atom 2"));
}

TEST_CASE("Parse orTest", "[Parser]")
{
    CHECK_THAT(dumpExpression("2 or 5"), Contains("orTest 'or'\n"
                                                  "|-lhs: atom 2\n"
                                                  "`-rhs: atom 5"));
}

TEST_CASE("Parse andTest", "[Parser]")
{
    CHECK_THAT(dumpExpression("2 and 5"), Contains("andTest 'and'\n"
                                                   "|-lhs: atom 2\n"
                                                   "`-rhs: atom 5"));
}

TEST_CASE("Parse conditional", "[Parser]")
{
    CHECK_THAT(dumpExpression("2 if 3 else 5"), Contains("conditional expression\n"
                                                         "|-value: atom 2\n"
                                                         "|-condition: atom 3\n"
                                                         "`-elseValue: atom 5"));
}

TEST_CASE("Parse lambda", "[Parser]")
{
    CHECK_THAT(dumpExpression("lambda: 3"), Contains("lambda expression\n"
                                                     "`-atom 3"));
}

TEST_CASE("Parse assignment statement", "[Parser]")
{
    CHECK_THAT(dumpStatement("a = 3"), Contains("assignment statement\n"
                                                "|-target a\n"
                                                "`-atom 3"));
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
    CHECK_THAT(dumpStatement("[a] = 3"), Contains("assignment statement\n"
                                                  "|-target square\n"
                                                  "| `-target a\n"
                                                  "`-atom 3"));
    PARSER_EMITS("(yield 5) = 3", pylir::Diag::CANNOT_ASSIGN_TO_N, "yield expression");
    PARSER_EMITS("(c for c in f) = 3", pylir::Diag::CANNOT_ASSIGN_TO_N, "generator expression");
}

TEST_CASE("Parse augmented assignment statement", "[Parser]")
{
    CHECK_THAT(dumpStatement("a += 3"), Contains("augmented assignment '+='\n"
                                                 "|-augtarget a\n"
                                                 "`-atom 3"));
    PARSER_EMITS("+= 3", pylir::Diag::EXPECTED_N_BEFORE_N, "identifier", "assignment");
    PARSER_EMITS("a,b += 3", pylir::Diag::CANNOT_ASSIGN_TO_N, "multiple values");
    PARSER_EMITS("*b += 3", pylir::Diag::CANNOT_ASSIGN_TO_N, "starred item");
    PARSER_EMITS("(b) += 3", pylir::Diag::CANNOT_ASSIGN_TO_N, "enclosure");
}

TEST_CASE("Parse annotated assignment", "[Parser]")
{
    CHECK_THAT(dumpStatement("a:b"), Contains("annotated assignment\n"
                                              "|-augtarget a\n"
                                              "`-atom b"));
    CHECK_THAT(dumpStatement("a:b = 3"), Contains("annotated assignment\n"
                                                  "|-augtarget a\n"
                                                  "|-atom b\n"
                                                  "`-atom 3"));
}

TEST_CASE("Parse assert statement", "[Parser]")
{
    CHECK_THAT(dumpStatement("assert 5"), Contains("assert statement\n"
                                                   "`-condition: atom 5"));
    CHECK_THAT(dumpStatement("assert 5,3"), Contains("assert statement\n"
                                                     "|-condition: atom 5\n"
                                                     "`-message: atom 3"));
}

TEST_CASE("Parse single word statements", "[Parser]")
{
    CHECK_THAT(dumpStatement("pass"), Contains("pass statement"));
    CHECK_THAT(dumpStatement("continue"), Contains("continue statement"));
    CHECK_THAT(dumpStatement("break"), Contains("break statement"));
}

TEST_CASE("Parse del statement", "[Parser]")
{
    CHECK_THAT(dumpStatement("del a"), Contains("del statement\n"
                                                "`-target a"));
}

TEST_CASE("Parse return statement", "[Parser]")
{
    CHECK_THAT(dumpStatement("return"), Contains("return statement"));
    CHECK_THAT(dumpStatement("return 5"), Contains("return statement\n"
                                                   "`-atom 5"));
}

TEST_CASE("Parse raise statement", "[Parser]")
{
    CHECK_THAT(dumpStatement("raise"), Contains("raise statement"));
    CHECK_THAT(dumpStatement("raise 5"), Contains("raise statement\n"
                                                  "`-exception: atom 5"));
    CHECK_THAT(dumpStatement("raise 5 from 3"), Contains("raise statement\n"
                                                         "|-exception: atom 5\n"
                                                         "`-expression: atom 3"));
}

TEST_CASE("Parse global statement", "[Parser]")
{
    CHECK_THAT(dumpStatement("global a,b,c"), Contains("global a, b, c"));
}

TEST_CASE("Parse nonlocal statement", "[Parser]")
{
    CHECK_THAT(dumpStatement("nonlocal a,b,c"), Contains("nonlocal a, b, c"));
}

TEST_CASE("Parse stmt list", "[Parser]")
{
    CHECK_THAT(dumpAll("a = 3;b = 4"), Contains("file input\n"
                                                "`-stmt list\n"
                                                "  |-assignment statement\n"
                                                "  | |-target a\n"
                                                "  | `-atom 3\n"
                                                "  `-assignment statement\n"
                                                "    |-target b\n"
                                                "    `-atom 4"));
}
