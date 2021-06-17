#include <catch2/catch.hpp>

#include <pylir/Parser/Dumper.hpp>
#include <pylir/Parser/Parser.hpp>

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
    SECTION("Integer")
    {
        CHECK_THAT(dumpExpression(".5"), Contains("atom 0.5"));
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
        CHECK_THAT(dumpExpression("(yield from 5)"), Contains("yield from\n"
                                                              "`-atom 5"));
        CHECK_THAT(dumpExpression("(yield)"), Contains("yield empty"));
        CHECK_THAT(dumpExpression("(yield 5)"), Contains("yield list\n"
                                                         "`-atom 5"));
        CHECK_THAT(dumpExpression("(yield 5,5,)"), Contains("yield list\n"
                                                            "|-atom 5\n"
                                                            "`-atom 5"));
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
                                                "`-positional arguments\n"
                                                "  `-atom b"));
    CHECK_THAT(dumpExpression("a(b)"), Contains("call\n"
                                                "|-atom a\n"
                                                "`-positional arguments\n"
                                                "  `-atom b"));
    CHECK_THAT(dumpExpression("a(b,c)"), Contains("call\n"
                                                  "|-atom a\n"
                                                  "`-positional arguments\n"
                                                  "  |-atom b\n"
                                                  "  `-atom c"));
    CHECK_THAT(dumpExpression("a(b,*c)"), Contains("call\n"
                                                   "|-atom a\n"
                                                   "`-positional arguments\n"
                                                   "  |-atom b\n"
                                                   "  `-starred\n"
                                                   "    `-atom c"));
    CHECK_THAT(dumpExpression("a(b,c = 3)"), Contains("call\n"
                                                      "|-atom a\n"
                                                      "|-positional arguments\n"
                                                      "| `-atom b\n"
                                                      "`-starred keywords\n"
                                                      "  `-keyword item c\n"
                                                      "    `-atom 3"));
    CHECK_THAT(dumpExpression("a(b,c = 3,*b)"), Contains("call\n"
                                                         "|-atom a\n"
                                                         "|-positional arguments\n"
                                                         "| `-atom b\n"
                                                         "`-starred keywords\n"
                                                         "  |-keyword item c\n"
                                                         "  | `-atom 3\n"
                                                         "  `-starred expression\n"
                                                         "    `-atom b"));
    CHECK_THAT(dumpExpression("a(**b,c = 3)"), Contains("call\n"
                                                        "|-atom a\n"
                                                        "`-keyword arguments\n"
                                                        "  |-mapped expression\n"
                                                        "  | `-atom b\n"
                                                        "  `-keyword item c\n"
                                                        "    `-atom 3"));
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

