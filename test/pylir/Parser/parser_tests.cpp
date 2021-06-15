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
}
