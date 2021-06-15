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
    }
}
