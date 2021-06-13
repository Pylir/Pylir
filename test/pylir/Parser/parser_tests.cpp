#include <catch2/catch.hpp>

#include <pylir/Parser/Dumper.hpp>
#include <pylir/Parser/Parser.hpp>

#define dumpExpression(source)                                \
    []() -> std::string                                       \
    {                                                         \
        pylir::Diag::Document document(source);               \
        pylir::Parser parser(document);                       \
        auto assignment = parser.parseAssignmentExpression(); \
        REQUIRE(assignment);                                  \
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
}
