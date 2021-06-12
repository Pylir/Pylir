#include <catch2/catch.hpp>

#include <pylir/Parser/Parser.hpp>

TEST_CASE("Parse atom", "[Parser]")
{
    SECTION("Identifier")
    {
        pylir::Diag::Document document("test");
        pylir::Parser parser(document);
        auto atom = parser.parseAtom();
        REQUIRE(atom);
        REQUIRE(std::holds_alternative<pylir::Syntax::Atom::Identifier>(atom->variant));
        CHECK(std::get<std::string>(std::get<pylir::Syntax::Atom::Identifier>(atom->variant).token.getValue())
              == "test");
    }
}
