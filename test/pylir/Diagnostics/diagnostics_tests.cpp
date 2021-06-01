#include <catch2/catch.hpp>

#include <pylir/Diagnostics/DiagnosticsBuilder.hpp>

#include <iostream>

TEST_CASE("Diagnostics labels", "[Diag]")
{
    SECTION("Simple")
    {
        pylir::Diag::Document document("A normal text", "filename");
        auto result = pylir::Diag::DiagnosticsBuilder(document, 2, "A message").addLabel(2, 8, "Label").emitError();
        CHECK_THAT(result, Catch::Contains("   1 | A normal\n"
                                           "     |   ~~~~~~\n"
                                           "     |      |\n"
                                           "     |      Label",
                                           Catch::CaseSensitive::Yes));
    }
}
