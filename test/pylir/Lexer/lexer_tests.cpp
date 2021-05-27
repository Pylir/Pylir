
#include <pylir/Lexer/Lexer.hpp>

#include <catch2/catch.hpp>

TEST_CASE("Lex comments", "[Lexer]")
{
    pylir::Lexer lexer("# comment", 0);
    CHECK(lexer.begin() == lexer.end());
}
