
#include <pylir/Lexer/Lexer.hpp>

#include <catch2/catch.hpp>

TEST_CASE("Lex comments", "[Lexer]")
{
    SECTION("Comment to end of line")
    {
        pylir::Lexer lexer("# comment\n"
                           "",
                           1);
        std::vector result(lexer.begin(), lexer.end());
        REQUIRE(result.size() == 2);
        auto& token = result.front();
        CHECK(token.getTokenType() == pylir::TokenType::Newline);
        CHECK(token.getFileId() == 1);
        CHECK(token.getOffset() == 9);
        CHECK(std::holds_alternative<std::monostate>(token.getValue()));
    }
    SECTION("Comment to end of file")
    {
        pylir::Lexer lexer("# comment", 1);
        std::vector result(lexer.begin(), lexer.end());
        REQUIRE(result.size() == 1);
        auto& token = result.front();
        CHECK(token.getTokenType() == pylir::TokenType::Newline);
        CHECK(token.getFileId() == 1);
        CHECK(token.getOffset() == 9);
        CHECK(std::holds_alternative<std::monostate>(token.getValue()));
    }
}

TEST_CASE("Lex line continuation", "[Lexer]")
{
    SECTION("Correct")
    {
        pylir::Lexer lexer("\\\n"
                           "",
                           1);
        std::vector result(lexer.begin(), lexer.end());
        REQUIRE(result.size() == 1);
        CHECK(result.front().getOffset() == 2); // It should be the EOF newline, not the one after the backslash
    }
}

TEST_CASE("Lex identifiers", "[Lexer]")
{
    SECTION("Unicode")
    {
        pylir::Lexer lexer("株式会社", 1);
        std::vector result(lexer.begin(), lexer.end());
        REQUIRE(result.size() == 2);
        auto& identifier = result[0];
        CHECK(identifier.getTokenType() == pylir::TokenType::Identifier);
        auto* str = std::get_if<std::string>(&identifier.getValue());
        REQUIRE(str);
        CHECK(*str == "株式会社");
    }
    SECTION("Normalized")
    {
        pylir::Lexer lexer("ＫＡＤＯＫＡＷＡ", 1);
        std::vector result(lexer.begin(), lexer.end());
        REQUIRE(result.size() == 2);
        auto& identifier = result[0];
        CHECK(identifier.getTokenType() == pylir::TokenType::Identifier);
        auto* str = std::get_if<std::string>(&identifier.getValue());
        REQUIRE(str);
        CHECK(*str == "KADOKAWA");
    }
}
