
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

TEST_CASE("Lex keywords", "[Lexer]")
{
    pylir::Lexer lexer(
        "False None True and as assert async await break class continue def del elif else except finally for from global if import in is lambda nonlocal not or pass raise return try while with yield",
        1);
    std::vector<pylir::TokenType> result;
    std::transform(lexer.begin(), lexer.end(), std::back_inserter(result),
                   [](const pylir::Token& token) { return token.getTokenType(); });
    CHECK_THAT(result,
               Catch::Equals(std::vector<pylir::TokenType>{
                   pylir::TokenType::FalseKeyword,  pylir::TokenType::NoneKeyword,     pylir::TokenType::TrueKeyword,
                   pylir::TokenType::AndKeyword,    pylir::TokenType::AsKeyword,       pylir::TokenType::AssertKeyword,
                   pylir::TokenType::AsyncKeyword,  pylir::TokenType::AwaitKeyword,    pylir::TokenType::BreakKeyword,
                   pylir::TokenType::ClassKeyword,  pylir::TokenType::ContinueKeyword, pylir::TokenType::DefKeyword,
                   pylir::TokenType::DelKeyword,    pylir::TokenType::ElifKeyword,     pylir::TokenType::ElseKeyword,
                   pylir::TokenType::ExceptKeyword, pylir::TokenType::FinallyKeyword,  pylir::TokenType::ForKeyword,
                   pylir::TokenType::FromKeyword,   pylir::TokenType::GlobalKeyword,   pylir::TokenType::IfKeyword,
                   pylir::TokenType::ImportKeyword, pylir::TokenType::InKeyword,       pylir::TokenType::IsKeyword,
                   pylir::TokenType::LambdaKeyword, pylir::TokenType::NonlocalKeyword, pylir::TokenType::NotKeyword,
                   pylir::TokenType::OrKeyword,     pylir::TokenType::PassKeyword,     pylir::TokenType::RaiseKeyword,
                   pylir::TokenType::ReturnKeyword, pylir::TokenType::TryKeyword,      pylir::TokenType::WhileKeyword,
                   pylir::TokenType::WithKeyword,   pylir::TokenType::YieldKeyword,    pylir::TokenType::Newline}));
}
