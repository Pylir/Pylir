
#include <catch2/catch.hpp>

#include <pylir/Diagnostics/DiagnosticMessages.hpp>
#include <pylir/Lexer/Lexer.hpp>

#include <iostream>

#include <fmt/format.h>

#define LEXER_EMITS(source, ...)                                              \
    [](std::string str)                                                       \
    {                                                                         \
        pylir::Diag::Document document(str);                                  \
        pylir::Lexer lexer(document);                                         \
        for (auto& token : lexer)                                             \
        {                                                                     \
            if (token.getTokenType() == pylir::TokenType::SyntaxError)        \
            {                                                                 \
                auto& error = std::get<std::string>(token.getValue());        \
                std::cerr << error;                                           \
                CHECK_THAT(error, Catch::Contains(fmt::format(__VA_ARGS__))); \
                return;                                                       \
            }                                                                 \
        }                                                                     \
        FAIL("No error emitted");                                             \
    }(source)

TEST_CASE("Lex comments", "[Lexer]")
{
    SECTION("Comment to end of line")
    {
        pylir::Diag::Document document("# comment\n"
                                       "");
        pylir::Lexer lexer(document, 1);
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
        pylir::Diag::Document document("# comment");
        pylir::Lexer lexer(document, 1);
        std::vector result(lexer.begin(), lexer.end());
        REQUIRE(result.size() == 1);
        auto& token = result.front();
        CHECK(token.getTokenType() == pylir::TokenType::Newline);
        CHECK(token.getFileId() == 1);
        CHECK(token.getOffset() == 9);
        CHECK(std::holds_alternative<std::monostate>(token.getValue()));
    }
}

TEST_CASE("Lex newlines", "[Lexer]")
{
    pylir::Diag::Document document("Windows\r\n"
                                   "Unix\n"
                                   "OldMac\r"
                                   "");
    pylir::Lexer lexer(document, 1);
    std::vector result(lexer.begin(), lexer.end());
    REQUIRE(result.size() == 7);
    CHECK(result[0].getTokenType() == pylir::TokenType::Identifier);
    CHECK(result[0].getOffset() == 0);
    CHECK(result[0].getSize() == 7);
    CHECK(result[1].getTokenType() == pylir::TokenType::Newline);
    CHECK(result[1].getOffset() == 7);
    CHECK(result[1].getSize() == 1);
    CHECK(result[2].getTokenType() == pylir::TokenType::Identifier);
    CHECK(result[2].getOffset() == 8);
    CHECK(result[2].getSize() == 4);
    CHECK(result[3].getTokenType() == pylir::TokenType::Newline);
    CHECK(result[3].getOffset() == 12);
    CHECK(result[3].getSize() == 1);
    CHECK(result[4].getTokenType() == pylir::TokenType::Identifier);
    CHECK(result[4].getOffset() == 13);
    CHECK(result[4].getSize() == 6);
    CHECK(result[5].getTokenType() == pylir::TokenType::Newline);
    CHECK(result[5].getOffset() == 19);
    CHECK(result[5].getSize() == 1);
}

TEST_CASE("Lex line continuation", "[Lexer]")
{
    SECTION("Correct")
    {
        pylir::Diag::Document document("\\\n"
                                       "");
        pylir::Lexer lexer(document, 1);
        std::vector result(lexer.begin(), lexer.end());
        REQUIRE(result.size() == 1);
        CHECK(result.front().getOffset() == 2); // It should be the EOF newline, not the one after the backslash
    }
    LEXER_EMITS("test\n"
                "\\",
                pylir::Diag::UNEXPECTED_EOF_WHILE_PARSING);
    LEXER_EMITS("test\n"
                "\\a",
                pylir::Diag::UNEXPECTED_CHARACTER_AFTER_LINE_CONTINUATION_CHARACTER);
}

TEST_CASE("Lex identifiers", "[Lexer]")
{
    SECTION("Unicode")
    {
        pylir::Diag::Document document("株式会社");
        pylir::Lexer lexer(document, 1);
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
        pylir::Diag::Document document("ＫＡＤＯＫＡＷＡ");
        pylir::Lexer lexer(document, 1);
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
    pylir::Diag::Document document(
        "False None True and as assert async await break class continue def del elif else except finally for from global if import in is lambda nonlocal not or pass raise return try while with yield");
    pylir::Lexer lexer(document, 1);
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

TEST_CASE("Lex string literals", "[Lexer]")
{
    SECTION("Normal")
    {
        SECTION("Quote")
        {
            pylir::Diag::Document document("'a text'");
            pylir::Lexer lexer(document);
            std::vector<pylir::Token> result(lexer.begin(), lexer.end());
            REQUIRE_FALSE(result.empty());
            auto& first = result[0];
            CHECK(first.getTokenType() == pylir::TokenType::StringLiteral);
            auto* str = std::get_if<std::string>(&first.getValue());
            REQUIRE(str);
            CHECK(*str == "a text");
        }
        SECTION("Double quote")
        {
            pylir::Diag::Document document("\"a text\"");
            pylir::Lexer lexer(document);
            std::vector<pylir::Token> result(lexer.begin(), lexer.end());
            REQUIRE_FALSE(result.empty());
            auto& first = result[0];
            CHECK(first.getTokenType() == pylir::TokenType::StringLiteral);
            auto* str = std::get_if<std::string>(&first.getValue());
            REQUIRE(str);
            CHECK(*str == "a text");
        }
        SECTION("Triple quote")
        {
            pylir::Diag::Document document("'''a text'''");
            pylir::Lexer lexer(document);
            std::vector<pylir::Token> result(lexer.begin(), lexer.end());
            REQUIRE_FALSE(result.empty());
            auto& first = result[0];
            CHECK(first.getTokenType() == pylir::TokenType::StringLiteral);
            auto* str = std::get_if<std::string>(&first.getValue());
            REQUIRE(str);
            CHECK(*str == "a text");
        }
        SECTION("Triple double quote")
        {
            pylir::Diag::Document document("\"\"\"a text\"\"\"");
            pylir::Lexer lexer(document);
            std::vector<pylir::Token> result(lexer.begin(), lexer.end());
            REQUIRE_FALSE(result.empty());
            auto& first = result[0];
            CHECK(first.getTokenType() == pylir::TokenType::StringLiteral);
            auto* str = std::get_if<std::string>(&first.getValue());
            REQUIRE(str);
            CHECK(*str == "a text");
        }
        LEXER_EMITS("'a text", pylir::Diag::EXPECTED_END_OF_LITERAL);
        LEXER_EMITS("'''a text", pylir::Diag::EXPECTED_END_OF_LITERAL);
    }
    SECTION("Newline")
    {
        pylir::Diag::Document document("'''\n"
                                       "a text\n"
                                       "'''");
        pylir::Lexer lexer(document);
        std::vector<pylir::Token> result(lexer.begin(), lexer.end());
        REQUIRE_FALSE(result.empty());
        auto& first = result[0];
        CHECK(first.getTokenType() == pylir::TokenType::StringLiteral);
        auto* str = std::get_if<std::string>(&first.getValue());
        REQUIRE(str);
        CHECK(*str == "\na text\n");
        LEXER_EMITS("'a text\n'", pylir::Diag::NEWLINE_NOT_ALLOWED_IN_LITERAL);
    }
    SECTION("Simple escapes")
    {
        pylir::Diag::Document document("'\\\\\\'\\\"\\a\\b\\f\\n\\r\\t\\v\\newline'");
        pylir::Lexer lexer(document);
        std::vector<pylir::Token> result(lexer.begin(), lexer.end());
        REQUIRE_FALSE(result.empty());
        auto& first = result[0];
        CHECK(first.getTokenType() == pylir::TokenType::StringLiteral);
        auto* str = std::get_if<std::string>(&first.getValue());
        REQUIRE(str);
        CHECK(*str == "\\'\"\a\b\f\n\r\t\v\n");
    }
    SECTION("Unicode name")
    {
        pylir::Diag::Document document("'\\N{Man in Business Suit Levitating}'");
        pylir::Lexer lexer(document);
        std::vector<pylir::Token> result(lexer.begin(), lexer.end());
        REQUIRE_FALSE(result.empty());
        auto& first = result[0];
        CHECK(first.getTokenType() == pylir::TokenType::StringLiteral);
        auto* str = std::get_if<std::string>(&first.getValue());
        REQUIRE(str);
        CHECK(*str == "\U0001F574");
        LEXER_EMITS("'\\N'", pylir::Diag::EXPECTED_OPEN_BRACE_AFTER_BACKSLASH_N);
        LEXER_EMITS("'\\N{wdwadwad}'", pylir::Diag::UNICODE_NAME_N_NOT_FOUND, "wdwadwad");
    }
    SECTION("Hex characters")
    {
        pylir::Diag::Document document("'\\xA7'");
        pylir::Lexer lexer(document);
        std::vector<pylir::Token> result(lexer.begin(), lexer.end());
        REQUIRE_FALSE(result.empty());
        auto& first = result[0];
        CHECK(first.getTokenType() == pylir::TokenType::StringLiteral);
        auto* str = std::get_if<std::string>(&first.getValue());
        REQUIRE(str);
        CHECK(*str == "§");
    }
    SECTION("Octal characters")
    {
        pylir::Diag::Document document("'\\247'");
        pylir::Lexer lexer(document);
        std::vector<pylir::Token> result(lexer.begin(), lexer.end());
        REQUIRE_FALSE(result.empty());
        auto& first = result[0];
        CHECK(first.getTokenType() == pylir::TokenType::StringLiteral);
        auto* str = std::get_if<std::string>(&first.getValue());
        REQUIRE(str);
        CHECK(*str == "§");
    }
    SECTION("Unicode escape")
    {
        SECTION("Small")
        {
            pylir::Diag::Document document("'\\u00A7'");
            pylir::Lexer lexer(document);
            std::vector<pylir::Token> result(lexer.begin(), lexer.end());
            REQUIRE_FALSE(result.empty());
            auto& first = result[0];
            CHECK(first.getTokenType() == pylir::TokenType::StringLiteral);
            auto* str = std::get_if<std::string>(&first.getValue());
            REQUIRE(str);
            CHECK(*str == "§");
        }
        SECTION("Big")
        {
            pylir::Diag::Document document("'\\U0001F574'");
            pylir::Lexer lexer(document);
            std::vector<pylir::Token> result(lexer.begin(), lexer.end());
            REQUIRE_FALSE(result.empty());
            auto& first = result[0];
            CHECK(first.getTokenType() == pylir::TokenType::StringLiteral);
            auto* str = std::get_if<std::string>(&first.getValue());
            REQUIRE(str);
            CHECK(*str == "\U0001F574");
        }
        LEXER_EMITS("'\\u343'", pylir::Diag::EXPECTED_N_MORE_HEX_CHARACTERS, 1);
        LEXER_EMITS("'\\ud869'", pylir::Diag::U_PLUS_N_IS_NOT_A_VALID_UNICODE_CODEPOINT, 0xd869);
    }
    SECTION("Raw strings")
    {
        SECTION("Immediately before")
        {
            pylir::Diag::Document document("r'\\\\\\\"\\a\\b\\f\\n\\r\\t\\v\\newline\\u343\\N'");
            pylir::Lexer lexer(document);
            std::vector<pylir::Token> result(lexer.begin(), lexer.end());
            REQUIRE_FALSE(result.empty());
            auto& first = result[0];
            CHECK(first.getTokenType() == pylir::TokenType::StringLiteral);
            auto* str = std::get_if<std::string>(&first.getValue());
            REQUIRE(str);
            CHECK(*str == "\\\\\\\"\\a\\b\\f\\n\\r\\t\\v\\newline\\u343\\N");
        }
        SECTION("Not immediately before")
        {
            pylir::Diag::Document document("r '\\n'");
            pylir::Lexer lexer(document);
            std::vector<pylir::Token> result(lexer.begin(), lexer.end());
            REQUIRE(result.size() >= 2);
            auto& first = result[0];
            CHECK(first.getTokenType() == pylir::TokenType::Identifier);
            auto* str = std::get_if<std::string>(&first.getValue());
            REQUIRE(str);
            CHECK(*str == "r");
            auto& second = result[1];
            CHECK(second.getTokenType() == pylir::TokenType::StringLiteral);
            str = std::get_if<std::string>(&second.getValue());
            REQUIRE(str);
            CHECK(*str == "\n");
        }
    }
    SECTION("Byte literals")
    {
        SECTION("Normal")
        {
            pylir::Diag::Document document("b'\\xC2\\xA7'");
            pylir::Lexer lexer(document);
            std::vector<pylir::Token> result(lexer.begin(), lexer.end());
            REQUIRE_FALSE(result.empty());
            auto& first = result[0];
            CHECK(first.getTokenType() == pylir::TokenType::BytesLiteral);
            auto* str = std::get_if<std::string>(&first.getValue());
            REQUIRE(str);
            CHECK(*str == "\xC2\xA7");
        }
        SECTION("Raw")
        {
            pylir::Diag::Document document("rb'\\xC2\\xA7'");
            pylir::Lexer lexer(document);
            std::vector<pylir::Token> result(lexer.begin(), lexer.end());
            REQUIRE_FALSE(result.empty());
            auto& first = result[0];
            CHECK(first.getTokenType() == pylir::TokenType::BytesLiteral);
            auto* str = std::get_if<std::string>(&first.getValue());
            REQUIRE(str);
            CHECK(*str == "\\xC2\\xA7");
        }
        LEXER_EMITS("b'§'", pylir::Diag::ONLY_ASCII_VALUES_ARE_ALLOWED_IN_BYTE_LITERALS);
        LEXER_EMITS("b'§'", pylir::Diag::USE_HEX_OR_OCTAL_ESCAPES_INSTEAD);
        LEXER_EMITS("b'§'", "\\xC2\\xA7");
    }
}

TEST_CASE("Lex integers", "[Lexer]")
{
    SECTION("Decimal")
    {
        pylir::Diag::Document document("30");
        pylir::Lexer lexer(document, 1);
        std::vector result(lexer.begin(), lexer.end());
        REQUIRE(result.size() == 2);
        auto& number = result[0];
        CHECK(number.getTokenType() == pylir::TokenType::IntegerLiteral);
        auto* apInt = std::get_if<llvm::APInt>(&number.getValue());
        REQUIRE(apInt);
        CHECK(apInt->getZExtValue() == 30);
    }
    SECTION("Binary")
    {
        pylir::Diag::Document document("0b10");
        pylir::Lexer lexer(document, 1);
        std::vector result(lexer.begin(), lexer.end());
        REQUIRE(result.size() == 2);
        auto& number = result[0];
        CHECK(number.getTokenType() == pylir::TokenType::IntegerLiteral);
        auto* apInt = std::get_if<llvm::APInt>(&number.getValue());
        REQUIRE(apInt);
        CHECK(apInt->getZExtValue() == 2);
    }
    SECTION("Octal")
    {
        pylir::Diag::Document document("0o30");
        pylir::Lexer lexer(document, 1);
        std::vector result(lexer.begin(), lexer.end());
        REQUIRE(result.size() == 2);
        auto& number = result[0];
        CHECK(number.getTokenType() == pylir::TokenType::IntegerLiteral);
        auto* apInt = std::get_if<llvm::APInt>(&number.getValue());
        REQUIRE(apInt);
        CHECK(apInt->getZExtValue() == 030);
    }
    SECTION("Hex")
    {
        pylir::Diag::Document document("0x30");
        pylir::Lexer lexer(document, 1);
        std::vector result(lexer.begin(), lexer.end());
        REQUIRE(result.size() == 2);
        auto& number = result[0];
        CHECK(number.getTokenType() == pylir::TokenType::IntegerLiteral);
        auto* apInt = std::get_if<llvm::APInt>(&number.getValue());
        REQUIRE(apInt);
        CHECK(apInt->getZExtValue() == 0x30);
    }
    LEXER_EMITS("0z3", pylir::Diag::INVALID_NUMBER_PREFIX_N, "0z");
    SECTION("Underline")
    {
        pylir::Diag::Document document("0x_3_0");
        pylir::Lexer lexer(document, 1);
        std::vector result(lexer.begin(), lexer.end());
        REQUIRE(result.size() == 2);
        auto& number = result[0];
        CHECK(number.getTokenType() == pylir::TokenType::IntegerLiteral);
        auto* apInt = std::get_if<llvm::APInt>(&number.getValue());
        REQUIRE(apInt);
        CHECK(apInt->getZExtValue() == 0x30);
        LEXER_EMITS("0x3__0", pylir::Diag::UNDERSCORE_ONLY_ALLOWED_BETWEEN_DIGITS);
    }
    SECTION("Null")
    {
        pylir::Diag::Document document("0000000000000");
        pylir::Lexer lexer(document, 1);
        std::vector result(lexer.begin(), lexer.end());
        REQUIRE(result.size() == 2);
        auto& number = result[0];
        CHECK(number.getTokenType() == pylir::TokenType::IntegerLiteral);
        auto* apInt = std::get_if<llvm::APInt>(&number.getValue());
        REQUIRE(apInt);
        CHECK(apInt->getZExtValue() == 0);
        LEXER_EMITS("00000000001", pylir::Diag::NUMBER_WITH_LEADING_ZEROS_NOT_ALLOWED);
    }
    LEXER_EMITS("0x3ll", pylir::Diag::INVALID_INTEGER_SUFFIX, "ll");
}
