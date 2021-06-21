
#pragma once

#include <llvm/ADT/APInt.h>

#include <pylir/Support/Macros.hpp>
#include <pylir/Support/Variant.hpp>

#include <cstdint>
#include <string>
#include <utility>
#include <variant>

#include <fmt/format.h>

namespace pylir
{
enum class TokenType : std::uint8_t
{
    SyntaxError,
    Newline,
    Identifier,
    FalseKeyword,
    NoneKeyword,
    TrueKeyword,
    AndKeyword,
    AsKeyword,
    AssertKeyword,
    AsyncKeyword,
    AwaitKeyword,
    BreakKeyword,
    ClassKeyword,
    ContinueKeyword,
    DefKeyword,
    DelKeyword,
    ElifKeyword,
    ElseKeyword,
    ExceptKeyword,
    FinallyKeyword,
    ForKeyword,
    FromKeyword,
    GlobalKeyword,
    IfKeyword,
    ImportKeyword,
    InKeyword,
    IsKeyword,
    LambdaKeyword,
    NonlocalKeyword,
    NotKeyword,
    OrKeyword,
    PassKeyword,
    RaiseKeyword,
    ReturnKeyword,
    TryKeyword,
    WhileKeyword,
    WithKeyword,
    YieldKeyword,
    StringLiteral,
    ByteLiteral,
    IntegerLiteral,
    FloatingPointLiteral,
    ComplexLiteral,
    Plus,
    Minus,
    Star,
    PowerOf,
    Divide,
    IntDivide,
    Remainder,
    AtSign,
    ShiftLeft,
    ShiftRight,
    BitAnd,
    BitOr,
    BitXor,
    BitNegate,
    Walrus,
    LessThan,
    GreaterThan,
    LessOrEqual,
    GreaterOrEqual,
    Equal,
    NotEqual,
    OpenParentheses,
    CloseParentheses,
    OpenSquareBracket,
    CloseSquareBracket,
    OpenBrace,
    CloseBrace,
    Comma,
    Colon,
    Dot,
    SemiColon,
    Assignment,
    Arrow,
    PlusAssignment,
    MinusAssignment,
    TimesAssignment,
    DivideAssignment,
    IntDivideAssignment,
    RemainderAssignment,
    AtAssignment,
    BitAndAssignment,
    BitOrAssignment,
    BitXorAssignment,
    ShiftRightAssignment,
    ShiftLeftAssignment,
    PowerOfAssignment,
    Indent,
    Dedent
};

class BaseToken
{
    int m_offset;
    int m_size;
    int m_fileId;

public:
    BaseToken(int offset, int size, int fileId) : m_offset(offset), m_size(size), m_fileId(fileId) {}

    [[nodiscard]] int getOffset() const
    {
        return m_offset;
    }

    [[nodiscard]] int getSize() const
    {
        return m_size;
    }

    [[nodiscard]] int getFileId() const
    {
        return m_fileId;
    }
};

class Token : public BaseToken
{
    TokenType m_tokenType;

public:
    using Variant = std::variant<std::monostate, std::string, llvm::APInt, double>;

private:
    Variant m_value;

public:
    Token(int offset, int size, int fileId, TokenType tokenType, Variant value = {})
        : BaseToken(offset, size, fileId), m_tokenType(tokenType), m_value(std::move(value))
    {
    }

    [[nodiscard]] TokenType getTokenType() const
    {
        return m_tokenType;
    }

    [[nodiscard]] const Variant& getValue() const
    {
        return m_value;
    }
};

class IdentifierToken : public BaseToken
{
    std::string m_value;

public:
    explicit IdentifierToken(const Token& token)
        : BaseToken(token.getOffset(), token.getSize(), token.getFileId()),
          m_value(pylir::get<std::string>(token.getValue()))
    {
    }

    explicit IdentifierToken(Token&& token)
        : BaseToken(token.getOffset(), token.getSize(), token.getFileId()),
          m_value(pylir::get<std::string>(std::move(token).getValue()))
    {
    }

    [[nodiscard]] TokenType getTokenType() const
    {
        return TokenType::Identifier;
    }

    [[nodiscard]] std::string_view getValue() const
    {
        return m_value;
    }
};

namespace Diag
{
template <class T, class>
struct LocationProvider;

template <class T>
struct LocationProvider<T, std::enable_if_t<std::is_base_of_v<BaseToken, T>>>
{
    static std::pair<std::size_t, std::size_t> getRange(const BaseToken& value) noexcept
    {
        return {value.getOffset(), value.getOffset() + value.getSize()};
    }
};
} // namespace Diag

} // namespace pylir

template <>
struct fmt::formatter<pylir::TokenType> : formatter<std::string_view>
{
    bool quote = false;

    constexpr auto parse(fmt::format_parse_context& ctx)
    {
        auto iter = ctx.begin();
        if (iter != ctx.end() && (*iter == 'q' || *iter == 'l'))
        {
            quote = *iter++ == 'q';
        }
        ctx.advance_to(iter);
        return fmt::formatter<std::string_view>::parse(ctx);
    }

    template <typename FormatContext>
    constexpr auto format(pylir::TokenType tokenType, FormatContext& ctx)
    {
        std::string_view name;
        switch (tokenType)
        {
            case pylir::TokenType::SyntaxError: PYLIR_UNREACHABLE;
            case pylir::TokenType::Newline: name = "newline"; break;
            case pylir::TokenType::Identifier: name = "identifier"; break;
            case pylir::TokenType::FalseKeyword: name = quote ? "'false'" : "false"; break;
            case pylir::TokenType::NoneKeyword: name = quote ? "'none'" : "none"; break;
            case pylir::TokenType::TrueKeyword: name = quote ? "'true'" : "true"; break;
            case pylir::TokenType::AndKeyword: name = quote ? "'and'" : "and"; break;
            case pylir::TokenType::AsKeyword: name = quote ? "'as'" : "as"; break;
            case pylir::TokenType::AssertKeyword: name = quote ? "'assert'" : "assert"; break;
            case pylir::TokenType::AsyncKeyword: name = quote ? "'async'" : "async"; break;
            case pylir::TokenType::AwaitKeyword: name = quote ? "'await'" : "await"; break;
            case pylir::TokenType::BreakKeyword: name = quote ? "'break'" : "break"; break;
            case pylir::TokenType::ClassKeyword: name = quote ? "'class'" : "class"; break;
            case pylir::TokenType::ContinueKeyword: name = quote ? "'continue'" : "continue"; break;
            case pylir::TokenType::DefKeyword: name = quote ? "'def'" : "def"; break;
            case pylir::TokenType::DelKeyword: name = quote ? "'del'" : "del"; break;
            case pylir::TokenType::ElifKeyword: name = quote ? "'elif'" : "elif"; break;
            case pylir::TokenType::ElseKeyword: name = quote ? "'else'" : "else"; break;
            case pylir::TokenType::ExceptKeyword: name = quote ? "'except'" : "expect"; break;
            case pylir::TokenType::FinallyKeyword: name = quote ? "'finally'" : "finally"; break;
            case pylir::TokenType::ForKeyword: name = quote ? "'for'" : "for"; break;
            case pylir::TokenType::FromKeyword: name = quote ? "'from'" : "from"; break;
            case pylir::TokenType::GlobalKeyword: name = quote ? "'global'" : "global"; break;
            case pylir::TokenType::IfKeyword: name = quote ? "'if'" : "if"; break;
            case pylir::TokenType::ImportKeyword: name = quote ? "'import'" : "import"; break;
            case pylir::TokenType::InKeyword: name = quote ? "'in'" : "in"; break;
            case pylir::TokenType::IsKeyword: name = quote ? "'is'" : "is"; break;
            case pylir::TokenType::LambdaKeyword: name = quote ? "'lambda'" : "lambda"; break;
            case pylir::TokenType::NonlocalKeyword: name = quote ? "'nonlocal'" : "nonlocal"; break;
            case pylir::TokenType::NotKeyword: name = quote ? "'not'" : "not"; break;
            case pylir::TokenType::OrKeyword: name = quote ? "'or'" : "or"; break;
            case pylir::TokenType::PassKeyword: name = quote ? "'pass'" : "pass"; break;
            case pylir::TokenType::RaiseKeyword: name = quote ? "'raise'" : "raise"; break;
            case pylir::TokenType::ReturnKeyword: name = quote ? "'return'" : "return"; break;
            case pylir::TokenType::TryKeyword: name = quote ? "'try'" : "try"; break;
            case pylir::TokenType::WhileKeyword: name = quote ? "'while'" : "while"; break;
            case pylir::TokenType::WithKeyword: name = quote ? "'with'" : "with"; break;
            case pylir::TokenType::YieldKeyword: name = quote ? "'yield'" : "yield"; break;
            case pylir::TokenType::StringLiteral: name = "string literal"; break;
            case pylir::TokenType::ByteLiteral: name = "byte literal"; break;
            case pylir::TokenType::IntegerLiteral: name = "integer literal"; break;
            case pylir::TokenType::FloatingPointLiteral: name = "floating point literal"; break;
            case pylir::TokenType::ComplexLiteral: name = "complex literal"; break;
            case pylir::TokenType::Plus: name = quote ? "'+'" : "+"; break;
            case pylir::TokenType::Minus: name = quote ? "'-'" : "-"; break;
            case pylir::TokenType::Star: name = quote ? "'*'" : "*"; break;
            case pylir::TokenType::PowerOf: name = quote ? "'**'" : "**"; break;
            case pylir::TokenType::Divide: name = quote ? "'/'" : "/"; break;
            case pylir::TokenType::IntDivide: name = quote ? "'//'" : "//"; break;
            case pylir::TokenType::Remainder: name = quote ? "'%'" : "%"; break;
            case pylir::TokenType::AtSign: name = quote ? "'@'" : "@"; break;
            case pylir::TokenType::ShiftLeft: name = quote ? "'<<'" : "<<"; break;
            case pylir::TokenType::ShiftRight: name = quote ? "'>>'" : ">>"; break;
            case pylir::TokenType::BitAnd: name = quote ? "'&'" : "&"; break;
            case pylir::TokenType::BitOr: name = quote ? "'|'" : "|"; break;
            case pylir::TokenType::BitXor: name = quote ? "'^'" : "^"; break;
            case pylir::TokenType::BitNegate: name = quote ? "'~'" : "~"; break;
            case pylir::TokenType::Walrus: name = quote ? "':='" : ":="; break;
            case pylir::TokenType::LessThan: name = quote ? "'<'" : "<"; break;
            case pylir::TokenType::GreaterThan: name = quote ? "'>'" : ">"; break;
            case pylir::TokenType::LessOrEqual: name = quote ? "'<='" : "<="; break;
            case pylir::TokenType::GreaterOrEqual: name = quote ? "'>='" : ">="; break;
            case pylir::TokenType::Equal: name = quote ? "'=='" : "=="; break;
            case pylir::TokenType::NotEqual: name = quote ? "'!='" : "!="; break;
            case pylir::TokenType::OpenParentheses: name = quote ? "'('" : "("; break;
            case pylir::TokenType::CloseParentheses: name = quote ? "')'" : ")"; break;
            case pylir::TokenType::OpenSquareBracket: name = quote ? "'['" : "["; break;
            case pylir::TokenType::CloseSquareBracket: name = quote ? "']'" : "]"; break;
            case pylir::TokenType::OpenBrace: name = quote ? "'{'" : "{"; break;
            case pylir::TokenType::CloseBrace: name = quote ? "'}'" : "}"; break;
            case pylir::TokenType::Comma: name = quote ? "','" : ","; break;
            case pylir::TokenType::Colon: name = quote ? "':'" : ":"; break;
            case pylir::TokenType::Dot: name = quote ? "'.'" : "."; break;
            case pylir::TokenType::SemiColon: name = quote ? "';'" : ";"; break;
            case pylir::TokenType::Assignment: name = quote ? "'='" : "="; break;
            case pylir::TokenType::Arrow: name = quote ? "'->'" : "->"; break;
            case pylir::TokenType::PlusAssignment: name = quote ? "'+='" : "+="; break;
            case pylir::TokenType::MinusAssignment: name = quote ? "'-='" : "-="; break;
            case pylir::TokenType::TimesAssignment: name = quote ? "'*='" : "*="; break;
            case pylir::TokenType::DivideAssignment: name = quote ? "'/='" : "/="; break;
            case pylir::TokenType::IntDivideAssignment: name = quote ? "'//='" : "//="; break;
            case pylir::TokenType::RemainderAssignment: name = quote ? "'%='" : "%="; break;
            case pylir::TokenType::AtAssignment: name = quote ? "'@='" : "@="; break;
            case pylir::TokenType::BitAndAssignment: name = quote ? "'&='" : "&="; break;
            case pylir::TokenType::BitOrAssignment: name = quote ? "'|='" : "|="; break;
            case pylir::TokenType::BitXorAssignment: name = quote ? "'^='" : "^="; break;
            case pylir::TokenType::ShiftRightAssignment: name = quote ? "'>>='" : ">>="; break;
            case pylir::TokenType::ShiftLeftAssignment: name = quote ? "'<<='" : "<<="; break;
            case pylir::TokenType::PowerOfAssignment: name = quote ? "'**='" : "**="; break;
            default: PYLIR_UNREACHABLE;
        }
        return formatter<std::string_view>::format(name, ctx);
    }
};
