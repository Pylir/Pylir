
#pragma once

#include <llvm/ADT/APInt.h>

#include <cstdint>
#include <string>
#include <utility>
#include <variant>

namespace pylir
{
enum class TokenType : std::uint8_t
{
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
};

class Token
{
    int m_offset;
    int m_size;
    int m_fileId;
    TokenType m_tokenType;

public:
    using Variant = std::variant<std::monostate, std::string, llvm::APInt, double>;

private:
    Variant m_value;

public:
    Token(int offset, int size, int fileId, TokenType tokenType, Variant value = {})
        : m_offset(offset), m_size(size), m_fileId(fileId), m_tokenType(tokenType), m_value(std::move(value))
    {
    }

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

    [[nodiscard]] TokenType getTokenType() const
    {
        return m_tokenType;
    }

    [[nodiscard]] const Variant& getValue() const
    {
        return m_value;
    }
};

} // namespace pylir
