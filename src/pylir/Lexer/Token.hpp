//  Licensed under the Apache License v2.0 with LLVM Exceptions.
//  See https://llvm.org/LICENSE.txt for license information.
//  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <llvm/ADT/MapVector.h>
#include <llvm/ADT/SetVector.h>

#include <pylir/Diagnostics/LocationProvider.hpp>
#include <pylir/Support/BigInt.hpp>
#include <pylir/Support/Macros.hpp>
#include <pylir/Support/Variant.hpp>

#include <cstdint>
#include <string>
#include <utility>
#include <variant>

#include <fmt/format.h>

namespace pylir {
enum class TokenType : std::uint8_t {
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

class BaseToken {
  std::uint32_t m_offset;
  std::uint32_t m_size;

public:
  BaseToken(std::uint32_t offset, std::uint32_t size)
      : m_offset(offset), m_size(size) {}

  [[nodiscard]] std::uint32_t getOffset() const {
    return m_offset;
  }

  [[nodiscard]] std::uint32_t getSize() const {
    return m_size;
  }
};

class Token : public BaseToken {
  TokenType m_tokenType;

public:
  using Variant = std::variant<std::monostate, std::string, BigInt, double>;

private:
  Variant m_value;

public:
  Token(uint32_t offset, uint32_t size, TokenType tokenType, Variant value = {})
      : BaseToken(offset, size), m_tokenType(tokenType),
        m_value(std::move(value)) {}

  [[nodiscard]] TokenType getTokenType() const {
    return m_tokenType;
  }

  [[nodiscard]] const Variant& getValue() const {
    return m_value;
  }
};

class IdentifierToken : public BaseToken {
  std::string m_value;

  friend struct llvm::DenseMapInfo<IdentifierToken>;

  explicit IdentifierToken(std::string value)
      : BaseToken(0, 0), m_value(std::move(value)) {}

public:
  explicit IdentifierToken(const Token& token)
      : BaseToken(token.getOffset(), token.getSize()),
        m_value(pylir::get<std::string>(token.getValue())) {}

  explicit IdentifierToken(Token&& token)
      : BaseToken(token.getOffset(), token.getSize()),
        m_value(pylir::get<std::string>(std::move(token).getValue())) {}

  [[nodiscard]] TokenType getTokenType() const {
    return TokenType::Identifier;
  }

  [[nodiscard]] std::string_view getValue() const {
    return m_value;
  }
};

} // namespace pylir

template <>
struct llvm::DenseMapInfo<pylir::IdentifierToken> {
  static pylir::IdentifierToken getEmptyKey() {
    return pylir::IdentifierToken("");
  }

  static pylir::IdentifierToken getTombstoneKey() {
    return pylir::IdentifierToken("0");
  }

  static unsigned getHashValue(const pylir::IdentifierToken& val) {
    return llvm::hash_combine_range(val.getValue().begin(),
                                    val.getValue().end());
  }

  static bool isEqual(const pylir::IdentifierToken& lhs,
                      const pylir::IdentifierToken& rhs) {
    return lhs.getValue() == rhs.getValue();
  }
};

namespace pylir {

using IdentifierSet = llvm::SetVector<IdentifierToken>;
template <class T>
using IdentifierMap = llvm::MapVector<IdentifierToken, T>;

namespace Diag {
template <class T>
struct LocationProvider<T, std::enable_if_t<std::is_base_of_v<BaseToken, T>>> {
  static Location getRange(const BaseToken& value) noexcept {
    return Location(value.getOffset(), value.getOffset() + value.getSize());
  }

  using non_lazy = void;
};
} // namespace Diag

} // namespace pylir

template <>
struct fmt::formatter<pylir::TokenType> : formatter<std::string_view> {
  bool quote = false;

  constexpr auto parse(fmt::format_parse_context& ctx) {
    const auto* iter = ctx.begin();
    if (iter != ctx.end() && (*iter == 'q' || *iter == 'l'))
      quote = *iter++ == 'q';

    ctx.advance_to(iter);
    return fmt::formatter<std::string_view>::parse(ctx);
  }

  template <typename FormatContext>
  constexpr auto format(pylir::TokenType tokenType, FormatContext& ctx) {
    std::string_view name;
    switch (tokenType) {
    case pylir::TokenType::SyntaxError: PYLIR_UNREACHABLE;
    case pylir::TokenType::Newline: name = "newline"; break;
    case pylir::TokenType::Identifier: name = "identifier"; break;
    case pylir::TokenType::FalseKeyword:
      name = quote ? "'False'" : "False";
      break;
    case pylir::TokenType::NoneKeyword: name = quote ? "'None'" : "None"; break;
    case pylir::TokenType::TrueKeyword: name = quote ? "'True'" : "True"; break;
    case pylir::TokenType::AndKeyword: name = quote ? "'and'" : "and"; break;
    case pylir::TokenType::AsKeyword: name = quote ? "'as'" : "as"; break;
    case pylir::TokenType::AssertKeyword:
      name = quote ? "'assert'" : "assert";
      break;
    case pylir::TokenType::AsyncKeyword:
      name = quote ? "'async'" : "async";
      break;
    case pylir::TokenType::AwaitKeyword:
      name = quote ? "'await'" : "await";
      break;
    case pylir::TokenType::BreakKeyword:
      name = quote ? "'break'" : "break";
      break;
    case pylir::TokenType::ClassKeyword:
      name = quote ? "'class'" : "class";
      break;
    case pylir::TokenType::ContinueKeyword:
      name = quote ? "'continue'" : "continue";
      break;
    case pylir::TokenType::DefKeyword: name = quote ? "'def'" : "def"; break;
    case pylir::TokenType::DelKeyword: name = quote ? "'del'" : "del"; break;
    case pylir::TokenType::ElifKeyword: name = quote ? "'elif'" : "elif"; break;
    case pylir::TokenType::ElseKeyword: name = quote ? "'else'" : "else"; break;
    case pylir::TokenType::ExceptKeyword:
      name = quote ? "'except'" : "expect";
      break;
    case pylir::TokenType::FinallyKeyword:
      name = quote ? "'finally'" : "finally";
      break;
    case pylir::TokenType::ForKeyword: name = quote ? "'for'" : "for"; break;
    case pylir::TokenType::FromKeyword: name = quote ? "'from'" : "from"; break;
    case pylir::TokenType::GlobalKeyword:
      name = quote ? "'global'" : "global";
      break;
    case pylir::TokenType::IfKeyword: name = quote ? "'if'" : "if"; break;
    case pylir::TokenType::ImportKeyword:
      name = quote ? "'import'" : "import";
      break;
    case pylir::TokenType::InKeyword: name = quote ? "'in'" : "in"; break;
    case pylir::TokenType::IsKeyword: name = quote ? "'is'" : "is"; break;
    case pylir::TokenType::LambdaKeyword:
      name = quote ? "'lambda'" : "lambda";
      break;
    case pylir::TokenType::NonlocalKeyword:
      name = quote ? "'nonlocal'" : "nonlocal";
      break;
    case pylir::TokenType::NotKeyword: name = quote ? "'not'" : "not"; break;
    case pylir::TokenType::OrKeyword: name = quote ? "'or'" : "or"; break;
    case pylir::TokenType::PassKeyword: name = quote ? "'pass'" : "pass"; break;
    case pylir::TokenType::RaiseKeyword:
      name = quote ? "'raise'" : "raise";
      break;
    case pylir::TokenType::ReturnKeyword:
      name = quote ? "'return'" : "return";
      break;
    case pylir::TokenType::TryKeyword: name = quote ? "'try'" : "try"; break;
    case pylir::TokenType::WhileKeyword:
      name = quote ? "'while'" : "while";
      break;
    case pylir::TokenType::WithKeyword: name = quote ? "'with'" : "with"; break;
    case pylir::TokenType::YieldKeyword:
      name = quote ? "'yield'" : "yield";
      break;
    case pylir::TokenType::StringLiteral: name = "string literal"; break;
    case pylir::TokenType::ByteLiteral: name = "byte literal"; break;
    case pylir::TokenType::IntegerLiteral: name = "integer literal"; break;
    case pylir::TokenType::FloatingPointLiteral:
      name = "floating point literal";
      break;
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
    case pylir::TokenType::CloseSquareBracket:
      name = quote ? "']'" : "]";
      break;
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
    case pylir::TokenType::DivideAssignment:
      name = quote ? "'/='" : "/=";
      break;
    case pylir::TokenType::IntDivideAssignment:
      name = quote ? "'//='" : "//=";
      break;
    case pylir::TokenType::RemainderAssignment:
      name = quote ? "'%='" : "%=";
      break;
    case pylir::TokenType::AtAssignment: name = quote ? "'@='" : "@="; break;
    case pylir::TokenType::BitAndAssignment:
      name = quote ? "'&='" : "&=";
      break;
    case pylir::TokenType::BitOrAssignment: name = quote ? "'|='" : "|="; break;
    case pylir::TokenType::BitXorAssignment:
      name = quote ? "'^='" : "^=";
      break;
    case pylir::TokenType::ShiftRightAssignment:
      name = quote ? "'>>='" : ">>=";
      break;
    case pylir::TokenType::ShiftLeftAssignment:
      name = quote ? "'<<='" : "<<=";
      break;
    case pylir::TokenType::PowerOfAssignment:
      name = quote ? "'**='" : "**=";
      break;
    case pylir::TokenType::Indent: name = "indent"; break;
    case pylir::TokenType::Dedent: name = "dedent"; break;
    }
    return formatter<std::string_view>::format(name, ctx);
  }
};
