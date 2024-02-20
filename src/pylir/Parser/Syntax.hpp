//  Licensed under the Apache License v2.0 with LLVM Exceptions.
//  See https://llvm.org/LICENSE.txt for license information.
//  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <llvm/ADT/STLExtras.h>

#include <pylir/Lexer/Token.hpp>
#include <pylir/Support/AbstractIntrusiveVariant.hpp>

namespace pylir::Syntax {

struct Expression
    : public AbstractIntrusiveVariant<
          Expression, struct BinOp, struct Atom, struct AttributeRef,
          struct Subscription, struct Slice, struct Assignment,
          struct Conditional, struct Call, struct Lambda, struct UnaryOp,
          struct Yield, struct Generator, struct TupleConstruct,
          struct ListDisplay, struct SetDisplay, struct DictDisplay,
          struct Comparison> {
  using AbstractIntrusiveVariant::AbstractIntrusiveVariant;
};

struct Atom : Expression::Base<Atom> {
  Token token;
};

struct BinOp : Expression::Base<BinOp> {
  IntrVarPtr<Expression> lhs;
  Token operation;
  IntrVarPtr<Expression> rhs;
};

struct UnaryOp : Expression::Base<UnaryOp> {
  Token operation;
  IntrVarPtr<Expression> expression;
};

struct Comparison : Expression::Base<Comparison> {
  IntrVarPtr<Expression> first;
  struct Operator {
    Token firstToken;
    std::optional<Token> secondToken;
  };
  std::vector<std::pair<Operator, IntrVarPtr<Expression>>> rest;
};

/// While this reuses the structs defined for 'Expression', due to semantic and
/// syntactic restriction for the target grammar it may only be one of:
/// * TupleConstruct
/// * ListDisplay without comprehension
/// * Atom but only an Identifier token
/// * Slice
/// * Subscription
/// * AttributeRef
///
/// defined recursively.
using Target = Expression;

template <class T>
constexpr bool validTargetType() {
  return llvm::is_one_of<T, Atom, Subscription, Slice, AttributeRef,
                         TupleConstruct, ListDisplay>{};
}

struct AttributeRef : Expression::Base<AttributeRef> {
  IntrVarPtr<Expression> object;
  BaseToken dot;
  IdentifierToken identifier;
};

struct Subscription : Expression::Base<Subscription> {
  IntrVarPtr<Expression> object;
  BaseToken openSquareBracket;
  IntrVarPtr<Expression> index;
  BaseToken closeSquareBracket;
};

struct Slice : Expression::Base<Slice> {
  IntrVarPtr<Expression> maybeLowerBound;
  BaseToken firstColon;
  IntrVarPtr<Expression> maybeUpperBound;
  std::optional<BaseToken> maybeSecondColon;
  IntrVarPtr<Expression> maybeStride;
};

struct Assignment : Expression::Base<Assignment> {
  IdentifierToken variable;
  BaseToken walrus;
  IntrVarPtr<Expression> expression;
};

struct Conditional : Expression::Base<Conditional> {
  IntrVarPtr<Expression> trueValue;
  BaseToken ifToken;
  IntrVarPtr<Expression> condition;
  BaseToken elseToken;
  IntrVarPtr<Expression> elseValue;
};

struct Argument {
  std::optional<IdentifierToken> maybeName;
  std::optional<Token> maybeExpansionsOrEqual;
  IntrVarPtr<Expression> expression;
};

struct CompIf;

struct CompFor {
  std::optional<BaseToken> awaitToken;
  BaseToken forToken;
  IntrVarPtr<Target> targets;
  BaseToken inToken;
  IntrVarPtr<Expression> test;
  std::variant<std::monostate, std::unique_ptr<CompFor>,
               std::unique_ptr<CompIf>>
      compIter;
};

struct CompIf {
  BaseToken ifToken;
  IntrVarPtr<Expression> test;
  std::variant<std::monostate, std::unique_ptr<CompFor>,
               std::unique_ptr<CompIf>>
      compIter;
};

struct Comprehension {
  IntrVarPtr<Expression> expression;
  CompFor compFor;
};

struct Call : Expression::Base<Call> {
  IntrVarPtr<Expression> expression;
  BaseToken openParenth;
  std::variant<std::vector<Argument>, Comprehension> variant;
  BaseToken closeParenth;
};

struct Parameter {
  enum Kind {
    Normal,
    PosOnly,
    KeywordOnly,
    PosRest,
    KeywordRest,
  } kind;
  std::optional<BaseToken> maybeStars;
  IdentifierToken name;
  std::optional<BaseToken> maybeColon;
  IntrVarPtr<Expression> maybeType;
  IntrVarPtr<Expression> maybeDefault;
};

struct Scope {
  enum Kind { Local, Cell, NonLocal, Global, Unknown };

  IdentifierMap<Kind> identifiers;
};

struct Lambda : Expression::Base<Lambda> {
  BaseToken lambdaKeyword;
  std::vector<Parameter> parameters;
  BaseToken colon;
  IntrVarPtr<Expression> expression;

  Scope scope;
};

struct Yield : Expression::Base<Yield> {
  BaseToken yieldToken;
  std::optional<BaseToken> fromToken;
  IntrVarPtr<Expression> maybeExpression;
};

struct Generator : Expression::Base<Generator> {
  BaseToken openParenth;
  IntrVarPtr<Expression> expression;
  CompFor compFor;
  BaseToken closeParenth;
};

struct StarredItem {
  std::optional<BaseToken> maybeStar;
  IntrVarPtr<Expression> expression;
};

struct TupleConstruct : Expression::Base<TupleConstruct> {
  // This and 'maybeCloseBracket' are guaranteed to be active when 'items' is
  // empty.
  std::optional<BaseToken> maybeOpenBracket;
  std::vector<StarredItem> items;
  std::optional<BaseToken> maybeCloseBracket;
};

struct ListDisplay : Expression::Base<ListDisplay> {
  BaseToken openSquare;
  std::variant<std::vector<StarredItem>, Comprehension> variant;
  BaseToken closeSquare;
};

struct SetDisplay : Expression::Base<SetDisplay> {
  BaseToken openBrace;
  std::variant<std::vector<StarredItem>, Comprehension> variant;
  BaseToken closeBrace;
};

struct DictDisplay : Expression::Base<DictDisplay> {
  BaseToken openBrace;

  struct KeyDatum {
    IntrVarPtr<Expression> key;
    BaseToken colonOrPowerOf;
    IntrVarPtr<Expression> maybeValue;
  };

  struct DictComprehension {
    IntrVarPtr<Expression> first;
    BaseToken colon;
    IntrVarPtr<Expression> second;
    CompFor compFor;
  };

  std::variant<std::vector<KeyDatum>, DictComprehension> variant;
  BaseToken closeBrace;
};

struct SimpleStmt
    : AbstractIntrusiveVariant<
          SimpleStmt, struct ExpressionStmt, struct AssertStmt,
          struct AssignmentStmt, struct SingleTokenStmt, struct DelStmt,
          struct ReturnStmt, struct RaiseStmt, struct ImportStmt,
          struct FutureStmt, struct GlobalOrNonLocalStmt> {
  using AbstractIntrusiveVariant::AbstractIntrusiveVariant;
};

struct ExpressionStmt : SimpleStmt::Base<ExpressionStmt> {
  IntrVarPtr<Expression> expression;
};

struct AssertStmt : SimpleStmt::Base<AssertStmt> {
  BaseToken assertKeyword;
  IntrVarPtr<Expression> condition;
  IntrVarPtr<Expression> maybeMessage;
};

struct AssignmentStmt : SimpleStmt::Base<AssignmentStmt> {
  std::vector<std::pair<IntrVarPtr<Target>, Token>> targets;
  IntrVarPtr<Expression> maybeAnnotation;
  IntrVarPtr<Expression> maybeExpression;
};

struct SingleTokenStmt : SimpleStmt::Base<SingleTokenStmt> {
  Token token;
};

struct DelStmt : SimpleStmt::Base<DelStmt> {
  BaseToken del;
  IntrVarPtr<Target> targetList;
};

struct ReturnStmt : SimpleStmt::Base<ReturnStmt> {
  BaseToken returnKeyword;
  IntrVarPtr<Expression> maybeExpression;
};

struct RaiseStmt : SimpleStmt::Base<RaiseStmt> {
  BaseToken raise;
  IntrVarPtr<Expression> maybeException;
  IntrVarPtr<Expression> maybeCause;
};

struct ImportStmt : SimpleStmt::Base<ImportStmt> {
  struct Module {
    std::vector<IdentifierToken> identifiers;
  };

  struct RelativeModule {
    std::vector<BaseToken> dots;
    std::optional<Module> module;
  };

  struct ImportAs {
    BaseToken import;
    std::vector<std::pair<Module, std::optional<IdentifierToken>>> modules;
  };

  struct FromImport {
    BaseToken from;
    RelativeModule relativeModule;
    BaseToken import;
    std::vector<std::pair<IdentifierToken, std::optional<IdentifierToken>>>
        imports;
  };

  struct ImportAll {
    BaseToken from;
    RelativeModule relativeModule;
    BaseToken import;
    BaseToken star;
  };

  std::variant<ImportAs, FromImport, ImportAll> variant;
};

struct FutureStmt : SimpleStmt::Base<FutureStmt> {
  BaseToken from;
  BaseToken future;
  BaseToken import;
  std::vector<std::pair<IdentifierToken, std::optional<IdentifierToken>>>
      imports;
};

struct GlobalOrNonLocalStmt : SimpleStmt::Base<GlobalOrNonLocalStmt> {
  Token token;
  std::vector<IdentifierToken> identifiers;
};

struct Suite;

struct CompoundStmt
    : AbstractIntrusiveVariant<CompoundStmt, struct IfStmt, struct WhileStmt,
                               struct ForStmt, struct TryStmt, struct WithStmt,
                               struct FuncDef, struct ClassDef> {
  using AbstractIntrusiveVariant::AbstractIntrusiveVariant;
};

struct IfStmt : CompoundStmt::Base<IfStmt> {
  BaseToken ifKeyword;
  IntrVarPtr<Expression> condition;
  BaseToken colon;
  std::unique_ptr<Suite> suite;
  struct Elif {
    BaseToken elif;
    IntrVarPtr<Expression> condition;
    BaseToken colon;
    std::unique_ptr<Suite> suite;
  };
  std::vector<Elif> elifs;
  struct Else {
    BaseToken elseKeyowrd;
    BaseToken colon;
    std::unique_ptr<Suite> suite;
  };
  std::optional<Else> elseSection;
};

struct WhileStmt : CompoundStmt::Base<WhileStmt> {
  BaseToken whileKeyword;
  IntrVarPtr<Expression> condition;
  BaseToken colon;
  std::unique_ptr<Suite> suite;
  std::optional<IfStmt::Else> elseSection;
};

struct ForStmt : CompoundStmt::Base<ForStmt> {
  std::optional<BaseToken> maybeAsyncKeyword;
  BaseToken forKeyword;
  IntrVarPtr<Target> targetList;
  BaseToken inKeyword;
  IntrVarPtr<Expression> expression;
  BaseToken colon;
  std::unique_ptr<Suite> suite;
  std::optional<IfStmt::Else> elseSection;
};

struct TryStmt : CompoundStmt::Base<TryStmt> {
  BaseToken tryKeyword;
  BaseToken colon;
  std::unique_ptr<Suite> suite;
  struct ExceptArgs {
    BaseToken exceptKeyword;
    IntrVarPtr<Expression> filter;
    std::optional<IdentifierToken> maybeName;
    BaseToken colon;
    std::unique_ptr<Suite> suite;
  };
  std::vector<ExceptArgs> excepts;
  struct ExceptAll {
    BaseToken exceptKeyword;
    BaseToken colon;
    std::unique_ptr<Suite> suite;
  };
  std::optional<ExceptAll> maybeExceptAll;
  std::optional<IfStmt::Else> elseSection;
  struct Finally {
    BaseToken finally;
    BaseToken colon;
    std::unique_ptr<Suite> suite;
  };
  std::optional<Finally> finally;
};

struct WithStmt : CompoundStmt::Base<WithStmt> {
  std::optional<BaseToken> maybeAsyncKeyword;
  BaseToken withKeyword;
  struct WithItem {
    IntrVarPtr<Expression> expression;
    IntrVarPtr<Target> maybeTarget;
  };
  std::vector<WithItem> items;
  BaseToken colon;
  std::unique_ptr<Suite> suite;
};

struct Decorator {
  BaseToken atSign;
  IntrVarPtr<Expression> expression;
  BaseToken newline;
};

struct FuncDef : CompoundStmt::Base<FuncDef> {
  std::vector<Decorator> decorators;
  std::optional<BaseToken> maybeAsyncKeyword;
  BaseToken def;
  IdentifierToken funcName;
  BaseToken openParenth;
  std::vector<Parameter> parameterList;
  BaseToken closeParenth;
  IntrVarPtr<Expression> maybeSuffix;
  BaseToken colon;
  std::unique_ptr<Suite> suite;

  Scope scope;
  bool isConst = false;
  bool isExported = false;
};

struct ClassDef : CompoundStmt::Base<ClassDef> {
  std::vector<Decorator> decorators;
  BaseToken classKeyword;
  IdentifierToken className;
  struct Inheritance {
    BaseToken openParenth;
    std::vector<Argument> argumentList;
    BaseToken closeParenth;
  };
  std::optional<Inheritance> inheritance;
  BaseToken colon;
  std::unique_ptr<Suite> suite;

  Scope scope;
  bool isConst = false;
  bool isExported = false;
};

struct Suite {
  using Variant =
      std::variant<IntrVarPtr<SimpleStmt>, IntrVarPtr<CompoundStmt>>;
  std::vector<Variant> statements;
};

struct FileInput {
  Suite input;
  IdentifierSet globals;
};

struct Intrinsic {
  /// Name of the intrinsic. These are all identifier joined with dots.
  /// Includes the 'pylir.intr' prefix.
  std::string name;
  /// All identifier tokens making up the name. Main use-case is for the
  /// purpose of the location in the source code.
  llvm::SmallVector<IdentifierToken> identifiers;
};

/// Checks whether 'expression' is a reference to an intrinsic. An intrinsic
/// consists of a series of attribute references resulting in the syntax:
/// "pylir" `.` "intr" { `.` identifier }.
/// Returns an empty optional if the expression is not an intrinsic reference.
std::optional<Intrinsic>
checkForIntrinsic(const Syntax::Expression& expression);

} // namespace pylir::Syntax

namespace pylir::Diag {

template <>
struct LocationProvider<Syntax::TupleConstruct> {
  static std::pair<std::size_t, std::size_t>
  getRange(const Syntax::TupleConstruct& value) noexcept;
};

template <>
struct LocationProvider<Syntax::Lambda> {
  static std::pair<std::size_t, std::size_t>
  getRange(const Syntax::Lambda& value) noexcept;
};

template <class T>
struct LocationProvider<T, std::enable_if_t<IsAbstractVariantConcrete<T>{}>> {
  static std::pair<std::size_t, std::size_t> getRange(const T& value) noexcept {
    return value.match([&](auto&& sub) { return rangeLoc(sub); });
  }

  static std::size_t getPoint(const T& value) noexcept {
    return value.match([&](auto&& sub) { return pointLoc(sub); });
  }
};

template <>
struct LocationProvider<Syntax::Conditional> {
  static std::pair<std::size_t, std::size_t>
  getRange(const Syntax::Conditional& value) noexcept;
};

template <>
struct LocationProvider<Syntax::BinOp> {
  static std::pair<std::size_t, std::size_t>
  getRange(const Syntax::BinOp& value) noexcept;
};

template <>
struct LocationProvider<Syntax::Comparison> {
  static std::pair<std::size_t, std::size_t>
  getRange(const Syntax::Comparison& value) noexcept;
};

template <>
struct LocationProvider<Syntax::UnaryOp> {
  static std::pair<std::size_t, std::size_t>
  getRange(const Syntax::UnaryOp& value) noexcept;
};

template <>
struct LocationProvider<Syntax::Call> {
  static std::pair<std::size_t, std::size_t>
  getRange(const Syntax::Call& value) noexcept;
};

template <>
struct LocationProvider<Syntax::DictDisplay> {
  static std::pair<std::size_t, std::size_t>
  getRange(const Syntax::DictDisplay& value) noexcept;
};

template <>
struct LocationProvider<Syntax::SetDisplay> {
  static std::pair<std::size_t, std::size_t>
  getRange(const Syntax::SetDisplay& value) noexcept;
};

template <>
struct LocationProvider<Syntax::ListDisplay> {
  static std::pair<std::size_t, std::size_t>
  getRange(const Syntax::ListDisplay& value) noexcept;
};

template <>
struct LocationProvider<Syntax::Yield> {
  static std::pair<std::size_t, std::size_t>
  getRange(const Syntax::Yield& value) noexcept;
};

template <>
struct LocationProvider<Syntax::Generator> {
  static std::pair<std::size_t, std::size_t>
  getRange(const Syntax::Generator& value) noexcept;
};

template <>
struct LocationProvider<Syntax::Slice> {
  static std::pair<std::size_t, std::size_t>
  getRange(const Syntax::Slice& value) noexcept;
};

template <>
struct LocationProvider<Syntax::Subscription> {
  static std::pair<std::size_t, std::size_t>
  getRange(const Syntax::Subscription& value) noexcept;
};

template <>
struct LocationProvider<Syntax::AttributeRef> {
  static std::pair<std::size_t, std::size_t>
  getRange(const Syntax::AttributeRef& value) noexcept;
};

template <>
struct LocationProvider<Syntax::Atom> {
  static std::pair<std::size_t, std::size_t>
  getRange(const Syntax::Atom& value) noexcept;
};

template <>
struct LocationProvider<Syntax::StarredItem> {
  static std::pair<std::size_t, std::size_t>
  getRange(const Syntax::StarredItem& value) noexcept;
};

template <>
struct LocationProvider<Syntax::Assignment> {
  static std::pair<std::size_t, std::size_t>
  getRange(const Syntax::Assignment& value) noexcept;
};

template <>
struct LocationProvider<Syntax::Argument> {
  static std::pair<std::size_t, std::size_t>
  getRange(const Syntax::Argument& value) noexcept;
};

template <>
struct LocationProvider<Syntax::Parameter> {
  static std::pair<std::size_t, std::size_t>
  getRange(const Syntax::Parameter& value) noexcept;
};

template <>
struct LocationProvider<Syntax::RaiseStmt> {
  static std::pair<std::size_t, std::size_t>
  getRange(const Syntax::RaiseStmt& value) noexcept;

  static std::size_t getPoint(const Syntax::RaiseStmt& value) noexcept {
    return pointLoc(value.raise);
  }
};

template <>
struct LocationProvider<Syntax::ReturnStmt> {
  static std::pair<std::size_t, std::size_t>
  getRange(const Syntax::ReturnStmt& value) noexcept;

  static std::size_t getPoint(const Syntax::ReturnStmt& value) noexcept {
    return pointLoc(value.returnKeyword);
  }
};

template <>
struct LocationProvider<Syntax::SingleTokenStmt> {
  static std::pair<std::size_t, std::size_t>
  getRange(const Syntax::SingleTokenStmt& value) noexcept;

  static std::size_t getPoint(const Syntax::SingleTokenStmt& value) noexcept {
    return pointLoc(value.token);
  }
};

template <>
struct LocationProvider<Syntax::AssignmentStmt> {
  static std::pair<std::size_t, std::size_t>
  getRange(const Syntax::AssignmentStmt& value) noexcept;
};

template <>
struct LocationProvider<Syntax::IfStmt> {
  static std::pair<std::size_t, std::size_t>
  getRange(const Syntax::IfStmt& value) noexcept;

  static std::size_t getPoint(const Syntax::IfStmt& value) noexcept {
    return pointLoc(value.ifKeyword);
  }
};

template <>
struct LocationProvider<Syntax::Suite> {
  static std::pair<std::size_t, std::size_t>
  getRange(const Syntax::Suite& value) noexcept;
};

template <>
struct LocationProvider<Syntax::FileInput> {
  static std::pair<std::size_t, std::size_t>
  getRange(const Syntax::FileInput& value) noexcept;
};

template <>
struct LocationProvider<Syntax::ExpressionStmt> {
  static std::pair<std::size_t, std::size_t>
  getRange(const Syntax::ExpressionStmt& value) noexcept;
};

template <>
struct LocationProvider<Syntax::AssertStmt> {
  static std::pair<std::size_t, std::size_t>
  getRange(const Syntax::AssertStmt& value) noexcept;

  static std::size_t getPoint(const Syntax::AssertStmt& value) noexcept {
    return pointLoc(value.assertKeyword);
  }
};

template <>
struct LocationProvider<Syntax::DelStmt> {
  static std::pair<std::size_t, std::size_t>
  getRange(const Syntax::DelStmt& value) noexcept;

  static std::size_t getPoint(const Syntax::DelStmt& value) noexcept {
    return pointLoc(value.del);
  }
};

template <>
struct LocationProvider<Syntax::ImportStmt> {
  static std::pair<std::size_t, std::size_t>
  getRange(const Syntax::ImportStmt& value) noexcept;

  static std::size_t getPoint(const Syntax::ImportStmt& value) noexcept;
};

template <>
struct LocationProvider<Syntax::FutureStmt> {
  static std::pair<std::size_t, std::size_t>
  getRange(const Syntax::FutureStmt& value) noexcept;

  static std::size_t getPoint(const Syntax::FutureStmt& value) noexcept;
};

template <>
struct LocationProvider<Syntax::GlobalOrNonLocalStmt> {
  static std::pair<std::size_t, std::size_t>
  getRange(const Syntax::GlobalOrNonLocalStmt& value) noexcept;

  static std::size_t
  getPoint(const Syntax::GlobalOrNonLocalStmt& value) noexcept {
    return pointLoc(value.token);
  }
};

template <>
struct LocationProvider<Syntax::WhileStmt> {
  static std::pair<std::size_t, std::size_t>
  getRange(const Syntax::WhileStmt& value) noexcept;

  static std::size_t getPoint(const Syntax::WhileStmt& value) noexcept {
    return pointLoc(value.whileKeyword);
  }
};

template <>
struct LocationProvider<Syntax::ForStmt> {
  static std::pair<std::size_t, std::size_t>
  getRange(const Syntax::ForStmt& value) noexcept;

  static std::size_t getPoint(const Syntax::ForStmt& value) noexcept {
    return pointLoc(value.forKeyword);
  }
};

template <>
struct LocationProvider<Syntax::TryStmt> {
  static std::pair<std::size_t, std::size_t>
  getRange(const Syntax::TryStmt& value) noexcept;

  static std::size_t getPoint(const Syntax::TryStmt& value) noexcept {
    return pointLoc(value.tryKeyword);
  }
};

template <>
struct LocationProvider<Syntax::WithStmt> {
  static std::pair<std::size_t, std::size_t>
  getRange(const Syntax::WithStmt& value) noexcept;

  static std::size_t getPoint(const Syntax::WithStmt& value) noexcept {
    return pointLoc(value.withKeyword);
  }
};

template <>
struct LocationProvider<Syntax::FuncDef> {
  static std::pair<std::size_t, std::size_t>
  getRange(const Syntax::FuncDef& value) noexcept;

  static std::size_t getPoint(const Syntax::FuncDef& value) noexcept {
    return pointLoc(value.funcName);
  }
};

template <>
struct LocationProvider<Syntax::Decorator> {
  static std::pair<std::size_t, std::size_t>
  getRange(const Syntax::Decorator& value) noexcept;
};

template <>
struct LocationProvider<Syntax::ClassDef> {
  static std::pair<std::size_t, std::size_t>
  getRange(const Syntax::ClassDef& value) noexcept;

  static std::size_t getPoint(const Syntax::ClassDef& value) noexcept {
    return pointLoc(value.className);
  }
};

template <>
struct LocationProvider<Syntax::CompFor> {
  static std::pair<std::size_t, std::size_t>
  getRange(const Syntax::CompFor& value) noexcept;
};

template <>
struct LocationProvider<Syntax::CompIf> {
  static std::pair<std::size_t, std::size_t>
  getRange(const Syntax::CompIf& value) noexcept;
};

template <>
struct LocationProvider<Syntax::Comprehension> {
  static std::pair<std::size_t, std::size_t>
  getRange(const Syntax::Comprehension& value) noexcept;
};

template <>
struct LocationProvider<Syntax::Intrinsic> {
  static std::pair<std::size_t, std::size_t>
  getRange(const Syntax::Intrinsic& value) noexcept {
    return rangeLoc(value.identifiers);
  }
};

} // namespace pylir::Diag
