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
          struct Comparison, struct Intrinsic> {
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

struct Intrinsic : Expression::Base<Intrinsic> {
  /// Name of the intrinsic. These are all identifier joined with dots.
  /// Includes the 'pylir.intr' prefix.
  std::string name;
  /// All identifier tokens making up the name. Main use-case is for the
  /// purpose of the location in the source code.
  llvm::SmallVector<IdentifierToken> identifiers;
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
  /// module ::= { identifier "." } identifier
  struct Module {
    std::vector<IdentifierToken> identifiers;
  };

  /// relative_module ::=  { "." } module |  "." { "." }
  struct RelativeModule {
    std::vector<BaseToken> dots;
    std::optional<Module> module;
  };

  /// "import" module ["as" identifier] { "," module ["as" identifier] }
  struct ImportAs {
    BaseToken import;
    std::vector<std::pair<Module, std::optional<IdentifierToken>>> modules;
  };

  /// "from" relative_module "import" identifier ["as" identifier]
  ///       { "," identifier ["as" identifier] }
  /// | "from" relative_module "import"
  ///   "(" identifier ["as" identifier]
  ///     { "," identifier ["as" identifier] }
  ///   [","] ")"
  struct FromImport {
    BaseToken from;
    RelativeModule relativeModule;
    BaseToken import;
    std::vector<std::pair<IdentifierToken, std::optional<IdentifierToken>>>
        imports;
  };

  /// "from" relative_module "import" "*"
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

} // namespace pylir::Syntax

namespace pylir::Diag {

template <>
struct LocationProvider<Syntax::TupleConstruct> {
  static Location getRange(const Syntax::TupleConstruct& value) noexcept {
    return {value.maybeOpenBracket, value.items, value.maybeCloseBracket};
  }
};

template <>
struct LocationProvider<Syntax::Lambda> {
  static Location getRange(const Syntax::Lambda& value) noexcept {
    return {value.lambdaKeyword, value.expression};
  }
};

template <class T>
struct LocationProvider<T, std::enable_if_t<IsAbstractVariantConcrete<T>{}>> {
  static Location getRange(const T& value) noexcept {
    return value.match([&](auto&& sub) { return rangeLoc(sub); });
  }

  static std::optional<std::size_t> getPoint(const T& value) noexcept {
    return value.match([&](auto&& sub) { return pointLoc(sub); });
  }
};

template <>
struct LocationProvider<Syntax::Conditional> {
  static Location getRange(const Syntax::Conditional& value) noexcept {
    return {value.trueValue, value.elseValue};
  }
};

template <>
struct LocationProvider<Syntax::BinOp> {
  static Location getRange(const Syntax::BinOp& value) noexcept {
    return {value.lhs, value.rhs};
  }
};

template <>
struct LocationProvider<Syntax::Comparison> {
  static Location getRange(const Syntax::Comparison& value) noexcept {
    return {value.first, value.rest.back().second};
  }
};

template <>
struct LocationProvider<Syntax::UnaryOp> {
  static Location getRange(const Syntax::UnaryOp& value) noexcept {
    return {value.operation, value.expression};
  }
};

template <>
struct LocationProvider<Syntax::Call> {
  static Location getRange(const Syntax::Call& value) noexcept {
    return {value.expression, value.closeParenth};
  }
};

template <>
struct LocationProvider<Syntax::DictDisplay> {
  static Location getRange(const Syntax::DictDisplay& value) noexcept {
    return {value.openBrace, value.closeBrace};
  }
};

template <>
struct LocationProvider<Syntax::SetDisplay> {
  static Location getRange(const Syntax::SetDisplay& value) noexcept {
    return {value.openBrace, value.closeBrace};
  }
};

template <>
struct LocationProvider<Syntax::ListDisplay> {
  static Location getRange(const Syntax::ListDisplay& value) noexcept {
    return {value.openSquare, value.closeSquare};
  }
};

template <>
struct LocationProvider<Syntax::Yield> {
  static Location getRange(const Syntax::Yield& value) noexcept {
    return {value.yieldToken, value.maybeExpression};
  }
};

template <>
struct LocationProvider<Syntax::Generator> {
  static Location getRange(const Syntax::Generator& value) noexcept {
    return {value.openParenth, value.closeParenth};
  }
};

template <>
struct LocationProvider<Syntax::Slice> {
  static Location getRange(const Syntax::Slice& value) noexcept {
    return {value.maybeLowerBound, value.firstColon, value.maybeUpperBound,
            value.maybeStride};
  }
};

template <>
struct LocationProvider<Syntax::Subscription> {
  static Location getRange(const Syntax::Subscription& value) noexcept {
    return {value.object, value.closeSquareBracket};
  }
};

template <>
struct LocationProvider<Syntax::AttributeRef> {
  static Location getRange(const Syntax::AttributeRef& value) noexcept {
    return {value.object, value.identifier};
  }
};

template <>
struct LocationProvider<Syntax::Atom> {
  static Location getRange(const Syntax::Atom& value) noexcept {
    return rangeLoc(value.token);
  }
};

template <>
struct LocationProvider<Syntax::StarredItem> {
  static Location getRange(const Syntax::StarredItem& value) noexcept {
    return {value.maybeStar, value.expression};
  }
};

template <>
struct LocationProvider<Syntax::Assignment> {
  static Location getRange(const Syntax::Assignment& value) noexcept {
    return {value.variable, value.expression};
  }
};

template <>
struct LocationProvider<Syntax::Argument> {
  static Location getRange(const Syntax::Argument& value) noexcept {
    return {value.maybeName, value.maybeExpansionsOrEqual, value.expression};
  }
};

template <>
struct LocationProvider<Syntax::Parameter> {
  static Location getRange(const Syntax::Parameter& value) noexcept {
    return {value.maybeStars, value.name, value.maybeType, value.maybeDefault};
  }
};

template <>
struct LocationProvider<Syntax::RaiseStmt> {
  static Location getRange(const Syntax::RaiseStmt& value) noexcept {
    return {value.raise, value.maybeException, value.maybeCause};
  }
};

template <>
struct LocationProvider<Syntax::ReturnStmt> {
  static Location getRange(const Syntax::ReturnStmt& value) noexcept {
    return {value.returnKeyword, value.maybeExpression};
  }
};

template <>
struct LocationProvider<Syntax::SingleTokenStmt> {
  static Location getRange(const Syntax::SingleTokenStmt& value) noexcept {
    return rangeLoc(value.token);
  }
};

template <>
struct LocationProvider<Syntax::AssignmentStmt> {
  static Location getRange(const Syntax::AssignmentStmt& value) noexcept {
    return {value.targets, value.maybeAnnotation, value.maybeExpression};
  }
};

template <>
struct LocationProvider<Syntax::IfStmt> {
  static Location getRange(const Syntax::IfStmt& value) noexcept {
    return {value.ifKeyword, value.suite, value.elifs, value.elseSection};
  }

  static std::optional<std::size_t>
  getPoint(const Syntax::IfStmt& value) noexcept {
    return pointLoc(value.ifKeyword);
  }
};

template <>
struct LocationProvider<Syntax::IfStmt::Elif> {
  static Location getRange(const Syntax::IfStmt::Elif& value) noexcept {
    return {value.elif, value.suite};
  }
};

template <>
struct LocationProvider<Syntax::IfStmt::Else> {
  static Location getRange(const Syntax::IfStmt::Else& value) noexcept {
    return {value.elseKeyowrd, value.suite};
  }
};

template <>
struct LocationProvider<Syntax::Suite> {
  static Location getRange(const Syntax::Suite& value) noexcept {
    return rangeLoc(value.statements);
  }
};

template <>
struct LocationProvider<Syntax::FileInput> {
  static Location getRange(const Syntax::FileInput& value) noexcept {
    return rangeLoc(value.input);
  }
};

template <>
struct LocationProvider<Syntax::ExpressionStmt> {
  static Location getRange(const Syntax::ExpressionStmt& value) noexcept {
    return rangeLoc(value.expression);
  }
};

template <>
struct LocationProvider<Syntax::AssertStmt> {
  static Location getRange(const Syntax::AssertStmt& value) noexcept {
    return {value.assertKeyword, value.condition, value.maybeMessage};
  }

  static std::optional<std::size_t>
  getPoint(const Syntax::AssertStmt& value) noexcept {
    return pointLoc(value.assertKeyword);
  }
};

template <>
struct LocationProvider<Syntax::DelStmt> {
  static Location getRange(const Syntax::DelStmt& value) noexcept {
    return {value.del, value.targetList};
  }

  static std::optional<std::size_t>
  getPoint(const Syntax::DelStmt& value) noexcept {
    return pointLoc(value.del);
  }
};

template <>
struct LocationProvider<Syntax::ImportStmt> {
  static Location getRange(const Syntax::ImportStmt& value) noexcept {
    return match(
        value.variant,
        [](const Syntax::ImportStmt::ImportAs& importAs) -> Location {
          return {importAs.import, importAs.modules};
        },
        [](const Syntax::ImportStmt::ImportAll& importAll) -> Location {
          return {importAll.from, importAll.star};
        },
        [](const Syntax::ImportStmt::FromImport& fromImport) -> Location {
          return {fromImport.from, fromImport.imports};
        });
  }

  static std::optional<std::size_t>
  getPoint(const Syntax::ImportStmt& value) noexcept {
    return pylir::match(value.variant, [](const auto& value) {
      return pointLoc(value.import);
    });
  }
};

template <>
struct LocationProvider<Syntax::ImportStmt::Module> {
  static Location getRange(const Syntax::ImportStmt::Module& value) noexcept {
    return rangeLoc(value.identifiers);
  }
};

template <>
struct LocationProvider<Syntax::FutureStmt> {
  static Location getRange(const Syntax::FutureStmt& value) noexcept {
    return {value.from, value.imports};
  }

  static std::optional<std::size_t>
  getPoint(const Syntax::FutureStmt& value) noexcept {
    return pointLoc(value.future);
  }
};

template <>
struct LocationProvider<Syntax::GlobalOrNonLocalStmt> {
  static Location getRange(const Syntax::GlobalOrNonLocalStmt& value) noexcept {
    return {value.token, value.identifiers};
  }

  static std::optional<std::size_t>
  getPoint(const Syntax::GlobalOrNonLocalStmt& value) noexcept {
    return pointLoc(value.token);
  }
};

template <>
struct LocationProvider<Syntax::WhileStmt> {
  static Location getRange(const Syntax::WhileStmt& value) noexcept {
    return {value.whileKeyword, value.suite};
  }

  static std::optional<std::size_t>
  getPoint(const Syntax::WhileStmt& value) noexcept {
    return pointLoc(value.whileKeyword);
  }
};

template <>
struct LocationProvider<Syntax::ForStmt> {
  static Location getRange(const Syntax::ForStmt& value) noexcept {
    return {value.maybeAsyncKeyword, value.forKeyword, value.suite,
            value.elseSection};
  }

  static std::optional<std::size_t>
  getPoint(const Syntax::ForStmt& value) noexcept {
    return pointLoc(value.forKeyword);
  }
};

template <>
struct LocationProvider<Syntax::TryStmt> {
  static Location getRange(const Syntax::TryStmt& value) noexcept {
    return {value.tryKeyword, value.excepts, value.maybeExceptAll,
            value.elseSection, value.finally};
  }

  static std::optional<std::size_t>
  getPoint(const Syntax::TryStmt& value) noexcept {
    return pointLoc(value.tryKeyword);
  }
};

template <>
struct LocationProvider<Syntax::TryStmt::ExceptArgs> {
  static Location getRange(const Syntax::TryStmt::ExceptArgs& value) noexcept {
    return {value.exceptKeyword, value.suite};
  }
};

template <>
struct LocationProvider<Syntax::TryStmt::ExceptAll> {
  static Location getRange(const Syntax::TryStmt::ExceptAll& value) noexcept {
    return {value.exceptKeyword, value.suite};
  }
};

template <>
struct LocationProvider<Syntax::TryStmt::Finally> {
  static Location getRange(const Syntax::TryStmt::Finally& value) noexcept {
    return {value.finally, value.suite};
  }
};

template <>
struct LocationProvider<Syntax::WithStmt> {
  static Location getRange(const Syntax::WithStmt& value) noexcept {
    return {value.maybeAsyncKeyword, value.withKeyword, value.suite};
  }

  static std::optional<std::size_t>
  getPoint(const Syntax::WithStmt& value) noexcept {
    return pointLoc(value.withKeyword);
  }
};

template <>
struct LocationProvider<Syntax::FuncDef> {
  static Location getRange(const Syntax::FuncDef& value) noexcept {
    return {value.decorators, value.maybeAsyncKeyword, value.def, value.suite};
  }

  static std::optional<std::size_t>
  getPoint(const Syntax::FuncDef& value) noexcept {
    return pointLoc(value.funcName);
  }
};

template <>
struct LocationProvider<Syntax::Decorator> {
  static Location getRange(const Syntax::Decorator& value) noexcept {
    return {value.atSign, value.expression};
  }
};

template <>
struct LocationProvider<Syntax::ClassDef> {
  static Location getRange(const Syntax::ClassDef& value) noexcept {
    return {value.decorators, value.classKeyword, value.suite};
  }

  static std::optional<std::size_t>
  getPoint(const Syntax::ClassDef& value) noexcept {
    return pointLoc(value.className);
  }
};

template <>
struct LocationProvider<Syntax::CompFor> {
  static Location getRange(const Syntax::CompFor& value) noexcept {
    return {value.awaitToken, value.forToken, value.test, value.compIter};
  }
};

template <>
struct LocationProvider<Syntax::CompIf> {
  static Location getRange(const Syntax::CompIf& value) noexcept {
    return {value.ifToken, value.test, value.compIter};
  }
};

template <>
struct LocationProvider<Syntax::Comprehension> {
  static Location getRange(const Syntax::Comprehension& value) noexcept {
    return {value.expression, value.compFor};
  }
};

template <>
struct LocationProvider<Syntax::Intrinsic> {
  static Location getRange(const Syntax::Intrinsic& value) noexcept {
    return rangeLoc(value.identifiers);
  }
};

} // namespace pylir::Diag
