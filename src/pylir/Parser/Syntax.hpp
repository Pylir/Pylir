// Copyright 2022 Markus Böck
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <pylir/Lexer/Token.hpp>

#include <memory>
#include <optional>
#include <variant>

namespace pylir::Syntax
{
struct Expression;

template <class T>
struct CommaList
{
    std::unique_ptr<T> firstExpr;
    std::vector<std::pair<BaseToken, std::unique_ptr<T>>> remainingExpr;
    std::optional<BaseToken> trailingComma;
};

using ExpressionList = CommaList<Expression>;

struct Enclosure;

struct Atom
{
    struct Literal
    {
        Token token;
    };

    std::variant<Literal, IdentifierToken, std::unique_ptr<Enclosure>> variant;
};

struct Primary;

struct AttributeRef
{
    std::unique_ptr<Primary> primary;
    BaseToken dot;
    IdentifierToken identifier;
};

struct Subscription
{
    std::unique_ptr<Primary> primary;
    BaseToken openSquareBracket;
    ExpressionList expressionList;
    BaseToken closeSquareBracket;
};

struct Slicing
{
    std::unique_ptr<Primary> primary;
    BaseToken openSquareBracket;
    struct ProperSlice
    {
        std::unique_ptr<Expression> optionalLowerBound;
        BaseToken firstColon;
        std::unique_ptr<Expression> optionalUpperBound;
        BaseToken secondColon;
        std::unique_ptr<Expression> optionalStride;
    };
    CommaList<std::variant<ProperSlice, Expression>> sliceList;
    BaseToken closeSquareBracket;
};

struct Comprehension;

struct AssignmentExpression;

struct ArgumentList
{
    struct PositionalItem
    {
        struct Star
        {
            BaseToken asterisk;
            std::unique_ptr<Expression> expression;
        };
        std::variant<std::unique_ptr<AssignmentExpression>, Star> variant;
    };

    struct PositionalArguments
    {
        PositionalItem firstItem;
        std::vector<std::pair<BaseToken, PositionalItem>> rest;
    };

    struct KeywordItem
    {
        IdentifierToken identifier;
        BaseToken assignmentOperator;
        std::unique_ptr<Expression> expression;
    };

    struct StarredAndKeywords
    {
        struct Expression
        {
            BaseToken asterisk;
            std::unique_ptr<Syntax::Expression> expression;
        };
        KeywordItem first;
        using Variant = std::variant<KeywordItem, Expression>;
        std::vector<std::pair<BaseToken, Variant>> rest;
    };

    struct KeywordArguments
    {
        struct Expression
        {
            BaseToken doubleAsterisk;
            std::unique_ptr<Syntax::Expression> expression;
        };
        Expression first;
        using Variant = std::variant<KeywordItem, Expression>;
        std::vector<std::pair<BaseToken, Variant>> rest;
    };

    std::optional<PositionalArguments> positionalArguments;
    std::optional<BaseToken> firstComma;
    std::optional<StarredAndKeywords> starredAndKeywords;
    std::optional<BaseToken> secondComma;
    std::optional<KeywordArguments> keywordArguments;
};

struct Call
{
    std::unique_ptr<Primary> primary;
    BaseToken openParentheses;
    std::variant<std::monostate, std::pair<ArgumentList, std::optional<BaseToken>>, std::unique_ptr<Comprehension>>
        variant;
    BaseToken closeParentheses;
};

struct Primary
{
    std::variant<Atom, AttributeRef, Subscription, Slicing, Call> variant;
};

struct AwaitExpr
{
    BaseToken awaitToken;
    Primary primary;
};

struct UExpr;

struct Power
{
    std::variant<AwaitExpr, Primary> variant;
    std::optional<std::pair<BaseToken, std::unique_ptr<UExpr>>> rightHand;
};

struct UExpr
{
    std::variant<Power, std::pair<Token, std::unique_ptr<UExpr>>> variant;
};

struct MExpr
{
    struct AtBin
    {
        std::unique_ptr<MExpr> lhs;
        BaseToken atToken;
        std::unique_ptr<MExpr> rhs;
    };

    struct BinOp
    {
        std::unique_ptr<MExpr> lhs;
        Token binToken;
        UExpr rhs;
    };

    std::variant<UExpr, std::unique_ptr<AtBin>, std::unique_ptr<BinOp>> variant;
};

struct AExpr
{
    struct BinOp
    {
        std::unique_ptr<AExpr> lhs;
        Token binToken;
        MExpr rhs;
    };

    std::variant<MExpr, std::unique_ptr<BinOp>> variant;
};

struct ShiftExpr
{
    struct BinOp
    {
        std::unique_ptr<ShiftExpr> lhs;
        Token binToken;
        AExpr rhs;
    };

    std::variant<AExpr, std::unique_ptr<BinOp>> variant;
};

struct AndExpr
{
    struct BinOp
    {
        std::unique_ptr<AndExpr> lhs;
        BaseToken bitAndToken;
        ShiftExpr rhs;
    };

    std::variant<ShiftExpr, std::unique_ptr<BinOp>> variant;
};

struct XorExpr
{
    struct BinOp
    {
        std::unique_ptr<XorExpr> lhs;
        BaseToken bitXorToken;
        AndExpr rhs;
    };

    std::variant<AndExpr, std::unique_ptr<BinOp>> variant;
};

struct OrExpr
{
    struct BinOp
    {
        std::unique_ptr<OrExpr> lhs;
        BaseToken bitOrToken;
        XorExpr rhs;
    };

    std::variant<XorExpr, std::unique_ptr<BinOp>> variant;
};

struct Comparison
{
    OrExpr left;
    struct Operator
    {
        Token firstToken;
        std::optional<Token> secondToken;
    };
    std::vector<std::pair<Operator, OrExpr>> rest;
};

struct NotTest
{
    std::variant<Comparison, std::pair<BaseToken, std::unique_ptr<NotTest>>> variant;
};

struct AndTest
{
    struct BinOp
    {
        std::unique_ptr<AndTest> lhs;
        BaseToken andToken;
        NotTest rhs;
    };

    std::variant<NotTest, std::unique_ptr<BinOp>> variant;
};

struct OrTest
{
    struct BinOp
    {
        std::unique_ptr<OrTest> lhs;
        BaseToken orToken;
        AndTest rhs;
    };

    std::variant<AndTest, std::unique_ptr<BinOp>> variant;
};

struct AssignmentExpression
{
    std::optional<std::pair<IdentifierToken, BaseToken>> identifierAndWalrus;
    std::unique_ptr<Expression> expression;
};

struct ConditionalExpression
{
    OrTest value;
    struct Suffix
    {
        BaseToken ifToken;
        std::unique_ptr<OrTest> test;
        BaseToken elseToken;
        std::unique_ptr<Expression> elseValue;
    };
    std::optional<Suffix> suffix;
};

struct LambdaExpression;

struct Expression
{
    std::variant<ConditionalExpression, std::unique_ptr<LambdaExpression>> variant;
};

struct ParameterList;

struct LambdaExpression
{
    BaseToken lambdaToken;
    std::unique_ptr<ParameterList> parameterList;
    BaseToken colonToken;
    Expression expression;
};

struct StarredItem
{
    std::variant<AssignmentExpression, std::pair<BaseToken, OrExpr>> variant;
};

using StarredList = CommaList<StarredItem>;

struct StarredExpression
{
    struct Items
    {
        std::vector<std::pair<StarredItem, BaseToken>> leading;
        std::unique_ptr<StarredItem> last;
    };
    std::variant<Expression, Items> variant;
};

struct Target;

using TargetList = CommaList<Target>;

struct CompIf;

struct CompFor
{
    std::optional<BaseToken> awaitToken;
    BaseToken forToken;
    TargetList targets;
    BaseToken inToken;
    OrTest orTest;
    std::variant<std::monostate, std::unique_ptr<CompFor>, std::unique_ptr<CompIf>> compIter;
};

struct CompIf
{
    BaseToken ifToken;
    OrTest orTest;
    std::variant<std::monostate, CompFor, std::unique_ptr<CompIf>> compIter;
};

struct Comprehension
{
    AssignmentExpression assignmentExpression;
    CompFor compFor;
};

struct YieldExpression
{
    BaseToken yieldToken;
    std::variant<std::monostate, ExpressionList, std::pair<BaseToken, Expression>> variant;
};

struct Enclosure
{
    struct ParenthForm
    {
        BaseToken openParenth;
        std::optional<StarredExpression> expression;
        BaseToken closeParenth;
    };

    struct ListDisplay
    {
        BaseToken openSquare;
        std::variant<std::monostate, StarredList, Comprehension> variant;
        BaseToken closeSquare;
    };

    struct SetDisplay
    {
        BaseToken openBrace;
        std::variant<StarredList, Comprehension> variant;
        BaseToken closeBrace;
    };

    struct DictDisplay
    {
        BaseToken openBrace;
        struct KeyDatum
        {
            struct Key
            {
                Expression first;
                BaseToken colon;
                Expression second;
            };
            struct Datum
            {
                BaseToken powerOf;
                OrExpr orExpr;
            };
            std::variant<Key, Datum> variant;
        };
        struct DictComprehension
        {
            Expression first;
            BaseToken colon;
            Expression second;
            CompFor compFor;
        };
        std::variant<std::monostate, CommaList<KeyDatum>, DictComprehension> variant;
        BaseToken closeBrace;
    };

    struct GeneratorExpression
    {
        BaseToken openParenth;
        Expression expression;
        CompFor compFor;
        BaseToken closeParenth;
    };

    struct YieldAtom
    {
        BaseToken openParenth;
        YieldExpression yieldExpression;
        BaseToken closeParenth;
    };

    std::variant<ParenthForm, ListDisplay, SetDisplay, DictDisplay, GeneratorExpression, YieldAtom> variant;
};

struct Target
{
    struct Parenth
    {
        BaseToken openParenth;
        std::optional<TargetList> targetList;
        BaseToken closeParenth;
    };

    struct Square
    {
        BaseToken openSquare;
        std::optional<TargetList> targetList;
        BaseToken closeSquare;
    };

    std::variant<IdentifierToken, Parenth, Square, AttributeRef, Subscription, Slicing,
                 std::pair<BaseToken, std::unique_ptr<Target>>>
        variant;
};


struct AssignmentStmt
{
    std::vector<std::pair<TargetList, BaseToken>> targets;
    std::variant<StarredExpression, YieldExpression> variant;
};

struct AugTarget
{
    std::variant<IdentifierToken, AttributeRef, Subscription, Slicing> variant;
};

struct AugmentedAssignmentStmt
{
    AugTarget augTarget;
    Token augOp;
    std::variant<ExpressionList, YieldExpression> variant;
};

struct AnnotatedAssignmentSmt
{
    AugTarget augTarget;
    BaseToken colon;
    Expression expression;
    std::optional<std::pair<BaseToken, std::variant<StarredExpression, YieldExpression>>> optionalAssignmentStmt;
};

struct AssertStmt
{
    BaseToken assertKeyword;
    Expression condition;
    std::optional<std::pair<BaseToken, Expression>> message;
};

struct PassStmt
{
    BaseToken pass;
};

struct DelStmt
{
    BaseToken del;
    TargetList targetList;
};

struct ReturnStmt
{
    BaseToken returnKeyword;
    std::optional<ExpressionList> expressions;
};

struct YieldStmt
{
    YieldExpression yieldExpression;
};

struct RaiseStmt
{
    BaseToken raise;
    std::optional<std::pair<Expression, std::optional<std::pair<BaseToken, Expression>>>> expressions;
};

struct BreakStmt
{
    BaseToken breakKeyword;
};

struct ContinueStmt
{
    BaseToken continueKeyword;
};

struct ImportStmt
{
    struct Module
    {
        std::vector<std::pair<IdentifierToken, BaseToken>> leading;
        IdentifierToken lastIdentifier;
    };

    struct RelativeModule
    {
        std::vector<BaseToken> dots;
        std::optional<Module> module;
    };

    struct ImportAsAs
    {
        BaseToken import;
        Module module;
        std::optional<std::pair<BaseToken, IdentifierToken>> name;
        struct Further
        {
            BaseToken comma;
            Module module;
            std::optional<std::pair<BaseToken, IdentifierToken>> name;
        };
        std::vector<Further> rest;
    };

    struct FromImportList
    {
        BaseToken from;
        RelativeModule relativeModule;
        BaseToken import;
        std::optional<BaseToken> openParenth;
        IdentifierToken identifier;
        std::optional<std::pair<BaseToken, IdentifierToken>> name;
        struct Further
        {
            BaseToken comma;
            IdentifierToken identifier;
            std::optional<std::pair<BaseToken, IdentifierToken>> name;
        };
        std::vector<Further> rest;
        std::optional<BaseToken> comma;
        std::optional<BaseToken> closeParenth;
    };

    struct FromImportAll
    {
        BaseToken from;
        RelativeModule relativeModule;
        BaseToken import;
        BaseToken star;
    };

    std::variant<ImportAsAs, FromImportList, FromImportAll> variant;
};

struct FutureStmt
{
    BaseToken from;
    BaseToken future;
    BaseToken import;
    std::optional<BaseToken> openParenth;
    IdentifierToken identifier;
    std::optional<std::pair<BaseToken, IdentifierToken>> name;
    std::vector<Syntax::ImportStmt::FromImportList::Further> rest;
    std::optional<BaseToken> comma;
    std::optional<BaseToken> closeParenth;
};

struct GlobalStmt
{
    BaseToken global;
    IdentifierToken identifier;
    std::vector<std::pair<BaseToken, IdentifierToken>> rest;
};

struct NonLocalStmt
{
    BaseToken nonLocal;
    IdentifierToken identifier;
    std::vector<std::pair<BaseToken, IdentifierToken>> rest;
};

struct SimpleStmt
{
    std::variant<StarredExpression, AssertStmt, AssignmentStmt, AugmentedAssignmentStmt, AnnotatedAssignmentSmt,
                 PassStmt, DelStmt, ReturnStmt, YieldStmt, RaiseStmt, BreakStmt, ContinueStmt, ImportStmt, FutureStmt,
                 GlobalStmt, NonLocalStmt>
        variant;
};

using StmtList = CommaList<SimpleStmt>;

struct Suite;

struct IfStmt
{
    BaseToken ifKeyword;
    AssignmentExpression condition;
    BaseToken colon;
    std::unique_ptr<Suite> suite;
    struct Elif
    {
        BaseToken elif;
        AssignmentExpression condition;
        BaseToken colon;
        std::unique_ptr<Suite> suite;
    };
    std::vector<Elif> elifs;
    struct Else
    {
        BaseToken elseKeyowrd;
        BaseToken colon;
        std::unique_ptr<Suite> suite;
    };
    std::optional<Else> elseSection;
};

struct WhileStmt
{
    BaseToken whileKeyword;
    AssignmentExpression condition;
    BaseToken colon;
    std::unique_ptr<Suite> suite;
    std::optional<IfStmt::Else> elseSection;
};

struct ForStmt
{
    BaseToken forKeyword;
    TargetList targetList;
    BaseToken inKeyword;
    ExpressionList expressionList;
    BaseToken colon;
    std::unique_ptr<Suite> suite;
    std::optional<IfStmt::Else> elseSection;
};

struct TryStmt
{
    BaseToken tryKeyword;
    BaseToken colon;
    std::unique_ptr<Suite> suite;
    struct Except
    {
        BaseToken exceptKeyword;
        std::optional<std::pair<Expression, std::optional<std::pair<BaseToken, IdentifierToken>>>> expression;
        BaseToken colon;
        std::unique_ptr<Suite> suite;
    };
    std::vector<Except> excepts;
    std::optional<IfStmt::Else> elseSection;
    struct Finally
    {
        BaseToken finally;
        BaseToken colon;
        std::unique_ptr<Suite> suite;
    };
    std::optional<Finally> finally;
};

struct WithStmt
{
    BaseToken withKeyword;
    struct WithItem
    {
        Expression expression;
        std::optional<std::pair<BaseToken, Target>> target;
    };
    WithItem first;
    std::vector<std::pair<BaseToken, WithItem>> rest;
    BaseToken colon;
    std::unique_ptr<Suite> suite;
};

struct ParameterList
{
    struct Parameter
    {
        IdentifierToken identifier;
        std::optional<std::pair<BaseToken, Expression>> type;
    };

    struct DefParameter
    {
        Parameter parameter;
        std::optional<std::pair<BaseToken, Expression>> defaultArg;
    };

    struct StarArgs
    {
        struct DoubleStar
        {
            BaseToken doubleStar;
            Parameter parameter;
            std::optional<BaseToken> comma;
        };

        struct Star
        {
            BaseToken star;
            std::optional<Parameter> parameter;
            std::vector<std::pair<BaseToken, DefParameter>> defParameters;
            struct Further
            {
                BaseToken comma;
                std::optional<DoubleStar> doubleStar;
            };
            std::optional<Further> further;
        };
        std::variant<Star, DoubleStar> variant;
    };

    struct NoPosOnly
    {
        struct DefParams
        {
            DefParameter first;
            std::vector<std::pair<BaseToken, DefParameter>> rest;
            std::optional<std::pair<BaseToken, std::optional<StarArgs>>> suffix;
        };
        std::variant<DefParams, StarArgs> variant;
    };

    struct PosOnly
    {
        DefParameter first;
        std::vector<std::pair<BaseToken, DefParameter>> rest;
        BaseToken comma;
        BaseToken slash;
        std::optional<std::pair<BaseToken, std::optional<NoPosOnly>>> suffix;
    };
    std::variant<PosOnly, NoPosOnly> variant;
};

struct Decorator
{
    BaseToken atSign;
    AssignmentExpression assignmentExpression;
    BaseToken newline;
};

struct FuncDef
{
    std::vector<Decorator> decorators;
    std::optional<BaseToken> async;
    BaseToken def;
    IdentifierToken funcName;
    BaseToken openParenth;
    std::optional<ParameterList> parameterList;
    BaseToken closeParenth;
    std::optional<std::pair<BaseToken, Expression>> suffix;
    BaseToken colon;
    std::unique_ptr<Suite> suite;

    IdentifierSet localVariables;
    IdentifierSet nonLocalVariables;
    IdentifierSet closures;
    IdentifierSet unknown; // only temporarily used
};

struct ClassDef
{
    std::vector<Decorator> decorators;
    BaseToken classKeyword;
    IdentifierToken className;
    struct Inheritance
    {
        BaseToken openParenth;
        std::optional<ArgumentList> argumentList;
        BaseToken closeParenth;
    };
    std::optional<Inheritance> inheritance;
    BaseToken colon;
    std::unique_ptr<Suite> suite;

    IdentifierSet localVariables;
    IdentifierSet nonLocalVariables;
    IdentifierSet unknown; // only temporarily used
};

struct AsyncForStmt
{
    BaseToken async;
    ForStmt forStmt;
};

struct AsyncWithStmt
{
    BaseToken async;
    WithStmt withStmt;
};

struct CompoundStmt
{
    std::variant<IfStmt, WhileStmt, ForStmt, TryStmt, WithStmt, FuncDef, ClassDef, AsyncForStmt, AsyncWithStmt> variant;
};

struct Statement
{
    struct SingleLine
    {
        StmtList stmtList;
        BaseToken newline;
    };
    std::variant<SingleLine, CompoundStmt> variant;
};

struct Suite
{
    struct SingleLine
    {
        StmtList stmtList;
        BaseToken newline;
    };

    struct MultiLine
    {
        BaseToken newline;
        BaseToken indent;
        std::vector<Statement> statements;
        BaseToken dedent;
    };
    std::variant<SingleLine, MultiLine> variant;
};

struct FileInput
{
    std::vector<std::variant<BaseToken, Statement>> input;
    IdentifierSet globals;
};

} // namespace pylir::Syntax

namespace pylir::Diag
{
template <class T, class>
struct LocationProvider;

template <>
struct LocationProvider<Syntax::Enclosure, void>
{
    static std::pair<std::size_t, std::size_t> getRange(const Syntax::Enclosure& value) noexcept;
};

template <>
struct LocationProvider<Syntax::LambdaExpression, void>
{
    static std::pair<std::size_t, std::size_t> getRange(const Syntax::LambdaExpression& value) noexcept;
};

template <>
struct LocationProvider<Syntax::Expression, void>
{
    static std::pair<std::size_t, std::size_t> getRange(const Syntax::Expression& value) noexcept;
};

template <>
struct LocationProvider<Syntax::ConditionalExpression, void>
{
    static std::pair<std::size_t, std::size_t> getRange(const Syntax::ConditionalExpression& value) noexcept;
};

template <>
struct LocationProvider<Syntax::OrTest, void>
{
    static std::pair<std::size_t, std::size_t> getRange(const Syntax::OrTest& value) noexcept;
};

template <>
struct LocationProvider<Syntax::AndTest, void>
{
    static std::pair<std::size_t, std::size_t> getRange(const Syntax::AndTest& value) noexcept;
};

template <>
struct LocationProvider<Syntax::NotTest, void>
{
    static std::pair<std::size_t, std::size_t> getRange(const Syntax::NotTest& value) noexcept;
};

template <>
struct LocationProvider<Syntax::Comparison, void>
{
    static std::pair<std::size_t, std::size_t> getRange(const Syntax::Comparison& value) noexcept;
};

template <>
struct LocationProvider<Syntax::OrExpr, void>
{
    static std::pair<std::size_t, std::size_t> getRange(const Syntax::OrExpr& value) noexcept;
};

template <>
struct LocationProvider<Syntax::XorExpr, void>
{
    static std::pair<std::size_t, std::size_t> getRange(const Syntax::XorExpr& value) noexcept;
};

template <>
struct LocationProvider<Syntax::AndExpr, void>
{
    static std::pair<std::size_t, std::size_t> getRange(const Syntax::AndExpr& value) noexcept;
};

template <>
struct LocationProvider<Syntax::ShiftExpr, void>
{
    static std::pair<std::size_t, std::size_t> getRange(const Syntax::ShiftExpr& value) noexcept;
};

template <>
struct LocationProvider<Syntax::AExpr, void>
{
    static std::pair<std::size_t, std::size_t> getRange(const Syntax::AExpr& value) noexcept;
};

template <>
struct LocationProvider<Syntax::MExpr, void>
{
    static std::pair<std::size_t, std::size_t> getRange(const Syntax::MExpr& binOp) noexcept;
};

template <>
struct LocationProvider<Syntax::UExpr, void>
{
    static std::pair<std::size_t, std::size_t> getRange(const Syntax::UExpr& power) noexcept;
};

template <>
struct LocationProvider<Syntax::Power, void>
{
    static std::pair<std::size_t, std::size_t> getRange(const Syntax::Power& value) noexcept;
};

template <>
struct LocationProvider<Syntax::AwaitExpr, void>
{
    static std::pair<std::size_t, std::size_t> getRange(const Syntax::AwaitExpr& value) noexcept;
};

template <>
struct LocationProvider<Syntax::Primary, void>
{
    static std::pair<std::size_t, std::size_t> getRange(const Syntax::Primary& value) noexcept;
};

template <>
struct LocationProvider<Syntax::Call, void>
{
    static std::pair<std::size_t, std::size_t> getRange(const Syntax::Call& value) noexcept;
};

template <>
struct LocationProvider<Syntax::Slicing, void>
{
    static std::pair<std::size_t, std::size_t> getRange(const Syntax::Slicing& value) noexcept;
};

template <>
struct LocationProvider<Syntax::Subscription, void>
{
    static std::pair<std::size_t, std::size_t> getRange(const Syntax::Subscription& value) noexcept;
};

template <>
struct LocationProvider<Syntax::AttributeRef, void>
{
    static std::pair<std::size_t, std::size_t> getRange(const Syntax::AttributeRef& value) noexcept;
};

template <>
struct LocationProvider<Syntax::Atom, void>
{
    static std::pair<std::size_t, std::size_t> getRange(const Syntax::Atom& value) noexcept;
};

template <>
struct LocationProvider<Syntax::StarredItem, void>
{
    static std::pair<std::size_t, std::size_t> getRange(const Syntax::StarredItem& starredItem) noexcept;
};

template <>
struct LocationProvider<Syntax::AssignmentExpression, void>
{
    static std::pair<std::size_t, std::size_t>
        getRange(const Syntax::AssignmentExpression& assignmentExpression) noexcept;
};

template <>
struct LocationProvider<Syntax::StarredExpression, void>
{
    static std::pair<std::size_t, std::size_t> getRange(const Syntax::StarredExpression& starredItem) noexcept;
};

template <>
struct LocationProvider<Syntax::ExpressionList, void>
{
    static std::pair<std::size_t, std::size_t> getRange(const Syntax::ExpressionList& expressionList) noexcept;
};

template <>
struct LocationProvider<Syntax::StarredList, void>
{
    static std::pair<std::size_t, std::size_t> getRange(const Syntax::StarredList& starredList) noexcept;
};

template <>
struct LocationProvider<Syntax::ArgumentList, void>
{
    static std::pair<std::size_t, std::size_t> getRange(const Syntax::ArgumentList& argumentList) noexcept;
};

template <>
struct LocationProvider<Syntax::ParameterList::Parameter, void>
{
    static std::pair<std::size_t, std::size_t> getRange(const Syntax::ParameterList::Parameter& parameter) noexcept;
};

template <>
struct LocationProvider<Syntax::ParameterList::DefParameter, void>
{
    static std::pair<std::size_t, std::size_t>
        getRange(const Syntax::ParameterList::DefParameter& defParameter) noexcept;
};

} // namespace pylir::Diag
