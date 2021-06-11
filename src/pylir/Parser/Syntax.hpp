
#pragma once

#include <pylir/Lexer/Token.hpp>

#include <variant>

namespace pylir::Syntax
{
struct Expression;

template <class T>
struct CommaList
{
    std::unique_ptr<T> firstExpr;
    std::vector<std::pair<Token, std::unique_ptr<T>>> remainingExpr;
    std::optional<Token> trailingComma;
};

using ExpressionList = CommaList<Expression>;

struct Enclosure;

struct Atom
{
    struct Literal
    {
        Token token;
    };

    struct Identifier
    {
        Token token;
    };

    std::variant<Literal, Identifier, std::unique_ptr<Enclosure>> variant;
};

struct Primary;

struct AttributeRef
{
    std::unique_ptr<Primary> primary;
    Token dot;
    Token identifier;
};

struct Subscription
{
    std::unique_ptr<Primary> primary;
    Token openSquareBracket;
    ExpressionList expressionList;
    Token closeSquareBracket;
};

struct Slicing
{
    std::unique_ptr<Primary> primary;
    Token openSquareBracket;
    struct ProperSlice
    {
        std::unique_ptr<Expression> optionalLowerBound;
        Token firstColon;
        std::unique_ptr<Expression> optionalUpperBound;
        Token secondColon;
        std::unique_ptr<Expression> optionalStride;
    };
    CommaList<std::variant<ProperSlice, Expression>> sliceList;
    Token closeSquareBracket;
};

struct Comprehension;

struct AssignmentExpression;

struct Call
{
    struct PositionalItem
    {
        struct Star
        {
            Token asterisk;
            std::unique_ptr<Expression> expression;
        };
        std::variant<std::unique_ptr<AssignmentExpression>, Star> variant;
    };

    struct PositionalArguments
    {
        PositionalItem firstItem;
        std::vector<std::pair<Token, PositionalItem>> rest;
    };

    struct KeywordItem
    {
        Token identifier;
        Token assignmentOperator;
        std::unique_ptr<Expression> expression;
    };

    struct StarredAndKeywords
    {
        struct Expression
        {
            Token asterisk;
            std::unique_ptr<Syntax::Expression> expression;
        };
        KeywordItem first;
        using Variant = std::variant<KeywordItem, Expression>;
        std::vector<std::pair<Token, Variant>> rest;
    };

    struct KeywordArguments
    {
        struct Expression
        {
            Token doubleAsterisk;
            std::unique_ptr<Syntax::Expression> expression;
        };
        Expression first;
        using Variant = std::variant<KeywordItem, Expression>;
        std::vector<std::pair<Token, Variant>> rest;
    };

    struct ArgumentList
    {
        std::optional<PositionalArguments> positionalArguments;
        std::optional<Token> firstComma;
        std::optional<StarredAndKeywords> starredAndKeywords;
        std::optional<Token> secondComma;
        std::optional<KeywordArguments> keywordArguments;
    };

    std::unique_ptr<Primary> primary;
    Token openParentheses;
    std::variant<std::monostate, std::pair<ArgumentList, std::optional<Token>>, std::unique_ptr<Comprehension>> variant;
    Token closeParentheses;
};

struct Primary
{
    std::variant<Atom, AttributeRef, Subscription, Slicing, Call> variant;
};

struct AwaitExpr
{
    Token awaitToken;
    Primary primary;
};

struct UExpr;

struct Power
{
    std::variant<AwaitExpr, Primary> variant;
    std::optional<std::pair<Token, std::unique_ptr<UExpr>>> rightHand;
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
        Token atToken;
        std::unique_ptr<MExpr> rhs;
    };

    struct BinOp
    {
        std::unique_ptr<MExpr> lhs;
        Token binToken;
        UExpr rhs;
    };

    std::variant<UExpr, AtBin, BinOp> variant;
};

struct AExpr
{
    struct BinOp
    {
        std::unique_ptr<AExpr> lhs;
        Token binToken;
        MExpr rhs;
    };

    std::variant<MExpr, BinOp, BinOp> variant;
};

struct ShiftExpr
{
    struct BinOp
    {
        std::unique_ptr<ShiftExpr> lhs;
        Token binToken;
        AExpr rhs;
    };

    std::variant<AExpr, BinOp> variant;
};

struct AndExpr
{
    struct BinOp
    {
        std::unique_ptr<AndExpr> lhs;
        Token bitAndToken;
        ShiftExpr rhs;
    };

    std::variant<ShiftExpr, BinOp> variant;
};

struct XorExpr
{
    struct BinOp
    {
        std::unique_ptr<XorExpr> lhs;
        Token bitXorToken;
        AndExpr rhs;
    };

    std::variant<AndExpr, BinOp> variant;
};

struct OrExpr
{
    struct BinOp
    {
        std::unique_ptr<OrExpr> lhs;
        Token bitOrToken;
        XorExpr rhs;
    };

    std::variant<XorExpr, BinOp> variant;
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
    std::variant<Comparison, std::pair<Token, std::unique_ptr<NotTest>>> variant;
};

struct AndTest
{
    struct BinOp
    {
        std::unique_ptr<AndExpr> lhs;
        Token andToken;
        NotTest rhs;
    };

    std::variant<NotTest, BinOp> variant;
};

struct OrTest
{
    struct BinOp
    {
        std::unique_ptr<OrExpr> lhs;
        Token orToken;
        AndTest rhs;
    };

    std::variant<AndTest, BinOp> variant;
};

struct AssignmentExpression
{
    std::optional<std::pair<Token, Token>> identifierAndWalrus;
    std::unique_ptr<Expression> expression;
};

inline bool firstInAssignmentExpression(TokenType tokenType);

struct ConditionalExpression
{
    OrTest value;
    struct Suffix
    {
        Token ifToken;
        OrTest test;
        Token elseToken;
        std::unique_ptr<Expression> elseValue;
    };
    std::optional<Suffix> suffix;
};

struct LambdaExpression;

struct Expression
{
    std::variant<ConditionalExpression, std::unique_ptr<LambdaExpression>> variant;
};

inline bool firstInExpression(TokenType tokenType)
{
    switch (tokenType)
    {
        case TokenType::LambdaKeyword:
        case TokenType::Minus:
        case TokenType::Plus:
        case TokenType::BitNegate:
        case TokenType::AwaitKeyword:
        case TokenType::StringLiteral:
        case TokenType::ByteLiteral:
        case TokenType::IntegerLiteral:
        case TokenType::FloatingPointLiteral:
        case TokenType::ComplexLiteral:
        case TokenType::Identifier:
        case TokenType::OpenParentheses:
        case TokenType::OpenSquareBracket:
        case TokenType::OpenBrace: return true;
        default: break;
    }
    return false;
}

struct ParameterList;

struct LambdaExpression
{
    Token lambdaToken;
    std::unique_ptr<ParameterList> parameterList;
    Token colonToken;
    Expression expression;
};

struct StarredItem
{
    std::variant<AssignmentExpression, std::pair<Token, OrExpr>> variant;
};

using StarredList = CommaList<StarredItem>;

struct StarredExpression
{
    struct Items
    {
        std::vector<std::pair<StarredItem, Token>> leading;
        std::optional<StarredItem> last;
    };
    std::variant<Expression, Items> variant;
};

struct Target;

using TargetList = CommaList<Target>;

struct CompIf;

struct CompFor
{
    std::optional<Token> awaitToken;
    Token forToken;
    TargetList targets;
    Token inToken;
    OrTest orTest;
    std::variant<std::monostate, std::unique_ptr<CompFor>, std::unique_ptr<CompIf>> compIter;
};

inline bool firstInCompFor(TokenType tokenType)
{
    switch (tokenType)
    {
        case TokenType::ForKeyword:
        case TokenType::AsyncKeyword: return true;
        default: break;
    }
    return false;
}

struct CompIf
{
    Token ifToken;
    OrTest orTest;
    std::variant<std::monostate, CompFor, std::unique_ptr<CompIf>> compIter;
};

struct Comprehension
{
    AssignmentExpression assignmentExpression;
    CompFor compFor;
};

inline bool firstInComprehension(TokenType tokenType)
{
    return firstInAssignmentExpression(tokenType);
}

struct Enclosure
{
    struct ParenthForm
    {
        Token openParenth;
        std::optional<StarredExpression> expression;
        Token closeParenth;
    };

    struct ListDisplay
    {
        Token openSquare;
        std::variant<std::monostate, StarredList, Comprehension> variant;
        Token closeSquare;
    };

    struct SetDisplay
    {
        Token openBrace;
        std::variant<std::monostate, StarredList, Comprehension> variant;
        Token closeBrace;
    };

    struct DictDisplay
    {
        Token openBrace;
        struct KeyDatum
        {
            struct Key
            {
                Expression first;
                Token colon;
                Expression second;
            };
            struct Datum
            {
                Token powerOf;
                OrExpr orExpr;
            };
            std::variant<Key, Datum> variant;
        };
        struct DictComprehension
        {
            Expression first;
            Token colon;
            Expression second;
            CompFor compFor;
        };
        std::variant<CommaList<KeyDatum>, DictComprehension> variant;
        Token closeBrace;
    };

    struct GeneratorExpression
    {
        Token openParenth;
        Expression expression;
        CompFor compFor;
        Token closeParenth;
    };

    struct YieldAtom
    {
        Token openParenth;
        struct YieldExpression
        {
            Token yieldToken;
            std::variant<ExpressionList, std::pair<Token, Expression>> variant;
        };
        Token closeParenth;
    };

    std::variant<ParenthForm, ListDisplay, SetDisplay, DictDisplay, GeneratorExpression, YieldAtom> variant;
};

// TODO:
struct Target
{
};

// TODO:
struct ParameterList
{
};

bool firstInAssignmentExpression(TokenType tokenType)
{
    return firstInExpression(tokenType);
}

} // namespace pylir::Syntax
