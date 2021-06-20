
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
    std::vector<std::pair<Token, std::unique_ptr<T>>> remainingExpr;
    std::optional<Token> trailingComma;
};

using ExpressionList = CommaList<Expression>;

struct Enclosure;

/**
 * atom      ::=  identifier | literal | enclosure
 */
struct Atom
{
    /**
     * literal ::=  stringliteral | bytesliteral
                    | integer | floatnumber | imagnumber
     */
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

/**
 * attributeref ::=  primary "." identifier
 */
struct AttributeRef
{
    std::unique_ptr<Primary> primary;
    Token dot;
    Token identifier;
};

/**
 * subscription ::=  primary "[" expression_list "]"
 */
struct Subscription
{
    std::unique_ptr<Primary> primary;
    Token openSquareBracket;
    ExpressionList expressionList;
    Token closeSquareBracket;
};

/**
 * slicing      ::=  primary "[" slice_list "]"
   slice_list   ::=  slice_item ("," slice_item)* [","]
   slice_item   ::=  expression | proper_slice
   proper_slice ::=  [lower_bound] ":" [upper_bound] [ ":" [stride] ]
   lower_bound  ::=  expression
   upper_bound  ::=  expression
   stride       ::=  expression
 */
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

/**
 * call                 ::=  primary "(" [argument_list [","] | comprehension] ")"
   argument_list        ::=  positional_arguments ["," starred_and_keywords]
                            ["," keywords_arguments]
                          | starred_and_keywords ["," keywords_arguments]
                          | keywords_arguments
   positional_arguments ::=  positional_item ("," positional_item)*
   positional_item      ::=  assignment_expression | "*" expression
   starred_and_keywords ::=  ("*" expression | keyword_item)
                          ("," "*" expression | "," keyword_item)*
   keywords_arguments   ::=  (keyword_item | "**" expression)
                          ("," keyword_item | "," "**" expression)*
   keyword_item         ::=  identifier "=" expression
 */
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

/**
 * primary ::=  atom | attributeref | subscription | slicing | call
 */
struct Primary
{
    std::variant<Atom, AttributeRef, Subscription, Slicing, Call> variant;
};

/**
 * await_expr ::=  "await" primary
 */
struct AwaitExpr
{
    Token awaitToken;
    Primary primary;
};

struct UExpr;

/**
 * power ::=  (await_expr | primary) ["**" u_expr]
 */
struct Power
{
    std::variant<AwaitExpr, Primary> variant;
    std::optional<std::pair<Token, std::unique_ptr<UExpr>>> rightHand;
};

/**
 * u_expr ::=  power | "-" u_expr | "+" u_expr | "~" u_expr
 */
struct UExpr
{
    std::variant<Power, std::pair<Token, std::unique_ptr<UExpr>>> variant;
};

/**
 * m_expr ::=  u_expr | m_expr "*" u_expr | m_expr "@" m_expr |
            m_expr "//" u_expr | m_expr "/" u_expr |
            m_expr "%" u_expr
 */
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

/**
 * a_expr ::=  m_expr | a_expr "+" m_expr | a_expr "-" m_expr
 */
struct AExpr
{
    struct BinOp
    {
        std::unique_ptr<AExpr> lhs;
        Token binToken;
        MExpr rhs;
    };

    std::variant<MExpr, BinOp> variant;
};

/**
 * shift_expr ::=  a_expr | shift_expr ("<<" | ">>") a_expr
 */
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

/**
 * and_expr ::=  shift_expr | and_expr "&" shift_expr
 */
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

/**
 * xor_expr ::=  and_expr | xor_expr "^" and_expr
 */
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

/**
 * or_expr  ::=  xor_expr | or_expr "|" xor_expr
 */
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

/**
 * comparison    ::=  or_expr (comp_operator or_expr)*
   comp_operator ::=  "<" | ">" | "==" | ">=" | "<=" | "!="
                   | "is" ["not"] | ["not"] "in"
 */
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

/**
 * not_test ::=  comparison | "not" not_test
 */
struct NotTest
{
    std::variant<Comparison, std::pair<Token, std::unique_ptr<NotTest>>> variant;
};

/**
 * and_test ::=  not_test | and_test "and" not_test
 */
struct AndTest
{
    struct BinOp
    {
        std::unique_ptr<AndTest> lhs;
        Token andToken;
        NotTest rhs;
    };

    std::variant<NotTest, BinOp> variant;
};

/**
 * or_test  ::=  and_test | or_test "or" and_test
 */
struct OrTest
{
    struct BinOp
    {
        std::unique_ptr<OrTest> lhs;
        Token orToken;
        AndTest rhs;
    };

    std::variant<AndTest, BinOp> variant;
};

/**
 * assignment_expression ::=  [identifier ":="] expression
 */
struct AssignmentExpression
{
    std::optional<std::pair<Token, Token>> identifierAndWalrus;
    std::unique_ptr<Expression> expression;
};

inline bool firstInAssignmentExpression(TokenType tokenType);

/**
 * conditional_expression ::=  or_test ["if" or_test "else" expression]
 */
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

/**
 * expression             ::=  conditional_expression | lambda_expr
 */
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

/**
 * lambda_expr ::=  "lambda" [parameter_list] ":" expression
 */
struct LambdaExpression
{
    Token lambdaToken;
    std::unique_ptr<ParameterList> parameterList;
    Token colonToken;
    Expression expression;
};

/**
 * starred_item       ::=  assignment_expression | "*" or_expr
 */
struct StarredItem
{
    std::variant<AssignmentExpression, std::pair<Token, OrExpr>> variant;
};

inline bool firstInStarredItem(TokenType tokenType)
{
    return tokenType == TokenType::Star || firstInAssignmentExpression(tokenType);
}

using StarredList = CommaList<StarredItem>;

/**
 * starred_expression ::=  expression | (starred_item ",")* [starred_item]
 */
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

/**
 * comp_for      ::=  ["async"] "for" target_list "in" or_test [comp_iter]
 */
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

/**
 * comp_if       ::=  "if" or_test [comp_iter]
 */
struct CompIf
{
    Token ifToken;
    OrTest orTest;
    std::variant<std::monostate, CompFor, std::unique_ptr<CompIf>> compIter;
};

/**
 * comprehension ::=  assignment_expression comp_for
 */
struct Comprehension
{
    AssignmentExpression assignmentExpression;
    CompFor compFor;
};

inline bool firstInComprehension(TokenType tokenType)
{
    return firstInAssignmentExpression(tokenType);
}

/**
 * yield_expression ::=  "yield" [expression_list | "from" expression]
 */
struct YieldExpression
{
    Token yieldToken;
    std::variant<std::monostate, ExpressionList, std::pair<Token, Expression>> variant;
};

/**
 * enclosure ::=  parenth_form | list_display | dict_display | set_display
               | generator_expression | yield_atom
 */
struct Enclosure
{
    /**
     * parenth_form ::=  "(" [starred_expression] ")"
     */
    struct ParenthForm
    {
        Token openParenth;
        std::optional<StarredExpression> expression;
        Token closeParenth;
    };

    /**
     * list_display ::=  "[" [starred_list | comprehension] "]"
     */
    struct ListDisplay
    {
        Token openSquare;
        std::variant<std::monostate, StarredList, Comprehension> variant;
        Token closeSquare;
    };

    /**
     * set_display ::=  "{" (starred_list | comprehension) "}"
     */
    struct SetDisplay
    {
        Token openBrace;
        std::variant<StarredList, Comprehension> variant;
        Token closeBrace;
    };

    /**
     * dict_display       ::=  "{" [key_datum_list | dict_comprehension] "}"
       key_datum_list     ::=  key_datum ("," key_datum)* [","]
       key_datum          ::=  expression ":" expression | "**" or_expr
       dict_comprehension ::=  expression ":" expression comp_for
     */
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
        std::variant<std::monostate, CommaList<KeyDatum>, DictComprehension> variant;
        Token closeBrace;
    };

    /**
     * generator_expression ::=  "(" expression comp_for ")"
     */
    struct GeneratorExpression
    {
        Token openParenth;
        Expression expression;
        CompFor compFor;
        Token closeParenth;
    };

    /**
     * yield_atom       ::=  "(" yield_expression ")"
     */
    struct YieldAtom
    {
        Token openParenth;
        YieldExpression yieldExpression;
        Token closeParenth;
    };

    std::variant<ParenthForm, ListDisplay, SetDisplay, DictDisplay, GeneratorExpression, YieldAtom> variant;
};

/**
 * target          ::=  identifier
                     | "(" [target_list] ")"
                     | "[" [target_list] "]"
                     | attributeref
                     | subscription
                     | slicing
                     | "*" target
 */

struct Target
{
    struct Parenth
    {
        Token openParenth;
        TargetList targetList;
        Token closeParenth;
    };

    struct Square
    {
        Token openSquare;
        TargetList targetList;
        Token closeSquare;
    };

    std::variant<Token, Parenth, Square, AttributeRef, Subscription, Slicing, std::pair<Token, std::unique_ptr<Target>>>
        variant;
};

/**
 * assignment_stmt ::=  (target_list "=")+ (starred_expression | yield_expression)
 */
struct AssignmentStmt
{
    std::vector<std::pair<TargetList, Token>> targets;
    std::variant<StarredExpression, YieldExpression> variant;
};

/**
 * augtarget                 ::=  identifier | attributeref | subscription | slicing
 */
struct AugTarget
{
    std::variant<Token, AttributeRef, Subscription, Slicing> variant;
};

/**
 * augmented_assignment_stmt ::=  augtarget augop (expression_list | yield_expression)
 * augop                     ::=  "+=" | "-=" | "*=" | "@=" | "/=" | "//=" | "%=" | "**="
                               | ">>=" | "<<=" | "&=" | "^=" | "|="
 */
struct AugmentedAssignmentStmt
{
    AugTarget augTarget;
    Token augOp;
    std::variant<ExpressionList, YieldExpression> variant;
};

/**
 * annotated_assignment_stmt ::=  augtarget ":" expression
                               ["=" (starred_expression | yield_expression)]
 */
struct AnnotatedAssignmentSmt
{
    AugTarget augTarget;
    Token colon;
    Expression expression;
    std::optional<std::pair<Token, std::variant<ExpressionList, YieldExpression>>> optionalAssignmentStmt;
};

/**
 * assert_stmt ::=  "assert" expression ["," expression]
 */
struct AssertStmt
{
    Token assertKeyword;
    Expression condition;
    std::optional<std::pair<Token, Expression>> message;
};

/**
 * pass_stmt ::=  "pass"
 */
struct PassStmt
{
    Token pass;
};

/**
 * del_stmt ::=  "del" target_list
 */
struct DelStmt
{
    Token del;
    TargetList targetList;
};

/**
 * return_stmt ::=  "return" [expression_list]
 */
struct ReturnStmt
{
    Token returnKeyword;
    std::optional<ExpressionList> expressions;
};

/**
 * yield_stmt ::=  yield_expression
 */
struct YieldStmt
{
    YieldExpression yieldExpression;
};

/**
 * raise_stmt ::=  "raise" [expression ["from" expression]]
 */
struct RaiseStmt
{
    Token raise;
    std::optional<std::pair<Expression, std::optional<std::pair<Token, Expression>>>> expressions;
};

/**
 * break_stmt ::=  "break"
 */
struct BreakStmt
{
    Token breakKeyword;
};

/**
 * continue_stmt ::=  "continue"
 */
struct ContinueStmt
{
    Token continueKeyword;
};

/**
 * import_stmt     ::=  "import" module ["as" identifier] ("," module ["as" identifier])*
                     | "from" relative_module "import" identifier ["as" identifier]
                     ("," identifier ["as" identifier])*
                     | "from" relative_module "import" "(" identifier ["as" identifier]
                     ("," identifier ["as" identifier])* [","] ")"
                     | "from" relative_module "import" "*"
   module          ::=  (identifier ".")* identifier
   relative_module ::=  "."* module | "."+
 */
struct ImportStmt
{
    struct Module
    {
        std::vector<std::pair<Token, Token>> leading;
        Token lastIdentifier;
    };

    struct RelativeModule
    {
        std::vector<Token> dots;
        std::optional<Module> module;
    };

    struct ImportAsAs
    {
        Token import;
        Module module;
        std::optional<std::pair<Token, Token>> name;
        struct Further
        {
            Token comma;
            Module module;
            std::optional<std::pair<Token, Token>> name;
        };
        std::vector<Further> rest;
    };

    struct FromImportList
    {
        Token from;
        RelativeModule relativeModule;
        Token import;
        std::optional<Token> openParenth;
        Token identifier;
        std::optional<std::pair<Token, Token>> name;
        struct Further
        {
            Token comma;
            Token identifier;
            std::optional<std::pair<Token, Token>> name;
        };
        std::vector<Further> rest;
        std::optional<Token> comma;
        std::optional<Token> closeParenth;
    };

    struct FromImportAll
    {
        Token from;
        RelativeModule relativeModule;
        Token import;
        Token star;
    };

    std::variant<ImportAsAs, FromImportList, FromImportAll> variant;
};

/**
 * future_stmt ::=  "from" "__future__" "import" feature ["as" identifier]
                 ("," feature ["as" identifier])*
                 | "from" "__future__" "import" "(" feature ["as" identifier]
                 ("," feature ["as" identifier])* [","] ")"
   feature     ::=  identifier
 */
struct FutureStmt
{
    Token from;
    Token future;
    Token import;
    std::optional<Token> openParenth;
    Token identifier;
    std::optional<std::pair<Token, Token>> name;
    struct Further
    {
        Token comma;
        Token identifier;
        std::optional<std::pair<Token, Token>> name;
    };
    std::vector<Further> rest;
    std::optional<Token> comma;
    std::optional<Token> closeParenth;
};

/**
 * global_stmt ::=  "global" identifier ("," identifier)*
 */
struct GlobalStmt
{
    Token global;
    Token identifier;
    std::vector<std::pair<Token, Token>> rest;
};

/**
 * nonlocal_stmt ::=  "nonlocal" identifier ("," identifier)*
 */
struct NonLocalStmt
{
    Token nonLocal;
    Token identifier;
    std::vector<std::pair<Token, Token>> rest;
};

/**
 * simple_stmt ::=  expression_stmt
                 | assert_stmt
                 | assignment_stmt
                 | augmented_assignment_stmt
                 | annotated_assignment_stmt
                 | pass_stmt
                 | del_stmt
                 | return_stmt
                 | yield_stmt
                 | raise_stmt
                 | break_stmt
                 | continue_stmt
                 | import_stmt
                 | future_stmt
                 | global_stmt
                 | nonlocal_stmt
 */
struct SimpleStmt
{
    std::variant<StarredExpression, AssertStmt, AssignmentStmt, AugmentedAssignmentStmt, AnnotatedAssignmentSmt,
                 PassStmt, DelStmt, ReturnStmt, YieldStmt, RaiseStmt, BreakStmt, ContinueStmt, ImportStmt, FutureStmt,
                 GlobalStmt, NonLocalStmt>
        variant;
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
