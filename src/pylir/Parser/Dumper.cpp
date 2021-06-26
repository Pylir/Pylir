#include "Dumper.hpp"

#include <pylir/Support/Variant.hpp>

#include <fmt/format.h>
#include <tcb/span.hpp>

namespace
{
std::vector<std::string_view> splitLines(std::string_view text)
{
    std::vector<std::string_view> result;
    std::size_t pos = 0;
    while ((pos = text.find('\n')) != std::string_view::npos)
    {
        result.push_back(text.substr(0, pos));
        text.remove_prefix(pos + 1);
    }
    result.push_back(text);
    return result;
}
} // namespace

std::string pylir::Dumper::addLastChild(std::string_view lastChildDump, std::optional<std::string_view>&& label)
{
    auto lines = splitLines(lastChildDump);
    std::string result;
    bool first = true;
    for (auto iter : lines)
    {
        if (first)
        {
            first = false;
            if (label)
            {
                result += "\n`-" + std::string(*label) + ": " + std::string(iter);
            }
            else
            {
                result += "\n`-" + std::string(iter);
            }
        }
        else
        {
            result += "\n  " + std::string(iter);
        }
    }
    return result;
}

std::string pylir::Dumper::addMiddleChild(std::string_view middleChildDump, std::optional<std::string_view>&& label)
{
    auto lines = splitLines(middleChildDump);
    std::string result;
    bool first = true;
    for (auto iter : lines)
    {
        if (first)
        {
            first = false;
            if (label)
            {
                result += "\n|-" + std::string(*label) + ": " + std::string(iter);
            }
            else
            {
                result += "\n|-" + std::string(iter);
            }
        }
        else
        {
            result += "\n| " + std::string(iter);
        }
    }
    return result;
}

std::string pylir::Dumper::dump(const pylir::Syntax::Atom& atom)
{
    return pylir::match(
        atom.variant,
        [](const IdentifierToken& identifier) -> std::string { return fmt::format("atom {}", identifier.getValue()); },
        [](const Syntax::Atom::Literal& literal) -> std::string
        {
            return pylir::match(
                literal.token.getValue(),
                [](double value) -> std::string { return fmt::format(FMT_STRING("atom {:#}"), value); },
                [](const llvm::APInt& apInt) -> std::string
                { return fmt::format("atom {}", apInt.toString(10, false)); },
                [&](const std::string& string) -> std::string
                {
                    if (literal.token.getTokenType() == TokenType::StringLiteral)
                    {
                        std::string result;
                        result.reserve(string.size());
                        for (auto character : string)
                        {
                            switch (character)
                            {
                                case '\'': result += "\\'"; break;
                                case '\\': result += "\\\\"; break;
                                case '\a': result += "\\a"; break;
                                case '\b': result += "\\b"; break;
                                case '\f': result += "\\f"; break;
                                case '\r': result += "\\r"; break;
                                case '\t': result += "\\t"; break;
                                case '\v': result += "\\v"; break;
                                case '\n': result += "\\n"; break;
                                default: result += character; break;
                            }
                        }
                        return fmt::format("atom '{}'", result);
                    }

                    std::string result;
                    result.reserve(string.size());
                    for (auto character : string)
                    {
                        switch (character)
                        {
                            case '\'': result += "\\'"; break;
                            case '\\': result += "\\\\"; break;
                            case '\a': result += "\\a"; break;
                            case '\b': result += "\\b"; break;
                            case '\f': result += "\\f"; break;
                            case '\r': result += "\\r"; break;
                            case '\t': result += "\\t"; break;
                            case '\v': result += "\\v"; break;
                            case '\n': result += "\\n"; break;
                            default:
                                std::uint32_t uchar = static_cast<std::uint8_t>(character);
                                // Control characters or ones that are not normal ascii
                                if (uchar <= 31 || uchar >= 127)
                                {
                                    result += fmt::format(FMT_STRING("\\x{:0^2X}"), uchar);
                                }
                                else
                                {
                                    result += character;
                                }
                                break;
                        }
                    }
                    return fmt::format("atom b'{}'", result);
                },
                [](std::monostate) -> std::string { PYLIR_UNREACHABLE; });
        },
        [&](const std::unique_ptr<Syntax::Enclosure>& enclosure) -> std::string { return dump(*enclosure); });
}

std::string pylir::Dumper::dump(const pylir::Syntax::Enclosure& enclosure)
{
    return pylir::match(
        enclosure.variant,
        [&](const Syntax::Enclosure::DictDisplay& dictDisplay) -> std::string
        {
            if (std::holds_alternative<std::monostate>(dictDisplay.variant))
            {
                return "dict display empty";
            }

            return "dict display"
                   + addLastChild(pylir::match(
                       dictDisplay.variant, [](std::monostate) -> std::string { PYLIR_UNREACHABLE; },
                       [&](const Syntax::Enclosure::DictDisplay::DictComprehension& comprehension) -> std::string
                       {
                           return "dict comprehension" + addMiddleChild(dump(comprehension.first))
                                  + addMiddleChild(dump(comprehension.second)
                                                   + addLastChild(dump(comprehension.compFor)));
                       },
                       [&](const Syntax::CommaList<Syntax::Enclosure::DictDisplay::KeyDatum>& commaList) -> std::string
                       {
                           return dump(
                               commaList,
                               [&](const Syntax::Enclosure::DictDisplay::KeyDatum& keyDatum) -> std::string
                               {
                                   return pylir::match(
                                       keyDatum.variant,
                                       [&](const Syntax::Enclosure::DictDisplay::KeyDatum::Key& key) -> std::string {
                                           return "key" + addMiddleChild(dump(key.first))
                                                  + addLastChild(dump(key.second));
                                       },
                                       [&](const Syntax::Enclosure::DictDisplay::KeyDatum::Datum& datum) -> std::string
                                       { return "datum" + addLastChild(dump(datum.orExpr)); });
                               },
                               "key datum list");
                       }));
        },
        [&](const Syntax::Enclosure::SetDisplay& setDisplay) -> std::string
        {
            std::string result = "set display";
            result += pylir::match(
                setDisplay.variant, [](const std::monostate&) -> std::string { PYLIR_UNREACHABLE; },
                [&](const auto& value) { return addLastChild(dump(value)); });
            return result;
        },
        [&](const Syntax::Enclosure::ListDisplay& listDisplay) -> std::string
        {
            if (std::holds_alternative<std::monostate>(listDisplay.variant))
            {
                return "list display empty";
            }
            std::string result = "list display";
            result += pylir::match(
                listDisplay.variant, [](const std::monostate&) -> std::string { PYLIR_UNREACHABLE; },
                [&](const auto& value) { return addLastChild(dump(value)); });
            return result;
        },
        [&](const Syntax::Enclosure::GeneratorExpression& generatorExpression) -> std::string
        {
            std::string result = "generator expression";
            result += addMiddleChild(dump(generatorExpression.expression));
            result += addLastChild(dump(generatorExpression.compFor));
            return result;
        },
        [&](const Syntax::Enclosure::YieldAtom& yieldAtom) -> std::string
        { return "yieldatom" + addLastChild(dump(yieldAtom.yieldExpression)); },
        [&](const Syntax::Enclosure::ParenthForm& parenthForm) -> std::string
        {
            if (!parenthForm.expression)
            {
                return "parenth empty";
            }
            else
            {
                std::string result = "parenth";
                result += addLastChild(dump(*parenthForm.expression));
                return result;
            }
        });
}

std::string pylir::Dumper::dump(const pylir::Syntax::Primary& primary)
{
    return pylir::match(primary.variant, [&](const auto& value) -> std::string { return dump(value); });
}

std::string pylir::Dumper::dump(const pylir::Syntax::AttributeRef& attribute)
{
    return fmt::format("attribute {}", attribute.identifier.getValue()) + addLastChild(dump(*attribute.primary));
}

std::string pylir::Dumper::dump(const pylir::Syntax::Subscription& subscription)
{
    return "subscription" + addMiddleChild(dump(*subscription.primary), "primary")
           + addLastChild(dump(subscription.expressionList), "index");
}

std::string pylir::Dumper::dump(const pylir::Syntax::Slicing& slicing)
{
    std::string result = "slicing";
    result += addMiddleChild(dump(*slicing.primary), "primary");
    result += addLastChild(
        dump(
            slicing.sliceList,
            [&](const auto& slice) -> std::string
            {
                return pylir::match(
                    slice,
                    [&](const Syntax::Slicing::ProperSlice& properSlice) -> std::string
                    {
                        std::string result = "proper slice";
                        if (properSlice.optionalLowerBound)
                        {
                            if (!properSlice.optionalUpperBound && !properSlice.optionalStride)
                            {
                                return result + addLastChild(dump(*properSlice.optionalLowerBound), "lowerBound");
                            }
                            else
                            {
                                result += addMiddleChild(dump(*properSlice.optionalLowerBound), "lowerBound");
                            }
                        }
                        if (properSlice.optionalUpperBound)
                        {
                            if (!properSlice.optionalStride)
                            {
                                return result + addLastChild(dump(*properSlice.optionalUpperBound), "upperBound");
                            }
                            else
                            {
                                result += addMiddleChild(dump(*properSlice.optionalUpperBound), "upperBound");
                            }
                        }
                        if (properSlice.optionalStride)
                        {
                            result += addLastChild(dump(*properSlice.optionalStride), "stride");
                        }
                        return result;
                    },
                    [&](const Syntax::Expression& expression) { return dump(expression); });
            },
            "proper slice list"),
        "index");
    return result;
}

std::string pylir::Dumper::dump(const pylir::Syntax::Comprehension& comprehension)
{
    return "comprehension" + addMiddleChild(dump(comprehension.assignmentExpression))
           + addLastChild(dump(comprehension.compFor));
}

std::string pylir::Dumper::dump(const pylir::Syntax::AssignmentExpression& assignmentExpression)
{
    if (!assignmentExpression.identifierAndWalrus)
    {
        return dump(*assignmentExpression.expression);
    }
    return fmt::format("assignment expression to {}", assignmentExpression.identifierAndWalrus->first.getValue())
           + addLastChild(dump(*assignmentExpression.expression));
}

std::string pylir::Dumper::dump(const pylir::Syntax::Call& call)
{
    std::string result = "call";
    if (std::holds_alternative<std::monostate>(call.variant))
    {
        return result + addLastChild(dump(*call.primary));
    }
    result += addMiddleChild(dump(*call.primary));
    if (auto* comprehension = std::get_if<std::unique_ptr<Syntax::Comprehension>>(&call.variant))
    {
        return result + addLastChild(dump(**comprehension));
    }
    auto& [argument, comma] = pylir::get<std::pair<Syntax::ArgumentList, std::optional<BaseToken>>>(call.variant);
    if (argument.positionalArguments)
    {
        std::string positional = "positional arguments";
        if (argument.positionalArguments->rest.empty())
        {
            positional += addLastChild(pylir::match(
                argument.positionalArguments->firstItem.variant,
                [&](const std::unique_ptr<Syntax::AssignmentExpression>& value) { return dump(*value); },
                [&](const Syntax::ArgumentList::PositionalItem::Star& star)
                { return "starred" + addLastChild(dump(*star.expression)); }));
        }
        else
        {
            positional += addMiddleChild(pylir::match(
                argument.positionalArguments->firstItem.variant,
                [&](const std::unique_ptr<Syntax::AssignmentExpression>& value) { return dump(*value); },
                [&](const Syntax::ArgumentList::PositionalItem::Star& star)
                { return "starred" + addLastChild(dump(*star.expression)); }));
            for (auto& [token, item] :
                 tcb::span(argument.positionalArguments->rest).first(argument.positionalArguments->rest.size() - 1))
            {
                positional += addMiddleChild(pylir::match(
                    item.variant,
                    [&](const std::unique_ptr<Syntax::AssignmentExpression>& value) { return dump(*value); },
                    [&](const Syntax::ArgumentList::PositionalItem::Star& star)
                    { return "starred" + addLastChild(dump(*star.expression)); }));
            }
            positional += addLastChild(pylir::match(
                argument.positionalArguments->rest.back().second.variant,
                [&](const std::unique_ptr<Syntax::AssignmentExpression>& value) { return dump(*value); },
                [&](const Syntax::ArgumentList::PositionalItem::Star& star)
                { return "starred" + addLastChild(dump(*star.expression)); }));
        }
        if (!argument.starredAndKeywords && !argument.keywordArguments)
        {
            return result + addLastChild(positional);
        }
        else
        {
            result += addMiddleChild(positional);
        }
    }
    if (argument.starredAndKeywords)
    {
        std::string starred = "starred keywords";
        std::string keyword = fmt::format("keyword item {}", argument.starredAndKeywords->first.identifier.getValue());
        keyword += addLastChild(dump(*argument.starredAndKeywords->first.expression));
        if (argument.starredAndKeywords->rest.empty())
        {
            starred += addLastChild(keyword);
        }
        else
        {
            starred += addMiddleChild(keyword);
            for (auto& [token, iter] :
                 tcb::span(argument.starredAndKeywords->rest).first(argument.starredAndKeywords->rest.size() - 1))
            {
                starred += addMiddleChild(pylir::match(
                    iter,
                    [&](const Syntax::ArgumentList::KeywordItem& keywordItem)
                    {
                        return fmt::format("keyword item {}", keywordItem.identifier.getValue())
                               + addLastChild(dump(*keywordItem.expression));
                    },
                    [&](const Syntax::ArgumentList::StarredAndKeywords::Expression& expression)
                    { return "starred expression" + addLastChild(dump(*expression.expression)); }));
            }
            starred += addLastChild(pylir::match(
                argument.starredAndKeywords->rest.back().second,
                [&](const Syntax::ArgumentList::KeywordItem& keywordItem)
                {
                    return fmt::format("keyword item {}", keywordItem.identifier.getValue())
                           + addLastChild(dump(*keywordItem.expression));
                },
                [&](const Syntax::ArgumentList::StarredAndKeywords::Expression& expression)
                { return "starred expression" + addLastChild(dump(*expression.expression)); }));
        }
        if (!argument.keywordArguments)
        {
            return result + addLastChild(starred);
        }
        else
        {
            result += addMiddleChild(starred);
        }
    }
    if (argument.keywordArguments)
    {
        std::string starred = "keyword arguments";
        std::string mapped = "mapped expression";
        mapped += addLastChild(dump(*argument.keywordArguments->first.expression));
        if (argument.keywordArguments->rest.empty())
        {
            starred += addLastChild(mapped);
        }
        else
        {
            starred += addMiddleChild(mapped);
            for (auto& [token, iter] :
                 tcb::span(argument.keywordArguments->rest).first(argument.keywordArguments->rest.size() - 1))
            {
                starred += addMiddleChild(pylir::match(
                    iter,
                    [&](const Syntax::ArgumentList::KeywordItem& keywordItem)
                    {
                        return fmt::format("keyword item {}", keywordItem.identifier.getValue())
                               + addLastChild(dump(*keywordItem.expression));
                    },
                    [&](const Syntax::ArgumentList::KeywordArguments::Expression& expression)
                    { return "mapped expression" + addLastChild(dump(*expression.expression)); }));
            }
            starred += addLastChild(pylir::match(
                argument.keywordArguments->rest.back().second,
                [&](const Syntax::ArgumentList::KeywordItem& keywordItem)
                {
                    return fmt::format("keyword item {}", keywordItem.identifier.getValue())
                           + addLastChild(dump(*keywordItem.expression));
                },
                [&](const Syntax::ArgumentList::KeywordArguments::Expression& expression)
                { return "mapped expression" + addLastChild(dump(*expression.expression)); }));
        }
        result += addLastChild(starred);
    }
    return result;
}

std::string pylir::Dumper::dump(const pylir::Syntax::AwaitExpr& awaitExpr)
{
    return fmt::format("await expression") + addLastChild(dump(awaitExpr.primary));
}

std::string pylir::Dumper::dump(const pylir::Syntax::UExpr& uExpr)
{
    if (auto* power = std::get_if<Syntax::Power>(&uExpr.variant))
    {
        return dump(*power);
    }
    auto& [token, rhs] = pylir::get<std::pair<Token, std::unique_ptr<Syntax::UExpr>>>(uExpr.variant);
    return fmt::format(FMT_STRING("unary {:q}"), token.getTokenType()) + addLastChild(dump(*rhs));
}

std::string pylir::Dumper::dump(const pylir::Syntax::Power& power)
{
    if (!power.rightHand)
    {
        return pylir::match(power.variant, [&](const auto& value) { return dump(value); });
    }
    return std::string("power")
           + addMiddleChild(pylir::match(power.variant, [&](const auto& value) { return dump(value); }), "base")
           + addLastChild(dump(*power.rightHand->second), "exponent");
}

std::string pylir::Dumper::dump(const pylir::Syntax::MExpr& mExpr)
{
    return pylir::match(
        mExpr.variant, [&](const Syntax::UExpr& uExpr) { return dump(uExpr); },
        [&](const std::unique_ptr<Syntax::MExpr::BinOp>& binOp)
        {
            return fmt::format("mexpr {:q}", binOp->binToken.getTokenType()) + addMiddleChild(dump(*binOp->lhs), "lhs")
                   + addLastChild(dump(binOp->rhs), "rhs");
        },
        [&](const std::unique_ptr<Syntax::MExpr::AtBin>& binOp)
        {
            return fmt::format("mexpr {:q}", TokenType::AtSign) + addMiddleChild(dump(*binOp->lhs), "lhs")
                   + addLastChild(dump(*binOp->rhs), "rhs");
        });
}

std::string pylir::Dumper::dump(const pylir::Syntax::AExpr& aExpr)
{
    return dumpBinOp(aExpr, "aexpr", &Token::getTokenType);
}

std::string pylir::Dumper::dump(const pylir::Syntax::ShiftExpr& shiftExpr)
{
    return dumpBinOp(shiftExpr, "shiftExpr", &Token::getTokenType);
}

std::string pylir::Dumper::dump(const pylir::Syntax::AndExpr& andExpr)
{
    return dumpBinOp(andExpr, "andExpr", [](auto&&) { return TokenType::BitAnd; });
}

std::string pylir::Dumper::dump(const pylir::Syntax::XorExpr& xorExpr)
{
    return dumpBinOp(xorExpr, "xorExpr", [](auto&&) { return TokenType::BitXor; });
}

std::string pylir::Dumper::dump(const pylir::Syntax::OrExpr& orExpr)
{
    return dumpBinOp(orExpr, "orExpr", [](auto&&) { return TokenType::BitOr; });
}

std::string pylir::Dumper::dump(const pylir::Syntax::Comparison& comparison)
{
    if (comparison.rest.empty())
    {
        return dump(comparison.left);
    }
    std::string result = "comparison";
    result += addMiddleChild(dump(comparison.left), "lhs");
    for (auto& [token, rhs] : tcb::span(comparison.rest).first(comparison.rest.size() - 1))
    {
        std::string name;
        if (token.secondToken)
        {
            name = fmt::format("{:q} {:q}", token.firstToken.getTokenType(), token.secondToken->getTokenType());
        }
        else
        {
            name = fmt::format("{:q}", token.firstToken.getTokenType());
        }
        result += addMiddleChild(dump(rhs), name);
    }
    std::string name;
    if (comparison.rest.back().first.secondToken)
    {
        name = fmt::format("{:q} {:q}", comparison.rest.back().first.firstToken.getTokenType(),
                           comparison.rest.back().first.secondToken->getTokenType());
    }
    else
    {
        name = fmt::format("{:q}", comparison.rest.back().first.firstToken.getTokenType());
    }
    result += addLastChild(dump(comparison.rest.back().second), name);
    return result;
}

std::string pylir::Dumper::dump(const pylir::Syntax::NotTest& notTest)
{
    return pylir::match(
        notTest.variant, [&](const Syntax::Comparison& comparison) { return dump(comparison); },
        [&](const auto& pair) { return "notTest" + addLastChild(dump(*pair.second)); });
}

std::string pylir::Dumper::dump(const pylir::Syntax::AndTest& andTest)
{
    return dumpBinOp(andTest, "andTest", [](auto&&) { return TokenType::AndKeyword; });
}

std::string pylir::Dumper::dump(const pylir::Syntax::OrTest& orTest)
{
    return dumpBinOp(orTest, "orTest", [](auto&&) { return TokenType::OrKeyword; });
}

std::string pylir::Dumper::dump(const pylir::Syntax::ConditionalExpression& conditionalExpression)
{
    if (!conditionalExpression.suffix)
    {
        return dump(conditionalExpression.value);
    }
    return fmt::format("conditional expression") + addMiddleChild(dump(conditionalExpression.value), "value")
           + addMiddleChild(dump(*conditionalExpression.suffix->test), "condition")
           + addLastChild(dump(*conditionalExpression.suffix->elseValue), "elseValue");
}

std::string pylir::Dumper::dump(const pylir::Syntax::LambdaExpression& lambdaExpression)
{
    // TODO: parameter list
    return fmt::format("lambda expression") + addLastChild(dump(lambdaExpression.expression));
}

std::string pylir::Dumper::dump(const pylir::Syntax::Expression& expression)
{
    return pylir::match(
        expression.variant, [&](const auto& value) { return dump(value); },
        [&](const std::unique_ptr<Syntax::LambdaExpression>& lambdaExpression) { return dump(*lambdaExpression); });
}

std::string pylir::Dumper::dump(const pylir::Syntax::StarredItem& starredItem)
{
    if (auto* assignment = std::get_if<Syntax::AssignmentExpression>(&starredItem.variant))
    {
        return dump(*assignment);
    }
    return "starred item"
           + addLastChild(dump(pylir::get<std::pair<BaseToken, Syntax::OrExpr>>(starredItem.variant).second));
}

std::string pylir::Dumper::dump(const pylir::Syntax::StarredExpression& starredExpression)
{
    if (auto* expression = std::get_if<Syntax::Expression>(&starredExpression.variant))
    {
        return dump(*expression);
    }
    auto& item = pylir::get<Syntax::StarredExpression::Items>(starredExpression.variant);
    std::string result = "starred expression";
    for (auto& iter : tcb::span(item.leading).first(item.leading.size() - 1))
    {
        result += addMiddleChild(dump(iter.first));
    }
    if (!item.last)
    {
        result += addLastChild(dump(item.leading.back().first));
    }
    else
    {
        result += addMiddleChild(dump(item.leading.back().first));
        result += addLastChild(dump(*item.last));
    }
    return result;
}

std::string pylir::Dumper::dump(const pylir::Syntax::CompIf& compIf)
{
    auto result = fmt::format("comp if");
    return pylir::match(
        compIf.compIter, [&](std::monostate) { return result + addLastChild(dump(compIf.orTest), "condition"); },
        [&](const Syntax::CompFor& compFor)
        { return result + addMiddleChild(dump(compIf.orTest), "condition") + addLastChild(dump(compFor)); },
        [&](const std::unique_ptr<Syntax::CompIf>& second)
        { return result + addMiddleChild(dump(compIf.orTest), "condition") + addLastChild(dump(*second)); });
}

std::string pylir::Dumper::dump(const pylir::Syntax::CompFor& compFor)
{
    std::string result;
    if (compFor.awaitToken)
    {
        result = "comp for await";
    }
    else
    {
        result = "comp for";
    }
    result += addMiddleChild(dump(compFor.targets));
    if (std::holds_alternative<std::monostate>(compFor.compIter))
    {
        return result + addLastChild(dump(compFor.orTest));
    }
    result += addMiddleChild(dump(compFor.orTest));
    result += addLastChild(pylir::match(
        compFor.compIter, [&](const auto& ptr) { return dump(*ptr); },
        [](std::monostate) -> std::string { PYLIR_UNREACHABLE; }));
    return result;
}

std::string pylir::Dumper::dump(const pylir::Syntax::Target& target)
{
    return pylir::match(
        target.variant,
        [&](const IdentifierToken& identifierToken) { return fmt::format("target {}", identifierToken.getValue()); },
        [&](const Syntax::Target::Parenth& parenth) -> std::string
        {
            if (!parenth.targetList)
            {
                return "target parenth empty";
            }
            return "target parenth" + addLastChild(dump(*parenth.targetList));
        },
        [&](const Syntax::Target::Square& square) -> std::string
        {
            if (!square.targetList)
            {
                return "target square empty";
            }
            return "target square" + addLastChild(dump(*square.targetList));
        },
        [&](const std::pair<BaseToken, std::unique_ptr<Syntax::Target>>& pair)
        { return "target starred" + addLastChild(dump(*pair.second)); },
        [&](const auto& value) { return dump(value); });
}

std::string pylir::Dumper::dump(const pylir::Syntax::SimpleStmt& simpleStmt)
{
    return pylir::match(simpleStmt.variant, [&](auto&& value) { return dump(value); });
}

std::string pylir::Dumper::dump(const pylir::Syntax::AssertStmt& assertStmt)
{
    std::string result = "assert statement";
    if (!assertStmt.message)
    {
        return result + addLastChild(dump(assertStmt.condition), "condition");
    }
    result += addMiddleChild(dump(assertStmt.condition), "condition");
    return result + addLastChild(dump(assertStmt.message->second), "message");
}

std::string pylir::Dumper::dump(const pylir::Syntax::PassStmt&)
{
    return "pass statement";
}

std::string pylir::Dumper::dump(const pylir::Syntax::DelStmt& delStmt)
{
    return "del statement" + addLastChild(dump(delStmt.targetList));
}

std::string pylir::Dumper::dump(const pylir::Syntax::ReturnStmt& returnStmt)
{
    if (!returnStmt.expressions)
    {
        return "return statement";
    }
    return "return statement" + addLastChild(dump(*returnStmt.expressions));
}

std::string pylir::Dumper::dump(const pylir::Syntax::YieldStmt& yieldStmt)
{
    return dump(yieldStmt.yieldExpression);
}

std::string pylir::Dumper::dump(const pylir::Syntax::RaiseStmt& raiseStmt)
{
    if (!raiseStmt.expressions)
    {
        return "raise statement";
    }
    if (!raiseStmt.expressions->second)
    {
        return "raise statement" + addLastChild(dump(raiseStmt.expressions->first), "exception");
    }
    return "raise statement" + addMiddleChild(dump(raiseStmt.expressions->first), "exception")
           + addLastChild(dump(raiseStmt.expressions->second->second), "expression");
}

std::string pylir::Dumper::dump(const pylir::Syntax::BreakStmt&)
{
    return "break statement";
}

std::string pylir::Dumper::dump(const pylir::Syntax::ContinueStmt&)
{
    return "continue statement";
}

std::string pylir::Dumper::dump(const pylir::Syntax::GlobalStmt& globalStmt)
{
    std::vector<std::string_view> identifiers{globalStmt.identifier.getValue()};
    std::transform(globalStmt.rest.begin(), globalStmt.rest.end(), std::back_inserter(identifiers),
                   [](const auto& pair) { return pair.second.getValue(); });
    return fmt::format(FMT_STRING("global {}"), fmt::join(identifiers, ", "));
}

std::string pylir::Dumper::dump(const pylir::Syntax::NonLocalStmt& nonLocalStmt)
{
    std::vector<std::string_view> identifiers{nonLocalStmt.identifier.getValue()};
    std::transform(nonLocalStmt.rest.begin(), nonLocalStmt.rest.end(), std::back_inserter(identifiers),
                   [](const auto& pair) { return pair.second.getValue(); });
    return fmt::format(FMT_STRING("nonlocal {}"), fmt::join(identifiers, ", "));
}

std::string pylir::Dumper::dump(const pylir::Syntax::ImportStmt& importStmt)
{
    auto dumpModule = [&](const Syntax::ImportStmt::Module& module)
    {
        std::vector<std::string_view> identifiers;
        std::transform(module.leading.begin(), module.leading.end(), std::back_inserter(identifiers),
                       [](const auto& pair) { return pair.first.getValue(); });
        identifiers.push_back(module.lastIdentifier.getValue());
        return fmt::format(FMT_STRING("module {}"), fmt::join(identifiers, "."));
    };

    auto dumpRelativeModule = [&](const Syntax::ImportStmt::RelativeModule& module)
    {
        auto dots = std::string(module.dots.size(), '.');
        if (!module.module)
        {
            return "relative module " + dots;
        }
        return "relative module " + dots + addLastChild(dumpModule(*module.module));
    };

    return pylir::match(
        importStmt.variant,
        [&](const Syntax::ImportStmt::ImportAsAs& importAsAs)
        {
            std::string result = "import";
            if (!importAsAs.name && importAsAs.rest.empty())
            {
                return result + addLastChild(dumpModule(importAsAs.module));
            }
            result += addMiddleChild(dumpModule(importAsAs.module));
            if (importAsAs.name && importAsAs.rest.empty())
            {
                return result + addLastChild(fmt::format("as {}", importAsAs.name->second.getValue()));
            }
            else
            {
                result += addMiddleChild(fmt::format("as {}", importAsAs.name->second.getValue()));
            }
            for (auto& further : tcb::span(importAsAs.rest).first(importAsAs.rest.size() - 1))
            {
                result += addMiddleChild(dumpModule(further.module));
                if (further.name)
                {
                    result += addMiddleChild(fmt::format("as {}", further.name->second.getValue()));
                }
            }
            if (!importAsAs.rest.back().name)
            {
                return result + addLastChild(dumpModule(importAsAs.rest.back().module));
            }
            result += addMiddleChild(dumpModule(importAsAs.rest.back().module));
            return result + addLastChild(fmt::format("as {}", importAsAs.rest.back().name->second.getValue()));
        },
        [&](const Syntax::ImportStmt::FromImportAll& importAll)
        { return "import all" + addLastChild(dumpRelativeModule(importAll.relativeModule)); },
        [&](const Syntax::ImportStmt::FromImportList& importList)
        {
            std::string result = "import list";
            result += addMiddleChild(dumpRelativeModule(importList.relativeModule));
            if (importList.rest.empty())
            {
                if (importList.name)
                {
                    return result
                           + addLastChild(fmt::format("{} as {}", importList.identifier.getValue(),
                                                      importList.identifier.getValue()));
                }
                return result + addLastChild(importList.identifier.getValue());
            }
            if (importList.name)
            {
                result += addMiddleChild(
                    fmt::format("{} as {}", importList.identifier.getValue(), importList.identifier.getValue()));
            }
            else
            {
                result += addMiddleChild(importList.identifier.getValue());
            }
            for (auto& further : tcb::span(importList.rest).first(importList.rest.size() - 1))
            {
                if (further.name)
                {
                    result += addMiddleChild(
                        fmt::format("{} as {}", further.identifier.getValue(), further.name->second.getValue()));
                }
                else
                {
                    result += addMiddleChild(further.identifier.getValue());
                }
            }
            if (importList.rest.back().name)
            {
                result += addLastChild(fmt::format("{} as {}", importList.rest.back().identifier.getValue(),
                                                   importList.rest.back().name->second.getValue()));
            }
            else
            {
                result += addLastChild(importList.rest.back().identifier.getValue());
            }
            return result;
        });
}

std::string pylir::Dumper::dump(const pylir::Syntax::FutureStmt& futureStmt)
{
    std::string result = "future list";
    if (futureStmt.rest.empty())
    {
        if (futureStmt.name)
        {
            return result
                   + addLastChild(
                       fmt::format("{} as {}", futureStmt.identifier.getValue(), futureStmt.identifier.getValue()));
        }
        return result + addLastChild(futureStmt.identifier.getValue());
    }
    if (futureStmt.name)
    {
        result +=
            addMiddleChild(fmt::format("{} as {}", futureStmt.identifier.getValue(), futureStmt.identifier.getValue()));
    }
    else
    {
        result += addMiddleChild(futureStmt.identifier.getValue());
    }
    for (auto& further : tcb::span(futureStmt.rest).first(futureStmt.rest.size() - 1))
    {
        if (further.name)
        {
            result +=
                addMiddleChild(fmt::format("{} as {}", further.identifier.getValue(), further.name->second.getValue()));
        }
        else
        {
            result += addMiddleChild(further.identifier.getValue());
        }
    }
    if (futureStmt.rest.back().name)
    {
        result += addLastChild(fmt::format("{} as {}", futureStmt.rest.back().identifier.getValue(),
                                           futureStmt.rest.back().name->second.getValue()));
    }
    else
    {
        result += addLastChild(futureStmt.rest.back().identifier.getValue());
    }
    return result;
}

std::string pylir::Dumper::dump(const pylir::Syntax::AssignmentStmt& assignmentStmt)
{
    std::string result = "assignment statement";
    for (auto& iter : assignmentStmt.targets)
    {
        result += addMiddleChild(dump(iter.first));
    }
    return result + addLastChild(pylir::match(assignmentStmt.variant, [&](auto&& value) { return dump(value); }));
}

std::string pylir::Dumper::dump(const pylir::Syntax::YieldExpression& yieldExpression)
{
    return pylir::match(
        yieldExpression.variant, [](std::monostate) -> std::string { return "yield empty"; },
        [&](const std::pair<BaseToken, Syntax::Expression>& expression) -> std::string
        {
            std::string result = "yield from";
            result += addLastChild(dump(expression.second));
            return result;
        },
        [&](const Syntax::ExpressionList& list) -> std::string
        {
            std::string result = "yield list";
            if (list.remainingExpr.empty())
            {
                result += addLastChild(dump(*list.firstExpr));
            }
            else
            {
                result += addMiddleChild(dump(*list.firstExpr));
                for (auto& iter : tcb::span(list.remainingExpr).first(list.remainingExpr.size() - 1))
                {
                    result += addMiddleChild(dump(*iter.second));
                }
                result += addLastChild(dump(*list.remainingExpr.back().second));
            }
            return result;
        });
}

std::string pylir::Dumper::dump(const pylir::Syntax::AugmentedAssignmentStmt& augmentedAssignmentStmt)
{
    std::string result = fmt::format("augmented assignment {:q}", augmentedAssignmentStmt.augOp.getTokenType());
    result += addMiddleChild(dump(augmentedAssignmentStmt.augTarget));
    result +=
        addLastChild(pylir::match(augmentedAssignmentStmt.variant, [&](const auto& value) { return dump(value); }));
    return result;
}

std::string pylir::Dumper::dump(const pylir::Syntax::AugTarget& augTarget)
{
    return pylir::match(
        augTarget.variant,
        [&](const IdentifierToken& identifierToken) { return "augtarget " + std::string(identifierToken.getValue()); },
        [&](const auto& value) { return dump(value); });
}

std::string pylir::Dumper::dump(const pylir::Syntax::AnnotatedAssignmentSmt& annotatedAssignmentSmt)
{
    std::string result = "annotated assignment";
    result += addMiddleChild(dump(annotatedAssignmentSmt.augTarget));
    if (!annotatedAssignmentSmt.optionalAssignmentStmt)
    {
        return result + addLastChild(dump(annotatedAssignmentSmt.expression));
    }
    result += addMiddleChild(dump(annotatedAssignmentSmt.expression));
    result += addLastChild(pylir::match(annotatedAssignmentSmt.optionalAssignmentStmt->second,
                                        [&](const auto& value) { return dump(value); }));
    return result;
}
