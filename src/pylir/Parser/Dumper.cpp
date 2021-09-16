#include "Dumper.hpp"

#include <llvm/ADT/SmallString.h>

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

std::string dumpVariables(const pylir::IdentifierSet& tokens)
{
    std::vector<std::string> variables(tokens.size());
    std::transform(tokens.begin(), tokens.end(), variables.begin(),
                   [](const pylir::IdentifierToken& token) { return std::string(token.getValue()); });
    std::sort(variables.begin(), variables.end());
    PYLIR_ASSERT(!variables.empty());
    std::string text = variables[0];
    for (std::size_t i = 1; i < variables.size(); i++)
    {
        text += ", ";
        text += variables[i];
    }
    return text;
}
} // namespace

std::string pylir::Dumper::Builder::addLastChild(std::string_view lastChildDump,
                                                 std::optional<std::string_view>&& label)
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

std::string pylir::Dumper::Builder::addMiddleChild(std::string_view middleChildDump,
                                                   std::optional<std::string_view>&& label)
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

std::string pylir::Dumper::Builder::emit() const
{
    if (m_children.empty())
    {
        return m_title;
    }
    auto result = m_title;
    for (auto& iter : tcb::span(m_children).first(m_children.size() - 1))
    {
        result += addMiddleChild(iter.first, iter.second);
    }
    return result + addLastChild(m_children.back().first, m_children.back().second);
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
                [](const BigInt& bigInt) -> std::string { return fmt::format("atom {}", bigInt.toString()); },
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
                [&](std::monostate) -> std::string
                {
                    switch (literal.token.getTokenType())
                    {
                        case TokenType::NoneKeyword: return "atom None";
                        case TokenType::TrueKeyword: return "atom True";
                        case TokenType::FalseKeyword: return "atom False";
                        default: PYLIR_UNREACHABLE;
                    }
                });
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

            return createBuilder("dict display")
                .add(pylir::match(
                    dictDisplay.variant, [](std::monostate) -> std::string { PYLIR_UNREACHABLE; },
                    [&](const Syntax::Enclosure::DictDisplay::DictComprehension& comprehension) -> std::string
                    {
                        return createBuilder("dict comprehension")
                            .add(comprehension.first)
                            .add(comprehension.second)
                            .add(comprehension.compFor)
                            .emit();
                    },
                    [&](const Syntax::CommaList<Syntax::Enclosure::DictDisplay::KeyDatum>& commaList) -> std::string
                    {
                        return dump(
                            commaList,
                            [&](const Syntax::Enclosure::DictDisplay::KeyDatum& keyDatum) -> std::string
                            {
                                return pylir::match(
                                    keyDatum.variant,
                                    [&](const Syntax::Enclosure::DictDisplay::KeyDatum::Key& key) -> std::string
                                    { return createBuilder("key").add(key.first).add(key.second).emit(); },
                                    [&](const Syntax::Enclosure::DictDisplay::KeyDatum::Datum& datum) -> std::string
                                    { return createBuilder("datum").add(datum.orExpr).emit(); });
                            },
                            "key datum list");
                    }))
                .emit();
        },
        [&](const Syntax::Enclosure::SetDisplay& setDisplay) -> std::string
        {
            auto result = createBuilder("set display");
            pylir::match(setDisplay.variant, [&](const auto& value) { result.add(value); });
            return result.emit();
        },
        [&](const Syntax::Enclosure::ListDisplay& listDisplay) -> std::string
        {
            if (std::holds_alternative<std::monostate>(listDisplay.variant))
            {
                return "list display empty";
            }
            auto result = createBuilder("list display");
            pylir::match(
                listDisplay.variant, [](const std::monostate&) { PYLIR_UNREACHABLE; },
                [&](const auto& value) { result.add(value); });
            return result.emit();
        },
        [&](const Syntax::Enclosure::GeneratorExpression& generatorExpression) -> std::string
        {
            return createBuilder("generator expression")
                .add(generatorExpression.expression)
                .add(generatorExpression.compFor)
                .emit();
        },
        [&](const Syntax::Enclosure::YieldAtom& yieldAtom) -> std::string
        { return createBuilder("yieldatom").add(yieldAtom.yieldExpression).emit(); },
        [&](const Syntax::Enclosure::ParenthForm& parenthForm) -> std::string
        {
            if (!parenthForm.expression)
            {
                return "parenth empty";
            }
            else
            {
                return createBuilder("parenth").add(*parenthForm.expression).emit();
            }
        });
}

std::string pylir::Dumper::dump(const pylir::Syntax::Primary& primary)
{
    return pylir::match(primary.variant, [&](const auto& value) -> std::string { return dump(value); });
}

std::string pylir::Dumper::dump(const pylir::Syntax::AttributeRef& attribute)
{
    return createBuilder("attribute {}", attribute.identifier.getValue()).add(*attribute.primary).emit();
}

std::string pylir::Dumper::dump(const pylir::Syntax::Subscription& subscription)
{
    return createBuilder("subscription")
        .add(*subscription.primary, "primary")
        .add(subscription.expressionList, "index")
        .emit();
}

std::string pylir::Dumper::dump(const pylir::Syntax::Slicing& slicing)
{
    return createBuilder("slicing")
        .add(*slicing.primary, "primary")
        .add(dump(
                 slicing.sliceList,
                 [&](const auto& slice) -> std::string
                 {
                     return pylir::match(
                         slice,
                         [&](const Syntax::Slicing::ProperSlice& properSlice) -> std::string
                         {
                             auto builder = createBuilder("proper slice");
                             if (properSlice.optionalLowerBound)
                             {
                                 builder.add(*properSlice.optionalLowerBound, "lowerBound");
                             }
                             if (properSlice.optionalUpperBound)
                             {
                                 builder.add(*properSlice.optionalUpperBound, "upperBound");
                             }
                             if (properSlice.optionalStride)
                             {
                                 builder.add(*properSlice.optionalStride, "stride");
                             }
                             return builder.emit();
                         },
                         [&](const Syntax::Expression& expression) { return dump(expression); });
                 },
                 "proper slice list"),
             "index")
        .emit();
}

std::string pylir::Dumper::dump(const pylir::Syntax::Comprehension& comprehension)
{
    return createBuilder("comprehension").add(comprehension.assignmentExpression).add(comprehension.compFor).emit();
}

std::string pylir::Dumper::dump(const pylir::Syntax::AssignmentExpression& assignmentExpression)
{
    if (!assignmentExpression.identifierAndWalrus)
    {
        return dump(*assignmentExpression.expression);
    }
    return createBuilder("assignment expression to {}", assignmentExpression.identifierAndWalrus->first.getValue())
        .add(*assignmentExpression.expression)
        .emit();
}

std::string pylir::Dumper::dump(const pylir::Syntax::ArgumentList& argument)
{
    auto builder = createBuilder("argument list");
    if (argument.positionalArguments)
    {
        auto positional =
            createBuilder("positional arguments")
                .add(pylir::match(
                    argument.positionalArguments->firstItem.variant,
                    [&](const std::unique_ptr<Syntax::AssignmentExpression>& value) { return dump(*value); },
                    [&](const Syntax::ArgumentList::PositionalItem::Star& star)
                    { return createBuilder("starred").add(*star.expression).emit(); }));
        for (auto& [token, item] : argument.positionalArguments->rest)
        {
            positional.add(pylir::match(
                item.variant, [&](const std::unique_ptr<Syntax::AssignmentExpression>& value) { return dump(*value); },
                [&](const Syntax::ArgumentList::PositionalItem::Star& star)
                { return createBuilder("starred").add(*star.expression).emit(); }));
        }
        builder.add(positional);
    }
    if (argument.starredAndKeywords)
    {
        auto starred = createBuilder("starred keywords");
        auto keyword = createBuilder("keyword item {}", argument.starredAndKeywords->first.identifier.getValue())
                           .add(*argument.starredAndKeywords->first.expression);
        starred.add(keyword);
        for (auto& [token, iter] : argument.starredAndKeywords->rest)
        {
            starred.add(pylir::match(
                iter,
                [&](const Syntax::ArgumentList::KeywordItem& keywordItem) {
                    return createBuilder("keyword item {}", keywordItem.identifier.getValue())
                        .add(*keywordItem.expression)
                        .emit();
                },
                [&](const Syntax::ArgumentList::StarredAndKeywords::Expression& expression)
                { return createBuilder("starred expression").add(*expression.expression).emit(); }));
        }
        builder.add(starred);
    }
    if (argument.keywordArguments)
    {
        auto starred = createBuilder("keyword arguments");
        auto mapped = createBuilder("mapped expression").add(*argument.keywordArguments->first.expression);
        starred.add(mapped);
        for (auto& [token, iter] : argument.keywordArguments->rest)
        {
            starred.add(pylir::match(
                iter,
                [&](const Syntax::ArgumentList::KeywordItem& keywordItem) {
                    return createBuilder("keyword item {}", keywordItem.identifier.getValue())
                        .add(*keywordItem.expression)
                        .emit();
                },
                [&](const Syntax::ArgumentList::KeywordArguments::Expression& expression)
                { return createBuilder("mapped expression").add(*expression.expression).emit(); }));
        }
        builder.add(starred);
    }
    return builder.emit();
}

std::string pylir::Dumper::dump(const pylir::Syntax::Call& call)
{
    auto builder = createBuilder("call");
    builder.add(*call.primary);
    if (std::holds_alternative<std::monostate>(call.variant))
    {
        return builder.emit();
    }
    if (auto* comprehension = std::get_if<std::unique_ptr<Syntax::Comprehension>>(&call.variant))
    {
        return builder.add(**comprehension).emit();
    }
    auto& [argument, comma] = pylir::get<std::pair<Syntax::ArgumentList, std::optional<BaseToken>>>(call.variant);
    return builder.add(argument).emit();
}

std::string pylir::Dumper::dump(const pylir::Syntax::AwaitExpr& awaitExpr)
{
    return createBuilder("await expression").add(awaitExpr.primary).emit();
}

std::string pylir::Dumper::dump(const pylir::Syntax::UExpr& uExpr)
{
    if (auto* power = std::get_if<Syntax::Power>(&uExpr.variant))
    {
        return dump(*power);
    }
    auto& [token, rhs] = pylir::get<std::pair<Token, std::unique_ptr<Syntax::UExpr>>>(uExpr.variant);
    return createBuilder(FMT_STRING("unary {:q}"), token.getTokenType()).add(*rhs).emit();
}

std::string pylir::Dumper::dump(const pylir::Syntax::Power& power)
{
    if (!power.rightHand)
    {
        return pylir::match(power.variant, [&](const auto& value) { return dump(value); });
    }
    return createBuilder("power")
        .add(pylir::match(power.variant, [&](const auto& value) { return dump(value); }), "base")
        .add(*power.rightHand->second, "exponent")
        .emit();
}

std::string pylir::Dumper::dump(const pylir::Syntax::MExpr& mExpr)
{
    return pylir::match(
        mExpr.variant, [&](const Syntax::UExpr& uExpr) { return dump(uExpr); },
        [&](const std::unique_ptr<Syntax::MExpr::BinOp>& binOp)
        {
            return createBuilder("mexpr {:q}", binOp->binToken.getTokenType())
                .add(*binOp->lhs, "lhs")
                .add(binOp->rhs, "rhs")
                .emit();
        },
        [&](const std::unique_ptr<Syntax::MExpr::AtBin>& binOp) {
            return createBuilder("mexpr {:q}", TokenType::AtSign)
                .add(*binOp->lhs, "lhs")
                .add(*binOp->rhs, "rhs")
                .emit();
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
    auto result = createBuilder("comparison");
    result.add(comparison.left, "lhs");
    for (auto& [token, rhs] : comparison.rest)
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
        result.add(rhs, name);
    }
    return result.emit();
}

std::string pylir::Dumper::dump(const pylir::Syntax::NotTest& notTest)
{
    return pylir::match(
        notTest.variant, [&](const Syntax::Comparison& comparison) { return dump(comparison); },
        [&](const auto& pair) { return createBuilder("notTest").add(*pair.second).emit(); });
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
    return createBuilder("conditional expression")
        .add(conditionalExpression.value, "value")
        .add(*conditionalExpression.suffix->test, "condition")
        .add(*conditionalExpression.suffix->elseValue, "elseValue")
        .emit();
}

std::string pylir::Dumper::dump(const pylir::Syntax::LambdaExpression& lambdaExpression)
{
    auto builder = createBuilder("lambda expression");
    if (lambdaExpression.parameterList)
    {
        builder.add(*lambdaExpression.parameterList);
    }
    return builder.add(lambdaExpression.expression).emit();
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
    return createBuilder("starred item")
        .add(pylir::get<std::pair<BaseToken, Syntax::OrExpr>>(starredItem.variant).second)
        .emit();
}

std::string pylir::Dumper::dump(const pylir::Syntax::StarredExpression& starredExpression)
{
    if (auto* expression = std::get_if<Syntax::Expression>(&starredExpression.variant))
    {
        return dump(*expression);
    }
    auto& item = pylir::get<Syntax::StarredExpression::Items>(starredExpression.variant);
    auto result = createBuilder("starred expression");
    for (auto& iter : item.leading)
    {
        result.add(iter.first);
    }
    if (item.last)
    {
        result.add(*item.last);
    }
    return result.emit();
}

std::string pylir::Dumper::dump(const pylir::Syntax::CompIf& compIf)
{
    auto result = createBuilder("comp if");
    return pylir::match(
        compIf.compIter, [&](std::monostate) { return result.add(compIf.orTest, "condition").emit(); },
        [&](const Syntax::CompFor& compFor) { return result.add(compIf.orTest, "condition").add(compFor).emit(); },
        [&](const std::unique_ptr<Syntax::CompIf>& second)
        { return result.add(compIf.orTest, "condition").add(*second).emit(); });
}

std::string pylir::Dumper::dump(const pylir::Syntax::CompFor& compFor)
{
    std::string title;
    if (compFor.awaitToken)
    {
        title = "comp for await";
    }
    else
    {
        title = "comp for";
    }
    auto builder = createBuilder("{}", title).add(compFor.targets);
    if (std::holds_alternative<std::monostate>(compFor.compIter))
    {
        return builder.add(compFor.orTest).emit();
    }
    builder.add(compFor.orTest);
    builder.add(pylir::match(
        compFor.compIter, [&](const auto& ptr) { return dump(*ptr); },
        [](std::monostate) -> std::string { PYLIR_UNREACHABLE; }));
    return builder.emit();
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
            return createBuilder("target parenth").add(*parenth.targetList).emit();
        },
        [&](const Syntax::Target::Square& square) -> std::string
        {
            if (!square.targetList)
            {
                return "target square empty";
            }
            return createBuilder("target square").add(*square.targetList).emit();
        },
        [&](const std::pair<BaseToken, std::unique_ptr<Syntax::Target>>& pair)
        { return createBuilder("target starred").add(*pair.second).emit(); },
        [&](const auto& value) { return dump(value); });
}

std::string pylir::Dumper::dump(const pylir::Syntax::SimpleStmt& simpleStmt)
{
    return pylir::match(simpleStmt.variant, [&](auto&& value) { return dump(value); });
}

std::string pylir::Dumper::dump(const pylir::Syntax::AssertStmt& assertStmt)
{
    auto result = createBuilder("assert statement").add(assertStmt.condition, "condition");
    if (assertStmt.message)
    {
        result.add(assertStmt.message->second, "message");
    }
    return result.emit();
}

std::string pylir::Dumper::dump(const pylir::Syntax::PassStmt&)
{
    return "pass statement";
}

std::string pylir::Dumper::dump(const pylir::Syntax::DelStmt& delStmt)
{
    return createBuilder("del statement").add(delStmt.targetList).emit();
}

std::string pylir::Dumper::dump(const pylir::Syntax::ReturnStmt& returnStmt)
{
    if (!returnStmt.expressions)
    {
        return "return statement";
    }
    return createBuilder("return statement").add(*returnStmt.expressions).emit();
}

std::string pylir::Dumper::dump(const pylir::Syntax::YieldStmt& yieldStmt)
{
    return dump(yieldStmt.yieldExpression);
}

std::string pylir::Dumper::dump(const pylir::Syntax::RaiseStmt& raiseStmt)
{
    auto builder = createBuilder("raise statement");
    if (raiseStmt.expressions)
    {
        builder.add(raiseStmt.expressions->first, "exception");
        if (raiseStmt.expressions->second)
        {
            builder.add(raiseStmt.expressions->second->second, "expression");
        }
    }
    return builder.emit();
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
        auto builder = createBuilder("relative module {}", dots);
        if (!module.module)
        {
            builder.add(dumpModule(*module.module));
        }
        return builder.emit();
    };

    return pylir::match(
        importStmt.variant,
        [&](const Syntax::ImportStmt::ImportAsAs& importAsAs)
        {
            auto result = createBuilder("import");
            result.add(dumpModule(importAsAs.module));
            if (importAsAs.name)
            {
                result.add(fmt::format("as {}", importAsAs.name->second.getValue()));
            }
            for (auto& further : importAsAs.rest)
            {
                result.add(dumpModule(further.module));
                if (further.name)
                {
                    result.add(fmt::format("as {}", further.name->second.getValue()));
                }
            }
            return result.emit();
        },
        [&](const Syntax::ImportStmt::FromImportAll& importAll)
        { return createBuilder("import all").add(dumpRelativeModule(importAll.relativeModule)).emit(); },
        [&](const Syntax::ImportStmt::FromImportList& importList)
        {
            auto result = createBuilder("import list");
            result.add(dumpRelativeModule(importList.relativeModule));
            if (importList.name)
            {
                result.add(fmt::format("{} as {}", importList.identifier.getValue(), importList.identifier.getValue()));
            }
            else
            {
                result.add(importList.identifier.getValue());
            }
            for (auto& further : importList.rest)
            {
                if (further.name)
                {
                    result.add(fmt::format("{} as {}", further.identifier.getValue(), further.name->second.getValue()));
                }
                else
                {
                    result.add(further.identifier.getValue());
                }
            }
            return result.emit();
        });
}

std::string pylir::Dumper::dump(const pylir::Syntax::FutureStmt& futureStmt)
{
    auto result = createBuilder("future list");
    if (futureStmt.name)
    {
        result.add(fmt::format("{} as {}", futureStmt.identifier.getValue(), futureStmt.identifier.getValue()));
    }
    else
    {
        result.add(futureStmt.identifier.getValue());
    }
    for (auto& further : futureStmt.rest)
    {
        if (further.name)
        {
            result.add(fmt::format("{} as {}", further.identifier.getValue(), further.name->second.getValue()));
        }
        else
        {
            result.add(further.identifier.getValue());
        }
    }
    return result.emit();
}

std::string pylir::Dumper::dump(const pylir::Syntax::AssignmentStmt& assignmentStmt)
{
    auto result = createBuilder("assignment statement");
    for (auto& iter : assignmentStmt.targets)
    {
        result.add(iter.first);
    }
    return result.add(pylir::match(assignmentStmt.variant, [&](auto&& value) { return dump(value); })).emit();
}

std::string pylir::Dumper::dump(const pylir::Syntax::YieldExpression& yieldExpression)
{
    return pylir::match(
        yieldExpression.variant, [](std::monostate) -> std::string { return "yield empty"; },
        [&](const std::pair<BaseToken, Syntax::Expression>& expression) -> std::string
        { return createBuilder("yield from").add(expression.second).emit(); },
        [&](const Syntax::ExpressionList& list) -> std::string
        {
            auto result = createBuilder("yield list").add(*list.firstExpr);
            for (auto& iter : list.remainingExpr)
            {
                result.add(*iter.second);
            }
            return result.emit();
        });
}

std::string pylir::Dumper::dump(const pylir::Syntax::AugmentedAssignmentStmt& augmentedAssignmentStmt)
{
    return createBuilder("augmented assignment {:q}", augmentedAssignmentStmt.augOp.getTokenType())
        .add(augmentedAssignmentStmt.augTarget)
        .add(pylir::match(augmentedAssignmentStmt.variant, [&](const auto& value) { return dump(value); }))
        .emit();
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
    auto result = createBuilder("annotated assignment")
                      .add(annotatedAssignmentSmt.augTarget)
                      .add(annotatedAssignmentSmt.expression);
    if (annotatedAssignmentSmt.optionalAssignmentStmt)
    {
        result.add(pylir::match(annotatedAssignmentSmt.optionalAssignmentStmt->second,
                                [&](const auto& value) { return dump(value); }));
    }
    return result.emit();
}

std::string pylir::Dumper::dump(const pylir::Syntax::FileInput& fileInput)
{
    auto builder = createBuilder("file input");
    if (!fileInput.globals.empty())
    {
        builder.add(dumpVariables(fileInput.globals), "globals");
    }
    for (auto& iter : fileInput.input)
    {
        if (auto* statement = std::get_if<Syntax::Statement>(&iter))
        {
            builder.add(*statement);
        }
    }
    return builder.emit();
}

std::string pylir::Dumper::dump(const pylir::Syntax::Statement& statement)
{
    return pylir::match(
        statement.variant, [&](const Syntax::CompoundStmt& compoundStmt) { return dump(compoundStmt); },
        [&](const Syntax::Statement::SingleLine& singleLine) { return dump(singleLine.stmtList); });
}

std::string pylir::Dumper::dump(const pylir::Syntax::CompoundStmt& compoundStmt)
{
    return pylir::match(compoundStmt.variant, [&](const auto& value) { return dump(value); });
}

std::string pylir::Dumper::dump(const pylir::Syntax::StmtList& stmtList)
{
    return dump(
        stmtList, [&](const auto& value) { return dump(value); }, "stmt list");
}

std::string pylir::Dumper::dump(const pylir::Syntax::Suite& suite)
{
    return pylir::match(
        suite.variant,
        [&](const Syntax::Suite::MultiLine& multiLine)
        {
            auto builder = createBuilder("suite");
            for (auto& iter : multiLine.statements)
            {
                builder.add(iter);
            }
            return builder.emit();
        },
        [&](const Syntax::Suite::SingleLine& singleLine) { return dump(singleLine.stmtList); });
}

std::string pylir::Dumper::dump(const pylir::Syntax::IfStmt& ifStmt)
{
    auto builder = createBuilder("if stmt").add(ifStmt.condition).add(*ifStmt.suite);
    for (auto& iter : ifStmt.elifs)
    {
        builder.add(iter.condition).add(*iter.suite);
    }
    if (ifStmt.elseSection)
    {
        builder.add(*ifStmt.elseSection->suite);
    }
    return builder.emit();
}

std::string pylir::Dumper::dump(const pylir::Syntax::WhileStmt& whileStmt)
{
    auto builder = createBuilder("while stmt").add(whileStmt.condition).add(*whileStmt.suite);
    if (whileStmt.elseSection)
    {
        builder.add(*whileStmt.elseSection->suite);
    }
    return builder.emit();
}

std::string pylir::Dumper::dump(const pylir::Syntax::ForStmt& forStmt)
{
    auto builder = createBuilder("for stmt").add(forStmt.targetList).add(forStmt.expressionList).add(*forStmt.suite);
    if (forStmt.elseSection)
    {
        builder.add(*forStmt.elseSection->suite);
    }
    return builder.emit();
}

std::string pylir::Dumper::dump(const pylir::Syntax::TryStmt& tryStmt)
{
    auto builder = createBuilder("try stmt").add(*tryStmt.suite);
    for (auto& iter : tryStmt.excepts)
    {
        auto except = createBuilder("except");
        if (iter.expression)
        {
            except.add(iter.expression->first);
            if (iter.expression->second)
            {
                except.add(fmt::format("as {}", iter.expression->second->second.getValue()));
            }
        }
        builder.add(except.add(*iter.suite));
    }
    if (tryStmt.elseSection)
    {
        builder.add(*tryStmt.elseSection->suite, "else");
    }
    if (tryStmt.finally)
    {
        builder.add(*tryStmt.finally->suite, "finally");
    }
    return builder.emit();
}

std::string pylir::Dumper::dump(const pylir::Syntax::WithStmt& withStmt)
{
    auto builder = createBuilder("with stmt");
    auto dumpWithItem = [&](const Syntax::WithStmt::WithItem& item)
    {
        auto builder = createBuilder("with item").add(item.expression);
        if (item.target)
        {
            builder.add(item.target->second, "as");
        }
        return builder.emit();
    };
    builder.add(dumpWithItem(withStmt.first));
    for (auto& [token, item] : withStmt.rest)
    {
        builder.add(dumpWithItem(item));
    }
    builder.add(*withStmt.suite);
    return builder.emit();
}

std::string pylir::Dumper::dump(const pylir::Syntax::ParameterList& parameterList)
{
    auto builder = createBuilder("parameter list");

    auto dumpParameter = [&](const Syntax::ParameterList::Parameter& parameter)
    {
        auto builder = createBuilder("parameter {}", parameter.identifier.getValue());
        if (parameter.type)
        {
            builder.add(parameter.type->second);
        }
        return builder.emit();
    };

    auto dumpDefParameter = [&](const Syntax::ParameterList::DefParameter& defParameter)
    {
        if (!defParameter.defaultArg)
        {
            return dumpParameter(defParameter.parameter);
        }
        return createBuilder("def parameter")
            .add(dumpParameter(defParameter.parameter))
            .add(defParameter.defaultArg->second)
            .emit();
    };

    auto dumpParameterListStarArgs = [&](const Syntax::ParameterList::StarArgs& list)
    {
        auto builder = createBuilder("parameter list star args");
        pylir::match(
            list.variant,
            [&](const Syntax::ParameterList::StarArgs::DoubleStar& doubleStar)
            { builder.add(dumpParameter(doubleStar.parameter), "double starred"); },
            [&](const Syntax::ParameterList::StarArgs::Star& star)
            {
                auto temp = createBuilder("starred");
                if (star.parameter)
                {
                    temp.add(dumpParameter(*star.parameter), "starred");
                }
                for (auto& [token, item] : star.defParameters)
                {
                    temp.add(dumpDefParameter(item));
                }
                if (star.further && star.further->doubleStar)
                {
                    temp.add(dumpParameter(star.further->doubleStar->parameter), "double starred");
                }
                builder.add(temp);
            });
        return builder.emit();
    };
    auto dumpParameterListNoPosOnly = [&](const Syntax::ParameterList::NoPosOnly& list)
    {
        return pylir::match(
            list.variant,
            [&](const Syntax::ParameterList::StarArgs& starArgs) { return dumpParameterListStarArgs(starArgs); },
            [&](const Syntax::ParameterList::NoPosOnly::DefParams& defParams)
            {
                auto builder = createBuilder("no pos only").add(dumpDefParameter(defParams.first));
                for (auto& [token, item] : defParams.rest)
                {
                    builder.add(dumpDefParameter(item));
                }
                if (defParams.suffix && defParams.suffix->second)
                {
                    builder.add(dumpParameterListStarArgs(*defParams.suffix->second));
                }
                return builder.emit();
            });
    };

    return pylir::match(
        parameterList.variant,
        [&](const Syntax::ParameterList::NoPosOnly& noPosOnly) { return dumpParameterListNoPosOnly(noPosOnly); },
        [&](const Syntax::ParameterList::PosOnly& posOnly)
        {
            auto builder = createBuilder("pos only").add(dumpDefParameter(posOnly.first));
            for (auto& [token, iter] : posOnly.rest)
            {
                builder.add(dumpDefParameter(iter));
            }
            if (posOnly.suffix && posOnly.suffix->second)
            {
                builder.add(dumpParameterListNoPosOnly(*posOnly.suffix->second));
            }
            return builder.emit();
        });
}

std::string pylir::Dumper::dump(const pylir::Syntax::Decorator& decorator)
{
    return createBuilder("decorator").add(decorator.assignmentExpression).emit();
}

std::string pylir::Dumper::dump(const pylir::Syntax::FuncDef& funcDef)
{
    std::string title;
    if (funcDef.async)
    {
        title = fmt::format("async function {}", funcDef.funcName.getValue());
    }
    else
    {
        title = fmt::format("function {}", funcDef.funcName.getValue());
    }
    auto builder = createBuilder("{}", title);
    for (auto& iter : funcDef.decorators)
    {
        builder.add(iter);
    }
    if (funcDef.parameterList)
    {
        builder.add(*funcDef.parameterList);
    }
    if (funcDef.suffix)
    {
        builder.add(funcDef.suffix->second, "suffix");
    }
    if (!funcDef.localVariables.empty())
    {
        builder.add(dumpVariables(funcDef.localVariables), "locals");
    }
    if (!funcDef.nonLocalVariables.empty())
    {
        builder.add(dumpVariables(funcDef.nonLocalVariables), "nonlocals");
    }
    return builder.add(*funcDef.suite).emit();
}

std::string pylir::Dumper::dump(const Syntax::ClassDef& classDef)
{
    auto builder = createBuilder("class {}", classDef.className.getValue());
    for (auto& iter : classDef.decorators)
    {
        builder.add(iter);
    }
    if (classDef.inheritance && classDef.inheritance->argumentList)
    {
        builder.add(*classDef.inheritance->argumentList);
    }
    if (!classDef.localVariables.empty())
    {
        builder.add(dumpVariables(classDef.localVariables), "locals");
    }
    if (!classDef.nonLocalVariables.empty())
    {
        builder.add(dumpVariables(classDef.nonLocalVariables), "nonlocals");
    }
    builder.add(*classDef.suite);
    return builder.emit();
}

std::string pylir::Dumper::dump(const pylir::Syntax::AsyncForStmt& asyncForStmt)
{
    return createBuilder("async for").add(asyncForStmt.forStmt).emit();
}

std::string pylir::Dumper::dump(const pylir::Syntax::AsyncWithStmt& asyncWithStmt)
{
    return createBuilder("async with").add(asyncWithStmt.withStmt).emit();
}
