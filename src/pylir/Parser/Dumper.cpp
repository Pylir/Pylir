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

std::string pylir::Dumper::addLastChild(std::string_view lastChildDump)
{
    auto lines = splitLines(lastChildDump);
    std::string result;
    bool first = true;
    for (auto iter : lines)
    {
        if (first)
        {
            first = false;
            result += "\n`-" + std::string(iter);
        }
        else
        {
            result += "\n  " + std::string(iter);
        }
    }
    return result;
}

std::string pylir::Dumper::addMiddleChild(std::string_view middleChildDump)
{
    auto lines = splitLines(middleChildDump);
    std::string result;
    bool first = true;
    for (auto iter : lines)
    {
        if (first)
        {
            first = false;
            result += "\n|-" + std::string(iter);
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
        [](const Syntax::Atom::Identifier& identifier) -> std::string
        { return fmt::format("atom {}", pylir::get<std::string>(identifier.token.getValue())); },
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
        [](const auto&) -> std::string
        {
            // TODO:
            PYLIR_UNREACHABLE;
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
        {
            return pylir::match(
                yieldAtom.variant, [](std::monostate) -> std::string { return "yield empty"; },
                [&](const std::pair<Token, Syntax::Expression>& expression) -> std::string
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
        },
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
    return std::string();
}

std::string pylir::Dumper::dump(const pylir::Syntax::Subscription& subscription)
{
    return std::string();
}

std::string pylir::Dumper::dump(const pylir::Syntax::Slicing& slicing)
{
    return std::string();
}

std::string pylir::Dumper::dump(const pylir::Syntax::Comprehension& comprehension)
{
    return std::string();
}

std::string pylir::Dumper::dump(const pylir::Syntax::AssignmentExpression& assignmentExpression)
{
    if (!assignmentExpression.identifierAndWalrus)
    {
        return dump(*assignmentExpression.expression);
    }
    return std::string();
}

std::string pylir::Dumper::dump(const pylir::Syntax::Call& call)
{
    return std::string();
}

std::string pylir::Dumper::dump(const pylir::Syntax::AwaitExpr& awaitExpr)
{
    return std::string();
}

std::string pylir::Dumper::dump(const pylir::Syntax::UExpr& uExpr)
{
    if (auto* power = std::get_if<Syntax::Power>(&uExpr.variant))
    {
        return dump(*power);
    }
    return std::string();
}

std::string pylir::Dumper::dump(const pylir::Syntax::Power& power)
{
    if (!power.rightHand)
    {
        return pylir::match(power.variant, [&](const auto& value) { return dump(value); });
    }
    return std::string();
}

std::string pylir::Dumper::dump(const pylir::Syntax::MExpr& mExpr)
{
    if (auto* uExpr = std::get_if<Syntax::UExpr>(&mExpr.variant))
    {
        return dump(*uExpr);
    }
    return std::string();
}

std::string pylir::Dumper::dump(const pylir::Syntax::AExpr& aExpr)
{
    if (auto* mExpr = std::get_if<Syntax::MExpr>(&aExpr.variant))
    {
        return dump(*mExpr);
    }
    return std::string();
}

std::string pylir::Dumper::dump(const pylir::Syntax::ShiftExpr& shiftExpr)
{
    if (auto* aExpr = std::get_if<Syntax::AExpr>(&shiftExpr.variant))
    {
        return dump(*aExpr);
    }
    return std::string();
}

std::string pylir::Dumper::dump(const pylir::Syntax::AndExpr& andExpr)
{
    if (auto* shiftExpr = std::get_if<Syntax::ShiftExpr>(&andExpr.variant))
    {
        return dump(*shiftExpr);
    }
    return std::string();
}

std::string pylir::Dumper::dump(const pylir::Syntax::XorExpr& xorExpr)
{
    if (auto* andExpr = std::get_if<Syntax::AndExpr>(&xorExpr.variant))
    {
        return dump(*andExpr);
    }
    return std::string();
}

std::string pylir::Dumper::dump(const pylir::Syntax::OrExpr& orExpr)
{
    if (auto* xorExpr = std::get_if<Syntax::XorExpr>(&orExpr.variant))
    {
        return dump(*xorExpr);
    }
    return std::string();
}

std::string pylir::Dumper::dump(const pylir::Syntax::Comparison& comparison)
{
    if (comparison.rest.empty())
    {
        return dump(comparison.left);
    }
    return std::string();
}

std::string pylir::Dumper::dump(const pylir::Syntax::NotTest& notTest)
{
    if (auto* comparison = std::get_if<Syntax::Comparison>(&notTest.variant))
    {
        return dump(*comparison);
    }
    return std::string();
}

std::string pylir::Dumper::dump(const pylir::Syntax::AndTest& andTest)
{
    if (auto* notTest = std::get_if<Syntax::NotTest>(&andTest.variant))
    {
        return dump(*notTest);
    }
    return std::string();
}

std::string pylir::Dumper::dump(const pylir::Syntax::OrTest& orTest)
{
    if (auto* andTest = std::get_if<Syntax::AndTest>(&orTest.variant))
    {
        return dump(*andTest);
    }
    return std::string();
}

std::string pylir::Dumper::dump(const pylir::Syntax::ConditionalExpression& conditionalExpression)
{
    if (!conditionalExpression.suffix)
    {
        return dump(conditionalExpression.value);
    }
    return std::string();
}

std::string pylir::Dumper::dump(const pylir::Syntax::LambdaExpression& lambdaExpression)
{
    return std::string();
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
           + addLastChild(dump(pylir::get<std::pair<Token, Syntax::OrExpr>>(starredItem.variant).second));
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
    return std::string();
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
    // TODO: targets
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
