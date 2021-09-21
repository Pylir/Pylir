#include "Syntax.hpp"

#include <pylir/Diagnostics/DiagnosticsBuilder.hpp>

std::pair<std::size_t, std::size_t>
    pylir::Diag::LocationProvider<pylir::Syntax::Enclosure, void>::getRange(const Syntax::Enclosure& value) noexcept
{
    return pylir::match(
        value.variant,
        [](const Syntax::Enclosure::ParenthForm& parenthForm) -> std::pair<std::size_t, std::size_t> {
            return {range(parenthForm.openParenth).first, range(parenthForm.closeParenth).second};
        },
        [](const Syntax::Enclosure::GeneratorExpression& parenthForm) -> std::pair<std::size_t, std::size_t> {
            return {range(parenthForm.openParenth).first, range(parenthForm.closeParenth).second};
        },
        [](const Syntax::Enclosure::YieldAtom& parenthForm) -> std::pair<std::size_t, std::size_t> {
            return {range(parenthForm.openParenth).first, range(parenthForm.closeParenth).second};
        },
        [](const Syntax::Enclosure::ListDisplay& parenthForm) -> std::pair<std::size_t, std::size_t> {
            return {range(parenthForm.openSquare).first, range(parenthForm.closeSquare).second};
        },
        [](const Syntax::Enclosure::SetDisplay& parenthForm) -> std::pair<std::size_t, std::size_t> {
            return {range(parenthForm.openBrace).first, range(parenthForm.closeBrace).second};
        },
        [](const Syntax::Enclosure::DictDisplay& parenthForm) -> std::pair<std::size_t, std::size_t> {
            return {range(parenthForm.openBrace).first, range(parenthForm.closeBrace).second};
        });
}

std::pair<std::size_t, std::size_t> pylir::Diag::LocationProvider<pylir::Syntax::LambdaExpression, void>::getRange(
    const Syntax::LambdaExpression& value) noexcept
{
    return {range(value.lambdaToken).first, range(value.expression).second};
}

std::pair<std::size_t, std::size_t>
    pylir::Diag::LocationProvider<pylir::Syntax::Expression, void>::getRange(const Syntax::Expression& value) noexcept
{
    return pylir::match(
        value.variant,
        [](const Syntax::ConditionalExpression& conditionalExpression) { return range(conditionalExpression); },
        [](const std::unique_ptr<Syntax::LambdaExpression>& ptr) { return range(*ptr); });
}

std::pair<std::size_t, std::size_t> pylir::Diag::LocationProvider<pylir::Syntax::ConditionalExpression, void>::getRange(
    const Syntax::ConditionalExpression& value) noexcept
{
    if (!value.suffix)
    {
        return range(value.value);
    }
    return {range(value.value).first, range(*value.suffix->elseValue).second};
}

namespace
{
template <class ThisClass>
auto rangeOfBin(const ThisClass& thisClass)
{
    return pylir::match(
        thisClass.variant,
        [](const std::unique_ptr<typename ThisClass::BinOp>& binOp) -> std::pair<std::size_t, std::size_t>
        {
            auto& [lhs, token, rhs] = *binOp;
            return {pylir::Diag::range(*lhs).first, pylir::Diag::range(rhs).second};
        },
        [](const auto& value) -> std::pair<std::size_t, std::size_t> { return pylir::Diag::range(value); });
}
} // namespace

std::pair<std::size_t, std::size_t>
    pylir::Diag::LocationProvider<pylir::Syntax::OrTest, void>::getRange(const Syntax::OrTest& value) noexcept
{
    return rangeOfBin(value);
}

std::pair<std::size_t, std::size_t>
    pylir::Diag::LocationProvider<pylir::Syntax::AndTest, void>::getRange(const Syntax::AndTest& value) noexcept
{
    return rangeOfBin(value);
}

std::pair<std::size_t, std::size_t>
    pylir::Diag::LocationProvider<pylir::Syntax::OrExpr, void>::getRange(const Syntax::OrExpr& value) noexcept
{
    return rangeOfBin(value);
}

std::pair<std::size_t, std::size_t>
    pylir::Diag::LocationProvider<pylir::Syntax::XorExpr, void>::getRange(const Syntax::XorExpr& value) noexcept
{
    return rangeOfBin(value);
}

std::pair<std::size_t, std::size_t>
    pylir::Diag::LocationProvider<pylir::Syntax::AndExpr, void>::getRange(const Syntax::AndExpr& value) noexcept
{
    return rangeOfBin(value);
}

std::pair<std::size_t, std::size_t>
    pylir::Diag::LocationProvider<pylir::Syntax::ShiftExpr, void>::getRange(const Syntax::ShiftExpr& value) noexcept
{
    return rangeOfBin(value);
}

std::pair<std::size_t, std::size_t>
    pylir::Diag::LocationProvider<pylir::Syntax::AExpr, void>::getRange(const Syntax::AExpr& value) noexcept
{
    return rangeOfBin(value);
}

std::pair<std::size_t, std::size_t>
    pylir::Diag::LocationProvider<pylir::Syntax::NotTest, void>::getRange(const Syntax::NotTest& value) noexcept
{
    return pylir::match(
        value.variant, [](const Syntax::Comparison& comparison) { return range(comparison); },
        [](const std::pair<BaseToken, std::unique_ptr<Syntax::NotTest>>& pair) -> std::pair<std::size_t, std::size_t> {
            return {range(pair.first).first, range(*pair.second).second};
        });
}

std::pair<std::size_t, std::size_t>
    pylir::Diag::LocationProvider<pylir::Syntax::Comparison, void>::getRange(const Syntax::Comparison& value) noexcept
{
    if (value.rest.empty())
    {
        return range(value.left);
    }
    return {range(value.left).first, range(value.rest.back().second).second};
}

std::pair<std::size_t, std::size_t>
    pylir::Diag::LocationProvider<pylir::Syntax::MExpr, void>::getRange(const Syntax::MExpr& value) noexcept
{
    return pylir::match(
        value.variant, [](const Syntax::UExpr& uExpr) { return range(uExpr); },
        [](const std::unique_ptr<Syntax::MExpr::AtBin>& atBin) -> std::pair<std::size_t, std::size_t> {
            return {range(*atBin->lhs).first, range(*atBin->rhs).second};
        },
        [](const std::unique_ptr<Syntax::MExpr::BinOp>& binOp) -> std::pair<std::size_t, std::size_t> {
            return {range(*binOp->lhs).first, range(binOp->rhs).second};
        });
}

std::pair<std::size_t, std::size_t>
    pylir::Diag::LocationProvider<pylir::Syntax::UExpr, void>::getRange(const Syntax::UExpr& value) noexcept
{
    return pylir::match(
        value.variant, [](const Syntax::Power& power) { return range(power); },
        [](const std::pair<Token, std::unique_ptr<Syntax::UExpr>>& pair) -> std::pair<std::size_t, std::size_t> {
            return {range(pair.first).first, range(*pair.second).second};
        });
}

std::pair<std::size_t, std::size_t>
    pylir::Diag::LocationProvider<pylir::Syntax::Power, void>::getRange(const Syntax::Power& value) noexcept
{
    auto first = pylir::match(value.variant, [](auto&& value) { return range(value); });
    if (!value.rightHand)
    {
        return first;
    }
    return {first.first, range(*value.rightHand->second).second};
}

std::pair<std::size_t, std::size_t>
    pylir::Diag::LocationProvider<pylir::Syntax::AwaitExpr, void>::getRange(const Syntax::AwaitExpr& value) noexcept
{
    return {range(value.awaitToken).first, range(value.primary).second};
}

std::pair<std::size_t, std::size_t>
    pylir::Diag::LocationProvider<pylir::Syntax::Primary, void>::getRange(const Syntax::Primary& value) noexcept
{
    return pylir::match(value.variant, [](auto&& value) { return range(value); });
}

std::pair<std::size_t, std::size_t>
    pylir::Diag::LocationProvider<pylir::Syntax::Atom, void>::getRange(const Syntax::Atom& value) noexcept
{
    return pylir::match(
        value.variant, [](const Syntax::Atom::Literal& literal) { return range(literal.token); },
        [](const IdentifierToken& identifier) { return range(identifier); },
        [](const std::unique_ptr<Syntax::Enclosure>& enclosure) { return range(*enclosure); });
}

std::pair<std::size_t, std::size_t>
    pylir::Diag::LocationProvider<pylir::Syntax::Call, void>::getRange(const Syntax::Call& value) noexcept
{
    return {range(*value.primary).second, range(value.closeParentheses).second};
}

std::pair<std::size_t, std::size_t>
    pylir::Diag::LocationProvider<pylir::Syntax::Slicing, void>::getRange(const Syntax::Slicing& value) noexcept
{
    return {range(*value.primary).second, range(value.closeSquareBracket).second};
}

std::pair<std::size_t, std::size_t> pylir::Diag::LocationProvider<pylir::Syntax::Subscription, void>::getRange(
    const Syntax::Subscription& value) noexcept
{
    return {range(*value.primary).second, range(value.closeSquareBracket).second};
}

std::pair<std::size_t, std::size_t> pylir::Diag::LocationProvider<pylir::Syntax::AttributeRef, void>::getRange(
    const Syntax::AttributeRef& value) noexcept
{
    return {range(*value.primary).second, range(value.identifier).second};
}

std::pair<std::size_t, std::size_t>
    pylir::Diag::LocationProvider<pylir::Syntax::StarredItem, void>::getRange(const Syntax::StarredItem& value) noexcept
{
    return pylir::match(
        value.variant,
        [](const Syntax::AssignmentExpression& assignmentExpression) { return range(assignmentExpression); },
        [](const std::pair<BaseToken, Syntax::OrExpr>& pair) -> std::pair<std::size_t, std::size_t> {
            return {range(pair.first).first, range(pair.second).second};
        });
}

std::pair<std::size_t, std::size_t> pylir::Diag::LocationProvider<pylir::Syntax::AssignmentExpression, void>::getRange(
    const Syntax::AssignmentExpression& value) noexcept
{
    if (!value.identifierAndWalrus)
    {
        return range(*value.expression);
    }
    return {range(value.identifierAndWalrus->first).first, range(*value.expression).second};
}

std::pair<std::size_t, std::size_t> pylir::Diag::LocationProvider<pylir::Syntax::StarredExpression, void>::getRange(
    const Syntax::StarredExpression& value) noexcept
{
    return pylir::match(
        value.variant, [](const Syntax::Expression& expression) { return range(expression); },
        [](const Syntax::StarredExpression::Items& items) -> std::pair<std::size_t, std::size_t>
        {
            if (!items.leading.empty())
            {
                if (items.last)
                {
                    return {range(items.leading.front().first).first, range(*items.last).second};
                }
                return {range(items.leading.front().first).first, range(items.leading.back().second).second};
            }
            if (items.last)
            {
                return range(*items.last);
            }
            return {0, 0};
        });
}

std::pair<std::size_t, std::size_t> pylir::Diag::LocationProvider<pylir::Syntax::ExpressionList, void>::getRange(
    const Syntax::ExpressionList& expressionList) noexcept
{
    auto first = range(*expressionList.firstExpr);
    if (expressionList.trailingComma)
    {
        return {first.first, range(*expressionList.trailingComma).second};
    }
    if (expressionList.remainingExpr.empty())
    {
        return first;
    }
    return {first.first, range(*expressionList.remainingExpr.back().second).second};
}

std::pair<std::size_t, std::size_t> pylir::Diag::LocationProvider<pylir::Syntax::StarredList, void>::getRange(
    const Syntax::StarredList& starredList) noexcept
{
    auto first = range(*starredList.firstExpr);
    if (starredList.trailingComma)
    {
        return {first.first, range(*starredList.trailingComma).second};
    }
    if (starredList.remainingExpr.empty())
    {
        return first;
    }
    return {first.first, range(*starredList.remainingExpr.back().second).second};
}

std::pair<std::size_t, std::size_t> pylir::Diag::LocationProvider<pylir::Syntax::ArgumentList, void>::getRange(
    const Syntax::ArgumentList& argumentList) noexcept
{
    auto handlePositionalItem = [&](const Syntax::ArgumentList::PositionalItem& item)
    {
        return pylir::match(
            item.variant, [&](const Syntax::ArgumentList::PositionalItem::Star& star) { return range(star.asterisk); },
            [&](const std::unique_ptr<Syntax::AssignmentExpression>& assignmentExpression)
            { return range(*assignmentExpression); });
    };
    std::pair<std::size_t, std::size_t> first;
    if (argumentList.positionalArguments)
    {
        first = handlePositionalItem(argumentList.positionalArguments->firstItem);
    }
    else if (argumentList.starredAndKeywords)
    {
        first = range(argumentList.starredAndKeywords->first.identifier);
    }
    else if (argumentList.keywordArguments)
    {
        first = range(argumentList.keywordArguments->first.doubleAsterisk);
    }

    std::pair<std::size_t, std::size_t> last;
    if (argumentList.keywordArguments)
    {
        last = range(argumentList.keywordArguments->first.doubleAsterisk);
    }
    else if (argumentList.starredAndKeywords)
    {
        last = range(argumentList.starredAndKeywords->first.identifier);
    }
    else if (argumentList.positionalArguments)
    {
        last = handlePositionalItem(argumentList.positionalArguments->firstItem);
    }
    return {first.first, last.second};
}

std::pair<std::size_t, std::size_t>
    pylir::Diag::LocationProvider<pylir::Syntax::ParameterList::Parameter, void>::getRange(
        const Syntax::ParameterList::Parameter& parameter) noexcept
{
    if (!parameter.type)
    {
        return range(parameter.identifier);
    }
    return {range(parameter.identifier).first, range(parameter.type->second).second};
}

std::pair<std::size_t, std::size_t>
    pylir::Diag::LocationProvider<pylir::Syntax::ParameterList::DefParameter, void>::getRange(
        const Syntax::ParameterList::DefParameter& defParameter) noexcept
{
    if (!defParameter.defaultArg)
    {
        return range(defParameter.parameter);
    }
    return {range(defParameter.parameter).first, range(defParameter.defaultArg->second).second};
}
