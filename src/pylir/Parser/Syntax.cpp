#include "Syntax.hpp"

using namespace pylir::Diag;
using namespace pylir::Syntax;

std::pair<std::size_t, std::size_t> LocationProvider<TupleConstruct>::getRange(const TupleConstruct& value) noexcept
{
    if (value.maybeOpenBracket)
    {
        return {range(*value.maybeOpenBracket).first, range(*value.maybeCloseBracket).second};
    }
    PYLIR_ASSERT(!value.items.empty());
    return {range(value.items.front()).first, range(value.items.back()).second};
}

std::pair<std::size_t, std::size_t> LocationProvider<Lambda>::getRange(const Lambda& value) noexcept
{
    return {range(value.lambdaKeyword).first, range(*value.expression).second};
}

std::pair<std::size_t, std::size_t> LocationProvider<Expression>::getRange(const Expression& value) noexcept
{
    return value.match([&](auto&& sub) { return range(sub); });
}

std::pair<std::size_t, std::size_t> LocationProvider<Conditional>::getRange(const Conditional& value) noexcept
{
    return {range(*value.trueValue).first, range(*value.elseValue).second};
}

std::pair<std::size_t, std::size_t> LocationProvider<BinOp>::getRange(const BinOp& value) noexcept
{
    return {range(*value.lhs).first, range(*value.rhs).second};
}

std::pair<std::size_t, std::size_t> LocationProvider<Comparison>::getRange(const Comparison& value) noexcept
{
    return {range(*value.first).first, range(*value.rest.back().second).second};
}

std::pair<std::size_t, std::size_t> LocationProvider<UnaryOp>::getRange(const UnaryOp& value) noexcept
{
    return {range(value.operation).first, range(*value.expression).second};
}

std::pair<std::size_t, std::size_t> LocationProvider<Call>::getRange(const Call& value) noexcept
{
    return {range(*value.expression).first, range(value.closeParenth).second};
}

std::pair<std::size_t, std::size_t> LocationProvider<DictDisplay>::getRange(const DictDisplay& value) noexcept
{
    return {range(value.openBrace).first, range(value.closeBrace).second};
}

std::pair<std::size_t, std::size_t> LocationProvider<SetDisplay>::getRange(const SetDisplay& value) noexcept
{
    return {range(value.openBrace).first, range(value.closeBrace).second};
}

std::pair<std::size_t, std::size_t> LocationProvider<ListDisplay>::getRange(const ListDisplay& value) noexcept
{
    return {range(value.openSquare).first, range(value.closeSquare).second};
}

std::pair<std::size_t, std::size_t> LocationProvider<Yield>::getRange(const Yield& value) noexcept
{
    return {range(value.yieldToken).first, range(*value.maybeExpression).second};
}

std::pair<std::size_t, std::size_t> LocationProvider<Generator>::getRange(const Generator& value) noexcept
{
    return {range(value.openParenth).first, range(value.closeParenth).second};
}

std::pair<std::size_t, std::size_t> LocationProvider<Slice>::getRange(const Slice& value) noexcept
{
    std::size_t start;
    std::size_t end;
    if (value.maybeLowerBound)
    {
        start = range(*value.maybeLowerBound).first;
    }
    else
    {
        start = range(value.firstColon).first;
    }
    if (value.maybeStride)
    {
        end = range(*value.maybeStride).second;
    }
    else if (value.maybeUpperBound)
    {
        end = range(*value.maybeUpperBound).second;
    }
    else
    {
        end = range(value.firstColon).second;
    }
    return {start, end};
}

std::pair<std::size_t, std::size_t> LocationProvider<Subscription>::getRange(const Subscription& value) noexcept
{
    return {range(*value.object).first, range(value.closeSquareBracket).second};
}

std::pair<std::size_t, std::size_t> LocationProvider<AttributeRef>::getRange(const AttributeRef& value) noexcept
{
    return {range(*value.object).first, range(value.identifier).second};
}

std::pair<std::size_t, std::size_t> LocationProvider<Atom>::getRange(const Atom& value) noexcept
{
    return range(value.token);
}

std::pair<std::size_t, std::size_t> LocationProvider<StarredItem>::getRange(const StarredItem& value) noexcept
{
    if (value.maybeStar)
    {
        return {range(*value.maybeStar).first, range(*value.expression).second};
    }
    return range(*value.expression);
}

std::pair<std::size_t, std::size_t> LocationProvider<Assignment>::getRange(const Assignment& value) noexcept
{
    return {range(value.variable).first, range(*value.expression).second};
}

std::pair<std::size_t, std::size_t> LocationProvider<Argument>::getRange(const Argument& value) noexcept
{
    std::size_t start;
    if (value.maybeName)
    {
        start = range(*value.maybeName).first;
    }
    else if (value.maybeExpansionsOrEqual)
    {
        start = range(*value.maybeExpansionsOrEqual).first;
    }
    else
    {
        return range(*value.expression);
    }
    return {start, range(*value.expression).second};
}

std::pair<std::size_t, std::size_t> LocationProvider<Parameter>::getRange(const Parameter& value) noexcept
{
    std::size_t start;
    if (value.maybeStars)
    {
        start = range(*value.maybeStars).first;
    }
    else
    {
        start = range(value.name).first;
    }
    std::size_t end;
    if (value.maybeDefault)
    {
        end = range(*value.maybeDefault).second;
    }
    else if (value.maybeType)
    {
        end = range(*value.maybeType).second;
    }
    else
    {
        end = range(value.name).second;
    }
    return {start, end};
}
