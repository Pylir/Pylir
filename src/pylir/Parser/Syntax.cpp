#include "Syntax.hpp"

using namespace pylir::Diag;
using namespace pylir::Syntax;

std::pair<std::size_t, std::size_t> LocationProvider<TupleConstruct>::getRange(const TupleConstruct& value) noexcept
{
    if (value.maybeOpenBracket)
    {
        return {rangeLoc(*value.maybeOpenBracket).first, rangeLoc(*value.maybeCloseBracket).second};
    }
    PYLIR_ASSERT(!value.items.empty());
    return {rangeLoc(value.items.front()).first, rangeLoc(value.items.back()).second};
}

std::pair<std::size_t, std::size_t> LocationProvider<Lambda>::getRange(const Lambda& value) noexcept
{
    return {rangeLoc(value.lambdaKeyword).first, rangeLoc(*value.expression).second};
}

std::pair<std::size_t, std::size_t> LocationProvider<Conditional>::getRange(const Conditional& value) noexcept
{
    return {rangeLoc(*value.trueValue).first, rangeLoc(*value.elseValue).second};
}

std::pair<std::size_t, std::size_t> LocationProvider<BinOp>::getRange(const BinOp& value) noexcept
{
    return {rangeLoc(*value.lhs).first, rangeLoc(*value.rhs).second};
}

std::pair<std::size_t, std::size_t> LocationProvider<Comparison>::getRange(const Comparison& value) noexcept
{
    return {rangeLoc(*value.first).first, rangeLoc(*value.rest.back().second).second};
}

std::pair<std::size_t, std::size_t> LocationProvider<UnaryOp>::getRange(const UnaryOp& value) noexcept
{
    return {rangeLoc(value.operation).first, rangeLoc(*value.expression).second};
}

std::pair<std::size_t, std::size_t> LocationProvider<Call>::getRange(const Call& value) noexcept
{
    return {rangeLoc(*value.expression).first, rangeLoc(value.closeParenth).second};
}

std::pair<std::size_t, std::size_t> LocationProvider<DictDisplay>::getRange(const DictDisplay& value) noexcept
{
    return {rangeLoc(value.openBrace).first, rangeLoc(value.closeBrace).second};
}

std::pair<std::size_t, std::size_t> LocationProvider<SetDisplay>::getRange(const SetDisplay& value) noexcept
{
    return {rangeLoc(value.openBrace).first, rangeLoc(value.closeBrace).second};
}

std::pair<std::size_t, std::size_t> LocationProvider<ListDisplay>::getRange(const ListDisplay& value) noexcept
{
    return {rangeLoc(value.openSquare).first, rangeLoc(value.closeSquare).second};
}

std::pair<std::size_t, std::size_t> LocationProvider<Yield>::getRange(const Yield& value) noexcept
{
    return {rangeLoc(value.yieldToken).first, rangeLoc(*value.maybeExpression).second};
}

std::pair<std::size_t, std::size_t> LocationProvider<Generator>::getRange(const Generator& value) noexcept
{
    return {rangeLoc(value.openParenth).first, rangeLoc(value.closeParenth).second};
}

std::pair<std::size_t, std::size_t> LocationProvider<Slice>::getRange(const Slice& value) noexcept
{
    std::size_t start;
    std::size_t end;
    if (value.maybeLowerBound)
    {
        start = rangeLoc(*value.maybeLowerBound).first;
    }
    else
    {
        start = rangeLoc(value.firstColon).first;
    }
    if (value.maybeStride)
    {
        end = rangeLoc(*value.maybeStride).second;
    }
    else if (value.maybeUpperBound)
    {
        end = rangeLoc(*value.maybeUpperBound).second;
    }
    else
    {
        end = rangeLoc(value.firstColon).second;
    }
    return {start, end};
}

std::pair<std::size_t, std::size_t> LocationProvider<Subscription>::getRange(const Subscription& value) noexcept
{
    return {rangeLoc(*value.object).first, rangeLoc(value.closeSquareBracket).second};
}

std::pair<std::size_t, std::size_t> LocationProvider<AttributeRef>::getRange(const AttributeRef& value) noexcept
{
    return {rangeLoc(*value.object).first, rangeLoc(value.identifier).second};
}

std::pair<std::size_t, std::size_t> LocationProvider<Atom>::getRange(const Atom& value) noexcept
{
    return rangeLoc(value.token);
}

std::pair<std::size_t, std::size_t> LocationProvider<StarredItem>::getRange(const StarredItem& value) noexcept
{
    if (value.maybeStar)
    {
        return {rangeLoc(*value.maybeStar).first, rangeLoc(*value.expression).second};
    }
    return rangeLoc(*value.expression);
}

std::pair<std::size_t, std::size_t> LocationProvider<Assignment>::getRange(const Assignment& value) noexcept
{
    return {rangeLoc(value.variable).first, rangeLoc(*value.expression).second};
}

std::pair<std::size_t, std::size_t> LocationProvider<Argument>::getRange(const Argument& value) noexcept
{
    std::size_t start;
    if (value.maybeName)
    {
        start = rangeLoc(*value.maybeName).first;
    }
    else if (value.maybeExpansionsOrEqual)
    {
        start = rangeLoc(*value.maybeExpansionsOrEqual).first;
    }
    else
    {
        return rangeLoc(*value.expression);
    }
    return {start, rangeLoc(*value.expression).second};
}

std::pair<std::size_t, std::size_t> LocationProvider<Parameter>::getRange(const Parameter& value) noexcept
{
    std::size_t start;
    if (value.maybeStars)
    {
        start = rangeLoc(*value.maybeStars).first;
    }
    else
    {
        start = rangeLoc(value.name).first;
    }
    std::size_t end;
    if (value.maybeDefault)
    {
        end = rangeLoc(*value.maybeDefault).second;
    }
    else if (value.maybeType)
    {
        end = rangeLoc(*value.maybeType).second;
    }
    else
    {
        end = rangeLoc(value.name).second;
    }
    return {start, end};
}

std::pair<std::size_t, std::size_t> LocationProvider<RaiseStmt>::getRange(const RaiseStmt& value) noexcept
{
    auto start = rangeLoc(value.raise).first;
    std::size_t end;
    if (value.maybeCause)
    {
        end = rangeLoc(*value.maybeCause).second;
    }
    else if (value.maybeException)
    {
        end = rangeLoc(*value.maybeException).second;
    }
    else
    {
        end = rangeLoc(value.raise).second;
    }
    return {start, end};
}

std::pair<std::size_t, std::size_t> LocationProvider<ReturnStmt>::getRange(const ReturnStmt& value) noexcept
{
    std::size_t start = rangeLoc(value.returnKeyword).first;
    std::size_t end;
    if (value.maybeExpression)
    {
        end = rangeLoc(*value.maybeExpression).second;
    }
    else
    {
        end = rangeLoc(value.returnKeyword).second;
    }
    return {start, end};
}

std::pair<std::size_t, std::size_t> LocationProvider<SingleTokenStmt>::getRange(const SingleTokenStmt& value) noexcept
{
    return rangeLoc(value.token);
}

std::pair<std::size_t, std::size_t> LocationProvider<AssignmentStmt>::getRange(const AssignmentStmt& value) noexcept
{
    auto start = rangeLoc(*value.targets.front().first).first;
    std::size_t end;
    if (value.maybeExpression)
    {
        end = rangeLoc(*value.maybeExpression).second;
    }
    else if (value.maybeAnnotation)
    {
        end = rangeLoc(*value.maybeAnnotation).second;
    }
    else
    {
        end = rangeLoc(value.targets.back().second).second;
    }
    return {start, end};
}

std::pair<std::size_t, std::size_t> LocationProvider<IfStmt>::getRange(const IfStmt& value) noexcept
{
    auto start = rangeLoc(value.ifKeyword).first;
    std::size_t end;
    if (value.elseSection)
    {
        end = rangeLoc(*value.elseSection->suite).second;
    }
    else if (!value.elifs.empty())
    {
        end = rangeLoc(*value.elifs.back().suite).second;
    }
    else
    {
        end = rangeLoc(*value.suite).second;
    }
    return {start, end};
}

std::pair<std::size_t, std::size_t> LocationProvider<Suite>::getRange(const Suite& value) noexcept
{
    auto getLoc = [](const auto& variant)
    { return pylir::match(variant, [](const auto& ptr) { return rangeLoc(*ptr); }); };
    return {getLoc(value.statements.front()).first, getLoc(value.statements.back()).second};
}

std::pair<std::size_t, std::size_t> LocationProvider<ExpressionStmt>::getRange(const ExpressionStmt& value) noexcept
{
    return rangeLoc(*value.expression);
}

std::pair<std::size_t, std::size_t> LocationProvider<AssertStmt>::getRange(const AssertStmt& value) noexcept
{
    auto start = rangeLoc(value.assertKeyword).first;
    std::size_t end;
    if (value.maybeMessage)
    {
        end = rangeLoc(*value.maybeMessage).second;
    }
    else
    {
        end = rangeLoc(*value.condition).second;
    }
    return {start, end};
}

std::pair<std::size_t, std::size_t> LocationProvider<DelStmt>::getRange(const DelStmt& value) noexcept
{
    return {rangeLoc(value.del).first, rangeLoc(*value.targetList).second};
}

std::pair<std::size_t, std::size_t> LocationProvider<ImportStmt>::getRange(const ImportStmt& value) noexcept
{
    return pylir::match(
        value.variant,
        [](const ImportStmt::ImportAs& importAs) -> std::pair<std::size_t, std::size_t>
        {
            auto start = rangeLoc(importAs.import).first;
            std::size_t end;
            if (importAs.modules.back().second)
            {
                end = rangeLoc(*importAs.modules.back().second).second;
            }
            else
            {
                end = rangeLoc(importAs.modules.back().first.identifiers.back()).second;
            }
            return {start, end};
        },
        [](const ImportStmt::ImportAll& importAll) -> std::pair<std::size_t, std::size_t> {
            return {rangeLoc(importAll.from).first, rangeLoc(importAll.star).second};
        },
        [](const ImportStmt::FromImport& fromImport) -> std::pair<std::size_t, std::size_t>
        {
            auto start = rangeLoc(fromImport.from).first;
            std::size_t end;
            if (fromImport.imports.back().second)
            {
                end = rangeLoc(*fromImport.imports.back().second).second;
            }
            else
            {
                end = rangeLoc(fromImport.imports.back().first).second;
            }
            return {start, end};
        });
}

std::size_t LocationProvider<ImportStmt>::getPoint(const ImportStmt& value) noexcept
{
    return pylir::match(value.variant, [](const auto& value) { return pointLoc(value.import); });
}

std::pair<std::size_t, std::size_t>
    LocationProvider<GlobalOrNonLocalStmt>::getRange(const GlobalOrNonLocalStmt& value) noexcept
{
    return {rangeLoc(value.token).first, rangeLoc(value.identifiers.back()).second};
}

std::pair<std::size_t, std::size_t> LocationProvider<ForStmt>::getRange(const ForStmt& value) noexcept
{
    std::size_t start;
    if (value.maybeAsyncKeyword)
    {
        start = rangeLoc(*value.maybeAsyncKeyword).first;
    }
    else
    {
        start = rangeLoc(value.forKeyword).first;
    }
    std::size_t end;
    if (value.elseSection)
    {
        end = rangeLoc(*value.elseSection->suite).second;
    }
    else
    {
        end = rangeLoc(*value.suite).second;
    }
    return {start, end};
}

std::pair<std::size_t, std::size_t> LocationProvider<TryStmt>::getRange(const TryStmt& value) noexcept
{
    auto start = rangeLoc(value.tryKeyword).first;
    std::size_t end;
    if (value.finally)
    {
        end = rangeLoc(*value.finally->suite).second;
    }
    else if (value.elseSection)
    {
        end = rangeLoc(*value.elseSection->suite).second;
    }
    else if (value.maybeExceptAll)
    {
        end = rangeLoc(*value.maybeExceptAll->suite).second;
    }
    else
    {
        end = rangeLoc(*value.excepts.back().suite).second;
    }
    return {start, end};
}

std::pair<std::size_t, std::size_t> LocationProvider<WhileStmt>::getRange(const WhileStmt& value) noexcept
{
    return {rangeLoc(value.whileKeyword).first, rangeLoc(*value.suite).second};
}

std::pair<std::size_t, std::size_t> LocationProvider<WithStmt>::getRange(const WithStmt& value) noexcept
{
    std::size_t start;
    if (value.maybeAsyncKeyword)
    {
        start = rangeLoc(*value.maybeAsyncKeyword).first;
    }
    else
    {
        start = rangeLoc(value.withKeyword).first;
    }
    return {start, rangeLoc(*value.suite).second};
}

std::pair<std::size_t, std::size_t> LocationProvider<FuncDef>::getRange(const FuncDef& value) noexcept
{
    std::size_t start;
    if (!value.decorators.empty())
    {
        start = rangeLoc(value.decorators.front()).first;
    }
    else if (value.maybeAsyncKeyword)
    {
        start = rangeLoc(*value.maybeAsyncKeyword).first;
    }
    else
    {
        start = rangeLoc(value.def).first;
    }
    return {start, rangeLoc(*value.suite).second};
}

std::pair<std::size_t, std::size_t> LocationProvider<ClassDef>::getRange(const ClassDef& value) noexcept
{
    std::size_t start;
    if (!value.decorators.empty())
    {
        start = rangeLoc(value.decorators.front()).first;
    }
    else
    {
        start = rangeLoc(value.classKeyword).first;
    }
    return {start, rangeLoc(*value.suite).second};
}

std::pair<std::size_t, std::size_t> LocationProvider<Decorator>::getRange(const Decorator& value) noexcept
{
    return {rangeLoc(value.atSign).first, rangeLoc(*value.expression).second};
}

std::pair<std::size_t, std::size_t> LocationProvider<CompFor>::getRange(const CompFor& value) noexcept
{
    std::size_t start;
    if (value.awaitToken)
    {
        start = rangeLoc(*value.awaitToken).first;
    }
    else
    {
        start = rangeLoc(value.forToken).first;
    }
    std::size_t end = pylir::match(
        value.compIter, [&](std::monostate) { return rangeLoc(*value.test).second; },
        [](const auto& ptr) { return rangeLoc(*ptr).second; });
    return {start, end};
}

std::pair<std::size_t, std::size_t> LocationProvider<CompIf>::getRange(const CompIf& value) noexcept
{
    std::size_t start = rangeLoc(value.ifToken).first;
    std::size_t end = pylir::match(
        value.compIter, [&](std::monostate) { return rangeLoc(*value.test).second; },
        [](const auto& ptr) { return rangeLoc(*ptr).second; });
    return {start, end};
}

std::pair<std::size_t, std::size_t> LocationProvider<Comprehension>::getRange(const Comprehension& value) noexcept
{
    return {rangeLoc(*value.expression).first, rangeLoc(value.compFor).second};
}
