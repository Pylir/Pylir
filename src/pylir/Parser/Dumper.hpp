#pragma once

#include <pylir/Support/Variant.hpp>

#include <string>

#include <tcb/span.hpp>

#include "Syntax.hpp"

namespace pylir
{
class Dumper
{
    std::string addMiddleChild(std::string_view middleChildDump,
                               std::optional<std::string_view>&& label = std::nullopt);

    std::string addLastChild(std::string_view lastChildDump, std::optional<std::string_view>&& label = std::nullopt);

    template <class T, class Func>
    std::string dump(const Syntax::CommaList<T>& list, Func dump, std::string_view name)
    {
        if (list.remainingExpr.empty())
        {
            return dump(*list.firstExpr);
        }
        std::string result = std::string(name);
        result += addMiddleChild(dump(*list.firstExpr));
        for (auto& iter : tcb::span(list.remainingExpr).first(list.remainingExpr.size() - 1))
        {
            result += addMiddleChild(dump(*iter.second));
        }
        result += addLastChild(dump(*list.remainingExpr.back().second));
        return result;
    }

    template <class ThisClass, class TokenTypeGetter>
    std::string dumpBinOp(const ThisClass& thisClass, std::string_view name, TokenTypeGetter tokenTypeGetter)
    {
        return pylir::match(
            thisClass.variant, [&](const auto& previous) { return dump(previous); },
            [&](const typename ThisClass::BinOp& binOp)
            {
                auto& [lhs, token, rhs] = binOp;
                return fmt::format(FMT_STRING("{} {:q}"), name, std::invoke(tokenTypeGetter, token))
                       + addMiddleChild(dump(*lhs), "lhs") + addLastChild(dump(rhs), "rhs");
            });
    }

public:
    std::string dump(const Syntax::Atom& atom);

    std::string dump(const Syntax::Enclosure& enclosure);

    std::string dump(const Syntax::YieldExpression& yieldExpression);

    std::string dump(const Syntax::Primary& primary);

    std::string dump(const Syntax::AttributeRef& attribute);

    std::string dump(const Syntax::Subscription& subscription);

    std::string dump(const Syntax::Slicing& slicing);

    std::string dump(const Syntax::Comprehension& comprehension);

    std::string dump(const Syntax::AssignmentExpression& assignmentExpression);

    std::string dump(const Syntax::Call& call);

    std::string dump(const Syntax::AwaitExpr& awaitExpr);

    std::string dump(const Syntax::UExpr& uExpr);

    std::string dump(const Syntax::Power& power);

    std::string dump(const Syntax::MExpr& mExpr);

    std::string dump(const Syntax::AExpr& aExpr);

    std::string dump(const Syntax::ShiftExpr& shiftExpr);

    std::string dump(const Syntax::AndExpr& andExpr);

    std::string dump(const Syntax::XorExpr& xorExpr);

    std::string dump(const Syntax::OrExpr& orExpr);

    std::string dump(const Syntax::Comparison& comparison);

    std::string dump(const Syntax::NotTest& notTest);

    std::string dump(const Syntax::AndTest& andTest);

    std::string dump(const Syntax::OrTest& orTest);

    std::string dump(const Syntax::ConditionalExpression& conditionalExpression);

    std::string dump(const Syntax::LambdaExpression& lambdaExpression);

    std::string dump(const Syntax::Expression& expression);

    std::string dump(const Syntax::StarredItem& starredItem);

    std::string dump(const Syntax::StarredExpression& starredExpression);

    std::string dump(const Syntax::CompIf& compIf);

    std::string dump(const Syntax::CompFor& compFor);

    std::string dump(const Syntax::StarredList& starredList)
    {
        return dump(
            starredList, [&](auto&& value) { return dump(value); }, "starred list");
    }

    std::string dump(const Syntax::ExpressionList& expressionList)
    {
        return dump(
            expressionList, [&](auto&& value) { return dump(value); }, "expression list");
    }

    std::string dump(const Syntax::Target& square);

    std::string dump(const Syntax::TargetList& targetList)
    {
        return dump(
            targetList, [&](auto&& value) { return dump(value); }, "target list");
    }

    std::string dump(const Syntax::SimpleStmt& simpleStmt);

    std::string dump(const Syntax::AssertStmt& assertStmt);

    std::string dump(const Syntax::AssignmentStmt& assignmentStmt);

    std::string dump(const Syntax::AugTarget& augTarget);

    std::string dump(const Syntax::AugmentedAssignmentStmt& augmentedAssignmentStmt);

    std::string dump(const Syntax::AnnotatedAssignmentSmt& annotatedAssignmentSmt);

    std::string dump(const Syntax::PassStmt& passStmt);

    std::string dump(const Syntax::DelStmt& delStmt);

    std::string dump(const Syntax::ReturnStmt& returnStmt);

    std::string dump(const Syntax::YieldStmt& yieldStmt);

    std::string dump(const Syntax::RaiseStmt& raiseStmt);

    std::string dump(const Syntax::BreakStmt& breakStmt);

    std::string dump(const Syntax::ContinueStmt& continueStmt);

    std::string dump(const Syntax::ImportStmt& importStmt);

    std::string dump(const Syntax::FutureStmt& futureStmt);

    std::string dump(const Syntax::GlobalStmt& globalStmt);

    std::string dump(const Syntax::NonLocalStmt& nonLocalStmt);
};

} // namespace pylir
