#pragma once

#include <pylir/Support/Variant.hpp>

#include <string>

#include <tcb/span.hpp>

#include "Syntax.hpp"

namespace pylir
{
class Dumper;

namespace detail
{
template <class T, class U = Dumper, class = void>
struct CanDump : std::false_type
{
};

template <class T, class U>
struct CanDump<T, U, std::void_t<decltype(std::declval<U>().dump(std::declval<T>()))>> : std::true_type
{
};
} // namespace detail

class Dumper
{
    class Builder
    {
        Dumper* m_dumper;
        std::string m_title;
        std::vector<std::pair<std::string, std::optional<std::string>>> m_children;

        static std::string addMiddleChild(std::string_view middleChildDump,
                                          std::optional<std::string_view>&& label = std::nullopt);

        static std::string addLastChild(std::string_view lastChildDump,
                                        std::optional<std::string_view>&& label = std::nullopt);

    public:
        template <class S, class... Args>
        Builder(Dumper* dumper, const S& s, Args&&... args)
            : m_dumper(dumper), m_title(fmt::format(s, std::forward<Args>(args)...))
        {
        }

        Builder& add(std::string_view view, std::optional<std::string_view>&& label = std::nullopt)
        {
            m_children.emplace_back(view, label);
            return *this;
        }

        template <class C, std::enable_if_t<detail::CanDump<C>{}>* = nullptr>
        Builder& add(const C& object, std::optional<std::string_view>&& label = std::nullopt)
        {
            m_children.emplace_back(m_dumper->dump(object), label);
            return *this;
        }

        Builder& add(const Builder& other, std::optional<std::string_view>&& label = std::nullopt)
        {
            m_children.emplace_back(other.emit(), label);
            return *this;
        }

        std::string emit() const;
    };

    template <class S, class... Args>
    Builder createBuilder(const S& s, Args&&... args)
    {
        return Builder(this, s, std::forward<Args>(args)...);
    }

    friend class Builder;

    template <class T, class Func>
    std::string dump(const Syntax::CommaList<T>& list, Func dump, std::string_view name)
    {
        if (list.remainingExpr.empty())
        {
            return dump(*list.firstExpr);
        }
        auto builder = createBuilder("{}", name);
        builder.add(dump(*list.firstExpr));
        for (auto& iter : list.remainingExpr)
        {
            builder.add(dump(*iter.second));
        }
        return builder.emit();
    }

    template <class ThisClass, class TokenTypeGetter>
    std::string dumpBinOp(const ThisClass& thisClass, std::string_view name, TokenTypeGetter tokenTypeGetter)
    {
        return pylir::match(
            thisClass.variant, [&](const auto& previous) { return dump(previous); },
            [&](const std::unique_ptr<typename ThisClass::BinOp>& binOp)
            {
                auto& [lhs, token, rhs] = *binOp;
                return createBuilder(FMT_STRING("{} {:q}"), name, std::invoke(tokenTypeGetter, token))
                    .add(*lhs, "lhs")
                    .add(rhs, "rhs")
                    .emit();
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

    std::string dump(const Syntax::ArgumentList& argumentList);

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
