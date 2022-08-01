// Copyright 2022 Markus Böck
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "Syntax.hpp"

namespace pylir::Syntax
{
template <class CRTP>
class Visitor
{
    CRTP* getImpl()
    {
        return static_cast<CRTP*>(this);
    }

protected:
    template <class... Args>
    void visit(const std::variant<Args...>& variant)
    {
        pylir::match(variant, [this](const auto& value) { getImpl()->visit(value); });
    }

    template <class T, std::enable_if_t<is_abstract_variant_concrete<T>{}>* = nullptr>
    void visit(const T& variant)
    {
        variant.match([this](const auto& value) { getImpl()->visit(value); });
    }

public:
    void visit(const Atom&) {}

    void visit(const AttributeRef& ref)
    {
        getImpl()->visit(*ref.object);
    }

    void visit(const Subscription& subscription)
    {
        getImpl()->visit(*subscription.object);
        getImpl()->visit(*subscription.index);
    }

    void visit(const ExpressionStmt& expressionStmt)
    {
        getImpl()->visit(*expressionStmt.expression);
    }

    void visit(const Slice& slice)
    {
        if (slice.maybeLowerBound)
        {
            getImpl()->visit(*slice.maybeLowerBound);
        }
        if (slice.maybeUpperBound)
        {
            getImpl()->visit(*slice.maybeUpperBound);
        }
        if (slice.maybeStride)
        {
            getImpl()->visit(*slice.maybeStride);
        }
    }

    void visit(const Argument& argument)
    {
        getImpl()->visit(*argument.expression);
    }

    void visit(const Call& call)
    {
        getImpl()->visit(*call.expression);
        pylir::match(
            call.variant,
            [&](const std::vector<Argument>& arguments)
            {
                for (auto& iter : arguments)
                {
                    getImpl()->visit(iter);
                }
            },
            [&](const Comprehension& comprehension) { getImpl()->visit(comprehension); });
    }

    void visit(const Comparison& comparison)
    {
        getImpl()->visit(*comparison.first);
        for (const auto& [op, expr] : comparison.rest)
        {
            (void)op;
            getImpl()->visit(*expr);
        }
    }

    void visit(const BinOp& binOp)
    {
        getImpl()->visit(*binOp.lhs);
        getImpl()->visit(*binOp.rhs);
    }

    void visit(const UnaryOp& unaryOp)
    {
        getImpl()->visit(*unaryOp.expression);
    }

    void visit(const Generator& generator)
    {
        getImpl()->visit(*generator.expression);
        getImpl()->visit(generator.compFor);
    }

    void visit(const ListDisplay& listDisplay)
    {
        if (auto* comprehension = std::get_if<Comprehension>(&listDisplay.variant))
        {
            getImpl()->visit(*comprehension);
            return;
        }
        for (const auto& iter : pylir::get<std::vector<StarredItem>>(listDisplay.variant))
        {
            getImpl()->visit(iter);
        }
    }

    void visit(const SetDisplay& setDisplay)
    {
        if (auto* comprehension = std::get_if<Comprehension>(&setDisplay.variant))
        {
            getImpl()->visit(*comprehension);
            return;
        }
        for (const auto& iter : pylir::get<std::vector<StarredItem>>(setDisplay.variant))
        {
            getImpl()->visit(iter);
        }
    }

    void visit(const DictDisplay& dictDisplay)
    {
        if (auto* comprehension = std::get_if<DictDisplay::DictComprehension>(&dictDisplay.variant))
        {
            getImpl()->visit(*comprehension->first);
            getImpl()->visit(*comprehension->second);
            getImpl()->visit(comprehension->compFor);
            return;
        }
        for (const auto& iter : pylir::get<std::vector<DictDisplay::KeyDatum>>(dictDisplay.variant))
        {
            getImpl()->visit(*iter.key);
            if (iter.maybeValue)
            {
                getImpl()->visit(*iter.maybeValue);
            }
        }
    }

    void visit(const Assignment& assignmentExpression)
    {
        getImpl()->visit(*assignmentExpression.expression);
    }

    void visit(const Conditional& conditionalExpression)
    {
        getImpl()->visit(*conditionalExpression.trueValue);
        getImpl()->visit(*conditionalExpression.condition);
        getImpl()->visit(*conditionalExpression.elseValue);
    }

    void visit(const Lambda& lambdaExpression)
    {
        for (auto& iter : lambdaExpression.parameters)
        {
            getImpl()->visit(iter);
        }
        getImpl()->visit(*lambdaExpression.expression);
    }

    void visit(const StarredItem& starredItem)
    {
        getImpl()->visit(*starredItem.expression);
    }

    void visit(const TupleConstruct& tuple)
    {
        for (auto& iter : tuple.items)
        {
            getImpl()->visit(iter);
        }
    }

    void visit(const CompFor& compFor)
    {
        getImpl()->visit(*compFor.targets);
        getImpl()->visit(*compFor.test);
        pylir::match(
            compFor.compIter, [](std::monostate) {}, [&](const auto& ptr) { getImpl()->visit(*ptr); });
    }

    void visit(const CompIf& compIf)
    {
        getImpl()->visit(*compIf.test);
        pylir::match(
            compIf.compIter, [](std::monostate) {}, [&](const auto& ptr) { getImpl()->visit(*ptr); });
    }

    void visit(const Comprehension& comprehension)
    {
        getImpl()->visit(*comprehension.expression);
        getImpl()->visit(comprehension.compFor);
    }

    void visit(const Yield& yieldExpression)
    {
        if (yieldExpression.maybeExpression)
        {
            getImpl()->visit(*yieldExpression.maybeExpression);
        }
    }

    void visit(const AssignmentStmt& assignmentStmt)
    {
        for (const auto& [targetList, token] : assignmentStmt.targets)
        {
            (void)token;
            getImpl()->visit(*targetList);
        }
        if (assignmentStmt.maybeAnnotation)
        {
            getImpl()->visit(*assignmentStmt.maybeAnnotation);
        }
        if (assignmentStmt.maybeExpression)
        {
            getImpl()->visit(*assignmentStmt.maybeExpression);
        }
    }

    void visit(const AssertStmt& assertStmt)
    {
        getImpl()->visit(*assertStmt.condition);
        if (assertStmt.maybeMessage)
        {
            getImpl()->visit(*assertStmt.maybeMessage);
        }
    }

    void visit(const SingleTokenStmt&) {}

    void visit(const DelStmt& delStmt)
    {
        getImpl()->visit(*delStmt.targetList);
    }

    void visit(const ReturnStmt& returnStmt)
    {
        if (returnStmt.maybeExpression)
        {
            getImpl()->visit(*returnStmt.maybeExpression);
        }
    }

    void visit(const RaiseStmt& raiseStmt)
    {
        if (raiseStmt.maybeException)
        {
            getImpl()->visit(*raiseStmt.maybeException);
        }
        if (raiseStmt.maybeCause)
        {
            getImpl()->visit(*raiseStmt.maybeCause);
        }
    }

    void visit(const ImportStmt&) {}

    void visit(const FutureStmt&) {}

    void visit(const GlobalOrNonLocalStmt&) {}

    void visit(const IfStmt& ifStmt)
    {
        getImpl()->visit(*ifStmt.condition);
        getImpl()->visit(*ifStmt.suite);
        for (const auto& elif : ifStmt.elifs)
        {
            getImpl()->visit(*elif.condition);
            getImpl()->visit(*elif.suite);
        }
        if (ifStmt.elseSection)
        {
            getImpl()->visit(*ifStmt.elseSection->suite);
        }
    }

    void visit(const WhileStmt& whileStmt)
    {
        getImpl()->visit(*whileStmt.condition);
        getImpl()->visit(*whileStmt.suite);
        if (whileStmt.elseSection)
        {
            getImpl()->visit(*whileStmt.elseSection->suite);
        }
    }

    void visit(const ForStmt& forStmt)
    {
        getImpl()->visit(*forStmt.targetList);
        getImpl()->visit(*forStmt.expression);
        getImpl()->visit(*forStmt.suite);
        if (forStmt.elseSection)
        {
            getImpl()->visit(*forStmt.elseSection->suite);
        }
    }

    void visit(const TryStmt& tryStmt)
    {
        getImpl()->visit(*tryStmt.suite);
        for (const auto& except : tryStmt.excepts)
        {
            getImpl()->visit(*except.filter);
            getImpl()->visit(*except.suite);
        }
        if (tryStmt.maybeExceptAll)
        {
            getImpl()->visit(*tryStmt.maybeExceptAll->suite);
        }
        if (tryStmt.elseSection)
        {
            getImpl()->visit(*tryStmt.elseSection->suite);
        }
        if (tryStmt.finally)
        {
            getImpl()->visit(*tryStmt.finally->suite);
        }
    }

    void visit(const WithStmt& withStmt)
    {
        for (const auto& withItem : withStmt.items)
        {
            getImpl()->visit(*withItem.expression);
            if (withItem.maybeTarget)
            {
                getImpl()->visit(*withItem.maybeTarget);
            }
        }
        getImpl()->visit(*withStmt.suite);
    }

    void visit(const Parameter& parameter)
    {
        if (parameter.maybeDefault)
        {
            getImpl()->visit(*parameter.maybeDefault);
        }
        if (parameter.maybeType)
        {
            getImpl()->visit(*parameter.maybeType);
        }
    }

    void visit(const Decorator& decorator)
    {
        getImpl()->visit(*decorator.expression);
    }

    void visit(const FuncDef& funcDef)
    {
        for (auto& iter : funcDef.decorators)
        {
            getImpl()->visit(iter);
        }
        for (auto& iter : funcDef.parameterList)
        {
            getImpl()->visit(iter);
        }
        if (funcDef.maybeSuffix)
        {
            getImpl()->visit(*funcDef.maybeSuffix);
        }
        getImpl()->visit(*funcDef.suite);
    }

    void visit(const ClassDef& classDef)
    {
        for (auto& iter : classDef.decorators)
        {
            getImpl()->visit(iter);
        }
        if (classDef.inheritance)
        {
            for (auto& iter : classDef.inheritance->argumentList)
            {
                getImpl()->visit(iter);
            }
        }
        getImpl()->visit(*classDef.suite);
    }

    void visit(const Suite& suite)
    {
        for (auto& iter : suite.statements)
        {
            pylir::match(iter, [&](const auto& stmt) { getImpl()->visit(*stmt); });
        }
    }

    void visit(const FileInput& fileInput)
    {
        getImpl()->visit(fileInput.input);
    }
};
} // namespace pylir::Syntax
