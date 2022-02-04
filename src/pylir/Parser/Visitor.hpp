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

public:
    template <class T>
    void visit(const CommaList<T>& commaList)
    {
        getImpl()->visit(*commaList.firstExpr);
        for (auto& [token, ptr] : commaList.remainingExpr)
        {
            (void)token;
            getImpl()->visit(*ptr);
        }
    }

    void visit(const Atom& atom)
    {
        if (const auto* enclosure = std::get_if<std::unique_ptr<Enclosure>>(&atom.variant))
        {
            getImpl()->visit(**enclosure);
        }
    }

    void visit(const AttributeRef& ref)
    {
        getImpl()->visit(*ref.primary);
    }

    void visit(const Subscription& subscription)
    {
        getImpl()->visit(*subscription.primary);
        getImpl()->visit(subscription.expressionList);
    }

    void visit(const Slicing::ProperSlice& properSlice)
    {
        if (properSlice.optionalLowerBound)
        {
            getImpl()->visit(*properSlice.optionalLowerBound);
        }
        if (properSlice.optionalUpperBound)
        {
            getImpl()->visit(*properSlice.optionalUpperBound);
        }
        if (properSlice.optionalStride)
        {
            getImpl()->visit(*properSlice.optionalStride);
        }
    }

    void visit(const Slicing& slicing)
    {
        getImpl()->visit(*slicing.primary);
        getImpl()->visit(slicing.sliceList);
    }

    void visit(const ArgumentList& argumentList)
    {
        if (argumentList.positionalArguments)
        {
            auto handlePositionalItem = [&](const ArgumentList::PositionalItem& positionalItem)
            {
                pylir::match(
                    positionalItem.variant,
                    [&](const std::unique_ptr<AssignmentExpression>& expression) { getImpl()->visit(*expression); },
                    [&](const ArgumentList::PositionalItem::Star& star) { getImpl()->visit(*star.expression); });
            };
            handlePositionalItem(argumentList.positionalArguments->firstItem);
            for (const auto& [token, item] : argumentList.positionalArguments->rest)
            {
                (void)token;
                handlePositionalItem(item);
            }
        }
        auto handleKeywordItem = [&](const ArgumentList::KeywordItem& keywordItem)
        { getImpl()->visit(*keywordItem.expression); };
        if (argumentList.starredAndKeywords)
        {
            handleKeywordItem(argumentList.starredAndKeywords->first);
            for (const auto& [token, variant] : argumentList.starredAndKeywords->rest)
            {
                (void)token;
                pylir::match(variant, handleKeywordItem,
                             [&](const ArgumentList::StarredAndKeywords::Expression& expression)
                             { getImpl()->visit(*expression.expression); });
            }
        }
        if (argumentList.keywordArguments)
        {
            auto handleExpression = [&](const ArgumentList::KeywordArguments::Expression& expression)
            { getImpl()->visit(*expression.expression); };
            handleExpression(argumentList.keywordArguments->first);
            for (const auto& [token, variant] : argumentList.keywordArguments->rest)
            {
                (void)token;
                pylir::match(variant, handleKeywordItem, handleExpression);
            }
        }
    }

    void visit(const Call& call)
    {
        getImpl()->visit(*call.primary);
        pylir::match(
            call.variant, [](std::monostate) {},
            [&](const std::pair<ArgumentList, std::optional<BaseToken>>& pair) { getImpl()->visit(pair.first); },
            [&](const std::unique_ptr<Comprehension>& comprehension) { getImpl()->visit(*comprehension); });
    }

    void visit(const Primary& primary)
    {
        getImpl()->visit(primary.variant);
    }

    void visit(const AwaitExpr& awaitExpr)
    {
        getImpl()->visit(awaitExpr.primary);
    }

    void visit(const Power& power)
    {
        getImpl()->visit(power.variant);
        if (power.rightHand)
        {
            getImpl()->visit(*power.rightHand->second);
        }
    }

    void visit(const UExpr& uExpr)
    {
        pylir::match(
            uExpr.variant, [&](const Power& power) { getImpl()->visit(power); },
            [&](const std::pair<Token, std::unique_ptr<UExpr>>& pair) { getImpl()->visit(*pair.second); });
    }

    void visit(const MExpr& mExpr)
    {
        pylir::match(
            mExpr.variant, [&](const UExpr& uExpr) { getImpl()->visit(uExpr); },
            [&](const std::unique_ptr<MExpr::AtBin>& ptr)
            {
                getImpl()->visit(*ptr->lhs);
                getImpl()->visit(*ptr->rhs);
            },
            [&](const std::unique_ptr<MExpr::BinOp>& ptr)
            {
                getImpl()->visit(*ptr->lhs);
                getImpl()->visit(ptr->rhs);
            });
    }

    void visit(const AExpr& aExpr)
    {
        pylir::match(
            aExpr.variant, [&](const MExpr& mExpr) { getImpl()->visit(mExpr); },
            [&](const std::unique_ptr<AExpr::BinOp>& ptr)
            {
                getImpl()->visit(*ptr->lhs);
                getImpl()->visit(ptr->rhs);
            });
    }

    void visit(const ShiftExpr& shiftExpr)
    {
        pylir::match(
            shiftExpr.variant, [&](const AExpr& aExpr) { getImpl()->visit(aExpr); },
            [&](const std::unique_ptr<ShiftExpr::BinOp>& ptr)
            {
                getImpl()->visit(*ptr->lhs);
                getImpl()->visit(ptr->rhs);
            });
    }

    void visit(const AndExpr& andExpr)
    {
        pylir::match(
            andExpr.variant, [&](const ShiftExpr& shiftExpr) { getImpl()->visit(shiftExpr); },
            [&](const std::unique_ptr<AndExpr::BinOp>& ptr)
            {
                getImpl()->visit(*ptr->lhs);
                getImpl()->visit(ptr->rhs);
            });
    }

    void visit(const XorExpr& xorExpr)
    {
        pylir::match(
            xorExpr.variant, [&](const AndExpr& andExpr) { getImpl()->visit(andExpr); },
            [&](const std::unique_ptr<XorExpr::BinOp>& ptr)
            {
                getImpl()->visit(*ptr->lhs);
                getImpl()->visit(ptr->rhs);
            });
    }

    void visit(const OrExpr& orExpr)
    {
        pylir::match(
            orExpr.variant, [&](const XorExpr& xorExpr) { getImpl()->visit(xorExpr); },
            [&](const std::unique_ptr<OrExpr::BinOp>& ptr)
            {
                getImpl()->visit(*ptr->lhs);
                getImpl()->visit(ptr->rhs);
            });
    }

    void visit(const Comparison& comparison)
    {
        getImpl()->visit(comparison.left);
        for (const auto& [op, expr] : comparison.rest)
        {
            (void)op;
            getImpl()->visit(expr);
        }
    }

    void visit(const NotTest& notTest)
    {
        pylir::match(
            notTest.variant, [&](const Comparison& comparison) { getImpl()->visit(comparison); },
            [&](const std::pair<BaseToken, std::unique_ptr<NotTest>>& pair) { getImpl()->visit(*pair.second); });
    }

    void visit(const AndTest& andTest)
    {
        pylir::match(
            andTest.variant, [&](const NotTest& notTest) { getImpl()->visit(notTest); },
            [&](const std::unique_ptr<AndTest::BinOp>& ptr)
            {
                getImpl()->visit(*ptr->lhs);
                getImpl()->visit(ptr->rhs);
            });
    }

    void visit(const OrTest& orTest)
    {
        pylir::match(
            orTest.variant, [&](const AndTest& andTest) { getImpl()->visit(andTest); },
            [&](const std::unique_ptr<OrTest::BinOp>& ptr)
            {
                getImpl()->visit(*ptr->lhs);
                getImpl()->visit(ptr->rhs);
            });
    }

    void visit(const AssignmentExpression& assignmentExpression)
    {
        getImpl()->visit(*assignmentExpression.expression);
    }

    void visit(const ConditionalExpression& conditionalExpression)
    {
        getImpl()->visit(conditionalExpression.value);
        if (conditionalExpression.suffix)
        {
            getImpl()->visit(*conditionalExpression.suffix->test);
            getImpl()->visit(*conditionalExpression.suffix->elseValue);
        }
    }

    void visit(const Expression& expression)
    {
        pylir::match(
            expression.variant,
            [&](const ConditionalExpression& conditionalExpression) { getImpl()->visit(conditionalExpression); },
            [&](const std::unique_ptr<LambdaExpression>& lambdaExpression) { getImpl()->visit(*lambdaExpression); });
    }

    void visit(const LambdaExpression& lambdaExpression)
    {
        if (lambdaExpression.parameterList)
        {
            getImpl()->visit(*lambdaExpression.parameterList);
        }
        getImpl()->visit(lambdaExpression.expression);
    }

    void visit(const StarredItem& starredItem)
    {
        pylir::match(
            starredItem.variant,
            [&](const AssignmentExpression& assignmentExpression) { getImpl()->visit(assignmentExpression); },
            [&](const std::pair<BaseToken, OrExpr>& pair) { getImpl()->visit(pair.second); });
    }

    void visit(const StarredExpression& expression)
    {
        pylir::match(
            expression.variant,
            [&](const StarredExpression::Items& items)
            {
                for (const auto& [item, token] : items.leading)
                {
                    (void)token;
                    getImpl()->visit(item);
                }
                if (items.last)
                {
                    getImpl()->visit(*items.last);
                }
            },
            [&](const Expression& expression) { getImpl()->visit(expression); });
    }

    void visit(const CompFor& compFor)
    {
        getImpl()->visit(compFor.targets);
        getImpl()->visit(compFor.orTest);
        pylir::match(
            compFor.compIter, [](std::monostate) {},
            [&](const std::unique_ptr<CompFor>& compFor) { getImpl()->visit(*compFor); },
            [&](const std::unique_ptr<CompIf>& compIf) { getImpl()->visit(*compIf); });
    }

    void visit(const CompIf& compIf)
    {
        getImpl()->visit(compIf.orTest);
        pylir::match(
            compIf.compIter, [](std::monostate) {}, [&](const CompFor& compFor) { getImpl()->visit(compFor); },
            [&](const std::unique_ptr<CompIf>& compIf) { getImpl()->visit(*compIf); });
    }

    void visit(const Comprehension& comprehension)
    {
        getImpl()->visit(comprehension.assignmentExpression);
        getImpl()->visit(comprehension.compFor);
    }

    void visit(const YieldExpression& yieldExpression)
    {
        pylir::match(
            yieldExpression.variant, [](std::monostate) {},
            [&](const ExpressionList& expressionList) { getImpl()->visit(expressionList); },
            [&](const std::pair<BaseToken, Expression>& pair) { getImpl()->visit(pair.second); });
    }

    void visit(const Enclosure& enclosure)
    {
        pylir::match(
            enclosure.variant,
            [&](const Enclosure::ParenthForm& parenthForm)
            {
                if (parenthForm.expression)
                {
                    getImpl()->visit(*parenthForm.expression);
                }
            },
            [&](const Enclosure::ListDisplay& listDisplay)
            {
                pylir::match(
                    listDisplay.variant, [](std::monostate) {},
                    [&](const StarredList& starredList) { getImpl()->visit(starredList); },
                    [&](const Comprehension& comprehension) { getImpl()->visit(comprehension); });
            },
            [&](const Enclosure::SetDisplay& setDisplay) { getImpl()->visit(setDisplay.variant); },
            [&](const Enclosure::DictDisplay& dictDisplay)
            {
                pylir::match(
                    dictDisplay.variant, [](std::monostate) {},
                    [&](const CommaList<Enclosure::DictDisplay::KeyDatum>& commaList)
                    {
                        auto handleKeyDatum = [&](const Enclosure::DictDisplay::KeyDatum& keyDatum)
                        {
                            pylir::match(
                                keyDatum.variant,
                                [&](const Enclosure::DictDisplay::KeyDatum::Key& key)
                                {
                                    getImpl()->visit(key.first);
                                    getImpl()->visit(key.second);
                                },
                                [&](const Enclosure::DictDisplay::KeyDatum::Datum& datum)
                                { getImpl()->visit(datum.orExpr); });
                        };
                        handleKeyDatum(*commaList.firstExpr);
                        for (const auto& [token, expr] : commaList.remainingExpr)
                        {
                            (void)token;
                            handleKeyDatum(*expr);
                        }
                    },
                    [&](const Enclosure::DictDisplay::DictComprehension& comprehension)
                    {
                        getImpl()->visit(comprehension.first);
                        getImpl()->visit(comprehension.second);
                        getImpl()->visit(comprehension.compFor);
                    });
            },
            [&](const Enclosure::GeneratorExpression& generatorExpression)
            {
                getImpl()->visit(generatorExpression.expression);
                getImpl()->visit(generatorExpression.compFor);
            },
            [&](const Enclosure::YieldAtom& yieldAtom) { getImpl()->visit(yieldAtom.yieldExpression); });
    }

    void visit(const Target& target)
    {
        pylir::match(
            target.variant, [](const IdentifierToken&) {},
            [&](const Target::Parenth& parenth)
            {
                if (parenth.targetList)
                {
                    getImpl()->visit(*parenth.targetList);
                }
            },
            [&](const Target::Square& square)
            {
                if (square.targetList)
                {
                    getImpl()->visit(*square.targetList);
                }
            },
            [&](const std::pair<BaseToken, std::unique_ptr<Target>>& pair) { getImpl()->visit(*pair.second); },
            [&](const auto& other) { getImpl()->visit(other); });
    }

    void visit(const AssignmentStmt& assignmentStmt)
    {
        for (const auto& [targetList, token] : assignmentStmt.targets)
        {
            (void)token;
            getImpl()->visit(targetList);
        }
        getImpl()->visit(assignmentStmt.variant);
    }

    void visit(const AugTarget& augTarget)
    {
        pylir::match(
            augTarget.variant, [](const IdentifierToken&) {}, [&](const auto& other) { getImpl()->visit(other); });
    }

    void visit(const AugmentedAssignmentStmt& augmentedAssignmentStmt)
    {
        getImpl()->visit(augmentedAssignmentStmt.augTarget);
        getImpl()->visit(augmentedAssignmentStmt.variant);
    }

    void visit(const AnnotatedAssignmentSmt& annotatedAssignmentSmt)
    {
        getImpl()->visit(annotatedAssignmentSmt.augTarget);
        getImpl()->visit(annotatedAssignmentSmt.expression);
        if (annotatedAssignmentSmt.optionalAssignmentStmt)
        {
            getImpl()->visit(annotatedAssignmentSmt.optionalAssignmentStmt->second);
        }
    }

    void visit(const AssertStmt& assertStmt)
    {
        getImpl()->visit(assertStmt.condition);
        if (assertStmt.message)
        {
            getImpl()->visit(assertStmt.message->second);
        }
    }

    void visit(const PassStmt&) {}

    void visit(const DelStmt& delStmt)
    {
        getImpl()->visit(delStmt.targetList);
    }

    void visit(const ReturnStmt& returnStmt)
    {
        if (returnStmt.expressions)
        {
            getImpl()->visit(*returnStmt.expressions);
        }
    }

    void visit(const YieldStmt& yieldStmt)
    {
        getImpl()->visit(yieldStmt.yieldExpression);
    }

    void visit(const RaiseStmt& raiseStmt)
    {
        if (raiseStmt.expressions)
        {
            getImpl()->visit(raiseStmt.expressions->first);
            if (raiseStmt.expressions->second)
            {
                getImpl()->visit(raiseStmt.expressions->second->second);
            }
        }
    }

    void visit(const BreakStmt&) {}

    void visit(const ContinueStmt&) {}

    void visit(const ImportStmt&) {}

    void visit(const FutureStmt&) {}

    void visit(const GlobalStmt&) {}

    void visit(const NonLocalStmt&) {}

    void visit(const SimpleStmt& simpleStmt)
    {
        getImpl()->visit(simpleStmt.variant);
    }

    void visit(const IfStmt& ifStmt)
    {
        getImpl()->visit(ifStmt.condition);
        getImpl()->visit(*ifStmt.suite);
        for (const auto& elif : ifStmt.elifs)
        {
            getImpl()->visit(elif.condition);
            getImpl()->visit(*elif.suite);
        }
        if (ifStmt.elseSection)
        {
            getImpl()->visit(*ifStmt.elseSection->suite);
        }
    }

    void visit(const WhileStmt& whileStmt)
    {
        getImpl()->visit(whileStmt.condition);
        getImpl()->visit(*whileStmt.suite);
        if (whileStmt.elseSection)
        {
            getImpl()->visit(*whileStmt.elseSection->suite);
        }
    }

    void visit(const ForStmt& forStmt)
    {
        getImpl()->visit(forStmt.targetList);
        getImpl()->visit(forStmt.expressionList);
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
            if (except.expression)
            {
                getImpl()->visit(except.expression->first);
            }
            getImpl()->visit(*except.suite);
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
        auto handleWithItem = [&](const WithStmt::WithItem& withItem)
        {
            getImpl()->visit(withItem.expression);
            if (withItem.target)
            {
                getImpl()->visit(withItem.target->second);
            }
        };
        handleWithItem(withStmt.first);
        for (const auto& [token, withItem] : withStmt.rest)
        {
            (void)token;
            handleWithItem(withItem);
        }
        getImpl()->visit(*withStmt.suite);
    }

    void visit(const ParameterList::Parameter& parameter)
    {
        if (parameter.type)
        {
            getImpl()->visit(parameter.type->second);
        }
    }

    void visit(const ParameterList::DefParameter& defParameter)
    {
        getImpl()->visit(defParameter.parameter);
        if (defParameter.defaultArg)
        {
            getImpl()->visit(defParameter.defaultArg->second);
        }
    }

    void visit(const ParameterList::StarArgs& starArgs)
    {
        pylir::match(
            starArgs.variant,
            [&](const ParameterList::StarArgs::DoubleStar& doubleStar) { getImpl()->visit(doubleStar.parameter); },
            [&](const ParameterList::StarArgs::Star& star)
            {
                if (star.parameter)
                {
                    getImpl()->visit(*star.parameter);
                }
                for (const auto& [token, parameter] : star.defParameters)
                {
                    (void)token;
                    getImpl()->visit(parameter);
                }
                if (star.further && star.further->doubleStar)
                {
                    getImpl()->visit(star.further->doubleStar->parameter);
                }
            });
    }

    void visit(const ParameterList::NoPosOnly& noPosOnly)
    {
        pylir::match(
            noPosOnly.variant, [&](const ParameterList::StarArgs& starArgs) { getImpl()->visit(starArgs); },
            [&](const ParameterList::NoPosOnly::DefParams& defParams)
            {
                getImpl()->visit(defParams.first);
                for (const auto& [token, parameter] : defParams.rest)
                {
                    (void)token;
                    getImpl()->visit(parameter);
                }
                if (defParams.suffix && defParams.suffix->second)
                {
                    getImpl()->visit(*defParams.suffix->second);
                }
            });
    }

    void visit(const ParameterList::PosOnly& posOnly)
    {
        getImpl()->visit(posOnly.first);
        for (const auto& [token, parameter] : posOnly.rest)
        {
            (void)token;
            getImpl()->visit(parameter);
        }
        if (posOnly.suffix && posOnly.suffix->second)
        {
            getImpl()->visit(*posOnly.suffix->second);
        }
    }

    void visit(const ParameterList& parameterList)
    {
        getImpl()->visit(parameterList.variant);
    }

    void visit(const Decorator& decorator)
    {
        getImpl()->visit(decorator.assignmentExpression);
    }

    void visit(const FuncDef& funcDef)
    {
        std::for_each(funcDef.decorators.begin(), funcDef.decorators.end(),
                      [&](const Decorator& decorator) { getImpl()->visit(decorator); });
        if (funcDef.parameterList)
        {
            getImpl()->visit(*funcDef.parameterList);
        }
        if (funcDef.suffix)
        {
            getImpl()->visit(funcDef.suffix->second);
        }
        getImpl()->visit(*funcDef.suite);
    }

    void visit(const ClassDef& classDef)
    {
        std::for_each(classDef.decorators.begin(), classDef.decorators.end(),
                      [&](const Decorator& decorator) { getImpl()->visit(decorator); });
        if (classDef.inheritance && classDef.inheritance->argumentList)
        {
            getImpl()->visit(*classDef.inheritance->argumentList);
        }
        getImpl()->visit(*classDef.suite);
    }

    void visit(const AsyncForStmt& asyncForStmt)
    {
        getImpl()->visit(asyncForStmt.forStmt);
    }

    void visit(const AsyncWithStmt& asyncWithStmt)
    {
        getImpl()->visit(asyncWithStmt.withStmt);
    }

    void visit(const CompoundStmt& compoundStmt)
    {
        getImpl()->visit(compoundStmt.variant);
    }

    void visit(const Statement& statement)
    {
        pylir::match(
            statement.variant, [&](const Statement::SingleLine& singleLine) { getImpl()->visit(singleLine.stmtList); },
            [&](const CompoundStmt& compoundStmt) { getImpl()->visit(compoundStmt); });
    }

    void visit(const Suite& suite)
    {
        pylir::match(
            suite.variant, [&](const Suite::SingleLine& singleLine) { getImpl()->visit(singleLine.stmtList); },
            [&](const Suite::MultiLine& multiLine)
            {
                for (const auto& iter : multiLine.statements)
                {
                    getImpl()->visit(iter);
                }
            });
    }

    void visit(const FileInput& fileInput)
    {
        for (const auto& iter : fileInput.input)
        {
            if (const auto* statement = std::get_if<Statement>(&iter))
            {
                getImpl()->visit(*statement);
            }
        }
    }
};
} // namespace pylir::Syntax
