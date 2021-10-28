#include "CodeGen.hpp"

#include <mlir/Dialect/StandardOps/IR/Ops.h>

#include <llvm/ADT/ScopeExit.h>

#include <pylir/Optimizer/PylirPy/IR/PylirPyAttributes.hpp>
#include <pylir/Optimizer/PylirPy/IR/PylirPyDialect.hpp>
#include <pylir/Optimizer/PylirPy/IR/PylirPyOps.hpp>
#include <pylir/Parser/Visitor.hpp>
#include <pylir/Support/ValueReset.hpp>

pylir::CodeGen::CodeGen(mlir::MLIRContext* context, Diag::Document& document)
    : m_builder(
        [&]
        {
            context->loadDialect<pylir::Py::PylirPyDialect>();
            context->loadDialect<mlir::StandardOpsDialect>();
            return context;
        }()),
      m_module(mlir::ModuleOp::create(m_builder.getUnknownLoc())),
      m_document(&document)
{
    for (auto& iter : Builtins::allBuiltins)
    {
        if (!iter.isPublic)
        {
            continue;
        }
        auto pos = iter.name.find_last_of('.');
        if (pos == std::string_view::npos)
        {
            pos = 0;
        }
        else
        {
            pos++;
        }
        m_builtinNamespace.emplace(iter.name.substr(pos), iter.name);
    }
}

mlir::ModuleOp pylir::CodeGen::visit(const pylir::Syntax::FileInput& fileInput)
{
    m_builder.setInsertionPointToEnd(m_module.getBody());
    createBuiltinsImpl();

    for (auto& token : fileInput.globals)
    {
        auto op = m_builder.create<Py::GlobalHandleOp>(getLoc(token, token), formQualifiedName(token.getValue()),
                                                       mlir::StringAttr{});
        getCurrentScope().emplace(token.getValue(), Identifier{Kind::Global, op.getOperation()});
    }

    auto initFunc = mlir::FuncOp::create(m_builder.getUnknownLoc(), formQualifiedName("__init__"),
                                         mlir::FunctionType::get(m_builder.getContext(), {}, {}));
    auto reset = implementFunction(initFunc);

    for (auto& iter : fileInput.input)
    {
        if (auto* statement = std::get_if<Syntax::Statement>(&iter))
        {
            visit(*statement);
            if (!m_builder.getInsertionBlock())
            {
                break;
            }
        }
    }
    if (needsTerminator())
    {
        m_builder.create<mlir::ReturnOp>(m_builder.getUnknownLoc());
    }

    return m_module;
}

void pylir::CodeGen::visit(const Syntax::Statement& statement)
{
    if (!m_builder.getInsertionBlock())
    {
        return;
    }
    pylir::match(
        statement.variant, [&](const Syntax::CompoundStmt& compoundStmt) { visit(compoundStmt); },
        [&](const Syntax::Statement::SingleLine& singleLine) { visit(singleLine.stmtList); });
}

void pylir::CodeGen::visit(const Syntax::StmtList& stmtList)
{
    if (!m_builder.getInsertionBlock())
    {
        return;
    }
    visit(*stmtList.firstExpr);
    for (auto& iter : stmtList.remainingExpr)
    {
        if (!m_builder.getInsertionBlock())
        {
            return;
        }
        visit(*iter.second);
    }
}

void pylir::CodeGen::visit(const Syntax::CompoundStmt& compoundStmt)
{
    pylir::match(compoundStmt.variant, [&](const auto& value) { visit(value); });
}

void pylir::CodeGen::visit(const Syntax::SimpleStmt& simpleStmt)
{
    pylir::match(
        simpleStmt.variant,
        [](const auto&)
        {
            // TODO
            PYLIR_UNREACHABLE;
        },
        [&](const Syntax::RaiseStmt& raiseStmt)
        {
            if (!raiseStmt.expressions)
            {
                // TODO: Get current exception via sys.exc_info()
                PYLIR_UNREACHABLE;
            }
            auto expression = visit(raiseStmt.expressions->first);
            if (!expression)
            {
                return;
            }
            // TODO: attach __cause__ and __context__
            auto loc = getLoc(raiseStmt, raiseStmt.raise);
            auto typeOf = m_builder.create<Py::TypeOfOp>(loc, expression);
            auto typeObject = m_builder.create<Py::GetGlobalValueOp>(loc, Builtins::Type.name);
            auto isTypeSubclass = buildSubclassCheck(loc, typeOf, typeObject);
            BlockPtr isType, instanceBlock;
            instanceBlock->addArgument(m_builder.getType<Py::DynamicType>());
            m_builder.create<mlir::CondBranchOp>(loc, isTypeSubclass, isType, instanceBlock,
                                                 mlir::ValueRange{expression});

            {
                implementBlock(isType);
                auto baseException = m_builder.create<Py::GetGlobalValueOp>(loc, Builtins::BaseException.name);
                auto isBaseException = buildSubclassCheck(loc, expression, baseException);
                BlockPtr typeError, createException;
                m_builder.create<mlir::CondBranchOp>(loc, isBaseException, createException, typeError);

                {
                    implementBlock(typeError);
                    auto exception = buildException(loc, Builtins::TypeError.name, {});
                    raiseException(exception);
                }

                implementBlock(createException);
                auto tuple = m_builder.create<Py::ConstantOp>(loc, Py::TupleAttr::get(m_builder.getContext(), {}));
                auto dict = m_builder.create<Py::ConstantOp>(loc, Py::DictAttr::get(m_builder.getContext(), {}));
                auto exception = buildCall(loc, expression, tuple, dict);
                m_builder.create<mlir::BranchOp>(loc, instanceBlock, mlir::ValueRange{exception});
            }

            implementBlock(instanceBlock);
            typeOf = m_builder.create<Py::TypeOfOp>(loc, instanceBlock->getArgument(0));
            auto baseException = m_builder.create<Py::GetGlobalValueOp>(loc, Builtins::BaseException.name);
            auto isBaseException = buildSubclassCheck(loc, typeOf, baseException);
            BlockPtr typeError, raiseBlock;
            m_builder.create<mlir::CondBranchOp>(loc, isBaseException, raiseBlock, typeError);

            {
                implementBlock(typeError);
                auto exception = buildException(loc, Builtins::TypeError.name, {});
                raiseException(exception);
            }

            implementBlock(raiseBlock);
            raiseException(instanceBlock->getArgument(0));
        },
        [&](const Syntax::ReturnStmt& returnStmt)
        {
            auto loc = getLoc(returnStmt, returnStmt.returnKeyword);
            if (!returnStmt.expressions)
            {
                executeFinallyBlocks(false);
                auto none = m_builder.create<Py::GetGlobalValueOp>(loc, Builtins::None.name);
                m_builder.create<mlir::ReturnOp>(loc, mlir::ValueRange{none});
                m_builder.clearInsertionPoint();
                return;
            }
            auto value = visit(*returnStmt.expressions);
            if (!value)
            {
                return;
            }
            executeFinallyBlocks(true);
            m_builder.create<mlir::ReturnOp>(loc, mlir::ValueRange{value});
            m_builder.clearInsertionPoint();
        },
        [&](const Syntax::BreakStmt& breakStmt)
        {
            executeFinallyBlocks();
            auto loc = getLoc(breakStmt, breakStmt.breakKeyword);
            m_builder.create<mlir::BranchOp>(loc, m_currentLoop.breakBlock);
            m_builder.clearInsertionPoint();
        },
        [&](const Syntax::ContinueStmt& continueStmt)
        {
            executeFinallyBlocks();
            auto loc = getLoc(continueStmt, continueStmt.continueKeyword);
            m_builder.create<mlir::BranchOp>(loc, m_currentLoop.continueBlock);
            m_builder.clearInsertionPoint();
        },
        [&](const Syntax::NonLocalStmt&) {},
        [&](const Syntax::GlobalStmt& globalStmt)
        {
            if (m_scope.size() == 1)
            {
                return;
            }
            auto handleIdentifier = [&](const IdentifierToken& token)
            {
                auto result = m_scope[0].find(token.getValue());
                PYLIR_ASSERT(result != m_scope[0].end());
                getCurrentScope().insert(*result);
            };
            handleIdentifier(globalStmt.identifier);
            for (auto& [token, identifier] : globalStmt.rest)
            {
                (void)token;
                handleIdentifier(identifier);
            }
        },
        [&](const Syntax::PassStmt&) {}, [&](const Syntax::StarredExpression& expression) { visit(expression); },
        [&](const Syntax::AssignmentStmt& statement) { visit(statement); });
}

void pylir::CodeGen::assignTarget(const Syntax::Target& target, mlir::Value value)
{
    pylir::match(
        target.variant, [&](const IdentifierToken& identifierToken) { writeIdentifier(identifierToken, value); },
        [&](const Syntax::Target::Parenth& parenth)
        {
            if (parenth.targetList)
            {
                assignTarget(*parenth.targetList, value);
                return;
            }
            // TODO
            PYLIR_UNREACHABLE;
        },
        [&](const Syntax::Subscription& subscription)
        {
            auto container = visit(*subscription.primary);
            if (!container)
            {
                return;
            }
            auto indices = visit(subscription.expressionList);
            if (!container)
            {
                return;
            }

            auto loc = getLoc(subscription, subscription);
            auto type = m_builder.create<Py::TypeOfOp>(loc, container);
            buildSpecialMethodCall(
                loc, "__setitem__", type,
                m_builder.create<Py::MakeTupleOp>(loc, std::vector<Py::IterArg>{container, indices, value}),
                m_builder.create<Py::MakeDictOp>(loc));
        },
        [&](const Syntax::Target::Square& square)
        {
            if (square.targetList)
            {
                assignTarget(*square.targetList, value);
                return;
            }
            // TODO
            PYLIR_UNREACHABLE;
        },
        [&](const auto&)
        {
            // TODO
            PYLIR_UNREACHABLE;
        });
}

void pylir::CodeGen::assignTarget(const Syntax::TargetList& targetList, mlir::Value value)
{
    if (targetList.remainingExpr.empty() && !targetList.trailingComma)
    {
        assignTarget(*targetList.firstExpr, value);
        return;
    }
    // TODO
    PYLIR_UNREACHABLE;
}

void pylir::CodeGen::visit(const Syntax::AssignmentStmt& assignmentStmt)
{
    auto rhs = pylir::match(assignmentStmt.variant, [&](const auto& value) { return visit(value); });
    if (!rhs)
    {
        return;
    }
    for (auto& [list, token] : assignmentStmt.targets)
    {
        assignTarget(list, rhs);
        if (!m_builder.getInsertionBlock())
        {
            return;
        }
    }
}

template <mlir::Value (pylir::CodeGen::*op)(mlir::Location, const std::vector<pylir::Py::IterArg>&)>
mlir::Value pylir::CodeGen::visit(const Syntax::StarredList& starredList)
{
    auto loc = getLoc(starredList, starredList);
    std::vector<Py::IterArg> operands;
    auto handleItem = [&](const Syntax::StarredItem& item)
    {
        return pylir::match(
            item.variant,
            [&](const Syntax::AssignmentExpression& assignment)
            {
                auto value = visit(assignment);
                if (!value)
                {
                    return false;
                }
                operands.push_back(value);
                return true;
            },
            [&](const std::pair<BaseToken, Syntax::OrExpr>& pair)
            {
                auto value = visit(pair.second);
                if (!value)
                {
                    return false;
                }
                operands.push_back(Py::IterExpansion{value});
                return true;
            });
    };
    if (!handleItem(*starredList.firstExpr))
    {
        return {};
    }
    for (auto& iter : starredList.remainingExpr)
    {
        if (!handleItem(*iter.second))
        {
            return {};
        }
    }
    return std::invoke(op, *this, loc, operands);
}

mlir::Value pylir::CodeGen::visit(const Syntax::StarredExpression& starredExpression)
{
    return pylir::match(
        starredExpression.variant,
        [&](const Syntax::StarredExpression::Items& items) -> mlir::Value
        {
            if (items.leading.empty())
            {
                if (!items.last)
                {
                    return {};
                }
                return visit(pylir::get<Syntax::AssignmentExpression>(items.last->variant));
            }
            auto loc = getLoc(starredExpression, starredExpression);
            std::vector<Py::IterArg> operands;
            auto handleItem = [&](const Syntax::StarredItem& item)
            {
                return pylir::match(
                    item.variant,
                    [&](const Syntax::AssignmentExpression& assignment)
                    {
                        auto value = visit(assignment);
                        if (!value)
                        {
                            return false;
                        }
                        operands.emplace_back(value);
                        return true;
                    },
                    [&](const std::pair<BaseToken, Syntax::OrExpr>& pair)
                    {
                        auto value = visit(pair.second);
                        if (!value)
                        {
                            return false;
                        }
                        operands.emplace_back(Py::IterExpansion{value});
                        return true;
                    });
            };
            for (auto& iter : items.leading)
            {
                if (!handleItem(iter.first))
                {
                    return {};
                }
            }
            if (items.last)
            {
                if (!handleItem(*items.last))
                {
                    return {};
                }
            }
            return makeTuple(loc, operands);
        },
        [&](const Syntax::Expression& expression) { return visit(expression); });
}

mlir::Value pylir::CodeGen::visit(const Syntax::ExpressionList& expressionList)
{
    if (!expressionList.trailingComma && expressionList.remainingExpr.empty())
    {
        return visit(*expressionList.firstExpr);
    }

    auto loc = getLoc(expressionList, expressionList);
    std::vector<mlir::Value> operands(1 + expressionList.remainingExpr.size());
    operands[0] = visit(*expressionList.firstExpr);
    if (!operands[0])
    {
        return {};
    }
    for (auto& iter : llvm::enumerate(expressionList.remainingExpr))
    {
        auto value = visit(*iter.value().second);
        if (!value)
        {
            return {};
        }
        operands[iter.index() + 1] = value;
    }
    return m_builder.create<Py::MakeTupleOp>(loc, operands, m_builder.getI32ArrayAttr({}));
}

mlir::Value pylir::CodeGen::visit(const Syntax::YieldExpression& yieldExpression)
{
    return {};
}

mlir::Value pylir::CodeGen::visit(const Syntax::Expression& expression)
{
    return pylir::match(
        expression.variant,
        [&](const Syntax::ConditionalExpression& conditionalExpression) { return visit(conditionalExpression); },
        [&](const std::unique_ptr<Syntax::LambdaExpression>& lambdaExpression) -> mlir::Value
        {
            // TODO return visit(*lambdaExpression);
            PYLIR_UNREACHABLE;
        });
}

mlir::Value pylir::CodeGen::visit(const Syntax::ConditionalExpression& expression)
{
    if (!expression.suffix)
    {
        return visit(expression.value);
    }
    auto loc = getLoc(expression, expression.suffix->ifToken);
    auto condition = toI1(loc, visit(*expression.suffix->test));
    if (!condition)
    {
        return {};
    }
    auto found = BlockPtr{};
    auto elseBlock = BlockPtr{};
    auto thenBlock = BlockPtr{};
    thenBlock->addArgument(m_builder.getType<Py::DynamicType>());

    m_builder.create<mlir::CondBranchOp>(loc, condition, found, elseBlock);

    implementBlock(found);
    auto trueValue = visit(expression.value);
    if (trueValue)
    {
        m_builder.create<mlir::BranchOp>(loc, thenBlock, trueValue);
    }

    implementBlock(elseBlock);
    auto falseValue = visit(*expression.suffix->elseValue);
    if (falseValue)
    {
        m_builder.create<mlir::BranchOp>(loc, thenBlock, falseValue);
    }

    if (thenBlock->hasNoPredecessors())
    {
        return {};
    }
    implementBlock(thenBlock);
    return thenBlock->getArgument(0);
}

mlir::Value pylir::CodeGen::visit(const pylir::Syntax::OrTest& expression)
{
    return pylir::match(
        expression.variant, [&](const Syntax::AndTest& andTest) { return visit(andTest); },
        [&](const std::unique_ptr<Syntax::OrTest::BinOp>& binOp) -> mlir::Value
        {
            auto loc = getLoc(expression, binOp->orToken);
            auto lhs = visit(*binOp->lhs);
            if (!lhs)
            {
                return {};
            }
            auto found = BlockPtr{};
            found->addArgument(m_builder.getType<Py::DynamicType>());
            auto rhsTry = BlockPtr{};
            m_builder.create<mlir::CondBranchOp>(loc, toI1(loc, lhs), found, lhs, rhsTry, mlir::ValueRange{});

            implementBlock(rhsTry);
            auto rhs = visit(binOp->rhs);
            if (rhs)
            {
                m_builder.create<mlir::BranchOp>(loc, found, rhs);
            }

            implementBlock(found);
            return found->getArgument(0);
        });
}

mlir::Value pylir::CodeGen::visit(const pylir::Syntax::AndTest& expression)
{
    return pylir::match(
        expression.variant, [&](const Syntax::NotTest& notTest) { return visit(notTest); },
        [&](const std::unique_ptr<Syntax::AndTest::BinOp>& binOp) -> mlir::Value
        {
            auto loc = getLoc(expression, binOp->andToken);
            auto lhs = visit(*binOp->lhs);
            if (!lhs)
            {
                return {};
            }
            auto found = BlockPtr{};
            found->addArgument(m_builder.getType<Py::DynamicType>());
            auto rhsTry = BlockPtr{};
            m_builder.create<mlir::CondBranchOp>(loc, toI1(loc, lhs), rhsTry, mlir::ValueRange{}, found,
                                                 mlir::ValueRange{lhs});

            implementBlock(rhsTry);
            auto rhs = visit(binOp->rhs);
            if (rhs)
            {
                m_builder.create<mlir::BranchOp>(loc, found, rhs);
            }

            implementBlock(found);
            return found->getArgument(0);
        });
}

mlir::Value pylir::CodeGen::visit(const Syntax::NotTest& expression)
{
    return pylir::match(
        expression.variant, [&](const Syntax::Comparison& comparison) { return visit(comparison); },
        [&](const std::pair<BaseToken, std::unique_ptr<Syntax::NotTest>>& pair) -> mlir::Value
        {
            auto loc = getLoc(expression, pair.first);
            auto value = toI1(loc, visit(*pair.second));
            auto one = m_builder.create<mlir::ConstantOp>(loc, m_builder.getBoolAttr(true));
            auto inverse = m_builder.create<mlir::XOrOp>(loc, one, value);
            return m_builder.create<Py::BoolFromI1Op>(loc, inverse);
        });
}

mlir::Value pylir::CodeGen::visit(const pylir::Syntax::Comparison& comparison)
{
    if (comparison.rest.empty())
    {
        return visit(comparison.left);
    }
    mlir::Value result;
    auto first = visit(comparison.left);
    if (!first)
    {
        return {};
    }
    auto previousRHS = first;
    for (auto& [op, rhs] : comparison.rest)
    {
        auto loc = getLoc(op.firstToken, op.firstToken);

        BlockPtr found;
        if (result)
        {
            found->addArgument(m_builder.getType<Py::DynamicType>());
            auto rhsTry = BlockPtr{};
            m_builder.create<mlir::CondBranchOp>(loc, toI1(loc, result), found, result, rhsTry, mlir::ValueRange{});
            implementBlock(rhsTry);
        }

        enum class Comp
        {
            Lt,
            Gt,
            Eq,
            Ne,
            Ge,
            Le,
            Is,
            In,
        };
        bool invert = false;
        Comp comp;
        switch (op.firstToken.getTokenType())
        {
            case TokenType::LessThan: comp = Comp::Lt; break;
            case TokenType::LessOrEqual: comp = Comp::Le; break;
            case TokenType::GreaterThan: comp = Comp::Gt; break;
            case TokenType::GreaterOrEqual: comp = Comp::Ge; break;
            case TokenType::Equal: comp = Comp::Eq; break;
            case TokenType::NotEqual: comp = Comp::Ne; break;
            case TokenType::IsKeyword: comp = Comp::Is; break;
            case TokenType::InKeyword: comp = Comp::In; break;
            default: PYLIR_UNREACHABLE;
        }
        if (op.secondToken)
        {
            invert = true;
        }
        auto other = visit(rhs);
        if (other)
        {
            mlir::Value cmp;
            /* TODO:
            switch (comp)
            {
                case Comp::Lt: cmp = m_builder.create<Py::LessOp>(loc, previousRHS, other); break;
                case Comp::Gt: cmp = m_builder.create<Py::GreaterOp>(loc, previousRHS, other); break;
                case Comp::Eq: cmp = m_builder.create<Py::EqualOp>(loc, previousRHS, other); break;
                case Comp::Ne: cmp = m_builder.create<Py::NotEqualOp>(loc, previousRHS, other); break;
                case Comp::Ge: cmp = m_builder.create<Py::GreaterEqualOp>(loc, previousRHS, other); break;
                case Comp::Le: cmp = m_builder.create<Py::LessEqualOp>(loc, previousRHS, other); break;
                case Comp::Is:
                    cmp = m_builder.create<Py::BoolFromI1Op>(loc, m_builder.create<Py::IsOp>(loc, previousRHS, other));
                    break;
                case Comp::In: cmp = m_builder.create<Py::InOp>(loc, previousRHS, other); break;
            }
            if (invert)
            {
                cmp = m_builder.create<Py::InvertOp>(loc, toBool(cmp));
            }
             */
            previousRHS = other;
            if (!result)
            {
                result = cmp;
                continue;
            }
            m_builder.create<mlir::BranchOp>(loc, found, cmp);
        }

        implementBlock(found);
        result = found->getArgument(0);
        if (!other)
        {
            break;
        }
    }
    return result;
}

mlir::Value pylir::CodeGen::visit(const Syntax::OrExpr& orExpr)
{
    return pylir::match(
        orExpr.variant, [&](const Syntax::XorExpr& xorExpr) { return visit(xorExpr); },
        [&](const std::unique_ptr<Syntax::OrExpr::BinOp>& binOp) -> mlir::Value
        {
            auto lhs = visit(*binOp->lhs);
            auto rhs = visit(binOp->rhs);
            auto loc = getLoc(orExpr, binOp->bitOrToken);
            // TODO: return m_builder.create<Py::OrOp>(loc, lhs, rhs);
            PYLIR_UNREACHABLE;
        });
}

mlir::Value pylir::CodeGen::visit(const Syntax::XorExpr& xorExpr)
{
    return pylir::match(
        xorExpr.variant, [&](const Syntax::AndExpr& andExpr) { return visit(andExpr); },
        [&](const std::unique_ptr<Syntax::XorExpr::BinOp>& binOp) -> mlir::Value
        {
            auto lhs = visit(*binOp->lhs);
            auto rhs = visit(binOp->rhs);
            auto loc = getLoc(xorExpr, binOp->bitXorToken);
            // TODO: return m_builder.create<Py::XorOp>(loc, lhs, rhs);
            PYLIR_UNREACHABLE;
        });
}

mlir::Value pylir::CodeGen::visit(const Syntax::AndExpr& andExpr)
{
    return pylir::match(
        andExpr.variant, [&](const Syntax::ShiftExpr& shiftExpr) { return visit(shiftExpr); },
        [&](const std::unique_ptr<Syntax::AndExpr::BinOp>& binOp) -> mlir::Value
        {
            auto lhs = visit(*binOp->lhs);
            auto rhs = visit(binOp->rhs);
            auto loc = getLoc(andExpr, binOp->bitAndToken);
            // TODO: return m_builder.create<Py::AndOp>(loc, lhs, rhs);
            PYLIR_UNREACHABLE;
        });
}

mlir::Value pylir::CodeGen::visit(const Syntax::ShiftExpr& shiftExpr)
{
    return pylir::match(
        shiftExpr.variant, [&](const Syntax::AExpr& aExpr) { return visit(aExpr); },
        [&](const std::unique_ptr<Syntax::ShiftExpr::BinOp>& binOp) -> mlir::Value
        {
            auto lhs = visit(*binOp->lhs);
            auto rhs = visit(binOp->rhs);
            auto loc = getLoc(shiftExpr, binOp->binToken);
            switch (binOp->binToken.getTokenType())
            {
                case TokenType::ShiftLeft:  // TODO: return m_builder.create<Py::LShiftOp>(loc, lhs, rhs);
                case TokenType::ShiftRight: // TODO: return m_builder.create<Py::RShiftOp>(loc, lhs, rhs);
                default: PYLIR_UNREACHABLE;
            }
        });
}

mlir::Value pylir::CodeGen::visit(const Syntax::AExpr& aExpr)
{
    return pylir::match(
        aExpr.variant, [&](const Syntax::MExpr& mExpr) { return visit(mExpr); },
        [&](const std::unique_ptr<Syntax::AExpr::BinOp>& binOp) -> mlir::Value
        {
            auto lhs = visit(*binOp->lhs);
            auto rhs = visit(binOp->rhs);
            auto loc = getLoc(aExpr, binOp->binToken);
            switch (binOp->binToken.getTokenType())
            {
                case TokenType::Plus:  // TODO: return m_builder.create<Py::AddOp>(loc, lhs, rhs);
                case TokenType::Minus: // TODO: return m_builder.create<Py::SubOp>(loc, lhs, rhs);
                default: PYLIR_UNREACHABLE;
            }
        });
}

mlir::Value pylir::CodeGen::visit(const Syntax::MExpr& mExpr)
{
    return pylir::match(
        mExpr.variant, [&](const Syntax::UExpr& uExpr) { return visit(uExpr); },
        [&](const std::unique_ptr<Syntax::MExpr::BinOp>& binOp) -> mlir::Value
        {
            auto lhs = visit(*binOp->lhs);
            auto rhs = visit(binOp->rhs);
            auto loc = getLoc(mExpr, binOp->binToken);
            switch (binOp->binToken.getTokenType())
            {
                case TokenType::Star:      // TODO: return m_builder.create<Py::MulOp>(loc, lhs, rhs);
                case TokenType::Divide:    // TODO: return m_builder.create<Py::TrueDivOp>(loc, lhs, rhs);
                case TokenType::IntDivide: // TODO: return m_builder.create<Py::FloorDivOp>(loc, lhs, rhs);
                case TokenType::Remainder: // TODO: return m_builder.create<Py::ModuloOp>(loc, lhs, rhs);
                default: PYLIR_UNREACHABLE;
            }
        },
        [&](const std::unique_ptr<Syntax::MExpr::AtBin>& atBin) -> mlir::Value
        {
            auto lhs = visit(*atBin->lhs);
            auto rhs = visit(*atBin->rhs);
            auto loc = getLoc(mExpr, atBin->atToken);
            // TODO: return m_builder.create<Py::MatMulOp>(loc, lhs, rhs);
            PYLIR_UNREACHABLE;
        });
}

mlir::Value pylir::CodeGen::visit(const Syntax::UExpr& uExpr)
{
    return pylir::match(
        uExpr.variant, [&](const Syntax::Power& power) { return visit(power); },
        [&](const std::pair<Token, std::unique_ptr<Syntax::UExpr>>& pair) -> mlir::Value
        {
            auto value = visit(*pair.second);
            switch (pair.first.getTokenType())
            {
                case TokenType::Minus:
                {
                    // TODO
                    PYLIR_UNREACHABLE;
                }
                case TokenType::Plus:

                    // TODO
                    PYLIR_UNREACHABLE;
                case TokenType::BitNegate:

                    // TODO
                    PYLIR_UNREACHABLE;
                default: PYLIR_UNREACHABLE;
            }
        });
}

mlir::Value pylir::CodeGen::visit(const Syntax::Power& power)
{
    if (!power.rightHand)
    {
        return pylir::match(power.variant, [&](const auto& value) { return visit(value); });
    }
    // TODO
    PYLIR_UNREACHABLE;
}

mlir::Value pylir::CodeGen::visit(const Syntax::AwaitExpr& awaitExpr)
{
    // TODO
    PYLIR_UNREACHABLE;
}

mlir::Value pylir::CodeGen::visit(const Syntax::Primary& primary)
{
    return pylir::match(
        primary.variant, [&](const Syntax::Atom& atom) { return visit(atom); },
        [&](const Syntax::Subscription& subscription) { return visit(subscription); },
        [&](const Syntax::Call& call) -> mlir::Value
        {
            auto loc = getLoc(call, call.openParentheses);
            auto callable = visit(*call.primary);
            if (!callable)
            {
                return {};
            }
            auto [tuple, keywords] = pylir::match(
                call.variant,
                [&](std::monostate) -> std::pair<mlir::Value, mlir::Value> {
                    return {m_builder.create<Py::MakeTupleOp>(loc), m_builder.create<Py::MakeDictOp>(loc)};
                },
                [&](const std::pair<Syntax::ArgumentList, std::optional<BaseToken>>& pair)
                    -> std::pair<mlir::Value, mlir::Value> { return visit(pair.first); },
                [&](const std::unique_ptr<Syntax::Comprehension>& comprehension) -> std::pair<mlir::Value, mlir::Value>
                {
                    // TODO:
                    PYLIR_UNREACHABLE;
                });
            if (!tuple || !keywords)
            {
                return {};
            }
            return buildCall(loc, callable, tuple, keywords);
        },
        [&](const auto&) -> mlir::Value
        {
            // TODO
            PYLIR_UNREACHABLE;
        });
}

void pylir::CodeGen::writeIdentifier(const IdentifierToken& identifierToken, mlir::Value value)
{
    auto loc = getLoc(identifierToken, identifierToken);
    if (m_classNamespace)
    {
        auto str = m_builder.create<Py::ConstantOp>(loc, m_builder.getStringAttr(identifierToken.getValue()));
        m_builder.create<Py::DictSetItemOp>(loc, m_classNamespace, str, value);
        return;
    }

    auto result = getCurrentScope().find(identifierToken.getValue());
    // Should not be possible
    PYLIR_ASSERT(result != getCurrentScope().end());

    mlir::Value handle;
    switch (result->second.kind)
    {
        case Global:
            handle = m_builder.create<Py::GetGlobalHandleOp>(
                loc, m_builder.getSymbolRefAttr(result->second.op.get<mlir::Operation*>()));
            break;
        case StackAlloc: handle = result->second.op.get<mlir::Value>(); break;
        case Cell:
            m_builder.create<Py::SetAttrOp>(loc, value, result->second.op.get<mlir::Value>(), "cell_contents");
            return;
    }
    m_builder.create<Py::StoreOp>(loc, value, handle);
}

mlir::Value pylir::CodeGen::readIdentifier(const IdentifierToken& identifierToken)
{
    auto loc = getLoc(identifierToken, identifierToken);
    BlockPtr classNamespaceFound;
    ScopeContainer::value_type* scope;
    if (m_classNamespace)
    {
        classNamespaceFound->addArgument(m_builder.getType<Py::DynamicType>());
        auto str = m_builder.create<Py::ConstantOp>(loc, m_builder.getStringAttr(identifierToken.getValue()));
        auto tryGet = m_builder.create<Py::DictTryGetItemOp>(loc, m_classNamespace, str);
        auto elseBlock = BlockPtr{};
        m_builder.create<mlir::CondBranchOp>(loc, tryGet.found(), classNamespaceFound, tryGet.result(), elseBlock,
                                             mlir::ValueRange{});
        implementBlock(elseBlock);

        // if not found in locals, it does not import free variables but rather goes straight to the global scope
        scope = &m_scope[0];
    }
    else
    {
        scope = &getCurrentScope();
    }
    auto result = scope->find(identifierToken.getValue());
    if (result == scope->end())
    {
        if (auto builtin = m_builtinNamespace.find(identifierToken.getValue()); builtin != m_builtinNamespace.end())
        {
            auto builtinValue = m_builder.create<Py::GetGlobalValueOp>(loc, builtin->second);
            if (!m_classNamespace)
            {
                return builtinValue;
            }
            m_builder.create<mlir::BranchOp>(loc, classNamespaceFound, mlir::ValueRange{builtinValue});
            implementBlock(classNamespaceFound);
            return classNamespaceFound->getArgument(0);
        }
        else
        {
            auto exception = buildException(loc, Builtins::NameError.name, /*TODO: string arg*/ {});
            raiseException(exception);
            if (!m_classNamespace)
            {
                // TODO: This could probably lead to issues? As this is an expression and would reset the insertion
                //       point as well as return a null mlir::Value.
                return {};
            }
        }
        implementBlock(classNamespaceFound);
        return classNamespaceFound->getArgument(0);
    }
    mlir::Value handle;
    switch (result->second.kind)
    {
        case Global:
            handle = m_builder.create<Py::GetGlobalHandleOp>(
                loc, m_builder.getSymbolRefAttr(result->second.op.get<mlir::Operation*>()));
            break;
        case StackAlloc: handle = result->second.op.get<mlir::Value>(); break;
        case Cell:
        {
            auto getAttrOp =
                m_builder.create<Py::GetAttrOp>(loc, result->second.op.get<mlir::Value>(), "cell_contents");
            auto success = BlockPtr{};
            auto failure = BlockPtr{};
            m_builder.create<mlir::CondBranchOp>(loc, getAttrOp.success(), success, failure);

            implementBlock(failure);
            auto exception = buildException(loc, Builtins::UnboundLocalError.name, /*TODO: string arg*/ {});
            raiseException(exception);

            implementBlock(success);
            return getAttrOp.result();
        }
    }
    auto condition = m_builder.create<Py::IsUnboundHandleOp>(loc, handle);
    auto unbound = BlockPtr{};
    auto found = BlockPtr{};
    m_builder.create<mlir::CondBranchOp>(loc, condition, unbound, found);

    implementBlock(unbound);
    if (result->second.kind == Global)
    {
        auto exception = buildException(loc, Builtins::NameError.name, /*TODO: string arg*/ {});
        raiseException(exception);
    }
    else
    {
        auto exception = buildException(loc, Builtins::UnboundLocalError.name, /*TODO: string arg*/ {});
        raiseException(exception);
    }

    implementBlock(found);
    if (!m_classNamespace)
    {
        return m_builder.create<Py::LoadOp>(loc, handle);
    }
    m_builder.create<mlir::BranchOp>(loc, classNamespaceFound,
                                     mlir::ValueRange{m_builder.create<Py::LoadOp>(loc, handle)});

    implementBlock(classNamespaceFound);
    return classNamespaceFound->getArgument(0);
}

mlir::Value pylir::CodeGen::visit(const pylir::Syntax::Atom& atom)
{
    return pylir::match(
        atom.variant,
        [&](const Syntax::Atom::Literal& literal) -> mlir::Value
        {
            auto location = getLoc(atom, literal.token);
            switch (literal.token.getTokenType())
            {
                case TokenType::IntegerLiteral:
                {
                    return m_builder.create<Py::ConstantOp>(
                        location,
                        Py::IntAttr::get(m_builder.getContext(), pylir::get<BigInt>(literal.token.getValue())));
                }
                case TokenType::FloatingPointLiteral:
                {
                    return m_builder.create<Py::ConstantOp>(
                        location, m_builder.getF64FloatAttr(pylir::get<double>(literal.token.getValue())));
                }
                case TokenType::ComplexLiteral:
                {
                    // TODO:
                    PYLIR_UNREACHABLE;
                }
                case TokenType::StringLiteral:
                {
                    return m_builder.create<Py::ConstantOp>(
                        location, m_builder.getStringAttr(pylir::get<std::string>(literal.token.getValue())));
                }
                case TokenType::ByteLiteral:
                    // TODO:
                    PYLIR_UNREACHABLE;
                case TokenType::TrueKeyword:
                {
                    return m_builder.create<Py::ConstantOp>(location, Py::BoolAttr::get(m_builder.getContext(), true));
                }
                case TokenType::FalseKeyword:
                {
                    return m_builder.create<Py::ConstantOp>(location, Py::BoolAttr::get(m_builder.getContext(), false));
                }
                case TokenType::NoneKeyword:
                    return m_builder.create<Py::GetGlobalValueOp>(location, Builtins::None.name);
                default: PYLIR_UNREACHABLE;
            }
        },
        [&](const IdentifierToken& identifierToken) -> mlir::Value { return readIdentifier(identifierToken); },
        [&](const std::unique_ptr<Syntax::Enclosure>& enclosure) -> mlir::Value { return visit(*enclosure); });
}

mlir::Value pylir::CodeGen::visit(const pylir::Syntax::Subscription& subscription)
{
    auto container = visit(*subscription.primary);
    if (!container)
    {
        return {};
    }
    auto indices = visit(subscription.expressionList);
    if (!container)
    {
        return {};
    }

    auto loc = getLoc(subscription, subscription);
    auto type = m_builder.create<Py::TypeOfOp>(loc, container);
    return buildSpecialMethodCall(loc, "__getitem__", type,
                                  m_builder.create<Py::MakeTupleOp>(loc, std::vector<Py::IterArg>{container, indices}),
                                  m_builder.create<Py::MakeDictOp>(loc));
}

mlir::Value pylir::CodeGen::toI1(mlir::Location loc, mlir::Value value)
{
    auto boolean = toBool(loc, value);
    return m_builder.create<Py::BoolToI1Op>(boolean.getLoc(), boolean);
}

mlir::Value pylir::CodeGen::toBool(mlir::Location loc, mlir::Value value)
{
    auto type = m_builder.create<Py::TypeOfOp>(loc, value);
    auto tuple = m_builder.create<Py::MakeTupleOp>(loc, std::vector<Py::IterArg>{value});
    auto maybeBool =
        buildSpecialMethodCall(loc, "__bool__", type, tuple,
                               m_builder.create<Py::ConstantOp>(loc, Py::DictAttr::get(m_builder.getContext(), {})));
    auto typeOfResult = m_builder.create<Py::TypeOfOp>(loc, maybeBool);
    auto booleanType = m_builder.create<Py::GetGlobalValueOp>(loc, Builtins::Bool.name);
    auto isBool = m_builder.create<Py::IsOp>(loc, typeOfResult, booleanType);
    BlockPtr isBoolBlock, typeErrorBlock;
    m_builder.create<mlir::CondBranchOp>(loc, isBool, isBoolBlock, typeErrorBlock);

    implementBlock(typeErrorBlock);
    auto exception = buildException(loc, Builtins::TypeError.name, {});
    raiseException(exception);

    implementBlock(isBoolBlock);
    return maybeBool;
}

mlir::Value pylir::CodeGen::visit(const pylir::Syntax::Enclosure& enclosure)
{
    return pylir::match(
        enclosure.variant,
        [&](const Syntax::Enclosure::ParenthForm& parenthForm) -> mlir::Value
        {
            if (parenthForm.expression)
            {
                return visit(*parenthForm.expression);
            }
            auto loc = getLoc(parenthForm, parenthForm.openParenth);
            return m_builder.create<Py::MakeTupleOp>(loc);
        },
        [&](const Syntax::Enclosure::ListDisplay& listDisplay) -> mlir::Value
        {
            auto loc = getLoc(listDisplay, listDisplay.openSquare);
            return pylir::match(
                listDisplay.variant,
                [&](std::monostate) -> mlir::Value { return m_builder.create<Py::MakeListOp>(loc); },
                [&](const Syntax::StarredList& list) -> mlir::Value { return visit<&CodeGen::makeList>(list); },
                [&](const Syntax::Comprehension&) -> mlir::Value
                {
                    // TODO:
                    PYLIR_UNREACHABLE;
                });
        },
        [&](const Syntax::Enclosure::SetDisplay& setDisplay) -> mlir::Value
        {
            auto loc = getLoc(setDisplay, setDisplay.openBrace);
            return pylir::match(
                setDisplay.variant,
                [&](const Syntax::StarredList& list) -> mlir::Value { return visit<&CodeGen::makeSet>(list); },
                [&](const Syntax::Comprehension&) -> mlir::Value
                {
                    // TODO:
                    PYLIR_UNREACHABLE;
                });
        },
        [&](const Syntax::Enclosure::DictDisplay& dictDisplay) -> mlir::Value
        {
            auto loc = getLoc(dictDisplay, dictDisplay.openBrace);
            return pylir::match(
                dictDisplay.variant,
                [&](std::monostate) -> mlir::Value
                { return m_builder.create<Py::MakeDictOp>(loc, std::vector<Py::DictArg>{}); },
                [&](const Syntax::CommaList<Syntax::Enclosure::DictDisplay::KeyDatum>& list) -> mlir::Value
                {
                    std::vector<Py::DictArg> result;
                    result.reserve(list.remainingExpr.size() + 1);
                    auto handleOne = [&](const Syntax::Enclosure::DictDisplay::KeyDatum& keyDatum)
                    {
                        return pylir::match(
                            keyDatum.variant,
                            [&](const Syntax::Enclosure::DictDisplay::KeyDatum::Key& key)
                            {
                                auto first = visit(key.first);
                                if (!first)
                                {
                                    return false;
                                }
                                auto second = visit(key.second);
                                if (!second)
                                {
                                    return false;
                                }
                                result.push_back(std::pair{first, second});
                                return true;
                            },
                            [&](const Syntax::Enclosure::DictDisplay::KeyDatum::Datum& key)
                            {
                                auto mapping = visit(key.orExpr);
                                if (!mapping)
                                {
                                    return false;
                                }
                                result.push_back(Py::MappingExpansion{mapping});
                                return true;
                            });
                    };
                    if (!handleOne(*list.firstExpr))
                    {
                        return {};
                    }
                    for (auto& [token, iter] : list.remainingExpr)
                    {
                        (void)token;
                        if (!handleOne(*iter))
                        {
                            return {};
                        }
                    }
                    return m_builder.create<Py::MakeDictOp>(loc, result);
                },
                [&](const Syntax::Enclosure::DictDisplay::DictComprehension&) -> mlir::Value
                {
                    // TODO:
                    PYLIR_UNREACHABLE;
                });
        },
        [&](const auto&) -> mlir::Value
        {
            // TODO:
            PYLIR_UNREACHABLE;
        });
}

mlir::Value pylir::CodeGen::visit(const pylir::Syntax::AssignmentExpression& assignmentExpression)
{
    if (!assignmentExpression.identifierAndWalrus)
    {
        return visit(*assignmentExpression.expression);
    }
    auto value = visit(*assignmentExpression.expression);
    if (!value)
    {
        return {};
    }
    writeIdentifier(assignmentExpression.identifierAndWalrus->first, value);
    return value;
}

void pylir::CodeGen::visit(const Syntax::IfStmt& ifStmt)
{
    auto condition = visit(ifStmt.condition);
    if (!condition)
    {
        return;
    }
    auto trueBlock = BlockPtr{};
    BlockPtr thenBlock;
    auto exitBlock = llvm::make_scope_exit(
        [&]
        {
            if (!thenBlock->hasNoPredecessors())
            {
                implementBlock(thenBlock);
            }
        });
    mlir::Block* elseBlock;
    if (!ifStmt.elseSection && ifStmt.elifs.empty())
    {
        elseBlock = thenBlock;
    }
    else
    {
        elseBlock = new mlir::Block;
    }
    auto loc = getLoc(ifStmt.ifKeyword, ifStmt.ifKeyword);
    m_builder.create<mlir::CondBranchOp>(loc, toI1(loc, condition), trueBlock, elseBlock);

    implementBlock(trueBlock);
    visit(*ifStmt.suite);
    if (needsTerminator())
    {
        m_builder.create<mlir::BranchOp>(loc, thenBlock);
    }
    if (thenBlock == elseBlock)
    {
        return;
    }
    implementBlock(elseBlock);
    for (auto& iter : llvm::enumerate(ifStmt.elifs))
    {
        loc = getLoc(iter.value().elif, iter.value().elif);
        condition = visit(iter.value().condition);
        if (!condition)
        {
            return;
        }
        trueBlock = BlockPtr{};
        if (iter.index() == ifStmt.elifs.size() - 1 && !ifStmt.elseSection)
        {
            elseBlock = thenBlock;
        }
        else
        {
            elseBlock = new mlir::Block;
        }

        m_builder.create<mlir::CondBranchOp>(loc, toI1(loc, condition), trueBlock, elseBlock);

        implementBlock(trueBlock);
        visit(*iter.value().suite);
        if (needsTerminator())
        {
            m_builder.create<mlir::BranchOp>(loc, thenBlock);
        }
        if (thenBlock != elseBlock)
        {
            implementBlock(elseBlock);
        }
    }
    if (ifStmt.elseSection)
    {
        visit(*ifStmt.elseSection->suite);
        if (needsTerminator())
        {
            m_builder.create<mlir::BranchOp>(loc, thenBlock);
        }
    }
}

void pylir::CodeGen::visit(const Syntax::WhileStmt& whileStmt)
{
    auto loc = getLoc(whileStmt, whileStmt.whileKeyword);
    auto conditionBlock = BlockPtr{};
    auto thenBlock = BlockPtr{};
    auto exitBlock = llvm::make_scope_exit(
        [&]
        {
            if (!thenBlock->hasNoPredecessors())
            {
                implementBlock(thenBlock);
            }
        });
    m_builder.create<mlir::BranchOp>(loc, conditionBlock);

    implementBlock(conditionBlock);
    auto condition = visit(whileStmt.condition);
    if (!condition)
    {
        return;
    }
    mlir::Block* elseBlock;
    if (whileStmt.elseSection)
    {
        elseBlock = new mlir::Block;
    }
    else
    {
        elseBlock = thenBlock;
    }
    auto body = BlockPtr{};
    m_builder.create<mlir::CondBranchOp>(loc, toI1(loc, condition), body, elseBlock);

    implementBlock(body);
    std::optional exit = pylir::ValueReset(m_currentLoop);
    m_currentLoop = {thenBlock, conditionBlock};
    visit(*whileStmt.suite);
    if (needsTerminator())
    {
        m_builder.create<mlir::BranchOp>(loc, conditionBlock);
    }
    exit.reset();
    if (elseBlock == thenBlock)
    {
        return;
    }
    implementBlock(elseBlock);
    visit(*whileStmt.elseSection->suite);
    if (needsTerminator())
    {
        m_builder.create<mlir::BranchOp>(loc, thenBlock);
    }
}

void pylir::CodeGen::visit(const pylir::Syntax::ForStmt& forStmt)
{
    auto iterable = visit(forStmt.expressionList);
    if (!iterable)
    {
        return;
    }
    auto loc = getLoc(forStmt, forStmt.forKeyword);
    auto type = m_builder.create<Py::TypeOfOp>(loc, iterable);
    auto typeMRO = m_builder.create<Py::GetAttrOp>(loc, type, "__mro__").result();
    auto iterMethod = m_builder.create<Py::MROLookupOp>(loc, typeMRO, "__iter__");
    BlockPtr notIterableBlock, iterableBlock;
    m_builder.create<mlir::CondBranchOp>(loc, iterMethod.success(), iterableBlock, notIterableBlock);

    {
        implementBlock(notIterableBlock);
        auto exception = buildException(loc, Builtins::TypeError.name, {});
        raiseException(exception);
    }

    implementBlock(iterableBlock);
    auto iterObject = buildCall(loc, iterMethod.result(),
                                m_builder.create<Py::MakeTupleOp>(loc, std::vector<Py::IterArg>{iterMethod.result()}),
                                m_builder.create<Py::ConstantOp>(loc, Py::DictAttr::get(m_builder.getContext(), {})));
    auto nextMethod = m_builder.create<Py::MROLookupOp>(loc, typeMRO, "__next__");
    BlockPtr notNextBlock, condition;
    m_builder.create<mlir::CondBranchOp>(loc, nextMethod.success(), condition, notNextBlock);

    {
        implementBlock(notNextBlock);
        auto exception = buildException(loc, Builtins::TypeError.name, {});
        raiseException(exception);
    }

    implementBlock(condition);
    BlockPtr exceptionHandler, thenBlock;
    auto implementThenBlock = llvm::make_scope_exit(
        [&]
        {
            if (!thenBlock->hasNoPredecessors())
            {
                implementBlock(thenBlock);
            }
        });

    exceptionHandler->addArgument(m_builder.getType<Py::DynamicType>());
    std::optional reset = pylir::ValueReset(m_currentExceptBlock);
    m_currentExceptBlock = exceptionHandler;
    auto next =
        buildCall(loc, nextMethod.result(),
                  m_builder.create<Py::MakeTupleOp>(loc, std::vector<Py::IterArg>{nextMethod.result(), iterObject}),
                  m_builder.create<Py::ConstantOp>(loc, Py::DictAttr::get(m_builder.getContext(), {})));
    reset.reset();
    assignTarget(forStmt.targetList, next);
    mlir::Block* elseBlock;
    if (forStmt.elseSection)
    {
        elseBlock = new mlir::Block;
    }
    else
    {
        elseBlock = thenBlock;
    }
    BlockPtr body;
    m_builder.create<mlir::BranchOp>(loc, body);

    implementBlock(body);
    std::optional exit = pylir::ValueReset(m_currentLoop);
    m_currentLoop = {thenBlock, condition};
    visit(*forStmt.suite);
    if (needsTerminator())
    {
        m_builder.create<mlir::BranchOp>(loc, condition);
    }
    exit.reset();
    if (!exceptionHandler->hasNoPredecessors())
    {
        implementBlock(exceptionHandler);
        auto exception = exceptionHandler->getArgument(0);
        auto exceptionType = m_builder.create<Py::TypeOfOp>(loc, exception);
        auto stopIteration = m_builder.create<Py::GetGlobalValueOp>(loc, Builtins::StopIteration.name);
        auto isStopIteration = buildSubclassCheck(loc, exceptionType, stopIteration);
        BlockPtr reraiseBlock, exitBlock;
        m_builder.create<mlir::CondBranchOp>(loc, isStopIteration, exitBlock, reraiseBlock);

        implementBlock(reraiseBlock);
        raiseException(exception);

        implementBlock(exitBlock);
        m_builder.create<mlir::BranchOp>(loc, elseBlock);
    }
    if (elseBlock == thenBlock)
    {
        return;
    }
    implementBlock(elseBlock);
    visit(*forStmt.elseSection->suite);
    if (needsTerminator())
    {
        m_builder.create<mlir::BranchOp>(loc, thenBlock);
    }
}

void pylir::CodeGen::visit(const pylir::Syntax::TryStmt& tryStmt)
{
    BlockPtr exceptionHandler;
    exceptionHandler->addArgument(m_builder.getType<Py::DynamicType>());
    std::optional reset = pylir::ValueReset(m_currentExceptBlock);
    auto lambda = [&] { m_finallyBlocks.pop_back(); };
    std::optional<decltype(llvm::make_scope_exit(lambda))> popFinally;
    if (tryStmt.finally)
    {
        m_finallyBlocks.push_back({&*tryStmt.finally, m_currentLoop, m_currentExceptBlock});
        popFinally.emplace(llvm::make_scope_exit(lambda));
    }
    m_currentExceptBlock = exceptionHandler;
    visit(*tryStmt.suite);

    auto enterFinallyCode = [&]
    {
        auto back = m_finallyBlocks.back();
        m_finallyBlocks.pop_back();
        auto tuple = std::make_tuple(llvm::make_scope_exit([back, this] { m_finallyBlocks.push_back(back); }),
                                     pylir::ValueReset(m_currentExceptBlock));
        m_currentExceptBlock = back.parentExceptBlock;
        return tuple;
    };

    if (needsTerminator())
    {
        if (tryStmt.elseSection)
        {
            visit(*tryStmt.elseSection->suite);
            if (needsTerminator() && tryStmt.finally)
            {
                auto finalSection = enterFinallyCode();
                visit(*tryStmt.finally->suite);
            }
        }
        else if (tryStmt.finally)
        {
            auto finalSection = enterFinallyCode();
            visit(*tryStmt.finally->suite);
        }
    }

    BlockPtr continueBlock;
    auto exitBlock = llvm::make_scope_exit(
        [&]
        {
            if (!continueBlock->hasNoPredecessors())
            {
                implementBlock(continueBlock);
            }
        });
    if (needsTerminator())
    {
        m_builder.create<mlir::BranchOp>(getLoc(tryStmt, tryStmt.tryKeyword), continueBlock);
    }

    if (exceptionHandler->hasNoPredecessors())
    {
        return;
    }

    implementBlock(exceptionHandler);
    // Exceptions thrown in exception handlers (including the expression after except) are propagated upwards and not
    // handled by this block
    reset.reset();

    for (auto& iter : tryStmt.excepts)
    {
        auto loc = getLoc(iter, iter.exceptKeyword);
        if (!iter.expression)
        {
            visit(*iter.suite);
            if (needsTerminator())
            {
                if (tryStmt.finally)
                {
                    auto finallySection = enterFinallyCode();
                    visit(*tryStmt.finally->suite);
                }
                if (needsTerminator())
                {
                    m_builder.create<mlir::BranchOp>(loc, continueBlock);
                }
            }
            continue;
        }
        auto value = visit(iter.expression->first);
        if (!value)
        {
            return;
        }
        if (iter.expression->second)
        {
            // TODO: Python requires this identifier to be unbound at the end of the exception handler as if done in
            //       a finally section
            writeIdentifier(iter.expression->second->second, exceptionHandler->getArgument(0));
        }
        auto tupleType = m_builder.create<Py::GetGlobalValueOp>(loc, Builtins::Tuple.name);
        auto isTuple = m_builder.create<Py::IsOp>(loc, m_builder.create<Py::TypeOfOp>(loc, value), tupleType);
        auto tupleBlock = BlockPtr{};
        auto exceptionBlock = BlockPtr{};
        m_builder.create<mlir::CondBranchOp>(loc, isTuple, tupleBlock, exceptionBlock);

        BlockPtr skipBlock;
        BlockPtr suiteBlock;
        {
            implementBlock(exceptionBlock);
            // TODO: check value is a type
            auto baseException = m_builder.create<Py::GetGlobalValueOp>(loc, Builtins::BaseException.name);
            auto isSubclass = buildSubclassCheck(loc, value, baseException);
            BlockPtr raiseBlock;
            BlockPtr noTypeErrorBlock;
            m_builder.create<mlir::CondBranchOp>(loc, isSubclass, noTypeErrorBlock, raiseBlock);

            implementBlock(raiseBlock);
            auto exception = buildException(loc, Builtins::TypeError.name, {});
            raiseException(exception);

            implementBlock(noTypeErrorBlock);
            auto exceptionType = m_builder.create<Py::TypeOfOp>(loc, exceptionHandler->getArgument(0));
            isSubclass = buildSubclassCheck(loc, exceptionType, value);
            m_builder.create<mlir::CondBranchOp>(loc, isSubclass, suiteBlock, skipBlock);
        }
        {
            implementBlock(tupleBlock);
            auto baseException = m_builder.create<Py::GetGlobalValueOp>(loc, Builtins::BaseException.name);
            BlockPtr noTypeErrorsBlock;
            buildTupleForEach(loc, value, noTypeErrorsBlock, {},
                              [&](mlir::Value entry)
                              {
                                  // TODO: check entry is a type
                                  auto isSubclass = buildSubclassCheck(loc, entry, baseException);
                                  BlockPtr raiseBlock;
                                  BlockPtr noTypeErrorBlock;
                                  m_builder.create<mlir::CondBranchOp>(loc, isSubclass, noTypeErrorBlock, raiseBlock);

                                  implementBlock(raiseBlock);
                                  auto exception = buildException(loc, Builtins::TypeError.name, {});
                                  raiseException(exception);

                                  implementBlock(noTypeErrorBlock);
                              });
            implementBlock(noTypeErrorsBlock);
            auto exceptionType = m_builder.create<Py::TypeOfOp>(loc, exceptionHandler->getArgument(0));
            buildTupleForEach(loc, value, skipBlock, {},
                              [&](mlir::Value entry)
                              {
                                  auto isSubclass = buildSubclassCheck(loc, exceptionType, entry);
                                  BlockPtr continueLoop;
                                  m_builder.create<mlir::CondBranchOp>(loc, isSubclass, suiteBlock, continueLoop);
                                  implementBlock(continueLoop);
                              });
        }

        implementBlock(suiteBlock);
        visit(*iter.suite);
        if (needsTerminator())
        {
            if (tryStmt.finally)
            {
                auto finallySection = enterFinallyCode();
                visit(*tryStmt.finally->suite);
            }
            if (needsTerminator())
            {
                m_builder.create<mlir::BranchOp>(loc, continueBlock);
            }
        }
        implementBlock(skipBlock);
    }
    if (needsTerminator())
    {
        if (tryStmt.finally)
        {
            auto finallyCode = enterFinallyCode();
            visit(*tryStmt.finally->suite);
        }
        if (needsTerminator())
        {
            m_builder.create<Py::RaiseOp>(getLoc(tryStmt, tryStmt.tryKeyword), exceptionHandler->getArgument(0));
        }
    }
}

void pylir::CodeGen::visit(const pylir::Syntax::WithStmt& withStmt) {}

void pylir::CodeGen::visit(const pylir::Syntax::FuncDef& funcDef)
{
    std::vector<Py::IterArg> defaultParameters;
    std::vector<Py::DictArg> keywordOnlyDefaultParameters;
    std::vector<IdentifierToken> functionParametersTokens;
    std::vector<FunctionParameter> functionParameters;
    if (funcDef.parameterList)
    {
        class ParamVisitor : public Syntax::Visitor<ParamVisitor>
        {
        public:
            std::vector<Py::IterArg>& defaultParameters;
            std::vector<Py::DictArg>& keywordOnlyDefaultParameters;
            std::vector<FunctionParameter>& functionParameters;
            std::vector<IdentifierToken>& functionParametersTokens;
            std::function<mlir::Value(const Syntax::Expression&)> calcCallback;
            mlir::OpBuilder& builder;
            std::function<mlir::Location(const IdentifierToken&)> locCallback;
            FunctionParameter::Kind kind{};

            using Visitor::visit;

            void visit(const Syntax::ParameterList::Parameter& param)
            {
                if (!builder.getInsertionBlock())
                {
                    return;
                }
                functionParameters.push_back({std::string(param.identifier.getValue()), kind, false});
                functionParametersTokens.push_back(param.identifier);
            }

            void visit(const Syntax::ParameterList::DefParameter& defParameter)
            {
                if (!builder.getInsertionBlock())
                {
                    return;
                }
                Visitor::visit(defParameter);
                if (!defParameter.defaultArg)
                {
                    return;
                }
                if (!builder.getInsertionBlock())
                {
                    return;
                }
                functionParameters.back().hasDefaultParam = true;
                auto value = calcCallback(defParameter.defaultArg->second);
                if (!value)
                {
                    return;
                }
                if (kind != FunctionParameter::KeywordOnly)
                {
                    defaultParameters.push_back(value);
                    return;
                }
                auto name =
                    builder.create<Py::ConstantOp>(locCallback(defParameter.parameter.identifier),
                                                   builder.getStringAttr(defParameter.parameter.identifier.getValue()));
                keywordOnlyDefaultParameters.push_back(std::pair{name, value});
            }

            void visit(const Syntax::ParameterList::PosOnly& posOnlyNode)
            {
                if (!builder.getInsertionBlock())
                {
                    return;
                }
                kind = FunctionParameter::PosOnly;
                Visitor::visit(posOnlyNode);
            }

            void visit(const Syntax::ParameterList::NoPosOnly& noPosOnly)
            {
                if (!builder.getInsertionBlock())
                {
                    return;
                }
                kind = FunctionParameter::Normal;
                Visitor::visit(noPosOnly);
            }

            void visit(const Syntax::ParameterList::StarArgs& star)
            {
                if (!builder.getInsertionBlock())
                {
                    return;
                }
                auto doubleStarHandler = [&](const Syntax::ParameterList::StarArgs::DoubleStar& doubleStar)
                {
                    visit(doubleStar.parameter);
                    if (!builder.getInsertionBlock())
                    {
                        return;
                    }
                    functionParameters.back().kind = FunctionParameter::KeywordRest;
                };
                pylir::match(
                    star.variant,
                    [&](const Syntax::ParameterList::StarArgs::Star& star)
                    {
                        kind = FunctionParameter::KeywordOnly;
                        if (star.parameter)
                        {
                            visit(*star.parameter);
                            if (!builder.getInsertionBlock())
                            {
                                return;
                            }
                            functionParameters.back().kind = FunctionParameter::PosRest;
                        }
                        for (auto& iter : llvm::make_second_range(star.defParameters))
                        {
                            visit(iter);
                            if (!builder.getInsertionBlock())
                            {
                                return;
                            }
                        }
                        if (star.further && star.further->doubleStar)
                        {
                            doubleStarHandler(*star.further->doubleStar);
                        }
                    },
                    doubleStarHandler);
            }
        } visitor{{},
                  defaultParameters,
                  keywordOnlyDefaultParameters,
                  functionParameters,
                  functionParametersTokens,
                  [this](const Syntax::Expression& expression) { return visit(expression); },
                  m_builder,
                  [this](const IdentifierToken& token) { return getLoc(token, token); }};
        visitor.visit(*funcDef.parameterList);
        if (!m_builder.getInsertionBlock())
        {
            return;
        }
    }
    auto loc = getLoc(funcDef.funcName, funcDef.funcName);
    auto qualifiedName = formQualifiedName(std::string(funcDef.funcName.getValue()));
    std::vector<IdentifierToken> usedClosures;
    mlir::FuncOp func;
    {
        pylir::ValueReset namespaceReset(m_classNamespace);
        m_classNamespace = {};
        func = mlir::FuncOp::create(
            loc, formImplName(qualifiedName + "$impl"),
            m_builder.getFunctionType(
                std::vector<mlir::Type>(1 + functionParameters.size(), m_builder.getType<Py::DynamicType>()),
                {m_builder.getType<Py::DynamicType>()}));
        func.setVisibility(mlir::SymbolTable::Visibility::Private);
        auto reset = implementFunction(func);

        m_scope.emplace_back();
        m_qualifierStack.emplace_back(funcDef.funcName.getValue());
        m_qualifierStack.push_back("<locals>");
        auto exit = llvm::make_scope_exit(
            [&]
            {
                m_scope.pop_back();
                m_qualifierStack.pop_back();
                m_qualifierStack.pop_back();
            });
        auto locals = funcDef.localVariables;
        auto closures = funcDef.closures;
        for (auto [name, value] : llvm::zip(functionParametersTokens, llvm::drop_begin(func.getArguments())))
        {
            if (funcDef.closures.count(name))
            {
                auto closureType = m_builder.create<Py::GetGlobalValueOp>(loc, Builtins::Cell.name);
                auto tuple = m_builder.create<Py::MakeTupleOp>(loc, std::vector<Py::IterArg>{closureType, value});
                auto emptyDict = m_builder.create<Py::ConstantOp>(loc, Py::DictAttr::get(m_builder.getContext(), {}));
                auto newMethod = m_builder.create<Py::GetAttrOp>(loc, closureType, "__new__").result();
                auto cell =
                    m_builder
                        .create<mlir::CallIndirectOp>(loc, m_builder.create<Py::FunctionGetFunctionOp>(loc, newMethod),
                                                      mlir::ValueRange{newMethod, tuple, emptyDict})
                        ->getResult(0);
                m_scope.back().emplace(name.getValue(), Identifier{Kind::Cell, cell});
                closures.erase(name);
            }
            else
            {
                auto allocaOp = m_builder.create<Py::AllocaOp>(getLoc(name, name));
                m_scope.back().emplace(name.getValue(), Identifier{Kind::StackAlloc, mlir::Value{allocaOp}});
                m_builder.create<Py::StoreOp>(loc, value, allocaOp);
                locals.erase(name);
            }
        }
        for (auto& iter : locals)
        {
            m_scope.back().emplace(
                iter.getValue(),
                Identifier{Kind::StackAlloc, mlir::Value{m_builder.create<Py::AllocaOp>(getLoc(iter, iter))}});
        }
        for (auto& iter : closures)
        {
            auto closureType = m_builder.create<Py::GetGlobalValueOp>(loc, Builtins::Cell.name);
            auto tuple = m_builder.create<Py::MakeTupleOp>(loc, std::vector<Py::IterArg>{closureType});
            auto emptyDict = m_builder.create<Py::ConstantOp>(loc, Py::DictAttr::get(m_builder.getContext(), {}));
            auto newMethod = m_builder.create<Py::GetAttrOp>(loc, closureType, "__new__").result();
            auto cell =
                m_builder
                    .create<mlir::CallIndirectOp>(loc, m_builder.create<Py::FunctionGetFunctionOp>(loc, newMethod),
                                                  mlir::ValueRange{newMethod, tuple, emptyDict})
                    ->getResult(0);
            m_scope.back().emplace(iter.getValue(), Identifier{Kind::Cell, mlir::Value{cell}});
        }
        if (!funcDef.nonLocalVariables.empty())
        {
            auto self = func.getArgument(0);
            auto closureTuple = m_builder.create<Py::GetAttrOp>(loc, self, "__closure__");
            for (auto& iter : llvm::enumerate(funcDef.nonLocalVariables))
            {
                auto constant = m_builder.create<mlir::ConstantOp>(loc, m_builder.getIndexAttr(iter.index()));
                auto cell = m_builder.create<Py::TupleIntegerGetItemOp>(loc, closureTuple.result(), constant);
                m_scope.back().emplace(iter.value().getValue(), Identifier{Kind::Cell, mlir::Value{cell}});
                usedClosures.push_back(iter.value());
            }
        }

        visit(*funcDef.suite);
        if (needsTerminator())
        {
            m_builder.create<mlir::ReturnOp>(
                loc, mlir::ValueRange{m_builder.create<Py::GetGlobalValueOp>(loc, Builtins::None.name)});
        }
        func = buildFunctionCC(loc, formImplName(qualifiedName + "$cc"), func, functionParameters);
    }
    mlir::Value value = m_builder.create<Py::MakeFuncOp>(loc, m_builder.getSymbolRefAttr(func));
    m_builder.create<Py::SetAttrOp>(
        loc, m_builder.create<Py::ConstantOp>(loc, m_builder.getStringAttr(funcDef.funcName.getValue())), value,
        m_builder.getStringAttr("__name__"));
    m_builder.create<Py::SetAttrOp>(loc, m_builder.create<Py::ConstantOp>(loc, m_builder.getStringAttr(qualifiedName)),
                                    value, "__qualname__");
    {
        mlir::Value defaults;
        if (defaultParameters.empty())
        {
            defaults = m_builder.create<Py::GetGlobalValueOp>(loc, Builtins::None.name);
        }
        else
        {
            defaults = m_builder.create<Py::MakeTupleOp>(loc, defaultParameters);
        }
        m_builder.create<Py::SetAttrOp>(loc, defaults, value, "__defaults__");
    }
    {
        mlir::Value kwDefaults;
        if (keywordOnlyDefaultParameters.empty())
        {
            kwDefaults = m_builder.create<Py::GetGlobalValueOp>(loc, Builtins::None.name);
        }
        else
        {
            kwDefaults = m_builder.create<Py::MakeDictOp>(loc, keywordOnlyDefaultParameters);
        }
        m_builder.create<Py::SetAttrOp>(loc, kwDefaults, value, "__kwdefaults__");
    }
    {
        mlir::Value closure;
        if (usedClosures.empty())
        {
            closure = m_builder.create<Py::GetGlobalValueOp>(loc, Builtins::None.name);
        }
        else
        {
            std::vector<Py::IterArg> args(usedClosures.size());
            std::transform(usedClosures.begin(), usedClosures.end(), args.begin(),
                           [&](const IdentifierToken& token) -> Py::IterArg
                           {
                               auto result = getCurrentScope().find(token.getValue());
                               PYLIR_ASSERT(result != getCurrentScope().end());
                               return result->second.op.get<mlir::Value>();
                           });
            closure = m_builder.create<Py::MakeTupleOp>(loc, args);
        }
        m_builder.create<Py::SetAttrOp>(loc, closure, value, "__closure__");
    }
    for (auto& iter : llvm::reverse(funcDef.decorators))
    {
        auto decLoc = getLoc(iter.atSign, iter.atSign);
        auto decorator = visit(iter.assignmentExpression);
        if (!decorator)
        {
            return;
        }
        value = buildCall(decLoc, decorator, m_builder.create<Py::MakeTupleOp>(decLoc, std::vector<Py::IterArg>{value}),
                          m_builder.create<Py::ConstantOp>(decLoc, Py::DictAttr::get(m_builder.getContext(), {})));
    }
    writeIdentifier(funcDef.funcName, value);
}

void pylir::CodeGen::visit(const pylir::Syntax::ClassDef& classDef)
{
    auto loc = getLoc(classDef, classDef.className);
    mlir::Value bases, keywords;
    if (classDef.inheritance && classDef.inheritance->argumentList)
    {
        std::tie(bases, keywords) = visit(*classDef.inheritance->argumentList);
    }
    else
    {
        bases = m_builder.create<Py::ConstantOp>(loc, Py::TupleAttr::get(m_builder.getContext(), {}));
        keywords = m_builder.create<Py::ConstantOp>(loc, Py::DictAttr::get(m_builder.getContext(), {}));
    }
    auto qualifiedName = formQualifiedName(classDef.className.getValue());
    auto name = m_builder.create<Py::ConstantOp>(loc, m_builder.getStringAttr(qualifiedName));

    mlir::FuncOp func;
    {
        func = mlir::FuncOp::create(
            loc, formImplName(qualifiedName + "$impl"),
            m_builder.getFunctionType(
                std::vector<mlir::Type>(2 /* cell tuple + namespace dict */, m_builder.getType<Py::DynamicType>()),
                {m_builder.getType<Py::DynamicType>()}));
        func.setVisibility(mlir::SymbolTable::Visibility::Private);
        auto reset = implementFunction(func);
        m_scope.emplace_back();
        m_qualifierStack.emplace_back(classDef.className.getValue());
        auto exit = llvm::make_scope_exit(
            [&]
            {
                m_scope.pop_back();
                m_qualifierStack.pop_back();
            });
        pylir::ValueReset namespaceReset(m_classNamespace);
        m_classNamespace = func.getArgument(1);

        visit(*classDef.suite);
        m_builder.create<mlir::ReturnOp>(loc, m_classNamespace);
    }
    auto value = m_builder.create<Py::MakeClassOp>(loc, m_builder.getSymbolRefAttr(func), name, bases, keywords);
    writeIdentifier(classDef.className, value);
}

void pylir::CodeGen::visit(const pylir::Syntax::AsyncForStmt& asyncForStmt) {}

void pylir::CodeGen::visit(const pylir::Syntax::AsyncWithStmt& asyncWithStmt) {}

void pylir::CodeGen::visit(const Syntax::Suite& suite)
{
    pylir::match(
        suite.variant, [&](const Syntax::Suite::SingleLine& singleLine) { visit(singleLine.stmtList); },
        [&](const Syntax::Suite::MultiLine& singleLine)
        {
            for (auto& iter : singleLine.statements)
            {
                visit(iter);
            }
        });
}

std::pair<mlir::Value, mlir::Value> pylir::CodeGen::visit(const pylir::Syntax::ArgumentList& argumentList)
{
    auto loc = getLoc(argumentList, argumentList);
    std::vector<Py::IterArg> iterArgs;
    std::vector<Py::DictArg> dictArgs;
    if (argumentList.positionalArguments)
    {
        auto handlePositionalItem = [&](const Syntax::ArgumentList::PositionalItem& positionalItem)
        {
            return pylir::match(
                positionalItem.variant,
                [&](const std::unique_ptr<Syntax::AssignmentExpression>& expression)
                {
                    auto iter = visit(*expression);
                    if (!iter)
                    {
                        return false;
                    }
                    iterArgs.emplace_back(iter);
                    return true;
                },
                [&](const Syntax::ArgumentList::PositionalItem::Star& star)
                {
                    auto iter = visit(*star.expression);
                    if (!iter)
                    {
                        return false;
                    }
                    iterArgs.push_back(Py::IterExpansion{iter});
                    return true;
                });
        };
        if (!handlePositionalItem(argumentList.positionalArguments->firstItem))
        {
            return {{}, {}};
        }
        for (auto& [token, rest] : argumentList.positionalArguments->rest)
        {
            (void)token;
            if (!handlePositionalItem(rest))
            {
                return {{}, {}};
            }
        }
    }
    auto handleKeywordItem = [&](const Syntax::ArgumentList::KeywordItem& keywordItem)
    {
        auto key = m_builder.create<Py::ConstantOp>(getLoc(keywordItem.identifier, keywordItem.identifier),
                                                    m_builder.getStringAttr(keywordItem.identifier.getValue()));
        auto value = visit(*keywordItem.expression);
        if (!value)
        {
            return false;
        }
        dictArgs.push_back(std::pair{key, value});
        return true;
    };
    if (argumentList.starredAndKeywords)
    {
        auto handleExpression = [&](const Syntax::ArgumentList::StarredAndKeywords::Expression& expression)
        {
            auto value = visit(*expression.expression);
            if (!value)
            {
                return false;
            }
            iterArgs.push_back(Py::IterExpansion{value});
            return true;
        };
        auto handleStarredAndKeywords = [&](const Syntax::ArgumentList::StarredAndKeywords::Variant& variant)
        { return pylir::match(variant, handleKeywordItem, handleExpression); };
        if (!handleKeywordItem(argumentList.starredAndKeywords->first))
        {
            return {{}, {}};
        }
        for (auto& [token, variant] : argumentList.starredAndKeywords->rest)
        {
            (void)token;
            if (!handleStarredAndKeywords(variant))
            {
                return {{}, {}};
            }
        }
    }
    if (argumentList.keywordArguments)
    {
        auto handleExpression = [&](const Syntax::ArgumentList::KeywordArguments::Expression& expression)
        {
            auto value = visit(*expression.expression);
            if (!value)
            {
                return false;
            }
            dictArgs.push_back(Py::MappingExpansion{value});
            return true;
        };
        auto handleKeywordArguments = [&](const Syntax::ArgumentList::KeywordArguments::Variant& variant)
        { return pylir::match(variant, handleKeywordItem, handleExpression); };
        if (!handleExpression(argumentList.keywordArguments->first))
        {
            return {{}, {}};
        }
        for (auto& [token, variant] : argumentList.keywordArguments->rest)
        {
            (void)token;
            if (!handleKeywordArguments(variant))
            {
                return {{}, {}};
            }
        }
    }
    return {makeTuple(loc, iterArgs), makeDict(loc, dictArgs)};
}

std::string pylir::CodeGen::formQualifiedName(std::string_view symbol)
{
    if (m_qualifierStack.empty())
    {
        return std::string(symbol);
    }
    std::string result;
    {
        llvm::raw_string_ostream os(result);
        llvm::interleave(m_qualifierStack, os, ".");
        os << "." << symbol;
    }
    return result;
}

std::string pylir::CodeGen::formImplName(std::string_view symbol)
{
    auto result = std::string(symbol);
    auto& index = m_implNames[result];
    result += "[" + std::to_string(index) + "]";
    index++;
    return result;
}

mlir::Value pylir::CodeGen::buildException(mlir::Location loc, std::string_view kind, std::vector<Py::IterArg> args)
{
    auto typeObj = m_builder.create<Py::GetGlobalValueOp>(loc, kind);
    args.emplace(args.begin(), typeObj);
    auto tuple = makeTuple(loc, args);
    auto dict = m_builder.create<Py::ConstantOp>(loc, Py::DictAttr::get(m_builder.getContext(), {}));
    auto newMethod = m_builder.create<Py::GetAttrOp>(loc, typeObj, "__new__").result();

    auto obj = m_builder
                   .create<mlir::CallIndirectOp>(loc, m_builder.create<Py::FunctionGetFunctionOp>(loc, newMethod),
                                                 mlir::ValueRange{newMethod, tuple, dict})
                   ->getResult(0);
    auto context = m_builder.create<Py::GetGlobalValueOp>(loc, Builtins::None.name);
    m_builder.create<Py::SetAttrOp>(loc, context, obj, "__context__");
    auto cause = m_builder.create<Py::GetGlobalValueOp>(loc, Builtins::None.name);
    m_builder.create<Py::SetAttrOp>(loc, cause, obj, "__cause__");
    return obj;
}

void pylir::CodeGen::raiseException(mlir::Value exceptionObject)
{
    if (m_currentExceptBlock)
    {
        m_builder.create<mlir::BranchOp>(exceptionObject.getLoc(), m_currentExceptBlock, exceptionObject);
    }
    else
    {
        m_builder.create<Py::RaiseOp>(exceptionObject.getLoc(), exceptionObject);
    }
    m_builder.clearInsertionPoint();
}

mlir::Value pylir::CodeGen::buildCall(mlir::Location loc, mlir::Value callable, mlir::Value tuple, mlir::Value dict)
{
    BlockPtr typeCall, notBound;
    auto functionObj = m_builder.create<Py::GetFunctionOp>(loc, callable);
    m_builder.create<mlir::CondBranchOp>(loc, functionObj.success(), typeCall, notBound);

    implementBlock(notBound);
    auto typeError = buildException(loc, Builtins::TypeError.name, {});
    raiseException(typeError);

    implementBlock(typeCall);
    auto function = m_builder.create<Py::FunctionGetFunctionOp>(loc, functionObj.result());
    if (!m_currentExceptBlock)
    {
        return m_builder
            .create<mlir::CallIndirectOp>(loc, function, mlir::ValueRange{functionObj.result(), tuple, dict})
            .getResult(0);
    }
    auto happyPath = BlockPtr{};
    auto result =
        m_builder.create<Py::InvokeIndirectOp>(loc, function, mlir::ValueRange{functionObj.result(), tuple, dict},
                                               mlir::ValueRange{}, mlir::ValueRange{}, happyPath, m_currentExceptBlock);
    implementBlock(happyPath);
    return result;
}

mlir::Value pylir::CodeGen::buildSpecialMethodCall(mlir::Location loc, llvm::Twine methodName, mlir::Value type,
                                                   mlir::Value tuple, mlir::Value dict)
{
    auto mroTuple = m_builder.create<Py::GetAttrOp>(loc, type, "__mro__").result();
    auto lookup = m_builder.create<Py::MROLookupOp>(loc, mroTuple, methodName.str());
    auto notFound = BlockPtr{};
    auto exec = BlockPtr{};
    m_builder.create<mlir::CondBranchOp>(loc, lookup.success(), exec, notFound);

    implementBlock(notFound);
    auto exception = buildException(loc, Builtins::TypeError.name, {});
    raiseException(exception);

    implementBlock(exec);
    return buildCall(loc, lookup.result(), tuple, dict);
}

mlir::FuncOp pylir::CodeGen::buildFunctionCC(mlir::Location loc, llvm::Twine name, mlir::FuncOp implementation,
                                             const std::vector<FunctionParameter>& parameters)
{
    auto cc = mlir::FuncOp::create(
        loc, name.str(),
        m_builder.getFunctionType({m_builder.getType<Py::DynamicType>(), m_builder.getType<Py::DynamicType>(),
                                   m_builder.getType<Py::DynamicType>()},
                                  {m_builder.getType<Py::DynamicType>()}));
    cc.setVisibility(mlir::SymbolTable::Visibility::Private);
    auto reset = implementFunction(cc);

    auto self = cc.getArgument(0);
    auto tuple = cc.getArgument(1);
    auto dict = cc.getArgument(2);

    auto defaultTuple = m_builder.create<Py::GetAttrOp>(loc, self, "__defaults__").result();
    auto kwDefaultDict = m_builder.create<Py::GetAttrOp>(loc, self, "__kwdefaults__").result();
    auto tupleLen = m_builder.create<Py::TupleIntegerLenOp>(loc, m_builder.getIndexType(), tuple);

    std::vector<mlir::Value> args{self};
    std::size_t posIndex = 0;
    std::size_t posDefaultsIndex = 0;
    for (auto& iter : parameters)
    {
        mlir::Value argValue;
        switch (iter.kind)
        {
            case FunctionParameter::Normal:
            case FunctionParameter::PosOnly:
            {
                auto constant = m_builder.create<mlir::ConstantOp>(loc, m_builder.getIndexAttr(posIndex++));
                auto isLess = m_builder.create<mlir::CmpIOp>(loc, mlir::CmpIPredicate::ult, constant, tupleLen);
                auto lessBlock = BlockPtr{};
                auto unboundBlock = BlockPtr{};
                m_builder.create<mlir::CondBranchOp>(loc, isLess, lessBlock, unboundBlock);

                auto resultBlock = BlockPtr{};
                resultBlock->addArgument(m_builder.getType<Py::DynamicType>());
                implementBlock(unboundBlock);
                auto unboundValue = m_builder.create<Py::ConstantOp>(loc, Py::UnboundAttr::get(m_builder.getContext()));
                m_builder.create<mlir::BranchOp>(loc, resultBlock, mlir::ValueRange{unboundValue});

                implementBlock(lessBlock);
                auto fetched = m_builder.create<Py::TupleIntegerGetItemOp>(loc, tuple, constant);
                m_builder.create<mlir::BranchOp>(loc, resultBlock, mlir::ValueRange{fetched});

                implementBlock(resultBlock);
                argValue = resultBlock->getArgument(0);
                if (iter.kind == FunctionParameter::PosOnly)
                {
                    break;
                }
                [[fallthrough]];
            }
            case FunctionParameter::KeywordOnly:
            {
                auto constant = m_builder.create<Py::ConstantOp>(loc, m_builder.getStringAttr(iter.name));
                auto lookup = m_builder.create<Py::DictTryGetItemOp>(loc, dict, constant);
                auto foundBlock = BlockPtr{};
                auto notFoundBlock = BlockPtr{};
                m_builder.create<mlir::CondBranchOp>(loc, lookup.found(), foundBlock, notFoundBlock);

                auto resultBlock = BlockPtr{};
                resultBlock->addArgument(m_builder.getType<Py::DynamicType>());
                implementBlock(notFoundBlock);
                auto unboundValue = m_builder.create<Py::ConstantOp>(loc, Py::UnboundAttr::get(m_builder.getContext()));
                m_builder.create<mlir::BranchOp>(loc, resultBlock, mlir::ValueRange{unboundValue});

                implementBlock(foundBlock);
                m_builder.create<Py::DictDelItemOp>(loc, dict, constant);
                // value can't be assigned both through a positional argument as well as keyword argument
                if (argValue)
                {
                    auto isUnbound = m_builder.create<Py::IsUnboundValueOp>(loc, argValue);
                    auto boundBlock = BlockPtr{};
                    m_builder.create<mlir::CondBranchOp>(loc, isUnbound, resultBlock, mlir::ValueRange{lookup.result()},
                                                         boundBlock, mlir::ValueRange{});

                    implementBlock(boundBlock);
                    auto exception = buildException(loc, Builtins::TypeError.name, {});
                    raiseException(exception);
                }
                else
                {
                    m_builder.create<mlir::BranchOp>(loc, resultBlock, mlir::ValueRange{lookup.result()});
                }

                implementBlock(resultBlock);
                argValue = resultBlock->getArgument(0);
                break;
            }
            case FunctionParameter::PosRest:
            {
                // if posIndex is 0 then no other positional arguments existed. As a shortcut we can then simply assign
                // the tuple to argValue instead of creating it from a list
                if (posIndex == 0)
                {
                    argValue = tuple;
                    break;
                }
                auto list = m_builder.create<Py::MakeListOp>(loc);
                auto start = m_builder.create<mlir::ConstantOp>(loc, m_builder.getIndexAttr(posIndex));
                auto conditionBlock = BlockPtr{};
                conditionBlock->addArgument(m_builder.getIndexType());
                m_builder.create<mlir::BranchOp>(loc, conditionBlock, mlir::ValueRange{start});

                implementBlock(conditionBlock);
                auto isLess = m_builder.create<mlir::CmpIOp>(loc, mlir::CmpIPredicate::ult,
                                                             conditionBlock->getArgument(0), tupleLen);
                auto lessBlock = BlockPtr{};
                auto endBlock = BlockPtr{};
                m_builder.create<mlir::CondBranchOp>(loc, isLess, lessBlock, endBlock);

                implementBlock(lessBlock);
                auto fetched = m_builder.create<Py::TupleIntegerGetItemOp>(loc, tuple, conditionBlock->getArgument(0));
                m_builder.create<Py::ListAppendOp>(loc, list, fetched);
                auto one = m_builder.create<mlir::ConstantOp>(loc, m_builder.getIndexAttr(1));
                auto incremented = m_builder.create<mlir::AddIOp>(loc, conditionBlock->getArgument(0), one);
                m_builder.create<mlir::BranchOp>(loc, conditionBlock, mlir::ValueRange{incremented});

                implementBlock(endBlock);
                argValue = m_builder.create<Py::ListToTupleOp>(loc, list);
                break;
            }
            case FunctionParameter::KeywordRest: argValue = dict; break;
        }
        switch (iter.kind)
        {
            case FunctionParameter::PosOnly:
            case FunctionParameter::Normal:
            case FunctionParameter::KeywordOnly:
            {
                auto isUnbound = m_builder.create<Py::IsUnboundValueOp>(loc, argValue);
                auto unboundBlock = BlockPtr{};
                auto boundBlock = BlockPtr{};
                boundBlock->addArgument(m_builder.getType<Py::DynamicType>());
                m_builder.create<mlir::CondBranchOp>(loc, isUnbound, unboundBlock, boundBlock,
                                                     mlir::ValueRange{argValue});

                implementBlock(unboundBlock);
                if (!iter.hasDefaultParam)
                {
                    auto exception = buildException(loc, Builtins::TypeError.name, {});
                    raiseException(exception);
                }
                else
                {
                    mlir::Value defaultArg;
                    switch (iter.kind)
                    {
                        case FunctionParameter::Normal:
                        case FunctionParameter::PosOnly:
                        {
                            auto index =
                                m_builder.create<mlir::ConstantOp>(loc, m_builder.getIndexAttr(posDefaultsIndex++));
                            defaultArg = m_builder.create<Py::TupleIntegerGetItemOp>(loc, defaultTuple, index);
                            break;
                        }
                        case FunctionParameter::KeywordOnly:
                        {
                            auto index = m_builder.create<Py::ConstantOp>(loc, m_builder.getStringAttr(iter.name));
                            auto lookup = m_builder.create<Py::DictTryGetItemOp>(loc, kwDefaultDict, index);
                            // TODO: __kwdefaults__ is writeable. This may not hold. I have no clue how and whether this
                            // also
                            //      affects __defaults__
                            defaultArg = lookup.result();
                            break;
                        }
                        default: PYLIR_UNREACHABLE;
                    }
                    m_builder.create<mlir::BranchOp>(loc, boundBlock, mlir::ValueRange{defaultArg});
                }

                implementBlock(boundBlock);
                args.push_back(boundBlock->getArgument(0));
                break;
            }
            case FunctionParameter::PosRest:
            case FunctionParameter::KeywordRest: args.push_back(argValue); break;
        }
    }

    auto result = m_builder.create<mlir::CallOp>(loc, implementation, args);
    m_builder.create<mlir::ReturnOp>(loc, result->getResults());
    return cc;
}

void pylir::CodeGen::executeFinallyBlocks(bool fullUnwind)
{
    // This whole sequence here is made quite complicated due to a few reasons:
    // try statements can be nested and they can execute ANY code. Including function returns.
    // If we were to simply execute all finally blocks in reverse this could easily lead to an infinite recursion in
    // the following case:
    //
    // try:
    //      ...
    // finally:
    //      return
    //
    // The return would lead us to executeFinallyBlocks here and it'd once again generate the finally that we are
    // currently executing. For that reason we are saving the current finally stack, pop one and generate that, and at
    // the end restore it for future statements.
    //
    // Further care needs to be taken for `raise` inside of finally:
    //
    // def foo():
    //    try: #1
    //        try: #2
    //            return
    //        finally:
    //            raise ValueError
    //    except ValueError:
    //        return "caught"
    //    finally:
    //        raise ValueError
    //
    // The finallies are basically executed as if outside the try block (even if we don't generate them as such)
    // which means exceptions raised within them are propagated upwards and not handled by their exception handler
    // but the enclosing one (if it exists)
    auto copy = m_finallyBlocks;
    auto reset = llvm::make_scope_exit([&] { m_finallyBlocks = std::move(copy); });

    for (auto iter = copy.rbegin();
         iter != copy.rend() && (fullUnwind || iter->parentLoop == m_currentLoop) && needsTerminator(); iter++)
    {
        pylir::ValueReset exceptReset(m_currentExceptBlock);
        m_currentExceptBlock = iter->parentExceptBlock;
        m_finallyBlocks.pop_back();
        visit(*iter->finallySuite->suite);
    }
}

mlir::Value pylir::CodeGen::makeTuple(mlir::Location loc, const std::vector<Py::IterArg>& args)
{
    if (!m_currentExceptBlock)
    {
        return m_builder.create<Py::MakeTupleOp>(loc, args);
    }
    if (std::all_of(args.begin(), args.end(),
                    [](const Py::IterArg& arg) { return std::holds_alternative<mlir::Value>(arg); }))
    {
        return m_builder.create<Py::MakeTupleOp>(loc, args);
    }
    auto happyPath = BlockPtr{};
    auto result = m_builder.create<Py::MakeTupleExOp>(loc, args, happyPath, mlir::ValueRange{}, m_currentExceptBlock,
                                                      mlir::ValueRange{});
    implementBlock(happyPath);
    return result;
}

mlir::Value pylir::CodeGen::makeList(mlir::Location loc, const std::vector<Py::IterArg>& args)
{
    if (!m_currentExceptBlock)
    {
        return m_builder.create<Py::MakeListOp>(loc, args);
    }
    if (std::all_of(args.begin(), args.end(),
                    [](const Py::IterArg& arg) { return std::holds_alternative<mlir::Value>(arg); }))
    {
        return m_builder.create<Py::MakeListOp>(loc, args);
    }
    auto happyPath = BlockPtr{};
    auto result = m_builder.create<Py::MakeListExOp>(loc, args, happyPath, mlir::ValueRange{}, m_currentExceptBlock,
                                                     mlir::ValueRange{});
    implementBlock(happyPath);
    return result;
}

mlir::Value pylir::CodeGen::makeSet(mlir::Location loc, const std::vector<Py::IterArg>& args)
{
    if (!m_currentExceptBlock)
    {
        return m_builder.create<Py::MakeSetOp>(loc, args);
    }
    if (std::all_of(args.begin(), args.end(),
                    [](const Py::IterArg& arg) { return std::holds_alternative<mlir::Value>(arg); }))
    {
        return m_builder.create<Py::MakeSetOp>(loc, args);
    }
    auto happyPath = BlockPtr{};
    auto result = m_builder.create<Py::MakeSetExOp>(loc, args, happyPath, mlir::ValueRange{}, m_currentExceptBlock,
                                                    mlir::ValueRange{});
    implementBlock(happyPath);
    return result;
}

mlir::Value pylir::CodeGen::makeDict(mlir::Location loc, const std::vector<Py::DictArg>& args)
{
    if (!m_currentExceptBlock)
    {
        return m_builder.create<Py::MakeDictOp>(loc, args);
    }
    if (std::all_of(args.begin(), args.end(),
                    [](const Py::DictArg& arg)
                    { return std::holds_alternative<std::pair<mlir::Value, mlir::Value>>(arg); }))
    {
        return m_builder.create<Py::MakeDictOp>(loc, args);
    }
    auto happyPath = BlockPtr{};
    auto result = m_builder.create<Py::MakeDictExOp>(loc, args, happyPath, mlir::ValueRange{}, m_currentExceptBlock,
                                                     mlir::ValueRange{});
    implementBlock(happyPath);
    return result;
}

mlir::Value pylir::CodeGen::buildSubclassCheck(mlir::Location loc, mlir::Value type, mlir::Value base)
{
    auto mro = m_builder.create<Py::GetAttrOp>(loc, type, "__mro__").result();
    return m_builder.create<Py::LinearContainsOp>(loc, mro, base);
}

void pylir::CodeGen::buildTupleForEach(mlir::Location loc, mlir::Value tuple, mlir::Block* endBlock,
                                       mlir::ValueRange endArgs,
                                       llvm::function_ref<void(mlir::Value)> iterationCallback)
{
    auto tupleSize = m_builder.create<Py::TupleIntegerLenOp>(loc, m_builder.getIndexType(), tuple);
    auto startConstant = m_builder.create<mlir::ConstantOp>(loc, m_builder.getIndexAttr(0));
    auto conditionBlock = BlockPtr{};
    conditionBlock->addArgument(m_builder.getIndexType());
    m_builder.create<mlir::BranchOp>(loc, conditionBlock, mlir::ValueRange{startConstant});

    implementBlock(conditionBlock);
    auto isLess =
        m_builder.create<mlir::CmpIOp>(loc, mlir::CmpIPredicate::ult, conditionBlock->getArgument(0), tupleSize);
    auto body = BlockPtr{};
    m_builder.create<mlir::CondBranchOp>(loc, isLess, body, endBlock, endArgs);

    implementBlock(body);
    auto entry = m_builder.create<Py::TupleIntegerGetItemOp>(loc, tuple, conditionBlock->getArgument(0));
    iterationCallback(entry);
    PYLIR_ASSERT(needsTerminator());
    auto one = m_builder.create<mlir::ConstantOp>(loc, m_builder.getIndexAttr(1));
    auto nextIter = m_builder.create<mlir::AddIOp>(loc, conditionBlock->getArgument(0), one);
    m_builder.create<mlir::BranchOp>(loc, conditionBlock, mlir::ValueRange{nextIter});
}
