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

    auto initFunc = m_currentFunc = mlir::FuncOp::create(m_builder.getUnknownLoc(), formQualifiedName("__init__"),
                                                         mlir::FunctionType::get(m_builder.getContext(), {}, {}));
    m_module.push_back(initFunc);

    m_builder.setInsertionPointToStart(initFunc.addEntryBlock());
    for (auto& iter : fileInput.input)
    {
        if (auto* statement = std::get_if<Syntax::Statement>(&iter))
        {
            visit(*statement);
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
    pylir::match(
        statement.variant, [&](const Syntax::CompoundStmt& compoundStmt) { visit(compoundStmt); },
        [&](const Syntax::Statement::SingleLine& singleLine) { visit(singleLine.stmtList); });
}

void pylir::CodeGen::visit(const Syntax::StmtList& stmtList)
{
    visit(*stmtList.firstExpr);
    for (auto& iter : stmtList.remainingExpr)
    {
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
        [&](const Syntax::ReturnStmt& returnStmt)
        {
            auto loc = getLoc(returnStmt, returnStmt.returnKeyword);
            if (!returnStmt.expressions)
            {
                auto none = m_builder.create<Py::GetGlobalValueOp>(loc, Builtins::None);
                m_builder.create<mlir::ReturnOp>(loc, mlir::ValueRange{none});
                return;
            }
            auto value = visit(*returnStmt.expressions);
            m_builder.create<mlir::ReturnOp>(loc, mlir::ValueRange{value});
        },
        [&](const Syntax::BreakStmt& breakStmt)
        {
            auto loc = getLoc(breakStmt, breakStmt.breakKeyword);
            m_builder.create<mlir::BranchOp>(loc, m_loopStack.back().breakBlock);
        },
        [&](const Syntax::ContinueStmt& continueStmt)
        {
            auto loc = getLoc(continueStmt, continueStmt.continueKeyword);
            m_builder.create<mlir::BranchOp>(loc, m_loopStack.back().continueBlock);
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
            auto indices = visit(subscription.expressionList);

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
    for (auto& [list, token] : assignmentStmt.targets)
    {
        assignTarget(list, rhs);
    }
}

template <class Op>
mlir::Value pylir::CodeGen::visit(const Syntax::StarredList& starredList)
{
    auto loc = getLoc(starredList, starredList);
    std::vector<std::int32_t> starred;
    std::vector<mlir::Value> operands;
    auto handleItem = [&](const Syntax::StarredItem& item, std::size_t index)
    {
        pylir::match(
            item.variant,
            [&](const Syntax::AssignmentExpression& assignment) { operands.push_back(visit(assignment)); },
            [&](const std::pair<BaseToken, Syntax::OrExpr>& pair)
            {
                operands.push_back(visit(pair.second));
                starred.push_back(index);
            });
    };
    handleItem(*starredList.firstExpr, 0);
    for (auto& iter : llvm::enumerate(starredList.remainingExpr))
    {
        handleItem(*iter.value().second, iter.index() + 1);
    }
    return m_builder.create<Op>(loc, operands, m_builder.getI32ArrayAttr(starred));
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
            std::vector<std::int32_t> starred;
            std::vector<mlir::Value> operands;
            auto handleItem = [&](const Syntax::StarredItem& item, std::size_t index)
            {
                pylir::match(
                    item.variant,
                    [&](const Syntax::AssignmentExpression& assignment) { operands.push_back(visit(assignment)); },
                    [&](const std::pair<BaseToken, Syntax::OrExpr>& pair)
                    {
                        operands.push_back(visit(pair.second));
                        starred.push_back(index);
                    });
            };
            for (auto& iter : llvm::enumerate(items.leading))
            {
                handleItem(iter.value().first, iter.index());
            }
            if (items.last)
            {
                handleItem(*items.last, items.leading.size());
            }
            return m_builder.create<Py::MakeTupleOp>(loc, operands, m_builder.getI32ArrayAttr(starred));
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
    std::transform(expressionList.remainingExpr.begin(), expressionList.remainingExpr.end(), operands.begin() + 1,
                   [&](const auto& pair) { return visit(*pair.second); });
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
    auto condition = toI1(visit(*expression.suffix->test));
    auto found = new mlir::Block;
    auto elseBlock = new mlir::Block;
    auto thenBlock = new mlir::Block;
    thenBlock->addArgument(m_builder.getType<Py::DynamicType>());

    m_builder.create<mlir::CondBranchOp>(loc, condition, found, elseBlock);

    m_currentFunc.push_back(found);
    m_builder.setInsertionPointToStart(found);
    m_builder.create<mlir::BranchOp>(loc, thenBlock, visit(expression.value));

    m_currentFunc.push_back(elseBlock);
    m_builder.setInsertionPointToStart(elseBlock);
    m_builder.create<mlir::BranchOp>(loc, thenBlock, visit(*expression.suffix->elseValue));

    m_currentFunc.push_back(thenBlock);
    m_builder.setInsertionPointToStart(thenBlock);
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
            auto found = new mlir::Block;
            found->addArgument(m_builder.getType<Py::DynamicType>());
            auto rhsTry = new mlir::Block;
            m_builder.create<mlir::CondBranchOp>(loc, toI1(lhs), found, lhs, rhsTry, mlir::ValueRange{});

            m_currentFunc.push_back(rhsTry);
            m_builder.setInsertionPointToStart(rhsTry);
            auto rhs = visit(binOp->rhs);
            m_builder.create<mlir::BranchOp>(loc, found, rhs);

            m_currentFunc.push_back(found);
            m_builder.setInsertionPointToStart(found);
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
            auto found = new mlir::Block;
            found->addArgument(m_builder.getType<Py::DynamicType>());
            auto rhsTry = new mlir::Block;
            m_builder.create<mlir::CondBranchOp>(loc, toI1(lhs), rhsTry, mlir::ValueRange{}, found,
                                                 mlir::ValueRange{lhs});

            m_currentFunc.push_back(rhsTry);
            m_builder.setInsertionPointToStart(rhsTry);
            auto rhs = visit(binOp->rhs);
            m_builder.create<mlir::BranchOp>(loc, found, rhs);

            m_currentFunc.push_back(found);
            m_builder.setInsertionPointToStart(found);
            return found->getArgument(0);
        });
}

mlir::Value pylir::CodeGen::visit(const Syntax::NotTest& expression)
{
    return pylir::match(
        expression.variant, [&](const Syntax::Comparison& comparison) { return visit(comparison); },
        [&](const std::pair<BaseToken, std::unique_ptr<Syntax::NotTest>>& pair) -> mlir::Value
        {
            auto value = toBool(visit(*pair.second));
            auto loc = getLoc(expression, pair.first);
            return m_builder.create<Py::InvertOp>(loc, value);
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
    auto previousRHS = first;
    for (auto& [op, rhs] : comparison.rest)
    {
        auto loc = getLoc(op.firstToken, op.firstToken);

        mlir::Block* found;
        if (result)
        {
            found = new mlir::Block;
            found->addArgument(m_builder.getType<Py::DynamicType>());
            auto rhsTry = new mlir::Block;
            m_builder.create<mlir::CondBranchOp>(loc, toI1(result), found, result, rhsTry, mlir::ValueRange{});

            m_currentFunc.push_back(rhsTry);
            m_builder.setInsertionPointToStart(rhsTry);
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
        mlir::Value cmp;
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
        previousRHS = other;
        if (!result)
        {
            result = cmp;
            continue;
        }
        m_builder.create<mlir::BranchOp>(loc, found, cmp);

        m_currentFunc.push_back(found);
        m_builder.setInsertionPointToStart(found);
        result = found->getArgument(0);
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
            return m_builder.create<Py::OrOp>(loc, lhs, rhs);
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
            return m_builder.create<Py::XorOp>(loc, lhs, rhs);
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
            return m_builder.create<Py::AndOp>(loc, lhs, rhs);
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
                case TokenType::ShiftLeft: return m_builder.create<Py::LShiftOp>(loc, lhs, rhs);
                case TokenType::ShiftRight: return m_builder.create<Py::RShiftOp>(loc, lhs, rhs);
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
                case TokenType::Plus: return m_builder.create<Py::AddOp>(loc, lhs, rhs);
                case TokenType::Minus: return m_builder.create<Py::SubOp>(loc, lhs, rhs);
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
                case TokenType::Star: return m_builder.create<Py::MulOp>(loc, lhs, rhs);
                case TokenType::Divide: return m_builder.create<Py::TrueDivOp>(loc, lhs, rhs);
                case TokenType::IntDivide: return m_builder.create<Py::FloorDivOp>(loc, lhs, rhs);
                case TokenType::Remainder: return m_builder.create<Py::ModuloOp>(loc, lhs, rhs);
                default: PYLIR_UNREACHABLE;
            }
        },
        [&](const std::unique_ptr<Syntax::MExpr::AtBin>& atBin) -> mlir::Value
        {
            auto lhs = visit(*atBin->lhs);
            auto rhs = visit(*atBin->rhs);
            auto loc = getLoc(mExpr, atBin->atToken);
            return m_builder.create<Py::MatMulOp>(loc, lhs, rhs);
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
    mlir::Block* classNamespaceFound = nullptr;
    ScopeContainer::value_type* scope;
    if (m_classNamespace)
    {
        classNamespaceFound = new mlir::Block;
        classNamespaceFound->addArgument(m_builder.getType<Py::DynamicType>());
        auto str = m_builder.create<Py::ConstantOp>(loc, m_builder.getStringAttr(identifierToken.getValue()));
        auto tryGet = m_builder.create<Py::DictTryGetItemOp>(loc, m_classNamespace, str);
        mlir::Block* elseBlock = new mlir::Block;
        m_builder.create<mlir::CondBranchOp>(loc, tryGet.found(), classNamespaceFound, tryGet.result(), elseBlock,
                                             mlir::ValueRange{});
        m_currentFunc.push_back(elseBlock);
        m_builder.setInsertionPointToStart(elseBlock);

        // if not found in locals, it does not import free variables but rather goes straight to the global scope.
        // beyond that it could also access the builtins scope which does not yet exist and idk if it has to and will
        // exist
        scope = &m_scope[0];
    }
    else
    {
        scope = &getCurrentScope();
    }
    auto result = scope->find(identifierToken.getValue());
    if (result == scope->end())
    {
        auto exception = buildException(loc, Builtins::NameError, /*TODO: string arg*/ {});
        raiseException(exception);
        if (!m_classNamespace)
        {
            return {};
        }
        m_currentFunc.push_back(classNamespaceFound);
        m_builder.setInsertionPointToStart(classNamespaceFound);
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
            auto success = new mlir::Block;
            auto failure = new mlir::Block;
            m_builder.create<mlir::CondBranchOp>(loc, getAttrOp.success(), success, failure);

            m_currentFunc.push_back(failure);
            m_builder.setInsertionPointToStart(failure);
            auto exception = buildException(loc, Builtins::UnboundLocalError, /*TODO: string arg*/ {});
            raiseException(exception);

            m_currentFunc.push_back(success);
            m_builder.setInsertionPointToStart(success);
            return getAttrOp.result();
        }
    }
    auto condition = m_builder.create<Py::IsUnboundHandleOp>(loc, handle);
    auto unbound = new mlir::Block;
    auto found = new mlir::Block;
    m_builder.create<mlir::CondBranchOp>(loc, condition, unbound, found);

    m_currentFunc.push_back(unbound);
    m_builder.setInsertionPointToStart(unbound);
    if (result->second.kind == Global)
    {
        auto exception = buildException(loc, Builtins::NameError, /*TODO: string arg*/ {});
        raiseException(exception);
    }
    else
    {
        auto exception = buildException(loc, Builtins::UnboundLocalError, /*TODO: string arg*/ {});
        raiseException(exception);
    }

    m_currentFunc.push_back(found);
    m_builder.setInsertionPointToStart(found);
    if (!classNamespaceFound)
    {
        return m_builder.create<Py::LoadOp>(loc, handle);
    }
    m_builder.create<mlir::BranchOp>(loc, classNamespaceFound,
                                     mlir::ValueRange{m_builder.create<Py::LoadOp>(loc, handle)});

    m_currentFunc.push_back(classNamespaceFound);
    m_builder.setInsertionPointToStart(classNamespaceFound);
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
                case TokenType::NoneKeyword: return m_builder.create<Py::GetGlobalValueOp>(location, Builtins::None);
                default: PYLIR_UNREACHABLE;
            }
        },
        [&](const IdentifierToken& identifierToken) -> mlir::Value { return readIdentifier(identifierToken); },
        [&](const std::unique_ptr<Syntax::Enclosure>& enclosure) -> mlir::Value { return visit(*enclosure); });
}

mlir::Value pylir::CodeGen::visit(const pylir::Syntax::Subscription& subscription)
{
    auto container = visit(*subscription.primary);
    auto indices = visit(subscription.expressionList);

    auto loc = getLoc(subscription, subscription);
    auto type = m_builder.create<Py::TypeOfOp>(loc, container);
    return buildSpecialMethodCall(loc, "__getitem__", type,
                                  m_builder.create<Py::MakeTupleOp>(loc, std::vector<Py::IterArg>{container, indices}),
                                  m_builder.create<Py::MakeDictOp>(loc));
}

mlir::Value pylir::CodeGen::toI1(mlir::Value value)
{
    auto boolean = toBool(value);
    return m_builder.create<Py::BoolToI1Op>(boolean.getLoc(), boolean);
}

mlir::Value pylir::CodeGen::toBool(mlir::Value value)
{
    return m_builder.create<Py::BoolOp>(value.getLoc(), value);
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
            return m_builder.create<Py::MakeTupleOp>(loc, mlir::ValueRange{}, m_builder.getI32ArrayAttr({}));
        },
        [&](const Syntax::Enclosure::ListDisplay& listDisplay) -> mlir::Value
        {
            auto loc = getLoc(listDisplay, listDisplay.openSquare);
            return pylir::match(
                listDisplay.variant,
                [&](std::monostate) -> mlir::Value
                { return m_builder.create<Py::MakeListOp>(loc, mlir::ValueRange{}, m_builder.getI32ArrayAttr({})); },
                [&](const Syntax::StarredList& list) -> mlir::Value { return visit<Py::MakeListOp>(list); },
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
                [&](const Syntax::StarredList& list) -> mlir::Value { return visit<Py::MakeSetOp>(list); },
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
                        pylir::match(
                            keyDatum.variant,
                            [&](const Syntax::Enclosure::DictDisplay::KeyDatum::Key& key)
                            {
                                auto first = visit(key.first);
                                auto second = visit(key.second);
                                result.push_back(std::pair{first, second});
                            },
                            [&](const Syntax::Enclosure::DictDisplay::KeyDatum::Datum& key)
                            { result.push_back(Py::MappingExpansion{visit(key.orExpr)}); });
                    };
                    handleOne(*list.firstExpr);
                    for (auto& [token, iter] : list.remainingExpr)
                    {
                        (void)token;
                        handleOne(*iter);
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
    writeIdentifier(assignmentExpression.identifierAndWalrus->first, value);
    return value;
}

void pylir::CodeGen::visit(const Syntax::IfStmt& ifStmt)
{
    auto condition = visit(ifStmt.condition);
    auto trueBlock = new mlir::Block;
    auto thenBlock = new mlir::Block;
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
    m_builder.create<mlir::CondBranchOp>(loc, toI1(condition), trueBlock, elseBlock);

    m_currentFunc.push_back(trueBlock);
    m_builder.setInsertionPointToStart(trueBlock);
    visit(*ifStmt.suite);
    if (needsTerminator())
    {
        m_builder.create<mlir::BranchOp>(loc, thenBlock);
    }
    if (thenBlock == elseBlock)
    {
        m_currentFunc.push_back(thenBlock);
        m_builder.setInsertionPointToStart(thenBlock);
        return;
    }
    m_currentFunc.push_back(elseBlock);
    m_builder.setInsertionPointToStart(elseBlock);
    for (auto& iter : llvm::enumerate(ifStmt.elifs))
    {
        loc = getLoc(iter.value().elif, iter.value().elif);
        condition = visit(iter.value().condition);
        trueBlock = new mlir::Block;
        if (iter.index() == ifStmt.elifs.size() - 1 && !ifStmt.elseSection)
        {
            elseBlock = thenBlock;
        }
        else
        {
            elseBlock = new mlir::Block;
        }

        m_builder.create<mlir::CondBranchOp>(loc, toI1(condition), trueBlock, elseBlock);

        m_currentFunc.push_back(trueBlock);
        m_builder.setInsertionPointToStart(trueBlock);
        visit(*iter.value().suite);
        if (needsTerminator())
        {
            m_builder.create<mlir::BranchOp>(loc, thenBlock);
        }
        if (thenBlock != elseBlock)
        {
            m_currentFunc.push_back(elseBlock);
            m_builder.setInsertionPointToStart(elseBlock);
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

    m_currentFunc.push_back(thenBlock);
    m_builder.setInsertionPointToStart(thenBlock);
}

void pylir::CodeGen::visit(const Syntax::WhileStmt& whileStmt)
{
    auto loc = getLoc(whileStmt, whileStmt.whileKeyword);
    auto conditionBlock = new mlir::Block;
    auto thenBlock = new mlir::Block;
    m_builder.create<mlir::BranchOp>(loc, conditionBlock);
    m_currentFunc.push_back(conditionBlock);

    m_builder.setInsertionPointToStart(conditionBlock);
    auto condition = visit(whileStmt.condition);
    mlir::Block* elseBlock;
    if (whileStmt.elseSection)
    {
        elseBlock = new mlir::Block;
    }
    else
    {
        elseBlock = thenBlock;
    }
    mlir::Block* body = new mlir::Block;
    m_builder.create<mlir::CondBranchOp>(loc, toI1(condition), body, elseBlock);

    m_currentFunc.push_back(body);
    m_builder.setInsertionPointToStart(body);
    m_loopStack.push_back({thenBlock, conditionBlock});
    std::optional exit = llvm::make_scope_exit([&] { m_loopStack.pop_back(); });
    visit(*whileStmt.suite);
    if (needsTerminator())
    {
        m_builder.create<mlir::BranchOp>(loc, conditionBlock);
    }
    exit.reset();

    m_currentFunc.push_back(elseBlock);
    m_builder.setInsertionPointToStart(elseBlock);
    if (elseBlock == thenBlock)
    {
        return;
    }
    visit(*whileStmt.elseSection->suite);
    if (needsTerminator())
    {
        m_builder.create<mlir::BranchOp>(loc, thenBlock);
    }
    m_currentFunc.push_back(thenBlock);
    m_builder.setInsertionPointToStart(thenBlock);
}

void pylir::CodeGen::visit(const pylir::Syntax::ForStmt& forStmt) {}

void pylir::CodeGen::visit(const pylir::Syntax::TryStmt& tryStmt) {}

void pylir::CodeGen::visit(const pylir::Syntax::WithStmt& withStmt) {}

void pylir::CodeGen::visit(const pylir::Syntax::FuncDef& funcDef)
{
    // TODO: adapter for calling convention
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
                functionParameters.push_back({std::string(param.identifier.getValue()), kind, false});
                functionParametersTokens.push_back(param.identifier);
            }

            void visit(const Syntax::ParameterList::DefParameter& defParameter)
            {
                Visitor::visit(defParameter);
                if (!defParameter.defaultArg)
                {
                    return;
                }
                functionParameters.back().hasDefaultParam = true;
                auto value = calcCallback(defParameter.defaultArg->second);
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
                kind = FunctionParameter::PosOnly;
                Visitor::visit(posOnlyNode);
            }

            void visit(const Syntax::ParameterList::NoPosOnly& noPosOnly)
            {
                kind = FunctionParameter::Normal;
                Visitor::visit(noPosOnly);
            }

            void visit(const Syntax::ParameterList::StarArgs& star)
            {
                auto doubleStarHandler = [&](const Syntax::ParameterList::StarArgs::DoubleStar& doubleStar)
                {
                    visit(doubleStar.parameter);
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
                            functionParameters.back().kind = FunctionParameter::PosRest;
                        }
                        for (auto& iter : llvm::make_second_range(star.defParameters))
                        {
                            visit(iter);
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
    }
    auto loc = getLoc(funcDef.funcName, funcDef.funcName);
    auto qualifiedName = formQualifiedName(std::string(funcDef.funcName.getValue()));
    std::vector<IdentifierToken> usedClosures;
    mlir::FuncOp func;
    {
        mlir::OpBuilder::InsertionGuard guard{m_builder};
        pylir::ValueReset reset(m_classNamespace, m_classNamespace);
        m_classNamespace = {};
        func = mlir::FuncOp::create(
            loc, formImplName(qualifiedName + "$impl"),
            m_builder.getFunctionType(
                std::vector<mlir::Type>(1 + functionParameters.size(), m_builder.getType<Py::DynamicType>()),
                {m_builder.getType<Py::DynamicType>()}));
        func.sym_visibilityAttr(m_builder.getStringAttr(("private")));
        m_module.push_back(func);
        pylir::ValueReset resetFunc(m_currentFunc, m_currentFunc);
        m_currentFunc = func;
        auto entry = func.addEntryBlock();
        m_builder.setInsertionPointToStart(entry);

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
                auto closureType = m_builder.create<Py::GetGlobalValueOp>(loc, Builtins::Cell);
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
            auto closureType = m_builder.create<Py::GetGlobalValueOp>(loc, Builtins::Cell);
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
                loc, mlir::ValueRange{m_builder.create<Py::GetGlobalValueOp>(loc, Builtins::None)});
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
            defaults = m_builder.create<Py::GetGlobalValueOp>(loc, Builtins::None);
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
            kwDefaults = m_builder.create<Py::GetGlobalValueOp>(loc, Builtins::None);
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
            closure = m_builder.create<Py::GetGlobalValueOp>(loc, Builtins::None);
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
        mlir::OpBuilder::InsertionGuard guard{m_builder};
        func = mlir::FuncOp::create(
            loc, formImplName(qualifiedName + "$impl"),
            m_builder.getFunctionType(
                std::vector<mlir::Type>(2 /* cell tuple + namespace dict */, m_builder.getType<Py::DynamicType>()),
                {m_builder.getType<Py::DynamicType>()}));
        func.sym_visibilityAttr(m_builder.getStringAttr(("private")));
        m_module.push_back(func);
        pylir::ValueReset resetFunc(m_currentFunc, m_currentFunc);
        m_currentFunc = func;
        auto entry = func.addEntryBlock();
        m_builder.setInsertionPointToStart(entry);

        m_scope.emplace_back();
        m_qualifierStack.emplace_back(classDef.className.getValue());
        auto exit = llvm::make_scope_exit(
            [&]
            {
                m_scope.pop_back();
                m_qualifierStack.pop_back();
            });
        pylir::ValueReset reset(m_classNamespace, m_classNamespace);
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
            pylir::match(
                positionalItem.variant,
                [&](const std::unique_ptr<Syntax::AssignmentExpression>& expression)
                { iterArgs.emplace_back(visit(*expression)); },
                [&](const Syntax::ArgumentList::PositionalItem::Star& star)
                { iterArgs.push_back(Py::IterExpansion{visit(*star.expression)}); });
        };
        handlePositionalItem(argumentList.positionalArguments->firstItem);
        for (auto& [token, rest] : argumentList.positionalArguments->rest)
        {
            (void)token;
            handlePositionalItem(rest);
        }
    }
    auto handleKeywordItem = [&](const Syntax::ArgumentList::KeywordItem& keywordItem)
    {
        auto key = m_builder.create<Py::ConstantOp>(getLoc(keywordItem.identifier, keywordItem.identifier),
                                                    m_builder.getStringAttr(keywordItem.identifier.getValue()));
        auto value = visit(*keywordItem.expression);
        dictArgs.push_back(std::pair{key, value});
    };
    if (argumentList.starredAndKeywords)
    {
        auto handleExpression = [&](const Syntax::ArgumentList::StarredAndKeywords::Expression& expression)
        { iterArgs.push_back(Py::IterExpansion{visit(*expression.expression)}); };
        auto handleStarredAndKeywords = [&](const Syntax::ArgumentList::StarredAndKeywords::Variant& variant)
        { pylir::match(variant, handleKeywordItem, handleExpression); };
        handleKeywordItem(argumentList.starredAndKeywords->first);
        for (auto& [token, variant] : argumentList.starredAndKeywords->rest)
        {
            (void)token;
            handleStarredAndKeywords(variant);
        }
    }
    if (argumentList.keywordArguments)
    {
        auto handleExpression = [&](const Syntax::ArgumentList::KeywordArguments::Expression& expression)
        { dictArgs.push_back(Py::MappingExpansion{visit(*expression.expression)}); };
        auto handleKeywordArguments = [&](const Syntax::ArgumentList::KeywordArguments::Variant& variant)
        { pylir::match(variant, handleKeywordItem, handleExpression); };
        handleExpression(argumentList.keywordArguments->first);
        for (auto& [token, variant] : argumentList.keywordArguments->rest)
        {
            (void)token;
            handleKeywordArguments(variant);
        }
    }
    return {m_builder.create<Py::MakeTupleOp>(loc, iterArgs), m_builder.create<Py::MakeDictOp>(loc, dictArgs)};
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
    auto tuple = m_builder.create<Py::MakeTupleOp>(loc, args);
    auto dict = m_builder.create<Py::ConstantOp>(loc, Py::DictAttr::get(m_builder.getContext(), {}));
    auto newMethod = m_builder.create<Py::GetAttrOp>(loc, typeObj, "__new__").result();

    auto obj = m_builder
                   .create<mlir::CallIndirectOp>(loc, m_builder.create<Py::FunctionGetFunctionOp>(loc, newMethod),
                                                 mlir::ValueRange{newMethod, tuple, dict})
                   ->getResult(0);
    auto context = m_builder.create<Py::GetGlobalValueOp>(loc, Builtins::None);
    m_builder.create<Py::SetAttrOp>(loc, context, obj, "__context__");
    auto cause = m_builder.create<Py::GetGlobalValueOp>(loc, Builtins::None);
    m_builder.create<Py::SetAttrOp>(loc, cause, obj, "__cause__");
    return obj;
}

void pylir::CodeGen::raiseException(mlir::Value exceptionObject)
{
    // TODO: branch to except handlers
    m_builder.create<Py::RaiseOp>(exceptionObject.getLoc(), exceptionObject);
    m_builder.clearInsertionPoint();
}

std::pair<mlir::Value, mlir::Value> pylir::CodeGen::buildMROLookup(mlir::Location loc, mlir::Value type,
                                                                   llvm::Twine attribute)
{
    auto mro = m_builder.create<Py::GetAttrOp>(loc, type, "__mro__").result();
    auto len = m_builder.create<Py::TupleIntegerLenOp>(loc, m_builder.getIndexType(), mro);
    auto zero = m_builder.create<mlir::ConstantOp>(loc, m_builder.getIndexAttr(0));

    auto condition = new mlir::Block;
    condition->addArgument(m_builder.getIndexType());
    m_builder.create<mlir::BranchOp>(loc, condition, mlir::ValueRange{zero});

    m_currentFunc.push_back(condition);
    m_builder.setInsertionPointToStart(condition);
    auto cmp = m_builder.create<mlir::CmpIOp>(loc, mlir::CmpIPredicate::ult, condition->getArgument(0), len);
    auto found = new mlir::Block;
    found->addArgument(m_builder.getType<Py::DynamicType>());
    auto body = new mlir::Block;
    auto unbound = m_builder.create<Py::UnboundValueOp>(loc);
    m_builder.create<mlir::CondBranchOp>(loc, cmp, body, found, mlir::ValueRange{unbound});

    m_currentFunc.push_back(body);
    m_builder.setInsertionPointToStart(body);
    auto entry = m_builder.create<Py::TupleIntegerGetItemOp>(loc, mro, condition->getArgument(0));
    auto fetch = m_builder.create<Py::GetAttrOp>(loc, entry, m_builder.getStringAttr(attribute));
    auto one = m_builder.create<mlir::ConstantOp>(loc, m_builder.getIndexAttr(1));
    auto incremented = m_builder.create<mlir::AddIOp>(loc, condition->getArgument(0), one);
    m_builder.create<mlir::CondBranchOp>(loc, fetch.success(), found, mlir::ValueRange{fetch.result()}, condition,
                                         mlir::ValueRange{incremented});

    m_currentFunc.push_back(found);
    m_builder.setInsertionPointToStart(found);
    auto success = m_builder.create<Py::IsUnboundValueOp>(loc, found->getArgument(0));
    return {found->getArgument(0), success};
}

mlir::Value pylir::CodeGen::buildCall(mlir::Location loc, mlir::Value callable, mlir::Value tuple, mlir::Value dict)
{
    auto condition = new mlir::Block;
    condition->addArgument(m_builder.getType<Py::DynamicType>());
    auto func = m_builder.create<Py::GetGlobalValueOp>(loc, Builtins::Function);
    m_builder.create<mlir::BranchOp>(loc, condition, mlir::ValueRange{callable});

    m_currentFunc.push_back(condition);
    m_builder.setInsertionPointToStart(condition);
    auto type = m_builder.create<Py::TypeOfOp>(loc, condition->getArgument(0));
    auto isFunction = m_builder.create<Py::IsOp>(loc, type, func);

    auto found = new mlir::Block;
    auto body = new mlir::Block;
    found->addArgument(m_builder.getType<Py::DynamicType>());
    m_builder.create<mlir::CondBranchOp>(loc, isFunction, found, mlir::ValueRange{condition->getArgument(0)}, body,
                                         mlir::ValueRange{});

    {
        m_currentFunc.push_back(body);
        m_builder.setInsertionPointToStart(body);
        auto [call, success] = buildMROLookup(loc, type, "__call__");
        auto unboundValue = m_builder.create<Py::UnboundValueOp>(loc);
        m_builder.create<mlir::CondBranchOp>(loc, success, condition, mlir::ValueRange{call}, found,
                                             mlir::ValueRange{unboundValue});
    }

    m_currentFunc.push_back(found);
    m_builder.setInsertionPointToStart(found);
    auto isUnbound = m_builder.create<Py::IsUnboundValueOp>(loc, found->getArgument(0));
    auto notBound = new mlir::Block;
    auto typeCall = new mlir::Block;
    m_builder.create<mlir::CondBranchOp>(loc, isUnbound, notBound, typeCall);

    m_currentFunc.push_back(notBound);
    m_builder.setInsertionPointToStart(notBound);
    auto typeError = buildException(loc, Builtins::TypeError, {});
    raiseException(typeError);

    m_currentFunc.push_back(typeCall);
    m_builder.setInsertionPointToStart(typeCall);
    auto function = m_builder.create<Py::FunctionGetFunctionOp>(loc, found->getArgument(0));
    return m_builder.create<mlir::CallIndirectOp>(loc, function, mlir::ValueRange{found->getArgument(0), tuple, dict})
        .getResult(0);
}

mlir::Value pylir::CodeGen::buildSpecialMethodCall(mlir::Location loc, llvm::Twine methodName, mlir::Value type,
                                                   mlir::Value tuple, mlir::Value dict)
{
    auto [method, found] = buildMROLookup(loc, type, methodName);
    auto notFound = new mlir::Block;
    auto exec = new mlir::Block;
    m_builder.create<mlir::CondBranchOp>(loc, found, exec, notFound);

    m_currentFunc.push_back(notFound);
    m_builder.setInsertionPointToStart(notFound);
    auto exception = buildException(loc, Builtins::TypeError, {});
    raiseException(exception);

    m_currentFunc.push_back(exec);
    m_builder.setInsertionPointToStart(exec);
    return buildCall(loc, method, tuple, dict);
}

mlir::FuncOp pylir::CodeGen::buildFunctionCC(mlir::Location loc, llvm::Twine name, mlir::FuncOp implementation,
                                             const std::vector<FunctionParameter>& parameters)
{
    mlir::OpBuilder::InsertionGuard guard(m_builder);
    m_builder.setInsertionPointToEnd(m_module.getBody());
    auto cc = m_builder.create<mlir::FuncOp>(
        loc, name.str(),
        m_builder.getFunctionType({m_builder.getType<Py::DynamicType>(), m_builder.getType<Py::DynamicType>(),
                                   m_builder.getType<Py::DynamicType>()},
                                  {m_builder.getType<Py::DynamicType>()}),
        m_builder.getStringAttr("private"));
    pylir::ValueReset reset(m_currentFunc, m_currentFunc);
    m_currentFunc = cc;
    m_builder.setInsertionPointToStart(cc.addEntryBlock());
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
                auto lessBlock = new mlir::Block;
                auto unboundBlock = new mlir::Block;
                m_builder.create<mlir::CondBranchOp>(loc, isLess, lessBlock, unboundBlock);

                auto resultBlock = new mlir::Block;
                resultBlock->addArgument(m_builder.getType<Py::DynamicType>());
                m_currentFunc.push_back(unboundBlock);
                m_builder.setInsertionPointToStart(unboundBlock);
                auto unboundValue = m_builder.create<Py::UnboundValueOp>(loc);
                m_builder.create<mlir::BranchOp>(loc, resultBlock, mlir::ValueRange{unboundValue});

                m_currentFunc.push_back(lessBlock);
                m_builder.setInsertionPointToStart(lessBlock);
                auto fetched = m_builder.create<Py::TupleIntegerGetItemOp>(loc, tuple, constant);
                m_builder.create<mlir::BranchOp>(loc, resultBlock, mlir::ValueRange{fetched});

                m_currentFunc.push_back(resultBlock);
                m_builder.setInsertionPointToStart(resultBlock);
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
                auto foundBlock = new mlir::Block;
                auto notFoundBlock = new mlir::Block;
                m_builder.create<mlir::CondBranchOp>(loc, lookup.found(), foundBlock, notFoundBlock);

                auto resultBlock = new mlir::Block;
                resultBlock->addArgument(m_builder.getType<Py::DynamicType>());
                m_currentFunc.push_back(notFoundBlock);
                m_builder.setInsertionPointToStart(notFoundBlock);
                auto unboundValue = m_builder.create<Py::UnboundValueOp>(loc);
                m_builder.create<mlir::BranchOp>(loc, resultBlock, mlir::ValueRange{unboundValue});

                m_currentFunc.push_back(foundBlock);
                m_builder.setInsertionPointToStart(foundBlock);
                m_builder.create<Py::DictDelItemOp>(loc, dict, constant);
                // value can't be assigned both through a positional argument as well as keyword argument
                if (argValue)
                {
                    auto isUnbound = m_builder.create<Py::IsUnboundValueOp>(loc, argValue);
                    auto boundBlock = new mlir::Block;
                    m_builder.create<mlir::CondBranchOp>(loc, isUnbound, resultBlock, mlir::ValueRange{lookup.result()},
                                                         boundBlock, mlir::ValueRange{});

                    m_currentFunc.push_back(boundBlock);
                    m_builder.setInsertionPointToStart(boundBlock);
                    auto exception = buildException(loc, Builtins::TypeError, {});
                    raiseException(exception);
                }
                else
                {
                    m_builder.create<mlir::BranchOp>(loc, resultBlock, mlir::ValueRange{lookup.result()});
                }

                m_currentFunc.push_back(resultBlock);
                m_builder.setInsertionPointToStart(resultBlock);
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
                auto conditionBlock = new mlir::Block;
                conditionBlock->addArgument(m_builder.getIndexType());
                m_builder.create<mlir::BranchOp>(loc, conditionBlock, mlir::ValueRange{start});

                m_currentFunc.push_back(conditionBlock);
                m_builder.setInsertionPointToStart(conditionBlock);
                auto isLess = m_builder.create<mlir::CmpIOp>(loc, mlir::CmpIPredicate::ult,
                                                             conditionBlock->getArgument(0), tupleLen);
                auto lessBlock = new mlir::Block;
                auto endBlock = new mlir::Block;
                m_builder.create<mlir::CondBranchOp>(loc, isLess, lessBlock, endBlock);

                m_currentFunc.push_back(lessBlock);
                m_builder.setInsertionPointToStart(lessBlock);
                auto fetched = m_builder.create<Py::TupleIntegerGetItemOp>(loc, tuple, conditionBlock->getArgument(0));
                m_builder.create<Py::ListAppendOp>(loc, list, fetched);
                auto one = m_builder.create<mlir::ConstantOp>(loc, m_builder.getIndexAttr(1));
                auto incremented = m_builder.create<mlir::AddIOp>(loc, conditionBlock->getArgument(0), one);
                m_builder.create<mlir::BranchOp>(loc, conditionBlock, mlir::ValueRange{incremented});

                m_currentFunc.push_back(endBlock);
                m_builder.setInsertionPointToStart(endBlock);
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
                auto unboundBlock = new mlir::Block;
                auto boundBlock = new mlir::Block;
                boundBlock->addArgument(m_builder.getType<Py::DynamicType>());
                m_builder.create<mlir::CondBranchOp>(loc, isUnbound, unboundBlock, boundBlock,
                                                     mlir::ValueRange{argValue});

                m_currentFunc.push_back(unboundBlock);
                m_builder.setInsertionPointToStart(unboundBlock);
                if (!iter.hasDefaultParam)
                {
                    auto exception = buildException(loc, Builtins::TypeError, {});
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

                m_currentFunc.push_back(boundBlock);
                m_builder.setInsertionPointToStart(boundBlock);
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
