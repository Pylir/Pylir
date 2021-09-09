#include "CodeGen.hpp"

#include <mlir/Dialect/StandardOps/IR/Ops.h>

#include <pylir/Optimizer/PylirPy/IR/PylirPyAttributes.hpp>
#include <pylir/Optimizer/PylirPy/IR/PylirPyDialect.hpp>
#include <pylir/Optimizer/PylirPy/IR/PylirPyOps.hpp>

#include "GlobalDiscoverer.hpp"

pylir::CodeGen::CodeGen(mlir::MLIRContext* context, Diag::Document& document)
    : m_builder(
        [&]
        {
            context->loadDialect<pylir::Py::PylirPyDialect>();
            context->loadDialect<mlir::StandardOpsDialect>();
            return context;
        }()),
      m_document(&document)
{
}

mlir::ModuleOp pylir::CodeGen::visit(const pylir::Syntax::FileInput& fileInput)
{
    m_module = mlir::ModuleOp::create(m_builder.getUnknownLoc());

    m_builder.setInsertionPointToEnd(m_module.getBody());
    GlobalDiscoverer discoverer(
        [&](const IdentifierToken& token)
        {
            if (getCurrentScope().count(token.getValue()))
            {
                return;
            }
            auto op = m_builder.create<Py::GlobalOp>(getLoc(token, token), token.getValue(), mlir::StringAttr{});
            getCurrentScope().insert({token.getValue(), op});
        });
    discoverer.visit(fileInput);

    auto initFunc = m_currentFunc = mlir::FuncOp::create(m_builder.getUnknownLoc(), "__init__",
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
    if (m_builder.getBlock() && !m_builder.getBlock()->back().hasTrait<mlir::OpTrait::IsTerminator>())
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
    // TODO
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
        [&](const Syntax::StarredExpression& expression) { visit(expression); },
        [&](const Syntax::AssignmentStmt& statement) { visit(statement); });
}

void pylir::CodeGen::assignTarget(const Syntax::Target& target, mlir::Value value)
{
    pylir::match(
        target.variant,
        [&](const IdentifierToken& identifierToken)
        {
            auto location = getLoc(identifierToken, identifierToken);
            auto handle = genIdentifierLookup(identifierToken);
            m_builder.create<Py::StoreOp>(location, value, handle);
        },
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

    m_currentFunc.getCallableRegion()->push_back(found);
    m_builder.setInsertionPointToStart(found);
    m_builder.create<mlir::BranchOp>(loc, thenBlock, visit(expression.value));

    m_currentFunc.getCallableRegion()->push_back(elseBlock);
    m_builder.setInsertionPointToStart(elseBlock);
    m_builder.create<mlir::BranchOp>(loc, thenBlock, visit(*expression.suffix->elseValue));

    m_currentFunc.getCallableRegion()->push_back(thenBlock);
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

            m_currentFunc.getCallableRegion()->push_back(rhsTry);
            m_builder.setInsertionPointToStart(rhsTry);
            auto rhs = visit(binOp->rhs);
            m_builder.create<mlir::BranchOp>(loc, found, rhs);

            m_currentFunc.getCallableRegion()->push_back(found);
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

            m_currentFunc.getCallableRegion()->push_back(rhsTry);
            m_builder.setInsertionPointToStart(rhsTry);
            auto rhs = visit(binOp->rhs);
            m_builder.create<mlir::BranchOp>(loc, found, rhs);

            m_currentFunc.getCallableRegion()->push_back(found);
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
            auto value = visit(*pair.second);
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
        auto loc = getLoc(op.firstToken, op.firstToken);
        switch (comp)
        {
            case Comp::Lt: cmp = m_builder.create<Py::LessOp>(loc, previousRHS, other); break;
            case Comp::Gt: cmp = m_builder.create<Py::GreaterOp>(loc, previousRHS, other); break;
            case Comp::Eq: cmp = m_builder.create<Py::EqualOp>(loc, previousRHS, other); break;
            case Comp::Ne: cmp = m_builder.create<Py::NotEqualOp>(loc, previousRHS, other); break;
            case Comp::Ge: cmp = m_builder.create<Py::GreaterEqualOp>(loc, previousRHS, other); break;
            case Comp::Le: cmp = m_builder.create<Py::LessEqualOp>(loc, previousRHS, other); break;
            case Comp::Is: cmp = m_builder.create<Py::IsOp>(loc, previousRHS, other); break;
            case Comp::In: cmp = m_builder.create<Py::InOp>(loc, previousRHS, other); break;
        }
        if (invert)
        {
        }
        previousRHS = other;
        if (!result)
        {
            result = cmp;
            continue;
        }

        // TODO short circuit and return value proper
        result = m_builder.create<mlir::AndOp>(getLoc(op, op.firstToken), result, cmp);
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
        [&](const auto&) -> mlir::Value
        {
            // TODO
            PYLIR_UNREACHABLE;
        });
}

mlir::Value pylir::CodeGen::genIdentifierLookup(const IdentifierToken& identifierToken)
{
    auto loc = getLoc(identifierToken, identifierToken);
    auto result = getCurrentScope().find(identifierToken.getValue());
    if (result == getCurrentScope().end())
    {
        // TODO
        PYLIR_UNREACHABLE;
    }
    mlir::Value handle;
    if (auto alloca = llvm::dyn_cast<Py::AllocaOp>(result->second))
    {
        handle = alloca;
    }
    else
    {
        auto global = llvm::cast<Py::GlobalOp>(result->second);
        handle = m_builder.create<Py::GetGlobalOp>(loc, m_builder.getSymbolRefAttr(global));
    }
    return handle;
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
                    // TODO:
                    PYLIR_UNREACHABLE;
                default: PYLIR_UNREACHABLE;
            }
        },
        [&](const IdentifierToken& identifierToken) -> mlir::Value
        {
            auto loc = getLoc(identifierToken, identifierToken);
            auto handle = genIdentifierLookup(identifierToken);
            return m_builder.create<Py::LoadOp>(loc, handle);
        },
        [&](const std::unique_ptr<Syntax::Enclosure>& enclosure) -> mlir::Value { return visit(*enclosure); });
}

mlir::Value pylir::CodeGen::visit(const pylir::Syntax::Subscription& subscription)
{
    auto container = visit(*subscription.primary);
    auto indices = visit(subscription.expressionList);

    auto loc = getLoc(subscription, subscription);
    return m_builder.create<Py::GetItemOp>(loc, container, indices);
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
    auto handle = genIdentifierLookup(assignmentExpression.identifierAndWalrus->first);
    m_builder.create<Py::StoreOp>(getLoc(assignmentExpression, assignmentExpression), value, handle);
    return value;
}
