#include "CodeGen.hpp"

#include <mlir/Dialect/SCF/SCF.h>
#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/IR/Block.h>

#include <pylir/Dialect/PylirAttributes.hpp>
#include <pylir/Dialect/PylirDialect.hpp>
#include <pylir/Dialect/PylirOps.hpp>

pylir::CodeGen::CodeGen(mlir::MLIRContext* context, Diag::Document& document)
    : m_builder(
        [&]
        {
            context->loadDialect<pylir::Dialect::PylirDialect>();
            context->loadDialect<mlir::scf::SCFDialect>();
            context->loadDialect<mlir::StandardOpsDialect>();
            return context;
        }()),
      m_document(&document)
{
}

mlir::ModuleOp pylir::CodeGen::visit(const pylir::Syntax::FileInput& fileInput)
{
    m_module = mlir::ModuleOp::create(m_builder.getUnknownLoc());
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
    if (m_builder.getBlock() && m_builder.getBlock()->back().isKnownNonTerminator())
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
    PYLIR_UNREACHABLE;
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
            auto result = getCurrentScope().find(identifierToken.getValue());
            if (result != getCurrentScope().end())
            {
                mlir::Value handle;
                if (auto alloca = llvm::dyn_cast<mlir::AllocaOp>(result->second))
                {
                    handle = alloca;
                }
                else
                {
                    auto global = llvm::cast<Dialect::GlobalOp>(result->second);
                    handle = m_builder.create<Dialect::HandleOfOp>(
                        location, mlir::FlatSymbolRefAttr::get(global.sym_name(), global.getContext()));
                }
                m_builder.create<Dialect::StoreOp>(location, value, handle);
                return;
            }
            if (m_scope.size() == 1)
            {
                // We are in the global scope
                auto op = Dialect::GlobalOp::create(location, identifierToken.getValue());
                getCurrentScope().insert({identifierToken.getValue(), op});
                m_module.push_back(op);
                auto handle = m_builder.create<Dialect::HandleOfOp>(
                    location, mlir::FlatSymbolRefAttr::get(op.sym_name(), op.getContext()));
                m_builder.create<Dialect::StoreOp>(location, value, handle);
                return;
            }

            auto alloca = m_builder.create<Dialect::AllocaOp>(location);
            m_builder.create<Dialect::StoreOp>(location, value, alloca);
            getCurrentScope().insert({identifierToken.getValue(), alloca});
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

mlir::Value pylir::CodeGen::visit(const Syntax::StarredExpression& starredExpression)
{
    return pylir::match(
        starredExpression.variant,
        [](const Syntax::StarredExpression::Items&) -> mlir::Value
        {
            // TODO
            PYLIR_UNREACHABLE;
        },
        [&](const Syntax::Expression& expression) { return visit(expression); });
}

mlir::Value pylir::CodeGen::visit(const Syntax::YieldExpression& yieldExpression) {}

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
    auto condition =
        m_builder.create<Dialect::BtoI1Op>(m_builder.getUnknownLoc(), toBool(visit(*expression.suffix->test)));
    mlir::Value thenValue;
    mlir::Value elseValue;

    auto tmpIfOp = m_builder.create<mlir::scf::IfOp>(
        getLoc(expression, expression.suffix->ifToken), condition,
        [&](mlir::OpBuilder&, mlir::Location) { thenValue = visit(expression.value); },
        [&](mlir::OpBuilder&, mlir::Location) { elseValue = visit(*expression.suffix->elseValue); });
    mlir::Type resultType;
    bool variant = true;
    if (thenValue.getType() == elseValue.getType())
    {
        resultType = thenValue.getType();
        variant = false;
    }
    else
    {
        resultType = Dialect::VariantType::get({thenValue.getType(), elseValue.getType()});
    }
    auto ifOp = m_builder.create<mlir::scf::IfOp>(tmpIfOp->mlir::Operation::getLoc(), resultType, condition, true);
    ifOp.thenRegion().takeBody(tmpIfOp.thenRegion());
    ifOp.elseRegion().takeBody(tmpIfOp.elseRegion());
    tmpIfOp->mlir::Operation::erase();
    auto guard = mlir::OpBuilder::InsertionGuard{m_builder};
    m_builder.setInsertionPointToEnd(&ifOp.thenRegion().back());
    if (variant)
    {
        thenValue = m_builder.create<Dialect::ToVariantOp>(m_builder.getUnknownLoc(), resultType, thenValue);
    }
    m_builder.create<mlir::scf::YieldOp>(m_builder.getUnknownLoc(), thenValue);
    m_builder.setInsertionPointToEnd(&ifOp.elseRegion().back());
    if (variant)
    {
        elseValue = m_builder.create<Dialect::ToVariantOp>(m_builder.getUnknownLoc(), resultType, elseValue);
    }
    m_builder.create<mlir::scf::YieldOp>(m_builder.getUnknownLoc(), elseValue);
    return ifOp.results()[0];
}

mlir::Value pylir::CodeGen::visit(const pylir::Syntax::OrTest& expression)
{
    return pylir::match(
        expression.variant, [&](const Syntax::AndTest& andTest) { return visit(andTest); },
        [&](const std::unique_ptr<Syntax::OrTest::BinOp>& binOp) -> mlir::Value
        {
            auto lhs = toBool(visit(*binOp->lhs));
            auto rhs = toBool(visit(binOp->rhs));
            return m_builder.create<Dialect::BOrOp>(getLoc(expression, binOp->orToken), lhs, rhs);
        });
}

mlir::Value pylir::CodeGen::visit(const pylir::Syntax::AndTest& expression)
{
    return pylir::match(
        expression.variant, [&](const Syntax::NotTest& notTest) { return visit(notTest); },
        [&](const std::unique_ptr<Syntax::AndTest::BinOp>& binOp) -> mlir::Value
        {
            auto lhs = toBool(visit(*binOp->lhs));
            auto rhs = toBool(visit(binOp->rhs));
            return m_builder.create<Dialect::BAndOp>(getLoc(expression, binOp->andToken), lhs, rhs);
        });
}

mlir::Value pylir::CodeGen::visit(const Syntax::NotTest& expression)
{
    return pylir::match(
        expression.variant, [&](const Syntax::Comparison& comparison) { return visit(comparison); },
        [&](const std::pair<BaseToken, std::unique_ptr<Syntax::NotTest>>& pair) -> mlir::Value
        {
            auto value = toBool(visit(*pair.second));
            return m_builder.create<Dialect::BNegOp>(getLoc(expression, pair.first), value);
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
            IsNot,
            In,
            NotIn,
        };
        Comp comp;
        if (!op.secondToken)
        {
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
        }
        else
        {
            if (op.firstToken.getTokenType() == TokenType::IsKeyword)
            {
                comp = Comp::IsNot;
            }
            else
            {
                PYLIR_ASSERT(op.firstToken.getTokenType() == TokenType::NotKeyword);
                comp = Comp::NotIn;
            }
        }
        auto other = visit(rhs);
        mlir::Value cmp;
        if (Dialect::isNumbers(previousRHS.getType()) && Dialect::isNumbers(other.getType()))
        {
            ensureInt(previousRHS);
            ensureInt(other);
            Dialect::CmpPredicate predicate;
            switch (comp)
            {
                case Comp::Lt: predicate = Dialect::CmpPredicate::LT; break;
                case Comp::Le: predicate = Dialect::CmpPredicate::LE; break;
                case Comp::Gt: predicate = Dialect::CmpPredicate::GT; break;
                case Comp::Ge: predicate = Dialect::CmpPredicate::GE; break;
                case Comp::Is:
                case Comp::Eq: predicate = Dialect::CmpPredicate::EQ; break;
                case Comp::IsNot:
                case Comp::Ne: predicate = Dialect::CmpPredicate::NE; break;
                default:
                    // TODO
                    PYLIR_UNREACHABLE;
            }
            if (Dialect::isIntegerLike(previousRHS.getType()) && Dialect::isIntegerLike(other.getType()))
            {
                cmp = m_builder.create<Dialect::ICmpOp>(getLoc(op, op.firstToken), predicate, previousRHS, other);
            }
            else if (previousRHS.getType().isa<Dialect::FloatType>() && other.getType().isa<Dialect::FloatType>())
            {
                cmp = m_builder.create<Dialect::FCmpOp>(getLoc(op, op.firstToken), predicate, previousRHS, other);
            }
            else
            {
                // TODO
                PYLIR_UNREACHABLE;
            }
        }
        if (!result)
        {
            result = cmp;
            continue;
        }

        result = m_builder.create<Dialect::BAndOp>(getLoc(op, op.firstToken), result, cmp);
        previousRHS = other;
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
            if (lhs.getType().isa<Dialect::BoolType>() && rhs.getType().isa<Dialect::BoolType>())
            {
                return m_builder.create<Dialect::BOrOp>(getLoc(orExpr, binOp->bitOrToken), lhs, rhs);
            }
            if (Dialect::isIntegerLike(lhs.getType()) && Dialect::isIntegerLike(rhs.getType()))
            {
                ensureInt(lhs);
                ensureInt(rhs);
                return m_builder.create<Dialect::IOrOp>(getLoc(orExpr, binOp->bitOrToken), lhs, rhs);
            }
            // TODO
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
            if (lhs.getType().isa<Dialect::BoolType>() && rhs.getType().isa<Dialect::BoolType>())
            {
                return m_builder.create<Dialect::BXorOp>(getLoc(xorExpr, binOp->bitXorToken), lhs, rhs);
            }
            if (Dialect::isIntegerLike(lhs.getType()) && Dialect::isIntegerLike(rhs.getType()))
            {
                ensureInt(lhs);
                ensureInt(rhs);
                return m_builder.create<Dialect::IXorOp>(getLoc(xorExpr, binOp->bitXorToken), lhs, rhs);
            }
            // TODO
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
            if (lhs.getType().isa<Dialect::BoolType>() && rhs.getType().isa<Dialect::BoolType>())
            {
                return m_builder.create<Dialect::BAndOp>(getLoc(andExpr, binOp->bitAndToken), lhs, rhs);
            }
            if (Dialect::isIntegerLike(lhs.getType()) && Dialect::isIntegerLike(rhs.getType()))
            {
                ensureInt(lhs);
                ensureInt(rhs);
                return m_builder.create<Dialect::IAndOp>(getLoc(andExpr, binOp->bitAndToken), lhs, rhs);
            }
            // TODO
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
            if (Dialect::isIntegerLike(lhs.getType()) && Dialect::isIntegerLike(rhs.getType()))
            {
                ensureInt(lhs);
                ensureInt(rhs);
                switch (binOp->binToken.getTokenType())
                {
                    case TokenType::ShiftLeft:
                        return m_builder.create<Dialect::IShlOp>(getLoc(shiftExpr, binOp->binToken), lhs, rhs);
                    case TokenType::ShiftRight:
                        return m_builder.create<Dialect::IShrOp>(getLoc(shiftExpr, binOp->binToken), lhs, rhs);
                    default: PYLIR_UNREACHABLE;
                }
            }
            // TODO
            PYLIR_UNREACHABLE;
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
            arithmeticConversion(lhs, rhs);
            auto loc = getLoc(aExpr, binOp->binToken);
            if (Dialect::isIntegerLike(lhs.getType()) && Dialect::isIntegerLike(rhs.getType()))
            {
                switch (binOp->binToken.getTokenType())
                {
                    case TokenType::Plus: return m_builder.create<Dialect::IAddOp>(loc, lhs, rhs);
                    case TokenType::Minus: return m_builder.create<Dialect::ISubOp>(loc, lhs, rhs);
                    default: PYLIR_UNREACHABLE;
                }
            }
            else if (lhs.getType().isa<Dialect::FloatType>() && rhs.getType().isa<Dialect::FloatType>())
            {
                switch (binOp->binToken.getTokenType())
                {
                    case TokenType::Plus: return m_builder.create<Dialect::FAddOp>(loc, lhs, rhs);
                    case TokenType::Minus: return m_builder.create<Dialect::FSubOp>(loc, lhs, rhs);
                    default: PYLIR_UNREACHABLE;
                }
            }
            auto lhsType = m_builder.create<Dialect::TypeOf>(loc, lhs);
            auto addFunc = m_builder.create<Dialect::GetAttr>(loc, lhsType, m_builder.getStringAttr("__attr__"));
            auto id = m_builder.create<Dialect::Identity>(loc, addFunc);
            auto noneConstant =
                m_builder.create<Dialect::ConstantOp>(loc, Dialect::NoneAttr::get(m_builder.getContext()));
            auto noneId = m_builder.create<Dialect::Identity>(loc, noneConstant);
            auto notImplementedConstant =
                m_builder.create<Dialect::ConstantOp>(loc, Dialect::NotImplementedAttr::get(m_builder.getContext()));
            auto notImplementedId = m_builder.create<Dialect::Identity>(loc, notImplementedConstant);
            auto notNone = m_builder.create<Dialect::ICmpOp>(loc, Dialect::CmpPredicate::NE, id, noneId);
            auto notNotImplemented =
                m_builder.create<Dialect::ICmpOp>(loc, Dialect::CmpPredicate::NE, id, notImplementedId);
            auto condition = m_builder.create<Dialect::BAndOp>(loc, notNone, notNotImplemented);
            //TODO
            PYLIR_UNREACHABLE;
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
            arithmeticConversion(lhs, rhs);
            if (Dialect::isIntegerLike(lhs.getType()) && Dialect::isIntegerLike(rhs.getType()))
            {
                switch (binOp->binToken.getTokenType())
                {
                    case TokenType::Star:
                        return m_builder.create<Dialect::IMulOp>(getLoc(mExpr, binOp->binToken), lhs, rhs);
                    case TokenType::Divide:
                        return m_builder.create<Dialect::IDivOp>(getLoc(mExpr, binOp->binToken), lhs, rhs);
                    case TokenType::IntDivide:
                        return m_builder.create<Dialect::IFloorDivOp>(getLoc(mExpr, binOp->binToken), lhs, rhs);
                    case TokenType::Remainder:
                        return m_builder.create<Dialect::IModOp>(getLoc(mExpr, binOp->binToken), lhs, rhs);
                    default: PYLIR_UNREACHABLE;
                }
            }
            else if (lhs.getType().isa<Dialect::FloatType>() && rhs.getType().isa<Dialect::FloatType>())
            {
                switch (binOp->binToken.getTokenType())
                {
                    case TokenType::Star:
                        return m_builder.create<Dialect::FMulOp>(getLoc(mExpr, binOp->binToken), lhs, rhs);
                    case TokenType::Divide:
                        return m_builder.create<Dialect::FDivOp>(getLoc(mExpr, binOp->binToken), lhs, rhs);
                    case TokenType::IntDivide:
                        return m_builder.create<Dialect::FFloorDivOp>(getLoc(mExpr, binOp->binToken), lhs, rhs);
                    case TokenType::Remainder:
                        return m_builder.create<Dialect::FModOp>(getLoc(mExpr, binOp->binToken), lhs, rhs);
                    default: PYLIR_UNREACHABLE;
                }
            }
            // TODO
            PYLIR_UNREACHABLE;
        },
        [&](const std::unique_ptr<Syntax::MExpr::AtBin>& atBin) -> mlir::Value
        {
            // TODO
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
                    if (Dialect::isIntegerLike(value.getType()))
                    {
                        ensureInt(value);
                        return m_builder.create<Dialect::ISubOp>(
                            getLoc(uExpr, pair.first),
                            m_builder.create<Dialect::ConstantOp>(
                                m_builder.getUnknownLoc(),
                                Dialect::IntegerAttr::get(m_builder.getContext(), llvm::APInt(2, 0))),
                            value);
                    }
                    else if (value.getType().isa<Dialect::FloatType>())
                    {
                        return m_builder.create<Dialect::FSubOp>(
                            getLoc(uExpr, pair.first),
                            m_builder.create<Dialect::ConstantOp>(m_builder.getUnknownLoc(),
                                                                  Dialect::FloatAttr::get(m_builder.getContext(), 0)),
                            value);
                    }
                    // TODO
                    PYLIR_UNREACHABLE;
                }
                case TokenType::Plus:
                    if (Dialect::isNumbers(value.getType()))
                    {
                        return value;
                    }
                    // TODO
                    PYLIR_UNREACHABLE;
                case TokenType::BitNegate:
                    if (Dialect::isIntegerLike(value.getType()))
                    {
                        ensureInt(value);
                        return m_builder.create<Dialect::INegOp>(getLoc(uExpr, pair.first), value);
                    }
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
        [&](const auto&) -> mlir::Value
        {
            // TODO
            PYLIR_UNREACHABLE;
        });
}

mlir::Value pylir::CodeGen::visit(const pylir::Syntax::Atom& atom)
{
    return pylir::match(
        atom.variant,
        [&](const Syntax::Atom::Literal& literal) -> mlir::Value
        {
            switch (literal.token.getTokenType())
            {
                case TokenType::IntegerLiteral:
                {
                    return m_builder.create<pylir::Dialect::ConstantOp>(
                        getLoc(atom, literal.token),
                        Dialect::IntegerAttr::get(m_builder.getContext(),
                                                  pylir::get<llvm::APInt>(literal.token.getValue())));
                }
                case TokenType::FloatingPointLiteral:
                {
                    return m_builder.create<pylir::Dialect::ConstantOp>(
                        getLoc(atom, literal.token),
                        Dialect::FloatAttr::get(m_builder.getContext(), pylir::get<double>(literal.token.getValue())));
                }
                case TokenType::ComplexLiteral:
                {
                    // TODO:
                    PYLIR_UNREACHABLE;
                }
                case TokenType::StringLiteral:
                    return m_builder.create<pylir::Dialect::ConstantOp>(
                        getLoc(atom, literal.token),
                        Dialect::StringAttr::get(m_builder.getContext(),
                                                 pylir::get<std::string>(literal.token.getValue())));
                case TokenType::ByteLiteral:
                    // TODO:
                    PYLIR_UNREACHABLE;
                case TokenType::TrueKeyword:
                    return m_builder.create<pylir::Dialect::ConstantOp>(
                        getLoc(atom, literal.token), Dialect::BoolAttr::get(m_builder.getContext(), true));
                case TokenType::FalseKeyword:
                    return m_builder.create<pylir::Dialect::ConstantOp>(
                        getLoc(atom, literal.token), Dialect::BoolAttr::get(m_builder.getContext(), false));
                case TokenType::NoneKeyword:
                    return m_builder.create<pylir::Dialect::ConstantOp>(getLoc(atom, literal.token),
                                                                        Dialect::NoneAttr::get(m_builder.getContext()));
                default: PYLIR_UNREACHABLE;
            }
        },
        [&](const IdentifierToken& identifierToken) -> mlir::Value
        {
            auto loc = getLoc(identifierToken, identifierToken);
            auto result = getCurrentScope().find(identifierToken.getValue());
            if (result == getCurrentScope().end())
            {
                // TODO
                PYLIR_UNREACHABLE;
            }
            mlir::Value handle;
            if (auto alloca = llvm::dyn_cast<Dialect::AllocaOp>(result->second))
            {
                handle = alloca;
            }
            else
            {
                auto global = llvm::cast<Dialect::GlobalOp>(result->second);
                handle = m_builder.create<Dialect::HandleOfOp>(
                    loc, mlir::FlatSymbolRefAttr::get(global.sym_name(), global.getContext()));
            }
            return m_builder.create<Dialect::LoadOp>(loc, handle);
        },
        [&](const std::unique_ptr<Syntax::Enclosure>& enclosure) -> mlir::Value { return visit(*enclosure); });
}

void pylir::CodeGen::arithmeticConversion(mlir::Value& lhs, mlir::Value& rhs)
{
    // TODO complex
    if (lhs.getType().isa<Dialect::FloatType>() || rhs.getType().isa<Dialect::FloatType>())
    {
        if (Dialect::isIntegerLike(lhs.getType()))
        {
            ensureInt(lhs);
            lhs = m_builder.create<Dialect::ItoFOp>(m_builder.getUnknownLoc(), lhs);
        }
        if (Dialect::isIntegerLike(rhs.getType()))
        {
            ensureInt(rhs);
            rhs = m_builder.create<Dialect::ItoFOp>(m_builder.getUnknownLoc(), rhs);
        }
        return;
    }
    if (Dialect::isIntegerLike(lhs.getType()) && Dialect::isIntegerLike(rhs.getType()))
    {
        ensureInt(lhs);
        ensureInt(rhs);
    }
}

void pylir::CodeGen::ensureInt(mlir::Value& value)
{
    if (value.getType().isa<Dialect::BoolType>())
    {
        value = m_builder.create<Dialect::BtoIOp>(m_builder.getUnknownLoc(), value);
    }
}

mlir::Value pylir::CodeGen::toBool(mlir::Value value)
{
    if (value.getType().isa<Dialect::BoolType>())
    {
        return value;
    }
    if (value.getType().isa<Dialect::IntegerType>())
    {
        return m_builder.create<Dialect::ICmpOp>(
            m_builder.getUnknownLoc(), Dialect::CmpPredicate::NE, value,
            m_builder.create<Dialect::ConstantOp>(
                m_builder.getUnknownLoc(), Dialect::IntegerAttr::get(m_builder.getContext(), llvm::APInt(2, 0))));
    }
    if (value.getType().isa<Dialect::FloatType>())
    {
        return m_builder.create<Dialect::FCmpOp>(
            m_builder.getUnknownLoc(), Dialect::CmpPredicate::NE, value,
            m_builder.create<Dialect::ConstantOp>(m_builder.getUnknownLoc(),
                                                  Dialect::FloatAttr::get(m_builder.getContext(), 0)));
    }
    // TODO
    PYLIR_UNREACHABLE;
}

mlir::Value pylir::CodeGen::visit(const pylir::Syntax::Enclosure& enclosure)
{
    return pylir::match(
        enclosure.variant,
        [&](const Syntax::Enclosure::ParenthForm& parenthForm)
        {
            if (parenthForm.expression)
            {
                return visit(*parenthForm.expression);
            }
            // TODO
            PYLIR_UNREACHABLE;
        },
        [&](const auto&) -> mlir::Value
        {
            // TODO
            PYLIR_UNREACHABLE;
        });
}