#include "CodeGen.hpp"

#include <pylir/Dialect/PylirAttributes.hpp>
#include <pylir/Dialect/PylirDialect.hpp>
#include <pylir/Dialect/PylirOps.hpp>
#include <pylir/Dialect/PylirTypes.hpp>

pylir::CodeGen::CodeGen(mlir::MLIRContext* context, Diag::Document& document)
    : m_builder(
        [&]
        {
            context->loadDialect<pylir::Dialect::PylirDialect>();
            return context;
        }()),
      m_document(&document)
{
}

mlir::ModuleOp pylir::CodeGen::visit(const pylir::Syntax::FileInput& fileInput)
{
    m_module = mlir::ModuleOp::create(m_builder.getUnknownLoc());
    auto initFunc = mlir::FuncOp::create(m_builder.getUnknownLoc(), "__init__",
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
        [&](const Syntax::StarredExpression& expression) { visit(expression); });
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
    // TODO
    PYLIR_UNREACHABLE;
}

mlir::Value pylir::CodeGen::visit(const pylir::Syntax::OrTest& expression)
{
    return pylir::match(
        expression.variant, [&](const Syntax::AndTest& andTest) { return visit(andTest); },
        [&](const auto&) -> mlir::Value
        {
            // TODO
            PYLIR_UNREACHABLE;
        });
}

mlir::Value pylir::CodeGen::visit(const pylir::Syntax::AndTest& expression)
{
    return pylir::match(
        expression.variant, [&](const Syntax::NotTest& notTest) { return visit(notTest); },
        [&](const auto&) -> mlir::Value
        {
            // TODO
            PYLIR_UNREACHABLE;
        });
}

mlir::Value pylir::CodeGen::visit(const Syntax::NotTest& expression)
{
    return pylir::match(
        expression.variant, [&](const Syntax::Comparison& comparison) { return visit(comparison); },
        [&](const auto&) -> mlir::Value
        {
            // TODO
            PYLIR_UNREACHABLE;
        });
}

mlir::Value pylir::CodeGen::visit(const pylir::Syntax::Comparison& comparison)
{
    if (comparison.rest.empty())
    {
        return visit(comparison.left);
    }
    // TODO
    PYLIR_UNREACHABLE;
}

mlir::Value pylir::CodeGen::visit(const Syntax::OrExpr& orExpr)
{
    return pylir::match(
        orExpr.variant, [&](const Syntax::XorExpr& xorExpr) { return visit(xorExpr); },
        [&](const std::unique_ptr<Syntax::OrExpr::BinOp>& binOp) -> mlir::Value
        {
            auto lhs = visit(*binOp->lhs);
            auto rhs = visit(binOp->rhs);
            if (lhs.getType().isa<Dialect::IntegerType>() && rhs.getType().isa<Dialect::IntegerType>())
            {
                return m_builder.createOrFold<Dialect::IOrOp>(getLoc(orExpr, binOp->bitOrToken), lhs, rhs);
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
            if (lhs.getType().isa<Dialect::IntegerType>() && rhs.getType().isa<Dialect::IntegerType>())
            {
                return m_builder.createOrFold<Dialect::IXorOp>(getLoc(xorExpr, binOp->bitXorToken), lhs, rhs);
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
            if (lhs.getType().isa<Dialect::IntegerType>() && rhs.getType().isa<Dialect::IntegerType>())
            {
                return m_builder.createOrFold<Dialect::IAndOp>(getLoc(andExpr, binOp->bitAndToken), lhs, rhs);
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
            if (lhs.getType().isa<Dialect::IntegerType>() && rhs.getType().isa<Dialect::IntegerType>())
            {
                switch (binOp->binToken.getTokenType())
                {
                    case TokenType::ShiftLeft:
                        return m_builder.createOrFold<Dialect::IShlOp>(getLoc(shiftExpr, binOp->binToken), lhs, rhs);
                    case TokenType::ShiftRight:
                        return m_builder.createOrFold<Dialect::IShrOp>(getLoc(shiftExpr, binOp->binToken), lhs, rhs);
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
            if (lhs.getType().isa<Dialect::IntegerType>() && rhs.getType().isa<Dialect::IntegerType>())
            {
                switch (binOp->binToken.getTokenType())
                {
                    case TokenType::Plus:
                        return m_builder.createOrFold<Dialect::IAddOp>(getLoc(aExpr, binOp->binToken), lhs, rhs);
                    case TokenType::Minus:
                        return m_builder.createOrFold<Dialect::ISubOp>(getLoc(aExpr, binOp->binToken), lhs, rhs);
                    default: PYLIR_UNREACHABLE;
                }
            }
            else if (lhs.getType().isa<Dialect::FloatType>() && rhs.getType().isa<Dialect::FloatType>())
            {
                switch (binOp->binToken.getTokenType())
                {
                    case TokenType::Plus:
                        return m_builder.createOrFold<Dialect::FAddOp>(getLoc(aExpr, binOp->binToken), lhs, rhs);
                    case TokenType::Minus:
                        return m_builder.createOrFold<Dialect::FSubOp>(getLoc(aExpr, binOp->binToken), lhs, rhs);
                    default: PYLIR_UNREACHABLE;
                }
            }
            // TODO
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
            if (lhs.getType().isa<Dialect::IntegerType>() && rhs.getType().isa<Dialect::IntegerType>())
            {
                switch (binOp->binToken.getTokenType())
                {
                    case TokenType::Star:
                        return m_builder.createOrFold<Dialect::IMulOp>(getLoc(mExpr, binOp->binToken), lhs, rhs);
                    case TokenType::Divide:
                        return m_builder.createOrFold<Dialect::IDivOp>(getLoc(mExpr, binOp->binToken), lhs, rhs);
                    case TokenType::IntDivide:
                        return m_builder.createOrFold<Dialect::IFloorDivOp>(getLoc(mExpr, binOp->binToken), lhs, rhs);
                    case TokenType::Remainder:
                        return m_builder.createOrFold<Dialect::IModOp>(getLoc(mExpr, binOp->binToken), lhs, rhs);
                    default: PYLIR_UNREACHABLE;
                }
            }
            else if (lhs.getType().isa<Dialect::FloatType>() && rhs.getType().isa<Dialect::FloatType>())
            {
                switch (binOp->binToken.getTokenType())
                {
                    case TokenType::Star:
                        return m_builder.createOrFold<Dialect::FMulOp>(getLoc(mExpr, binOp->binToken), lhs, rhs);
                    case TokenType::Divide:
                        return m_builder.createOrFold<Dialect::FDivOp>(getLoc(mExpr, binOp->binToken), lhs, rhs);
                    case TokenType::IntDivide:
                        return m_builder.createOrFold<Dialect::FFloorDivOp>(getLoc(mExpr, binOp->binToken), lhs, rhs);
                    case TokenType::Remainder:
                        return m_builder.createOrFold<Dialect::FModOp>(getLoc(mExpr, binOp->binToken), lhs, rhs);
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
                    if (value.getType().isa<Dialect::IntegerType>())
                    {
                        return m_builder.createOrFold<Dialect::ISubOp>(
                            getLoc(uExpr, pair.first),
                            m_builder.create<Dialect::ConstantOp>(
                                m_builder.getUnknownLoc(),
                                Dialect::IntegerAttr::get(m_builder.getContext(), llvm::APInt(2, 0))),
                            value);
                    }
                    else if (value.getType().isa<Dialect::FloatType>())
                    {
                        return m_builder.createOrFold<Dialect::FSubOp>(
                            getLoc(uExpr, pair.first),
                            m_builder.create<Dialect::ConstantOp>(m_builder.getUnknownLoc(),
                                                                  Dialect::FloatAttr::get(m_builder.getContext(), 0)),
                            value);
                    }
                    // TODO
                    PYLIR_UNREACHABLE;
                }
                case TokenType::Plus:
                    if (value.getType().isa<Dialect::IntegerType>() || value.getType().isa<Dialect::FloatType>())
                    {
                        return value;
                    }
                    // TODO
                    PYLIR_UNREACHABLE;
                case TokenType::BitNegate:
                    if (value.getType().isa<Dialect::IntegerType>())
                    {
                        return m_builder.createOrFold<Dialect::INegOp>(getLoc(uExpr, pair.first), value);
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
            // TODO
            PYLIR_UNREACHABLE;
        },
        [](const std::unique_ptr<Syntax::Enclosure>& enclosure) -> mlir::Value
        {
            // TODO
            PYLIR_UNREACHABLE;
        });
}

void pylir::CodeGen::arithmeticConversion(mlir::Value& lhs, mlir::Value& rhs)
{
    // TODO complex
    if (lhs.getType().isa<Dialect::FloatType>() || rhs.getType().isa<Dialect::FloatType>())
    {
        if (lhs.getType().isa<Dialect::IntegerType>())
        {
            lhs = m_builder.createOrFold<Dialect::ItoF>(m_builder.getUnknownLoc(), lhs);
        }
        if (rhs.getType().isa<Dialect::IntegerType>())
        {
            rhs = m_builder.createOrFold<Dialect::ItoF>(m_builder.getUnknownLoc(), rhs);
        }
        return;
    }
}
