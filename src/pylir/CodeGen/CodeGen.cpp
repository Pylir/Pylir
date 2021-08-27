#include "CodeGen.hpp"

#include <mlir/Dialect/StandardOps/IR/Ops.h>

#include <pylir/Optimizer/PylirMem/IR/PylirMemAttributes.hpp>
#include <pylir/Optimizer/PylirMem/IR/PylirMemDialect.hpp>
#include <pylir/Optimizer/PylirMem/IR/PylirMemTypeObjects.hpp>

pylir::CodeGen::CodeGen(mlir::MLIRContext* context, Diag::Document& document)
    : m_builder(
        [&]
        {
            context->loadDialect<pylir::Mem::PylirDialect>();
            context->loadDialect<mlir::StandardOpsDialect>();
            return context;
        }()),
      m_document(&document),
      m_refRefObject(Mem::PointerType::get(Mem::PointerType::get(m_builder.getType<Mem::ObjectType>())))
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
                if (auto alloca = llvm::dyn_cast<Mem::AllocaOp>(result->second))
                {
                    handle = alloca;
                }
                else
                {
                    auto global = llvm::cast<Mem::GlobalOp>(result->second);
                    handle = m_builder.create<Mem::GetGlobalOp>(location, m_refRefObject,
                                                                m_builder.getSymbolRefAttr(global));
                }
                m_builder.create<Mem::StoreOp>(location, value, handle);
                return;
            }
            if (m_scope.size() == 1)
            {
                // We are in the global scope
                Mem::GlobalOp op;
                {
                    mlir::OpBuilder::InsertionGuard guard{m_builder};
                    m_builder.setInsertionPointToEnd(m_module.getBody());
                    op = m_builder.create<Mem::GlobalOp>(location, identifierToken.getValue(), mlir::StringAttr{},
                                                         m_refRefObject.getElementType(), m_builder.getUnitAttr());
                }
                getCurrentScope().insert({identifierToken.getValue(), op});
                auto handle =
                    m_builder.create<Mem::GetGlobalOp>(location, m_refRefObject, m_builder.getSymbolRefAttr(op));
                m_builder.create<Mem::StoreOp>(location, value, handle);
                return;
            }

            auto alloca = m_builder.create<Mem::AllocaOp>(location, m_refRefObject.getElementType());
            m_builder.create<Mem::StoreOp>(location, value, alloca);
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

            mlir::Value count =
                m_builder.create<mlir::ConstantOp>(loc, m_builder.getIndexType(), m_builder.getIndexAttr(0));
            std::vector<mlir::Value> operands;
            auto handleItem = [&](const Syntax::StarredItem& item)
            {
                if (auto* assignment = std::get_if<Syntax::AssignmentExpression>(&item.variant))
                {
                    operands.push_back(visit(*assignment));
                    count = m_builder.create<mlir::AddIOp>(
                        loc, m_builder.getIndexType(), count,
                        m_builder.create<mlir::ConstantOp>(loc, m_builder.getIndexType(), m_builder.getIndexAttr(1)));
                    return;
                }
                auto list = visit(pylir::get<std::pair<BaseToken, Syntax::OrExpr>>(item.variant).second);
                // TODO check if iterable, add size to count, etc
            };
            for (auto& [item, comma] : items.leading)
            {
                handleItem(item);
            }
            if (items.last)
            {
                handleItem(*items.last);
            }
            auto tuple = m_builder.create<Mem::GCObjectAllocOp>(
                loc, count, m_builder.getSymbolRefAttr(Mem::getTupleTypeObject(m_module)));
            std::size_t index = 0;
            for (auto iter : operands)
            {
                auto constant =
                    m_builder.create<mlir::ConstantOp>(loc, m_builder.getIndexType(), m_builder.getIndexAttr(index++));
                m_builder.create<Mem::SetTupleItemOp>(loc, tuple, constant, iter);
            }
            return tuple;
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
    mlir::Value count = m_builder.create<mlir::ConstantOp>(
        loc, m_builder.getIndexType(), m_builder.getIndexAttr(expressionList.remainingExpr.size() + 1));
    std::vector<mlir::Value> operands(expressionList.remainingExpr.size() + 1);
    operands[0] = visit(*expressionList.firstExpr);
    std::transform(expressionList.remainingExpr.begin(), expressionList.remainingExpr.end(), operands.begin() + 1,
                   [&](const auto& pair) { return visit(*pair.second); });
    auto tuple = m_builder.create<Mem::GCObjectAllocOp>(loc, count,
                                                        m_builder.getSymbolRefAttr(Mem::getTupleTypeObject(m_module)));
    for (auto iter : llvm::enumerate(operands))
    {
        auto constant =
            m_builder.create<mlir::ConstantOp>(loc, m_builder.getIndexType(), m_builder.getIndexAttr(iter.index()));
        m_builder.create<Mem::SetTupleItemOp>(loc, tuple, constant, iter.value());
    }
    return tuple;
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

    //    auto result = m_builder.create<Mem::AllocaOp>(loc);
    //
    //    mlir::Block* thenBlock = new mlir::Block;
    //    mlir::Block* elseBlock = new mlir::Block;
    //    mlir::Block* continueBlock = new mlir::Block;
    //
    //    m_builder.create<mlir::CondBranchOp>(
    //        loc, m_builder.create<Mem::BtoI1Op>(loc, toBool(visit(*expression.suffix->test))), thenBlock,
    //        elseBlock);
    //
    //    m_currentFunc.getCallableRegion()->push_back(thenBlock);
    //    m_builder.setInsertionPointToStart(thenBlock);
    //    auto thenValue = visit(expression.value);
    //    m_builder.create<Mem::StoreOp>(loc, thenValue, result);
    //    m_builder.create<mlir::BranchOp>(loc, continueBlock);
    //
    //    m_currentFunc.getCallableRegion()->push_back(elseBlock);
    //    m_builder.setInsertionPointToStart(elseBlock);
    //    auto elseValue = visit(*expression.suffix->elseValue);
    //    m_builder.create<Mem::StoreOp>(loc, elseValue, result);
    //    m_builder.create<mlir::BranchOp>(loc, continueBlock);
    //
    //    m_currentFunc.getCallableRegion()->push_back(continueBlock);
    //    m_builder.setInsertionPointToStart(continueBlock);
    //    return m_builder.create<Mem::LoadOp>(loc, m_builder.getType<Mem::UnknownType>(), result);
    // TODO
    PYLIR_UNREACHABLE;
}

mlir::Value pylir::CodeGen::visit(const pylir::Syntax::OrTest& expression)
{
    return pylir::match(
        expression.variant, [&](const Syntax::AndTest& andTest) { return visit(andTest); },
        [&](const std::unique_ptr<Syntax::OrTest::BinOp>& binOp) -> mlir::Value
        {
            auto lhs = toBool(visit(*binOp->lhs));
            auto rhs = toBool(visit(binOp->rhs));
            return m_builder.create<mlir::OrOp>(getLoc(expression, binOp->orToken), lhs, rhs);
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
            return m_builder.create<mlir::AndOp>(getLoc(expression, binOp->andToken), lhs, rhs);
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
            auto one = m_builder.create<mlir::ConstantOp>(loc, m_builder.getBoolAttr(true));
            return m_builder.create<mlir::XOrOp>(loc, value, one);
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
        // TODO cmp
        if (!result)
        {
            result = cmp;
            continue;
        }

        result = m_builder.create<mlir::AndOp>(getLoc(op, op.firstToken), result, cmp);
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
            auto loc = getLoc(orExpr, binOp->bitOrToken);
            return genBinOp(loc, lhs, rhs, Mem::TypeSlotPredicate::Or, "__ror__");
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
            return genBinOp(loc, lhs, rhs, Mem::TypeSlotPredicate::Xor, "__rxor__");
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
            return genBinOp(loc, lhs, rhs, Mem::TypeSlotPredicate::And, "__rand__");
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
                case TokenType::ShiftLeft:
                    return genBinOp(loc, lhs, rhs, Mem::TypeSlotPredicate::LShift, "__rlshift__");
                case TokenType::ShiftRight:
                    return genBinOp(loc, lhs, rhs, Mem::TypeSlotPredicate::RShift, "__rrshift__");
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
                case TokenType::Plus: return genBinOp(loc, lhs, rhs, Mem::TypeSlotPredicate::Add, "__radd__");
                case TokenType::Minus: return genBinOp(loc, lhs, rhs, Mem::TypeSlotPredicate::Subtract, "__rsub__");
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
                case TokenType::Star: return genBinOp(loc, lhs, rhs, Mem::TypeSlotPredicate::Multiply, "__rmul__");
                case TokenType::Divide:
                    return genBinOp(loc, lhs, rhs, Mem::TypeSlotPredicate::TrueDivide, "__rtruediv__");
                case TokenType::IntDivide:
                    return genBinOp(loc, lhs, rhs, Mem::TypeSlotPredicate::FloorDivide, "__rfloordiv__");
                case TokenType::Remainder:
                    return genBinOp(loc, lhs, rhs, Mem::TypeSlotPredicate::Remainder, "__rmod__");
                default: PYLIR_UNREACHABLE;
            }
        },
        [&](const std::unique_ptr<Syntax::MExpr::AtBin>& atBin) -> mlir::Value
        {
            auto lhs = visit(*atBin->lhs);
            auto rhs = visit(*atBin->rhs);
            auto loc = getLoc(mExpr, atBin->atToken);
            return genBinOp(loc, lhs, rhs, Mem::TypeSlotPredicate::MatrixMultiply, "__rmatmul__");
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
                    auto intTypeObject = m_builder.create<Mem::DataOfOp>(location, Mem::getIntTypeObject(m_module));
                    return m_builder.create<Mem::IntegerConstant>(location, intTypeObject,
                                                                  pylir::get<llvm::APInt>(literal.token.getValue()));
                }
                case TokenType::FloatingPointLiteral:
                {
                    auto constant = m_builder.create<mlir::ConstantOp>(
                        location, m_builder.getF64FloatAttr(pylir::get<double>(literal.token.getValue())));
                    // TODO:
                    PYLIR_UNREACHABLE;
                }
                case TokenType::ComplexLiteral:
                {
                    // TODO:
                    PYLIR_UNREACHABLE;
                }
                case TokenType::StringLiteral:
                {
                    auto utf32 = Text::toUTF32String(pylir::get<std::string>(literal.token.getValue()));
                    std::vector<std::int32_t> values(utf32.begin(), utf32.end());
                    auto size = m_builder.create<mlir::ConstantOp>(location, m_builder.getIndexType(),
                                                                   m_builder.getIndexAttr(utf32.size()));
                    auto gcAlloc = m_builder.create<Mem::GCObjectAllocOp>(
                        location, size, m_builder.getSymbolRefAttr(Mem::getStringTypeObject(m_module)));
                    for (auto& iter : llvm::enumerate(values))
                    {
                        auto index = m_builder.create<mlir::ConstantOp>(location, m_builder.getIndexType(),
                                                                        m_builder.getIndexAttr(iter.index()));
                        auto constant = m_builder.create<mlir::ConstantOp>(location, m_builder.getI32Type(),
                                                                           m_builder.getI32IntegerAttr(iter.value()));
                        m_builder.create<Mem::SetStringItemOp>(location, gcAlloc, index, constant);
                    }
                    return gcAlloc;
                }
                case TokenType::ByteLiteral:
                    // TODO:
                    PYLIR_UNREACHABLE;
                case TokenType::TrueKeyword:
                {
                    auto boolTypeObject = m_builder.create<Mem::DataOfOp>(location, Mem::getBoolTypeObject(m_module));
                    return m_builder.create<Mem::IntegerConstant>(location, boolTypeObject, llvm::APInt(2, 1));
                }
                case TokenType::FalseKeyword:
                {
                    auto boolTypeObject = m_builder.create<Mem::DataOfOp>(location, Mem::getBoolTypeObject(m_module));
                    return m_builder.create<Mem::IntegerConstant>(location, boolTypeObject, llvm::APInt(2, 0));
                }
                case TokenType::NoneKeyword:
                    return m_builder.create<pylir::Mem::DataOfOp>(location, Mem::getNoneObject(m_module));
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
            if (auto alloca = llvm::dyn_cast<Mem::AllocaOp>(result->second))
            {
                handle = alloca;
            }
            else
            {
                auto global = llvm::cast<Mem::GlobalOp>(result->second);
                handle = m_builder.create<Mem::GetGlobalOp>(loc, m_refRefObject, m_builder.getSymbolRefAttr(global));
            }
            return m_builder.create<Mem::LoadOp>(loc, handle);
        },
        [&](const std::unique_ptr<Syntax::Enclosure>& enclosure) -> mlir::Value { return visit(*enclosure); });
}

mlir::Value pylir::CodeGen::visit(const pylir::Syntax::Subscription& subscription)
{
    auto container = visit(*subscription.primary);
    auto indices = visit(subscription.expressionList);

    auto loc = getLoc(subscription, subscription);
    auto type = m_builder.create<Mem::TypeOfOp>(loc, container);
    auto result = m_builder.create<Mem::GetTypeSlotOp>(loc, Mem::TypeSlotPredicate::GetItem, type);
    auto getItem = result.getResult(0);
    auto found = result.getResult(1);

    auto notFoundBlock = new mlir::Block;
    auto foundBlock = new mlir::Block;
    m_builder.create<mlir::CondBranchOp>(loc, found, foundBlock, notFoundBlock);

    m_currentFunc.getCallableRegion()->push_back(notFoundBlock);
    m_builder.setInsertionPointToStart(notFoundBlock);
    // TODO raise
    m_builder.create<mlir::ReturnOp>(loc);

    m_currentFunc.getCallableRegion()->push_back(foundBlock);
    m_builder.setInsertionPointToStart(foundBlock);
    return m_builder.create<mlir::CallIndirectOp>(loc, getItem, mlir::ValueRange{container, indices}).getResult(0);
}

mlir::Value pylir::CodeGen::toBool(mlir::Value value)
{
    // TODO
    PYLIR_UNREACHABLE;
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
            auto zero = m_builder.create<mlir::ConstantOp>(loc, m_builder.getIndexType(), m_builder.getIndexAttr(0));
            auto zeroSizedStorage = m_builder.create<Mem::GCObjectAllocOp>(
                loc, zero, m_builder.getSymbolRefAttr(Mem::getTupleTypeObject(m_module)));
            return zeroSizedStorage;
        },
        [&](const auto&) -> mlir::Value
        {
            // TODO
            PYLIR_UNREACHABLE;
        });
}

std::pair<mlir::Value, mlir::Value> pylir::CodeGen::lookupTypeSlot(mlir::Location loc, mlir::Value type,
                                                                   Mem::TypeSlotPredicate slot)
{
    auto mro = m_builder.create<Mem::GetTypeSlotOp>(loc, Mem::TypeSlotPredicate::Mro, type).result();
    auto size = m_builder.create<Mem::TupleSizeOp>(loc, mro);

    auto exitBlock = new mlir::Block;
    exitBlock->addArgument(Mem::GetTypeSlotOp::returnTypeFromPredicate(m_builder.getContext(), slot));
    exitBlock->addArgument(m_builder.getI1Type());

    auto loopBody = new mlir::Block;
    loopBody->addArgument(m_builder.getIndexType());
    auto zero = m_builder.create<mlir::ConstantOp>(loc, m_builder.getIndexType(), m_builder.getIndexAttr(0));
    m_builder.create<mlir::BranchOp>(loc, loopBody, mlir::ValueRange{zero});

    m_currentFunc.getCallableRegion()->push_back(loopBody);
    m_builder.setInsertionPointToStart(loopBody);

    auto thisBase = m_builder.create<Mem::GetTupleItemOp>(loc, mro, loopBody->getArgument(0));
    auto typeSlotOp = m_builder.create<Mem::GetTypeSlotOp>(loc, slot, thisBase);
    auto notFound = new mlir::Block;
    m_builder.create<mlir::CondBranchOp>(loc, typeSlotOp.found(), exitBlock,
                                         mlir::ValueRange{typeSlotOp.result(), typeSlotOp.found()}, notFound,
                                         mlir::ValueRange{});

    m_currentFunc.getCallableRegion()->push_back(notFound);
    m_builder.setInsertionPointToStart(notFound);
    auto one = m_builder.create<mlir::ConstantOp>(loc, m_builder.getIndexType(), m_builder.getIndexAttr(1));
    auto newIndex = m_builder.create<mlir::AddIOp>(loc, loopBody->getArgument(0), one);
    auto isLess = m_builder.create<mlir::CmpIOp>(loc, mlir::CmpIPredicate::ule, newIndex, size);
    m_builder.create<mlir::CondBranchOp>(loc, isLess, loopBody, mlir::ValueRange{newIndex}, exitBlock,
                                         mlir::ValueRange{typeSlotOp.result(), typeSlotOp.found()});

    m_currentFunc.getCallableRegion()->push_back(exitBlock);
    m_builder.setInsertionPointToStart(exitBlock);
    return {exitBlock->getArgument(0), exitBlock->getArgument(1)};
}

mlir::Value pylir::CodeGen::genBinOp(mlir::Location loc, mlir::Value lhs, mlir::Value rhs,
                                     Mem::TypeSlotPredicate operation, std::string_view)
{
    auto lhsType = m_builder.create<Mem::TypeOfOp>(loc, lhs);
    auto [func, wasFound] = lookupTypeSlot(loc, lhsType, operation);
    auto result = m_builder.create<Mem::AllocaOp>(loc, m_refRefObject);

    auto* foundBlock = new mlir::Block;
    auto* notFoundBlock = new mlir::Block;
    auto* endBlock = new mlir::Block;

    m_builder.create<mlir::CondBranchOp>(loc, wasFound, foundBlock, notFoundBlock);

    {
        m_currentFunc.getCallableRegion()->push_back(foundBlock);
        m_builder.setInsertionPointToStart(foundBlock);

        auto call = m_builder.create<mlir::CallIndirectOp>(loc, func, mlir::ValueRange{lhs, rhs}).getResult(0);
        auto notImplementedConstant = m_builder.create<Mem::DataOfOp>(loc, Mem::getNotImplementedObject(m_module));
        auto notImplementedId = m_builder.create<Mem::IdOp>(loc, notImplementedConstant);
        auto id = m_builder.create<Mem::IdOp>(loc, call);
        auto isNotImplemented = m_builder.create<mlir::CmpIOp>(loc, mlir::CmpIPredicate::eq, id, notImplementedId);
        auto continueBlock = mlir::OpBuilder{m_builder}.createBlock(m_currentFunc.getCallableRegion());
        m_builder.create<mlir::CondBranchOp>(loc, isNotImplemented, notFoundBlock, continueBlock);
        m_builder.setInsertionPointToStart(continueBlock);
        m_builder.create<Mem::StoreOp>(loc, call, result);
        m_builder.create<mlir::BranchOp>(loc, endBlock);
    }

    {
        m_currentFunc.getCallableRegion()->push_back(notFoundBlock);
        m_builder.setInsertionPointToStart(notFoundBlock);

        auto* raiseBlock = new mlir::Block;
        m_builder.create<mlir::BranchOp>(loc, raiseBlock);

        /*TODO __r*__operator
        auto rhsType = m_builder.create<Mem::TypeOfOp>(loc, rhs);
        auto tryRBlock = OpBuilder{m_builder}.createBlock(m_currentFunc.getCallableRegion());

        auto rhsTypeId = m_builder.create<Mem::IdOp>(loc, rhsType);
        auto lhsTypeId = m_builder.create<Mem::IdOp>(loc, lhsType);
        auto typeEqual = m_builder.create<Mem::ICmpOp>(loc, Mem::CmpPredicate::EQ, rhsTypeId, lhsTypeId);
        m_builder.create<mlir::CondBranchOp>(loc, m_builder.create<Mem::BtoI1Op>(loc, typeEqual), raiseBlock,
                                             tryRBlock);

        m_builder.setInsertionPointToStart(tryRBlock);
        slot = m_builder.create<Mem::GetAttrOp>(loc, rhsType, m_builder.getStringAttr(fallback));
        func = slot.getResult(0);
        wasFound = slot.getResult(1);
        auto continueBlock = OpBuilder{m_builder}.createBlock(m_currentFunc.getCallableRegion());
        m_builder.create<mlir::CondBranchOp>(loc, wasFound, continueBlock, raiseBlock);

        m_builder.setInsertionPointToStart(continueBlock);
        auto posArgs = m_builder.create<Mem::MakeTupleOp>(loc, m_builder.getType<Mem::TupleType>(),
                                                              mlir::ValueRange{rhs, lhs});
        auto emptyDict = m_builder.create<Mem::ConstantOp>(loc, Mem::DictAttr::get(m_builder.getContext(), {}));
        auto call = assureCallable(loc, func, posArgs, emptyDict);
        auto notImplementedConstant =
            m_builder.create<Mem::DataOfOp>(loc, Mem::getNotImplementedObject(m_module));
        auto notImplementedId = m_builder.create<Mem::IdOp>(loc, notImplementedConstant);
        auto id = m_builder.create<Mem::IdOp>(loc, call);
        auto isNotImplemented = m_builder.create<Mem::ICmpOp>(loc, Mem::CmpPredicate::EQ, id, notImplementedId);
        auto storeBlock = OpBuilder{m_builder}.createBlock(m_currentFunc.getCallableRegion());
        m_builder.create<mlir::CondBranchOp>(loc, m_builder.create<Mem::BtoI1Op>(loc, isNotImplemented), raiseBlock,
                                             storeBlock);

        m_builder.setInsertionPointToStart(storeBlock);
        m_builder.create<Mem::StoreOp>(loc, call, result);
        m_builder.create<mlir::BranchOp>(loc, endBlock);
         */

        m_currentFunc.getCallableRegion()->push_back(raiseBlock);
        m_builder.setInsertionPointToStart(raiseBlock);
        // TODO raise terminator
        m_builder.create<mlir::ReturnOp>(loc);
    }

    m_currentFunc.getCallableRegion()->push_back(endBlock);
    m_builder.setInsertionPointToStart(endBlock);
    return m_builder.create<Mem::LoadOp>(loc, result);
}

mlir::Value pylir::CodeGen::visit(const pylir::Syntax::AssignmentExpression& assignmentExpression)
{
    if (!assignmentExpression.identifierAndWalrus)
    {
        return visit(*assignmentExpression.expression);
    }
    // TODO
    PYLIR_UNREACHABLE;
}
