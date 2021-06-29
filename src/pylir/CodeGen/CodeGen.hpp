#pragma once

#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>

#include <pylir/Parser/Syntax.hpp>

namespace pylir
{
class CodeGen
{
    mlir::OpBuilder m_builder;
    mlir::ModuleOp m_module;

public:
    CodeGen(mlir::MLIRContext* context);

    mlir::ModuleOp visit(const Syntax::FileInput& fileInput);

    void visit(const Syntax::Statement& statement);

    void visit(const Syntax::StmtList& stmtList);

    void visit(const Syntax::CompoundStmt& compoundStmt);

    void visit(const Syntax::SimpleStmt& simpleStmt);

    mlir::Value visit(const Syntax::StarredExpression& starredExpression);

    mlir::Value visit(const Syntax::Expression& expression);

    mlir::Value visit(const Syntax::ConditionalExpression& expression);

    mlir::Value visit(const Syntax::OrTest& expression);

    mlir::Value visit(const Syntax::AndTest& expression);

    mlir::Value visit(const Syntax::NotTest& expression);

    mlir::Value visit(const Syntax::Comparison& comparison);

    mlir::Value visit(const Syntax::OrExpr& orExpr);

    mlir::Value visit(const Syntax::XorExpr& xorExpr);

    mlir::Value visit(const Syntax::AndExpr& andExpr);

    mlir::Value visit(const Syntax::ShiftExpr& shiftExpr);

    mlir::Value visit(const Syntax::AExpr& aExpr);

    mlir::Value visit(const Syntax::MExpr& mExpr);

    mlir::Value visit(const Syntax::UExpr& uExpr);

    mlir::Value visit(const Syntax::Power& power);

    mlir::Value visit(const Syntax::AwaitExpr& awaitExpr);

    mlir::Value visit(const Syntax::Primary& primary);

    mlir::Value visit(const Syntax::Atom& atom);
};

inline mlir::ModuleOp codegen(mlir::MLIRContext* context, const Syntax::FileInput& input)
{
    CodeGen codegen{context};
    return codegen.visit(input);
}

} // namespace pylir
