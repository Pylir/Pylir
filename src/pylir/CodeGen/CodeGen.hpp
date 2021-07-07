#pragma once

#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>

#include <pylir/Diagnostics/DiagnosticsBuilder.hpp>
#include <pylir/Parser/Syntax.hpp>

#include <unordered_map>

namespace pylir
{
class CodeGen
{
    mlir::OpBuilder m_builder;
    mlir::ModuleOp m_module;
    mlir::FuncOp m_currentFunc;
    Diag::Document* m_document;
    std::vector<std::unordered_map<std::string_view, mlir::Operation*>> m_scope{1};

    std::unordered_map<std::string_view, mlir::Operation*>& getCurrentScope()
    {
        return m_scope.back();
    }

    void arithmeticConversion(mlir::Value& lhs, mlir::Value& rhs);

    void ensureInt(mlir::Value& value);

    mlir::Value toBool(mlir::Value value);

    template <class AST, class FallBackLocation>
    mlir::Location getLoc(const AST& astObject, const FallBackLocation& fallBackLocation)
    {
        auto [line, col] = m_document->getLineCol(Diag::range(fallBackLocation).first);
        return mlir::OpaqueLoc::get(
            &astObject, m_builder.getFileLineColLoc(m_builder.getIdentifier(m_document->getFilename()), line, col));
    }

    void assignTarget(const Syntax::TargetList& targetList, mlir::Value value);

    void assignTarget(const Syntax::Target& target, mlir::Value value);

public:
    CodeGen(mlir::MLIRContext* context, Diag::Document& document);

    mlir::ModuleOp visit(const Syntax::FileInput& fileInput);

    void visit(const Syntax::Statement& statement);

    void visit(const Syntax::StmtList& stmtList);

    void visit(const Syntax::CompoundStmt& compoundStmt);

    void visit(const Syntax::SimpleStmt& simpleStmt);

    void visit(const Syntax::AssignmentStmt& assignmentStmt);

    mlir::Value visit(const Syntax::StarredExpression& starredExpression);

    mlir::Value visit(const Syntax::YieldExpression& yieldExpression);

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

    mlir::Value visit(const Syntax::Enclosure& enclosure);
};

inline mlir::ModuleOp codegen(mlir::MLIRContext* context, const Syntax::FileInput& input, Diag::Document& document)
{
    CodeGen codegen{context, document};
    return codegen.visit(input);
}

} // namespace pylir